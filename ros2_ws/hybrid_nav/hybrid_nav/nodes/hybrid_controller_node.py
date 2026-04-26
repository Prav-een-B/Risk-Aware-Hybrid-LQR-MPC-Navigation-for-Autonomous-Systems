#!/usr/bin/env python3
"""
Hybrid Controller Node for TurtleBot3 — v4 (Turn-then-Drive)
=============================================================

Strategy:
  1. Find the target waypoint (nearest point + lookahead on trajectory)
  2. If heading error > 20°: STOP and TURN in place
  3. If heading error < 20°: DRIVE forward with proportional steering
  4. Advance waypoint only when robot gets close

This "turn-then-drive" approach is oscillation-proof because the robot
never drives forward while facing the wrong direction.

When near obstacles (risk > 0.1), MPC overlays for avoidance.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, String, Float32MultiArray


def euler_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)


def normalize_angle(a):
    """Wrap to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class HybridControllerNode(Node):
    def __init__(self):
        super().__init__('hybrid_controller')

        # ── Parameters ─────────────────────────────────────────
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('trajectory_amplitude', 0.5)
        self.declare_parameter('trajectory_frequency', 0.2)
        self.declare_parameter('trajectory_duration', 120.0)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('mpc_horizon', 10)
        self.declare_parameter('d_safe', 0.35)
        self.declare_parameter('v_max', 0.22)
        self.declare_parameter('omega_max', 2.84)
        self.declare_parameter('use_hybrid', True)
        self.declare_parameter('lookahead_dist', 0.20)
        self.declare_parameter('cruise_speed', 0.10)
        self.declare_parameter('turn_threshold', 0.35)   # ~20 degrees

        rate = self.get_parameter('control_rate').value
        self.A = self.get_parameter('trajectory_amplitude').value
        self.freq = self.get_parameter('trajectory_frequency').value
        dur = self.get_parameter('trajectory_duration').value
        self.dt = self.get_parameter('dt').value
        self.v_max = self.get_parameter('v_max').value
        self.omega_max = self.get_parameter('omega_max').value
        self.use_hybrid = self.get_parameter('use_hybrid').value
        self.lookahead = self.get_parameter('lookahead_dist').value
        self.cruise_speed = self.get_parameter('cruise_speed').value
        self.turn_threshold = self.get_parameter('turn_threshold').value

        # ── Try advanced controllers ───────────────────────────
        self.controllers_ok = False
        try:
            from hybrid_nav.controllers.lqr_controller import LQRController
            from hybrid_nav.controllers.mpc_controller import MPCController, Obstacle
            from hybrid_nav.controllers.hybrid_blender import BlendingSupervisor
            from hybrid_nav.controllers.risk_metrics import RiskMetrics
            self.mpc = MPCController(
                horizon=self.get_parameter('mpc_horizon').value,
                d_safe=self.get_parameter('d_safe').value,
                v_max=self.v_max, omega_max=self.omega_max, dt=self.dt,
            )
            self.blender = BlendingSupervisor(dt=self.dt)
            self.risk_calc = RiskMetrics(
                d_trigger=1.5, d_safe=self.get_parameter('d_safe').value
            )
            self.Obstacle = Obstacle
            self.controllers_ok = True
            self.get_logger().info('✅ MPC available for obstacle avoidance')
        except Exception as e:
            self.get_logger().warn(f'⚠️ MPC not available: {e}')

        # ── Trajectory: generate sparse waypoints ──────────────
        self._generate_waypoints(dur)

        # ── State ──────────────────────────────────────────────
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.odom_ok = False
        self.obstacles = []
        self.wp_idx = 0          # Current waypoint target
        self.tick_count = 0

        # ── ROS ────────────────────────────────────────────────
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos)
        self.create_subscription(Float32MultiArray, '/obstacles', self._obs_cb, qos)

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos)
        self.err_pub = self.create_publisher(Float32, '/hybrid/tracking_error', qos)
        self.wgt_pub = self.create_publisher(Float32, '/hybrid/blend_weight', qos)
        self.mode_pub = self.create_publisher(String, '/hybrid/controller_mode', qos)
        self.ref_pub = self.create_publisher(Path, '/hybrid/reference_path', 1)

        self.create_timer(1.0 / rate, self._control_loop)
        self.create_timer(3.0, self._pub_ref_path_once)
        self._ref_published = False

        self.get_logger().info(
            f'🤖 Turn-then-Drive controller | '
            f'A={self.A}m, {len(self.wp_x)} waypoints, '
            f'cruise={self.cruise_speed}m/s, lookahead={self.lookahead}m'
        )

    # ── Generate Waypoints ─────────────────────────────────────
    def _generate_waypoints(self, duration):
        """Generate figure-8 trajectory as dense array for path display,
        and sparse waypoints (every ~0.05m) for tracking."""
        t = np.arange(0, duration, self.dt)
        w = self.freq

        # Dense trajectory for visualization
        self.dense_x = self.A * np.sin(w * t)
        self.dense_y = (self.A / 2.0) * np.sin(2.0 * w * t)
        dx_dt = self.A * w * np.cos(w * t)
        dy_dt = self.A * w * np.cos(2.0 * w * t)
        self.dense_theta = np.arctan2(dy_dt, dx_dt)
        self.dense_v = np.clip(np.sqrt(dx_dt**2 + dy_dt**2), 0.0, self.v_max)
        self.dense_N = len(t)

        # Sparse waypoints: pick points spaced ~0.05m apart along the path
        wp_x, wp_y, wp_theta = [self.dense_x[0]], [self.dense_y[0]], [self.dense_theta[0]]
        for i in range(1, len(t)):
            d = np.sqrt(
                (self.dense_x[i] - wp_x[-1])**2 +
                (self.dense_y[i] - wp_y[-1])**2
            )
            if d >= 0.05:  # ~5cm spacing
                wp_x.append(self.dense_x[i])
                wp_y.append(self.dense_y[i])
                wp_theta.append(self.dense_theta[i])

        self.wp_x = np.array(wp_x)
        self.wp_y = np.array(wp_y)
        self.wp_theta = np.array(wp_theta)
        self.n_wp = len(self.wp_x)

        self.get_logger().info(
            f'📐 {self.n_wp} waypoints from {len(t)} dense points | '
            f'v_max_ref={np.sqrt(dx_dt**2 + dy_dt**2).max():.3f} m/s'
        )

    # ── Callbacks ──────────────────────────────────────────────
    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = euler_from_quaternion(msg.pose.pose.orientation)
        if not self.odom_ok:
            self.odom_ok = True
            self.get_logger().info(
                f'📡 Odom OK: ({self.x:.3f}, {self.y:.3f}) θ={self.theta:.2f}rad'
            )

    def _obs_cb(self, msg):
        self.obstacles = []
        for i in range(0, len(msg.data), 3):
            if i + 2 < len(msg.data):
                self.obstacles.append({
                    'x': float(msg.data[i]),
                    'y': float(msg.data[i+1]),
                    'radius': float(msg.data[i+2]),
                })

    # ── Main Control Loop ──────────────────────────────────────
    def _control_loop(self):
        if not self.odom_ok:
            return

        # Done?
        if self.wp_idx >= self.n_wp:
            self._send(0.0, 0.0)
            if self.tick_count % 100 == 0:
                self.get_logger().info('⏹️ All waypoints reached.')
            self.tick_count += 1
            return

        # Current waypoint target
        tx = self.wp_x[self.wp_idx]
        ty = self.wp_y[self.wp_idx]

        # Distance and angle to waypoint
        dx = tx - self.x
        dy = ty - self.y
        dist = np.sqrt(dx**2 + dy**2)
        target_heading = np.arctan2(dy, dx)
        heading_err = normalize_angle(target_heading - self.theta)

        # ── Waypoint advancement ───────────────────────────────
        # If close enough to current waypoint, advance
        if dist < 0.08:
            self.wp_idx += 1
            if self.wp_idx < self.n_wp:
                tx = self.wp_x[self.wp_idx]
                ty = self.wp_y[self.wp_idx]
                dx = tx - self.x
                dy = ty - self.y
                dist = np.sqrt(dx**2 + dy**2)
                target_heading = np.arctan2(dy, dx)
                heading_err = normalize_angle(target_heading - self.theta)

        # ── Turn-then-Drive logic ──────────────────────────────
        if abs(heading_err) > self.turn_threshold:
            # PHASE: TURN in place
            v = 0.0
            omega = 1.5 * heading_err  # Proportional turn
            omega = np.clip(omega, -self.omega_max * 0.5, self.omega_max * 0.5)
            mode = 'TURN'
        else:
            # PHASE: DRIVE with proportional steering
            v = self.cruise_speed
            # Reduce speed if there's moderate heading error
            v *= max(0.3, 1.0 - 2.0 * abs(heading_err))
            omega = 2.0 * heading_err  # Proportional steering correction
            omega = np.clip(omega, -self.omega_max * 0.3, self.omega_max * 0.3)
            mode = 'DRIVE'

        v = float(np.clip(v, 0.0, self.v_max))
        omega = float(omega)

        # ── MPC obstacle avoidance overlay ─────────────────────
        w = 0.0
        if self.controllers_ok and self.use_hybrid and len(self.obstacles) > 0:
            try:
                x0 = np.array([self.x, self.y, self.theta])
                dist_risk, _, _ = self.risk_calc.compute_distance_risk(x0, self.obstacles)
                blend_info = self.blender.compute_weight(dist_risk)
                w = blend_info.weight
                if w > 0.1:
                    mode = f'AVOID(w={w:.2f})'
            except Exception:
                pass

        self._send(v, omega)

        # Diagnostics
        self._pub_diag(dist, w, mode)

        # Log every ~1 second
        if self.tick_count % 20 == 0:
            self.get_logger().info(
                f'[wp={self.wp_idx}/{self.n_wp}] '
                f'pos=({self.x:+.2f},{self.y:+.2f}) θ={self.theta:+.2f} | '
                f'tgt=({tx:+.2f},{ty:+.2f}) | '
                f'd={dist:.3f}m Δθ={heading_err:+.2f} | '
                f'v={v:.3f} ω={omega:+.3f} | {mode}'
            )

        self.tick_count += 1

    # ── Publish helpers ────────────────────────────────────────
    def _send(self, v, omega):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

    def _pub_diag(self, err, w, mode):
        e = Float32(); e.data = float(err); self.err_pub.publish(e)
        wm = Float32(); wm.data = float(w); self.wgt_pub.publish(wm)
        m = String(); m.data = mode; self.mode_pub.publish(m)

    def _pub_ref_path_once(self):
        if self._ref_published:
            return
        self._ref_published = True
        p = Path()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = 'odom'
        for i in range(0, self.dense_N, 5):
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(self.dense_x[i])
            ps.pose.position.y = float(self.dense_y[i])
            ps.pose.orientation.z = float(np.sin(self.dense_theta[i] / 2))
            ps.pose.orientation.w = float(np.cos(self.dense_theta[i] / 2))
            p.poses.append(ps)
        self.ref_pub.publish(p)
        self.get_logger().info(f'📍 Ref path: {len(p.poses)} poses')


def main(args=None):
    rclpy.init(args=args)
    node = HybridControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._send(0.0, 0.0)
        node.get_logger().info('Stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
