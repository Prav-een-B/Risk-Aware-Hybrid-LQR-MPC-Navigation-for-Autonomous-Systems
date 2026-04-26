#!/usr/bin/env python3
"""
Hybrid LQR-MPC Controller Node for TurtleBot3
==============================================

Uses a proven pure-pursuit base controller for reliable trajectory tracking,
with optional LQR/MPC/Blending overlay when controllers are available.

The pure-pursuit ensures the robot ALWAYS follows the path regardless of
controller tuning issues.
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
    return (a + np.pi) % (2 * np.pi) - np.pi


class HybridControllerNode(Node):
    def __init__(self):
        super().__init__('hybrid_controller')

        # Parameters
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
        self.declare_parameter('lookahead_dist', 0.15)    # Pure pursuit lookahead
        self.declare_parameter('base_speed', 0.12)        # Cruise speed

        rate = self.get_parameter('control_rate').value
        self.A = self.get_parameter('trajectory_amplitude').value
        self.freq = self.get_parameter('trajectory_frequency').value
        dur = self.get_parameter('trajectory_duration').value
        self.dt = self.get_parameter('dt').value
        self.v_max = self.get_parameter('v_max').value
        self.omega_max = self.get_parameter('omega_max').value
        self.use_hybrid = self.get_parameter('use_hybrid').value
        self.lookahead = self.get_parameter('lookahead_dist').value
        self.base_speed = self.get_parameter('base_speed').value

        # Try importing advanced controllers
        self.controllers_ok = False
        try:
            from hybrid_nav.controllers.lqr_controller import LQRController
            from hybrid_nav.controllers.mpc_controller import MPCController, Obstacle
            from hybrid_nav.controllers.hybrid_blender import BlendingSupervisor
            from hybrid_nav.controllers.risk_metrics import RiskMetrics

            self.lqr = LQRController(
                Q_diag=[10.0, 10.0, 1.0],
                R_diag=[1.0, 0.5],
                v_max=self.v_max, omega_max=self.omega_max, dt=self.dt,
            )
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
            self.get_logger().info('✅ Advanced controllers loaded')
        except Exception as e:
            self.get_logger().warn(f'⚠️ Advanced controllers not available: {e}')
            self.get_logger().info('Using pure-pursuit baseline controller')

        # Generate trajectory
        self._generate_trajectory(dur)

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.odom_received = False
        self.obstacles_dict = []
        self.start_time = None
        self.tick = 0
        self.closest_idx = 0  # Track closest point for smooth progression

        # ROS
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos)
        self.create_subscription(Float32MultiArray, '/obstacles', self._obs_cb, qos)

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos)
        self.err_pub = self.create_publisher(Float32, '/hybrid/tracking_error', qos)
        self.wgt_pub = self.create_publisher(Float32, '/hybrid/blend_weight', qos)
        self.mode_pub = self.create_publisher(String, '/hybrid/controller_mode', qos)
        self.ref_pub = self.create_publisher(Path, '/hybrid/reference_path', 1)
        self.pred_pub = self.create_publisher(Path, '/hybrid/predicted_path', qos)

        self.create_timer(1.0 / rate, self._control_loop)
        self.create_timer(3.0, self._pub_ref_path_once)
        self._ref_published = False

        self.get_logger().info(
            f'🤖 Ready | A={self.A}m freq={self.freq} '
            f'v_max={self.v_max} lookahead={self.lookahead}m'
        )

    # ── Trajectory ─────────────────────────────────────────────
    def _generate_trajectory(self, duration):
        t = np.arange(0, duration, self.dt)
        N = len(t)
        w = self.freq

        self.traj_x = self.A * np.sin(w * t)
        self.traj_y = (self.A / 2.0) * np.sin(2.0 * w * t)  # Clean figure-8

        # Analytical derivatives
        dx = self.A * w * np.cos(w * t)
        dy = self.A * w * np.cos(2.0 * w * t)
        self.traj_theta = np.arctan2(dy, dx)
        self.traj_v = np.clip(np.sqrt(dx**2 + dy**2), 0.0, self.v_max)
        self.N = N

        v_max_ref = np.sqrt(dx**2 + dy**2).max()
        self.get_logger().info(
            f'📐 Trajectory: {N} pts, {duration}s, v_max_ref={v_max_ref:.3f} m/s'
        )

    # ── Core: Find nearest + lookahead point ───────────────────
    def _find_target(self):
        """Find the lookahead target point on the trajectory.
        
        Uses nearest-point tracking with forward lookahead for smooth pursuit.
        """
        px, py = self.x, self.y

        # Search near the last closest index (± window for efficiency)
        search_start = max(0, self.closest_idx - 20)
        search_end = min(self.N, self.closest_idx + 100)

        # Find nearest point
        dx = self.traj_x[search_start:search_end] - px
        dy = self.traj_y[search_start:search_end] - py
        dists = np.sqrt(dx**2 + dy**2)
        local_min = np.argmin(dists)
        self.closest_idx = search_start + local_min
        nearest_dist = dists[local_min]

        # Find lookahead point (walk forward from nearest until distance >= lookahead)
        target_idx = self.closest_idx
        for i in range(self.closest_idx, min(self.closest_idx + 60, self.N)):
            d = np.sqrt((self.traj_x[i] - px)**2 + (self.traj_y[i] - py)**2)
            if d >= self.lookahead:
                target_idx = i
                break
        else:
            target_idx = min(self.closest_idx + 10, self.N - 1)

        return target_idx, nearest_dist

    # ── Pure Pursuit Controller ────────────────────────────────
    def _pure_pursuit(self, target_idx):
        """Compute v, omega using pure pursuit geometry."""
        tx = self.traj_x[target_idx]
        ty = self.traj_y[target_idx]

        # Vector from robot to target in world frame
        dx = tx - self.x
        dy = ty - self.y
        dist = np.sqrt(dx**2 + dy**2)

        # Desired heading
        target_theta = np.arctan2(dy, dx)
        alpha = normalize_angle(target_theta - self.theta)

        # Pure pursuit curvature: kappa = 2*sin(alpha) / L
        if dist > 0.01:
            curvature = 2.0 * np.sin(alpha) / max(dist, self.lookahead)
        else:
            curvature = 0.0

        # Speed: slow down for sharp turns and near-target
        speed = self.base_speed
        if abs(alpha) > 0.5:
            speed *= 0.5  # Slow down for sharp turns
        if abs(alpha) > 1.2:
            speed *= 0.3  # Near U-turn, go very slow

        omega = speed * curvature

        # Clamp
        v = float(np.clip(speed, 0.0, self.v_max))
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))

        return v, omega

    # ── Callbacks ──────────────────────────────────────────────
    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = euler_from_quaternion(msg.pose.pose.orientation)
        if not self.odom_received:
            self.odom_received = True
            self.start_time = self.get_clock().now()
            self.get_logger().info(
                f'📡 Odom: ({self.x:.2f}, {self.y:.2f}) θ={self.theta:.2f}'
            )

    def _obs_cb(self, msg):
        self.obstacles_dict = []
        for i in range(0, len(msg.data), 3):
            if i + 2 < len(msg.data):
                self.obstacles_dict.append({
                    'x': float(msg.data[i]),
                    'y': float(msg.data[i+1]),
                    'radius': float(msg.data[i+2]),
                })

    # ── Main Control Loop ──────────────────────────────────────
    def _control_loop(self):
        if not self.odom_received:
            return

        # Check if trajectory complete
        if self.closest_idx >= self.N - 5:
            self._send_cmd(0.0, 0.0)
            if self.tick % 100 == 0:
                self.get_logger().info('⏹️ Trajectory complete.')
            self.tick += 1
            return

        # Find target on trajectory
        target_idx, tracking_err = self._find_target()

        # ─── BASE: Pure pursuit (always works) ─────────────
        v_pp, omega_pp = self._pure_pursuit(target_idx)

        # ─── OVERLAY: Hybrid LQR-MPC (if available + active) ─
        w = 0.0
        mode = 'PURSUIT'

        if self.controllers_ok and self.use_hybrid and len(self.obstacles_dict) > 0:
            try:
                x0 = np.array([self.x, self.y, self.theta])
                # Risk check
                dist_risk, _, _ = self.risk_calc.compute_distance_risk(
                    x0, self.obstacles_dict
                )
                blend_info = self.blender.compute_weight(dist_risk)
                w = blend_info.weight

                if w > 0.1:
                    # MPC for obstacle avoidance
                    horizon = self.mpc.N
                    x_refs = np.zeros((horizon + 1, 3))
                    u_refs = np.zeros((horizon, 2))
                    for k in range(horizon + 1):
                        i = min(target_idx + k, self.N - 1)
                        x_refs[k] = [self.traj_x[i], self.traj_y[i], self.traj_theta[i]]
                    for k in range(horizon):
                        i = min(target_idx + k, self.N - 2)
                        u_refs[k] = [self.traj_v[i], 0.0]

                    mpc_obs = [
                        self.Obstacle(x=o['x'], y=o['y'], radius=o['radius'])
                        for o in self.obstacles_dict
                    ]
                    sol = self.mpc.solve(x0, x_refs, u_refs, mpc_obs)
                    u_mpc = sol.optimal_control

                    # Blend MPC with pure pursuit
                    v_blend = w * u_mpc[0] + (1.0 - w) * v_pp
                    omega_blend = w * u_mpc[1] + (1.0 - w) * omega_pp
                    v_pp = float(np.clip(v_blend, -self.v_max, self.v_max))
                    omega_pp = float(np.clip(omega_blend, -self.omega_max, self.omega_max))
                    mode = f'HYBRID(w={w:.2f})'

                    if sol.predicted_states is not None:
                        self._pub_predicted_path(sol.predicted_states)
            except Exception as e:
                if self.tick % 40 == 0:
                    self.get_logger().warn(f'MPC: {e}')

        # Send command
        self._send_cmd(v_pp, omega_pp)

        # Diagnostics
        e = Float32(); e.data = tracking_err; self.err_pub.publish(e)
        wm = Float32(); wm.data = float(w); self.wgt_pub.publish(wm)
        m = String(); m.data = mode; self.mode_pub.publish(m)

        # Log every ~1 second
        if self.tick % 20 == 0:
            tgt = target_idx
            self.get_logger().info(
                f'[{tgt:4d}] pos=({self.x:+.2f},{self.y:+.2f}) θ={self.theta:+.2f} | '
                f'tgt=({self.traj_x[tgt]:+.2f},{self.traj_y[tgt]:+.2f}) | '
                f'err={tracking_err:.3f}m | v={v_pp:.3f} ω={omega_pp:+.3f} | {mode}'
            )

        self.tick += 1

    # ── Publish ────────────────────────────────────────────────
    def _send_cmd(self, v, omega):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

    def _pub_ref_path_once(self):
        if self._ref_published:
            return
        self._ref_published = True
        p = Path()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = 'odom'
        for i in range(0, self.N, 5):
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(self.traj_x[i])
            ps.pose.position.y = float(self.traj_y[i])
            ps.pose.orientation.z = float(np.sin(self.traj_theta[i] / 2))
            ps.pose.orientation.w = float(np.cos(self.traj_theta[i] / 2))
            p.poses.append(ps)
        self.ref_pub.publish(p)
        self.get_logger().info(f'📍 Ref path: {len(p.poses)} poses')

    def _pub_predicted_path(self, states):
        p = Path()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = 'odom'
        for s in states:
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(s[0])
            ps.pose.position.y = float(s[1])
            ps.pose.orientation.z = float(np.sin(s[2] / 2))
            ps.pose.orientation.w = float(np.cos(s[2] / 2))
            p.poses.append(ps)
        self.pred_pub.publish(p)


def main(args=None):
    rclpy.init(args=args)
    node = HybridControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._send_cmd(0.0, 0.0)
        node.get_logger().info('Stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
