#!/usr/bin/env python3
"""
Hybrid LQR-MPC Controller Node for TurtleBot3
==============================================

Subscribes to /odom for robot state, runs the hybrid LQR-MPC-Blending
controller, and publishes velocity commands to /cmd_vel.

Key design:
  - Uses SIMULATION TIME to index into trajectory (not callback count)
  - Publishes TwistStamped (required by TurtleBot3 Gazebo bridge)
  - Verbose logging for debugging

Topics:
    Subscribes: /odom, /obstacles
    Publishes:  /cmd_vel, /hybrid/tracking_error, /hybrid/blend_weight,
                /hybrid/controller_mode, /hybrid/reference_path, /hybrid/predicted_path
"""

import numpy as np
import time as wall_time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, String, Float32MultiArray


def euler_from_quaternion(q):
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)


def normalize_angle(a):
    """Normalize angle to [-pi, pi]."""
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

        control_rate = self.get_parameter('control_rate').value
        self.A = self.get_parameter('trajectory_amplitude').value
        self.freq = self.get_parameter('trajectory_frequency').value
        duration = self.get_parameter('trajectory_duration').value
        self.dt = self.get_parameter('dt').value
        horizon = self.get_parameter('mpc_horizon').value
        d_safe = self.get_parameter('d_safe').value
        self.v_max = self.get_parameter('v_max').value
        self.omega_max = self.get_parameter('omega_max').value
        self.use_hybrid = self.get_parameter('use_hybrid').value

        # ── Import Controllers ─────────────────────────────────
        self.controllers_ok = False
        try:
            from hybrid_nav.controllers.lqr_controller import LQRController
            from hybrid_nav.controllers.mpc_controller import MPCController, Obstacle
            from hybrid_nav.controllers.hybrid_blender import BlendingSupervisor
            from hybrid_nav.controllers.risk_metrics import RiskMetrics

            self.lqr = LQRController(
                Q_diag=[10.0, 10.0, 1.0],
                R_diag=[1.0, 0.5],       # Higher R = less aggressive control
                v_max=self.v_max,
                omega_max=self.omega_max,
                dt=self.dt,
            )
            self.mpc = MPCController(
                horizon=horizon,
                d_safe=d_safe,
                v_max=self.v_max,
                omega_max=self.omega_max,
                dt=self.dt,
            )
            self.blender = BlendingSupervisor(dt=self.dt)
            self.risk_calc = RiskMetrics(d_trigger=1.5, d_safe=d_safe)
            self.Obstacle = Obstacle  # Store class for later use
            self.controllers_ok = True
            self.get_logger().info('✅ All controllers imported successfully')
        except Exception as e:
            self.get_logger().error(f'❌ Controller import failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.get_logger().warn('⚠️ Running in FALLBACK mode (P-control)')

        # ── Generate Trajectory ────────────────────────────────
        self._generate_trajectory(duration)

        # ── State ──────────────────────────────────────────────
        self.current_state = np.zeros(3)  # [x, y, theta]
        self.state_received = False
        self.obstacles_dict = []
        self.start_time = None  # Set on first odom
        self.callback_count = 0

        # ── ROS Setup ──────────────────────────────────────────
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos)
        self.obs_sub = self.create_subscription(
            Float32MultiArray, '/obstacles', self.obstacle_callback, qos)

        # TurtleBot3 Gazebo bridge requires TwistStamped
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos)
        self.error_pub = self.create_publisher(Float32, '/hybrid/tracking_error', qos)
        self.weight_pub = self.create_publisher(Float32, '/hybrid/blend_weight', qos)
        self.mode_pub = self.create_publisher(String, '/hybrid/controller_mode', qos)
        self.ref_path_pub = self.create_publisher(Path, '/hybrid/reference_path', 1)
        self.pred_path_pub = self.create_publisher(Path, '/hybrid/predicted_path', qos)

        self.control_timer = self.create_timer(1.0 / control_rate, self.control_callback)

        # Delayed reference path publish
        self.create_timer(3.0, self._publish_reference_path_once)
        self._ref_path_published = False

        self.get_logger().info(
            f'🤖 Controller started | rate={control_rate}Hz | '
            f'hybrid={self.use_hybrid} | v_max={self.v_max} | '
            f'A={self.A}m freq={self.freq}'
        )

    # ── Trajectory Generation ──────────────────────────────────
    def _generate_trajectory(self, duration):
        """Generate figure-8 reference with velocity-limited parameterization."""
        t = np.arange(0, duration, self.dt)
        N = len(t)

        self.ref_x = self.A * np.sin(self.freq * t)
        self.ref_y = self.A * np.sin(self.freq * t) * np.cos(self.freq * t)

        # Velocity via analytical derivatives (not np.gradient edge effects)
        dx = self.A * self.freq * np.cos(self.freq * t)
        # y = A * sin(wt) * cos(wt) = (A/2) * sin(2wt)
        dy = self.A * self.freq * np.cos(2.0 * self.freq * t)

        self.ref_theta = np.arctan2(dy, dx)
        # Don't unwrap — keep angles in [-pi, pi] for proper error computation
        self.ref_v = np.sqrt(dx**2 + dy**2)
        self.ref_omega = np.gradient(self.ref_theta, self.dt)
        # Clean omega discontinuities from atan2 jumps
        self.ref_omega = np.clip(self.ref_omega, -self.omega_max, self.omega_max)

        self.traj_len = N
        self.traj_duration = duration

        self.get_logger().info(
            f'📐 Trajectory: {N} points, {duration}s | '
            f'v_max_ref={self.ref_v.max():.3f} m/s | '
            f'v_mean_ref={self.ref_v.mean():.3f} m/s'
        )

    def _get_traj_index(self):
        """Get trajectory index from elapsed simulation time."""
        if self.start_time is None:
            return 0
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9
        idx = int(elapsed / self.dt)
        return min(idx, self.traj_len - 1)

    def _get_refs(self, idx, horizon):
        """Get reference trajectory window for MPC."""
        N = horizon + 1
        x_refs = np.zeros((N, 3))
        u_refs = np.zeros((N - 1, 2))
        for k in range(N):
            i = min(idx + k, self.traj_len - 1)
            x_refs[k] = [self.ref_x[i], self.ref_y[i], self.ref_theta[i]]
        for k in range(N - 1):
            i = min(idx + k, self.traj_len - 2)
            u_refs[k] = [self.ref_v[i], self.ref_omega[i]]
        return x_refs, u_refs

    # ── Callbacks ──────────────────────────────────────────────
    def odom_callback(self, msg):
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        self.current_state[2] = euler_from_quaternion(msg.pose.pose.orientation)
        if not self.state_received:
            self.state_received = True
            self.start_time = self.get_clock().now()
            self.get_logger().info(
                f'📡 First odom received: pos=({self.current_state[0]:.2f}, '
                f'{self.current_state[1]:.2f}), theta={self.current_state[2]:.2f}'
            )

    def obstacle_callback(self, msg):
        self.obstacles_dict = []
        data = msg.data
        for i in range(0, len(data), 3):
            if i + 2 < len(data):
                self.obstacles_dict.append({
                    'x': float(data[i]),
                    'y': float(data[i + 1]),
                    'radius': float(data[i + 2]),
                })

    def control_callback(self):
        if not self.state_received:
            return

        # Use simulation time to index trajectory
        idx = self._get_traj_index()

        if idx >= self.traj_len - 1:
            self._stop_robot()
            return

        x0 = self.current_state.copy()
        x_ref = np.array([self.ref_x[idx], self.ref_y[idx], self.ref_theta[idx]])
        u_ref = np.array([self.ref_v[idx], self.ref_omega[idx]])

        # Position tracking error
        pos_err = float(np.linalg.norm(x0[:2] - x_ref[:2]))
        theta_err = normalize_angle(x0[2] - x_ref[2])

        # Select controller
        if self.controllers_ok and self.use_hybrid:
            v, omega, w, mode = self._run_hybrid(idx, x0, x_ref, u_ref)
        elif self.controllers_ok:
            v, omega, w, mode = self._run_lqr_only(x0, x_ref, u_ref)
        else:
            v, omega, w, mode = self._run_fallback(x0, x_ref)

        # Publish command
        self._publish_cmd(v, omega)
        self._publish_diagnostics(pos_err, w, mode)

        # Verbose logging every ~1 second
        self.callback_count += 1
        if self.callback_count % 20 == 0:
            self.get_logger().info(
                f'[t={idx:4d}] pos=({x0[0]:+.2f},{x0[1]:+.2f}) θ={x0[2]:+.2f} | '
                f'ref=({x_ref[0]:+.2f},{x_ref[1]:+.2f}) θr={x_ref[2]:+.2f} | '
                f'err={pos_err:.3f}m Δθ={theta_err:+.2f} | '
                f'v={v:+.3f} ω={omega:+.3f} | {mode}'
            )

    # ── Controller Modes ───────────────────────────────────────
    def _run_hybrid(self, idx, x0, x_ref, u_ref):
        """Hybrid LQR-MPC-Blending control."""
        horizon = self.mpc.N
        x_refs, u_refs = self._get_refs(idx, horizon)

        # Risk assessment
        dist_risk, _, _ = self.risk_calc.compute_distance_risk(x0, self.obstacles_dict)
        blend_info = self.blender.compute_weight(dist_risk)
        w = blend_info.weight

        # LQR
        u_lqr = self._compute_lqr(x0, x_ref, u_ref)

        # MPC (only when risk is significant)
        if w > 0.05:
            try:
                mpc_obstacles = [
                    self.Obstacle(x=o['x'], y=o['y'], radius=o['radius'])
                    for o in self.obstacles_dict
                ]
                mpc_sol = self.mpc.solve(x0, x_refs, u_refs, mpc_obstacles)
                u_mpc = mpc_sol.optimal_control
                if mpc_sol.predicted_states is not None:
                    self._publish_predicted_path(mpc_sol.predicted_states)
            except Exception as e:
                self.get_logger().warn(f'MPC err: {e}', throttle_duration_sec=5.0)
                u_mpc = u_lqr
        else:
            u_mpc = u_lqr

        u = w * u_mpc + (1.0 - w) * u_lqr
        v = float(np.clip(u[0], -self.v_max, self.v_max))
        omega = float(np.clip(u[1], -self.omega_max, self.omega_max))
        return v, omega, w, f'HYBRID(w={w:.2f})'

    def _run_lqr_only(self, x0, x_ref, u_ref):
        """LQR-only control."""
        u = self._compute_lqr(x0, x_ref, u_ref)
        v = float(np.clip(u[0], -self.v_max, self.v_max))
        omega = float(np.clip(u[1], -self.omega_max, self.omega_max))
        return v, omega, 0.0, 'LQR'

    def _compute_lqr(self, x0, x_ref, u_ref):
        """Compute LQR control with gain recomputation."""
        v_r = u_ref[0] if abs(u_ref[0]) > 0.01 else 0.01
        self.lqr.compute_gain(v_r, x_ref[2])
        return self.lqr.compute_control(x0, x_ref, u_ref)

    def _run_fallback(self, x0, x_ref):
        """Simple P-control fallback."""
        dx = x_ref[0] - x0[0]
        dy = x_ref[1] - x0[1]
        dist = np.sqrt(dx**2 + dy**2)
        target_theta = np.arctan2(dy, dx)
        theta_err = normalize_angle(target_theta - x0[2])

        # Turn first, then drive
        if abs(theta_err) > 0.3:
            v = 0.0
            omega = float(np.clip(2.0 * theta_err, -self.omega_max, self.omega_max))
        else:
            v = float(np.clip(0.15 * dist, 0.0, self.v_max))
            omega = float(np.clip(1.5 * theta_err, -self.omega_max, self.omega_max))
        return v, omega, 0.0, 'FALLBACK'

    # ── Publishing ─────────────────────────────────────────────
    def _publish_cmd(self, v, omega):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = v
        cmd.twist.angular.z = omega
        self.cmd_pub.publish(cmd)

    def _stop_robot(self):
        self._publish_cmd(0.0, 0.0)
        if self.callback_count % 100 == 0:
            self.get_logger().info('⏹️ Trajectory complete, robot stopped.')

    def _publish_diagnostics(self, tracking_error, w, mode):
        e = Float32(); e.data = tracking_error; self.error_pub.publish(e)
        wm = Float32(); wm.data = float(w); self.weight_pub.publish(wm)
        m = String(); m.data = mode; self.mode_pub.publish(m)

    def _publish_reference_path_once(self):
        if self._ref_path_published:
            return
        self._ref_path_published = True
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
        for i in range(0, self.traj_len, 10):
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(self.ref_x[i])
            ps.pose.position.y = float(self.ref_y[i])
            ps.pose.orientation.z = float(np.sin(self.ref_theta[i] / 2))
            ps.pose.orientation.w = float(np.cos(self.ref_theta[i] / 2))
            path_msg.poses.append(ps)
        self.ref_path_pub.publish(path_msg)
        self.get_logger().info(f'📍 Reference path published ({len(path_msg.poses)} poses)')

    def _publish_predicted_path(self, states):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
        for s in states:
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(s[0])
            ps.pose.position.y = float(s[1])
            ps.pose.orientation.z = float(np.sin(s[2] / 2))
            ps.pose.orientation.w = float(np.cos(s[2] / 2))
            path_msg.poses.append(ps)
        self.pred_path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HybridControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._stop_robot()
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
