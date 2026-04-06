#!/usr/bin/env python3
"""
Hybrid controller ROS2 node.

This node exposes the risk-aware LQR-MPC blending logic through ROS2 so the
controller can be launched inside the containerized Gazebo workflow.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32, Float32MultiArray, String

from ..controllers.hybrid_blender import BlendingSupervisor
from ..controllers.lqr_controller import LQRController
from ..controllers.mpc_controller import MPCController, Obstacle
from ..controllers.risk_metrics import RiskMetrics
from ..logging.simulation_logger import SimulationLogger
from ..trajectory.reference_generator import ReferenceTrajectoryGenerator


class HybridControllerNode(Node):
    """ROS2 wrapper for the smooth hybrid LQR-MPC controller."""

    def __init__(self):
        super().__init__("hybrid_controller")

        self.declare_parameter("trajectory_type", "figure8")
        self.declare_parameter("checkpoint_preset", "diamond")
        self.declare_parameter("trajectory_duration", 30.0)
        self.declare_parameter("amplitude", 2.0)
        self.declare_parameter("frequency", 0.5)
        self.declare_parameter("nominal_speed", 0.8)
        self.declare_parameter("dt", 0.02)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("mpc_rate_divisor", 5)
        self.declare_parameter("v_max", 2.0)
        self.declare_parameter("omega_max", 3.0)
        self.declare_parameter("log_level", "INFO")

        dt = float(self.get_parameter("dt").value)
        control_rate = float(self.get_parameter("control_rate").value)
        self.mpc_rate_divisor = max(1, int(self.get_parameter("mpc_rate_divisor").value))

        trajectory_type = str(self.get_parameter("trajectory_type").value)
        checkpoint_preset = str(self.get_parameter("checkpoint_preset").value)
        trajectory_duration = float(self.get_parameter("trajectory_duration").value)
        amplitude = float(self.get_parameter("amplitude").value)
        frequency = float(self.get_parameter("frequency").value)
        nominal_speed = float(self.get_parameter("nominal_speed").value)
        v_max = float(self.get_parameter("v_max").value)
        omega_max = float(self.get_parameter("omega_max").value)
        log_level = str(self.get_parameter("log_level").value)

        self.lqr = LQRController(
            Q_diag=[15.0, 15.0, 8.0],
            R_diag=[0.1, 0.1],
            dt=dt,
            v_max=v_max,
            omega_max=omega_max,
        )
        self.mpc = MPCController(
            horizon=5,
            Q_diag=[80.0, 80.0, 120.0],
            R_diag=[0.1, 0.1],
            P_diag=[20.0, 20.0, 40.0],
            S_diag=[0.1, 0.5],
            J_diag=[0.05, 0.3],
            d_safe=0.3,
            slack_penalty=5000.0,
            dt=dt,
            v_max=v_max,
            omega_max=omega_max,
            solver="OSQP",
            block_size=2,
            w_max=0.05,
        )
        self.risk_metrics = RiskMetrics(
            d_safe=0.3,
            d_trigger=1.0,
            alpha=0.6,
            beta=0.4,
            threshold_low=0.2,
            threshold_medium=0.5,
        )
        self.blender = BlendingSupervisor(
            k_sigmoid=10.0,
            risk_threshold=0.3,
            dw_max=2.0,
            hysteresis_band=0.05,
            solver_time_limit=5.0,
            feasibility_decay=0.8,
            dt=dt,
        )
        self.traj_gen = ReferenceTrajectoryGenerator(
            A=amplitude,
            a=frequency,
            dt=dt,
            T_blend=0.5,
            trajectory_type=trajectory_type,
            checkpoint_preset=checkpoint_preset,
            nominal_speed=nominal_speed,
        )
        self.trajectory = self.traj_gen.generate(trajectory_duration)

        self.logger = SimulationLogger(
            log_dir="logs",
            log_level=log_level,
            node_name="hybrid_controller",
        )

        self.current_state = np.zeros(3)
        self.obstacles = []
        self.obstacle_dicts = []
        self.state_received = False
        self.current_traj_idx = 0
        self.timestep = 0
        self.mpc_solution = None

        qos = QoSProfile(depth=10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos)
        self.obstacle_sub = self.create_subscription(
            Float32MultiArray, "/mpc_obstacles", self.obstacle_callback, qos
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", qos)
        self.weight_pub = self.create_publisher(Float32, "/hybrid/blend_weight", qos)
        self.risk_pub = self.create_publisher(Float32, "/hybrid/risk", qos)
        self.mode_pub = self.create_publisher(String, "/hybrid/mode", qos)
        self.path_pub = self.create_publisher(Path, "/hybrid/predicted_path", qos)
        self.ref_pub = self.create_publisher(PoseStamped, "/hybrid/reference_pose", qos)

        self.control_timer = self.create_timer(1.0 / control_rate, self.control_callback)

        self.get_logger().info(
            "Hybrid controller initialized with %s trajectory (%d samples)"
            % (trajectory_type, len(self.trajectory))
        )

    def odom_callback(self, msg: Odometry):
        """Update the current state estimate from odometry."""
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_state[2] = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.state_received = True

    def obstacle_callback(self, msg: Float32MultiArray):
        """Update static obstacle data used by the hybrid controller."""
        self.obstacles = []
        self.obstacle_dicts = []
        data = list(msg.data)

        for idx in range(0, len(data), 3):
            if idx + 2 >= len(data):
                break
            obstacle = Obstacle(x=float(data[idx]), y=float(data[idx + 1]), radius=float(data[idx + 2]))
            self.obstacles.append(obstacle)
            self.obstacle_dicts.append(
                {"x": obstacle.x, "y": obstacle.y, "radius": obstacle.radius}
            )

    def control_callback(self):
        """Run one hybrid control step."""
        if not self.state_received:
            return

        if self.current_traj_idx >= len(self.trajectory) - 1:
            self.publish_stop()
            return

        x_ref, u_ref = self.traj_gen.get_reference_at_index(self.current_traj_idx)
        x_refs, u_refs = self.traj_gen.get_trajectory_segment(
            self.current_traj_idx, self.mpc.N + 1
        )

        u_lqr, error = self.lqr.compute_control_at_operating_point(
            self.current_state, x_ref, u_ref
        )

        solver_status = "optimal"
        solver_time_ms = 0.0
        feasibility_margin = 0.0
        predicted_states = None

        if self.timestep % self.mpc_rate_divisor == 0:
            self.mpc_solution = self.mpc.solve_with_ltv(
                self.current_state, x_refs, u_refs, self.obstacles
            )

        if self.mpc_solution is not None:
            solver_status = self.mpc_solution.status
            solver_time_ms = self.mpc_solution.solve_time_ms
            feasibility_margin = self.mpc_solution.feasibility_margin
            predicted_states = self.mpc_solution.predicted_states
            u_mpc = self.mpc_solution.optimal_control
        else:
            u_mpc = u_lqr

        assessment = self.risk_metrics.assess_risk(
            self.current_state, self.obstacle_dicts, predicted_states=predicted_states
        )
        u_blend, blend_info = self.blender.blend(
            u_lqr=u_lqr,
            u_mpc=u_mpc,
            risk=assessment.combined_risk,
            solver_status=solver_status,
            solver_time_ms=solver_time_ms,
            feasibility_margin=feasibility_margin,
        )

        self.publish_command(u_blend)
        self.publish_diagnostics(blend_info.weight, assessment.combined_risk, blend_info.mode)
        self.publish_reference_pose(x_ref)

        if predicted_states is not None:
            self.publish_predicted_path(predicted_states)

        self.logger.log_state(
            timestep=self.timestep,
            state=self.current_state,
            state_ref=x_ref,
            error=error,
        )
        self.logger.log_control(
            timestep=self.timestep,
            control=u_blend,
            controller_type="HYBRID",
            solve_time=solver_time_ms,
        )
        self.logger.log_hybrid_step(
            timestep=self.timestep,
            blend_weight=blend_info.weight,
            risk=assessment.combined_risk,
            mode=blend_info.mode,
        )

        if self.mpc_solution is not None and self.mpc_solution.slack_used:
            self.logger.log_constraint_event(
                timestep=self.timestep,
                constraint_type="slack_activated",
                details={"reason": "hybrid_mpc_feasibility"},
            )

        self.current_traj_idx += 1
        self.timestep += 1

    def publish_command(self, control: np.ndarray):
        """Publish a velocity command."""
        cmd = Twist()
        cmd.linear.x = float(control[0])
        cmd.angular.z = float(control[1])
        self.cmd_pub.publish(cmd)

    def publish_diagnostics(self, blend_weight: float, risk: float, mode: str):
        """Publish hybrid blending diagnostics."""
        weight_msg = Float32()
        weight_msg.data = float(blend_weight)
        self.weight_pub.publish(weight_msg)

        risk_msg = Float32()
        risk_msg.data = float(risk)
        self.risk_pub.publish(risk_msg)

        mode_msg = String()
        mode_msg.data = mode
        self.mode_pub.publish(mode_msg)

    def publish_reference_pose(self, state_ref: np.ndarray):
        """Publish the active reference pose for debugging."""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "odom"
        pose.pose.position.x = float(state_ref[0])
        pose.pose.position.y = float(state_ref[1])
        pose.pose.orientation.z = float(np.sin(state_ref[2] / 2.0))
        pose.pose.orientation.w = float(np.cos(state_ref[2] / 2.0))
        self.ref_pub.publish(pose)

    def publish_predicted_path(self, predicted_states: np.ndarray):
        """Publish the MPC-predicted path used inside the hybrid branch."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom"

        for state in predicted_states:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            pose.pose.orientation.z = float(np.sin(state[2] / 2.0))
            pose.pose.orientation.w = float(np.cos(state[2] / 2.0))
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_stop(self):
        """Command the simulated robot to stop once the trajectory is complete."""
        self.publish_command(np.zeros(2))


def main(args=None):
    rclpy.init(args=args)
    node = HybridControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.logger.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
