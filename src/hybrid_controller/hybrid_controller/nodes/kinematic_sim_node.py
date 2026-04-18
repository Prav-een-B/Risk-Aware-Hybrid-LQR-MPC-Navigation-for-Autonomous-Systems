#!/usr/bin/env python3
"""
Kinematic simulation node.

This node closes the ROS2 control loop for the container workflow by
integrating the differential-drive model from incoming /cmd_vel commands and
publishing /odom.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import TransformBroadcaster

from ..models.differential_drive import DifferentialDriveRobot


class KinematicSimulationNode(Node):
    """Lightweight odometry simulator for ROS2 controller demos."""

    def __init__(self):
        super().__init__("kinematic_simulator")

        self.declare_parameter("dt", 0.02)
        self.declare_parameter("v_max", 2.0)
        self.declare_parameter("omega_max", 3.0)
        self.declare_parameter("wheel_base", 0.3)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_theta", 0.0)

        dt = float(self.get_parameter("dt").value)
        v_max = float(self.get_parameter("v_max").value)
        omega_max = float(self.get_parameter("omega_max").value)
        wheel_base = float(self.get_parameter("wheel_base").value)

        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.current_state = np.array(
            [
                float(self.get_parameter("initial_x").value),
                float(self.get_parameter("initial_y").value),
                float(self.get_parameter("initial_theta").value),
            ],
            dtype=float,
        )
        self.current_control = np.zeros(2, dtype=float)
        self.dt = dt
        self.robot = DifferentialDriveRobot(
            v_max=v_max, omega_max=omega_max, wheel_base=wheel_base
        )

        qos = QoSProfile(depth=10)
        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", qos)
        self.path_pub = self.create_publisher(Path, "/sim/path", qos)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.timer = self.create_timer(self.dt, self.step_callback)

        self.get_logger().info("Kinematic simulator initialized")

    def cmd_callback(self, msg: Twist):
        """Cache the most recent commanded velocity."""
        self.current_control[0] = float(msg.linear.x)
        self.current_control[1] = float(msg.angular.z)

    def step_callback(self):
        """Advance the kinematic state and publish odometry."""
        self.current_state = self.robot.simulate_step(
            self.current_state, self.current_control, self.dt
        )

        now = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = float(self.current_state[0])
        odom.pose.pose.position.y = float(self.current_state[1])
        odom.pose.pose.orientation.z = float(np.sin(self.current_state[2] / 2.0))
        odom.pose.pose.orientation.w = float(np.cos(self.current_state[2] / 2.0))
        odom.twist.twist.linear.x = float(self.current_control[0])
        odom.twist.twist.angular.z = float(self.current_control[1])
        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header.stamp = now
            transform.header.frame_id = self.odom_frame
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.translation.z = 0.0
            transform.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

        path_point = PoseStamped()
        path_point.header.stamp = now
        path_point.header.frame_id = self.odom_frame
        path_point.pose = odom.pose.pose
        self.path_msg.header.stamp = now
        self.path_msg.poses.append(path_point)
        self.path_pub.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = KinematicSimulationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
