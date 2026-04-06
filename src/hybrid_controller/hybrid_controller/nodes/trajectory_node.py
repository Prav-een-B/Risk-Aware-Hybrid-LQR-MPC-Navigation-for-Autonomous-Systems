#!/usr/bin/env python3
"""
Trajectory Node
===============

ROS2 node that publishes reference trajectories for the navigation stack.

Publishers:
    /reference_trajectory (nav_msgs/Path): Full reference trajectory
    /current_reference (geometry_msgs/PoseStamped): Current reference pose

Parameters:
    amplitude: Spatial size of analytic trajectories (meters)
    frequency: Angular frequency of analytic trajectories (rad/s)
    duration: Total trajectory duration (seconds)
    trajectory_type: Reference family to publish
    checkpoint_preset: Preset used when trajectory_type=checkpoint_path
    nominal_speed: Desired speed along checkpoint paths (m/s)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray

import numpy as np
from ..trajectory.reference_generator import ReferenceTrajectoryGenerator


class TrajectoryPublisher(Node):
    """
    ROS2 node for publishing reference trajectories.
    """
    
    def __init__(self):
        super().__init__('trajectory_publisher')
        
        # Declare parameters
        self.declare_parameter('amplitude', 2.0)
        self.declare_parameter('frequency', 0.5)
        self.declare_parameter('duration', 20.0)
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('trajectory_type', 'figure8')
        self.declare_parameter('checkpoint_preset', 'diamond')
        self.declare_parameter('nominal_speed', 0.8)

        # Get parameters
        self.amplitude = self.get_parameter('amplitude').value
        self.frequency = self.get_parameter('frequency').value
        self.duration = self.get_parameter('duration').value
        self.dt = self.get_parameter('dt').value
        self.trajectory_type = self.get_parameter('trajectory_type').value
        self.checkpoint_preset = self.get_parameter('checkpoint_preset').value
        self.nominal_speed = self.get_parameter('nominal_speed').value

        # Initialize trajectory generator
        self.generator = ReferenceTrajectoryGenerator(
            A=self.amplitude,
            a=self.frequency,
            dt=self.dt,
            trajectory_type=self.trajectory_type,
            checkpoint_preset=self.checkpoint_preset,
            nominal_speed=self.nominal_speed,
        )
        
        # Generate trajectory
        self.trajectory = self.generator.generate(self.duration)
        self.current_idx = 0

        self.get_logger().info(
            f"Generated {self._describe_trajectory()} trajectory with "
            f"{len(self.trajectory)} points"
        )
        
        # QoS profile
        qos = QoSProfile(depth=10)
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/reference_trajectory', qos)
        self.pose_pub = self.create_publisher(PoseStamped, '/current_reference', qos)
        self.ref_vel_pub = self.create_publisher(Twist, '/reference_velocity', qos)
        
        # Timer for publishing current reference
        self.timer = self.create_timer(self.dt, self.timer_callback)
        
        # Publish full path once
        self.publish_full_path()
        
        self.get_logger().info("Trajectory publisher initialized")

    def _describe_trajectory(self) -> str:
        """Build a readable trajectory label for logs."""
        if self.trajectory_type == 'checkpoint_path':
            return f"{self.trajectory_type} ({self.checkpoint_preset})"
        return str(self.trajectory_type)
    
    def publish_full_path(self):
        """Publish the complete reference trajectory as a Path message."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
        
        for i in range(len(self.trajectory)):
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.pose.position.x = self.trajectory[i, 1]
            pose.pose.position.y = self.trajectory[i, 2]
            pose.pose.position.z = 0.0
            
            # Convert heading to quaternion (yaw only)
            theta = self.trajectory[i, 3]
            pose.pose.orientation.z = np.sin(theta / 2)
            pose.pose.orientation.w = np.cos(theta / 2)
            
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info("Published full reference trajectory")
    
    def timer_callback(self):
        """Publish current reference pose and velocity."""
        if self.current_idx >= len(self.trajectory):
            self.get_logger().info("Trajectory complete")
            return
        
        # Get current reference point
        ref = self.trajectory[self.current_idx]
        t, px, py, theta, v, omega = ref
        
        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = px
        pose_msg.pose.position.y = py
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = np.sin(theta / 2)
        pose_msg.pose.orientation.w = np.cos(theta / 2)
        
        self.pose_pub.publish(pose_msg)
        
        # Publish reference velocity
        vel_msg = Twist()
        vel_msg.linear.x = v
        vel_msg.angular.z = omega
        self.ref_vel_pub.publish(vel_msg)
        
        # Advance to next point
        self.current_idx += 1
    
    def reset(self):
        """Reset trajectory to beginning."""
        self.current_idx = 0
        self.get_logger().info("Trajectory reset to start")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
