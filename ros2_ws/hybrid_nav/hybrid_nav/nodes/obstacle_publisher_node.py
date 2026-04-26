#!/usr/bin/env python3
"""
Obstacle Publisher Node for TurtleBot3 Gazebo Simulation
========================================================

Publishes static obstacle positions for the hybrid controller.
These match obstacles placed in the Gazebo world file.

Publishes: /obstacles (std_msgs/Float32MultiArray)
           Format: [x1, y1, r1, x2, y2, r2, ...]
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray


class ObstaclePublisherNode(Node):
    def __init__(self):
        super().__init__('obstacle_publisher')

        self.declare_parameter('publish_rate', 2.0)
        publish_rate = self.get_parameter('publish_rate').value

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Float32MultiArray, '/obstacles', qos)

        # Obstacle positions matching the Gazebo world
        # Scaled for figure-8 with A=0.5m: x in [-0.5, 0.5], y in [-0.25, 0.25]
        # Format: [x, y, radius] for each obstacle
        self.obstacles = [
            [0.35,  0.15,  0.10],   # Right loop of figure-8
            [-0.25, -0.15, 0.10],   # Left loop of figure-8
            [0.45,  -0.10, 0.08],   # Near crossing point
        ]

        self.timer = self.create_timer(1.0 / publish_rate, self.publish_obstacles)
        self.get_logger().info(
            f'Obstacle publisher started: {len(self.obstacles)} obstacles'
        )

    def publish_obstacles(self):
        msg = Float32MultiArray()
        data = []
        for obs in self.obstacles:
            data.extend(obs)
        msg.data = [float(v) for v in data]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstaclePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
