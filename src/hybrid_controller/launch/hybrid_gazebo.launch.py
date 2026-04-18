#!/usr/bin/env python3
"""
Hybrid controller Gazebo launch file.

This launch file starts a classic Gazebo world for environment visualization,
the lightweight kinematic simulator that publishes /odom, and the ROS2 hybrid
controller stack.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, TimerAction
from launch.conditions import UnlessCondition
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    duration = LaunchConfiguration("duration")
    world = LaunchConfiguration("world")
    trajectory_type = LaunchConfiguration("trajectory_type")
    checkpoint_preset = LaunchConfiguration("checkpoint_preset")

    pkg_share = FindPackageShare("hybrid_controller")
    gazebo_share = FindPackageShare("gazebo_ros")
    world_path = PathJoinSubstitution([pkg_share, "worlds", world])

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([gazebo_share, "launch", "gzserver.launch.py"])
        ),
        launch_arguments={"world": world_path}.items(),
    )

    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([gazebo_share, "launch", "gzclient.launch.py"])
        ),
        condition=UnlessCondition(headless),
    )

    trajectory_node = Node(
        package="hybrid_controller",
        executable="trajectory_node",
        name="trajectory_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "trajectory_type": trajectory_type,
                "checkpoint_preset": checkpoint_preset,
                "duration": duration,
                "dt": 0.02,
                "amplitude": 2.0,
                "frequency": 0.5,
                "nominal_speed": 0.8,
            }
        ],
    )

    simulator_node = Node(
        package="hybrid_controller",
        executable="kinematic_sim_node",
        name="kinematic_simulator",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "dt": 0.02}],
    )

    state_estimator_node = Node(
        package="hybrid_controller",
        executable="state_estimator_node",
        name="state_estimator",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    hybrid_node = Node(
        package="hybrid_controller",
        executable="hybrid_node",
        name="hybrid_controller",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "trajectory_type": trajectory_type,
                "checkpoint_preset": checkpoint_preset,
                "trajectory_duration": duration,
                "dt": 0.02,
                "control_rate": 20.0,
                "mpc_rate_divisor": 5,
                "v_max": 2.0,
                "omega_max": 3.0,
            }
        ],
    )

    obstacle_publisher = Node(
        package="ros2topic",
        executable="ros2topic",
        name="obstacle_publisher",
        output="screen",
        arguments=[
            "pub",
            "-r",
            "1.0",
            "/mpc_obstacles",
            "std_msgs/msg/Float32MultiArray",
            "{data: [1.0, 0.5, 0.2, -0.5, -1.0, 0.25, 1.5, -0.3, 0.15]}",
        ],
    )

    shutdown_timer = TimerAction(
        period=duration,
        actions=[EmitEvent(event=Shutdown(reason="Timed hybrid Gazebo run complete"))],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("duration", default_value="20.0"),
            DeclareLaunchArgument("world", default_value="hybrid_obstacle.world"),
            DeclareLaunchArgument("trajectory_type", default_value="checkpoint_path"),
            DeclareLaunchArgument("checkpoint_preset", default_value="warehouse"),
            gazebo_server,
            gazebo_client,
            simulator_node,
            state_estimator_node,
            trajectory_node,
            hybrid_node,
            obstacle_publisher,
            shutdown_timer,
        ]
    )
