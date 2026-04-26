#!/usr/bin/env python3
"""
Launch: TurtleBot3 + Hybrid LQR-MPC Controller + Obstacles + RViz
=================================================================

Launches:
  1. Gazebo with custom obstacle world
  2. TurtleBot3 Burger spawn
  3. Robot state publisher (TF frames)
  4. Hybrid LQR-MPC controller node
  5. Obstacle publisher node
  6. RViz2 visualization

Usage:
    ros2 launch hybrid_nav turtlebot3_hybrid.launch.py
    ros2 launch hybrid_nav turtlebot3_hybrid.launch.py use_hybrid:=false
    ros2 launch hybrid_nav turtlebot3_hybrid.launch.py use_rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    hybrid_nav_dir = get_package_share_directory('hybrid_nav')
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, 'launch')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    # Check for custom world, fall back to TB3 empty world
    custom_world = os.path.join(hybrid_nav_dir, 'worlds', 'hybrid_obstacle_world.sdf')
    fallback_world = os.path.join(tb3_gazebo_dir, 'worlds', 'empty_world.world')
    world = custom_world if os.path.exists(custom_world) else fallback_world

    # Set Gazebo model path for TurtleBot3 models
    set_gz_model_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(tb3_gazebo_dir, 'models'),
    )

    # 1. Gazebo server (physics simulation)
    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r -s -v1 ', world],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # 2. Gazebo client (GUI visualization)
    gz_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v1 '}.items(),
    )

    # 3. Robot State Publisher (publishes TF from TurtleBot3 URDF)
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # 4. Spawn TurtleBot3 Burger in Gazebo
    spawn_turtlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose,
        }.items(),
    )

    # 5. Hybrid LQR-MPC Controller Node
    hybrid_controller = Node(
        package='hybrid_nav',
        executable='hybrid_controller_node',
        name='hybrid_controller',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'control_rate': 20.0,
            'trajectory_amplitude': 0.5,   # meters (smaller for TB3 speed)
            'trajectory_frequency': 0.2,   # rad/s (max v ~ 0.14 m/s < 0.22)
            'trajectory_duration': 120.0,  # seconds (2 min for slower traj)
            'dt': 0.05,
            'mpc_horizon': 10,
            'd_safe': 0.35,
            'v_max': 0.22,         # TurtleBot3 Burger max
            'omega_max': 2.84,     # TurtleBot3 Burger max
            'use_hybrid': True,
        }],
    )

    # 6. Obstacle Publisher Node
    obstacle_publisher = Node(
        package='hybrid_nav',
        executable='obstacle_publisher_node',
        name='obstacle_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # 7. RViz2 Visualization
    rviz_config = os.path.join(hybrid_nav_dir, 'rviz', 'hybrid_nav.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
    )

    # Build launch description
    ld = LaunchDescription()

    # Declare arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation clock'))
    ld.add_action(DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz2 visualization'))
    ld.add_action(DeclareLaunchArgument(
        'x_pose', default_value='0.0',
        description='Robot initial X position'))
    ld.add_action(DeclareLaunchArgument(
        'y_pose', default_value='0.0',
        description='Robot initial Y position'))

    # Add all actions
    ld.add_action(set_gz_model_path)
    ld.add_action(gz_server)
    ld.add_action(gz_client)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_turtlebot)
    ld.add_action(hybrid_controller)
    ld.add_action(obstacle_publisher)
    ld.add_action(rviz)

    return ld
