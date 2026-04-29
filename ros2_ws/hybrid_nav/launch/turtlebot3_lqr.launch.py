#!/usr/bin/env python3
"""
Launch: TurtleBot3 + Pure LQR Controller + Obstacles + RViz
===========================================================

Launches:
  1. Gazebo with custom obstacle world
  2. TurtleBot3 Burger spawn
  3. Robot state publisher
  4. PURE LQR controller node (no hybrid logic)
  5. Obstacle publisher node (for visualization only)
  6. RViz2 visualization
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
    hybrid_nav_dir = get_package_share_directory('hybrid_nav')
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')
    tb3_launch_dir = os.path.join(tb3_gazebo_dir, 'launch')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    world = os.path.join(hybrid_nav_dir, 'worlds', 'hybrid_obstacle_world.sdf')
    
    set_gz_model_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(tb3_gazebo_dir, 'models'),
    )

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v1 ', world], 'on_exit_shutdown': 'true'}.items(),
    )

    gz_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v1 '}.items(),
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    spawn_turtlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items(),
    )

    # 5. Pure LQR Controller Node
    lqr_controller = Node(
        package='hybrid_nav',
        executable='lqr_controller_node',  # USING THE NEW PURE LQR NODE
        name='lqr_controller',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'control_rate': 20.0,
            'trajectory_amplitude': 0.5,
            'trajectory_frequency': 0.2,
            'trajectory_duration': 1000.0,
            'dt': 0.05,
            'v_max': 0.22,
            'omega_max': 2.84,
            'q_x': 10.0,
            'q_y': 10.0,
            'q_theta': 5.0,
            'r_v': 1.0,
            'r_omega': 0.1,
        }],
    )

    obstacle_publisher = Node(
        package='hybrid_nav',
        executable='obstacle_publisher_node',
        name='obstacle_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    rviz_config = os.path.join(hybrid_nav_dir, 'rviz', 'hybrid_nav.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
    )

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))
    ld.add_action(DeclareLaunchArgument('use_rviz', default_value='true'))
    ld.add_action(DeclareLaunchArgument('x_pose', default_value='0.0'))
    ld.add_action(DeclareLaunchArgument('y_pose', default_value='0.0'))

    ld.add_action(set_gz_model_path)
    ld.add_action(gz_server)
    ld.add_action(gz_client)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_turtlebot)
    ld.add_action(lqr_controller)
    ld.add_action(obstacle_publisher)
    ld.add_action(rviz)

    return ld
