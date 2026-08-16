"""Launch the FCOS3D detector node.

Usage:
    ros2 launch fcos3d_ros2 detector.launch.py \
        checkpoint_file:=/ws/checkpoints/fcos3d_r101.pth \
        device:=cuda:0 fp16:=false

Contributor: Carlos Gonzales
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_CONFIG = (
    '/opt/mmdetection3d/configs/fcos3d/'
    'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py'
)


def generate_launch_description():
    args = [
        DeclareLaunchArgument('config_file', default_value=DEFAULT_CONFIG),
        DeclareLaunchArgument('checkpoint_file', default_value=''),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('fp16', default_value='false'),
        DeclareLaunchArgument('score_threshold', default_value='0.3'),
        DeclareLaunchArgument(
            'image_topic', default_value='/camera/front/image_raw'),
        DeclareLaunchArgument(
            'camera_info_topic', default_value='/camera/front/camera_info'),
    ]

    node = Node(
        package='fcos3d_ros2',
        executable='detector_node',
        name='fcos3d_detector',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'checkpoint_file': LaunchConfiguration('checkpoint_file'),
            'device': LaunchConfiguration('device'),
            'fp16': LaunchConfiguration('fp16'),
            'score_threshold': LaunchConfiguration('score_threshold'),
            'image_topic': LaunchConfiguration('image_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
        }],
    )

    return LaunchDescription(args + [node])
