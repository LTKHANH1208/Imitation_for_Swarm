from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare('box_pusher_tracking'), 'config', 'tracker.yaml'
    ])
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'detections_topic',
            default_value='/robot1/perception/orange_boxes/centers_base'
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/robot1/diff_drive_controller/odom'
        ),
        DeclareLaunchArgument('odom_frame_fallback', default_value='odom'),
        Node(
            package='box_pusher_tracking',
            executable='orange_box_tracker',
            name='orange_box_tracker_node',
            output='screen',
            parameters=[config, {
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
            }],
        ),
    ])
