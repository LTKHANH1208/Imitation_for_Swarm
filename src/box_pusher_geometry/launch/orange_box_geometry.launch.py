from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare('box_pusher_geometry'), 'config', 'geometry.yaml'
    ])
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('image_topic', default_value='/robot1/stereo/left/image_rect'),
        DeclareLaunchArgument('depth_topic', default_value='/robot1/stereo/depth'),
        DeclareLaunchArgument('mask_topic', default_value='/robot1/perception/orange_boxes/mask'),
        DeclareLaunchArgument('camera_info_topic', default_value='/robot1/stereo/left/camera_info_rect'),
        DeclareLaunchArgument('tracks_topic', default_value='/robot1/perception/orange_boxes/tracks'),
        DeclareLaunchArgument('odom_topic', default_value='/robot1/diff_drive_controller/odom'),
        Node(
            package='box_pusher_geometry',
            executable='orange_box_geometry',
            name='orange_box_geometry_node',
            output='screen',
            parameters=[config, {
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                'image_topic': LaunchConfiguration('image_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'mask_topic': LaunchConfiguration('mask_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'tracks_topic': LaunchConfiguration('tracks_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
            }],
        ),
    ])
