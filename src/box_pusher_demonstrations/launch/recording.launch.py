from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    del args, kwargs
    robot_ns = LaunchConfiguration('robot_namespace').perform(context).strip('/')
    dataset_root = LaunchConfiguration('dataset_root').perform(context)
    condition = LaunchConfiguration('condition').perform(context)
    operator_view = LaunchConfiguration('operator_view').perform(context)
    robot_count = int(LaunchConfiguration('robot_count').perform(context))
    world_seed = int(LaunchConfiguration('world_seed').perform(context))
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'

    root = '/' + robot_ns if robot_ns else ''
    topics = {
        'cmd_vel_topic': f'{root}/diff_drive_controller/cmd_vel',
        'left_image_topic': f'{root}/stereo/left/image_raw',
        'right_image_topic': f'{root}/stereo/right/image_raw',
        'left_rect_topic': f'{root}/stereo/left/image_rect',
        'right_rect_topic': f'{root}/stereo/right/image_rect',
        'depth_topic': f'{root}/stereo/depth',
        'left_info_topic': f'{root}/stereo/left/camera_info',
        'right_info_topic': f'{root}/stereo/right/camera_info',
        'left_rect_info_topic': f'{root}/stereo/left/camera_info_rect',
        'right_rect_info_topic': f'{root}/stereo/right/camera_info_rect',
        'imu_topic': f'{root}/imu/data',
        'joint_states_topic': f'{root}/joint_states',
        'odom_topic': f'{root}/diff_drive_controller/odom',
        'tracks_topic': f'{root}/perception/orange_boxes/tracks',
        'geometry_topic': f'{root}/perception/orange_boxes/geometry',
    }

    params = {
        'use_sim_time': use_sim_time,
        'dataset_root': dataset_root,
        'condition': condition,
        'operator_view': operator_view,
        'robot_count': robot_count,
        'world_seed': world_seed,
        'expected_control_hz': 5.0,
        'max_left_image_age_sec': 0.15,
        'max_optional_image_age_sec': 0.20,
        'max_depth_age_sec': 0.60,
        'max_aux_age_sec': 0.20,
        'policy_image_width': 96,
        'policy_image_height': 72,
        **topics,
    }

    return [
        Node(
            package='box_pusher_demonstrations',
            executable='demonstration_recorder',
            name='demonstration_recorder',
            namespace=robot_ns,
            output='screen',
            parameters=[params],
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_namespace', default_value='robot1'),
        DeclareLaunchArgument(
            'dataset_root',
            default_value='/home/ltk1208/Imitation_for_Swarm/datasets/demonstrations',
        ),
        DeclareLaunchArgument('condition', default_value='single'),
        DeclareLaunchArgument('operator_view', default_value='onboard'),
        DeclareLaunchArgument('robot_count', default_value='1'),
        DeclareLaunchArgument('world_seed', default_value='42'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        OpaqueFunction(function=_launch_setup),
    ])
