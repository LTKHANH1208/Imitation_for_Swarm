from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    base_bringup = PathJoinSubstitution([
        FindPackageShare('box_pusher_description'), 'launch', 'bringup.launch.py'
    ])
    world = PathJoinSubstitution([
        FindPackageShare('box_pusher_description'), 'worlds', 'box_collection_world.sdf'
    ])
    orange_perception_config = PathJoinSubstitution([
        FindPackageShare('box_pusher_description'), 'config', 'orange_box_perception.yaml'
    ])

    task_evaluator = Node(
        package='box_pusher_description', executable='task_evaluator_node.py',
        name='task_evaluator_node', output='screen',
        condition=IfCondition(LaunchConfiguration('use_task_evaluator')),
        parameters=[{
            'use_sim_time': True, 'box_count': 28,
            'pose_topic_prefix': '/model', 'pose_topic_suffix': '/pose',
            'carpet_half_extent_m': 0.50,
            'carpet_inner_half_extent_m': 0.25,
            'pit_half_extent_m': 0.25,
            'pit_center_x_m': 0.0, 'pit_center_y_m': 0.0,
            'pit_fallen_z_threshold_m': -0.02,
            'publish_rate_hz': 5.0,
        }],
    )

    collision_monitor = Node(
        package='box_pusher_description', executable='collision_monitor_node.py',
        name='collision_monitor_node', output='screen',
        condition=IfCondition(LaunchConfiguration('use_collision_monitor')),
        parameters=[{
            'use_sim_time': True, 'box_count': 28,
            'wall_names': ['wall_north', 'wall_south', 'wall_east', 'wall_west', 'pit_wall_north', 'pit_wall_south', 'pit_wall_east', 'pit_wall_west'],
            'event_gap_sec': 0.50, 'active_timeout_sec': 0.20,
            'publish_rate_hz': 5.0,
        }],
    )

    deadlock_monitor = Node(
        package='box_pusher_description', executable='deadlock_monitor_node.py',
        name='deadlock_monitor_node', output='screen',
        condition=IfCondition(LaunchConfiguration('use_deadlock_monitor')),
        parameters=[{
            'use_sim_time': True,
            'cmd_topic': '/robot1/diff_drive_controller/cmd_vel',
            'pose_topic': '/model/box_pusher/pose',
            'delivered_count_topic': '/box_collection/evaluator/delivered_count',
            'cmd_stale_timeout_sec': 0.75,
            'linear_command_threshold_mps': 0.04,
            'angular_command_threshold_rps': 0.10,
            'linear_motion_threshold_mps': 0.015,
            'angular_motion_threshold_rps': 0.05,
            'stuck_duration_threshold_sec': 3.0,
            'task_stall_threshold_sec': 20.0,
            'publish_rate_hz': 5.0,
        }],
    )


    stereo_rectify = Node(
        package='box_pusher_description', executable='stereo_rectify_node.py',
        name='stereo_rectify_node', output='screen',
        condition=IfCondition(LaunchConfiguration('use_stereo_rectification')),
        parameters=[{
            'use_sim_time': True,
            'left_image_topic': '/robot1/stereo/left/image_raw',
            'right_image_topic': '/robot1/stereo/right/image_raw',
            'left_info_topic': '/robot1/stereo/left/camera_info',
            'right_info_topic': '/robot1/stereo/right/camera_info',
            'left_rect_topic': '/robot1/stereo/left/image_rect',
            'right_rect_topic': '/robot1/stereo/right/image_rect',
            'left_rect_info_topic': '/robot1/stereo/left/camera_info_rect',
            'right_rect_info_topic': '/robot1/stereo/right/camera_info_rect',
            'left_frame_fallback': 'left_camera_optical_frame',
            'right_frame_fallback': 'right_camera_optical_frame',
            'sync_slop_sec': 0.04,
            'max_rate_hz': LaunchConfiguration('stereo_rectify_rate_hz'),
            'rectify_alpha': 0.0,
            'expected_baseline_m': 0.12,
            'baseline_tolerance_m': 0.02,
        }],
    )

    stereo_depth = Node(
        package='box_pusher_description', executable='stereo_depth_node.py',
        name='stereo_depth_node', output='screen',
        condition=IfCondition(LaunchConfiguration('use_stereo_depth')),
        parameters=[{
            'use_sim_time': True,
            'left_rect_topic': '/robot1/stereo/left/image_rect',
            'right_rect_topic': '/robot1/stereo/right/image_rect',
            'left_info_topic': '/robot1/stereo/left/camera_info_rect',
            'right_info_topic': '/robot1/stereo/right/camera_info_rect',
            'disparity_topic': '/robot1/stereo/disparity',
            'depth_topic': '/robot1/stereo/depth',
            'depth_preview_topic': '/robot1/stereo/depth_preview',
            'disparity_preview_topic': '/robot1/stereo/disparity_preview',
            'metrics_topic': '/robot1/stereo/depth/metrics',
            'ready_topic': '/robot1/stereo/depth/ready',
            'sync_slop_sec': 0.04,
            'max_rate_hz': LaunchConfiguration('stereo_depth_rate_hz'),
            'min_disparity': 0,
            'num_disparities': 96,
            'block_size': 5,
            'uniqueness_ratio': 8,
            'speckle_window_size': 80,
            'speckle_range': 2,
            'min_depth_m': 0.20,
            'max_depth_m': 8.0,
            'preview_max_depth_m': 5.0,
        }],
    )


    stereo_depth_validation = Node(
        package='box_pusher_description', executable='stereo_depth_validation_node.py',
        name='stereo_depth_validation_node', output='screen',
        condition=IfCondition(LaunchConfiguration('use_stereo_depth_validation')),
        parameters=[{
            'use_sim_time': True,
            'depth_topic': '/robot1/stereo/depth',
            'rectified_left_topic': '/robot1/stereo/left/image_rect',
            'rectified_left_info_topic': '/robot1/stereo/left/camera_info_rect',
            'rectification_report_topic': '/robot1/stereo/rectification/report',
            'robot_pose_topic': '/model/box_pusher/pose',
            'box_pose_topic_prefix': '/model/orange_box_',
            'box_pose_topic_suffix': '/pose',
            'box_count': 28,
            'box_size_x_m': 0.08,
            'box_size_y_m': 0.08,
            'box_size_z_m': 0.10,
            'base_frame': 'base_footprint',
            'left_optical_frame': 'left_camera_optical_frame',
            'max_ground_truth_age_sec': 0.75,
            'sample_radius_px': 1,
            'min_patch_valid_pixels': 3,
            'inlier_tolerance_m': 0.10,
            'max_samples_per_frame': 28,
            'publish_overlay': True,
            'metrics_topic': '/robot1/stereo/depth_validation/metrics',
            'overlay_topic': '/robot1/stereo/depth_validation/overlay',
            'ready_topic': '/robot1/stereo/depth_validation/ready',
        }],
    )

    orange_box_perception = Node(
        package='box_pusher_description',
        executable='orange_box_perception_node.py',
        name='orange_box_perception_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_orange_box_perception')),
        parameters=[orange_perception_config, {'use_sim_time': True}],
    )

    orange_box_perception_evaluator = Node(
        package='box_pusher_description',
        executable='orange_box_perception_evaluator_node.py',
        name='orange_box_perception_evaluator_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_perception_evaluator')),
        parameters=[{
            'use_sim_time': True,
            'perception_metrics_topic': '/robot1/perception/orange_boxes/metrics',
            'perception_overlay_topic': '/robot1/perception/orange_boxes/overlay',
            'camera_info_topic': '/robot1/stereo/left/camera_info_rect',
            'rectification_report_topic': '/robot1/stereo/rectification/report',
            'robot_pose_topic': '/model/box_pusher/pose',
            'box_pose_topic_prefix': '/model/orange_box_',
            'box_pose_topic_suffix': '/pose',
            'box_count': 28,
            'box_size_x_m': 0.08,
            'box_size_y_m': 0.08,
            'box_size_z_m': 0.10,
            'base_frame': 'base_footprint',
            'camera_frame': 'left_camera_optical_frame',
            'max_ground_truth_age_sec': 0.75,
            'evaluation_rate_hz': 2.0,
            'minimum_projected_area_px': 25.0,
            'visibility_grid_size': 5,
            'minimum_visible_ray_hits': 1,
            'minimum_match_iou': 0.10,
            'centroid_grid_size': 9,
            'metrics_topic': '/robot1/perception/orange_boxes/evaluation/metrics',
            'overlay_topic': '/robot1/perception/orange_boxes/evaluation/overlay',
            'ready_topic': '/robot1/perception/orange_boxes/evaluation/ready',
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_task_evaluator', default_value='true'),
        DeclareLaunchArgument('use_collision_monitor', default_value='true'),
        DeclareLaunchArgument('use_deadlock_monitor', default_value='true'),
        DeclareLaunchArgument('robot_x', default_value='-1.30'),
        DeclareLaunchArgument('robot_y', default_value='-1.30'),
        DeclareLaunchArgument('robot_z', default_value='0.01'),
        DeclareLaunchArgument('robot_yaw', default_value='0.0'),
        DeclareLaunchArgument('use_stereo_sync', default_value='true'),
        DeclareLaunchArgument('use_stereo_rectification', default_value='true'),
        DeclareLaunchArgument('use_stereo_depth', default_value='true'),
        DeclareLaunchArgument('use_stereo_depth_validation', default_value='true'),
        DeclareLaunchArgument('use_orange_box_perception', default_value='true'),
        DeclareLaunchArgument('use_perception_evaluator', default_value='true'),
        DeclareLaunchArgument('stereo_rectify_rate_hz', default_value='10.0'),
        DeclareLaunchArgument('stereo_depth_rate_hz', default_value='5.0'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_bringup),
            launch_arguments={
                'world': world,
                'robot_x': LaunchConfiguration('robot_x'),
                'robot_y': LaunchConfiguration('robot_y'),
                'robot_z': LaunchConfiguration('robot_z'),
                'robot_yaw': LaunchConfiguration('robot_yaw'),
                'use_stereo_sync': LaunchConfiguration('use_stereo_sync'),
            }.items(),
        ),
        task_evaluator,
        collision_monitor,
        deadlock_monitor,
        stereo_rectify,
        stereo_depth,
        stereo_depth_validation,
        orange_box_perception,
        orange_box_perception_evaluator,
    ])
