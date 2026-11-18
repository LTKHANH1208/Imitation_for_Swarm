from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('box_pusher_description')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    default_world_file = os.path.join(pkg_share, 'worlds', 'box_pusher_world.sdf')
    world_file = LaunchConfiguration('world')
    robot_x = LaunchConfiguration('robot_x')
    robot_y = LaunchConfiguration('robot_y')
    robot_z = LaunchConfiguration('robot_z')
    robot_yaw = LaunchConfiguration('robot_yaw')
    xacro_file = os.path.join(pkg_share, 'urdf', 'box_pusher_stereo.urdf.xacro')
    bridge_file = os.path.join(pkg_share, 'config', 'ros_gz_bridge.yaml')

    robot_namespace = 'robot1'
    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file,
            ' robot_namespace:=', robot_namespace,
            ' camera_baseline:=0.12 camera_toe_in:=0.06'
        ]),
        value_type=str
    )

    use_stereo_sync = LaunchConfiguration('use_stereo_sync')
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': [TextSubstitution(text='-r '), world_file]}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=robot_namespace,
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
        remappings=[('tf', '/tf'), ('tf_static', '/tf_static')]
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_box_pusher',
        output='screen',
        arguments=[
            '-name', 'box_pusher',
            '-topic', '/robot1/robot_description',
            '-x', robot_x, '-y', robot_y, '-z', robot_z, '-Y', robot_yaw,
            '-allow_renaming', 'true',
        ]
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_sensor_bridge',
        output='screen',
        arguments=['--ros-args', '-p', f'config_file:={bridge_file}']
    )

    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='spawn_joint_state_broadcaster',
        output='screen',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/robot1/controller_manager',
            '--controller-manager-timeout', '60'
        ]
    )

    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='spawn_diff_drive_controller',
        output='screen',
        arguments=[
            'diff_drive_controller',
            '--controller-manager', '/robot1/controller_manager',
            '--controller-manager-timeout', '60'
        ]
    )

    stereo_sync = Node(
        package='box_pusher_description',
        executable='stereo_sync_node.py',
        namespace=robot_namespace,
        name='stereo_sync_node',
        output='screen',
        condition=IfCondition(use_stereo_sync),
        parameters=[{'use_sim_time': True, 'sync_slop_sec': 0.04}]
    )

    # Start controllers only after Gazebo has accepted the robot entity.
    start_joint_state_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[TimerAction(period=2.0, actions=[joint_state_spawner])]
        )
    )
    start_diff_drive_after_joint_state = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_spawner,
            on_exit=[diff_drive_spawner]
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value=default_world_file,
            description='Absolute path to the Gazebo SDF world.'
        ),
        DeclareLaunchArgument('robot_x', default_value='0.0', description='Robot spawn X (m).'),
        DeclareLaunchArgument('robot_y', default_value='0.0', description='Robot spawn Y (m).'),
        DeclareLaunchArgument('robot_z', default_value='0.01', description='Robot spawn Z (m).'),
        DeclareLaunchArgument('robot_yaw', default_value='0.0', description='Robot spawn yaw (rad).'),
        DeclareLaunchArgument(
            'use_stereo_sync', default_value='true',
            description='Publish a side-by-side preview after synchronizing both camera streams.'
        ),
        start_gazebo,
        robot_state_publisher,
        bridge,
        stereo_sync,
        TimerAction(period=3.0, actions=[spawn_robot]),
        start_joint_state_after_spawn,
        start_diff_drive_after_joint_state,
    ])
