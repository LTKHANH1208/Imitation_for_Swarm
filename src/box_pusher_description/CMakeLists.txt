cmake_minimum_required(VERSION 3.8)
project(box_pusher_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config launch worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/stereo_sync_node.py
  scripts/cmd_vel_keyboard.py
  scripts/task_evaluator_node.py
  scripts/collision_monitor_node.py
  scripts/deadlock_monitor_node.py
  scripts/stereo_rectify_node.py
  scripts/stereo_depth_node.py
  scripts/stereo_depth_validation_node.py
  scripts/orange_box_perception_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/box_evaluator_geometry.py
  scripts/collision_geometry.py
  scripts/stereo_geometry.py
  scripts/depth_validation_geometry.py
  scripts/orange_box_perception_geometry.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
