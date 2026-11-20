# V2-S2.6 — Orange Box Geometric Estimation

This is an **additive ROS 2 package**. It does not modify `box_pusher_description`,
S1 evaluators, S2.3 perception, or the S2.5 tracking package.

## Purpose

S2.3 detects orange regions and estimates visible-surface 3-D points. S2.5 tracks
those points over time. S2.6 uses the registered rectified RGB, mask and metric
depth to fit a visible vertical face and estimate:

- visible face centre and outward normal;
- approximate full box centre, using the known 0.08 x 0.08 x 0.10 m box prior;
- observed face width/height;
- horizontal edge direction and approach heading;
- association with the local S2.5 track ID when track and odometry timestamps match.

The estimator **does not subscribe to Gazebo model poses, ground-truth evaluator
topics, or world-coordinate labels**. Gazebo ground truth remains exclusive to
evaluation nodes.

## Important geometry limitation

The boxes have a square 80 x 80 mm footprint and no asymmetric visual marker.
Their absolute yaw is therefore not uniquely observable from geometry alone:
rotating an unmarked square box by 90 degrees produces the same footprint.
S2.6 reports `yaw_edge_mod_pi_over_2_rad` and a separately estimated visible-face
normal / `approach_yaw_base_rad`. Do not interpret this as a unique 0–360 degree
box heading.

The full geometric centre is an **approximation**: it shifts the estimated
visible vertical face centre inward by half the known 80 mm box depth and uses
the floor-support prior for centre height (0.05 m). Width and height measurements
are observed extents and can be smaller when a face is occluded or clipped.
Confidence and plane-fit residuals are published so downstream systems can
reject weak estimates. For arbitrary boxes or boxes not resting on the floor,
this prior must be replaced by general 3-D shape reconstruction.

## Inputs

| Topic | Type | Use |
|---|---|---|
| `/robot1/stereo/left/image_rect` | `sensor_msgs/Image` | Rectified RGB for overlay |
| `/robot1/perception/orange_boxes/mask` | `sensor_msgs/Image` (`mono8`) | S2.3 orange segmentation |
| `/robot1/stereo/depth` | `sensor_msgs/Image` (`32FC1`, metres) | Stereo depth |
| `/robot1/stereo/left/camera_info_rect` | `sensor_msgs/CameraInfo` | Rectified projection and R1 |
| `/robot1/diff_drive_controller/odom` | `nav_msgs/Odometry` | Timestamped base-to-odom transform for track ID association |
| `/robot1/perception/orange_boxes/tracks` | `std_msgs/String` | S2.5 local track IDs/positions in odom |

Image, mask, depth and CameraInfo are approximately synchronized. The point cloud
is reconstructed in the rectified optical frame, rotated back with `R1.T`, then
transformed using the raw optical-camera TF into `base_footprint`.

## Outputs

| Topic | Type | Meaning |
|---|---|---|
| `/robot1/perception/orange_boxes/geometry` | `std_msgs/String` | Full per-detection geometry JSON |
| `/robot1/perception/orange_boxes/geometry/poses` | `geometry_msgs/PoseArray` | Approximate geometric centres in `base_footprint`; orientation is only yaw modulo 90 degrees |
| `/robot1/perception/orange_boxes/geometry/markers` | `visualization_msgs/MarkerArray` | Approximate box cubes, inward arrows and labels |
| `/robot1/perception/orange_boxes/geometry/overlay` | `sensor_msgs/Image` | RGB image with contour and estimated geometry labels |
| `/robot1/perception/orange_boxes/geometry/metrics` | `std_msgs/String` | Counts, fit quality and track association summary |
| `/robot1/perception/orange_boxes/geometry/ready` | `std_msgs/Bool` | True after an RGB-D frame is processed |

## Build

```bash
cd /home/ltk1208/Imitation_for_Swarm
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select box_pusher_geometry
source install/setup.bash
```

## Run

Keep the already working S2.4 and S2.5 sessions running once each. In a new
terminal, launch only S2.6:

```bash
cd /home/ltk1208/Imitation_for_Swarm
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch box_pusher_geometry orange_box_geometry.launch.py
```

If topic names differ, override them without editing source:

```bash
ros2 launch box_pusher_geometry orange_box_geometry.launch.py \
  mask_topic:=/robot1/perception/orange_boxes/mask \
  tracks_topic:=/robot1/perception/orange_boxes/tracks
```

Do **not** launch `box_collection.launch.py` again if Gazebo/S2.4 is already
running. Start the simulator once, the S2.5 tracker once, and this S2.6 node once.

## Verify

```bash
ros2 node list | sort | uniq -c | grep -E 'orange_box_(tracker|geometry)'
ros2 topic hz /robot1/perception/orange_boxes/geometry/overlay
ros2 topic echo /robot1/perception/orange_boxes/geometry/ready
ros2 topic echo /robot1/perception/orange_boxes/geometry/metrics
ros2 topic echo /robot1/perception/orange_boxes/geometry --once
```

Open the overlay with `rqt_image_view`. In RViz2 set Fixed Frame to
`base_footprint`, add MarkerArray, and select
`/robot1/perception/orange_boxes/geometry/markers`.

## Unit tests

```bash
cd /home/ltk1208/Imitation_for_Swarm
PYTHONPATH=src/box_pusher_geometry \
  python3 -m unittest discover -s src/box_pusher_geometry/test -v
```

The tests validate back-projection, rigid transforms, square-yaw ambiguity and
face fitting on a synthetic cuboid point cloud. They do not replace runtime
validation with the actual Gazebo camera and stereo disparity stream.
