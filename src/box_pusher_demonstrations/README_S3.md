# V2-S3: Teleoperation and Demonstration Recording

Workspace used by this project:

```bash
/home/ltk1208/Imitation_for_Swarm
```

This package introduces the first S3 collection tools without changing the V2
URDF, Gazebo world, controllers, stereo pipeline, tracking, or geometry nodes.

## Scope

- S3.1: safe terminal keyboard teleoperation, publishing `TwistStamped` at 5 Hz.
- S3.2: timestamp-aware recorder; one sample is created for each command received
  on the controller `cmd_vel` input topic.
- S3.3 foundation: episode metadata, folder structure, offline integrity checker.

The package does not train a policy. It creates clean, inspectable demonstration
episodes for later Single / Recovery / Static / Dynamic dataset construction.

## Important data protocol

Paper-style baseline sample:

```text
observation = left raw RGB resized to 96 x 72
action      = [linear.x (m/s), angular.z (rad/s)]
control rate = 5 Hz
```

V2 extension channels are also saved when available:

- right raw RGB;
- left/right rectified RGB;
- depth, stored as uint16 millimetres (`0` means invalid);
- raw and rectified CameraInfo calibration;
- IMU, joint_states, wheel odometry.

Those auxiliary channels are **not baseline policy inputs by default**. Ground-truth
Gazebo box poses, robot ground-truth pose, collision events and task-evaluator
metrics are deliberately not subscribed to or stored as policy observations.

For each action at timestamp `t`, the recorder chooses the latest sensor message
whose timestamp is `<= t`, never a future message. It stores each source stamp
and its age in milliseconds. A left raw RGB sample older than 0.15 s is rejected;
optional channels may be absent without discarding the RGB-action pair.

The action is the `TwistStamped` message observed on the controller input topic.
This records the published command, not a guaranteed low-level motor acceptance
acknowledgement.

## Install

Close any running build process, then:

```bash
cd /home/ltk1208/Imitation_for_Swarm
source /opt/ros/jazzy/setup.bash
unzip ~/Downloads/v2_s3_teleop_recording_overlay.zip -d src/
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select box_pusher_demonstrations
source install/setup.bash
```

Required Ubuntu/ROS packages are declared in `package.xml`; if rosdep does not
resolve Python OpenCV/CvBridge on your machine, install:

```bash
sudo apt install ros-jazzy-cv-bridge python3-opencv python3-numpy
```

## Runtime order

### Terminal 1: start the existing V2 simulation once

```bash
cd /home/ltk1208/Imitation_for_Swarm
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch box_pusher_description box_collection.launch.py
```

Do not launch a second copy of Gazebo or the stereo nodes.

### Terminal 2: start the recorder

```bash
cd /home/ltk1208/Imitation_for_Swarm
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch box_pusher_demonstrations recording.launch.py \
  condition:=single operator_view:=onboard robot_count:=1 world_seed:=42
```

The launch starts only the recorder; it does not start Gazebo or teleoperation.

### Terminal 3: show the onboard camera only

```bash
ros2 run rqt_image_view rqt_image_view
```

Select `/robot1/stereo/left/image_raw`. For a paper-style onboard-view
collection session, the operator should make decisions using this camera view,
not the global Gazebo scene. Keep the Gazebo GUI hidden/minimized if it exposes
the full environment.

### Terminal 4: keyboard teleoperation

Run from an interactive terminal (not as a background launch node):

```bash
cd /home/ltk1208/Imitation_for_Swarm
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run box_pusher_demonstrations keyboard_teleop --ros-args \
  -p use_sim_time:=true \
  -p cmd_vel_topic:=/robot1/diff_drive_controller/cmd_vel \
  -p publish_rate_hz:=5.0
```

Keys:

- `w/s`: forward/reverse;
- `a/d`: rotate left/right;
- `q/e`: forward-left / forward-right arc;
- `z/c`: reverse-left / reverse-right arc;
- `SPACE` or `x`: stop immediately;
- `+/-`: adjust linear speed;
- `[/]`: adjust angular speed;
- `ESC` or `Ctrl-C`: stop and quit.

Hold a key so the terminal's key-repeat events continue. If no key event arrives
for 0.45 s, the teleop node publishes zero velocity (deadman stop). This is
slightly shorter than the current diff-drive controller command timeout of 0.5 s.
Use low speed for initial collection. Defaults are 0.10 m/s and 0.45 rad/s,
capped at 0.20 m/s and 0.80 rad/s in this node.

### Terminal 5: start and stop an episode

Wait until camera and depth messages are visible, then start recording:

```bash
ros2 service call /robot1/demonstration_recorder/start std_srvs/srv/Trigger "{}"
```

Drive the robot while using only the onboard camera view. End the episode:

```bash
ros2 service call /robot1/demonstration_recorder/stop std_srvs/srv/Trigger "{}"
```

Check recorder status/counters:

```bash
ros2 topic echo /robot1/demonstration_recorder/metrics
ros2 topic echo /robot1/demonstration_recorder/recording
```

## Output structure

Default root:

```text
/home/ltk1208/Imitation_for_Swarm/datasets/demonstrations/
  single_YYYYMMDDTHHMMSS_xxxxxxxx/
    episode_metadata.json
    camera_calibration.json
    samples.jsonl
    rgb_left_raw/000000.png
    rgb_right_raw/000000.png
    rgb_left_rect/000000.png
    rgb_right_rect/000000.png
    policy_rgb_96x72/000000.png
    depth_mm/000000.png
```

Each JSONL row contains one action and the selected observation references,
all relevant timestamps/ages, action values, plus auxiliary IMU/joint/odometry
metadata. Missing optional channels are marked in `quality`.

Depth is stored as 16-bit PNG millimetres to reduce disk usage. Reconstruct
metres with `depth_m = depth_mm.astype(float32) / 1000.0`; zero is invalid.

## Inspect an episode

```bash
ros2 run box_pusher_demonstrations dataset_inspect \
  /home/ltk1208/Imitation_for_Swarm/datasets/demonstrations/<episode_folder>
```

The inspector checks metadata completion, monotonically increasing action
stamps, required baseline images and missing image files.

## Recommended first experiment

1. Start one simulation with one robot and fixed world seed if your world launch
   exposes a seed parameter.
2. Record a short 30-60 second `debug` episode first.
3. Check recorder counters and inspect the episode.
4. View a few `policy_rgb_96x72` images and verify the orange boxes are visible.
5. Confirm `action_stamp_ns` increases and `left_raw` `age_ms` stays below 150 ms.
6. Only then collect a real `single` demonstration episode.

Do not target 6K samples immediately. Validate synchronization, image/action
alignment, disk usage and operator procedure on short episodes first.

## Data integrity caveats

- Stop recording before resetting Gazebo simulation time or restarting the world.
  Start a new episode after reset; never mix two sim-time epochs in one episode.
- Use one teleoperation publisher for the action topic during a demonstration.
- The keyboard node publishes a zero command on shutdown and after its deadman
  timeout.
- The recorder does not subscribe to Gazebo ground truth. Ground truth remains
  in separate evaluation nodes and must not enter the baseline policy dataset.
- `operator_view=onboard` is metadata; the software cannot enforce that the
  operator did not look at the global Gazebo view.
- For multi-robot Static/Dynamic collection, record the actual robot count and
  condition in launch arguments. Dynamic collection requires a previously
  trained policy running on the other robots; it is not implemented by this
  S3.1/S3.2 package.
