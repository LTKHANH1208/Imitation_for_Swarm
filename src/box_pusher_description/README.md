# Box Pusher Stereo Robot — ROS 2 Jazzy + Gazebo Harmonic

Package này chứa mô hình robot vi sai đẩy hộp, world Gazebo, launch khởi động, bộ điều khiển hai bánh, bridge cảm biến và node kiểm tra đồng bộ stereo camera.

Mục tiêu là dùng cùng một mô tả robot cho giai đoạn mô phỏng, thu thập dữ liệu imitation learning và sau này thay hardware interface mô phỏng bằng driver phần cứng thật.

## 1. Các điểm đã sửa trong phiên bản này

### 1.1. Bánh tự do phía sau

Bản trước dùng một quả cầu gắn cứng vào thân. Cách đó không có khớp quay và khớp xoay lái, vì vậy nó có thể trượt thay vì hoạt động như caster đúng nghĩa.

Bản mới sử dụng cụm caster có hai bậc tự do thụ động:

- `rear_caster_swivel_joint`: khớp continuous xoay quanh trục Z, cho phép bánh tự đổi hướng.
- `rear_caster_wheel_joint`: khớp continuous quay quanh trục bánh Y, cho phép bánh lăn.
- Có cổ gá nối từ mặt dưới chassis xuống ổ xoay, càng đỡ hai bên bánh và độ trail 20 mm.
- Tâm bánh caster cách mặt sàn 18 mm; bán kính bánh 18 mm, vì vậy đáy bánh nằm tại mặt phẳng sàn danh định `z = 0`.

Hai bánh chủ động có bán kính 55 mm và tâm bánh cao 55 mm so với sàn. `base_footprint` nằm trên mặt phẳng sàn; tâm `base_link` cao 105 mm so với `base_footprint`. Như vậy, cả hai bánh chủ động và caster có cùng cao độ tiếp xúc danh định.

### 1.2. Camera có thân, gá và ống kính

Mỗi camera gồm vỏ hộp, viền mặt trước, ống kính dạng trụ, vòng giữ lens và mặt kính màu xanh đậm. Hai camera được đặt trên thanh gá stereo có hai trụ đỡ, thay vì chỉ là hai khối rời.

| Thông số | Giá trị mặc định |
|---|---:|
| Baseline | 120 mm |
| Toe-in mỗi camera | 0.06 rad ≈ 3.44° |
| Độ phân giải | 640 × 480 RGB |
| Tần số | 30 Hz |
| Horizontal FOV | 120° |

Camera trái đặt ở `+Y` và xoay vào giữa; camera phải đặt ở `-Y` và cũng xoay vào giữa. Optical frame tuân theo quy ước ROS: `+Z` hướng nhìn, `+X` sang phải, `+Y` hướng xuống.

**Lưu ý:** Đây là stereo rig dạng toe-in. Cần hiệu chuẩn intrinsics/extrinsics và rectify hai ảnh trước khi tính disparity hoặc depth theo đơn vị mét. Node preview trong package chỉ đồng bộ và ghép hai ảnh thô để kiểm tra, không giả lập kết quả depth.

### 1.3. Đầu đẩy hình chữ U

Cụm đẩy là một U-channel cứng, miệng mở về hướng trước robot (`+X`):

- Thanh ngang phía sau truyền lực đẩy vào mặt sau hộp.
- Hai ray dọc hai bên giúp giữ hộp không lệch khỏi vùng đẩy.
- Mỗi thanh có visual và collision riêng để Gazebo xử lý va chạm.
- Không có cơ cấu gắp, nâng hoặc joint điều khiển.

### 1.4. Kích thước chính

| Thành phần | Kích thước/thông số |
|---|---|
| Chassis | 300 × 200 × 70 mm |
| Hai bánh chủ động | Ø110 × 24 mm |
| Khoảng cách tâm bánh | 230 mm |
| Bánh caster | Ø36 × 10 mm |
| Caster trail | 20 mm |
| Pusher | Dạng U, bề rộng ngoài 200 mm, ray dài 148 mm |
| Stereo baseline | 120 mm |

Các thông số trên là giả định thiết kế ban đầu; cần chỉnh theo kích thước phần cứng thực tế trước khi dùng để so sánh định lượng.

## 2. Kiến trúc giao tiếp Gazebo ↔ ROS 2

```text
                     ROS 2
                       │
       TwistStamped    │  /robot1/diff_drive_controller/cmd_vel
                       ▼
              diff_drive_controller
                       │  wheel velocity command
                       ▼
                  ros2_control
                       │
                 gz_ros2_control
                       ▼
                Gazebo wheel joints
                       │
            ┌──────────┴──────────┐
            │                    │
      wheel position       wheel velocity
      (encoder-like)          (feedback)
            │                    │
            └──────────┬─────────┘
                       ▼
             joint_state_broadcaster
                 /robot1/joint_states

 Gazebo sensors ──ros_gz_bridge──► ROS 2
   left RGB image       /robot1/stereo/left/image_raw
   left CameraInfo      /robot1/stereo/left/camera_info
   right RGB image      /robot1/stereo/right/image_raw
   right CameraInfo     /robot1/stereo/right/camera_info
   IMU                  /robot1/imu/data
```

Có hai đường giao tiếp độc lập:

1. **Điều khiển và phản hồi bánh xe:** ROS 2 gửi vận tốc qua `diff_drive_controller`; `gz_ros2_control` áp lệnh lên hai joint bánh trong Gazebo. Góc và vận tốc khớp quay được trả lại ROS 2 thông qua `joint_state_broadcaster`.
2. **Cảm biến:** Gazebo xuất ảnh, CameraInfo và IMU; `ros_gz_bridge` chuyển chúng thành message ROS 2.

Không cần bridge wheel command qua `ros_gz_bridge` lần thứ hai, vì đường này đã được `gz_ros2_control` đảm nhiệm.

Trong mô phỏng, encoder được biểu diễn bởi góc bánh (rad) và vận tốc góc (rad/s). Đây là giá trị tương đương encoder, không phải số xung thô. Khi chuyển sang robot thật, hardware interface cần đổi tick encoder thành radian và rad/s theo độ phân giải encoder và tỉ số truyền.

## 3. Cài dependencies

Cấu hình đích: Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic.

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro \
  ros-jazzy-cv-bridge \
  ros-jazzy-message-filters \
  ros-jazzy-rqt-image-view \
  python3-opencv
```

Đặt thư mục `box_pusher_description` vào `~/ros2_ws/src/`, sau đó build:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select box_pusher_description
source install/setup.bash
```

## 4. Khởi động toàn bộ mô phỏng

```bash
ros2 launch box_pusher_description bringup.launch.py
```

Launch sẽ lần lượt khởi động:

1. Gazebo Harmonic cùng world có Physics, Sensors và IMU systems.
2. `robot_state_publisher` trong namespace `robot1`.
3. Robot được spawn vào Gazebo.
4. Bridge cho hai camera, hai CameraInfo, IMU và `/clock`.
5. `joint_state_broadcaster` và `diff_drive_controller` sau khi robot đã spawn.
6. `stereo_sync_node` để ghép cặp ảnh trái/phải và xuất ảnh preview.

Robot được spawn cao hơn sàn 10 mm để bộ giải vật lý có thể xử lý tiếp xúc bánh ban đầu mà không bắt đầu trong trạng thái xuyên sàn; trọng lực sẽ đưa robot xuống vị trí tiếp xúc.

Có thể tắt node preview:

```bash
ros2 launch box_pusher_description bringup.launch.py use_stereo_sync:=false
```

Cấu hình hiện tại dùng namespace đơn `/robot1`. Nếu đổi namespace, phải đổi đồng bộ trong Xacro, `config/ros_gz_bridge.yaml` và `launch/bringup.launch.py`.

## 5. Điều khiển hai bánh chính từ ROS 2

`diff_drive_controller` nhận vận tốc thân robot rồi quy đổi thành vận tốc góc của bánh trái/phải. Các thông số đang khai báo:

- Bán kính bánh: `0.055 m`.
- Khoảng cách hai bánh: `0.23 m`.
- Phản hồi odometry: dùng position feedback của hai wheel joint.
- Giới hạn vận tốc thân: `|linear.x| ≤ 0.35 m/s`, `|angular.z| ≤ 2.0 rad/s`.

Topic điều khiển trong ROS 2 Jazzy:

```text
/robot1/diff_drive_controller/cmd_vel
geometry_msgs/msg/TwistStamped
```

Package có node điều khiển bằng bàn phím. Mở terminal khác, source workspace rồi chạy:

```bash
ros2 run box_pusher_description cmd_vel_keyboard.py
```

Phím điều khiển: `w` tiến, `s` lùi, `a` xoay trái, `d` xoay phải, `x` dừng, `q` thoát. Node phát message có timestamp mới ở 10 Hz để không bị `cmd_vel_timeout` loại bỏ.

Policy imitation learning có thể publish cùng loại message vào cùng topic. `linear.x` tính bằng m/s, `angular.z` tính bằng rad/s. Khi chạy policy trong mô phỏng, đặt `use_sim_time:=true` để timestamp của `TwistStamped` sử dụng đồng hồ Gazebo. Có thể lấy mẫu hành động ở 5 Hz nếu cần tái tạo tần số hành động trong bài báo.

## 6. Đọc encoder giả lập và odometry

```bash
ros2 topic echo /robot1/joint_states
ros2 topic echo /robot1/diff_drive_controller/odom
ros2 control list_controllers -c /robot1/controller_manager
```

`/robot1/joint_states` chứa `left_wheel_joint` và `right_wheel_joint` với `position` và `velocity`. Đây là dữ liệu encoder giả lập. `diff_drive_controller` dùng position feedback để tính odometry và phát TF `odom -> base_footprint`.

## 7. Đọc và kiểm tra stereo camera

Các topic ảnh thô:

```text
/robot1/stereo/left/image_raw
/robot1/stereo/right/image_raw
```

Thông tin camera:

```text
/robot1/stereo/left/camera_info
/robot1/stereo/right/camera_info
```

Ảnh ghép để kiểm tra nhanh:

```text
/robot1/stereo/preview
```

Mở công cụ xem ảnh:

```bash
ros2 run rqt_image_view rqt_image_view
```

Chọn `/robot1/stereo/preview` trong danh sách topic. Có thể chọn trực tiếp `image_raw` của camera trái/phải để xem riêng từng luồng.

`stereo_sync_node` dùng `ApproximateTimeSynchronizer` để đồng bộ hai ảnh và hai CameraInfo theo timestamp, sau đó ghép ngang thành ảnh preview. Node này giúp xác nhận rằng cả hai camera đều hoạt động, kích thước ảnh tương thích và timestamp gần nhau. Nó **chưa** thực hiện stereo rectification, disparity, depth hay point cloud. Những bước đó cần intrinsics/extrinsics đã hiệu chuẩn.

## 8. IMU

Topic:

```text
/robot1/imu/data
```

Message: `sensor_msgs/msg/Imu`, tần số danh định 100 Hz. Đây là cảm biến IMU 6 trục mô phỏng với Gaussian noise nhỏ; không phải mô hình đã hiệu chuẩn cho một mã MPU cụ thể.

IMU không trực tiếp đo vị trí tuyệt đối. Cần kết hợp vận tốc góc/gia tốc với wheel odometry, rồi dùng bộ lọc như EKF của `robot_localization` để ước lượng trạng thái. Khi robot đẩy hộp, trượt bánh và lực tiếp xúc có thể làm odometry từ encoder sai lệch.

## 9. Các lệnh kiểm tra hữu ích

```bash
# Xacro -> URDF và kiểm tra cấu trúc link/joint
xacro $(ros2 pkg prefix box_pusher_description)/share/box_pusher_description/urdf/box_pusher_stereo.urdf.xacro > /tmp/box_pusher.urdf
check_urdf /tmp/box_pusher.urdf

# Kiểm tra các topic liên quan
ros2 topic list | grep -E 'stereo|imu|joint_states|diff_drive|clock'

# Kiểm tra tần số và message của camera/IMU
ros2 topic hz /robot1/stereo/left/image_raw
ros2 topic hz /robot1/stereo/right/image_raw
ros2 topic echo --once /robot1/stereo/left/camera_info
ros2 topic echo --once /robot1/imu/data

# Kiểm tra Gazebo Transport topics
 gz topic -l | grep -E 'stereo|imu'
```

Khi chạy Gazebo, quan sát robot ở góc nhìn cạnh để xác nhận hai bánh chủ động và bánh caster cùng chạm mặt sàn. Nếu robot nảy, lún hoặc trượt bất thường, cần tinh chỉnh contact friction, solver, khối lượng/quán tính và kích thước bánh theo phiên bản Gazebo đang cài.

## 10. Những phần cần thay khi chuyển sang robot thật

Đây là mô hình mô phỏng, chưa phải driver phần cứng. Khi chuyển sang robot thật:

- Thay `gz_ros2_control/GazeboSimSystem` bằng hardware interface điều khiển driver motor và đọc encoder.
- Đổi encoder tick thành wheel angle/rad/s theo độ phân giải và tỉ số truyền đo được.
- Publish dữ liệu MPU đã hiệu chuẩn dưới dạng `sensor_msgs/msg/Imu` trong frame `imu_link`.
- Hiệu chuẩn stereo camera và cập nhật baseline, toe-in, intrinsics, extrinsics, rectification theo cụm camera thật.
- Đo lại wheel radius, wheel separation, friction, mass và inertia của robot thật.
