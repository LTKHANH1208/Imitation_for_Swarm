#!/usr/bin/env python3
"""Rectify the V2 toe-in stereo pair using CameraInfo + TF extrinsics.

No checkerboard is needed for Gazebo: intrinsics arrive in CameraInfo and the
fixed left/right optical-frame transform is read from robot_state_publisher TF.
For real hardware, replace the nominal Gazebo calibration with a measured
checkerboard calibration before trusting metric depth.
"""

from __future__ import annotations

import copy
import json
import math
import time

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from stereo_geometry import camera_info_intrinsics, compute_rectification, quaternion_to_rotation_matrix


class StereoRectifyNode(Node):
    def __init__(self):
        super().__init__('stereo_rectify_node')
        self.declare_parameter('left_image_topic', '/robot1/stereo/left/image_raw')
        self.declare_parameter('right_image_topic', '/robot1/stereo/right/image_raw')
        self.declare_parameter('left_info_topic', '/robot1/stereo/left/camera_info')
        self.declare_parameter('right_info_topic', '/robot1/stereo/right/camera_info')
        self.declare_parameter('left_rect_topic', '/robot1/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic', '/robot1/stereo/right/image_rect')
        self.declare_parameter('left_rect_info_topic', '/robot1/stereo/left/camera_info_rect')
        self.declare_parameter('right_rect_info_topic', '/robot1/stereo/right/camera_info_rect')
        self.declare_parameter('left_frame_fallback', 'left_camera_optical_frame')
        self.declare_parameter('right_frame_fallback', 'right_camera_optical_frame')
        self.declare_parameter('sync_slop_sec', 0.04)
        self.declare_parameter('max_rate_hz', 15.0)
        self.declare_parameter('rectify_alpha', 0.0)
        self.declare_parameter('expected_baseline_m', 0.12)
        self.declare_parameter('baseline_tolerance_m', 0.02)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.maps = None
        self.signature = None
        self.last_process_time_ns = 0
        self.last_tf_warning_wall = 0.0
        self.pair_count = 0
        self.failed_count = 0

        self.left_pub = self.create_publisher(
            Image, self.get_parameter('left_rect_topic').value, qos_profile_sensor_data
        )
        self.right_pub = self.create_publisher(
            Image, self.get_parameter('right_rect_topic').value, qos_profile_sensor_data
        )
        self.left_info_pub = self.create_publisher(
            CameraInfo, self.get_parameter('left_rect_info_topic').value, qos_profile_sensor_data
        )
        self.right_info_pub = self.create_publisher(
            CameraInfo, self.get_parameter('right_rect_info_topic').value, qos_profile_sensor_data
        )
        status_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_pub = self.create_publisher(Bool, '/robot1/stereo/rectification/ready', status_qos)
        self.report_pub = self.create_publisher(String, '/robot1/stereo/rectification/report', status_qos)

        subs = [
            message_filters.Subscriber(self, Image, self.get_parameter('left_image_topic').value,
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, Image, self.get_parameter('right_image_topic').value,
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, CameraInfo, self.get_parameter('left_info_topic').value,
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, CameraInfo, self.get_parameter('right_info_topic').value,
                                       qos_profile=qos_profile_sensor_data),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=12,
            slop=float(self.get_parameter('sync_slop_sec').value),
            allow_headerless=False,
        )
        self.sync.registerCallback(self.on_pair)
        self.get_logger().info('Stereo rectifier waiting for synchronized images, CameraInfo and TF.')

    @staticmethod
    def _transform_to_rt(transform: TransformStamped):
        q = transform.transform.rotation
        t = transform.transform.translation
        rotation = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)
        return rotation, translation

    @staticmethod
    def _set_roi(camera_info: CameraInfo, roi_tuple):
        x, y, width, height = [int(v) for v in roi_tuple]
        camera_info.roi.x_offset = max(0, x)
        camera_info.roi.y_offset = max(0, y)
        camera_info.roi.width = max(0, width)
        camera_info.roi.height = max(0, height)
        camera_info.roi.do_rectify = True

    def _ensure_maps(self, left_msg, right_msg, left_info, right_info):
        height, width = left_msg.height, left_msg.width
        if right_msg.width != width or right_msg.height != height:
            raise ValueError('Left/right images have different dimensions')
        if left_info.width != width or left_info.height != height:
            raise ValueError('Left CameraInfo dimensions do not match the image')
        if right_info.width != width or right_info.height != height:
            raise ValueError('Right CameraInfo dimensions do not match the image')

        left_frame = left_info.header.frame_id or self.get_parameter('left_frame_fallback').value
        right_frame = right_info.header.frame_id or self.get_parameter('right_frame_fallback').value
        signature = (
            width, height, tuple(left_info.k), tuple(left_info.d),
            tuple(right_info.k), tuple(right_info.d), left_frame, right_frame,
        )
        if signature == self.signature and self.maps is not None:
            return left_frame, right_frame

        # lookup_transform(target, source) yields X_target = R * X_source + t.
        # This is exactly the convention expected by cv2.stereoRectify for
        # camera-1=left and camera-2=right.
        transform = self.tf_buffer.lookup_transform(right_frame, left_frame, Time())
        rotation, translation = self._transform_to_rt(transform)
        k_left, d_left = camera_info_intrinsics(left_info)
        k_right, d_right = camera_info_intrinsics(right_info)
        self.maps = compute_rectification(
            k_left, d_left, k_right, d_right,
            (width, height), rotation, translation,
            alpha=float(self.get_parameter('rectify_alpha').value),
        )

        expected = float(self.get_parameter('expected_baseline_m').value)
        tolerance = float(self.get_parameter('baseline_tolerance_m').value)
        actual = self.maps['baseline_m']
        if abs(actual - expected) > tolerance:
            self.get_logger().warning(
                f'Rectified baseline {actual:.4f} m differs from expected {expected:.4f} m '
                f'(tolerance {tolerance:.3f} m). Check Xacro and TF.'
            )

        self.signature = signature
        report = {
            'status': 'RECTIFICATION_READY',
            'image_size': [width, height],
            'left_frame': left_frame,
            'right_frame': right_frame,
            'raw_tf_translation_right_from_left_m': translation.tolist(),
            'raw_tf_baseline_m': self.maps['raw_baseline_m'],
            'rectified_baseline_m': actual,
            'rectified_focal_px': self.maps['focal_px'],
            'rectification_rotation_left': self.maps['r1'].reshape(-1).tolist(),
            'rectification_rotation_right': self.maps['r2'].reshape(-1).tolist(),
            'rectified_projection_left': self.maps['p1'].reshape(-1).tolist(),
            'rectified_projection_right': self.maps['p2'].reshape(-1).tolist(),
            'rectify_alpha': float(self.get_parameter('rectify_alpha').value),
        }
        msg = String()
        msg.data = json.dumps(report, separators=(',', ':'))
        self.report_pub.publish(msg)
        self.ready_pub.publish(Bool(data=True))
        self.get_logger().info(
            f'Rectification maps ready: {width}x{height}, '
            f'baseline={actual:.4f} m, f={self.maps["focal_px"]:.2f} px'
        )
        return left_frame, right_frame

    def on_pair(self, left_msg, right_msg, left_info, right_info):
        now_ns = self.get_clock().now().nanoseconds
        max_rate = float(self.get_parameter('max_rate_hz').value)
        if max_rate > 0.0 and self.last_process_time_ns > 0:
            if (now_ns - self.last_process_time_ns) < int(1e9 / max_rate):
                return
        self.last_process_time_ns = now_ns
        try:
            left_frame, right_frame = self._ensure_maps(
                left_msg, right_msg, left_info, right_info
            )
            left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
            rect_left = cv2.remap(
                left, self.maps['map_lx'], self.maps['map_ly'], cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            rect_right = cv2.remap(
                right, self.maps['map_rx'], self.maps['map_ry'], cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            left_out = self.bridge.cv2_to_imgmsg(rect_left, encoding='bgr8')
            right_out = self.bridge.cv2_to_imgmsg(rect_right, encoding='bgr8')
            left_out.header = copy.deepcopy(left_msg.header)
            right_out.header = copy.deepcopy(right_msg.header)
            self.left_pub.publish(left_out)
            self.right_pub.publish(right_out)

            left_info_out = copy.deepcopy(left_info)
            right_info_out = copy.deepcopy(right_info)
            left_info_out.header = copy.deepcopy(left_msg.header)
            right_info_out.header = copy.deepcopy(right_msg.header)
            left_info_out.r = self.maps['r1'].reshape(-1).tolist()
            right_info_out.r = self.maps['r2'].reshape(-1).tolist()
            left_info_out.p = self.maps['p1'].reshape(-1).tolist()
            right_info_out.p = self.maps['p2'].reshape(-1).tolist()
            self._set_roi(left_info_out, self.maps['roi_left'])
            self._set_roi(right_info_out, self.maps['roi_right'])
            self.left_info_pub.publish(left_info_out)
            self.right_info_pub.publish(right_info_out)

            self.pair_count += 1
            if self.pair_count == 1 or self.pair_count % 100 == 0:
                self.get_logger().info(
                    f'Rectified stereo pairs={self.pair_count}; frames={left_frame}, {right_frame}'
                )
        except TransformException as exc:
            self.failed_count += 1
            wall_now = time.monotonic()
            if wall_now - self.last_tf_warning_wall > 3.0:
                self.get_logger().warning(f'Waiting for stereo optical TF: {exc}')
                self.last_tf_warning_wall = wall_now
        except (ValueError, cv2.error) as exc:
            self.failed_count += 1
            self.get_logger().error(f'Stereo rectification failed: {exc}')
            self.ready_pub.publish(Bool(data=False))


def main(args=None):
    rclpy.init(args=args)
    node = StereoRectifyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()