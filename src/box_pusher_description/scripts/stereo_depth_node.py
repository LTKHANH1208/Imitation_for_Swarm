#!/usr/bin/env python3
"""Compute rectified stereo disparity and metric depth for robot V2."""

from __future__ import annotations

import copy
import json
import math

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
)
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest
from std_msgs.msg import Bool, String
from stereo_msgs.msg import DisparityImage

from stereo_geometry import depth_from_disparity


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')
        self.declare_parameter('left_rect_topic', '/robot1/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic', '/robot1/stereo/right/image_rect')
        self.declare_parameter('left_info_topic', '/robot1/stereo/left/camera_info_rect')
        self.declare_parameter('right_info_topic', '/robot1/stereo/right/camera_info_rect')
        self.declare_parameter('disparity_topic', '/robot1/stereo/disparity')
        self.declare_parameter('depth_topic', '/robot1/stereo/depth')
        self.declare_parameter('depth_preview_topic', '/robot1/stereo/depth_preview')
        self.declare_parameter('disparity_preview_topic', '/robot1/stereo/disparity_preview')
        self.declare_parameter('metrics_topic', '/robot1/stereo/depth/metrics')
        self.declare_parameter('ready_topic', '/robot1/stereo/depth/ready')
        self.declare_parameter('sync_slop_sec', 0.04)
        self.declare_parameter('max_rate_hz', 5.0)
        self.declare_parameter('min_disparity', 0)
        self.declare_parameter('num_disparities', 96)
        self.declare_parameter('block_size', 5)
        self.declare_parameter('uniqueness_ratio', 8)
        self.declare_parameter('speckle_window_size', 80)
        self.declare_parameter('speckle_range', 2)
        self.declare_parameter('min_depth_m', 0.20)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('preview_max_depth_m', 5.0)

        self.bridge = CvBridge()
        self.matcher = None
        self.matcher_signature = None
        self.last_process_time_ns = 0
        self.processed_count = 0
        self.skipped_count = 0

        self.disparity_pub = self.create_publisher(
            DisparityImage, self.get_parameter('disparity_topic').value, qos_profile_sensor_data
        )
        self.depth_pub = self.create_publisher(
            Image, self.get_parameter('depth_topic').value, qos_profile_sensor_data
        )
        self.preview_pub = self.create_publisher(
            Image, self.get_parameter('depth_preview_topic').value, qos_profile_sensor_data
        )
        self.disparity_preview_pub = self.create_publisher(
            Image, self.get_parameter('disparity_preview_topic').value, qos_profile_sensor_data
        )
        self.metrics_pub = self.create_publisher(String, self.get_parameter('metrics_topic').value, 10)
        status_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_pub = self.create_publisher(
            Bool, self.get_parameter('ready_topic').value, status_qos
        )

        subs = [
            message_filters.Subscriber(self, Image, self.get_parameter('left_rect_topic').value,
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, Image, self.get_parameter('right_rect_topic').value,
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, CameraInfo, self.get_parameter('left_info_topic').value,
                                       qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, CameraInfo, self.get_parameter('right_info_topic').value,
                                       qos_profile=qos_profile_sensor_data),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=8,
            slop=float(self.get_parameter('sync_slop_sec').value),
            allow_headerless=False,
        )
        self.sync.registerCallback(self.on_pair)
        self.get_logger().info('Stereo depth node waiting for rectified stereo pairs.')

    def _ensure_matcher(self, width: int, height: int):
        min_disp = int(self.get_parameter('min_disparity').value)
        num_disp = int(self.get_parameter('num_disparities').value)
        block = int(self.get_parameter('block_size').value)
        if num_disp <= 0 or num_disp % 16 != 0:
            raise ValueError('num_disparities must be a positive multiple of 16')
        if block < 3 or block % 2 == 0:
            raise ValueError('block_size must be an odd integer >= 3')
        if num_disp >= width:
            raise ValueError(f'num_disparities ({num_disp}) must be smaller than image width ({width})')
        signature = (width, height, min_disp, num_disp, block)
        if signature == self.matcher_signature and self.matcher is not None:
            return
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * block * block,
            P2=32 * block * block,
            disp12MaxDiff=1,
            preFilterCap=31,
            uniquenessRatio=int(self.get_parameter('uniqueness_ratio').value),
            speckleWindowSize=int(self.get_parameter('speckle_window_size').value),
            speckleRange=int(self.get_parameter('speckle_range').value),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.matcher_signature = signature
        self.get_logger().info(
            f'StereoSGBM configured: {width}x{height}, min_disp={min_disp}, '
            f'num_disp={num_disp}, block={block}'
        )

    @staticmethod
    def _roi_from_info(info: CameraInfo) -> RegionOfInterest:
        roi = RegionOfInterest()
        roi.x_offset = info.roi.x_offset
        roi.y_offset = info.roi.y_offset
        roi.width = info.roi.width
        roi.height = info.roi.height
        roi.do_rectify = info.roi.do_rectify
        return roi

    def on_pair(self, left_msg, right_msg, left_info, right_info):
        now_ns = self.get_clock().now().nanoseconds
        max_rate = float(self.get_parameter('max_rate_hz').value)
        if max_rate > 0.0 and self.last_process_time_ns > 0:
            if (now_ns - self.last_process_time_ns) < int(1e9 / max_rate):
                self.skipped_count += 1
                return
        self.last_process_time_ns = now_ns

        try:
            if left_msg.width != right_msg.width or left_msg.height != right_msg.height:
                raise ValueError('Rectified left/right image dimensions differ')
            width, height = int(left_msg.width), int(left_msg.height)
            self._ensure_matcher(width, height)
            if len(left_info.p) != 12 or len(right_info.p) != 12:
                raise ValueError('Rectified CameraInfo must contain 12 projection-matrix values')
            p_left = np.asarray(left_info.p, dtype=np.float64).reshape(3, 4)
            p_right = np.asarray(right_info.p, dtype=np.float64).reshape(3, 4)
            focal = float(p_left[0, 0])
            if focal <= 0.0:
                raise ValueError('Left rectified CameraInfo has invalid P[0,0]')
            baseline = abs(float(p_right[0, 3] / p_right[0, 0]))
            if not math.isfinite(baseline) or baseline <= 0.0:
                raise ValueError('Right rectified CameraInfo has invalid stereo baseline P[0,3]/P[0,0]')

            left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            disp_fixed = self.matcher.compute(left_gray, right_gray).astype(np.float32)
            disparity = disp_fixed / 16.0

            min_disp = float(self.get_parameter('min_disparity').value)
            max_disp = min_disp + float(self.get_parameter('num_disparities').value)
            min_depth = float(self.get_parameter('min_depth_m').value)
            max_depth = float(self.get_parameter('max_depth_m').value)
            depth, valid = depth_from_disparity(
                disparity, focal, baseline, min_disp + 0.5, max_disp,
                min_depth_m=min_depth, max_depth_m=max_depth,
            )

            disparity_msg = DisparityImage()
            disparity_msg.header = copy.deepcopy(left_msg.header)
            disparity_msg.image = self.bridge.cv2_to_imgmsg(disparity, encoding='32FC1')
            disparity_msg.image.header = copy.deepcopy(left_msg.header)
            disparity_msg.f = focal
            disparity_msg.t = baseline
            disparity_msg.valid_window = self._roi_from_info(left_info)
            disparity_msg.min_disparity = min_disp
            disparity_msg.max_disparity = max_disp
            disparity_msg.delta_d = 1.0 / 16.0
            self.disparity_pub.publish(disparity_msg)

            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
            depth_msg.header = copy.deepcopy(left_msg.header)
            self.depth_pub.publish(depth_msg)

            preview_max = float(self.get_parameter('preview_max_depth_m').value)
            preview = np.zeros((height, width), dtype=np.uint8)
            if preview_max > min_depth:
                safe_depth = np.nan_to_num(depth, nan=preview_max, posinf=preview_max, neginf=min_depth)
                scaled = np.clip((safe_depth - min_depth) / (preview_max - min_depth), 0.0, 1.0)
                preview = (255.0 * (1.0 - scaled)).astype(np.uint8)
                preview[~valid] = 0
            preview_color = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
            preview_msg = self.bridge.cv2_to_imgmsg(preview_color, encoding='bgr8')
            preview_msg.header = copy.deepcopy(left_msg.header)
            self.preview_pub.publish(preview_msg)

            # Visualization only: map the searched disparity interval to 8-bit.
            # Metric values remain in /disparity and /depth, not in this preview.
            disp_vis = np.zeros((height, width), dtype=np.uint8)
            disp_valid = np.isfinite(disparity) & (disparity > min_disp + 0.5) & (disparity < max_disp)
            if np.any(disp_valid):
                disp_scaled = (disparity - min_disp) / max(1.0, (max_disp - min_disp))
                disp_vis[disp_valid] = np.clip(255.0 * disp_scaled[disp_valid], 0, 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
            disp_preview_msg = self.bridge.cv2_to_imgmsg(disp_color, encoding='bgr8')
            disp_preview_msg.header = copy.deepcopy(left_msg.header)
            self.disparity_preview_pub.publish(disp_preview_msg)

            self.processed_count += 1
            valid_fraction = float(np.count_nonzero(valid)) / float(valid.size)
            valid_depths = depth[valid]
            metrics = {
                'processed_frames': self.processed_count,
                'skipped_for_rate_limit': self.skipped_count,
                'image_size': [width, height],
                'focal_px': focal,
                'baseline_m': baseline,
                'valid_depth_fraction': valid_fraction,
                'valid_depth_percent': 100.0 * valid_fraction,
                'median_depth_m': float(np.median(valid_depths)) if valid_depths.size else None,
                'min_depth_m': min_depth,
                'max_depth_m': max_depth,
            }
            metric_msg = String()
            metric_msg.data = json.dumps(metrics, separators=(',', ':'))
            self.metrics_pub.publish(metric_msg)
            if self.processed_count == 1 or self.processed_count % 50 == 0:
                self.get_logger().info(
                    f'Depth frames={self.processed_count}, valid={100.0 * valid_fraction:.1f}%, '
                    f'baseline={baseline:.3f} m'
                )
            self.ready_pub.publish(Bool(data=True))
        except (ValueError, cv2.error) as exc:
            self.get_logger().error(f'Stereo depth failed: {exc}')
            self.ready_pub.publish(Bool(data=False))


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
