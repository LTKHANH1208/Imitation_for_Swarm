#!/usr/bin/env python3
"""Compare stereo-derived metric depth against Gazebo box ground truth.

All Gazebo pose inputs are evaluation-only. They must never be remapped into
or supplied to the imitation-policy observation pipeline.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from rclpy.time import Time

from depth_validation_geometry import (
    box_corners_world,
    pose_to_rt,
    project_rectified_point,
    ray_oriented_box_intersection,
    quaternion_to_matrix,
)


class StereoDepthValidationNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_validation_node')
        self.declare_parameter('depth_topic', '/robot1/stereo/depth')
        self.declare_parameter('rectified_left_topic', '/robot1/stereo/left/image_rect')
        self.declare_parameter('rectified_left_info_topic', '/robot1/stereo/left/camera_info_rect')
        self.declare_parameter('rectification_report_topic', '/robot1/stereo/rectification/report')
        self.declare_parameter('robot_pose_topic', '/model/box_pusher/pose')
        self.declare_parameter('box_pose_topic_prefix', '/model/orange_box_')
        self.declare_parameter('box_pose_topic_suffix', '/pose')
        self.declare_parameter('box_count', 28)
        self.declare_parameter('box_size_x_m', 0.08)
        self.declare_parameter('box_size_y_m', 0.08)
        self.declare_parameter('box_size_z_m', 0.10)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('left_optical_frame', 'left_camera_optical_frame')
        self.declare_parameter('max_ground_truth_age_sec', 0.75)
        self.declare_parameter('sample_radius_px', 1)
        self.declare_parameter('min_patch_valid_pixels', 3)
        self.declare_parameter('inlier_tolerance_m', 0.10)
        self.declare_parameter('max_samples_per_frame', 28)
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('metrics_topic', '/robot1/stereo/depth_validation/metrics')
        self.declare_parameter('overlay_topic', '/robot1/stereo/depth_validation/overlay')
        self.declare_parameter('ready_topic', '/robot1/stereo/depth_validation/ready')

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.box_count = int(self.get_parameter('box_count').value)
        self.box_dims = np.array([
            float(self.get_parameter('box_size_x_m').value),
            float(self.get_parameter('box_size_y_m').value),
            float(self.get_parameter('box_size_z_m').value),
        ], dtype=np.float64)
        self.max_pose_age_ns = int(float(self.get_parameter('max_ground_truth_age_sec').value) * 1e9)
        self.box_poses: Dict[str, Tuple[Pose, int]] = {}
        self.robot_pose: Optional[Tuple[Pose, int]] = None
        self.left_info: Optional[CameraInfo] = None
        self.rectification_r1: Optional[np.ndarray] = None
        self.rectification_report: Optional[dict] = None
        self.base_to_camera: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.frame_count = 0
        self.sample_count_total = 0
        self.last_warn_wall = 0.0

        reliable_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.metrics_pub = self.create_publisher(
            String, str(self.get_parameter('metrics_topic').value), 10
        )
        self.ready_pub = self.create_publisher(
            Bool, str(self.get_parameter('ready_topic').value), reliable_latched
        )
        self.overlay_pub = self.create_publisher(
            Image, str(self.get_parameter('overlay_topic').value), qos_profile_sensor_data
        )

        # Publish a live left-image preview independently from the depth/GT
        # synchronizer. The previous implementation only published overlay
        # after ALL validation prerequisites were satisfied, so rqt_image_view
        # stayed blank whenever any pose/TF/metadata input was missing.
        self.latest_left_image_msg: Optional[Image] = None
        self.latest_status = 'WAITING_FOR_RECTIFIED_IMAGE'
        self.preview_subscription = self.create_subscription(
            Image,
            str(self.get_parameter('rectified_left_topic').value),
            self.left_preview_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('rectified_left_info_topic').value),
            self.left_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('rectification_report_topic').value),
            self.rectification_report_callback,
            reliable_latched,
        )
        self.create_subscription(
            Pose,
            str(self.get_parameter('robot_pose_topic').value),
            self.robot_pose_callback,
            10,
        )
        prefix = str(self.get_parameter('box_pose_topic_prefix').value)
        suffix = str(self.get_parameter('box_pose_topic_suffix').value)
        self.box_subscriptions = []
        for i in range(1, self.box_count + 1):
            name = f'orange_box_{i:02d}'
            topic = f'{prefix}{i:02d}{suffix}'
            self.box_subscriptions.append(self.create_subscription(
                Pose, topic,
                lambda msg, box_name=name: self.box_pose_callback(box_name, msg),
                10,
            ))

        image_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('rectified_left_topic').value),
            qos_profile=qos_profile_sensor_data,
        )
        depth_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('depth_topic').value),
            qos_profile=qos_profile_sensor_data,
        )
        self.image_sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub], queue_size=6, slop=0.08, allow_headerless=False
        )
        self.image_sync.registerCallback(self.image_callback)
        self.get_logger().info(
            'Depth validator started. Live rectified-left preview is independent of '
            'depth/ground-truth readiness. Gazebo poses are evaluation-only.'
        )

    def left_preview_callback(self, msg: Image):
        """Keep the display alive even while validation inputs are missing."""
        self.latest_left_image_msg = msg
        if self.frame_count == 0 and self.latest_status == 'WAITING_FOR_RECTIFIED_IMAGE':
            self.latest_status = 'RECEIVING_LEFT_WAITING_FOR_DEPTH_GROUND_TRUTH_SYNC'
        if bool(self.get_parameter('publish_overlay').value):
            self._publish_status_overlay(msg, self.latest_status)

    def _publish_status_overlay(self, image_msg: Image, status: str, frame=None):
        try:
            if frame is None:
                frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8').copy()
            else:
                frame = frame.copy()
            h, w = frame.shape[:2]
            label = f'STEREO DEPTH VALIDATION: {status}'
            cv2.rectangle(frame, (0, 0), (w, 30), (25, 25, 25), -1)
            cv2.putText(
                frame, label[:max(1, int(w / 8))], (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
            )
            out = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out.header = image_msg.header
            self.overlay_pub.publish(out)
        except Exception as exc:
            self.get_logger().warning(f'Could not publish overlay preview: {exc}')

    def left_info_callback(self, msg: CameraInfo):
        self.left_info = msg

    def rectification_report_callback(self, msg: String):
        try:
            report = json.loads(msg.data)
            r1 = np.asarray(report['rectification_rotation_left'], dtype=np.float64).reshape(3, 3)
            p_left = np.asarray(report['rectified_projection_left'], dtype=np.float64).reshape(3, 4)
            if not np.all(np.isfinite(r1)) or not np.all(np.isfinite(p_left)):
                raise ValueError('Report contains non-finite rectification matrices')
            self.rectification_r1 = r1
            self.rectification_report = report
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f'Invalid rectification report: {exc}')

    def robot_pose_callback(self, msg: Pose):
        self.robot_pose = (msg, self.get_clock().now().nanoseconds)

    def box_pose_callback(self, name: str, msg: Pose):
        self.box_poses[name] = (msg, self.get_clock().now().nanoseconds)

    def _ensure_base_camera_transform(self):
        if self.base_to_camera is not None:
            return self.base_to_camera
        base = str(self.get_parameter('base_frame').value)
        optical = str(self.get_parameter('left_optical_frame').value)
        transform: TransformStamped = self.tf_buffer.lookup_transform(base, optical, Time())
        t = transform.transform.translation
        q = transform.transform.rotation
        r = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)
        self.base_to_camera = (r, translation)
        return self.base_to_camera

    @staticmethod
    def _pose_fresh(received_ns: int, now_ns: int, max_age_ns: int) -> bool:
        age = now_ns - received_ns
        return 0 <= age <= max_age_ns

    def _project_box_ray(
        self,
        box_pose: Pose,
        world_r_camera: np.ndarray,
        world_t_camera: np.ndarray,
        r1: np.ndarray,
        p_left: np.ndarray,
        all_box_poses: Dict[str, Pose],
        target_name: str,
    ) -> Optional[Tuple[int, int, float, float]]:
        box_r, box_t = pose_to_rt(box_pose.position, box_pose.orientation)
        ray = box_t - world_t_camera
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm < 1e-6:
            return None
        hit = ray_oriented_box_intersection(
            world_t_camera, ray, box_r, box_t, self.box_dims
        )
        if hit is None:
            return None
        target_distance, hit_world = hit

        # Discard the target if another known box occludes the ray first.
        for other_name, other_pose in all_box_poses.items():
            if other_name == target_name:
                continue
            other_r, other_t = pose_to_rt(other_pose.position, other_pose.orientation)
            other_hit = ray_oriented_box_intersection(
                world_t_camera, ray, other_r, other_t, self.box_dims
            )
            if other_hit is not None and other_hit[0] < target_distance - 0.01:
                return None

        projected = project_rectified_point(
            hit_world, world_r_camera, world_t_camera, r1, p_left
        )
        if projected is None:
            return None
        u, v, expected_z = projected
        return int(round(u)), int(round(v)), expected_z, target_distance

    def image_callback(self, left_image_msg: Image, depth_msg: Image):
        now_ns = self.get_clock().now().nanoseconds
        max_age = self.max_pose_age_ns
        if self.robot_pose is None or not self._pose_fresh(self.robot_pose[1], now_ns, max_age):
            self._publish_waiting('WAITING_FOR_FRESH_ROBOT_POSE')
            return
        if self.left_info is None or self.rectification_r1 is None or self.rectification_report is None:
            self._publish_waiting('WAITING_FOR_RECTIFICATION_METADATA')
            return
        if len(self.box_poses) < self.box_count:
            self._publish_waiting('WAITING_FOR_BOX_GROUND_TRUTH', boxes_received=len(self.box_poses))
            return

        fresh_boxes = {
            name: pose
            for name, (pose, received_ns) in self.box_poses.items()
            if self._pose_fresh(received_ns, now_ns, max_age)
        }
        if len(fresh_boxes) < self.box_count:
            self._publish_waiting('WAITING_FOR_FRESH_BOX_GROUND_TRUTH', boxes_fresh=len(fresh_boxes))
            return

        try:
            base_r_camera, base_t_camera = self._ensure_base_camera_transform()
        except TransformException as exc:
            self._publish_waiting('WAITING_FOR_CAMERA_TF', error=str(exc))
            return

        robot_pose = self.robot_pose[0]
        world_r_base, world_t_base = pose_to_rt(robot_pose.position, robot_pose.orientation)
        world_r_camera = world_r_base @ base_r_camera
        world_t_camera = world_t_base + world_r_base @ base_t_camera

        p_left = np.asarray(self.left_info.p, dtype=np.float64).reshape(3, 4)
        r1 = self.rectification_r1
        if p_left[0, 0] <= 0.0:
            self._publish_waiting('INVALID_RECTIFIED_CAMERA_INFO')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            overlay = self.bridge.imgmsg_to_cv2(left_image_msg, desired_encoding='bgr8').copy()
        except Exception as exc:  # CvBridge raises several conversion exception types.
            self.get_logger().error(f'Could not convert depth/rectified image: {exc}')
            self._publish_waiting('IMAGE_CONVERSION_ERROR', error=str(exc))
            return

        height, width = depth.shape[:2]
        if overlay.shape[:2] != (height, width):
            self._publish_waiting('IMAGE_SIZE_MISMATCH')
            return

        records = []
        sample_radius = max(0, int(self.get_parameter('sample_radius_px').value))
        min_valid = max(1, int(self.get_parameter('min_patch_valid_pixels').value))
        max_samples = max(1, int(self.get_parameter('max_samples_per_frame').value))
        tolerance = float(self.get_parameter('inlier_tolerance_m').value)

        for name, box_pose in fresh_boxes.items():
            result = self._project_box_ray(
                box_pose, world_r_camera, world_t_camera, r1, p_left,
                fresh_boxes, name,
            )
            if result is None:
                continue
            u, v, expected_z, ray_distance = result
            if not (0 <= u < width and 0 <= v < height and expected_z > 0.0):
                continue
            x0, x1 = max(0, u - sample_radius), min(width, u + sample_radius + 1)
            y0, y1 = max(0, v - sample_radius), min(height, v + sample_radius + 1)
            patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float32)
            valid_patch = patch[np.isfinite(patch) & (patch > 0.0)]
            if valid_patch.size < min_valid:
                cv2.circle(overlay, (u, v), 4, (0, 0, 255), 1)
                continue
            measured_z = float(np.median(valid_patch))
            error = measured_z - expected_z
            abs_error = abs(error)
            records.append({
                'box': name,
                'pixel': [u, v],
                'expected_depth_m': expected_z,
                'measured_depth_m': measured_z,
                'signed_error_m': error,
                'absolute_error_m': abs_error,
                'inlier': abs_error <= tolerance,
            })
            color = (0, 220, 0) if abs_error <= tolerance else (0, 165, 255)
            cv2.circle(overlay, (u, v), 4, color, 2)
            cv2.putText(
                overlay, f'{name[-2:]} {measured_z:.2f}/{expected_z:.2f}m',
                (min(width - 120, u + 5), max(14, v - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )
            if len(records) >= max_samples:
                break

        errors = np.asarray([r['absolute_error_m'] for r in records], dtype=np.float64)
        signed = np.asarray([r['signed_error_m'] for r in records], dtype=np.float64)
        if errors.size:
            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(signed * signed)))
            median_abs = float(np.median(errors))
            p90 = float(np.percentile(errors, 90))
            inlier_pct = 100.0 * float(np.count_nonzero(errors <= tolerance)) / float(errors.size)
            status = 'VALIDATED' if errors.size >= 3 else 'INSUFFICIENT_VISIBLE_BOXES'
        else:
            mae = rmse = median_abs = p90 = None
            inlier_pct = None
            status = 'NO_VALID_VISIBLE_BOX_SAMPLES'

        self.frame_count += 1
        self.sample_count_total += len(records)
        report = {
            'status': status,
            'frame_count': self.frame_count,
            'samples_this_frame': len(records),
            'samples_total': self.sample_count_total,
            'known_boxes': len(fresh_boxes),
            'mean_absolute_error_m': mae,
            'root_mean_square_error_m': rmse,
            'median_absolute_error_m': median_abs,
            'p90_absolute_error_m': p90,
            'inlier_tolerance_m': tolerance,
            'inlier_percent': inlier_pct,
            'rectified_baseline_m': self.rectification_report.get('rectified_baseline_m'),
            'rectified_focal_px': float(p_left[0, 0]),
            'measurement_source': 'stereo_depth_32FC1',
            'reference_source': 'gazebo_ground_truth_box_obb_ray_intersection',
            'ground_truth_usage': 'evaluation_only_not_policy_input',
            'samples': records,
        }
        metrics_msg = String()
        metrics_msg.data = json.dumps(report, separators=(',', ':'), allow_nan=False)
        self.metrics_pub.publish(metrics_msg)
        self.ready_pub.publish(Bool(data=(len(records) >= 3)))

        self.latest_status = status
        if bool(self.get_parameter('publish_overlay').value):
            self._publish_status_overlay(left_image_msg, status, frame=overlay)

        if self.frame_count == 1 or self.frame_count % 20 == 0:
            self.get_logger().info(
                f'Depth validation {status}: samples={len(records)}, '
                f'MAE={mae if mae is not None else "n/a"}, '
                f'RMSE={rmse if rmse is not None else "n/a"}'
            )

    def _publish_waiting(self, status: str, **extra):
        self.latest_status = status
        report = {'status': status, 'ground_truth_usage': 'evaluation_only_not_policy_input'}
        report.update(extra)
        msg = String()
        msg.data = json.dumps(report, separators=(',', ':'), allow_nan=False)
        self.metrics_pub.publish(msg)
        self.ready_pub.publish(Bool(data=False))
        if (bool(self.get_parameter('publish_overlay').value)
                and self.latest_left_image_msg is not None):
            self._publish_status_overlay(self.latest_left_image_msg, status)


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthValidationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
