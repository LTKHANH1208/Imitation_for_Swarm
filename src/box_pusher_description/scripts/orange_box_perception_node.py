#!/usr/bin/env python3
"""Detect orange boxes in V2 rectified RGB-D images and estimate 3D surface points.

This node is perception-only: it subscribes to rectified RGB, stereo depth and
rectified CameraInfo. It never subscribes to Gazebo ground-truth poses.
The reported 3D point is the robust centroid of the visible surface pixels,
not the hidden geometric centre of the complete box.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from orange_box_perception_geometry import (
    find_orange_contours,
    make_orange_mask,
    project_depth_region_to_xyz,
    transform_point_xyz,
)


class OrangeBoxPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__('orange_box_perception_node')

        # Inputs: all RGB-D inputs must refer to the rectified left optical frame.
        self.declare_parameter('image_topic', '/robot1/stereo/left/image_rect')
        self.declare_parameter('depth_topic', '/robot1/stereo/depth')
        self.declare_parameter('camera_info_topic', '/robot1/stereo/left/camera_info_rect')
        self.declare_parameter('camera_frame', 'left_camera_optical_frame')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('sync_slop_sec', 0.08)
        self.declare_parameter('queue_size', 8)
        self.declare_parameter('max_rate_hz', 5.0)

        # Orange segmentation. Initial range suits the Gazebo material
        # ambient/diffuse approximately RGB=(0.95, 0.28, 0.025), but must be
        # tuned against the actual rendered camera image.
        self.declare_parameter('hue_min', 3)
        self.declare_parameter('hue_max', 30)
        self.declare_parameter('saturation_min', 100)
        self.declare_parameter('value_min', 60)
        self.declare_parameter('morph_kernel', 3)
        self.declare_parameter('min_area_px', 120.0)
        self.declare_parameter('min_width_px', 5)
        self.declare_parameter('min_height_px', 5)
        self.declare_parameter('min_extent', 0.20)
        self.declare_parameter('max_aspect_ratio', 4.0)
        self.declare_parameter('max_detections', 28)

        # Depth filtering.
        self.declare_parameter('min_depth_m', 0.20)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('min_valid_depth_pixels', 8)
        self.declare_parameter('erode_kernel', 3)

        # Output topics.
        self.declare_parameter('overlay_topic', '/robot1/perception/orange_boxes/overlay')
        self.declare_parameter('mask_topic', '/robot1/perception/orange_boxes/mask')
        self.declare_parameter('centers_camera_topic', '/robot1/perception/orange_boxes/centers_camera')
        self.declare_parameter('centers_base_topic', '/robot1/perception/orange_boxes/centers_base')
        self.declare_parameter('markers_topic', '/robot1/perception/orange_boxes/markers')
        self.declare_parameter('metrics_topic', '/robot1/perception/orange_boxes/metrics')
        self.declare_parameter('ready_topic', '/robot1/perception/orange_boxes/ready')

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_process_ns = 0
        self._frame_count = 0
        self._tf_warning_sent = False

        self.overlay_pub = self.create_publisher(
            Image, str(self.get_parameter('overlay_topic').value), qos_profile_sensor_data
        )
        self.mask_pub = self.create_publisher(
            Image, str(self.get_parameter('mask_topic').value), qos_profile_sensor_data
        )
        self.centers_camera_pub = self.create_publisher(
            PoseArray, str(self.get_parameter('centers_camera_topic').value), 10
        )
        self.centers_base_pub = self.create_publisher(
            PoseArray, str(self.get_parameter('centers_base_topic').value), 10
        )
        self.markers_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter('markers_topic').value), 10
        )
        self.metrics_pub = self.create_publisher(
            String, str(self.get_parameter('metrics_topic').value), 10
        )
        ready_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_pub = self.create_publisher(
            Bool, str(self.get_parameter('ready_topic').value), ready_qos
        )

        image_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('image_topic').value),
            qos_profile=qos_profile_sensor_data,
        )
        depth_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('depth_topic').value),
            qos_profile=qos_profile_sensor_data,
        )
        info_sub = message_filters.Subscriber(
            self, CameraInfo, str(self.get_parameter('camera_info_topic').value),
            qos_profile=qos_profile_sensor_data,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub, info_sub],
            queue_size=int(self.get_parameter('queue_size').value),
            slop=float(self.get_parameter('sync_slop_sec').value),
            allow_headerless=False,
        )
        self.sync.registerCallback(self.on_rgbd)
        self.get_logger().info(
            'Orange-box perception started: rectified RGB + metric depth + CameraInfo; '
            'Gazebo ground-truth is not subscribed.'
        )

    @staticmethod
    def _pose_from_xyz(xyz: np.ndarray) -> Pose:
        pose = Pose()
        pose.position.x = float(xyz[0])
        pose.position.y = float(xyz[1])
        pose.position.z = float(xyz[2])
        pose.orientation.w = 1.0
        return pose

    def _publish_empty(self, header, status: str, **extra) -> None:
        camera_array = PoseArray()
        camera_array.header = header
        self.centers_camera_pub.publish(camera_array)
        base_array = PoseArray()
        base_array.header.stamp = header.stamp
        base_array.header.frame_id = str(self.get_parameter('base_frame').value)
        self.centers_base_pub.publish(base_array)
        marker_array = MarkerArray()
        delete = Marker()
        delete.header = base_array.header
        delete.action = Marker.DELETEALL
        marker_array.markers.append(delete)
        self.markers_pub.publish(marker_array)
        report = {'status': status, 'detection_count': 0, 'detections': []}
        report.update(extra)
        msg = String()
        msg.data = json.dumps(report, separators=(',', ':'), allow_nan=False)
        self.metrics_pub.publish(msg)
        self.ready_pub.publish(Bool(data=False))

    def on_rgbd(self, image_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        now_ns = self.get_clock().now().nanoseconds
        max_rate = float(self.get_parameter('max_rate_hz').value)
        if max_rate > 0.0 and self._last_process_ns > 0:
            if now_ns - self._last_process_ns < int(1e9 / max_rate):
                return
        self._last_process_ns = now_ns

        try:
            if image_msg.width != depth_msg.width or image_msg.height != depth_msg.height:
                self._publish_empty(image_msg.header, 'IMAGE_DEPTH_SIZE_MISMATCH')
                return
            if len(info_msg.p) != 12:
                self._publish_empty(image_msg.header, 'INVALID_CAMERA_INFO_P')
                return
            p = np.asarray(info_msg.p, dtype=np.float64).reshape(3, 4)
            fx, fy = float(p[0, 0]), float(p[1, 1])
            cx, cy = float(p[0, 2]), float(p[1, 2])
            if fx <= 0.0 or fy <= 0.0:
                self._publish_empty(image_msg.header, 'INVALID_CAMERA_FOCAL_LENGTH')
                return
            if info_msg.header.frame_id and image_msg.header.frame_id and info_msg.header.frame_id != image_msg.header.frame_id:
                self.get_logger().warning(
                    f'Image/CameraInfo frame mismatch: {image_msg.header.frame_id} vs {info_msg.header.frame_id}'
                )

            bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            if depth.ndim != 2:
                raise ValueError(f'Expected 32FC1 depth, got shape={depth.shape}')

            mask = make_orange_mask(
                bgr,
                hue_min=int(self.get_parameter('hue_min').value),
                hue_max=int(self.get_parameter('hue_max').value),
                saturation_min=int(self.get_parameter('saturation_min').value),
                value_min=int(self.get_parameter('value_min').value),
                morph_kernel=int(self.get_parameter('morph_kernel').value),
            )
            candidates = find_orange_contours(
                mask,
                min_area_px=float(self.get_parameter('min_area_px').value),
                min_width_px=int(self.get_parameter('min_width_px').value),
                min_height_px=int(self.get_parameter('min_height_px').value),
                min_extent=float(self.get_parameter('min_extent').value),
                max_aspect_ratio=float(self.get_parameter('max_aspect_ratio').value),
            )
            candidates = candidates[:int(self.get_parameter('max_detections').value)]

            # Erode each component slightly so mixed boundary pixels are less likely
            # to corrupt median depth (especially at box/background edges).
            erode_k = max(1, int(self.get_parameter('erode_kernel').value))
            if erode_k % 2 == 0:
                erode_k += 1
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))

            camera_frame = image_msg.header.frame_id or str(self.get_parameter('camera_frame').value)
            base_frame = str(self.get_parameter('base_frame').value)
            camera_to_base = None
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    base_frame,
                    camera_frame,
                    Time.from_msg(image_msg.header.stamp),
                    timeout=Duration(seconds=0.08),
                )
                t = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                camera_to_base = (tf_msg.transform.translation, (q.x, q.y, q.z, q.w))
            except TransformException as exc:
                if not self._tf_warning_sent:
                    self.get_logger().warning(f'Cannot transform detections to {base_frame}: {exc}')
                    self._tf_warning_sent = True

            overlay = bgr.copy()
            camera_poses = PoseArray()
            camera_poses.header = image_msg.header
            base_poses = PoseArray()
            base_poses.header.stamp = image_msg.header.stamp
            base_poses.header.frame_id = base_frame
            markers = MarkerArray()
            delete = Marker()
            delete.header = base_poses.header
            delete.action = Marker.DELETEALL
            markers.markers.append(delete)
            detections: List[Dict[str, object]] = []
            valid_depth_count = 0

            min_depth = float(self.get_parameter('min_depth_m').value)
            max_depth = float(self.get_parameter('max_depth_m').value)
            min_valid = int(self.get_parameter('min_valid_depth_pixels').value)
            h_img, w_img = mask.shape[:2]

            for det_index, candidate in enumerate(candidates):
                contour = candidate['contour']
                x, y, w, h = candidate['bbox']
                component = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
                if erode_k > 1:
                    component = cv2.erode(component, erode_kernel, iterations=1)
                xyz, stats = project_depth_region_to_xyz(
                    depth, component, fx, fy, cx, cy,
                    min_depth_m=min_depth,
                    max_depth_m=max_depth,
                    min_valid_pixels=min_valid,
                )
                if xyz is None:
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(overlay, 'ORANGE? NO VALID DEPTH', (x, max(14, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
                    continue

                valid_depth_count += 1
                camera_poses.poses.append(self._pose_from_xyz(xyz))
                base_xyz = None
                if camera_to_base is not None:
                    trans, quat = camera_to_base
                    base_xyz = transform_point_xyz(
                        xyz,
                        [trans.x, trans.y, trans.z],
                        quat,
                    )
                    base_poses.poses.append(self._pose_from_xyz(base_xyz))

                    sphere = Marker()
                    sphere.header = base_poses.header
                    sphere.ns = 'orange_box_surface_centroids'
                    sphere.id = det_index
                    sphere.type = Marker.SPHERE
                    sphere.action = Marker.ADD
                    sphere.pose = self._pose_from_xyz(base_xyz)
                    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.045
                    sphere.color.r = 1.0
                    sphere.color.g = 0.45
                    sphere.color.b = 0.02
                    sphere.color.a = 0.95
                    sphere.lifetime = Duration(seconds=0.5).to_msg()
                    markers.markers.append(sphere)

                # 2D box and median-depth label.
                cv2.drawContours(overlay, [contour], -1, (0, 220, 0), 2)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 0), 1)
                pixel_center = [int(x + w // 2), int(y + h // 2)]
                label = f'orange_{det_index:02d} Z={float(stats["median_m"]):.2f}m'
                cv2.circle(overlay, tuple(pixel_center), 3, (255, 255, 0), -1)
                cv2.putText(overlay, label, (x, max(14, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 0), 1, cv2.LINE_AA)

                record: Dict[str, object] = {
                    'detection_index': det_index,
                    'bbox_xywh': [int(x), int(y), int(w), int(h)],
                    'bbox_center_uv': pixel_center,
                    'contour_area_px': float(candidate['area_px']),
                    'depth_median_m': float(stats['median_m']),
                    'depth_mad_m': float(stats['mad_m']),
                    'depth_valid_ratio': float(stats['valid_ratio']),
                    'visible_surface_centroid_camera_m': [float(v) for v in xyz],
                    'coordinate_frame_camera': camera_frame,
                    'point_semantics': 'visible_surface_centroid_not_geometric_box_center',
                }
                if base_xyz is not None:
                    record['visible_surface_centroid_base_m'] = [float(v) for v in base_xyz]
                    record['coordinate_frame_base'] = base_frame
                detections.append(record)

            self.overlay_pub.publish(self._with_header(
                self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8'), image_msg.header
            ))
            self.mask_pub.publish(self._with_header(self.bridge.cv2_to_imgmsg(mask, encoding='mono8'), image_msg.header))
            self.centers_camera_pub.publish(camera_poses)
            self.centers_base_pub.publish(base_poses)
            self.markers_pub.publish(markers)

            self._frame_count += 1
            report = {
                'status': 'OK' if valid_depth_count > 0 else ('NO_ORANGE_CANDIDATES' if not candidates else 'ORANGE_WITHOUT_VALID_DEPTH'),
                'frame_count': self._frame_count,
                'input_image_size': [int(w_img), int(h_img)],
                'candidate_count': len(candidates),
                'valid_depth_detection_count': valid_depth_count,
                'base_frame_transform_available': camera_to_base is not None,
                'hsv_thresholds': {
                    'hue_min': int(self.get_parameter('hue_min').value),
                    'hue_max': int(self.get_parameter('hue_max').value),
                    'saturation_min': int(self.get_parameter('saturation_min').value),
                    'value_min': int(self.get_parameter('value_min').value),
                },
                'coordinate_semantics': 'visible_surface_centroid_not_geometric_box_center',
                'ground_truth_used': False,
                'detections': detections,
            }
            metrics_msg = String()
            metrics_msg.data = json.dumps(report, separators=(',', ':'), allow_nan=False)
            self.metrics_pub.publish(metrics_msg)
            self.ready_pub.publish(Bool(data=True))

            if self._frame_count == 1 or self._frame_count % 30 == 0:
                self.get_logger().info(
                    f'Orange perception: candidates={len(candidates)}, '
                    f'valid_depth={valid_depth_count}, base_tf={camera_to_base is not None}'
                )
        except Exception as exc:
            self.get_logger().error(f'RGB-D perception callback failed: {type(exc).__name__}: {exc}')
            self._publish_empty(image_msg.header, 'PROCESSING_ERROR', error=str(exc))

    @staticmethod
    def _with_header(msg: Image, header) -> Image:
        msg.header = header
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrangeBoxPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
