#!/usr/bin/env python3
"""Estimate visible face geometry, square-box centre and approach direction.

Inputs are only rectified RGB/mask, metric stereo depth, CameraInfo, odometry
and the S2.5 local track list. Gazebo model poses/evaluator topics are never
subscribed. The square 80x80 mm footprint means object yaw is observable only
modulo 90 degrees unless an asymmetric visual marker is added to the box.
"""
from __future__ import annotations

from collections import deque
import json
import math
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, Point
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import (
    backproject_rectified_pixels,
    base_to_odom,
    fit_visible_vertical_face,
    interpolate_planar_pose,
    quaternion_from_yaw,
    quaternion_to_matrix,
    transform_points,
    transform_point,
    wrap_angle,
)


class OrangeBoxGeometryNode(Node):
    def __init__(self) -> None:
        super().__init__('orange_box_geometry_node')

        # S2.3 RGB-D perception output. All image topics are rectified/registered.
        self.declare_parameter('image_topic', '/robot1/stereo/left/image_rect')
        self.declare_parameter('depth_topic', '/robot1/stereo/depth')
        self.declare_parameter('mask_topic', '/robot1/perception/orange_boxes/mask')
        self.declare_parameter('camera_info_topic', '/robot1/stereo/left/camera_info_rect')
        self.declare_parameter('camera_frame', 'left_camera_optical_frame')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('sync_slop_sec', 0.08)
        self.declare_parameter('queue_size', 8)
        self.declare_parameter('max_rate_hz', 5.0)

        # Expected V2 task object dimensions: 80 x 80 x 100 mm.
        self.declare_parameter('box_xy_size_m', 0.08)
        self.declare_parameter('box_height_m', 0.10)
        self.declare_parameter('min_depth_m', 0.20)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('min_area_px', 120.0)
        self.declare_parameter('erode_kernel', 3)
        self.declare_parameter('min_points', 18)
        self.declare_parameter('max_points_per_detection', 800)
        self.declare_parameter('plane_threshold_m', 0.008)
        self.declare_parameter('max_normal_abs_z', 0.35)
        self.declare_parameter('ransac_iterations', 160)
        self.declare_parameter('min_face_width_m', 0.018)
        self.declare_parameter('max_face_width_m', 0.14)

        # S2.5 tracker/odom are used only to attach local IDs and compensate frame.
        self.declare_parameter('tracks_topic', '/robot1/perception/orange_boxes/tracks')
        self.declare_parameter('odom_topic', '/robot1/diff_drive_controller/odom')
        self.declare_parameter('track_match_gate_m', 0.30)
        self.declare_parameter('max_track_sync_sec', 0.30)
        self.declare_parameter('max_odom_sync_sec', 0.15)
        self.declare_parameter('odom_history_size', 300)
        self.declare_parameter('tf_timeout_sec', 0.06)

        self.declare_parameter('geometry_topic', '/robot1/perception/orange_boxes/geometry')
        self.declare_parameter('poses_topic', '/robot1/perception/orange_boxes/geometry/poses')
        self.declare_parameter('markers_topic', '/robot1/perception/orange_boxes/geometry/markers')
        self.declare_parameter('overlay_topic', '/robot1/perception/orange_boxes/geometry/overlay')
        self.declare_parameter('metrics_topic', '/robot1/perception/orange_boxes/geometry/metrics')
        self.declare_parameter('ready_topic', '/robot1/perception/orange_boxes/geometry/ready')

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_process_ns = 0
        self._odom: Deque[dict] = deque(maxlen=int(self.get_parameter('odom_history_size').value))
        self._track_snapshot: Optional[dict] = None
        self._track_stamp_ns: Optional[int] = None
        self._last_tf_warning = ''

        self.geometry_pub = self.create_publisher(String, str(self.get_parameter('geometry_topic').value), 10)
        self.poses_pub = self.create_publisher(PoseArray, str(self.get_parameter('poses_topic').value), 10)
        self.markers_pub = self.create_publisher(MarkerArray, str(self.get_parameter('markers_topic').value), 10)
        self.overlay_pub = self.create_publisher(Image, str(self.get_parameter('overlay_topic').value), qos_profile_sensor_data)
        self.metrics_pub = self.create_publisher(String, str(self.get_parameter('metrics_topic').value), 10)
        self.ready_pub = self.create_publisher(Bool, str(self.get_parameter('ready_topic').value), 10)

        self.odom_sub = self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value), self.on_odom, 50
        )
        self.tracks_sub = self.create_subscription(
            String, str(self.get_parameter('tracks_topic').value), self.on_tracks, 10
        )

        image_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('image_topic').value), qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('depth_topic').value), qos_profile=qos_profile_sensor_data
        )
        mask_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter('mask_topic').value), qos_profile=qos_profile_sensor_data
        )
        info_sub = message_filters.Subscriber(
            self, CameraInfo, str(self.get_parameter('camera_info_topic').value), qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub, mask_sub, info_sub],
            queue_size=int(self.get_parameter('queue_size').value),
            slop=float(self.get_parameter('sync_slop_sec').value),
            allow_headerless=False,
        )
        self.sync.registerCallback(self.on_rgbd)
        self.get_logger().info(
            'S2.6 geometry estimator started: RGB/mask/depth/CameraInfo + odom/tracks; '
            'Gazebo ground-truth is not subscribed.'
        )

    @staticmethod
    def _stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        item = {
            'stamp_ns': self._stamp_ns(msg.header.stamp),
            'frame_id': msg.header.frame_id,
            'child_frame_id': msg.child_frame_id,
            'xyz': (float(p.x), float(p.y), float(p.z)),
            'q': (float(q.x), float(q.y), float(q.z), float(q.w)),
        }
        if self._odom and item['stamp_ns'] <= self._odom[-1]['stamp_ns']:
            return
        self._odom.append(item)

    def on_tracks(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            stamp = payload.get('source_stamp')
            if isinstance(stamp, list) and len(stamp) == 2:
                stamp_ns = int(stamp[0]) * 1_000_000_000 + int(stamp[1])
            else:
                stamp_ns = None
            tracks = payload.get('tracks', [])
            if not isinstance(tracks, list):
                return
            clean = []
            for item in tracks:
                pos = item.get('position_odom_m')
                if not isinstance(pos, list) or len(pos) < 3:
                    continue
                vals = [float(v) for v in pos[:3]]
                if not all(math.isfinite(v) for v in vals):
                    continue
                clean.append({
                    'id': int(item.get('id', -1)),
                    'status': str(item.get('status', 'unknown')),
                    'position_odom_m': vals,
                    'time_since_seen_sec': float(item.get('time_since_seen_sec', 0.0)),
                })
            self._track_snapshot = {'tracks': clean}
            self._track_stamp_ns = stamp_ns
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f'Ignoring invalid S2.5 track message: {exc}')

    def _odom_at(self, stamp_ns: int) -> Optional[dict]:
        if not self._odom:
            return None
        samples = list(self._odom)
        stamps = [s['stamp_ns'] for s in samples]
        idx = int(np.searchsorted(stamps, stamp_ns))
        if idx == 0:
            chosen = samples[0]
            if abs(chosen['stamp_ns'] - stamp_ns) * 1e-9 > float(self.get_parameter('max_odom_sync_sec').value):
                return None
            return chosen
        if idx >= len(samples):
            chosen = samples[-1]
            if abs(chosen['stamp_ns'] - stamp_ns) * 1e-9 > float(self.get_parameter('max_odom_sync_sec').value):
                return None
            return chosen
        a, b = samples[idx-1], samples[idx]
        if a['frame_id'] != b['frame_id'] or a['child_frame_id'] != b['child_frame_id']:
            return None
        dt = b['stamp_ns'] - a['stamp_ns']
        if dt <= 0:
            return None
        alpha = (stamp_ns - a['stamp_ns']) / dt
        # Interpolate yaw only; V2 is planar differential drive.
        qa = quaternion_to_matrix(a['q'])
        qb = quaternion_to_matrix(b['q'])
        yaw_a = math.atan2(qa[1, 0], qa[0, 0])
        yaw_b = math.atan2(qb[1, 0], qb[0, 0])
        x, y, yaw = interpolate_planar_pose(
            a['xyz'][0], a['xyz'][1], yaw_a,
            b['xyz'][0], b['xyz'][1], yaw_b,
            alpha,
        )
        return {
            'stamp_ns': stamp_ns,
            'frame_id': a['frame_id'],
            'child_frame_id': a['child_frame_id'],
            'xyz': (x, y, (1-alpha)*a['xyz'][2] + alpha*b['xyz'][2]),
            'yaw': yaw,
            'q': None,
        }

    def _base_point_to_odom(self, point_base: Sequence[float], stamp, odom: dict) -> Optional[np.ndarray]:
        child = odom['child_frame_id']
        point_child = np.asarray(point_base, dtype=np.float64)
        if child != str(self.get_parameter('base_frame').value):
            try:
                tf = self.tf_buffer.lookup_transform(
                    child, str(self.get_parameter('base_frame').value),
                    Time.from_msg(stamp), timeout=Duration(seconds=float(self.get_parameter('tf_timeout_sec').value))
                )
                t = tf.transform.translation
                q = tf.transform.rotation
                point_child = transform_point((t.x, t.y, t.z), (q.x, q.y, q.z, q.w), point_child)
            except TransformException:
                return None
        if odom.get('yaw') is not None:
            x, y, z = odom['xyz']
            return base_to_odom(point_child, x, y, z, odom['yaw'])
        q = odom['q']
        return transform_point(odom['xyz'], q, point_child)

    def _match_track_ids(self, surface_points_base: List[np.ndarray], stamp_ns: int, header) -> List[Optional[int]]:
        result: List[Optional[int]] = [None] * len(surface_points_base)
        if self._track_snapshot is None or self._track_stamp_ns is None:
            return result
        max_sync = float(self.get_parameter('max_track_sync_sec').value)
        if abs(self._track_stamp_ns - stamp_ns) * 1e-9 > max_sync:
            return result
        odom = self._odom_at(stamp_ns)
        if odom is None:
            return result
        predicted = []
        for i, point in enumerate(surface_points_base):
            p_odom = self._base_point_to_odom(point, header.stamp, odom)
            if p_odom is not None:
                predicted.append((i, p_odom))
        tracks = self._track_snapshot['tracks']
        pairs = []
        for det_idx, p in predicted:
            for track_idx, track in enumerate(tracks):
                if track['status'] == 'coasting' or track['id'] < 0:
                    continue
                q = np.asarray(track['position_odom_m'], dtype=np.float64)
                d = float(np.linalg.norm(p[:2] - q[:2]))
                if d <= float(self.get_parameter('track_match_gate_m').value):
                    pairs.append((d, det_idx, track_idx))
        used_dets, used_tracks = set(), set()
        for _, det_idx, track_idx in sorted(pairs, key=lambda item: item[0]):
            if det_idx in used_dets or track_idx in used_tracks:
                continue
            result[det_idx] = int(tracks[track_idx]['id'])
            used_dets.add(det_idx)
            used_tracks.add(track_idx)
        return result

    def _publish_empty(self, header, status: str, extra: Optional[dict] = None) -> None:
        poses = PoseArray()
        poses.header = header
        poses.header.frame_id = str(self.get_parameter('base_frame').value)
        self.poses_pub.publish(poses)
        markers = MarkerArray()
        delete = Marker()
        delete.header = poses.header
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)
        self.markers_pub.publish(markers)
        payload = {'status': status, 'object_count': 0, 'frame_id': poses.header.frame_id}
        if extra:
            payload.update(extra)
        self._publish_json(self.geometry_pub, payload)
        self._publish_json(self.metrics_pub, payload)
        self.ready_pub.publish(Bool(data=False))

    def on_rgbd(self, image_msg: Image, depth_msg: Image, mask_msg: Image, info_msg: CameraInfo) -> None:
        stamp_ns = self._stamp_ns(image_msg.header.stamp)
        rate = float(self.get_parameter('max_rate_hz').value)
        now_ns = self.get_clock().now().nanoseconds
        if rate > 0 and self._last_process_ns and now_ns - self._last_process_ns < int(1e9 / rate):
            return
        self._last_process_ns = now_ns
        try:
            if not (image_msg.width == depth_msg.width == mask_msg.width and
                    image_msg.height == depth_msg.height == mask_msg.height):
                self._publish_empty(image_msg.header, 'RGB_DEPTH_MASK_SIZE_MISMATCH')
                return
            if len(info_msg.p) != 12 or len(info_msg.r) != 9:
                self._publish_empty(image_msg.header, 'INVALID_RECTIFIED_CAMERA_INFO')
                return
            projection = np.asarray(info_msg.p, dtype=np.float64).reshape(3, 4)
            fx, fy, cx, cy = float(projection[0, 0]), float(projection[1, 1]), float(projection[0, 2]), float(projection[1, 2])
            if fx <= 0 or fy <= 0:
                self._publish_empty(image_msg.header, 'INVALID_RECTIFIED_INTRINSICS')
                return
            bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
            if depth.shape != mask.shape:
                self._publish_empty(image_msg.header, 'DEPTH_MASK_SHAPE_MISMATCH')
                return

            r1 = np.asarray(info_msg.r, dtype=np.float64).reshape(3, 3)
            # Transform from raw optical camera frame to base; un-rectify cloud points first.
            camera_frame = image_msg.header.frame_id or str(self.get_parameter('camera_frame').value)
            base_frame = str(self.get_parameter('base_frame').value)
            tf = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, Time.from_msg(image_msg.header.stamp),
                timeout=Duration(seconds=float(self.get_parameter('tf_timeout_sec').value))
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            camera_origin_base = np.array([t.x, t.y, t.z], dtype=np.float64)

            kernel = max(1, int(self.get_parameter('erode_kernel').value))
            if kernel % 2 == 0:
                kernel += 1
            erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
            work_mask = cv2.erode(mask, erode, iterations=1) if kernel > 1 else mask
            contours, _ = cv2.findContours(work_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            min_area = float(self.get_parameter('min_area_px').value)
            min_depth = float(self.get_parameter('min_depth_m').value)
            max_depth = float(self.get_parameter('max_depth_m').value)
            box_xy = float(self.get_parameter('box_xy_size_m').value)
            box_h = float(self.get_parameter('box_height_m').value)
            max_pts = int(self.get_parameter('max_points_per_detection').value)
            records = []
            surface_base_points: List[np.ndarray] = []
            estimates = []
            overlay = bgr.copy()
            frame_index = 0

            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                component = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)
                ys, xs = np.nonzero(component)
                if xs.size == 0:
                    continue
                zvals = depth[ys, xs].astype(np.float64)
                valid = np.isfinite(zvals) & (zvals >= min_depth) & (zvals <= max_depth)
                xs, ys, zvals = xs[valid], ys[valid], zvals[valid]
                if len(zvals) < int(self.get_parameter('min_points').value):
                    continue
                median_z = float(np.median(zvals))
                mad_z = float(np.median(np.abs(zvals - median_z)))
                z_gate = max(0.04, 3.5 * 1.4826 * mad_z)
                keep = np.abs(zvals - median_z) <= z_gate
                xs, ys, zvals = xs[keep], ys[keep], zvals[keep]
                if len(zvals) < int(self.get_parameter('min_points').value):
                    continue
                if len(zvals) > max_pts:
                    inds = np.linspace(0, len(zvals)-1, max_pts).astype(int)
                    xs, ys, zvals = xs[inds], ys[inds], zvals[inds]

                points_rect = backproject_rectified_pixels(xs, ys, zvals, fx, fy, cx, cy)
                points_raw = (r1.T @ points_rect.T).T
                points_base = transform_points(
                    (t.x, t.y, t.z), (q.x, q.y, q.z, q.w), points_raw
                )
                surface = np.median(points_base, axis=0)
                fit = fit_visible_vertical_face(
                    points_base,
                    camera_origin_base,
                    box_xy_size_m=box_xy,
                    box_height_m=box_h,
                    plane_threshold_m=float(self.get_parameter('plane_threshold_m').value),
                    max_normal_abs_z=float(self.get_parameter('max_normal_abs_z').value),
                    min_points=int(self.get_parameter('min_points').value),
                    ransac_iterations=int(self.get_parameter('ransac_iterations').value),
                    seed=13 + frame_index,
                )
                if fit.success:
                    min_width = float(self.get_parameter('min_face_width_m').value)
                    max_width = float(self.get_parameter('max_face_width_m').value)
                    if not (min_width <= float(fit.observed_width_m) <= max_width):
                        fit.success = False
                        fit.reason = 'OBSERVED_FACE_WIDTH_OUT_OF_RANGE'
                record = {
                    'detection_index': frame_index,
                    'bbox_xywh': [int(x), int(y), int(w), int(h)],
                    'contour_area_px': area,
                    'point_count': int(len(points_base)),
                    'visible_surface_centroid_base_m': [float(v) for v in surface],
                    'geometry_valid': bool(fit.success),
                    'geometry_reason': fit.reason,
                    'geometry_method': 'ransac_vertical_face_plus_known_box_dimensions',
                    'known_box_size_prior_m': [box_xy, box_xy, box_h],
                    'yaw_observability': 'square_footprint_yaw_modulo_90deg_only',
                }
                if fit.success:
                    record.update({
                        'geometric_center_base_m': [float(v) for v in fit.geometric_center],
                        'visible_face_center_base_m': [float(v) for v in fit.surface_center],
                        'visible_face_outward_normal_base': [float(v) for v in fit.outward_normal],
                        'edge_direction_base': [float(v) for v in fit.edge_direction],
                        'yaw_edge_mod_pi_over_2_rad': float(fit.yaw_edge_mod_pi_over_2),
                        'approach_yaw_base_rad': float(fit.approach_yaw),
                        'observed_face_width_m': float(fit.observed_width_m),
                        'observed_face_height_m': float(fit.observed_height_m),
                        'plane_inlier_count': int(fit.plane_inlier_count),
                        'plane_inlier_ratio': float(fit.plane_inlier_ratio),
                        'plane_rmse_m': float(fit.plane_rmse_m),
                        'confidence': float(fit.confidence),
                        'center_z_method': 'known_height_half_prior_on_floor_supported_box',
                    })
                surface_base_points.append(surface)
                estimates.append(fit)
                records.append(record)
                cv2.drawContours(overlay, [contour], -1, (0, 220, 0) if fit.success else (0, 165, 255), 2)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 190, 0), 1)
                label = f'box{frame_index} ' + (f'yaw~{math.degrees(fit.yaw_edge_mod_pi_over_2):.1f}deg c={fit.confidence:.2f}' if fit.success else fit.reason)
                cv2.putText(overlay, label, (x, max(14, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (0, 220, 0) if fit.success else (0, 165, 255), 1, cv2.LINE_AA)
                frame_index += 1

            track_ids = self._match_track_ids(surface_base_points, stamp_ns, image_msg.header)
            for idx, track_id in enumerate(track_ids):
                records[idx]['track_id'] = track_id
                if track_id is not None:
                    x, y, w, h = records[idx]['bbox_xywh']
                    cv2.putText(overlay, f'ID {track_id}', (x, y+h+14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 0), 1, cv2.LINE_AA)

            poses = PoseArray()
            poses.header = image_msg.header
            poses.header.frame_id = base_frame
            markers = MarkerArray()
            delete = Marker()
            delete.header = poses.header
            delete.action = Marker.DELETEALL
            markers.markers.append(delete)
            valid_count = 0
            for idx, (record, fit) in enumerate(zip(records, estimates)):
                if not fit.success:
                    continue
                valid_count += 1
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = [float(v) for v in fit.geometric_center]
                qy = quaternion_from_yaw(float(fit.yaw_edge_mod_pi_over_2))
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qy
                poses.poses.append(pose)

                marker = Marker()
                marker.header = poses.header
                marker.ns = 'orange_box_geometry'
                marker.id = 3 * idx
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose = pose
                marker.scale.x = box_xy
                marker.scale.y = box_xy
                marker.scale.z = box_h
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.42, 0.02, 0.55)
                marker.lifetime = Duration(seconds=0.6).to_msg()
                markers.markers.append(marker)

                arrow = Marker()
                arrow.header = poses.header
                arrow.ns = 'orange_box_approach_normal'
                arrow.id = 3 * idx + 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                start = fit.surface_center
                end = fit.geometric_center
                arrow.points = [Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
                                Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = (0.008, 0.018, 0.025)
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (0.1, 0.9, 1.0, 0.95)
                arrow.lifetime = Duration(seconds=0.6).to_msg()
                markers.markers.append(arrow)

                label = Marker()
                label.header = poses.header
                label.ns = 'orange_box_geometry_label'
                label.id = 3 * idx + 2
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose = pose
                label.pose.position.z += box_h * 0.75
                label.scale.z = 0.045
                label.color.r, label.color.g, label.color.b, label.color.a = (1.0, 1.0, 1.0, 1.0)
                label.text = f"ID {record['track_id']}" if record['track_id'] is not None else f"det {idx}"
                label.lifetime = Duration(seconds=0.6).to_msg()
                markers.markers.append(label)

            self.poses_pub.publish(poses)
            self.markers_pub.publish(markers)
            self.overlay_pub.publish(self._with_header(self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8'), image_msg.header))

            payload = {
                'status': 'OK' if records else 'NO_ORANGE_MASK_COMPONENTS',
                'source_stamp': [int(image_msg.header.stamp.sec), int(image_msg.header.stamp.nanosec)],
                'frame_id': base_frame,
                'camera_frame': camera_frame,
                'candidate_count': len(records),
                'geometry_valid_count': valid_count,
                'track_ids_matched': sum(1 for v in track_ids if v is not None),
                'known_box_dimensions_prior_m': [box_xy, box_xy, box_h],
                'orientation_limit': 'unmarked square footprint: yaw only modulo 90 degrees; approach face normal is separately estimated',
                'ground_truth_used': False,
                'objects': records,
            }
            self._publish_json(self.geometry_pub, payload)
            self._publish_json(self.metrics_pub, {k: v for k, v in payload.items() if k != 'objects'})
            self.ready_pub.publish(Bool(data=True))
        except TransformException as exc:
            self._publish_empty(image_msg.header, 'CAMERA_TO_BASE_TF_UNAVAILABLE', {'error': str(exc)})
        except Exception as exc:
            self.get_logger().error(f'Geometry callback failed: {type(exc).__name__}: {exc}')
            self._publish_empty(image_msg.header, 'PROCESSING_ERROR', {'error': str(exc)})

    @staticmethod
    def _publish_json(pub, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload, separators=(',', ':'), allow_nan=False)
        pub.publish(msg)

    @staticmethod
    def _with_header(msg: Image, header) -> Image:
        msg.header = header
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrangeBoxGeometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
