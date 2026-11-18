#!/usr/bin/env python3
"""Quantitatively evaluate orange-box RGB-D detections against Gazebo ground truth.

This node is evaluator-only. Gazebo poses are never forwarded to or consumed by
orange_box_perception_node or the future imitation policy.

2-D evaluation: project each oriented Gazebo box into the rectified left image,
remove boxes fully occluded by other boxes (sampled ray visibility), then perform
greedy one-to-one IoU matching against perception bounding boxes.

3-D evaluation: for each true-positive detection with valid stereo depth, ray-cast
a grid over its detected image bbox against the Gazebo oriented boxes. The median
visible surface point of the matched box is the reference in rectified-camera
coordinates. Compare it with the detection node's rectified-camera XYZ estimate.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

from perception_evaluation_geometry import (
    pose_to_rt,
    project_obb_bbox,
    visible_fraction_for_box,
    visible_surface_centroid_for_bbox,
    greedy_iou_match,
)


class OrangeBoxPerceptionEvaluator(Node):
    def __init__(self) -> None:
        super().__init__('orange_box_perception_evaluator_node')

        self.declare_parameter('perception_metrics_topic', '/robot1/perception/orange_boxes/metrics')
        self.declare_parameter('perception_overlay_topic', '/robot1/perception/orange_boxes/overlay')
        self.declare_parameter('camera_info_topic', '/robot1/stereo/left/camera_info_rect')
        self.declare_parameter('rectification_report_topic', '/robot1/stereo/rectification/report')
        self.declare_parameter('robot_pose_topic', '/model/box_pusher/pose')
        self.declare_parameter('box_pose_topic_prefix', '/model/orange_box_')
        self.declare_parameter('box_pose_topic_suffix', '/pose')
        self.declare_parameter('box_count', 28)
        self.declare_parameter('box_size_x_m', 0.08)
        self.declare_parameter('box_size_y_m', 0.08)
        self.declare_parameter('box_size_z_m', 0.10)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('camera_frame', 'left_camera_optical_frame')
        self.declare_parameter('max_ground_truth_age_sec', 0.75)
        self.declare_parameter('evaluation_rate_hz', 2.0)
        self.declare_parameter('minimum_projected_area_px', 25.0)
        self.declare_parameter('visibility_grid_size', 5)
        self.declare_parameter('minimum_visible_ray_hits', 1)
        self.declare_parameter('minimum_match_iou', 0.10)
        self.declare_parameter('centroid_grid_size', 9)
        self.declare_parameter('max_metric_history', 10000)
        self.declare_parameter('metrics_topic', '/robot1/perception/orange_boxes/evaluation/metrics')
        self.declare_parameter('overlay_topic', '/robot1/perception/orange_boxes/evaluation/overlay')
        self.declare_parameter('ready_topic', '/robot1/perception/orange_boxes/evaluation/ready')

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
        self.robot_pose: Optional[Tuple[Pose, int]] = None
        self.box_poses: Dict[str, Tuple[Pose, int]] = {}
        self.camera_info: Optional[CameraInfo] = None
        self.rectification_report: Optional[dict] = None
        self.overlay_cache: Optional[Image] = None
        self.base_to_camera: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_source_stamp: Optional[Tuple[int, int]] = None
        self.last_eval_ns = 0
        self.frame_count = 0
        self.cum_tp = 0
        self.cum_fp = 0
        self.cum_fn = 0
        self.err_3d_history: List[float] = []
        self.err_z_history: List[float] = []
        self.last_wait_status = ''

        self.metrics_pub = self.create_publisher(
            String, str(self.get_parameter('metrics_topic').value), 10
        )
        self.overlay_pub = self.create_publisher(
            Image, str(self.get_parameter('overlay_topic').value), qos_profile_sensor_data
        )
        ready_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_pub = self.create_publisher(
            Bool, str(self.get_parameter('ready_topic').value), ready_qos
        )

        self.create_subscription(
            String,
            str(self.get_parameter('perception_metrics_topic').value),
            self.perception_metrics_callback,
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('perception_overlay_topic').value),
            self.overlay_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('rectification_report_topic').value),
            self.rectification_report_callback,
            latched_qos,
        )
        self.create_subscription(
            Pose,
            str(self.get_parameter('robot_pose_topic').value),
            self.robot_pose_callback,
            10,
        )
        self.box_subscriptions = []
        prefix = str(self.get_parameter('box_pose_topic_prefix').value)
        suffix = str(self.get_parameter('box_pose_topic_suffix').value)
        for i in range(1, self.box_count + 1):
            name = f'orange_box_{i:02d}'
            topic = f'{prefix}{i:02d}{suffix}'
            self.box_subscriptions.append(self.create_subscription(
                Pose, topic,
                lambda msg, box_name=name: self.box_pose_callback(box_name, msg),
                10,
            ))

        self.get_logger().info(
            'S2.4 perception evaluator started. Ground truth subscriptions are evaluator-only.'
        )

    def _publish_waiting(self, status: str, **extra) -> None:
        data = {'status': status, 'evaluation_enabled': False}
        data.update(extra)
        msg = String()
        msg.data = json.dumps(data, separators=(',', ':'), allow_nan=False)
        self.metrics_pub.publish(msg)
        self.ready_pub.publish(Bool(data=False))
        if status != self.last_wait_status:
            self.get_logger().info(f'S2.4 waiting: {status}')
            self.last_wait_status = status

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def rectification_report_callback(self, msg: String) -> None:
        try:
            report = json.loads(msg.data)
            r1 = np.asarray(report['rectification_rotation_left'], dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(r1)):
                raise ValueError('non-finite R1')
            self.rectification_report = report
            self.get_logger().info('Received complete rectification metadata (R1/P1).')
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().error(f'Invalid rectification report: {exc}')
            self.rectification_report = None

    def robot_pose_callback(self, msg: Pose) -> None:
        self.robot_pose = (msg, self.get_clock().now().nanoseconds)

    def box_pose_callback(self, name: str, msg: Pose) -> None:
        self.box_poses[name] = (msg, self.get_clock().now().nanoseconds)

    def overlay_callback(self, msg: Image) -> None:
        self.overlay_cache = msg

    @staticmethod
    def _is_fresh(received_ns: int, now_ns: int, max_age_ns: int) -> bool:
        age = now_ns - received_ns
        return 0 <= age <= max_age_ns

    def _wait_for_inputs(self, now_ns: int) -> Optional[str]:
        if self.camera_info is None or self.rectification_report is None:
            return 'WAITING_FOR_RECTIFICATION_METADATA'
        if self.robot_pose is None or not self._is_fresh(self.robot_pose[1], now_ns, self.max_pose_age_ns):
            return 'WAITING_FOR_FRESH_ROBOT_POSE'
        fresh = [
            name for name, (_, received) in self.box_poses.items()
            if self._is_fresh(received, now_ns, self.max_pose_age_ns)
        ]
        if len(fresh) < self.box_count:
            return 'WAITING_FOR_FRESH_BOX_GROUND_TRUTH'
        return None

    def _get_camera_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        base = str(self.get_parameter('base_frame').value)
        camera = str(self.get_parameter('camera_frame').value)
        tf_msg: TransformStamped = self.tf_buffer.lookup_transform(
            base, camera, Time(), timeout=Duration(seconds=0.08)
        )
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        from perception_evaluation_geometry import quaternion_to_matrix
        base_r_camera = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        base_t_camera = np.array([t.x, t.y, t.z], dtype=np.float64)
        world_r_base, world_t_base = pose_to_rt(
            self.robot_pose[0].position, self.robot_pose[0].orientation
        )
        world_r_camera = world_r_base @ base_r_camera
        world_t_camera = world_t_base + world_r_base @ base_t_camera
        return world_r_camera, world_t_camera

    @staticmethod
    def _parse_stamp(payload: dict) -> Optional[Tuple[int, int]]:
        stamp = payload.get('source_stamp')
        if not isinstance(stamp, (list, tuple)) or len(stamp) != 2:
            return None
        try:
            return int(stamp[0]), int(stamp[1])
        except (TypeError, ValueError):
            return None

    def _build_box_models(self, fresh_poses: dict):
        models = {}
        for name, pose in fresh_poses.items():
            r, t = pose_to_rt(pose.position, pose.orientation)
            models[name] = (r, t, self.box_dims)
        return models

    def _publish_eval_overlay(self, payload: dict, visible_gt: list, matches: list,
                              unmatched_gt: list, unmatched_det: list,
                              stamp: Optional[Tuple[int, int]]) -> None:
        if self.overlay_cache is None:
            return
        if stamp is not None:
            image_stamp = (int(self.overlay_cache.header.stamp.sec), int(self.overlay_cache.header.stamp.nanosec))
            delta_ns = abs((image_stamp[0] - stamp[0]) * 1_000_000_000 + image_stamp[1] - stamp[1])
            if delta_ns > 200_000_000:
                return
        try:
            image = self.bridge.imgmsg_to_cv2(self.overlay_cache, desired_encoding='bgr8').copy()
        except Exception as exc:
            self.get_logger().warning(f'Cannot convert perception overlay for S2.4: {exc}')
            return
        dets = payload.get('detections', [])
        # GT projected rectangles: blue; matched detection label with IoU.
        for gt in visible_gt:
            x, y, w, h = [int(round(v)) for v in gt['bbox_xywh']]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 80, 0), 1)
            cv2.putText(image, gt['name'][-2:], (x, min(image.shape[0] - 2, y + h + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 0), 1, cv2.LINE_AA)
        match_by_det = {m['detection_index']: m for m in matches}
        for det_idx, det in enumerate(dets):
            bbox = det.get('bbox_xywh')
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, w, h = [int(round(v)) for v in bbox]
            if det_idx in match_by_det:
                m = match_by_det[det_idx]
                gt = visible_gt[m['ground_truth_index']]
                color = (0, 255, 0)
                label = f"TP {gt['name'][-2:]} IoU={m['iou']:.2f}"
            else:
                color = (0, 0, 255)
                label = 'FP'
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        for gi in unmatched_gt:
            gt = visible_gt[gi]
            x, y, w, h = [int(round(v)) for v in gt['bbox_xywh']]
            cv2.putText(image, 'FN', (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'S2.4 TP={len(matches)} FP={len(unmatched_det)} FN={len(unmatched_gt)}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        out.header = self.overlay_cache.header
        self.overlay_pub.publish(out)

    def perception_metrics_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._publish_waiting('INVALID_PERCEPTION_JSON', error=str(exc))
            return

        stamp = self._parse_stamp(payload)
        if stamp is not None and stamp == self.last_source_stamp:
            return
        now_ns = self.get_clock().now().nanoseconds
        max_rate_hz = float(self.get_parameter('evaluation_rate_hz').value)
        if max_rate_hz > 0.0 and self.last_eval_ns > 0:
            if now_ns - self.last_eval_ns < int(1e9 / max_rate_hz):
                return
        self.last_eval_ns = now_ns

        waiting = self._wait_for_inputs(now_ns)
        if waiting:
            self._publish_waiting(waiting, boxes_received=len(self.box_poses))
            return

        try:
            info = self.camera_info
            if len(info.p) != 12:
                self._publish_waiting('INVALID_RECTIFIED_CAMERA_INFO_P')
                return
            p_left = np.asarray(info.p, dtype=np.float64).reshape(3, 4)
            if p_left[0, 0] <= 0.0 or p_left[1, 1] <= 0.0:
                self._publish_waiting('INVALID_RECTIFIED_FOCAL_LENGTH')
                return
            r1 = np.asarray(self.rectification_report['rectification_rotation_left'], dtype=np.float64).reshape(3, 3)
            image_size = payload.get('input_image_size', [info.width, info.height])
            width, height = int(image_size[0]), int(image_size[1])
            if width != int(info.width) or height != int(info.height):
                self._publish_waiting('IMAGE_CAMERA_INFO_SIZE_MISMATCH', image_size=image_size,
                                      camera_info_size=[int(info.width), int(info.height)])
                return

            world_r_camera_raw, world_t_camera = self._get_camera_world_pose()
            robot_pose = self.robot_pose[0]
            world_r_base, world_t_base = pose_to_rt(robot_pose.position, robot_pose.orientation)
            fresh_poses = {
                name: pose for name, (pose, received_ns) in self.box_poses.items()
                if self._is_fresh(received_ns, now_ns, self.max_pose_age_ns)
            }
            box_models = self._build_box_models(fresh_poses)

            visible_gt = []
            min_area = float(self.get_parameter('minimum_projected_area_px').value)
            grid = int(self.get_parameter('visibility_grid_size').value)
            min_ray_hits = int(self.get_parameter('minimum_visible_ray_hits').value)
            for name, (box_r, box_t, dims) in box_models.items():
                bbox = project_obb_bbox(
                    box_r, box_t, dims, world_r_camera_raw, world_t_camera,
                    r1, p_left, width, height,
                )
                if bbox is None or bbox[2] * bbox[3] < min_area:
                    continue
                fraction, target_hits, total_hits = visible_fraction_for_box(
                    name, bbox, world_t_camera, world_r_camera_raw, r1, p_left,
                    box_models, grid_size=grid,
                )
                if target_hits < min_ray_hits:
                    continue
                visible_gt.append({
                    'name': name,
                    'bbox_xywh': [float(v) for v in bbox],
                    'visible_ray_fraction': float(fraction),
                    'visible_ray_hits': int(target_hits),
                    'occlusion_ray_hits_total': int(total_hits),
                    'projected_area_px': float(bbox[2] * bbox[3]),
                })

            detections = []
            for d in payload.get('detections', []):
                bbox = d.get('bbox_xywh')
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                    continue
                detections.append(d)

            matches, unmatched_det, unmatched_gt = greedy_iou_match(
                detections, visible_gt,
                iou_threshold=float(self.get_parameter('minimum_match_iou').value),
            )
            tp, fp, fn = len(matches), len(unmatched_det), len(unmatched_gt)
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 1.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

            xyz_records = []
            centroid_grid = int(self.get_parameter('centroid_grid_size').value)
            for match in matches:
                det = detections[match['detection_index']]
                gt = visible_gt[match['ground_truth_index']]
                if not bool(det.get('depth_valid', det.get('visible_surface_centroid_rectified_camera_m') is not None)):
                    continue
                predicted = det.get('visible_surface_centroid_rectified_camera_m')
                if predicted is None and det.get('visible_surface_centroid_camera_m') is not None:
                    # Backward compatibility with early S2.3 metrics, where XYZ
                    # was emitted in raw optical coordinates.
                    predicted = (r1 @ np.asarray(det['visible_surface_centroid_camera_m'], dtype=np.float64)).tolist()
                if not isinstance(predicted, (list, tuple)) or len(predicted) != 3:
                    continue
                expected = visible_surface_centroid_for_bbox(
                    gt['name'], det['bbox_xywh'], world_t_camera,
                    world_r_camera_raw, r1, p_left, box_models,
                    grid_size=centroid_grid,
                )
                if expected is None:
                    continue
                pred = np.asarray(predicted, dtype=np.float64)
                err = pred - expected
                norm = float(np.linalg.norm(err))
                if not np.all(np.isfinite(err)) or not math.isfinite(norm):
                    continue
                record = {
                    'box': gt['name'],
                    'detection_index': int(det.get('detection_index', match['detection_index'])),
                    'bbox_iou': float(match['iou']),
                    'predicted_surface_centroid_rectified_camera_m': pred.tolist(),
                    'ground_truth_surface_centroid_rectified_camera_m': expected.tolist(),
                    'error_xyz_m': err.tolist(),
                    'euclidean_error_m': norm,
                    'absolute_z_error_m': abs(float(err[2])),
                    'ground_truth_depth_m': float(expected[2]),
                }
                xyz_records.append(record)
                self.err_3d_history.append(norm)
                self.err_z_history.append(abs(float(err[2])))

            history_limit = max(100, int(self.get_parameter('max_metric_history').value))
            if len(self.err_3d_history) > history_limit:
                self.err_3d_history = self.err_3d_history[-history_limit:]
                self.err_z_history = self.err_z_history[-history_limit:]

            self.cum_tp += tp
            self.cum_fp += fp
            self.cum_fn += fn
            cum_precision = self.cum_tp / (self.cum_tp + self.cum_fp) if self.cum_tp + self.cum_fp else 1.0
            cum_recall = self.cum_tp / (self.cum_tp + self.cum_fn) if self.cum_tp + self.cum_fn else 1.0
            cum_f1 = 2 * cum_precision * cum_recall / (cum_precision + cum_recall) if cum_precision + cum_recall else 0.0

            errors = np.asarray([r['euclidean_error_m'] for r in xyz_records], dtype=np.float64)
            z_errors = np.asarray([r['absolute_z_error_m'] for r in xyz_records], dtype=np.float64)
            hist3 = np.asarray(self.err_3d_history, dtype=np.float64)
            histz = np.asarray(self.err_z_history, dtype=np.float64)
            self.frame_count += 1
            report = {
                'status': 'OK',
                'evaluation_enabled': True,
                'frame_count': self.frame_count,
                'source_stamp': list(stamp) if stamp is not None else None,
                'ground_truth_usage': 'evaluation_only_not_perception_or_policy_input',
                'matching_method': 'greedy_one_to_one_projected_bbox_iou',
                'minimum_match_iou': float(self.get_parameter('minimum_match_iou').value),
                'occlusion_model': 'ray_visibility_against_28_ground_truth_boxes_only',
                'ground_truth_boxes_received': len(fresh_poses),
                'ground_truth_boxes_projected': len(box_models),
                'ground_truth_boxes_visible': len(visible_gt),
                'detections': len(detections),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision_frame': float(precision),
                'recall_frame': float(recall),
                'f1_frame': float(f1),
                'cumulative': {
                    'tp': self.cum_tp, 'fp': self.cum_fp, 'fn': self.cum_fn,
                    'precision': float(cum_precision), 'recall': float(cum_recall),
                    'f1': float(cum_f1),
                },
                'position_3d_samples_this_frame': len(xyz_records),
                'position_3d_mean_error_m': float(np.mean(errors)) if errors.size else None,
                'position_3d_rmse_m': float(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
                'position_3d_median_error_m': float(np.median(errors)) if errors.size else None,
                'position_3d_p90_error_m': float(np.percentile(errors, 90)) if errors.size else None,
                'position_z_mae_m': float(np.mean(z_errors)) if z_errors.size else None,
                'position_3d_cumulative_samples': int(hist3.size),
                'position_3d_cumulative_mean_error_m': float(np.mean(hist3)) if hist3.size else None,
                'position_3d_cumulative_rmse_m': float(np.sqrt(np.mean(hist3 ** 2))) if hist3.size else None,
                'position_3d_cumulative_p90_m': float(np.percentile(hist3, 90)) if hist3.size else None,
                'position_z_cumulative_mae_m': float(np.mean(histz)) if histz.size else None,
                'rectified_baseline_m': self.rectification_report.get('rectified_baseline_m'),
                'rectified_focal_px': float(p_left[0, 0]),
                'matches': [
                    {
                        'box': visible_gt[m['ground_truth_index']]['name'],
                        'detection_index': int(detections[m['detection_index']].get('detection_index', m['detection_index'])),
                        'bbox_iou': float(m['iou']),
                        'depth_valid': bool(detections[m['detection_index']].get('depth_valid', False)),
                    }
                    for m in matches
                ],
                'position_errors': xyz_records,
            }
            out = String()
            out.data = json.dumps(report, separators=(',', ':'), allow_nan=False)
            self.metrics_pub.publish(out)
            self.ready_pub.publish(Bool(data=True))
            self.last_wait_status = ''
            self.last_source_stamp = stamp
            self._publish_eval_overlay(payload, visible_gt, matches, unmatched_gt, unmatched_det, stamp)

            if self.frame_count == 1 or self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'S2.4 frame={self.frame_count} visible_gt={len(visible_gt)} '
                    f'dets={len(detections)} TP/FP/FN={tp}/{fp}/{fn} '
                    f'3D_samples={len(xyz_records)}'
                )
        except TransformException as exc:
            self._publish_waiting('WAITING_FOR_CAMERA_TF', error=str(exc))
        except Exception as exc:
            self.get_logger().error(f'S2.4 evaluation failed: {type(exc).__name__}: {exc}')
            self._publish_waiting('EVALUATION_ERROR', error=str(exc))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrangeBoxPerceptionEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
