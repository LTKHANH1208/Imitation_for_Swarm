#!/usr/bin/env python3
"""Track orange RGB-D detections over time in the robot's odometry frame.

Inputs are S2.3 perception PoseArray (in base_footprint) and differential-drive
Odometry. Gazebo model poses are deliberately NOT subscribed. Track positions
refer to the detector's visible-surface centroid, not a reconstructed full-box
geometric centre.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import bisect
import json
import math
from typing import Deque, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException

from .geometry import transform_point, quaternion_slerp
from .tracker_core import MultiObjectTracker


@dataclass
class OdomSample:
    stamp_ns: int
    frame_id: str
    child_frame_id: str
    translation: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]


class OrangeBoxTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__('orange_box_tracker_node')

        self.declare_parameter('detections_topic', '/robot1/perception/orange_boxes/centers_base')
        self.declare_parameter('odom_topic', '/robot1/diff_drive_controller/odom')
        self.declare_parameter('tracks_topic', '/robot1/perception/orange_boxes/tracks')
        self.declare_parameter('tracks_poses_topic', '/robot1/perception/orange_boxes/tracks/poses')
        self.declare_parameter('markers_topic', '/robot1/perception/orange_boxes/tracks/markers')
        self.declare_parameter('metrics_topic', '/robot1/perception/orange_boxes/tracking/metrics')
        self.declare_parameter('ready_topic', '/robot1/perception/orange_boxes/tracking/ready')
        self.declare_parameter('max_odom_sync_sec', 0.15)
        self.declare_parameter('odom_history_size', 300)
        self.declare_parameter('gate_distance_m', 0.35)
        self.declare_parameter('max_age_sec', 1.25)
        self.declare_parameter('min_hits', 2)
        self.declare_parameter('measurement_sigma_m', 0.045)
        self.declare_parameter('acceleration_noise', 0.8)
        self.declare_parameter('marker_lifetime_sec', 0.5)
        self.declare_parameter('show_tentative_tracks', True)
        self.declare_parameter('tf_lookup_timeout_sec', 0.03)

        self.tracker = MultiObjectTracker(
            gate_distance_m=float(self.get_parameter('gate_distance_m').value),
            max_age_sec=float(self.get_parameter('max_age_sec').value),
            min_hits=int(self.get_parameter('min_hits').value),
            measurement_sigma_m=float(self.get_parameter('measurement_sigma_m').value),
            acceleration_noise=float(self.get_parameter('acceleration_noise').value),
        )
        self.odom_history: Deque[OdomSample] = deque(
            maxlen=int(self.get_parameter('odom_history_size').value)
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_frame_stamp_ns = 0
        self._frame_count = 0
        self._dropped_frame_count = 0
        self._last_tf_warning = ''

        self.tracks_pub = self.create_publisher(
            String, str(self.get_parameter('tracks_topic').value), 10
        )
        self.poses_pub = self.create_publisher(
            PoseArray, str(self.get_parameter('tracks_poses_topic').value), 10
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

        self.odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            self.on_odom,
            50,
        )
        self.detections_sub = self.create_subscription(
            PoseArray,
            str(self.get_parameter('detections_topic').value),
            self.on_detections,
            10,
        )
        self.get_logger().info(
            'Orange-box tracker started; inputs are base-frame detections + wheel odometry. '
            'No Gazebo ground-truth pose topics are used.'
        )

    @staticmethod
    def stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        sample = OdomSample(
            stamp_ns=self.stamp_ns(msg.header.stamp),
            frame_id=msg.header.frame_id,
            child_frame_id=msg.child_frame_id,
            translation=(float(p.x), float(p.y), float(p.z)),
            quaternion=(float(q.x), float(q.y), float(q.z), float(q.w)),
        )
        if self.odom_history and sample.stamp_ns < self.odom_history[-1].stamp_ns:
            # Sim-time reset or bag rewind: clear stale history to avoid mixing epochs.
            self.odom_history.clear()
            self.tracker = MultiObjectTracker(
                gate_distance_m=float(self.get_parameter('gate_distance_m').value),
                max_age_sec=float(self.get_parameter('max_age_sec').value),
                min_hits=int(self.get_parameter('min_hits').value),
                measurement_sigma_m=float(self.get_parameter('measurement_sigma_m').value),
                acceleration_noise=float(self.get_parameter('acceleration_noise').value),
            )
            self.get_logger().warning('Odometry time moved backwards; tracker state reset.')
        self.odom_history.append(sample)

    def _odom_at(self, stamp_ns: int) -> Optional[OdomSample]:
        if not self.odom_history:
            return None
        samples = list(self.odom_history)
        stamps = [s.stamp_ns for s in samples]
        idx = bisect.bisect_left(stamps, stamp_ns)
        max_dt_ns = int(float(self.get_parameter('max_odom_sync_sec').value) * 1e9)

        if idx == 0:
            nearest = samples[0]
            return nearest if abs(nearest.stamp_ns - stamp_ns) <= max_dt_ns else None
        if idx >= len(samples):
            nearest = samples[-1]
            return nearest if abs(nearest.stamp_ns - stamp_ns) <= max_dt_ns else None

        before, after = samples[idx - 1], samples[idx]
        if before.frame_id != after.frame_id or before.child_frame_id != after.child_frame_id:
            nearest = min((before, after), key=lambda s: abs(s.stamp_ns - stamp_ns))
            return nearest if abs(nearest.stamp_ns - stamp_ns) <= max_dt_ns else None

        gap = after.stamp_ns - before.stamp_ns
        if gap <= 0:
            nearest = min((before, after), key=lambda s: abs(s.stamp_ns - stamp_ns))
            return nearest if abs(nearest.stamp_ns - stamp_ns) <= max_dt_ns else None
        if max(abs(stamp_ns - before.stamp_ns), abs(after.stamp_ns - stamp_ns)) > max_dt_ns:
            nearest = min((before, after), key=lambda s: abs(s.stamp_ns - stamp_ns))
            return nearest if abs(nearest.stamp_ns - stamp_ns) <= max_dt_ns else None

        alpha = (stamp_ns - before.stamp_ns) / float(gap)
        trans = tuple((1.0-alpha)*a + alpha*b for a, b in zip(before.translation, after.translation))
        quat = quaternion_slerp(before.quaternion, after.quaternion, alpha)
        return OdomSample(
            stamp_ns=stamp_ns,
            frame_id=before.frame_id,
            child_frame_id=before.child_frame_id,
            translation=trans,
            quaternion=quat,
        )

    def _point_to_odom(self, xyz: Sequence[float], detection_frame: str,
                       stamp, odom: OdomSample) -> Tuple[Optional[Tuple[float, float, float]], str]:
        point_in_child = tuple(float(v) for v in xyz)
        if detection_frame != odom.child_frame_id:
            try:
                tf_msg: TransformStamped = self.tf_buffer.lookup_transform(
                    odom.child_frame_id,
                    detection_frame,
                    Time.from_msg(stamp),
                    timeout=Duration(seconds=float(self.get_parameter('tf_lookup_timeout_sec').value)),
                )
                t = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                point_in_child = transform_point(
                    (t.x, t.y, t.z), (q.x, q.y, q.z, q.w), point_in_child
                )
            except TransformException as exc:
                return None, f'FRAME_TRANSFORM_UNAVAILABLE:{detection_frame}->{odom.child_frame_id}:{exc}'
        point_odom = transform_point(odom.translation, odom.quaternion, point_in_child)
        return point_odom, 'OK'

    def _publish_empty(self, header, status: str, extra=None) -> None:
        pose_array = PoseArray()
        pose_array.header.stamp = header.stamp
        pose_array.header.frame_id = 'odom'
        self.poses_pub.publish(pose_array)
        self._publish_markers([], header.stamp, 'odom')
        report = {
            'status': status,
            'frame_count': self._frame_count,
            'dropped_frame_count': self._dropped_frame_count,
            'active_tracks': len(self.tracker.tracks),
            'tracks': [],
        }
        if extra:
            report.update(extra)
        self._publish_json(self.tracks_pub, report)
        self._publish_json(self.metrics_pub, report)
        self.ready_pub.publish(Bool(data=False))

    def on_detections(self, msg: PoseArray) -> None:
        stamp_ns = self.stamp_ns(msg.header.stamp)
        if stamp_ns <= self._last_frame_stamp_ns and self._last_frame_stamp_ns > 0:
            self._dropped_frame_count += 1
            return
        self._last_frame_stamp_ns = stamp_ns
        odom = self._odom_at(stamp_ns)
        if odom is None:
            self._dropped_frame_count += 1
            self._publish_empty(msg.header, 'WAITING_FOR_SYNCHRONIZED_ODOMETRY', {
                'detection_frame': msg.header.frame_id,
                'odom_history_samples': len(self.odom_history),
                'max_odom_sync_sec': float(self.get_parameter('max_odom_sync_sec').value),
            })
            return
        if not msg.header.frame_id:
            self._dropped_frame_count += 1
            self._publish_empty(msg.header, 'DETECTION_FRAME_ID_EMPTY')
            return

        detections_odom = []
        frame_error = None
        for pose in msg.poses:
            p = pose.position
            point, status = self._point_to_odom(
                (p.x, p.y, p.z), msg.header.frame_id, msg.header.stamp, odom
            )
            if point is None:
                frame_error = status
                break
            detections_odom.append(point)
        if frame_error:
            self._dropped_frame_count += 1
            if frame_error != self._last_tf_warning:
                self.get_logger().warning(frame_error)
                self._last_tf_warning = frame_error
            self._publish_empty(msg.header, 'DETECTION_TO_ODOM_TRANSFORM_FAILED', {
                'error': frame_error,
                'detection_frame': msg.header.frame_id,
                'odom_child_frame': odom.child_frame_id,
                'odom_frame': odom.frame_id,
            })
            return

        update = self.tracker.update(detections_odom, stamp_ns)
        self._frame_count += 1
        pose_array = PoseArray()
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = odom.frame_id
        show_tentative = bool(self.get_parameter('show_tentative_tracks').value)

        published_tracks = []
        for track in sorted(self.tracker.tracks.values(), key=lambda t: t.track_id):
            if track.hits < int(self.get_parameter('min_hits').value) and not show_tentative:
                continue
            pose = Pose()
            pose.position.x = float(track.state[0])
            pose.position.y = float(track.state[1])
            pose.position.z = float(track.z)
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
            published_tracks.append(track.as_dict(stamp_ns))
        self.poses_pub.publish(pose_array)
        self._publish_markers(published_tracks, msg.header.stamp, odom.frame_id)

        report = {
            'status': 'TRACKING' if detections_odom else 'NO_VALID_DEPTH_DETECTIONS',
            'frame_count': self._frame_count,
            'dropped_frame_count': self._dropped_frame_count,
            'source_stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec],
            'detection_frame': msg.header.frame_id,
            'odom_frame': odom.frame_id,
            'odom_child_frame': odom.child_frame_id,
            'odometry_sync_error_sec': abs(odom.stamp_ns - stamp_ns) * 1e-9,
            'coordinate_semantics': 'tracked visible-surface centroids transformed into odom; not full geometric box centres',
            **update,
            'tracks': published_tracks,
        }
        self._publish_json(self.tracks_pub, report)
        metrics = {k: v for k, v in report.items() if k != 'tracks'}
        metrics['tracks_summary'] = [
            {'id': t['id'], 'status': t['status'], 'hits': t['hits'],
             'missed_frames': t['missed_frames'], 'time_since_seen_sec': t['time_since_seen_sec']}
            for t in published_tracks
        ]
        self._publish_json(self.metrics_pub, metrics)
        self.ready_pub.publish(Bool(data=True))

    def _publish_markers(self, tracks, stamp, frame_id: str) -> None:
        array = MarkerArray()
        delete = Marker()
        delete.header.stamp = stamp
        delete.header.frame_id = frame_id
        delete.action = Marker.DELETEALL
        array.markers.append(delete)
        lifetime = Duration(seconds=float(self.get_parameter('marker_lifetime_sec').value)).to_msg()
        for track in tracks:
            tid = int(track['id'])
            x, y, z = track['position_odom_m']
            status = str(track['status'])
            color = (0.1, 0.9, 0.2, 0.95) if status == 'confirmed' else (1.0, 0.65, 0.1, 0.55)

            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = frame_id
            sphere.ns = 'orange_box_tracks'
            sphere.id = tid * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(x)
            sphere.pose.position.y = float(y)
            sphere.pose.position.z = float(z)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.075
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = color
            sphere.lifetime = lifetime
            array.markers.append(sphere)

            label = Marker()
            label.header = sphere.header
            label.ns = 'orange_box_track_ids'
            label.id = tid * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose = sphere.pose
            label.pose.position.z += 0.07
            label.scale.z = 0.10
            label.color.r, label.color.g, label.color.b, label.color.a = color
            label.text = f'ID {tid} ({status})'
            label.lifetime = lifetime
            array.markers.append(label)
        self.markers_pub.publish(array)

    @staticmethod
    def _publish_json(publisher, payload) -> None:
        msg = String()
        msg.data = json.dumps(payload, separators=(',', ':'), allow_nan=False)
        publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrangeBoxTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
