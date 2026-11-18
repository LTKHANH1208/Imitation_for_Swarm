#!/usr/bin/env python3
"""Ground-truth evaluator for the 28-box Gazebo collection task.

Gazebo model poses are subscribed from per-box PosePublisher topics. Outputs
are evaluation-only and must not be connected to the imitation policy input.
"""
from __future__ import annotations

import json
from statistics import mean, median
from typing import Dict, Tuple

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String, UInt32

from box_evaluator_geometry import classify_box_center


class BoxCollectionEvaluator(Node):
    def __init__(self) -> None:
        super().__init__('box_collection_task_evaluator')

        self.declare_parameter('box_count', 28)
        self.declare_parameter('pose_topic_prefix', '/model')
        self.declare_parameter('pose_topic_suffix', '/pose')
        self.declare_parameter('carpet_half_extent_m', 0.50)
        self.declare_parameter('carpet_inner_half_extent_m', 0.25)
        self.declare_parameter('pit_half_extent_m', 0.25)
        self.declare_parameter('pit_center_x_m', 0.0)
        self.declare_parameter('pit_center_y_m', 0.0)
        self.declare_parameter('pit_fallen_z_threshold_m', -0.02)
        self.declare_parameter('publish_rate_hz', 5.0)

        self.box_count = int(self.get_parameter('box_count').value)
        self.topic_prefix = str(self.get_parameter('pose_topic_prefix').value).rstrip('/')
        self.topic_suffix = str(self.get_parameter('pose_topic_suffix').value)
        self.geometry = {
            'carpet_half_extent_m': float(self.get_parameter('carpet_half_extent_m').value),
            'carpet_inner_half_extent_m': float(self.get_parameter('carpet_inner_half_extent_m').value),
            'pit_half_extent_m': float(self.get_parameter('pit_half_extent_m').value),
            'pit_center_x_m': float(self.get_parameter('pit_center_x_m').value),
            'pit_center_y_m': float(self.get_parameter('pit_center_y_m').value),
            'pit_fallen_z_threshold_m': float(self.get_parameter('pit_fallen_z_threshold_m').value),
        }

        self.latest_pose: Dict[str, Tuple[float, float, float]] = {}
        self.delivered: Dict[str, str] = {}
        self.started_at_ns = None
        self.last_clock_ns = None
        self.delivery_times_sec = []

        self.ready_pub = self.create_publisher(Bool, '/box_collection/evaluator/ready', 10)
        self.count_pub = self.create_publisher(UInt32, '/box_collection/evaluator/delivered_count', 10)
        self.remaining_pub = self.create_publisher(UInt32, '/box_collection/evaluator/remaining_count', 10)
        self.progress_pub = self.create_publisher(Float32, '/box_collection/evaluator/progress', 10)
        self.metrics_pub = self.create_publisher(String, '/box_collection/evaluator/metrics', 10)
        self.events_pub = self.create_publisher(String, '/box_collection/evaluator/events', 10)

        self.box_subscriptions = []
        for index in range(1, self.box_count + 1):
            box_name = f'orange_box_{index:02d}'
            topic = f'{self.topic_prefix}/{box_name}{self.topic_suffix}'
            self.box_subscriptions.append(
                self.create_subscription(
                    Pose,
                    topic,
                    lambda msg, name=box_name: self.pose_callback(name, msg),
                    10,
                )
            )
            self.get_logger().info(f'Ground-truth subscription: {topic}')

        rate = max(0.2, float(self.get_parameter('publish_rate_hz').value))
        self.timer = self.create_timer(1.0 / rate, self.publish_metrics)
        self.get_logger().info(
            f'Evaluator configured for {self.box_count} boxes. '
            'Gazebo pose ground-truth is evaluation-only, not policy input.'
        )

    def pose_callback(self, box_name: str, msg: Pose) -> None:
        point = (float(msg.position.x), float(msg.position.y), float(msg.position.z))
        self.latest_pose[box_name] = point

        # Once delivered, a box stays counted for task-progress evaluation.
        if box_name in self.delivered:
            return

        state = classify_box_center(*point, **self.geometry)
        if state != 'ACTIVE':
            self.delivered[box_name] = state
            now_ns = self.get_clock().now().nanoseconds
            self.delivery_times_sec.append(now_ns / 1e9)
            event = {
                'event': 'box_delivered',
                'box_name': box_name,
                'delivery_mode': state,
                'position_m': {'x': point[0], 'y': point[1], 'z': point[2]},
                'sim_time_sec': now_ns / 1e9,
            }
            event_msg = String()
            event_msg.data = json.dumps(event, separators=(',', ':'))
            self.events_pub.publish(event_msg)
            self.get_logger().info(f'{box_name} delivered: {state}')

    def publish_metrics(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self.last_clock_ns is not None and now_ns < self.last_clock_ns:
            # Gazebo simulation reset: begin a fresh evaluation epoch.
            self.latest_pose.clear()
            self.delivered.clear()
            self.delivery_times_sec.clear()
            self.started_at_ns = None
            self.get_logger().warn('Simulation clock moved backwards; evaluator state reset.')
        self.last_clock_ns = now_ns

        ready = len(self.latest_pose) == self.box_count
        if ready and self.started_at_ns is None:
            self.started_at_ns = now_ns

        delivered_count = len(self.delivered)
        remaining_count = max(0, self.box_count - delivered_count)
        progress = delivered_count / float(self.box_count) if self.box_count else 0.0
        elapsed = (
            max(0.0, (now_ns - self.started_at_ns) / 1e9)
            if self.started_at_ns is not None else 0.0
        )
        status = 'WAITING_FOR_POSES' if not ready else ('COMPLETE' if remaining_count == 0 else 'RUNNING')

        ready_msg = Bool(); ready_msg.data = ready
        count_msg = UInt32(); count_msg.data = delivered_count
        remaining_msg = UInt32(); remaining_msg.data = remaining_count
        progress_msg = Float32(); progress_msg.data = float(progress)
        self.ready_pub.publish(ready_msg)
        self.count_pub.publish(count_msg)
        self.remaining_pub.publish(remaining_msg)
        self.progress_pub.publish(progress_msg)

        box_states = {}
        for index in range(1, self.box_count + 1):
            name = f'orange_box_{index:02d}'
            if name in self.delivered:
                state = self.delivered[name]
            elif name in self.latest_pose:
                state = 'ACTIVE'
            else:
                state = 'NO_POSE'
            box_states[name] = state

        now_sec = now_ns / 1e9
        delivery_times = list(self.delivery_times_sec)
        intervals = [b - a for a, b in zip(delivery_times[:-1], delivery_times[1:])]
        first_delivery_after_start = (
            max(0.0, delivery_times[0] - self.started_at_ns / 1e9)
            if delivery_times and self.started_at_ns is not None else None
        )
        rolling_count = sum(1 for stamp in delivery_times if now_sec - stamp <= 60.0)
        rolling_window_sec = min(60.0, elapsed)
        rolling_rate = (rolling_count * 60.0 / rolling_window_sec) if rolling_window_sec > 1e-9 else 0.0

        metrics = {
            'status': status,
            'expected_boxes': self.box_count,
            'received_box_poses': len(self.latest_pose),
            'delivered_boxes': delivered_count,
            'remaining_boxes': remaining_count,
            'progress_fraction': round(progress, 5),
            'progress_percent': round(progress * 100.0, 2),
            'elapsed_sim_seconds': round(elapsed, 3),
            'throughput_boxes_per_sim_minute': round(
                delivered_count / (elapsed / 60.0), 4
            ) if elapsed > 1e-9 else 0.0,
            'rolling_60s_delivered_boxes': rolling_count,
            'rolling_60s_throughput_boxes_per_minute': round(rolling_rate, 4),
            'first_delivery_after_start_sec': round(first_delivery_after_start, 3) if first_delivery_after_start is not None else None,
            'mean_inter_delivery_interval_sec': round(mean(intervals), 3) if intervals else None,
            'median_inter_delivery_interval_sec': round(median(intervals), 3) if intervals else None,
            'delivery_timestamps_retained': len(delivery_times),
            'delivery_modes': {
                'carpet': sum(1 for mode in self.delivered.values() if mode == 'DELIVERED_CARPET'),
                'pit': sum(1 for mode in self.delivered.values() if mode == 'DELIVERED_PIT'),
            },
            'box_states': box_states,
            'ground_truth_usage': 'evaluation_only_not_policy_input',
        }
        metrics_msg = String(); metrics_msg.data = json.dumps(metrics, separators=(',', ':'))
        self.metrics_pub.publish(metrics_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoxCollectionEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
