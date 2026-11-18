#!/usr/bin/env python3
"""Aggregate Gazebo contact sensor streams into debounced collision metrics.

The node subscribes to per-box and perimeter-wall Contact sensors. It is an
EVALUATION-ONLY ground-truth consumer; its outputs must never feed the policy.
"""
from __future__ import annotations
import json
from typing import Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, UInt32
from ros_gz_interfaces.msg import Contacts

from collision_geometry import entity_name, classify_pair, is_expected_ground_contact


class CollisionMonitor(Node):
    def __init__(self) -> None:
        super().__init__('box_collection_collision_monitor')
        self.declare_parameter('box_count', 28)
        self.declare_parameter('wall_names', ['wall_north', 'wall_south', 'wall_east', 'wall_west'])
        self.declare_parameter('event_gap_sec', 0.50)
        self.declare_parameter('active_timeout_sec', 0.20)
        self.declare_parameter('publish_rate_hz', 5.0)

        n = int(self.get_parameter('box_count').value)
        walls = list(self.get_parameter('wall_names').value)
        self.event_gap = float(self.get_parameter('event_gap_sec').value)
        self.active_timeout = float(self.get_parameter('active_timeout_sec').value)
        self.sensor_topics: List[str] = [
            f'/box_collection/contacts/boxes/orange_box_{i:02d}' for i in range(1, n + 1)
        ] + [f'/box_collection/contacts/walls/{name}' for name in walls]

        self.box_subscriptions = [
            self.create_subscription(Contacts, topic,
                lambda msg, src=topic: self.contact_callback(msg, src), 20)
            for topic in self.sensor_topics
        ]
        self.ready_pub = self.create_publisher(Bool, '/box_collection/collision/ready', 10)
        self.event_count_pub = self.create_publisher(UInt32, '/box_collection/collision/event_count', 10)
        self.metrics_pub = self.create_publisher(String, '/box_collection/collision/metrics', 10)
        self.events_pub = self.create_publisher(String, '/box_collection/collision/events', 20)

        self.last_seen: Dict[str, float] = {}
        self.pair_names: Dict[str, tuple] = {}
        self.pair_types: Dict[str, str] = {}
        self.category_counts = {
            'robot_box': 0, 'robot_wall': 0, 'robot_robot': 0,
            'box_box': 0, 'box_wall': 0, 'other': 0,
        }
        self.total_events = 0
        self.total_contact_samples = 0
        self.total_ignored_ground_samples = 0
        self.last_sim_time_sec = None
        self.sensor_messages = {topic: 0 for topic in self.sensor_topics}
        rate = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.timer = self.create_timer(1.0 / rate, self.publish_metrics)
        self.get_logger().info(
            f'Collision monitor subscribed to {len(self.sensor_topics)} Gazebo contact streams.'
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def contact_callback(self, msg: Contacts, source_topic: str) -> None:
        now = self._now_sec()
        self.sensor_messages[source_topic] = self.sensor_messages.get(source_topic, 0) + 1
        for contact in msg.contacts:
            name_a = entity_name(contact.collision1)
            name_b = entity_name(contact.collision2)
            if not name_a or not name_b:
                continue
            if is_expected_ground_contact(name_a, name_b):
                self.total_ignored_ground_samples += 1
                continue
            self.total_contact_samples += 1
            key = ' | '.join(sorted((name_a, name_b)))
            category = classify_pair(name_a, name_b)
            previous = self.last_seen.get(key)
            if previous is None or now - previous > self.event_gap:
                self.total_events += 1
                self.category_counts[category] += 1
                event = {
                    'event': 'collision_episode_started',
                    'category': category,
                    'collision1': name_a,
                    'collision2': name_b,
                    'sim_time_sec': round(now, 4),
                    'source_topic': source_topic,
                    'episode_index': self.total_events,
                }
                out = String(); out.data = json.dumps(event, separators=(',', ':'))
                self.events_pub.publish(out)
            self.last_seen[key] = now
            self.pair_names[key] = (name_a, name_b)
            self.pair_types[key] = category

    def publish_metrics(self) -> None:
        now = self._now_sec()
        if self.last_sim_time_sec is not None and now < self.last_sim_time_sec:
            self.last_seen.clear()
            self.pair_names.clear()
            self.pair_types.clear()
            self.category_counts = {key: 0 for key in self.category_counts}
            self.total_events = 0
            self.total_contact_samples = 0
            self.total_ignored_ground_samples = 0
            self.get_logger().warn('Simulation clock moved backwards; collision metrics reset.')
        self.last_sim_time_sec = now
        publishers = {
            topic: len(self.get_publishers_info_by_topic(topic)) > 0
            for topic in self.sensor_topics
        }
        connected_count = sum(publishers.values())
        active = []
        for key, last_seen in list(self.last_seen.items()):
            age = now - last_seen
            if age <= self.active_timeout:
                a, b = self.pair_names[key]
                active.append({
                    'category': self.pair_types[key],
                    'collision1': a, 'collision2': b,
                    'last_seen_age_sec': round(max(0.0, age), 4),
                })
        ready = connected_count == len(self.sensor_topics)
        ready_msg = Bool(); ready_msg.data = ready; self.ready_pub.publish(ready_msg)
        count_msg = UInt32(); count_msg.data = self.total_events; self.event_count_pub.publish(count_msg)
        metrics = {
            'status': 'READY' if ready else 'WAITING_FOR_CONTACT_BRIDGES',
            'expected_contact_streams': len(self.sensor_topics),
            'connected_contact_streams': connected_count,
            'received_contact_messages': sum(self.sensor_messages.values()),
            'collision_episodes_total': self.total_events,
            'collision_episodes_by_type': self.category_counts,
            'active_contact_pairs': active,
            'active_contact_pair_count': len(active),
            'contact_samples_total': self.total_contact_samples,
            'ignored_ground_contact_samples': self.total_ignored_ground_samples,
            'ground_truth_usage': 'evaluation_only_not_policy_input',
        }
        out = String(); out.data = json.dumps(metrics, separators=(',', ':'))
        self.metrics_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CollisionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
