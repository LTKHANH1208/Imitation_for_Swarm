#!/usr/bin/env python3
"""Evaluate-only stuck/progress-stall monitor using Gazebo model pose.

Wheel odometry alone can report motion while wheels slip against an obstacle.
Therefore this node uses a Gazebo PosePublisher stream for actual body motion.
That ground-truth stream must never be connected to the policy input.
"""
from __future__ import annotations
import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, TwistStamped
from std_msgs.msg import Bool, String, UInt32


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class DeadlockMonitor(Node):
    def __init__(self) -> None:
        super().__init__('box_collection_deadlock_monitor')
        self.declare_parameter('cmd_topic', '/robot1/diff_drive_controller/cmd_vel')
        self.declare_parameter('pose_topic', '/model/box_pusher/pose')
        self.declare_parameter('delivered_count_topic', '/box_collection/evaluator/delivered_count')
        self.declare_parameter('cmd_stale_timeout_sec', 0.75)
        self.declare_parameter('pose_stale_timeout_sec', 0.50)
        self.declare_parameter('linear_command_threshold_mps', 0.04)
        self.declare_parameter('angular_command_threshold_rps', 0.10)
        self.declare_parameter('linear_motion_threshold_mps', 0.015)
        self.declare_parameter('angular_motion_threshold_rps', 0.05)
        self.declare_parameter('stuck_duration_threshold_sec', 3.0)
        self.declare_parameter('task_stall_threshold_sec', 20.0)
        self.declare_parameter('publish_rate_hz', 5.0)

        cmd_topic = str(self.get_parameter('cmd_topic').value)
        pose_topic = str(self.get_parameter('pose_topic').value)
        delivered_topic = str(self.get_parameter('delivered_count_topic').value)
        self.cmd_stale = float(self.get_parameter('cmd_stale_timeout_sec').value)
        self.pose_stale = float(self.get_parameter('pose_stale_timeout_sec').value)
        self.lin_cmd_thr = float(self.get_parameter('linear_command_threshold_mps').value)
        self.ang_cmd_thr = float(self.get_parameter('angular_command_threshold_rps').value)
        self.lin_motion_thr = float(self.get_parameter('linear_motion_threshold_mps').value)
        self.ang_motion_thr = float(self.get_parameter('angular_motion_threshold_rps').value)
        self.stuck_thr = float(self.get_parameter('stuck_duration_threshold_sec').value)
        self.task_stall_thr = float(self.get_parameter('task_stall_threshold_sec').value)

        self.create_subscription(TwistStamped, cmd_topic, self.cmd_callback, 20)
        self.create_subscription(Pose, pose_topic, self.pose_callback, 20)
        self.create_subscription(UInt32, delivered_topic, self.progress_callback, 10)
        self.stuck_pub = self.create_publisher(Bool, '/box_collection/deadlock/robot_stuck', 10)
        self.stall_pub = self.create_publisher(Bool, '/box_collection/deadlock/task_stalled', 10)
        self.metrics_pub = self.create_publisher(String, '/box_collection/deadlock/metrics', 10)
        self.events_pub = self.create_publisher(String, '/box_collection/deadlock/events', 10)

        self.last_cmd_ns = None
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.last_pose = None  # (x, y, yaw, receive_time_ns)
        self.last_pose_ns = None
        self.measured_linear = 0.0
        self.measured_angular = 0.0
        self.stuck_since_ns = None
        self.stuck_latched = False
        self.deadlock_events = 0
        self.last_delivered_count = None
        self.last_progress_ns = None
        self.task_stalled_latched = False
        rate = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.timer = self.create_timer(1.0 / rate, self.evaluate)
        self.get_logger().info(
            f'Using command {cmd_topic} and Gazebo ground-truth pose {pose_topic}; '
            'ground-truth is evaluation-only.'
        )

    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def cmd_callback(self, msg: TwistStamped) -> None:
        self.last_cmd_ns = self.now_ns()
        self.cmd_linear = float(msg.twist.linear.x)
        self.cmd_angular = float(msg.twist.angular.z)

    def pose_callback(self, msg: Pose) -> None:
        now = self.now_ns()
        x, y = float(msg.position.x), float(msg.position.y)
        yaw = yaw_from_quaternion(msg.orientation)
        current = (x, y, yaw, now)
        if self.last_pose is not None:
            px, py, pyaw, previous_ns = self.last_pose
            dt = (now - previous_ns) / 1e9
            if 1e-4 < dt <= 1.0:
                self.measured_linear = math.hypot(x - px, y - py) / dt
                self.measured_angular = abs(wrap_angle(yaw - pyaw)) / dt
        self.last_pose = current
        self.last_pose_ns = now

    def progress_callback(self, msg: UInt32) -> None:
        now = self.now_ns()
        count = int(msg.data)
        if self.last_delivered_count is None or count != self.last_delivered_count:
            self.last_progress_ns = now
        self.last_delivered_count = count

    def publish_event(self, event_name: str, payload: dict) -> None:
        data = {'event': event_name, 'sim_time_sec': self.now_ns() / 1e9, **payload}
        out = String(); out.data = json.dumps(data, separators=(',', ':'))
        self.events_pub.publish(out)

    def evaluate(self) -> None:
        now = self.now_ns()
        cmd_fresh = self.last_cmd_ns is not None and (now - self.last_cmd_ns) / 1e9 <= self.cmd_stale
        pose_fresh = self.last_pose_ns is not None and (now - self.last_pose_ns) / 1e9 <= self.pose_stale
        command_active = cmd_fresh and (
            abs(self.cmd_linear) >= self.lin_cmd_thr or abs(self.cmd_angular) >= self.ang_cmd_thr
        )
        moving = pose_fresh and (
            self.measured_linear >= self.lin_motion_thr or self.measured_angular >= self.ang_motion_thr
        )
        qualifies = command_active and pose_fresh and not moving

        if qualifies:
            if self.stuck_since_ns is None:
                self.stuck_since_ns = now
            duration = max(0.0, (now - self.stuck_since_ns) / 1e9)
            if duration >= self.stuck_thr and not self.stuck_latched:
                self.stuck_latched = True
                self.deadlock_events += 1
                self.publish_event('robot_stuck_started', {
                    'duration_threshold_sec': self.stuck_thr,
                    'command_linear_mps': self.cmd_linear,
                    'command_angular_rps': self.cmd_angular,
                    'measured_ground_truth_linear_mps': self.measured_linear,
                    'measured_ground_truth_angular_rps': self.measured_angular,
                })
                self.get_logger().warn('Robot appears stuck: active command but Gazebo body pose is stationary.')
        else:
            if self.stuck_latched:
                duration = max(0.0, (now - self.stuck_since_ns) / 1e9) if self.stuck_since_ns else 0.0
                self.publish_event('robot_stuck_ended', {'episode_duration_sec': round(duration, 3)})
            self.stuck_since_ns = None
            self.stuck_latched = False

        duration = max(0.0, (now - self.stuck_since_ns) / 1e9) if self.stuck_since_ns is not None else 0.0
        task_stalled = False
        time_since_progress = None
        if command_active and self.last_progress_ns is not None:
            time_since_progress = max(0.0, (now - self.last_progress_ns) / 1e9)
            task_stalled = time_since_progress >= self.task_stall_thr
        if task_stalled and not self.task_stalled_latched:
            self.task_stalled_latched = True
            self.publish_event('task_progress_stalled', {
                'time_since_last_delivery_sec': round(time_since_progress, 3),
                'threshold_sec': self.task_stall_thr,
            })
        elif not task_stalled and self.task_stalled_latched:
            self.task_stalled_latched = False
            self.publish_event('task_progress_resumed', {})

        stuck_msg = Bool(); stuck_msg.data = self.stuck_latched; self.stuck_pub.publish(stuck_msg)
        stall_msg = Bool(); stall_msg.data = task_stalled; self.stall_pub.publish(stall_msg)
        metrics = {
            'status': 'READY' if pose_fresh else 'WAITING_FOR_GAZEBO_ROBOT_POSE',
            'command_active': bool(command_active),
            'command_linear_mps': round(self.cmd_linear, 4),
            'command_angular_rps': round(self.cmd_angular, 4),
            'ground_truth_linear_speed_mps': round(self.measured_linear, 4),
            'ground_truth_angular_speed_rps': round(self.measured_angular, 4),
            'robot_stuck': bool(self.stuck_latched),
            'current_stuck_duration_sec': round(duration, 3),
            'robot_stuck_episodes_total': self.deadlock_events,
            'task_progress_stalled': bool(task_stalled),
            'time_since_last_delivery_sec': round(time_since_progress, 3) if time_since_progress is not None else None,
            'stuck_detection_method': 'Gazebo ground-truth model pose finite difference, not wheel odometry',
            'ground_truth_usage': 'evaluation_only_not_policy_input',
        }
        out = String(); out.data = json.dumps(metrics, separators=(',', ':'))
        self.metrics_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeadlockMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
