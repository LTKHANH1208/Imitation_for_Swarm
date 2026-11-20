#!/usr/bin/env python3
"""Safe terminal keyboard teleoperation for the V2 differential-drive robot.

The node publishes geometry_msgs/TwistStamped at 5 Hz by default. A command is
kept alive only while terminal key-repeat events continue to arrive. If no key
is received before the deadman timeout, the node publishes zero velocity.

Keys:
  w/s : forward/reverse
  a/d : rotate left/right
  q/e : forward-left / forward-right arc
  z/c : reverse-left / reverse-right arc
  SPACE or x : immediate stop
  +/- : adjust linear speed
  [/ ]: adjust angular speed
  ESC or Ctrl-C : stop and quit
"""
from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")
        self.declare_parameter(
            "cmd_vel_topic", "/robot1/diff_drive_controller/cmd_vel"
        )
        self.declare_parameter("frame_id", "base_footprint")
        self.declare_parameter("linear_speed", 0.10)
        self.declare_parameter("angular_speed", 0.45)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("deadman_timeout_sec", 0.45)
        self.declare_parameter("max_linear_speed", 0.20)
        self.declare_parameter("max_angular_speed", 0.80)

        topic = str(self.get_parameter("cmd_vel_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_speed = float(self.get_parameter("angular_speed").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.deadman_timeout_sec = float(
            self.get_parameter("deadman_timeout_sec").value
        )
        self.max_linear_speed = float(
            self.get_parameter("max_linear_speed").value
        )
        self.max_angular_speed = float(
            self.get_parameter("max_angular_speed").value
        )

        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be > 0")
        if self.deadman_timeout_sec <= 0.0:
            raise ValueError("deadman_timeout_sec must be > 0")

        self.publisher = self.create_publisher(TwistStamped, topic, 10)
        self._lock = threading.Lock()
        self._linear = 0.0
        self._angular = 0.0
        self._last_key_monotonic = 0.0
        self._active = False
        self._timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish)
        self.get_logger().info(
            f"Publishing TwistStamped to {topic} at {self.publish_rate_hz:.1f} Hz; "
            f"deadman={self.deadman_timeout_sec:.2f}s"
        )

    def _set_command(self, linear: float, angular: float) -> None:
        with self._lock:
            self._linear = max(-self.max_linear_speed, min(self.max_linear_speed, linear))
            self._angular = max(
                -self.max_angular_speed, min(self.max_angular_speed, angular)
            )
            self._last_key_monotonic = time.monotonic()
            self._active = (abs(self._linear) > 1e-9 or abs(self._angular) > 1e-9)

    def _stop(self) -> None:
        with self._lock:
            self._linear = 0.0
            self._angular = 0.0
            self._active = False
            self._last_key_monotonic = 0.0

    def handle_key(self, key: str) -> str:
        """Apply one key. Returns 'quit' only for ESC/Ctrl-C."""
        key = key.lower()
        v = self.linear_speed
        w = self.angular_speed
        commands = {
            "w": (v, 0.0),
            "s": (-v, 0.0),
            "a": (0.0, w),
            "d": (0.0, -w),
            "q": (v, w),
            "e": (v, -w),
            "z": (-v, w),
            "c": (-v, -w),
        }
        if key in commands:
            self._set_command(*commands[key])
        elif key in ("x", " "):
            self._stop()
        elif key == "+":
            self.linear_speed = min(self.max_linear_speed, self.linear_speed + 0.02)
            self.get_logger().info(f"Linear speed: {self.linear_speed:.2f} m/s")
        elif key == "-":
            self.linear_speed = max(0.02, self.linear_speed - 0.02)
            self.get_logger().info(f"Linear speed: {self.linear_speed:.2f} m/s")
        elif key == "]":
            self.angular_speed = min(self.max_angular_speed, self.angular_speed + 0.05)
            self.get_logger().info(f"Angular speed: {self.angular_speed:.2f} rad/s")
        elif key == "[":
            self.angular_speed = max(0.10, self.angular_speed - 0.05)
            self.get_logger().info(f"Angular speed: {self.angular_speed:.2f} rad/s")
        elif key == "\x1b" or key == "\x03":
            self._stop()
            return "quit"
        elif key == "?":
            print(
                "\n[w/s] forward/reverse  [a/d] rotate  [q/e] arcs  "
                "[z/c] reverse arcs  [SPACE/x] stop  [+/-] linear speed  "
                "[/] angular speed  [ESC/Ctrl-C] quit\n",
                flush=True,
            )
        return "continue"

    def _publish(self) -> None:
        with self._lock:
            if self._active and (
                time.monotonic() - self._last_key_monotonic > self.deadman_timeout_sec
            ):
                self._linear = 0.0
                self._angular = 0.0
                self._active = False
            linear = self._linear
            angular = self._angular

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.publisher.publish(msg)

    def publish_stop_burst(self, count: int = 3) -> None:
        self._stop()
        for _ in range(count):
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.publisher.publish(msg)
            time.sleep(0.03)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    if not sys.stdin.isatty():
        node.get_logger().error(
            "Keyboard teleop requires an interactive terminal. Run with `ros2 run` "
            "in a terminal, not as a background launch process."
        )
        node.destroy_node()
        rclpy.shutdown()
        return

    old_settings = termios.tcgetattr(sys.stdin.fileno())
    print(
        "\nV2 keyboard teleop (5 Hz)\n"
        " w/s: forward/reverse | a/d: rotate | q/e: forward arcs\n"
        " z/c: reverse arcs | SPACE/x: stop | +/-: linear speed\n"
        " [/]: angular speed | ESC/Ctrl-C: quit\n"
        "Hold a key (terminal key-repeat); deadman auto-stops after timeout.\n",
        flush=True,
    )
    try:
        tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if ready:
                key = sys.stdin.read(1)
                if node.handle_key(key) == "quit":
                    break
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop_burst()
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
