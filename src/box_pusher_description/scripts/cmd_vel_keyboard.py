#!/usr/bin/env python3
"""Small terminal teleop publishing the Jazzy TwistStamped diff-drive input.

Keys: w forward, s reverse, a rotate left, d rotate right, x stop, q quit.
"""

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.parameter import Parameter


class KeyboardDrive(Node):
    def __init__(self, robot_namespace: str):
        super().__init__(
            'box_pusher_keyboard_teleop',
            parameter_overrides=[Parameter('use_sim_time', value=True)]
        )
        topic = f'/{robot_namespace.strip("/")}/diff_drive_controller/cmd_vel'
        self.publisher = self.create_publisher(TwistStamped, topic, 10)
        self.linear = 0.0
        self.angular = 0.0
        self.timer = self.create_timer(0.1, self.publish_command)
        self.get_logger().info(f'Publishing TwistStamped to {topic}')

    def set_key(self, key: str):
        speeds = {
            'w': (0.20, 0.0),
            's': (-0.20, 0.0),
            'a': (0.0, 0.75),
            'd': (0.0, -0.75),
            'x': (0.0, 0.0),
        }
        if key in speeds:
            self.linear, self.angular = speeds[key]

    def publish_command(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = self.linear
        msg.twist.angular.z = self.angular
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardDrive('robot1')
    if not sys.stdin.isatty():
        node.get_logger().error('Run this executable from an interactive terminal.')
        node.destroy_node()
        rclpy.shutdown()
        return

    old_settings = termios.tcgetattr(sys.stdin)
    print('\n[w] forward  [s] reverse  [a/d] turn  [x] stop  [q] quit\n')
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                key = sys.stdin.read(1).lower()
                if key == 'q':
                    node.linear = 0.0
                    node.angular = 0.0
                    for _ in range(3):
                        node.publish_command()
                    break
                node.set_key(key)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.linear = 0.0
        node.angular = 0.0
        for _ in range(3):
            node.publish_command()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
