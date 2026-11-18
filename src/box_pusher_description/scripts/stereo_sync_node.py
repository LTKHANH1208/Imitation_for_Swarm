#!/usr/bin/env python3
"""Approximate-time synchronization and visual preview for the stereo pair.

This node does not invent stereo calibration or depth. It verifies that both
images and their CameraInfo messages arrive together, then publishes a
side-by-side preview for quick inspection in rqt_image_view.
"""

import cv2
import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo


class StereoSyncNode(Node):
    def __init__(self):
        super().__init__('stereo_sync_node')
        self.declare_parameter('sync_slop_sec', 0.04)
        self.declare_parameter('left_image_topic', 'stereo/left/image_raw')
        self.declare_parameter('right_image_topic', 'stereo/right/image_raw')
        self.declare_parameter('left_info_topic', 'stereo/left/camera_info')
        self.declare_parameter('right_info_topic', 'stereo/right/camera_info')
        self.declare_parameter('preview_topic', 'stereo/preview')

        self.bridge = CvBridge()
        self.pair_count = 0
        self.preview_pub = self.create_publisher(
            Image, self.get_parameter('preview_topic').value, qos_profile_sensor_data
        )

        left_img = message_filters.Subscriber(
            self, Image, self.get_parameter('left_image_topic').value,
            qos_profile=qos_profile_sensor_data
        )
        right_img = message_filters.Subscriber(
            self, Image, self.get_parameter('right_image_topic').value,
            qos_profile=qos_profile_sensor_data
        )
        left_info = message_filters.Subscriber(
            self, CameraInfo, self.get_parameter('left_info_topic').value,
            qos_profile=qos_profile_sensor_data
        )
        right_info = message_filters.Subscriber(
            self, CameraInfo, self.get_parameter('right_info_topic').value,
            qos_profile=qos_profile_sensor_data
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [left_img, right_img, left_info, right_info],
            queue_size=12,
            slop=float(self.get_parameter('sync_slop_sec').value),
            allow_headerless=False
        )
        self.sync.registerCallback(self.on_stereo_pair)
        self.get_logger().info('Waiting for synchronized left/right images and CameraInfo.')

    def on_stereo_pair(self, left_msg, right_msg, left_info, right_info):
        try:
            left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
            if left.shape[:2] != right.shape[:2]:
                self.get_logger().warning('Left/right image dimensions differ; skipping preview.')
                return

            # Add a thin divider so the two camera views are visually distinct.
            divider = 255 * (left[:, :2, :] * 0 + 1)
            pair = cv2.hconcat([left, divider, right])
            preview = self.bridge.cv2_to_imgmsg(pair, encoding='bgr8')
            preview.header = left_msg.header
            self.preview_pub.publish(preview)

            self.pair_count += 1
            if self.pair_count == 1 or self.pair_count % 100 == 0:
                self.get_logger().info(
                    f'Stereo pairs={self.pair_count}, '
                    f'left={left_info.width}x{left_info.height}, '
                    f'right={right_info.width}x{right_info.height}'
                )
        except Exception as exc:  # Keep the node alive if one corrupt frame arrives.
            self.get_logger().error(f'Could not process stereo pair: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = StereoSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
