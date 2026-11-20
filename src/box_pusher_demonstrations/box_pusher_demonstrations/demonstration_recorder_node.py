#!/usr/bin/env python3
"""Timestamp-aware demonstration recorder for V2 box-pushing.

Every received TwistStamped command is treated as an action sample. For each
command at time t, the recorder selects the latest sensor message with stamp
<= t. The paper-style policy observation is the left raw RGB image resized to
96x72. Stereo/depth/IMU/encoder/odometry are preserved as auxiliary channels.
Gazebo ground-truth poses, collision events and evaluator outputs are
intentionally NOT subscribed to or written into the policy dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger

from .sync_utils import StampedCache, message_stamp_ns


def _vector3_to_dict(v) -> Dict[str, float]:
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def _quaternion_to_dict(q) -> Dict[str, float]:
    return {"x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w)}


def _camera_info_to_dict(msg: Optional[CameraInfo]) -> Optional[Dict[str, Any]]:
    if msg is None:
        return None
    return {
        "stamp_ns": message_stamp_ns(msg),
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": msg.distortion_model,
        "d": [float(x) for x in msg.d],
        "k": [float(x) for x in msg.k],
        "r": [float(x) for x in msg.r],
        "p": [float(x) for x in msg.p],
        "binning_x": int(msg.binning_x),
        "binning_y": int(msg.binning_y),
        "roi": {
            "x_offset": int(msg.roi.x_offset),
            "y_offset": int(msg.roi.y_offset),
            "height": int(msg.roi.height),
            "width": int(msg.roi.width),
            "do_rectify": bool(msg.roi.do_rectify),
        },
    }


def _imu_to_dict(msg: Optional[Imu]) -> Optional[Dict[str, Any]]:
    if msg is None:
        return None
    return {
        "stamp_ns": message_stamp_ns(msg),
        "frame_id": msg.header.frame_id,
        "orientation_xyzw": _quaternion_to_dict(msg.orientation),
        "orientation_covariance": [float(x) for x in msg.orientation_covariance],
        "angular_velocity_rad_s": _vector3_to_dict(msg.angular_velocity),
        "angular_velocity_covariance": [float(x) for x in msg.angular_velocity_covariance],
        "linear_acceleration_m_s2": _vector3_to_dict(msg.linear_acceleration),
        "linear_acceleration_covariance": [float(x) for x in msg.linear_acceleration_covariance],
    }


def _joint_state_to_dict(msg: Optional[JointState]) -> Optional[Dict[str, Any]]:
    if msg is None:
        return None
    return {
        "stamp_ns": message_stamp_ns(msg),
        "name": list(msg.name),
        "position_rad": [float(x) for x in msg.position],
        "velocity_rad_s": [float(x) for x in msg.velocity],
        "effort": [float(x) for x in msg.effort],
    }


def _odom_to_dict(msg: Optional[Odometry]) -> Optional[Dict[str, Any]]:
    if msg is None:
        return None
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return {
        "stamp_ns": message_stamp_ns(msg),
        "frame_id": msg.header.frame_id,
        "child_frame_id": msg.child_frame_id,
        "pose": {
            "position_m": _vector3_to_dict(p),
            "orientation_xyzw": _quaternion_to_dict(q),
            "covariance": [float(x) for x in msg.pose.covariance],
        },
        "twist": {
            "linear_m_s": _vector3_to_dict(msg.twist.twist.linear),
            "angular_rad_s": _vector3_to_dict(msg.twist.twist.angular),
            "covariance": [float(x) for x in msg.twist.covariance],
        },
    }


class DemonstrationRecorder(Node):
    def __init__(self):
        super().__init__("demonstration_recorder")

        self.declare_parameter(
            "dataset_root",
            str(Path.home() / "Imitation_for_Swarm" / "datasets" / "demonstrations"),
        )
        self.declare_parameter("condition", "single")
        self.declare_parameter("operator_view", "onboard")
        self.declare_parameter("robot_count", 1)
        self.declare_parameter("world_seed", -1)
        self.declare_parameter("expected_control_hz", 5.0)
        self.declare_parameter("control_method", "keyboard")
        self.declare_parameter("max_left_image_age_sec", 0.15)
        self.declare_parameter("max_optional_image_age_sec", 0.20)
        self.declare_parameter("max_depth_age_sec", 0.60)
        self.declare_parameter("max_aux_age_sec", 0.20)
        self.declare_parameter("policy_image_width", 96)
        self.declare_parameter("policy_image_height", 72)
        self.declare_parameter("cmd_vel_topic", "/robot1/diff_drive_controller/cmd_vel")
        self.declare_parameter("left_image_topic", "/robot1/stereo/left/image_raw")
        self.declare_parameter("right_image_topic", "/robot1/stereo/right/image_raw")
        self.declare_parameter("left_rect_topic", "/robot1/stereo/left/image_rect")
        self.declare_parameter("right_rect_topic", "/robot1/stereo/right/image_rect")
        self.declare_parameter("depth_topic", "/robot1/stereo/depth")
        self.declare_parameter("left_info_topic", "/robot1/stereo/left/camera_info")
        self.declare_parameter("right_info_topic", "/robot1/stereo/right/camera_info")
        self.declare_parameter("left_rect_info_topic", "/robot1/stereo/left/camera_info_rect")
        self.declare_parameter("right_rect_info_topic", "/robot1/stereo/right/camera_info_rect")
        self.declare_parameter("imu_topic", "/robot1/imu/data")
        self.declare_parameter("joint_states_topic", "/robot1/joint_states")
        self.declare_parameter("odom_topic", "/robot1/diff_drive_controller/odom")
        self.declare_parameter("tracks_topic", "/robot1/perception/orange_boxes/tracks")
        self.declare_parameter("geometry_topic", "/robot1/perception/orange_boxes/geometry")

        self.dataset_root = Path(str(self.get_parameter("dataset_root").value)).expanduser()
        self.condition = str(self.get_parameter("condition").value)
        self.operator_view = str(self.get_parameter("operator_view").value)
        self.robot_count = int(self.get_parameter("robot_count").value)
        self.world_seed = int(self.get_parameter("world_seed").value)
        self.control_hz = float(self.get_parameter("expected_control_hz").value)
        self.control_method = str(self.get_parameter("control_method").value)
        self.max_left_age = float(self.get_parameter("max_left_image_age_sec").value)
        self.max_optional_image_age = float(
            self.get_parameter("max_optional_image_age_sec").value
        )
        self.max_depth_age = float(self.get_parameter("max_depth_age_sec").value)
        self.max_aux_age = float(self.get_parameter("max_aux_age_sec").value)
        self.policy_width = int(self.get_parameter("policy_image_width").value)
        self.policy_height = int(self.get_parameter("policy_image_height").value)

        if self.condition not in {"single", "recovery", "static", "dynamic", "debug"}:
            raise ValueError("condition must be single/recovery/static/dynamic/debug")
        if self.operator_view not in {"onboard", "global"}:
            raise ValueError("operator_view must be onboard or global")
        if self.robot_count < 1:
            raise ValueError("robot_count must be >= 1")
        if self.control_hz <= 0.0:
            raise ValueError("expected_control_hz must be > 0")

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Bounded caches hold ROS messages until an action arrives.
        self.caches = {
            "left_raw": StampedCache[Image](maxlen=30),
            "right_raw": StampedCache[Image](maxlen=30),
            "left_rect": StampedCache[Image](maxlen=20),
            "right_rect": StampedCache[Image](maxlen=20),
            "depth": StampedCache[Image](maxlen=15),
            "imu": StampedCache[Imu](maxlen=30),
            "joint_states": StampedCache[JointState](maxlen=20),
            "odom": StampedCache[Odometry](maxlen=20),
        }
        self.camera_infos: Dict[str, Optional[CameraInfo]] = {
            "left_raw": None,
            "right_raw": None,
            "left_rect": None,
            "right_rect": None,
        }
        self._bridge = CvBridge()
        self._active = False
        self._episode_dir: Optional[Path] = None
        self._samples_file = None
        self._episode_start_ns = 0
        self._episode_start_wall = ""
        self._sample_index = 0
        self._last_action_stamp_ns = 0
        self._counts = {
            "cmd_messages": 0,
            "saved_samples": 0,
            "dropped_no_left_image": 0,
            "dropped_stale_left_image": 0,
            "dropped_non_monotonic_action": 0,
            "image_write_failures": 0,
            "optional_right_missing": 0,
            "optional_rectified_missing": 0,
            "optional_depth_missing": 0,
        }
        self._last_error = ""

        def topic(param_name: str) -> str:
            return str(self.get_parameter(param_name).value)

        self.create_subscription(
            Image, topic("left_image_topic"),
            lambda msg: self.caches["left_raw"].append(msg), sensor_qos
        )
        self.create_subscription(
            Image, topic("right_image_topic"),
            lambda msg: self.caches["right_raw"].append(msg), sensor_qos
        )
        self.create_subscription(
            Image, topic("left_rect_topic"),
            lambda msg: self.caches["left_rect"].append(msg), sensor_qos
        )
        self.create_subscription(
            Image, topic("right_rect_topic"),
            lambda msg: self.caches["right_rect"].append(msg), sensor_qos
        )
        self.create_subscription(
            Image, topic("depth_topic"),
            lambda msg: self.caches["depth"].append(msg), sensor_qos
        )
        self.create_subscription(
            Imu, topic("imu_topic"),
            lambda msg: self.caches["imu"].append(msg), sensor_qos
        )
        self.create_subscription(
            JointState, topic("joint_states_topic"),
            lambda msg: self.caches["joint_states"].append(msg), reliable_qos
        )
        self.create_subscription(
            Odometry, topic("odom_topic"),
            lambda msg: self.caches["odom"].append(msg), reliable_qos
        )

        self.create_subscription(
            CameraInfo, topic("left_info_topic"),
            lambda msg: self._set_camera_info("left_raw", msg), sensor_qos
        )
        self.create_subscription(
            CameraInfo, topic("right_info_topic"),
            lambda msg: self._set_camera_info("right_raw", msg), sensor_qos
        )
        self.create_subscription(
            CameraInfo, topic("left_rect_info_topic"),
            lambda msg: self._set_camera_info("left_rect", msg), sensor_qos
        )
        self.create_subscription(
            CameraInfo, topic("right_rect_info_topic"),
            lambda msg: self._set_camera_info("right_rect", msg), sensor_qos
        )

        self.create_subscription(
            TwistStamped, topic("cmd_vel_topic"), self._on_action, reliable_qos
        )
        self.metrics_pub = self.create_publisher(String, "metrics", 10)
        self.active_pub = self.create_publisher(Bool, "recording", 10)
        self.create_service(Trigger, "start", self._start_recording)
        self.create_service(Trigger, "stop", self._stop_recording)
        self.create_timer(1.0, self._publish_metrics)

        self.get_logger().info("Demonstration recorder ready")
        self.get_logger().info(f"Action input: {topic('cmd_vel_topic')}")
        self.get_logger().info(f"Policy baseline observation: {topic('left_image_topic')} -> "
                               f"{self.policy_width}x{self.policy_height}")
        self.get_logger().info(f"Dataset root: {self.dataset_root}")

    def _set_camera_info(self, key: str, msg: CameraInfo) -> None:
        self.camera_infos[key] = msg

    def _write_json(self, path: Path, value: Dict[str, Any]) -> None:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    def _calibration_payload(self) -> Dict[str, Any]:
        payload = {
            key: _camera_info_to_dict(msg)
            for key, msg in self.camera_infos.items()
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    def _start_recording(self, request, response):
        del request
        if self._active:
            response.success = False
            response.message = "Recording is already active. Stop the current episode first."
            return response
        if len(self.caches["left_raw"]) == 0:
            response.success = False
            response.message = "No left RGB image received yet. Start Gazebo/camera pipeline first."
            return response

        try:
            self.dataset_root.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)
            stamp = now.strftime("%Y%m%dT%H%M%S")
            episode_id = f"{self.condition}_{stamp}_{uuid.uuid4().hex[:8]}"
            self._episode_dir = self.dataset_root / episode_id
            self._episode_dir.mkdir(parents=True, exist_ok=False)
            for folder in (
                "rgb_left_raw",
                "rgb_right_raw",
                "rgb_left_rect",
                "rgb_right_rect",
                "policy_rgb_96x72",
                "depth_mm",
            ):
                (self._episode_dir / folder).mkdir()
            self._samples_file = (self._episode_dir / "samples.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )
            self._episode_start_ns = int(self.get_clock().now().nanoseconds)
            self._episode_start_wall = now.isoformat()
            self._sample_index = 0
            self._last_action_stamp_ns = 0
            self._counts = {key: 0 for key in self._counts}
            self._last_error = ""
            self._active = True
            metadata = self._base_metadata(episode_id)
            metadata["status"] = "recording"
            self._write_json(self._episode_dir / "episode_metadata.json", metadata)
            self._write_json(self._episode_dir / "camera_calibration.json", self._calibration_payload())
            self.get_logger().info(f"Recording started: {self._episode_dir}")
            response.success = True
            response.message = str(self._episode_dir)
        except Exception as exc:  # noqa: BLE001
            self._active = False
            self._last_error = repr(exc)
            response.success = False
            response.message = f"Could not start recording: {exc}"
        return response

    def _base_metadata(self, episode_id: str) -> Dict[str, Any]:
        return {
            "format": "box_pusher_demonstration_v1",
            "episode_id": episode_id,
            "condition": self.condition,
            "operator_view": self.operator_view,
            "robot_count_in_scene": self.robot_count,
            "world_seed": self.world_seed,
            "expected_control_hz": self.control_hz,
            "control_method": self.control_method,
            "observation_action_alignment": "latest_sensor_stamp_at_or_before_action_stamp",
            "paper_baseline_policy_input": {
                "topic": str(self.get_parameter("left_image_topic").value),
                "representation": "single_front_rgb",
                "resize_width": self.policy_width,
                "resize_height": self.policy_height,
                "policy_input_default": True,
            },
            "v2_auxiliary_channels": [
                "right_rgb", "left_rectified_rgb", "right_rectified_rgb", "depth_mm",
                "imu", "joint_states", "wheel_odometry", "camera_calibration",
            ],
            "auxiliary_channels_policy_input_default": False,
            "action": {
                "topic": str(self.get_parameter("cmd_vel_topic").value),
                "message_type": "geometry_msgs/msg/TwistStamped",
                "label": "linear_x_m_s_and_angular_z_rad_s",
                "semantics": "command observed on controller cmd_vel input topic; controller acceptance is not independently acknowledged",
            },
            "topics": {
                name: str(self.get_parameter(param).value)
                for name, param in {
                    "left_rgb": "left_image_topic",
                    "right_rgb": "right_image_topic",
                    "left_rectified_rgb": "left_rect_topic",
                    "right_rectified_rgb": "right_rect_topic",
                    "depth": "depth_topic",
                    "imu": "imu_topic",
                    "joint_states": "joint_states_topic",
                    "odometry": "odom_topic",
                }.items()
            },
            "ground_truth_policy_leakage": False,
            "ground_truth_topics_subscribed": [],
            "created_wall_time_utc": self._episode_start_wall,
            "start_ros_time_ns": self._episode_start_ns,
            "notes": [
                "Gazebo ground-truth object/robot poses and evaluator outputs are not recorded as policy observations.",
                "The paper-style baseline uses only policy_rgb_96x72 and the recorded action.",
                "Auxiliary stereo/depth/IMU/encoder/odometry data are kept for separate V2 extension experiments.",
            ],
        }

    def _select(self, cache_name: str, target_ns: int, max_age: float):
        return self.caches[cache_name].latest_at_or_before(target_ns, max_age)

    def _write_color_image(self, folder: str, index: int, msg: Image) -> Optional[Dict[str, Any]]:
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = self._episode_dir / folder / f"{index:06d}.png"
        if not cv2.imwrite(str(path), image):
            raise IOError(f"cv2.imwrite failed: {path}")
        return {
            "path": str(path.relative_to(self._episode_dir)),
            "stamp_ns": message_stamp_ns(msg),
            "age_ms": None,
            "source_encoding": msg.encoding,
            "width": int(msg.width),
            "height": int(msg.height),
            "storage": "lossless_png_rgb",
        }

    def _save_sample(self, action: TwistStamped) -> bool:
        action_ns = message_stamp_ns(action)
        if action_ns <= 0:
            self._counts["dropped_non_monotonic_action"] += 1
            return False
        if self._last_action_stamp_ns and action_ns <= self._last_action_stamp_ns:
            self._counts["dropped_non_monotonic_action"] += 1
            return False

        left = self._select("left_raw", action_ns, self.max_left_age)
        if left is None:
            if len(self.caches["left_raw"]) == 0:
                self._counts["dropped_no_left_image"] += 1
            else:
                self._counts["dropped_stale_left_image"] += 1
            return False

        index = self._sample_index
        record: Dict[str, Any] = {
            "sample_index": index,
            "action_stamp_ns": action_ns,
            "action_stamp_sec": action_ns / 1e9,
            "elapsed_sim_sec": (action_ns - self._episode_start_ns) / 1e9,
            "action": {
                "linear": _vector3_to_dict(action.twist.linear),
                "angular": _vector3_to_dict(action.twist.angular),
                "linear_x_m_s": float(action.twist.linear.x),
                "angular_z_rad_s": float(action.twist.angular.z),
                "frame_id": action.header.frame_id,
                "topic": str(self.get_parameter("cmd_vel_topic").value),
                "source": "observed_controller_cmd_vel_input",
            },
            "observation": {},
            "auxiliary": {},
            "quality": {},
        }
        image_specs = [
            ("left_raw", "rgb_left_raw", self.max_left_age, True),
            ("right_raw", "rgb_right_raw", self.max_optional_image_age, False),
            ("left_rect", "rgb_left_rect", self.max_optional_image_age, False),
            ("right_rect", "rgb_right_rect", self.max_optional_image_age, False),
        ]
        try:
            for cache_name, folder, max_age, required in image_specs:
                selected = left if cache_name == "left_raw" else self._select(cache_name, action_ns, max_age)
                if selected is None:
                    if cache_name == "right_raw":
                        self._counts["optional_right_missing"] += 1
                    if cache_name in ("left_rect", "right_rect"):
                        self._counts["optional_rectified_missing"] += 1
                    if required:
                        return False
                    continue
                image_record = self._write_color_image(folder, index, selected.message)
                image_record["age_ms"] = round(selected.age_sec * 1000.0, 3)
                record["observation"][cache_name] = image_record

            # Paper-style single-camera RGB input: raw left RGB resized to 96x72.
            raw_bgr = self._bridge.imgmsg_to_cv2(left.message, desired_encoding="bgr8")
            policy_bgr = cv2.resize(
                raw_bgr,
                (self.policy_width, self.policy_height),
                interpolation=cv2.INTER_AREA,
            )
            policy_path = self._episode_dir / "policy_rgb_96x72" / f"{index:06d}.png"
            if not cv2.imwrite(str(policy_path), policy_bgr):
                raise IOError(f"cv2.imwrite failed: {policy_path}")
            record["observation"]["policy_rgb_96x72"] = {
                "path": str(policy_path.relative_to(self._episode_dir)),
                "source": "left_raw_rgb",
                "stamp_ns": left.stamp_ns,
                "age_ms": round(left.age_sec * 1000.0, 3),
                "width": self.policy_width,
                "height": self.policy_height,
                "storage": "lossless_png_rgb",
            }

            depth = self._select("depth", action_ns, self.max_depth_age)
            if depth is not None:
                depth_array = self._bridge.imgmsg_to_cv2(
                    depth.message, desired_encoding="passthrough"
                )
                depth_array = np.asarray(depth_array, dtype=np.float32)
                valid = np.isfinite(depth_array) & (depth_array > 0.0)
                depth_mm = np.zeros(depth_array.shape, dtype=np.uint16)
                depth_mm[valid] = np.clip(
                    np.rint(depth_array[valid] * 1000.0), 1, 65535
                ).astype(np.uint16)
                depth_path = self._episode_dir / "depth_mm" / f"{index:06d}.png"
                if not cv2.imwrite(str(depth_path), depth_mm):
                    raise IOError(f"cv2.imwrite failed: {depth_path}")
                record["observation"]["depth_mm"] = {
                    "path": str(depth_path.relative_to(self._episode_dir)),
                    "stamp_ns": depth.stamp_ns,
                    "age_ms": round(depth.age_sec * 1000.0, 3),
                    "encoding": "16UC1",
                    "scale": "millimetres_per_unit",
                    "invalid_value": 0,
                    "valid_fraction": float(np.count_nonzero(valid) / max(1, depth_array.size)),
                }
            else:
                self._counts["optional_depth_missing"] += 1

            for cache_name, output_name in (
                ("imu", "imu"),
                ("joint_states", "joint_states"),
                ("odom", "wheel_odometry"),
            ):
                selected = self._select(cache_name, action_ns, self.max_aux_age)
                if selected is None:
                    record["quality"][f"{output_name}_fresh"] = False
                    continue
                record["quality"][f"{output_name}_fresh"] = True
                record["auxiliary"][output_name] = {
                    "age_ms": round(selected.age_sec * 1000.0, 3),
                    "data": (
                        _imu_to_dict(selected.message)
                        if cache_name == "imu"
                        else _joint_state_to_dict(selected.message)
                        if cache_name == "joint_states"
                        else _odom_to_dict(selected.message)
                    ),
                }

            left_stamp = record["observation"]["left_raw"]["stamp_ns"]
            right_stamp = record["observation"].get("right_raw", {}).get("stamp_ns")
            left_rect_stamp = record["observation"].get("left_rect", {}).get("stamp_ns")
            depth_stamp = record["observation"].get("depth_mm", {}).get("stamp_ns")
            record["quality"].update({
                "left_rgb_fresh": True,
                "stereo_pair_delta_ms": (abs(left_stamp - right_stamp) / 1e6) if right_stamp else None,
                "rgbd_pair_delta_ms": (abs(left_rect_stamp - depth_stamp) / 1e6) if left_rect_stamp and depth_stamp else None,
                "right_rgb_present": "right_raw" in record["observation"],
                "rectified_rgb_present": "left_rect" in record["observation"],
                "depth_present": "depth_mm" in record["observation"],
            })
            self._samples_file.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
            self._samples_file.flush()
            self._sample_index += 1
            self._counts["saved_samples"] += 1
            self._last_action_stamp_ns = action_ns
            return True
        except Exception as exc:  # noqa: BLE001
            self._counts["image_write_failures"] += 1
            self._last_error = repr(exc)
            self.get_logger().error(f"Failed to save sample {index}: {exc}")
            return False

    def _on_action(self, msg: TwistStamped) -> None:
        if not self._active:
            return
        self._counts["cmd_messages"] += 1
        self._save_sample(msg)

    def _publish_metrics(self) -> None:
        active = Bool()
        active.data = self._active
        self.active_pub.publish(active)
        payload = {
            "recording": self._active,
            "condition": self.condition,
            "episode_dir": str(self._episode_dir) if self._episode_dir else None,
            "sample_index_next": self._sample_index,
            **self._counts,
            "last_error": self._last_error,
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.metrics_pub.publish(out)

    def _stop_recording(self, request, response):
        del request
        if not self._active:
            response.success = False
            response.message = "No active recording episode."
            return response
        try:
            self._active = False
            if self._samples_file is not None:
                self._samples_file.flush()
                self._samples_file.close()
                self._samples_file = None
            metadata_path = self._episode_dir / "episode_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["status"] = "complete"
            metadata["end_wall_time_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["end_ros_time_ns"] = int(self.get_clock().now().nanoseconds)
            metadata["sample_count"] = self._sample_index
            metadata["recorder_counters"] = self._counts
            metadata["last_error"] = self._last_error
            self._write_json(metadata_path, metadata)
            self._write_json(self._episode_dir / "camera_calibration.json", self._calibration_payload())
            response.success = True
            response.message = (
                f"Saved {self._sample_index} samples to {self._episode_dir}"
            )
            self.get_logger().info(response.message)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Error while closing episode: {exc}"
            self.get_logger().error(response.message)
        return response

    def destroy_node(self):
        if self._active:
            response = Trigger.Response()
            self._stop_recording(Trigger.Request(), response)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DemonstrationRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
