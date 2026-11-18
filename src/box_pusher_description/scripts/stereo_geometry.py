#!/usr/bin/env python3
"""Geometry helpers shared by the V2 stereo ROS 2 nodes.

This module deliberately contains no ROS imports so the numerical parts can be
unit-tested on a plain Python/OpenCV installation.
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert an (x, y, z, w) unit quaternion to a 3x3 rotation matrix."""
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid zero-length or non-finite quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_info_intrinsics(camera_info) -> Tuple[np.ndarray, np.ndarray]:
    """Extract K and D from a sensor_msgs/CameraInfo-like object."""
    k = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
    d = np.asarray(camera_info.d, dtype=np.float64)
    if d.size == 0:
        d = np.zeros(5, dtype=np.float64)
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
        raise ValueError("CameraInfo contains invalid focal lengths / intrinsic matrix")
    if not np.all(np.isfinite(d)):
        raise ValueError("CameraInfo contains non-finite distortion coefficients")
    return k, d


def compute_rectification(
    k_left: np.ndarray,
    d_left: np.ndarray,
    k_right: np.ndarray,
    d_right: np.ndarray,
    image_size: Tuple[int, int],
    rotation_right_left: np.ndarray,
    translation_right_left: np.ndarray,
    alpha: float = 0.0,
):
    """Compute OpenCV stereo rectification maps using left->right TF extrinsics.

    Convention:
      X_right = R_right_left @ X_left + T_right_left

    `translation_right_left` is the transform translation returned by
    lookup_transform(target=right_optical_frame, source=left_optical_frame).
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image size must be positive")
    rotation_right_left = np.asarray(rotation_right_left, dtype=np.float64).reshape(3, 3)
    translation_right_left = np.asarray(translation_right_left, dtype=np.float64).reshape(3, 1)
    baseline = float(np.linalg.norm(translation_right_left))
    if not math.isfinite(baseline) or baseline < 1e-4:
        raise ValueError(f"Stereo baseline is invalid: {baseline}")

    r1, r2, p1, p2, q, roi_left, roi_right = cv2.stereoRectify(
        np.asarray(k_left, dtype=np.float64),
        np.asarray(d_left, dtype=np.float64),
        np.asarray(k_right, dtype=np.float64),
        np.asarray(d_right, dtype=np.float64),
        (int(width), int(height)),
        rotation_right_left,
        translation_right_left,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        k_left, d_left, r1, p1, (int(width), int(height)), cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        k_right, d_right, r2, p2, (int(width), int(height)), cv2.CV_32FC1
    )
    rectified_baseline = abs(float(p2[0, 3] / p2[0, 0]))
    if rectified_baseline < 1e-4:
        raise ValueError("Rectified projection matrices imply a near-zero baseline")

    return {
        "r1": r1,
        "r2": r2,
        "p1": p1,
        "p2": p2,
        "q": q,
        "roi_left": roi_left,
        "roi_right": roi_right,
        "map_lx": map_lx,
        "map_ly": map_ly,
        "map_rx": map_rx,
        "map_ry": map_ry,
        "baseline_m": rectified_baseline,
        "focal_px": float(p1[0, 0]),
        "raw_baseline_m": baseline,
    }


def depth_from_disparity(
    disparity_px: np.ndarray,
    focal_px: float,
    baseline_m: float,
    min_disparity: float,
    max_disparity: float,
    min_depth_m: float = 0.20,
    max_depth_m: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return metric depth (float32, NaN invalid) and a valid-pixel mask."""
    if not math.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError("Focal length must be finite and positive")
    if not math.isfinite(baseline_m) or baseline_m <= 0.0:
        raise ValueError("Baseline must be finite and positive")
    if min_depth_m <= 0.0 or max_depth_m <= min_depth_m:
        raise ValueError("Depth limits must satisfy 0 < min_depth < max_depth")

    disp = np.asarray(disparity_px, dtype=np.float32)
    valid = (
        np.isfinite(disp)
        & (disp > max(0.0, float(min_disparity)))
        & (disp < float(max_disparity))
    )
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    depth[valid] = (float(focal_px) * float(baseline_m)) / disp[valid]
    valid &= np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    depth[~valid] = np.nan
    return depth, valid
