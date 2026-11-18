#!/usr/bin/env python3
"""Pure geometry helpers for comparing stereo depth with Gazebo ground truth."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def pose_to_rt(position, orientation) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ROS Pose position/orientation to R_world_object, t_world_object."""
    r = quaternion_to_matrix(
        orientation.x, orientation.y, orientation.z, orientation.w
    )
    t = np.array([position.x, position.y, position.z], dtype=np.float64)
    return r, t


def ray_oriented_box_intersection(
    ray_origin_world: np.ndarray,
    ray_direction_world: np.ndarray,
    box_rotation_world: np.ndarray,
    box_center_world: np.ndarray,
    box_dimensions_xyz: np.ndarray,
) -> Optional[Tuple[float, np.ndarray]]:
    """Return nearest nonnegative ray hit (distance, world point) with an OBB.

    The ray direction is normalized internally. Slab intersection is performed
    in the box's local coordinates, so rotated Gazebo boxes are supported.
    """
    origin = np.asarray(ray_origin_world, dtype=np.float64).reshape(3)
    direction = np.asarray(ray_direction_world, dtype=np.float64).reshape(3)
    direction_norm = float(np.linalg.norm(direction))
    if not math.isfinite(direction_norm) or direction_norm < 1e-12:
        return None
    direction = direction / direction_norm

    rotation = np.asarray(box_rotation_world, dtype=np.float64).reshape(3, 3)
    center = np.asarray(box_center_world, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(box_dimensions_xyz, dtype=np.float64).reshape(3)
    if np.any(half <= 0.0):
        raise ValueError("Box dimensions must be positive")

    # World -> object local coordinates.
    local_origin = rotation.T @ (origin - center)
    local_direction = rotation.T @ direction
    t_near = -math.inf
    t_far = math.inf
    for axis in range(3):
        o = float(local_origin[axis])
        d = float(local_direction[axis])
        h = float(half[axis])
        if abs(d) < 1e-12:
            if o < -h or o > h:
                return None
            continue
        t1 = (-h - o) / d
        t2 = (h - o) / d
        t_near = max(t_near, min(t1, t2))
        t_far = min(t_far, max(t1, t2))
        if t_near > t_far:
            return None
    if t_far < 0.0:
        return None
    distance = t_near if t_near >= 0.0 else t_far
    hit = origin + distance * direction
    return float(distance), hit


def project_rectified_point(
    point_world: np.ndarray,
    world_rotation_camera: np.ndarray,
    world_translation_camera: np.ndarray,
    rectification_rotation_left: np.ndarray,
    projection_left: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Project world point into rectified left image; return (u, v, Z_rect)."""
    p_world = np.asarray(point_world, dtype=np.float64).reshape(3)
    r_wc = np.asarray(world_rotation_camera, dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(world_translation_camera, dtype=np.float64).reshape(3)
    r1 = np.asarray(rectification_rotation_left, dtype=np.float64).reshape(3, 3)
    pmat = np.asarray(projection_left, dtype=np.float64).reshape(3, 4)

    point_camera = r_wc.T @ (p_world - t_wc)
    point_rectified = r1 @ point_camera
    if not np.all(np.isfinite(point_rectified)) or point_rectified[2] <= 1e-6:
        return None
    projected = pmat @ np.array([
        point_rectified[0], point_rectified[1], point_rectified[2], 1.0
    ], dtype=np.float64)
    if not np.all(np.isfinite(projected)) or abs(projected[2]) < 1e-12:
        return None
    u = float(projected[0] / projected[2])
    v = float(projected[1] / projected[2])
    return u, v, float(point_rectified[2])


def box_corners_world(
    rotation_world_box: np.ndarray,
    center_world: np.ndarray,
    dimensions_xyz: np.ndarray,
) -> np.ndarray:
    """Generate eight OBB corners in world coordinates."""
    r = np.asarray(rotation_world_box, dtype=np.float64).reshape(3, 3)
    c = np.asarray(center_world, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(dimensions_xyz, dtype=np.float64).reshape(3)
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    return (r @ (signs * half).T).T + c
