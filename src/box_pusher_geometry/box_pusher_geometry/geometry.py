"""Pure NumPy geometry utilities for estimating an orange box from RGB-D points.

The estimator intentionally reports the square-box yaw modulo pi/2 because an
unmarked 80 x 80 mm box is rotationally symmetric under 90-degree rotations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def normalize_quaternion(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(q), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quaternion must contain xyzw")
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError("invalid quaternion")
    return q / n


def quaternion_to_matrix(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def wrap_square_yaw(yaw: float) -> float:
    """Canonical representative of an unmarked square's yaw modulo pi/2."""
    period = 0.5 * math.pi
    value = float(yaw) % period
    return value if value <= 0.5 * period else value - period


def transform_point(translation: Sequence[float], quaternion_xyzw: Sequence[float], point: Sequence[float]) -> np.ndarray:
    t = np.asarray(translation, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    if t.shape != (3,) or p.shape != (3,):
        raise ValueError("translation and point must be length 3")
    return quaternion_to_matrix(quaternion_xyzw) @ p + t


def transform_points(translation: Sequence[float], quaternion_xyzw: Sequence[float], points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    return (quaternion_to_matrix(quaternion_xyzw) @ points.T).T + np.asarray(translation, dtype=np.float64)


def backproject_rectified_pixels(
    u: np.ndarray, v: np.ndarray, depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Back-project rectified pixel coordinates into rectified optical XYZ."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(depth_m, dtype=np.float64)
    if not (u.shape == v.shape == z.shape):
        raise ValueError("u, v and depth arrays must have the same shape")
    if fx <= 0 or fy <= 0:
        raise ValueError("focal lengths must be positive")
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack((x, y, z))


@dataclass
class FaceEstimate:
    success: bool
    reason: str
    surface_center: Optional[np.ndarray] = None
    geometric_center: Optional[np.ndarray] = None
    outward_normal: Optional[np.ndarray] = None
    edge_direction: Optional[np.ndarray] = None
    yaw_edge_mod_pi_over_2: Optional[float] = None
    approach_yaw: Optional[float] = None
    observed_width_m: Optional[float] = None
    observed_height_m: Optional[float] = None
    plane_inlier_count: int = 0
    plane_inlier_ratio: float = 0.0
    plane_rmse_m: Optional[float] = None
    confidence: float = 0.0


def fit_visible_vertical_face(
    points_base: np.ndarray,
    camera_origin_base: Sequence[float],
    box_xy_size_m: float = 0.08,
    box_height_m: float = 0.10,
    plane_threshold_m: float = 0.008,
    max_normal_abs_z: float = 0.35,
    min_points: int = 18,
    ransac_iterations: int = 160,
    max_ransac_points: int = 500,
    seed: int = 13,
) -> FaceEstimate:
    """Fit the strongest near-vertical planar box face and infer its centre.

    Assumptions: box is an 80x80x100 mm orange cuboid resting on the floor.
    The selected plane is a visible side face. Its normal is oriented toward the
    camera; the geometric centre is estimated 40 mm behind that face. Since the
    horizontal footprint is square and unmarked, yaw is only observable modulo
    90 degrees. This is an estimate, not a general arbitrary-object 6-DoF pose.
    """
    pts = np.asarray(points_base, dtype=np.float64)
    cam = np.asarray(camera_origin_base, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or cam.shape != (3,):
        raise ValueError("points must be Nx3 and camera origin length 3")
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < min_points:
        return FaceEstimate(False, "INSUFFICIENT_3D_POINTS")
    if len(pts) > max_ransac_points:
        take = np.linspace(0, len(pts)-1, max_ransac_points).astype(int)
        sample_pts = pts[take]
    else:
        sample_pts = pts

    rng = np.random.default_rng(seed)
    best_mask = None
    best_count = 0
    n = len(sample_pts)
    for _ in range(max(1, ransac_iterations)):
        idx = rng.choice(n, 3, replace=False)
        a, b, c = sample_pts[idx]
        normal = np.cross(b-a, c-a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(float(normal[2])) > max_normal_abs_z:
            continue
        distances = np.abs((sample_pts - a) @ normal)
        inliers = distances <= plane_threshold_m
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_mask = inliers

    required = max(min_points, int(math.ceil(0.18 * n)))
    if best_mask is None or best_count < required:
        return FaceEstimate(False, "NO_STABLE_VERTICAL_FACE", plane_inlier_count=best_count,
                            plane_inlier_ratio=best_count / max(1, n))

    inliers = sample_pts[best_mask]
    origin = np.mean(inliers, axis=0)
    centered = inliers - origin
    # SVD refines the face plane normal.
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    normal /= max(1e-12, float(np.linalg.norm(normal)))
    if abs(float(normal[2])) > max_normal_abs_z:
        return FaceEstimate(False, "REFINED_FACE_NOT_VERTICAL", plane_inlier_count=best_count,
                            plane_inlier_ratio=best_count / max(1, n))

    # Orient the outward face normal toward the optical camera centre.
    face_center = np.median(inliers, axis=0)
    to_camera = cam - face_center
    if float(np.dot(normal, to_camera)) < 0.0:
        normal = -normal

    # Principal horizontal direction of the selected vertical face.
    xy = inliers[:, :2]
    xy_center = np.mean(xy, axis=0)
    xy_centered = xy - xy_center
    cov_xy = (xy_centered.T @ xy_centered) / max(1, len(xy_centered)-1)
    evals, evecs = np.linalg.eigh(cov_xy)
    edge = evecs[:, int(np.argmax(evals))]
    edge = edge / max(1e-12, float(np.linalg.norm(edge)))
    # Resolve sign for stable output (yaw still modulo 90 degrees).
    if edge[0] < 0.0 or (abs(edge[0]) < 1e-9 and edge[1] < 0.0):
        edge = -edge
    edge3 = np.array([edge[0], edge[1], 0.0], dtype=np.float64)
    edge_yaw = math.atan2(float(edge[1]), float(edge[0]))
    yaw_mod = wrap_square_yaw(edge_yaw)
    approach_yaw = math.atan2(float(-normal[1]), float(-normal[0]))

    along = xy_centered @ edge
    observed_width = float(np.percentile(along, 95) - np.percentile(along, 5))
    observed_height = float(np.percentile(inliers[:, 2], 95) - np.percentile(inliers[:, 2], 5))
    residuals = (inliers - face_center) @ normal
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ratio = float(best_count / max(1, n))

    # The visible surface face centre is shifted inward by half the known box depth.
    # z uses the floor-support prior; measured plane z is also retained separately.
    geometric_center = face_center - normal * (0.5 * float(box_xy_size_m))
    geometric_center[2] = 0.5 * float(box_height_m)

    width_score = math.exp(-abs(observed_width - box_xy_size_m) / max(0.02, box_xy_size_m))
    plane_score = math.exp(-rmse / max(1e-6, plane_threshold_m))
    point_score = min(1.0, best_count / 80.0)
    confidence = float(np.clip(0.40*ratio + 0.35*plane_score + 0.25*width_score, 0.0, 1.0) * point_score)

    return FaceEstimate(
        success=True,
        reason="VERTICAL_FACE_FIT",
        surface_center=face_center,
        geometric_center=geometric_center,
        outward_normal=normal,
        edge_direction=edge3,
        yaw_edge_mod_pi_over_2=yaw_mod,
        approach_yaw=approach_yaw,
        observed_width_m=observed_width,
        observed_height_m=observed_height,
        plane_inlier_count=best_count,
        plane_inlier_ratio=ratio,
        plane_rmse_m=rmse,
        confidence=confidence,
    )


def interpolate_planar_pose(
    x0: float, y0: float, yaw0: float,
    x1: float, y1: float, yaw1: float,
    alpha: float,
) -> Tuple[float, float, float]:
    a = max(0.0, min(1.0, float(alpha)))
    dyaw = wrap_angle(yaw1 - yaw0)
    return (x0 + a*(x1-x0), y0 + a*(y1-y0), wrap_angle(yaw0 + a*dyaw))


def base_to_odom(point_base: Sequence[float], x: float, y: float, z: float, yaw: float) -> np.ndarray:
    p = np.asarray(point_base, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c*p[0] - s*p[1], y + s*p[0] + c*p[1], z + p[2]], dtype=np.float64)
