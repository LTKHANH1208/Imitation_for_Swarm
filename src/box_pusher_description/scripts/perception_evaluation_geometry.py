#!/usr/bin/env python3
"""Geometry helpers for quantitative RGB-D perception evaluation.

These helpers use Gazebo poses only as evaluator ground truth. No function is
used by the orange perception policy/input path.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError('Invalid quaternion')
    x, y, z, w = q / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def pose_to_rt(position, orientation) -> Tuple[np.ndarray, np.ndarray]:
    r = quaternion_to_matrix(orientation.x, orientation.y, orientation.z, orientation.w)
    t = np.array([position.x, position.y, position.z], dtype=np.float64)
    return r, t


def box_corners_world(rotation_world_box: np.ndarray, center_world: np.ndarray,
                      dimensions_xyz: Sequence[float]) -> np.ndarray:
    r = np.asarray(rotation_world_box, dtype=np.float64).reshape(3, 3)
    c = np.asarray(center_world, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(dimensions_xyz, dtype=np.float64).reshape(3)
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    return (r @ (signs * half).T).T + c


def ray_oriented_box_intersection(ray_origin_world: np.ndarray,
                                  ray_direction_world: np.ndarray,
                                  box_rotation_world: np.ndarray,
                                  box_center_world: np.ndarray,
                                  box_dimensions_xyz: Sequence[float]
                                  ) -> Optional[Tuple[float, np.ndarray]]:
    origin = np.asarray(ray_origin_world, dtype=np.float64).reshape(3)
    direction = np.asarray(ray_direction_world, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(direction))
    if not math.isfinite(n) or n < 1e-12:
        return None
    direction = direction / n
    r = np.asarray(box_rotation_world, dtype=np.float64).reshape(3, 3)
    center = np.asarray(box_center_world, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(box_dimensions_xyz, dtype=np.float64).reshape(3)
    local_o = r.T @ (origin - center)
    local_d = r.T @ direction
    t_near, t_far = -math.inf, math.inf
    for axis in range(3):
        o, d, h = float(local_o[axis]), float(local_d[axis]), float(half[axis])
        if abs(d) < 1e-12:
            if o < -h or o > h:
                return None
            continue
        a, b = (-h - o) / d, (h - o) / d
        t_near = max(t_near, min(a, b))
        t_far = min(t_far, max(a, b))
        if t_near > t_far:
            return None
    if t_far < 0.0:
        return None
    distance = t_near if t_near >= 0.0 else t_far
    hit = origin + distance * direction
    return float(distance), hit


# Corner indexing from box_corners_world: sign order is x, then y, then z.
_BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


def _project_rect_point(point_rect: np.ndarray, projection: np.ndarray) -> Optional[Tuple[float, float]]:
    p = np.asarray(point_rect, dtype=np.float64).reshape(3)
    pmat = np.asarray(projection, dtype=np.float64).reshape(3, 4)
    if p[2] <= 1e-6:
        return None
    h = pmat @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    if not np.all(np.isfinite(h)) or abs(float(h[2])) < 1e-12:
        return None
    return float(h[0] / h[2]), float(h[1] / h[2])


def project_obb_bbox(box_rotation_world: np.ndarray, box_center_world: np.ndarray,
                     dimensions_xyz: Sequence[float], world_rotation_camera_raw: np.ndarray,
                     world_translation_camera: np.ndarray, rectification_rotation: np.ndarray,
                     projection: np.ndarray, image_width: int, image_height: int,
                     near_plane_m: float = 0.03
                     ) -> Optional[Tuple[float, float, float, float]]:
    """Project an oriented 3-D box to a clipped axis-aligned rectified-image bbox."""
    corners = box_corners_world(box_rotation_world, box_center_world, dimensions_xyz)
    r_wc = np.asarray(world_rotation_camera_raw, dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(world_translation_camera, dtype=np.float64).reshape(3)
    r1 = np.asarray(rectification_rotation, dtype=np.float64).reshape(3, 3)
    rect_points = [r1 @ (r_wc.T @ (p - t_wc)) for p in corners]
    clipped_points: List[np.ndarray] = [p for p in rect_points if p[2] >= near_plane_m]
    for ia, ib in _BOX_EDGES:
        a, b = rect_points[ia], rect_points[ib]
        za, zb = float(a[2]), float(b[2])
        if (za < near_plane_m and zb >= near_plane_m) or (zb < near_plane_m and za >= near_plane_m):
            denom = zb - za
            if abs(denom) > 1e-12:
                alpha = (near_plane_m - za) / denom
                clipped_points.append(a + alpha * (b - a))
    pixels = []
    for p in clipped_points:
        uv = _project_rect_point(p, projection)
        if uv is not None and math.isfinite(uv[0]) and math.isfinite(uv[1]):
            pixels.append(uv)
    if len(pixels) < 2:
        return None
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    x0 = max(0.0, min(float(image_width - 1), min(xs)))
    y0 = max(0.0, min(float(image_height - 1), min(ys)))
    x1 = max(0.0, min(float(image_width), max(xs)))
    y1 = max(0.0, min(float(image_height), max(ys)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def rectified_pixel_ray_world(u: float, v: float, projection: np.ndarray,
                              rectification_rotation: np.ndarray,
                              world_rotation_camera_raw: np.ndarray
                              ) -> np.ndarray:
    p = np.asarray(projection, dtype=np.float64).reshape(3, 4)
    r1 = np.asarray(rectification_rotation, dtype=np.float64).reshape(3, 3)
    r_wc = np.asarray(world_rotation_camera_raw, dtype=np.float64).reshape(3, 3)
    fx, fy = float(p[0, 0]), float(p[1, 1])
    cx, cy = float(p[0, 2]), float(p[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('Invalid rectified focal length')
    ray_rect = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)
    ray_raw = r1.T @ ray_rect
    ray_world = r_wc @ ray_raw
    n = float(np.linalg.norm(ray_world))
    if n < 1e-12:
        raise ValueError('Invalid pixel ray')
    return ray_world / n


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(x) for x in a]
    bx, by, bw, bh = [float(x) for x in b]
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return float(inter / union) if union > 1e-12 else 0.0


def greedy_iou_match(detections: List[dict], ground_truth: List[dict],
                     iou_threshold: float = 0.10) -> Tuple[List[dict], List[int], List[int]]:
    candidates = []
    for di, det in enumerate(detections):
        for gi, gt in enumerate(ground_truth):
            score = bbox_iou(det['bbox_xywh'], gt['bbox_xywh'])
            if score >= iou_threshold:
                candidates.append((score, di, gi))
    candidates.sort(reverse=True)
    used_d, used_g, matches = set(), set(), []
    for score, di, gi in candidates:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        matches.append({'detection_index': di, 'ground_truth_index': gi, 'iou': float(score)})
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    unmatched_g = [i for i in range(len(ground_truth)) if i not in used_g]
    return matches, unmatched_d, unmatched_g


def nearest_box_hit(origin: np.ndarray, direction: np.ndarray,
                    boxes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
                    ) -> Optional[Tuple[str, float, np.ndarray]]:
    nearest = None
    for name, (rotation, center, dimensions) in boxes.items():
        hit = ray_oriented_box_intersection(origin, direction, rotation, center, dimensions)
        if hit is None:
            continue
        distance, point = hit
        if nearest is None or distance < nearest[1]:
            nearest = (name, distance, point)
    return nearest


def visible_fraction_for_box(name: str, bbox: Sequence[float],
                             camera_origin_world: np.ndarray,
                             world_rotation_camera_raw: np.ndarray,
                             rectification_rotation: np.ndarray,
                             projection: np.ndarray,
                             boxes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                             grid_size: int = 3) -> Tuple[float, int, int]:
    x, y, w, h = [float(v) for v in bbox]
    total_hits, target_hits = 0, 0
    n = max(2, int(grid_size))
    for gy in range(n):
        for gx in range(n):
            u = x + (gx + 0.5) * w / n
            v = y + (gy + 0.5) * h / n
            ray = rectified_pixel_ray_world(u, v, projection, rectification_rotation,
                                            world_rotation_camera_raw)
            nearest = nearest_box_hit(camera_origin_world, ray, boxes)
            if nearest is not None:
                total_hits += 1
                if nearest[0] == name:
                    target_hits += 1
    fraction = target_hits / total_hits if total_hits else 0.0
    return float(fraction), target_hits, total_hits


def visible_surface_centroid_for_bbox(name: str, bbox: Sequence[float],
                                      camera_origin_world: np.ndarray,
                                      world_rotation_camera_raw: np.ndarray,
                                      rectification_rotation: np.ndarray,
                                      projection: np.ndarray,
                                      boxes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                      grid_size: int = 9) -> Optional[np.ndarray]:
    """Median GT surface point in rectified camera coords within a detection bbox."""
    x, y, w, h = [float(v) for v in bbox]
    r_wc = np.asarray(world_rotation_camera_raw, dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(camera_origin_world, dtype=np.float64).reshape(3)
    r1 = np.asarray(rectification_rotation, dtype=np.float64).reshape(3, 3)
    points_rect = []
    n = max(2, int(grid_size))
    for gy in range(n):
        for gx in range(n):
            u = x + (gx + 0.5) * w / n
            v = y + (gy + 0.5) * h / n
            ray = rectified_pixel_ray_world(u, v, projection, r1, r_wc)
            nearest = nearest_box_hit(t_wc, ray, boxes)
            if nearest is None or nearest[0] != name:
                continue
            hit_world = nearest[2]
            point_raw = r_wc.T @ (hit_world - t_wc)
            point_rect = r1 @ point_raw
            if np.all(np.isfinite(point_rect)) and point_rect[2] > 0.0:
                points_rect.append(point_rect)
    if len(points_rect) < 3:
        return None
    return np.median(np.asarray(points_rect, dtype=np.float64), axis=0)
