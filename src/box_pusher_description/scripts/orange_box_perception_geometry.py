#!/usr/bin/env python3
"""Pure OpenCV/NumPy helpers for V2 orange-box RGB-D perception."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def make_orange_mask(
    bgr: np.ndarray,
    hue_min: int = 3,
    hue_max: int = 30,
    saturation_min: int = 100,
    value_min: int = 60,
    morph_kernel: int = 3,
) -> np.ndarray:
    """Return a uint8 mask for orange pixels in an OpenCV BGR image."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Expected a HxWx3 BGR image')
    if not (0 <= hue_min <= 179 and 0 <= hue_max <= 179):
        raise ValueError('OpenCV hue bounds must be in [0, 179]')
    if saturation_min < 0 or saturation_min > 255 or value_min < 0 or value_min > 255:
        raise ValueError('Saturation/value thresholds must be in [0,255]')
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if hue_min <= hue_max:
        mask = cv2.inRange(
            hsv,
            np.array([hue_min, saturation_min, value_min], dtype=np.uint8),
            np.array([hue_max, 255, 255], dtype=np.uint8),
        )
    else:
        # Support hue intervals crossing OpenCV's 179 -> 0 boundary.
        low = cv2.inRange(
            hsv,
            np.array([hue_min, saturation_min, value_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        high = cv2.inRange(
            hsv,
            np.array([0, saturation_min, value_min], dtype=np.uint8),
            np.array([hue_max, 255, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(low, high)

    k = max(1, int(morph_kernel))
    if k % 2 == 0:
        k += 1
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def find_orange_contours(
    mask: np.ndarray,
    min_area_px: float = 120.0,
    min_width_px: int = 5,
    min_height_px: int = 5,
    min_extent: float = 0.20,
    max_aspect_ratio: float = 4.0,
) -> List[Dict[str, object]]:
    """Find plausible orange object blobs and return sorted bounding regions."""
    if mask is None or mask.ndim != 2:
        raise ValueError('Expected a 2D binary mask')
    result = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[-2]
    candidates: List[Dict[str, object]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area_px):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < int(min_width_px) or h < int(min_height_px):
            continue
        extent = area / float(max(1, w * h))
        aspect = w / float(max(1, h))
        if extent < float(min_extent) or aspect > float(max_aspect_ratio) or aspect < 1.0 / float(max_aspect_ratio):
            continue
        candidates.append({
            'contour': contour,
            'area_px': area,
            'bbox': (int(x), int(y), int(w), int(h)),
            'extent': extent,
            'aspect_ratio': aspect,
        })
    candidates.sort(key=lambda item: float(item['area_px']), reverse=True)
    return candidates


def robust_depth_samples(
    depth: np.ndarray,
    region_mask: np.ndarray,
    min_depth_m: float = 0.20,
    max_depth_m: float = 5.0,
    min_valid_pixels: int = 8,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Return valid depth values and robust summary statistics."""
    if depth.ndim != 2 or region_mask.ndim != 2 or depth.shape != region_mask.shape:
        raise ValueError('Depth and region mask must be equal-sized 2D arrays')
    selected = depth[(region_mask > 0)]
    total = int(selected.size)
    valid = selected[np.isfinite(selected) & (selected >= min_depth_m) & (selected <= max_depth_m)]
    if valid.size < int(min_valid_pixels):
        return None, {
            'total_pixels': float(total),
            'valid_pixels': float(valid.size),
            'valid_ratio': float(valid.size / max(1, total)),
            'median_m': float('nan'),
            'mad_m': float('nan'),
        }
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    # Reject isolated stereo mismatches while retaining some natural depth spread.
    robust_limit = max(0.025, 3.5 * 1.4826 * mad)
    inliers = valid[np.abs(valid - median) <= robust_limit]
    if inliers.size >= int(min_valid_pixels):
        valid = inliers
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
    return valid.astype(np.float32), {
        'total_pixels': float(total),
        'valid_pixels': float(valid.size),
        'valid_ratio': float(valid.size / max(1, total)),
        'median_m': median,
        'mad_m': mad,
    }


def project_depth_region_to_xyz(
    depth: np.ndarray,
    region_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth_m: float = 0.20,
    max_depth_m: float = 5.0,
    min_valid_pixels: int = 8,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Project valid pixels to camera XYZ and return robust visible-surface centroid.

    Coordinates follow the ROS optical-frame convention: X right, Y down, Z forward.
    This is the centroid of the visible surface, not the hidden geometric centre
    of the full 3D box.
    """
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError('fx and fy must be positive')
    if depth.shape != region_mask.shape:
        raise ValueError('Depth and region mask dimensions differ')
    ys, xs = np.nonzero(region_mask > 0)
    if xs.size == 0:
        return None, {'total_pixels': 0.0, 'valid_pixels': 0.0, 'valid_ratio': 0.0,
                      'median_m': float('nan'), 'mad_m': float('nan')}
    z_all = depth[ys, xs].astype(np.float32)
    valid = np.isfinite(z_all) & (z_all >= min_depth_m) & (z_all <= max_depth_m)
    if int(np.count_nonzero(valid)) < int(min_valid_pixels):
        return None, {
            'total_pixels': float(xs.size),
            'valid_pixels': float(np.count_nonzero(valid)),
            'valid_ratio': float(np.count_nonzero(valid) / max(1, xs.size)),
            'median_m': float('nan'), 'mad_m': float('nan'),
        }
    xs_v = xs[valid].astype(np.float32)
    ys_v = ys[valid].astype(np.float32)
    zs_v = z_all[valid]
    median_z = float(np.median(zs_v))
    mad_z = float(np.median(np.abs(zs_v - median_z)))
    limit = max(0.025, 3.5 * 1.4826 * mad_z)
    inlier = np.abs(zs_v - median_z) <= limit
    if int(np.count_nonzero(inlier)) >= int(min_valid_pixels):
        xs_v, ys_v, zs_v = xs_v[inlier], ys_v[inlier], zs_v[inlier]
    x3 = (xs_v - float(cx)) * zs_v / float(fx)
    y3 = (ys_v - float(cy)) * zs_v / float(fy)
    points = np.column_stack((x3, y3, zs_v)).astype(np.float64)
    # Coordinate-wise median is robust to a small number of mismatched pixels.
    centroid = np.median(points, axis=0)
    stats = {
        'total_pixels': float(xs.size),
        'valid_pixels': float(zs_v.size),
        'valid_ratio': float(zs_v.size / max(1, xs.size)),
        'median_m': float(np.median(zs_v)),
        'mad_m': float(np.median(np.abs(zs_v - np.median(zs_v)))),
    }
    return centroid, stats


def transform_point_xyz(point: Sequence[float], translation: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Apply a TransformStamped transform to a 3D point."""
    px, py, pz = [float(v) for v in point]
    tx, ty, tz = [float(v) for v in translation]
    qx, qy, qz, qw = [float(v) for v in quaternion_xyzw]
    norm = float(np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw))
    if norm < 1e-12:
        raise ValueError('Invalid zero-norm quaternion')
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    rotation = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return rotation @ np.array([px, py, pz], dtype=np.float64) + np.array([tx, ty, tz], dtype=np.float64)
