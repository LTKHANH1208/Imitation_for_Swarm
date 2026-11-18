"""Small dependency-light 3-D rigid-transform helpers used by the tracker."""
from __future__ import annotations
from typing import Iterable, Tuple
import math


def normalize_quaternion(q: Iterable[float]) -> Tuple[float, float, float, float]:
    qx, qy, qz, qw = (float(v) for v in q)
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-12:
        raise ValueError("zero-norm quaternion")
    return qx/norm, qy/norm, qz/norm, qw/norm


def rotate_vector(q: Iterable[float], v: Iterable[float]) -> Tuple[float, float, float]:
    qx, qy, qz, qw = normalize_quaternion(q)
    vx, vy, vz = (float(x) for x in v)
    # q * [v,0] * conjugate(q), using the equivalent cross-product form.
    tx = 2.0 * (qy*vz - qz*vy)
    ty = 2.0 * (qz*vx - qx*vz)
    tz = 2.0 * (qx*vy - qy*vx)
    rx = vx + qw*tx + (qy*tz - qz*ty)
    ry = vy + qw*ty + (qz*tx - qx*tz)
    rz = vz + qw*tz + (qx*ty - qy*tx)
    return rx, ry, rz


def transform_point(translation: Iterable[float], quaternion: Iterable[float], point: Iterable[float]):
    rotated = rotate_vector(quaternion, point)
    tx, ty, tz = (float(v) for v in translation)
    return tx + rotated[0], ty + rotated[1], tz + rotated[2]


def quaternion_slerp(q0: Iterable[float], q1: Iterable[float], alpha: float):
    a = normalize_quaternion(q0)
    b = normalize_quaternion(q1)
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-x for x in b)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        vals = tuple((1.0-alpha)*x + alpha*y for x, y in zip(a, b))
        return normalize_quaternion(vals)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    wa = math.sin((1.0-alpha)*theta) / sin_theta
    wb = math.sin(alpha*theta) / sin_theta
    return normalize_quaternion(tuple(wa*x + wb*y for x, y in zip(a, b)))
