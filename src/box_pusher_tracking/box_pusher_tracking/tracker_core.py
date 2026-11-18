"""Lightweight 2-D constant-velocity Kalman tracker and Hungarian assignment.

The tracker operates on XY positions expressed in a stable odometry frame.
Input measurements are visible-surface centroids from the RGB-D detector, not
necessarily the geometric centres of complete boxes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import math
import numpy as np


@dataclass
class Track:
    track_id: int
    state: np.ndarray                 # [x, y, vx, vy]
    covariance: np.ndarray            # 4x4
    z: float
    first_stamp_ns: int
    last_stamp_ns: int
    hits: int = 1
    missed: int = 0
    last_measurement_stamp_ns: int = 0
    status: str = "tentative"

    def as_dict(self, now_ns: int) -> Dict[str, object]:
        return {
            "id": self.track_id,
            "status": self.status,
            "position_odom_m": [float(self.state[0]), float(self.state[1]), float(self.z)],
            "velocity_odom_mps": [float(self.state[2]), float(self.state[3])],
            "age_sec": max(0.0, (now_ns - self.first_stamp_ns) * 1e-9),
            "time_since_seen_sec": max(0.0, (now_ns - self.last_measurement_stamp_ns) * 1e-9),
            "hits": self.hits,
            "missed_frames": self.missed,
        }


def hungarian_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Minimum-cost assignment for a rectangular matrix with rows <= cols.

    Returns (row, column) pairs. This is the O(n^2 m) potentials algorithm.
    The tracker always appends dummy columns, so its matrix satisfies n <= m.
    """
    a = np.asarray(cost, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    n, m = a.shape
    if n == 0 or m == 0:
        return []
    if n > m:
        raise ValueError("hungarian_assignment requires rows <= columns")
    if not np.all(np.isfinite(a)):
        raise ValueError("cost matrix must contain only finite values")

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result: List[Tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            result.append((int(p[j] - 1), int(j - 1)))
    return result


class MultiObjectTracker:
    def __init__(
        self,
        gate_distance_m: float = 0.35,
        max_age_sec: float = 1.25,
        min_hits: int = 2,
        measurement_sigma_m: float = 0.045,
        acceleration_noise: float = 0.8,
        initial_velocity_sigma_mps: float = 0.5,
    ) -> None:
        if gate_distance_m <= 0 or max_age_sec <= 0 or min_hits < 1:
            raise ValueError("invalid tracker configuration")
        self.gate_distance_m = float(gate_distance_m)
        self.max_age_sec = float(max_age_sec)
        self.min_hits = int(min_hits)
        self.measurement_sigma_m = float(measurement_sigma_m)
        self.acceleration_noise = float(acceleration_noise)
        self.initial_velocity_sigma_mps = float(initial_velocity_sigma_mps)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.created_total = 0
        self.deleted_total = 0

    def _predict_track(self, track: Track, stamp_ns: int) -> None:
        dt = max(0.0, min(1.0, (stamp_ns - track.last_stamp_ns) * 1e-9))
        if dt <= 0.0:
            return
        f = np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
        )
        q = self.acceleration_noise ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        process = q * np.array(
            [[dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
             [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
             [dt3 / 2.0, 0.0, dt2, 0.0],
             [0.0, dt3 / 2.0, 0.0, dt2]], dtype=np.float64
        )
        track.state = f @ track.state
        track.covariance = f @ track.covariance @ f.T + process
        track.last_stamp_ns = stamp_ns

    def _new_track(self, xyz: Sequence[float], stamp_ns: int) -> Track:
        state = np.array([xyz[0], xyz[1], 0.0, 0.0], dtype=np.float64)
        covariance = np.diag([
            self.measurement_sigma_m ** 2,
            self.measurement_sigma_m ** 2,
            self.initial_velocity_sigma_mps ** 2,
            self.initial_velocity_sigma_mps ** 2,
        ]).astype(np.float64)
        track = Track(
            track_id=self.next_id,
            state=state,
            covariance=covariance,
            z=float(xyz[2]),
            first_stamp_ns=stamp_ns,
            last_stamp_ns=stamp_ns,
            hits=1,
            missed=0,
            last_measurement_stamp_ns=stamp_ns,
            status="tentative" if self.min_hits > 1 else "confirmed",
        )
        self.tracks[track.track_id] = track
        self.next_id += 1
        self.created_total += 1
        return track

    def _update_track(self, track: Track, xyz: Sequence[float], stamp_ns: int) -> None:
        h = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        r = np.eye(2, dtype=np.float64) * (self.measurement_sigma_m ** 2)
        z_meas = np.array([xyz[0], xyz[1]], dtype=np.float64)
        innovation = z_meas - h @ track.state
        s = h @ track.covariance @ h.T + r
        k = track.covariance @ h.T @ np.linalg.inv(s)
        track.state = track.state + k @ innovation
        identity = np.eye(4, dtype=np.float64)
        # Joseph form keeps covariance symmetric and positive semidefinite.
        kh = identity - k @ h
        track.covariance = kh @ track.covariance @ kh.T + k @ r @ k.T
        track.z = 0.65 * track.z + 0.35 * float(xyz[2])
        track.hits += 1
        track.missed = 0
        track.last_measurement_stamp_ns = stamp_ns
        track.status = "confirmed" if track.hits >= self.min_hits else "tentative"

    def update(self, detections_xyz: Sequence[Sequence[float]], stamp_ns: int) -> Dict[str, object]:
        detections = [tuple(float(v) for v in xyz[:3]) for xyz in detections_xyz]
        for xyz in detections:
            if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
                raise ValueError("each detection must contain three finite coordinates")

        # Predict all tracks to this frame timestamp.
        track_ids = sorted(self.tracks)
        for track_id in track_ids:
            self._predict_track(self.tracks[track_id], stamp_ns)

        matches: List[Tuple[int, int]] = []
        assigned_detection_indices = set()
        if track_ids:
            n_tracks = len(track_ids)
            n_dets = len(detections)
            gate2 = self.gate_distance_m ** 2
            # Each track gets a private-cost option among dummy columns; unmatched
            # detections become new tracks after assignment.
            costs = np.full((n_tracks, n_dets + n_tracks), gate2 + 1e3, dtype=np.float64)
            for row, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                for col, det in enumerate(detections):
                    d2 = (track.state[0] - det[0]) ** 2 + (track.state[1] - det[1]) ** 2
                    if d2 <= gate2:
                        costs[row, col] = d2
                costs[row, n_dets + row] = gate2
            assignments = hungarian_assignment(costs)
            for row, col in assignments:
                if col < n_dets and costs[row, col] <= gate2:
                    track_id = track_ids[row]
                    self._update_track(self.tracks[track_id], detections[col], stamp_ns)
                    matches.append((track_id, col))
                    assigned_detection_indices.add(col)
                else:
                    track = self.tracks[track_ids[row]]
                    track.missed += 1
                    track.status = "coasting" if track.hits >= self.min_hits else "tentative"

        for det_index, det in enumerate(detections):
            if det_index not in assigned_detection_indices:
                self._new_track(det, stamp_ns)

        expired = []
        for track_id, track in self.tracks.items():
            since_seen = (stamp_ns - track.last_measurement_stamp_ns) * 1e-9
            if since_seen > self.max_age_sec:
                expired.append(track_id)
        for track_id in expired:
            del self.tracks[track_id]
            self.deleted_total += 1

        confirmed = sum(1 for t in self.tracks.values() if t.hits >= self.min_hits)
        return {
            "input_detections": len(detections),
            "association_matches": len(matches),
            "new_tracks_this_frame": len(detections) - len(assigned_detection_indices),
            "active_tracks": len(self.tracks),
            "confirmed_tracks": confirmed,
            "tentative_tracks": len(self.tracks) - confirmed,
            "coasting_tracks": sum(1 for t in self.tracks.values() if t.status == "coasting"),
            "created_tracks_total": self.created_total,
            "deleted_tracks_total": self.deleted_total,
            "tracks": [self.tracks[k].as_dict(stamp_ns) for k in sorted(self.tracks)],
        }
