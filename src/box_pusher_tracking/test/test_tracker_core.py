import math
import unittest
import numpy as np

from box_pusher_tracking.geometry import transform_point, quaternion_slerp
from box_pusher_tracking.tracker_core import MultiObjectTracker, hungarian_assignment


class TestGeometry(unittest.TestCase):
    def test_transform_point_translation_only(self):
        result = transform_point((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (0.5, -0.5, 0.2))
        np.testing.assert_allclose(result, (1.5, 1.5, 3.2), atol=1e-9)

    def test_transform_point_quarter_turn_z(self):
        q = (0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4))
        result = transform_point((0.0, 0.0, 0.0), q, (1.0, 0.0, 0.0))
        np.testing.assert_allclose(result, (0.0, 1.0, 0.0), atol=1e-8)

    def test_quaternion_slerp_identity_to_half_turn(self):
        q0 = (0.0, 0.0, 0.0, 1.0)
        q1 = (0.0, 0.0, 1.0, 0.0)
        mid = quaternion_slerp(q0, q1, 0.5)
        self.assertAlmostEqual(abs(mid[2]), math.sqrt(0.5), places=6)
        self.assertAlmostEqual(abs(mid[3]), math.sqrt(0.5), places=6)


class TestAssignment(unittest.TestCase):
    def test_hungarian_global_assignment(self):
        cost = np.array([[4.0, 1.0, 9.0], [2.0, 7.0, 8.0]])
        pairs = hungarian_assignment(cost)
        chosen = {(r, c) for r, c in pairs}
        self.assertIn((0, 1), chosen)
        self.assertIn((1, 0), chosen)

    def test_hungarian_rejects_more_rows_than_columns(self):
        with self.assertRaises(ValueError):
            hungarian_assignment(np.zeros((3, 2)))


class TestMultiObjectTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = MultiObjectTracker(
            gate_distance_m=0.35,
            max_age_sec=0.5,
            min_hits=2,
            measurement_sigma_m=0.03,
        )

    def test_track_id_persists_under_small_motion(self):
        t0 = 1_000_000_000
        a = self.tracker.update([(1.0, 0.0, 0.05)], t0)
        first_id = a['tracks'][0]['id']
        b = self.tracker.update([(1.03, 0.0, 0.05)], t0 + 100_000_000)
        self.assertEqual(b['association_matches'], 1)
        self.assertEqual(b['tracks'][0]['id'], first_id)
        self.assertEqual(b['tracks'][0]['status'], 'confirmed')

    def test_two_tracks_do_not_swap_on_crossing_nearby_paths(self):
        t0 = 2_000_000_000
        self.tracker.update([(0.0, -0.08, 0.05), (0.0, 0.08, 0.05)], t0)
        self.tracker.update([(0.04, -0.08, 0.05), (0.04, 0.08, 0.05)], t0 + 100_000_000)
        tracks = {t['id']: t for t in self.tracker.update(
            [(0.08, -0.08, 0.05), (0.08, 0.08, 0.05)], t0 + 200_000_000
        )['tracks']}
        self.assertLess(tracks[1]['position_odom_m'][1], 0.0)
        self.assertGreater(tracks[2]['position_odom_m'][1], 0.0)

    def test_missed_track_coasts_then_expires(self):
        t0 = 3_000_000_000
        self.tracker.update([(0.5, 0.5, 0.05)], t0)
        self.tracker.update([(0.5, 0.5, 0.05)], t0 + 100_000_000)
        miss = self.tracker.update([], t0 + 200_000_000)
        self.assertEqual(miss['coasting_tracks'], 1)
        expired = self.tracker.update([], t0 + 900_000_000)
        self.assertEqual(expired['active_tracks'], 0)
        self.assertEqual(expired['deleted_tracks_total'], 1)

    def test_out_of_gate_detection_creates_new_track(self):
        t0 = 4_000_000_000
        self.tracker.update([(0.0, 0.0, 0.05)], t0)
        result = self.tracker.update([(1.0, 0.0, 0.05)], t0 + 100_000_000)
        self.assertEqual(result['active_tracks'], 2)
        self.assertEqual(result['association_matches'], 0)

    def test_non_finite_detection_rejected(self):
        with self.assertRaises(ValueError):
            self.tracker.update([(float('nan'), 0.0, 0.0)], 5_000_000_000)


if __name__ == '__main__':
    unittest.main()
