import math
import unittest
import numpy as np

from box_pusher_geometry.geometry import (
    backproject_rectified_pixels,
    base_to_odom,
    fit_visible_vertical_face,
    quaternion_to_matrix,
    wrap_square_yaw,
)


class GeometryTests(unittest.TestCase):
    def test_backprojection(self):
        pts = backproject_rectified_pixels(np.array([320.0]), np.array([240.0]),
                                           np.array([2.0]), 200.0, 200.0, 320.0, 240.0)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 2.0])

    def test_base_to_odom_translation_and_yaw(self):
        out = base_to_odom([1.0, 0.0, 0.1], 2.0, 3.0, 0.0, math.pi/2)
        np.testing.assert_allclose(out, [2.0, 4.0, 0.1], atol=1e-9)

    def test_square_yaw_modulo_ninety_degrees(self):
        self.assertAlmostEqual(wrap_square_yaw(math.radians(92.0)), math.radians(2.0), places=6)

    def test_quaternion_matrix_identity(self):
        np.testing.assert_allclose(quaternion_to_matrix([0, 0, 0, 1]), np.eye(3), atol=1e-12)

    def test_vertical_face_fit_known_cuboid(self):
        # Visible front face at x=1.0 m. Camera lies behind it at x=0.
        yy, zz = np.meshgrid(np.linspace(-0.04, 0.04, 17), np.linspace(0.0, 0.10, 19))
        xx = np.ones_like(yy)
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        fit = fit_visible_vertical_face(points, [0.0, 0.0, 0.05],
                                        box_xy_size_m=0.08, box_height_m=0.10,
                                        plane_threshold_m=0.002, min_points=18,
                                        ransac_iterations=100)
        self.assertTrue(fit.success, fit.reason)
        self.assertAlmostEqual(fit.geometric_center[0], 1.04, delta=0.012)
        self.assertAlmostEqual(fit.geometric_center[1], 0.0, delta=0.01)
        self.assertAlmostEqual(fit.geometric_center[2], 0.05, places=6)
        self.assertAlmostEqual(fit.observed_width_m, 0.08, delta=0.01)
        self.assertAlmostEqual(fit.observed_height_m, 0.10, delta=0.01)
        self.assertAlmostEqual(abs(fit.outward_normal[0]), 1.0, delta=0.02)

    def test_insufficient_points(self):
        fit = fit_visible_vertical_face(np.zeros((5, 3)), [0, 0, 0])
        self.assertFalse(fit.success)
        self.assertEqual(fit.reason, 'INSUFFICIENT_3D_POINTS')

    def test_rotated_face_reports_square_yaw_modulo_ninety(self):
        theta = math.radians(32.0)
        edge = np.array([math.cos(theta), math.sin(theta), 0.0])
        outward = np.array([-math.sin(theta), math.cos(theta), 0.0])
        yy, zz = np.meshgrid(np.linspace(-0.04, 0.04, 17), np.linspace(0.0, 0.10, 19))
        center = np.array([1.2, -0.3, 0.05])
        points = center + yy.ravel()[:, None] * edge + (zz.ravel()-0.05)[:, None] * np.array([0.0, 0.0, 1.0])
        camera = center + 0.8 * outward
        fit = fit_visible_vertical_face(points, camera, plane_threshold_m=0.002, min_points=18)
        self.assertTrue(fit.success, fit.reason)
        self.assertAlmostEqual(fit.yaw_edge_mod_pi_over_2, theta, delta=math.radians(2.0))
        self.assertGreater(float(np.dot(fit.outward_normal, outward)), 0.95)

    def test_plane_fit_rejects_scattered_depth_outliers(self):
        yy, zz = np.meshgrid(np.linspace(-0.04, 0.04, 17), np.linspace(0.0, 0.10, 19))
        face = np.column_stack((np.ones(yy.size), yy.ravel(), zz.ravel()))
        rng = np.random.default_rng(4)
        outliers = np.column_stack((rng.uniform(0.8, 1.3, 35),
                                    rng.uniform(-0.12, 0.12, 35),
                                    rng.uniform(-0.02, 0.14, 35)))
        points = np.vstack((face, outliers))
        fit = fit_visible_vertical_face(points, [0.0, 0.0, 0.05],
                                        plane_threshold_m=0.003, min_points=18,
                                        ransac_iterations=250)
        self.assertTrue(fit.success, fit.reason)
        self.assertGreater(fit.plane_inlier_ratio, 0.65)
        self.assertAlmostEqual(fit.geometric_center[0], 1.04, delta=0.015)


if __name__ == '__main__':
    unittest.main()
