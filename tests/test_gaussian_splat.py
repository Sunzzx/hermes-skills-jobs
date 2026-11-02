"""Comprehensive Unit and Integration Test Suite for 3D Gaussian Splatting & Real-Time Radiance Field Engine."""

from importlib.machinery import SourceFileLoader
import math
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "gaussian-splat"

loader = SourceFileLoader("gaussian_splat", str(BIN_PATH))
gs_mod = types.ModuleType(loader.name)
loader.exec_module(gs_mod)

normalize_quaternion = gs_mod.normalize_quaternion
quat_to_rot_matrix = gs_mod.quat_to_rot_matrix
quat_from_axis_angle = gs_mod.quat_from_axis_angle
build_3d_covariance = gs_mod.build_3d_covariance
eval_sh_deg1 = gs_mod.eval_sh_deg1
Camera = gs_mod.Camera
GaussianCloud = gs_mod.GaussianCloud
GaussianRasterizer = gs_mod.GaussianRasterizer
GaussianFitter = gs_mod.GaussianFitter
TerminalRenderer = gs_mod.TerminalRenderer
generate_galaxy = gs_mod.generate_galaxy
generate_torus_knot = gs_mod.generate_torus_knot
generate_teapot = gs_mod.generate_teapot
generate_planetary = gs_mod.generate_planetary
run_verification_suite = gs_mod.run_verification_suite


class TestQuaternionAndRotation(unittest.TestCase):
    def test_quaternion_normalization(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_norm = normalize_quaternion(q)
        self.assertAlmostEqual(np.linalg.norm(q_norm), 1.0, places=9)
        self.assertAlmostEqual(q_norm[0], 1.0, places=9)

    def test_so3_orthogonality_and_determinant(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        R = quat_to_rot_matrix(q)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=7)
        ortho = R @ R.T
        np.testing.assert_allclose(ortho, np.eye(3), atol=1e-7)

    def test_axis_angle_construction(self):
        angle = math.pi * 0.5
        axis = np.array([0.0, 1.0, 0.0])
        q = quat_from_axis_angle(axis, angle)
        R = quat_to_rot_matrix(q)
        v = np.array([1.0, 0.0, 0.0])
        v_rot = R @ v
        np.testing.assert_allclose(v_rot, np.array([0.0, 0.0, -1.0]), atol=1e-6)


class Test3DCovariance(unittest.TestCase):
    def test_covariance_symmetry(self):
        scales = np.array([[0.2, 0.1, 0.05], [0.4, 0.4, 0.1]])
        quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0]])
        cov = build_3d_covariance(scales, quats)
        self.assertEqual(cov.shape, (2, 3, 3))
        for i in range(2):
            np.testing.assert_allclose(cov[i], cov[i].T, atol=1e-9)

    def test_covariance_positive_semidefinite(self):
        scales = np.array([[0.3, 0.2, 0.1]])
        quats = np.array([[0.5, 0.5, 0.5, 0.5]])
        cov = build_3d_covariance(scales, quats)[0]
        eigvals = np.linalg.eigvalsh(cov)
        self.assertTrue(np.all(eigvals >= -1e-12))
        np.testing.assert_allclose(np.sort(eigvals), np.sort(scales[0] ** 2), atol=1e-6)


class TestCameraModel(unittest.TestCase):
    def test_camera_intrinsics_and_focal_length(self):
        cam = Camera(width=80, height=40, fov_deg=60.0)
        expected_fx = (40.0) / math.tan(math.radians(30.0))
        self.assertAlmostEqual(cam.focal_x, expected_fx, places=5)
        self.assertEqual(cam.cx, 40.0)
        self.assertEqual(cam.cy, 20.0)

    def test_view_matrix_eye_to_origin(self):
        cam = Camera(position=np.array([0.0, 0.0, 5.0]), target=np.array([0.0, 0.0, 0.0]))
        R_cam, t_cam = cam.get_view_matrix()
        p_cam = R_cam @ cam.position + t_cam
        np.testing.assert_allclose(p_cam, np.zeros(3), atol=1e-7)

    def test_camera_orbit_generation(self):
        cam = Camera.create_orbit(radius=4.0, azimuth_deg=90.0, elevation_deg=0.0)
        self.assertAlmostEqual(cam.position[0], 4.0, places=5)
        self.assertAlmostEqual(cam.position[1], 0.0, places=5)
        self.assertAlmostEqual(cam.position[2], 0.0, places=5)


class TestSphericalHarmonics(unittest.TestCase):
    def test_dc_constant_sh0(self):
        sh = np.zeros((1, 4, 3))
        sh[0, 0, :] = 0.5 / gs_mod.C0
        dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        colors = eval_sh_deg1(np.repeat(sh, 2, axis=0), dirs)
        np.testing.assert_allclose(colors[0], [1.0, 1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(colors[1], [1.0, 1.0, 1.0], atol=1e-5)

    def test_degree1_view_dependency(self):
        sh = np.zeros((1, 4, 3))
        sh[0, 1, 1] = 1.0  # Y component on green
        dir_up = np.array([[0.0, 1.0, 0.0]])
        dir_down = np.array([[0.0, -1.0, 0.0]])
        c_up = eval_sh_deg1(sh, dir_up)[0, 1]
        c_down = eval_sh_deg1(sh, dir_down)[0, 1]
        self.assertGreater(c_up, c_down)


class TestRasterizationAndAlphaCompositing(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(width=40, height=20, position=np.array([0.0, 0.0, 3.0]))

    def test_project_gaussians_screen_bounds(self):
        cloud = GaussianCloud(
            positions=np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]]),  # Second is offscreen
            scales=np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            opacities=np.array([0.8, 0.8])
        )
        proj = GaussianRasterizer.project_gaussians(cloud, self.camera)
        self.assertTrue(proj["valid"])
        self.assertEqual(len(proj["x_px"]), 1)
        self.assertAlmostEqual(proj["x_px"][0], self.camera.cx, places=1)
        self.assertAlmostEqual(proj["y_px"][0], self.camera.cy, places=1)

    def test_render_output_properties(self):
        cloud = GaussianCloud(
            positions=np.array([[0.0, 0.0, 0.0]]),
            scales=np.array([[0.2, 0.2, 0.2]]),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            opacities=np.array([0.9]),
            colors=np.array([[0.8, 0.2, 0.1]])
        )
        out = GaussianRasterizer.render(cloud, self.camera)
        self.assertEqual(out.rgb.shape, (20, 40, 3))
        self.assertEqual(out.depth.shape, (20, 40))
        self.assertEqual(out.alpha.shape, (20, 40))
        self.assertEqual(out.visible_count, 1)
        self.assertGreater(out.render_time_ms, 0.0)
        center_px = out.rgb[self.camera.height // 2, self.camera.width // 2]
        self.assertGreater(center_px[0], 0.5)

    def test_depth_sorting_and_occlusion(self):
        # Front: red at z=0.5. Back: blue at z=0.0 (farther from camera at z=3.0)
        cloud = GaussianCloud(
            positions=np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]),
            scales=np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            opacities=np.array([0.7, 0.9]),
            colors=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )
        out = GaussianRasterizer.render(cloud, self.camera)
        center_px = out.rgb[self.camera.height // 2, self.camera.width // 2]
        self.assertGreater(center_px[0], center_px[2])


class TestDifferentiableOptimizer(unittest.TestCase):
    def test_gradient_step_decreases_error(self):
        camera = Camera(width=30, height=15, position=np.array([0.0, 0.0, 3.0]))
        target_img = np.ones((15, 30, 3), dtype=np.float64) * 0.9
        cloud = GaussianCloud(
            positions=np.array([[0.0, 0.0, 0.0]]),
            scales=np.array([[0.3, 0.3, 0.3]]),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            opacities=np.array([0.8]),
            colors=np.array([[0.1, 0.1, 0.1]])
        )
        l_init, _ = GaussianFitter.fit_step(cloud, target_img, camera, lr=0.0)
        l_final = l_init
        for _ in range(5):
            l_final, _ = GaussianFitter.fit_step(cloud, target_img, camera, lr=0.2)
        self.assertLess(l_final, l_init)


class TestProceduralGeneratorsAndPLY(unittest.TestCase):
    def test_galaxy_generation(self):
        galaxy = generate_galaxy(n_stars=300)
        self.assertEqual(galaxy.n_gaussians, 300)
        self.assertEqual(galaxy.positions.shape, (300, 3))
        self.assertEqual(galaxy.colors.shape, (300, 3))

    def test_torus_generation(self):
        torus = generate_torus_knot(p=2, q=3, n_samples=250)
        self.assertEqual(torus.n_gaussians, 250)

    def test_teapot_generation(self):
        teapot = generate_teapot(n_points=200)
        self.assertEqual(teapot.n_gaussians, 200)

    def test_planetary_generation(self):
        planet = generate_planetary(n_points=200)
        self.assertEqual(planet.n_gaussians, 200)

    def test_ply_export_and_file_structure(self):
        cloud = GaussianCloud(
            positions=np.array([[1.0, 2.0, 3.0]]),
            scales=np.array([[0.1, 0.1, 0.1]]),
            rotations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            opacities=np.array([0.8]),
            colors=np.array([[0.5, 0.5, 0.5]])
        )
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply_path = f.name
        try:
            cloud.export_ply(ply_path)
            self.assertTrue(os.path.exists(ply_path))
            with open(ply_path) as f:
                content = f.read()
            self.assertTrue(content.startswith("ply\nformat ascii 1.0"))
            self.assertIn("element vertex 1", content)
            self.assertIn("end_header", content)
        finally:
            if os.path.exists(ply_path):
                os.remove(ply_path)


class TestFormalVerificationSuite(unittest.TestCase):
    def test_formal_invariants_pass_100_percent(self):
        self.assertTrue(run_verification_suite())


if __name__ == '__main__':
    unittest.main()
