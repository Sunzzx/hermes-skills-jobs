"""
Unit and Integration Tests for Kerr Relativistic Geodesic Raytracer & Black Hole Engine.
"""

import math
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "kerr-sim"

# Load module dynamically
kerr_globals = {}
with open(CLI, "r", encoding="utf-8") as f:
    exec(f.read(), kerr_globals)

KerrSpacetime = kerr_globals["KerrSpacetime"]
KerrVisualizer = kerr_globals["KerrVisualizer"]
verify_kerr_invariants = kerr_globals["verify_kerr_invariants"]
run_benchmark = kerr_globals["run_benchmark"]


class TestKerrGeodesicEngine(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_kerr_spacetime_geometry(self):
        # Schwarzschild limit (a = 0)
        sp0 = KerrSpacetime(a=0.0)
        self.assertAlmostEqual(sp0.r_plus, 2.0, places=4)
        self.assertAlmostEqual(sp0.r_minus, 0.0, places=4)
        self.assertAlmostEqual(sp0.r_isco, 6.0, places=4)
        self.assertAlmostEqual(sp0.r_ph_pro, 3.0, places=4)
        self.assertAlmostEqual(sp0.r_ph_retro, 3.0, places=4)

        # High-spin Kerr (a = 0.998)
        sp_ext = KerrSpacetime(a=0.998)
        expected_r_plus = 1.0 + math.sqrt(1.0 - 0.998**2)
        self.assertAlmostEqual(sp_ext.r_plus, expected_r_plus, places=4)
        self.assertLess(sp_ext.r_isco, 1.30)
        self.assertLess(sp_ext.r_ph_pro, 1.15)
        self.assertGreater(sp_ext.r_ph_retro, 3.90)

        # Ergosphere geometry
        sp9 = KerrSpacetime(a=0.9)
        self.assertAlmostEqual(sp9.ergosphere_radius(0.0), sp9.r_plus, places=4)
        self.assertAlmostEqual(sp9.ergosphere_radius(math.pi * 0.5), 2.0, places=4)

    def test_bardeen_analytical_shadow(self):
        # a = 0 gives circle of radius sqrt(27)*M ~ 5.196
        sp0 = KerrSpacetime(a=0.0)
        alp0, bet0 = sp0.analytical_shadow(math.radians(90.0), num_points=100)
        radii = np.sqrt(alp0**2 + bet0**2)
        expected_R = math.sqrt(27.0)
        self.assertTrue(np.allclose(radii, expected_R, rtol=1e-3))

        # a = 0.9 gives asymmetric shadow
        sp9 = KerrSpacetime(a=0.9)
        alp9, bet9 = sp9.analytical_shadow(math.radians(85.0), num_points=400)
        alp_min = float(np.min(alp9))
        alp_max = float(np.max(alp9))
        self.assertLess(alp_min, -2.6)
        self.assertGreater(alp_min, -3.0)
        self.assertGreater(alp_max, 6.5)
        self.assertLess(alp_max, 7.0)
        # Centroid shift from frame dragging
        centroid = (alp_max + alp_min) / 2.0
        self.assertGreater(centroid, 1.8)

    def test_carter_quartic_and_shadow_classification(self):
        sp = KerrSpacetime(a=0.9)
        theta0 = math.radians(85.0)
        sin_th = math.sin(theta0)
        cos_th = math.cos(theta0)

        # Center of shadow alpha=0, beta=0
        xi0 = 0.0
        eta0 = -sp.a2 * cos_th**2
        roots0 = sp.solve_radial_roots(xi0, eta0)
        self.assertEqual(len(roots0), 0)  # Plunges directly into horizon

        # Far outside shadow alpha=10, beta=0
        xi_out = -10.0 * sin_th
        eta_out = (100.0 - sp.a2) * cos_th**2
        roots_out = sp.solve_radial_roots(xi_out, eta_out)
        self.assertGreater(len(roots_out), 0)  # Scatters at periapsis
        self.assertGreater(roots_out[-1], sp.r_plus)

    def test_raytracing_geodesics(self):
        sp = KerrSpacetime(a=0.9)
        theta0 = math.radians(85.0)

        # Ray inside shadow
        res_in = sp.trace_ray(0.0, 0.0, theta0=theta0, r0=30.0)
        self.assertTrue(res_in.in_shadow)
        self.assertLessEqual(res_in.r_min, sp.r_plus + 1e-2)

        # Ray outside shadow
        res_out = sp.trace_ray(10.0, 0.0, theta0=theta0, r0=30.0)
        self.assertFalse(res_out.in_shadow)
        self.assertGreater(res_out.r_min, sp.r_plus)

        # Disk crossing check
        res_disk = sp.trace_ray(-6.0, 1.0, theta0=theta0, r0=30.0, r_disk_out=15.0)
        self.assertTrue(res_disk.hit_disk)
        self.assertGreaterEqual(res_disk.disk_r, sp.r_isco)
        self.assertLessEqual(res_disk.disk_r, 15.0)

    def test_relativistic_doppler_and_beaming(self):
        sp = KerrSpacetime(a=0.9)
        theta0 = math.radians(85.0)

        ray_app = sp.trace_ray(-8.0, 1.0, theta0=theta0, r0=30.0)
        ray_rec = sp.trace_ray(8.0, 1.0, theta0=theta0, r0=30.0)
        self.assertTrue(ray_app.hit_disk)
        self.assertTrue(ray_rec.hit_disk)

        # Blueshift vs Redshift
        self.assertGreater(ray_app.doppler_g, 1.0)
        self.assertLess(ray_rec.doppler_g, 1.0)

        # Relativistic beaming ratio I = g^4 * I_emit
        self.assertGreater(ray_app.intensity, 3.0 * ray_rec.intensity)

    def test_visualizer_braille_and_colormaps(self):
        # Grid of shape 8x4 -> 2x2 Braille chars
        pixel_grid = np.ones((8, 4), dtype=float)
        intensity_grid = np.full((8, 4), 0.75, dtype=float)
        lines = KerrVisualizer.render_braille_grid(pixel_grid, intensity_grid, palette="inferno")
        self.assertEqual(len(lines), 2)
        self.assertTrue(len(lines[0]) > 0)

        # Colormaps
        r, g, b = KerrVisualizer.inferno_rgb(0.5)
        self.assertTrue(0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255)
        r, g, b = KerrVisualizer.blackbody_rgb(0.8)
        self.assertTrue(0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255)

    def test_formal_invariants_verification(self):
        self.assertTrue(verify_kerr_invariants())

    def test_cli_subcommands(self):
        # Verify command
        res = self.run_cli("verify")
        self.assertEqual(res.returncode, 0)
        self.assertIn("7/7 Tests Passed", res.stdout)

        # Ascii command
        res = self.run_cli("ascii")
        self.assertEqual(res.returncode, 0)
        self.assertIn("KERR BLACK HOLE", res.stdout)

        # Shadow command
        res = self.run_cli("shadow", "--cols", "20", "--rows", "10")
        self.assertEqual(res.returncode, 0)

        # Disk command
        res = self.run_cli("disk", "--cols", "20", "--rows", "10")
        self.assertEqual(res.returncode, 0)

        # Bench command
        res = self.run_cli("bench", "-n", "100")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Benchmark completed successfully", res.stdout)


if __name__ == "__main__":
    unittest.main()
