import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "lbm-cfd"

# Load module directly for unit tests
lbm_globals = {}
with open(CLI, "r", encoding="utf-8") as f:
    exec(f.read(), lbm_globals)

LatticeBoltzmannD2Q9 = lbm_globals["LatticeBoltzmannD2Q9"]
run_poiseuille_validation = lbm_globals["run_poiseuille_validation"]
BrailleCanvas = lbm_globals["BrailleCanvas"]
C_X = lbm_globals["C_X"]
C_Y = lbm_globals["C_Y"]
W = lbm_globals["W"]


class TestLatticeBoltzmannD2Q9(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False
        )

    def test_cli_info_command(self):
        res = self.run_cli("info")
        self.assertEqual(res.returncode, 0)
        self.assertIn("D2Q9 Theoretical Formulation", res.stdout)
        self.assertIn("BGK", res.stdout)
        self.assertIn("Momentum Exchange Method", res.stdout)

    def test_cli_poiseuille_validation(self):
        res = self.run_cli("poiseuille", "--steps", "300", "--ny", "31", "--nx", "51")
        self.assertEqual(res.returncode, 0)
        self.assertIn("PASSED: Parabolic velocity profile verified", res.stdout)
        self.assertIn("Determination Coeff R^2", res.stdout)

    def test_cli_benchmark_command(self):
        res = self.run_cli("benchmark")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Throughput (MLUPS)", res.stdout)
        self.assertIn("100 x 40", res.stdout)

    def test_cli_vortex_command(self):
        res = self.run_cli("vortex", "--steps", "200", "--width", "120", "--height", "40")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Aerodynamic & Hydrodynamic Analysis", res.stdout)
        self.assertIn("Reynolds Number", res.stdout)

    def test_cli_sim_headless(self):
        res = self.run_cli("sim", "--steps", "40", "--vis", "none", "--width", "100", "--height", "30")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Simulation Complete", res.stdout)
        self.assertIn("Final Drag Coeff", res.stdout)

    def test_equilibrium_conservation(self):
        """Test that feq reproduces exact density and momentum."""
        rho = np.array([[1.0, 1.05], [0.95, 1.02]])
        ux = np.array([[0.05, -0.02], [0.03, 0.0]])
        uy = np.array([[0.01, 0.04], [-0.01, -0.02]])

        feq = LatticeBoltzmannD2Q9.compute_equilibrium(rho, ux, uy)
        self.assertEqual(feq.shape, (9, 2, 2))

        # Zeroth moment sum(feq) == rho
        rho_calc = np.sum(feq, axis=0)
        np.testing.assert_allclose(rho_calc, rho, rtol=1e-12)

        # First moment sum(feq * c) == rho * u
        mom_x = np.sum(feq * C_X[:, None, None], axis=0)
        mom_y = np.sum(feq * C_Y[:, None, None], axis=0)
        np.testing.assert_allclose(mom_x, rho * ux, rtol=1e-12)
        np.testing.assert_allclose(mom_y, rho * uy, rtol=1e-12)

    def test_equilibrium_symmetry_at_rest(self):
        """At rest (u=0), feq must equal w_i * rho exactly."""
        rho = np.ones((10, 10))
        ux = np.zeros((10, 10))
        uy = np.zeros((10, 10))

        feq = LatticeBoltzmannD2Q9.compute_equilibrium(rho, ux, uy)
        for i in range(9):
            np.testing.assert_allclose(feq[i], W[i] * rho, rtol=1e-14)

    def test_obstacle_halfway_bounceback(self):
        """Test that obstacle nodes invert velocity directions upon bounce-back."""
        sim = LatticeBoltzmannD2Q9(nx=40, ny=20, reynolds=50.0, obstacle_type="cylinder", cylinder_d=6.0)
        self.assertTrue(np.any(sim.body_mask))
        self.assertTrue(np.all(sim.obstacle[0, :]))
        self.assertTrue(np.all(sim.obstacle[-1, :]))

        # Run one step
        sim.collide_and_stream()
        self.assertEqual(sim.step_count, 1)
        self.assertTrue(np.all(sim.ux[sim.obstacle] == 0.0))
        self.assertTrue(np.all(sim.uy[sim.obstacle] == 0.0))

    def test_poiseuille_analytical_accuracy(self):
        """Verify R^2 > 0.995 on laminar channel Poiseuille validation."""
        res = run_poiseuille_validation(ny=31, nx=51, u_max=0.04, steps=400)
        self.assertTrue(res["passed"])
        self.assertGreater(res["r_squared"], 0.995)
        self.assertLess(res["mse"], 1e-4)

    def test_vorticity_calculation(self):
        """Verify curl calculation on known analytical velocity field."""
        sim = LatticeBoltzmannD2Q9(nx=30, ny=30, obstacle_type="none")
        Y, X = np.ogrid[:30, :30]
        # u_x = -y, u_y = x => vorticity = du_y/dx - du_x/dy = 1 - (-1) = 2.0
        sim.ux = -Y.astype(np.float64)
        sim.uy = X.astype(np.float64)
        vort = sim.compute_vorticity()
        # Interior points should equal 2.0
        np.testing.assert_allclose(vort[2:-2, 2:-2], 2.0, rtol=1e-4)

    def test_braille_canvas_rendering(self):
        """Verify Braille subpixel rendering produces valid Unicode output."""
        canvas = BrailleCanvas(char_width=20, char_height=10)
        grid = np.zeros((40, 40))
        grid[10:30, 10:30] = 0.05
        obs = np.zeros((40, 40), dtype=bool)
        obs[15:25, 15:25] = True

        lines = canvas.render_field(grid, obs, vmin=-0.05, vmax=0.05)
        self.assertEqual(len(lines), 10)
        # Verify unicode braille character presence
        has_braille_or_block = any(chr(0x2800) <= ch <= chr(0x28FF) or ch == '█' for line in lines for ch in line)
        self.assertTrue(has_braille_or_block)


if __name__ == "__main__":
    unittest.main()
