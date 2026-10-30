import math
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "fluid2d"

# Load module directly for unit tests
fluid2d_globals = {}
with open(CLI, "r", encoding="utf-8") as f:
    exec(f.read(), fluid2d_globals)

NavierStokes2D = fluid2d_globals["NavierStokes2D"]
run_taylor_green_validation = fluid2d_globals["run_taylor_green_validation"]
BrailleCanvas = fluid2d_globals["BrailleCanvas"]


class TestNavierStokes2D(unittest.TestCase):
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
        self.assertIn("Chorin Projection", res.stdout)
        self.assertIn("Marker-and-Cell", res.stdout)
        self.assertIn("Operator Splitting Scheme", res.stdout)

    def test_cli_taylor_green_analytical_validation(self):
        res = self.run_cli("taylor-green", "--grid", "48", "--steps", "40", "--validate")
        self.assertEqual(res.returncode, 0)
        self.assertIn("PASSED", res.stdout)
        self.assertIn("Max Divergence:", res.stdout)

    def test_cli_cavity_simulation(self):
        res = self.run_cli("cavity", "--grid", "32", "--steps", "40", "--vis", "none")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Lid-Driven Cavity Simulation", res.stdout)
        self.assertIn("Centerline Analysis", res.stdout)

    def test_cli_vortex_simulation(self):
        res = self.run_cli("vortex", "--nx", "60", "--ny", "24", "--steps", "30", "--vis", "none")
        self.assertEqual(res.returncode, 0)
        self.assertIn("Von Kármán Vortex Street", res.stdout)
        self.assertIn("Peak Vorticity", res.stdout)

    def test_cli_benchmark_command(self):
        res = self.run_cli("benchmark", "--grid", "32", "--steps", "20")
        self.assertEqual(res.returncode, 0)
        self.assertIn("MCUPS", res.stdout)
        self.assertIn("Step Latency", res.stdout)

    def test_mac_staggered_grid_shapes(self):
        nx, ny = 32, 24
        sim = NavierStokes2D(nx=nx, ny=ny, lx=1.0, ly=0.75)
        self.assertEqual(sim.u.shape, (ny, nx + 1))
        self.assertEqual(sim.v.shape, (ny + 1, nx))
        self.assertEqual(sim.p.shape, (ny, nx))
        self.assertEqual(sim.obstacle.shape, (ny, nx))

    def test_helmholtz_hodge_projection_divergence_free(self):
        """Test that projection drives arbitrary velocity field to solenoidal (div < 1e-4)."""
        nx, ny = 36, 36
        sim = NavierStokes2D(nx=nx, ny=ny, lx=1.0, ly=1.0, nu=0.01, bc_type="cavity")
        # Distort velocity field with high non-divergence-free noise
        np.random.seed(42)
        sim.u[:, 1:-1] = np.random.randn(ny, nx - 1) * 0.5
        sim.v[1:-1, :] = np.random.randn(ny - 1, nx) * 0.5
        sim._apply_boundary_conditions()

        div_before = np.max(np.abs(sim.get_divergence()))
        self.assertGreater(div_before, 1.0)

        # Solve Poisson and project
        dt = 0.005
        sim.solve_pressure_poisson(dt, max_iter=150)
        sim.project(dt)

        div_after = np.max(np.abs(sim.get_divergence()))
        self.assertLess(div_after, 1e-4)

    def test_kinetic_energy_monotonic_viscous_decay(self):
        """Without external forcing, kinetic energy must strictly decay under viscosity."""
        sim = NavierStokes2D(nx=32, ny=32, lx=2.0 * math.pi, ly=2.0 * math.pi, nu=0.05, bc_type="periodic")
        x_u = np.linspace(0, 2 * math.pi, 33)
        y_u = np.linspace(0.5 * sim.dy, 2 * math.pi - 0.5 * sim.dy, 32)
        X_u, Y_u = np.meshgrid(x_u, y_u)
        sim.u = -np.cos(X_u) * np.sin(Y_u)

        x_v = np.linspace(0.5 * sim.dx, 2 * math.pi - 0.5 * sim.dx, 32)
        y_v = np.linspace(0, 2 * math.pi, 33)
        X_v, Y_v = np.meshgrid(x_v, y_v)
        sim.v = np.sin(X_v) * np.cos(Y_v)

        e0 = sim.get_kinetic_energy()
        for _ in range(20):
            sim.step(dt=0.01)
        e1 = sim.get_kinetic_energy()
        self.assertLess(e1, e0)

    def test_braille_canvas_rendering(self):
        canvas = BrailleCanvas(char_width=20, char_height=10)
        grid = np.sin(np.linspace(0, 3.14, 20)[:, None]) * np.cos(np.linspace(0, 3.14, 20)[None, :])
        lines = canvas.render(grid, palette="vorticity")
        self.assertEqual(len(lines), 10)
        self.assertTrue(any(len(line) > 0 for line in lines))


if __name__ == "__main__":
    unittest.main()
