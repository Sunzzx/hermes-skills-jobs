"""Comprehensive Unit and Integration Test Suite for Muon Optimizer Engine.""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

# Dynamic import of bin/muon-optimizer
REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = REPO_ROOT / "bin" / "muon-optimizer"

loader = SourceFileLoader("muon_optimizer", str(BIN_PATH))
spec = importlib.util.spec_from_loader("muon_optimizer", loader)
muon_optimizer = importlib.util.module_from_spec(spec)
sys.modules["muon_optimizer"] = muon_optimizer
loader.exec_module(muon_optimizer)

newton_schulz5 = muon_optimizer.newton_schulz5
exact_polar_decomposition = muon_optimizer.exact_polar_decomposition
compute_spectral_metrics = muon_optimizer.compute_spectral_metrics
Parameter = muon_optimizer.Parameter
MuonOptimizer = muon_optimizer.MuonOptimizer
TwoLayerMLP = muon_optimizer.TwoLayerMLP


class TestNewtonSchulz5(unittest.TestCase):
    """Tests for the 5th-order Newton-Schulz polynomial iteration.""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_square_matrix_orthogonalization(self):
        G = self.rng.normal(0.0, 1.0, size=(64, 64))
        O = newton_schulz5(G, steps=5)
        self.assertEqual(O.shape, (64, 64))
        metrics = compute_spectral_metrics(O)
        # Singular values should be tightly clustered in [0.65, 1.15]
        self.assertGreater(metrics["min_singular_value"], 0.60)
        self.assertLess(metrics["max_singular_value"], 1.20)
        self.assertLess(metrics["condition_number"], 2.0)

    def test_wide_matrix_orthogonalization(self):
        G = self.rng.normal(0.0, 1.0, size=(32, 96))
        O = newton_schulz5(G, steps=5)
        self.assertEqual(O.shape, (32, 96))
        metrics = compute_spectral_metrics(O)
        self.assertLess(metrics["condition_number"], 2.2)

    def test_tall_matrix_orthogonalization(self):
        G = self.rng.normal(0.0, 1.0, size=(96, 32))
        O = newton_schulz5(G, steps=5)
        self.assertEqual(O.shape, (96, 32))
        metrics = compute_spectral_metrics(O)
        self.assertLess(metrics["condition_number"], 2.2)
        # Aspect ratio scaling factor for 96x32 is sqrt(3) ~ 1.732
        expected_scale = math.sqrt(96 / 32)
        self.assertAlmostEqual(metrics["mean_singular_value"], expected_scale * 0.85, delta=0.4)

    def test_reproducibility(self):
        G = self.rng.normal(0.0, 1.0, size=(40, 40))
        O1 = newton_schulz5(G, steps=5)
        O2 = newton_schulz5(G, steps=5)
        np.testing.assert_array_equal(O1, O2)

    def test_dtype_preservation(self):
        G_f32 = self.rng.normal(0.0, 1.0, size=(20, 20)).astype(np.float32)
        O_f32 = newton_schulz5(G_f32, steps=5)
        self.assertEqual(O_f32.dtype, np.float32)

        G_f64 = self.rng.normal(0.0, 1.0, size=(20, 20)).astype(np.float64)
        O_f64 = newton_schulz5(G_f64, steps=5)
        self.assertEqual(O_f64.dtype, np.float64)

    def test_invalid_dimension_raises(self):
        v = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            newton_schulz5(v)

    def test_polar_decomposition_svd_alignment(self):
        G = self.rng.normal(0.0, 1.0, size=(48, 48))
        O = newton_schulz5(G, steps=5)
        Q_svd, S = exact_polar_decomposition(G)
        rel_diff = np.linalg.norm(O - Q_svd) / np.linalg.norm(Q_svd)
        # NS5 approximates polar factor within ~20% relative error in 5 steps
        self.assertLess(rel_diff, 0.30)


class TestMuonOptimizer(unittest.TestCase):
    """Unit and integration tests for hybrid Muon + AdamW optimizer.""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.W = Parameter("weight", self.rng.normal(0.0, 0.1, size=(32, 64)))
        self.b = Parameter("bias", np.zeros(32))

    def test_parameter_classification(self):
        self.assertTrue(self.W.is_matrix)
        self.assertFalse(self.b.is_matrix)

    def test_hybrid_parameter_routing(self):
        opt = MuonOptimizer([self.W, self.b])
        self.assertEqual(len(opt.muon_params), 1)
        self.assertEqual(opt.muon_params[0].name, "weight")
        self.assertEqual(len(opt.adamw_params), 1)
        self.assertEqual(opt.adamw_params[0].name, "bias")

    def test_zero_grad(self):
        self.W.grad = np.ones_like(self.W.data)
        self.b.grad = np.ones_like(self.b.data)
        opt = MuonOptimizer([self.W, self.b])
        opt.zero_grad()
        np.testing.assert_array_equal(self.W.grad, 0.0)
        np.testing.assert_array_equal(self.b.grad, 0.0)

    def test_decoupled_weight_decay(self):
        # When gradient is zero, weight decays by (1 - lr * wd)
        self.W.grad = np.zeros_like(self.W.data)
        self.b.grad = np.zeros_like(self.b.data)
        initial_W = self.W.data.copy()
        initial_b = self.b.data.copy()

        lr = 0.02
        wd = 0.05
        opt = MuonOptimizer([self.W, self.b], lr=lr, weight_decay=wd, adamw_lr=0.01, adamw_weight_decay=wd)
        opt.step()

        expected_W = initial_W * (1.0 - lr * wd)
        np.testing.assert_allclose(self.W.data, expected_W, rtol=1e-5)

    def test_nesterov_momentum_step(self):
        grad = self.rng.normal(0.0, 1.0, size=(16, 16))
        p = Parameter("W_test", np.eye(16))
        p.grad = grad.copy()

        opt = MuonOptimizer([p], lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0)
        opt.step()

        state = opt.muon_states["W_test"]
        self.assertEqual(state.step, 1)
        self.assertIsNotNone(state.momentum_buffer)
        # Buffer should equal (1 - 0.9) * grad
        expected_buf = 0.1 * grad
        np.testing.assert_allclose(state.momentum_buffer, expected_buf, rtol=1e-5)

    def test_adamw_vector_step(self):
        b = Parameter("bias_test", np.ones(8))
        b.grad = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        opt = MuonOptimizer([b], adamw_lr=0.01, adamw_betas=(0.9, 0.999), adamw_weight_decay=0.0)
        initial_b = b.data.copy()
        opt.step()

        # AdamW step must update parameter values
        self.assertFalse(np.array_equal(b.data, initial_b))
        self.assertEqual(opt.adamw_states["bias_test"].step, 1)

    def test_state_dict_serialization(self):
        mlp = TwoLayerMLP(din=16, dhidden=32, dout=16, seed=42)
        opt1 = MuonOptimizer(mlp.params, lr=0.02, adamw_lr=0.003)

        # Run 2 steps to populate states
        X = self.rng.normal(0.0, 1.0, size=(4, 16))
        pred, cache = mlp.forward(X)
        mlp.backward(cache, pred)
        opt1.step()

        state = opt1.state_dict()
        opt2 = MuonOptimizer(mlp.params)
        opt2.load_state_dict(state)

        self.assertEqual(opt2.lr, 0.02)
        self.assertEqual(opt2.adamw_lr, 0.003)
        self.assertEqual(opt2.muon_states["W1"].step, 1)
        self.assertEqual(opt2.adamw_states["b1"].step, 1)
        np.testing.assert_array_equal(
            opt1.muon_states["W1"].momentum_buffer,
            opt2.muon_states["W1"].momentum_buffer,
        )


class TestNeuralConvergence(unittest.TestCase):
    """Tests optimization dynamics on non-linear MLP models.""

    def test_mlp_training_convergence(self):
        mlp = TwoLayerMLP(din=32, dhidden=64, dout=32, seed=99)
        opt = MuonOptimizer(mlp.params, lr=0.03, adamw_lr=0.005)
        rng = np.random.default_rng(101)
        X = rng.normal(0.0, 1.0, size=(16, 32))
        Y = rng.normal(0.0, 1.0, size=(16, 32))

        pred0, cache0 = mlp.forward(X)
        loss0 = float(np.mean((pred0 - Y) ** 2))

        for _ in range(25):
            pred, cache = mlp.forward(X)
            mlp.backward(cache, 2.0 * (pred - Y))
            opt.step()

        pred_final, _ = mlp.forward(X)
        loss_final = float(np.mean((pred_final - Y) ** 2))

        # Loss should reduce substantially
        self.assertLess(loss_final, loss0 * 0.70)


class TestCLIExecution(unittest.TestCase):
    """Functional tests executing the bin/muon-optimizer CLI binary.""

    def test_cli_status(self):
        res = subprocess.run([str(BIN_PATH), "status"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Muon", res.stdout)
        self.assertIn("Newton-Schulz", res.stdout)

    def test_cli_verify(self):
        res = subprocess.run([str(BIN_PATH), "verify"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("PASSED", res.stdout)

    def test_cli_orthogonalize(self):
        res = subprocess.run(
            [str(BIN_PATH), "orthogonalize", "--rows", "16", "--cols", "32", "--steps", "3"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Condition Number", res.stdout)

    def test_cli_train_step(self):
        res = subprocess.run([str(BIN_PATH), "train-step"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Step completed", res.stdout)


if __name__ == "__main__":
    unittest.main()
