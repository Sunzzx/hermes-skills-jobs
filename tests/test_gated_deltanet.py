"""Unit and integration tests for Gated DeltaNet & Parallel Delta Rule Engine."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "deltanet"

spec = importlib.util.spec_from_loader("deltanet", SourceFileLoader("deltanet", str(CLI)))
assert spec is not None and spec.loader is not None
deltanet = importlib.util.module_from_spec(spec)
sys.modules["deltanet"] = deltanet
spec.loader.exec_module(deltanet)

GatedDeltaRuleSequential = deltanet.GatedDeltaRuleSequential
GatedDeltaRuleParallel = deltanet.GatedDeltaRuleParallel
GatedDeltaNetLayer = deltanet.GatedDeltaNetLayer
CausalLinearAttention = deltanet.CausalLinearAttention
associative_memory_probe = deltanet.associative_memory_probe
verify_invariants = deltanet.verify_invariants
l2_normalize = deltanet.l2_normalize
rel_err = deltanet.rel_err
sigmoid = deltanet.sigmoid
softplus = deltanet.softplus


class TestNumerics(unittest.TestCase):
    def test_sigmoid_bounds(self):
        x = np.array([-100.0, 0.0, 100.0])
        s = sigmoid(x)
        self.assertTrue(np.all(s >= 0.0) and np.all(s <= 1.0))
        self.assertAlmostEqual(float(s[1]), 0.5, places=7)

    def test_softplus_positive(self):
        x = np.linspace(-5, 5, 21)
        sp = softplus(x)
        self.assertTrue(np.all(sp >= 0.0))
        np.testing.assert_allclose(sp[x > 20], x[x > 20], atol=1e-6)

    def test_l2_normalize(self):
        rng = np.random.RandomState(0)
        x = rng.randn(10, 8)
        n = l2_normalize(x, axis=-1)
        norms = np.linalg.norm(n, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


class TestSequentialDeltaRule(unittest.TestCase):
    def test_unit_key_perfect_write(self):
        d = 16
        rng = np.random.RandomState(1)
        k = l2_normalize(rng.randn(d))
        v = rng.randn(d)
        seq = GatedDeltaRuleSequential(d, d, readout="post")
        y, _ = seq.step(k, v, beta=1.0, alpha=1.0)
        np.testing.assert_allclose(y, v, atol=1e-12)

    def test_zero_beta_pure_decay(self):
        d_k, d_v = 8, 10
        rng = np.random.RandomState(2)
        seq = GatedDeltaRuleSequential(d_k, d_v)
        seq.W = rng.randn(d_v, d_k)
        before = seq.W.copy()
        k = l2_normalize(rng.randn(d_k))
        seq.step(k, rng.randn(d_v), beta=0.0, alpha=0.25)
        np.testing.assert_allclose(seq.W, 0.25 * before, atol=1e-12)

    def test_causal_invariance(self):
        rng = np.random.RandomState(3)
        L, d_k, d_v = 24, 8, 8
        K = l2_normalize(rng.randn(L, d_k), axis=-1)
        V = rng.randn(L, d_v)
        beta = rng.uniform(0.3, 1.0, L)
        alpha = rng.uniform(0.8, 1.0, L)
        Y1, _ = GatedDeltaRuleSequential(d_k, d_v).forward(K, V, beta, alpha)
        V2 = V.copy()
        V2[-4:] += 50.0
        Y2, _ = GatedDeltaRuleSequential(d_k, d_v).forward(K, V2, beta, alpha)
        np.testing.assert_allclose(Y1[:-4], Y2[:-4], atol=1e-12)

    def test_householder_form_matches_outer(self):
        """W <- a W; W <- W + b outer(v - Wk, k)  ==  W @ (a(I-b kk^T)) + b outer(v,k) after scale order."""
        rng = np.random.RandomState(4)
        d_k, d_v = 6, 5
        W0 = rng.randn(d_v, d_k)
        k = l2_normalize(rng.randn(d_k))
        v = rng.randn(d_v)
        a, b = 0.9, 0.7
        # path A (sequential step semantics)
        W_a = a * W0
        u = W_a @ k
        W_a = W_a + b * np.outer(v - u, k)
        # path B (matrix form)
        W_b = (a * W0) @ (np.eye(d_k) - b * np.outer(k, k)) + b * np.outer(v, k)
        np.testing.assert_allclose(W_a, W_b, atol=1e-12)


class TestParallelEquivalence(unittest.TestCase):
    def test_seq_par_match(self):
        rng = np.random.RandomState(5)
        L, d_k, d_v = 40, 12, 10
        K = l2_normalize(rng.randn(L, d_k), axis=-1)
        V = rng.randn(L, d_v)
        beta = rng.uniform(0.2, 1.0, L)
        alpha = rng.uniform(0.6, 1.0, L)
        Ys, Ws = GatedDeltaRuleSequential(d_k, d_v).forward(K, V, beta, alpha)
        Yp, Wp = GatedDeltaRuleParallel(d_k, d_v).forward(K, V, beta, alpha)
        self.assertLess(rel_err(Ys, Yp), 1e-10)
        self.assertLess(rel_err(Ws, Wp), 1e-10)

    def test_chunk_match(self):
        rng = np.random.RandomState(6)
        L, d_k, d_v = 64, 8, 8
        K = l2_normalize(rng.randn(L, d_k), axis=-1)
        V = rng.randn(L, d_v)
        beta = rng.uniform(0.4, 1.0, L)
        alpha = np.full(L, 0.92)
        Ys, _ = GatedDeltaRuleSequential(d_k, d_v).forward(K, V, beta, alpha)
        Yc, _ = GatedDeltaRuleParallel(d_k, d_v).forward_chunk_fused(
            K, V, beta, alpha, chunk_size=7
        )
        self.assertLess(rel_err(Ys, Yc), 1e-10)


class TestAssociativeMemory(unittest.TestCase):
    def test_probe_high_cosine(self):
        res = associative_memory_probe(d_k=32, d_v=32, n_pairs=8, seed=9, beta=1.0, alpha=1.0)
        self.assertGreater(res["mean_cosine"], 0.85)
        self.assertLess(res["mse"], 1.0)


class TestLayer(unittest.TestCase):
    def test_forward_shape_and_finite(self):
        layer = GatedDeltaNetLayer(d_model=32, d_k=32, d_v=32, n_heads=4, seed=0)
        X = np.random.RandomState(0).randn(16, 32)
        Y = layer.forward(X)
        self.assertEqual(Y.shape, X.shape)
        self.assertTrue(np.isfinite(Y).all())

    def test_gate_ranges(self):
        layer = GatedDeltaNetLayer(d_model=16, d_k=16, d_v=16, n_heads=2, seed=1)
        X = np.random.RandomState(1).randn(8, 16)
        p = layer.project(X)
        self.assertTrue(float(p["beta"].min()) >= 0.0)
        self.assertTrue(float(p["beta"].max()) <= 1.0)
        self.assertTrue(float(p["alpha"].min()) > 0.0)
        self.assertTrue(float(p["alpha"].max()) <= 1.0 + 1e-12)


class TestLinearAttentionBaseline(unittest.TestCase):
    def test_runs_and_differs(self):
        rng = np.random.RandomState(8)
        L, d = 20, 8
        K = l2_normalize(rng.randn(L, d), axis=-1)
        V = rng.randn(L, d)
        beta = np.ones(L)
        alpha = np.ones(L)
        Yd, _ = GatedDeltaRuleSequential(d, d).forward(K, V, beta, alpha)
        Yl = CausalLinearAttention(d, d).forward(K, V, alpha=alpha)
        self.assertGreater(rel_err(Yd, Yl), 1e-3)


class TestInvariantSuite(unittest.TestCase):
    def test_full_suite(self):
        res = verify_invariants(verbose=False)
        self.assertTrue(res["ok"], msg=f"failed={res['failed']}")


class TestCLI(unittest.TestCase):
    def test_verify_cli_exit0(self):
        r = subprocess.run(
            [sys.executable, str(CLI), "verify"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("PASSED", r.stdout)

    def test_benchmark_cli(self):
        r = subprocess.run(
            [sys.executable, str(CLI), "benchmark", "--seq-len", "32", "--dim", "8", "--repeats", "1"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)

    def test_inspect_cli(self):
        r = subprocess.run(
            [sys.executable, str(CLI), "inspect", "--dim", "64", "--heads", "4"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("params", r.stdout)


if __name__ == "__main__":
    unittest.main()
