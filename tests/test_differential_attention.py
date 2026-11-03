"""Comprehensive Unit and Integration Test Suite for Differential Attention (Diff-Transformer) Engine."""

from __future__ import annotations

import subprocess
import sys
import unittest
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "diff-attn"

sys.path.insert(0, str(ROOT / "bin"))
loader = SourceFileLoader("diff_attn", str(CLI))
spec = importlib.util.spec_from_loader("diff_attn", loader)
assert spec is not None
diff_module = importlib.util.module_from_spec(spec)
sys.modules["diff_attn"] = diff_module
loader.exec_module(diff_module)

DiffAttnConfig = diff_module.DiffAttnConfig
DifferentialKVCache = diff_module.DifferentialKVCache
DifferentialAttention = diff_module.DifferentialAttention
StandardAttentionBaseline = diff_module.StandardAttentionBaseline
softmax = diff_module.softmax
rms_norm = diff_module.rms_norm
precompute_rope_freqs = diff_module.precompute_rope_freqs
apply_rope = diff_module.apply_rope
apply_rope_sequence = diff_module.apply_rope_sequence


class TestMathematicalPrimitives(unittest.TestCase):
    """Tests for mathematical kernels and RoPE rotations."""

    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_softmax_numerical_stability(self):
        # Extremely large positive and negative numbers
        x = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]])
        probs = softmax(x, axis=-1)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(2), atol=1e-7)

    def test_rms_norm_scale_invariance(self):
        x = self.rng.randn(2, 8, 64) * 12.0
        weight = np.ones(64)
        normed = rms_norm(x, weight, eps=1e-8)
        rms_vals = np.sqrt(np.mean(normed ** 2, axis=-1))
        np.testing.assert_allclose(rms_vals, np.ones((2, 8)), atol=1e-5)

    def test_rope_rotation_norm_preservation(self):
        dim = 32
        cos, sin = precompute_rope_freqs(dim, 64)
        v = self.rng.randn(1, 1, 4, dim)
        v_norm_before = np.linalg.norm(v, axis=-1)
        v_rot = apply_rope(v, cos, sin, pos=17)
        v_norm_after = np.linalg.norm(v_rot, axis=-1)
        np.testing.assert_allclose(v_norm_before, v_norm_after, atol=1e-7)


class TestDifferentialAttentionEngine(unittest.TestCase):
    """Tests for Differential Attention architecture and invariants."""

    def setUp(self):
        self.config = DiffAttnConfig(
            dim=256,
            n_heads=4,
            lambda_init=0.8,
            max_seq_len=128,
            use_rope=True,
            learnable_lambda=True,
        )
        self.engine = DifferentialAttention(self.config, seed=1337)
        self.rng = np.random.RandomState(99)

    def test_config_validation(self):
        # Non-divisible dimensions should raise
        with self.assertRaises(ValueError):
            DiffAttnConfig(dim=255, n_heads=4)
        with self.assertRaises(ValueError):
            # head_dim must be even
            DiffAttnConfig(dim=12, n_heads=4)  # head_dim = 3 (odd)

    def test_differential_row_sum_invariant(self):
        # For each head h, sum(A_diff) == 1 - lambda_h
        B, S, D = 2, 16, self.config.dim
        x = self.rng.randn(B, S, D)
        _, maps = self.engine.forward(x, causal=False, return_maps=True)
        A_diff = maps["A_diff"]  # (B, H, S, S)
        lambdas = maps["lambdas"]  # (H,)

        row_sums = np.sum(A_diff, axis=-1)
        expected = np.broadcast_to(1.0 - lambdas[None, :, None], row_sums.shape)
        np.testing.assert_allclose(row_sums, expected, atol=1e-7)

    def test_common_mode_rejection_property(self):
        # When Q1 K1 and Q2 K2 are subjected to identical ambient distraction,
        # the distraction response is suppressed by factor (1 - lambda).
        lam = float(np.mean(self.engine.compute_lambda()))
        ambient_scores = np.ones((1, self.config.n_heads, 1, 32)) * 3.0
        a1 = softmax(ambient_scores, axis=-1)
        a2 = softmax(ambient_scores, axis=-1)
        a_diff = a1 - self.engine.compute_lambda()[None, :, None, None] * a2

        cm_total = float(np.mean(np.sum(a_diff, axis=-1)))
        self.assertAlmostEqual(cm_total, 1.0 - lam, places=5)
        # Attenuation should be at least 70% for lambda ~ 0.8
        self.assertLess(cm_total, 0.35)

    def test_causal_masking_enforcement(self):
        B, S, D = 1, 8, self.config.dim
        x = self.rng.randn(B, S, D)
        _, maps = self.engine.forward(x, causal=True, return_maps=True)
        A1 = maps["A1"]
        A2 = maps["A2"]
        A_diff = maps["A_diff"]

        for r in range(S):
            for c in range(r + 1, S):
                self.assertEqual(A1[0, 0, r, c], 0.0)
                self.assertEqual(A2[0, 0, r, c], 0.0)
                self.assertEqual(A_diff[0, 0, r, c], 0.0)

    def test_kv_cache_autoregressive_parity(self):
        seq_len = 6
        x_seq = self.rng.randn(1, seq_len, self.config.dim)

        # Batch causal forward
        out_full = self.engine.forward(x_seq, causal=True)

        # Incremental generation token by token
        cache = DifferentialKVCache(self.config, batch_size=1)
        out_steps = []
        for t in range(seq_len):
            xt = x_seq[:, t:t+1, :]
            ot = self.engine.step_forward(xt, cache, pos=t)
            out_steps.append(ot)
        out_incremental = np.concatenate(out_steps, axis=1)

        max_diff = np.max(np.abs(out_full - out_incremental))
        self.assertLess(max_diff, 1e-6)

    def test_kv_cache_reset(self):
        cache = DifferentialKVCache(self.config, batch_size=1)
        k1 = self.rng.randn(1, 1, 4, 32)
        k2 = self.rng.randn(1, 1, 4, 32)
        v = self.rng.randn(1, 1, 4, 64)
        cache.update(k1, k2, v)
        self.assertEqual(cache.cur_len, 1)
        cache.reset()
        self.assertEqual(cache.cur_len, 0)
        self.assertIsNone(cache.k1_cache)

    def test_learnable_lambda_bounds(self):
        lams = self.engine.compute_lambda()
        self.assertEqual(len(lams), self.config.n_heads)
        self.assertTrue(np.all(lams >= 0.01))
        self.assertTrue(np.all(lams <= 0.99))


class TestCLIExecution(unittest.TestCase):
    """Tests invoking CLI commands via subprocess."""

    def test_cli_verify(self):
        res = subprocess.run([sys.executable, str(CLI), "verify"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"verify failed: {res.stderr}")
        self.assertIn("ALL 5/5 MATHEMATICAL INVARIANTS PERFECTLY VERIFIED", res.stdout)

    def test_cli_inspect(self):
        res = subprocess.run([sys.executable, str(CLI), "inspect"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"inspect failed: {res.stderr}")
        self.assertIn("Differential Attention Map A_diff", res.stdout)

    def test_cli_needle(self):
        res = subprocess.run([sys.executable, str(CLI), "needle", "--context", "32", "--pos", "10"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"needle failed: {res.stderr}")
        self.assertIn("Differential Attention delivers", res.stdout)


if __name__ == '__main__':
    unittest.main()
