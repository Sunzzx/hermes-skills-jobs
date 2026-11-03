"""Comprehensive Unit and Integration Test Suite for Multi-Head Latent Attention (MLA) Engine."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "mla-engine"

sys.path.insert(0, str(ROOT / "bin"))
mla_module = SourceFileLoader("mla_engine", str(CLI)).load_module()

MLAConfig = mla_module.MLAConfig
MLAKVCache = mla_module.MLAKVCache
MultiHeadLatentAttention = mla_module.MultiHeadLatentAttention
StandardMHABaseline = mla_module.StandardMHABaseline
GroupedQueryAttentionBaseline = mla_module.GroupedQueryAttentionBaseline
rms_norm = mla_module.rms_norm
precompute_rope_freqs = mla_module.precompute_rope_freqs
apply_rope = mla_module.apply_rope


class TestMLAPrimitives(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_rms_norm_invariance(self):
        x = self.rng.randn(4, 16, 64) * 5.0
        weight = np.ones(64)
        normed = rms_norm(x, weight, eps=1e-8)
        rms_val = np.sqrt(np.mean(normed ** 2, axis=-1))
        np.testing.assert_allclose(rms_val, 1.0, atol=1e-5)

    def test_rope_frequencies_and_pos0(self):
        cos, sin = precompute_rope_freqs(dim=32, max_seq_len=64, theta=10000.0)
        self.assertEqual(cos.shape, (64, 16))
        self.assertEqual(sin.shape, (64, 16))
        # At pos 0, cos is 1 and sin is 0
        np.testing.assert_allclose(cos[0], 1.0)
        np.testing.assert_allclose(sin[0], 0.0)

    def test_rope_norm_preservation(self):
        cos, sin = precompute_rope_freqs(dim=16, max_seq_len=32)
        v = self.rng.randn(1, 1, 16)
        rotated = apply_rope(v, cos, sin, pos=5)
        norm_orig = np.linalg.norm(v)
        norm_rot = np.linalg.norm(rotated)
        np.testing.assert_allclose(norm_orig, norm_rot, atol=1e-6)


class TestMLAKVCache(unittest.TestCase):
    def setUp(self):
        self.config = MLAConfig(dim=256, n_heads=4, q_lora_rank=64, kv_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32)
        self.cache = MLAKVCache(self.config, batch_size=2)

    def test_cache_append_and_dimensions(self):
        c_kv = np.random.randn(2, 1, 64)
        k_rope = np.random.randn(2, 1, 16)

        self.assertEqual(self.cache.length, 0)
        self.cache.append(c_kv, k_rope)
        self.assertEqual(self.cache.length, 1)
        self.assertEqual(self.cache.get_c_kv().shape, (2, 1, 64))
        self.assertEqual(self.cache.get_k_rope().shape, (2, 1, 16))

        # Append second token
        self.cache.append(c_kv, k_rope)
        self.assertEqual(self.cache.length, 2)
        self.assertEqual(self.cache.get_c_kv().shape, (2, 2, 64))
        self.assertEqual(self.cache.get_k_rope().shape, (2, 2, 16))

    def test_cache_memory_calculation(self):
        c_kv = np.random.randn(2, 1, 64)
        k_rope = np.random.randn(2, 1, 16)
        for _ in range(10):
            self.cache.append(c_kv, k_rope)

        # 10 tokens * 2 batch * (64 + 16) elems * 4 bytes
        expected_bytes = 10 * 2 * 80 * 4
        self.assertEqual(self.cache.memory_bytes(dtype_size=4), expected_bytes)


class TestMultiHeadLatentAttention(unittest.TestCase):
    def setUp(self):
        self.config = MLAConfig(
            dim=128,
            n_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=16,
            max_seq_len=64,
        )
        self.mla = MultiHeadLatentAttention(self.config, seed=42)

    def test_forward_decompressed_output_shape(self):
        x = np.random.randn(2, 8, self.config.dim)
        out = self.mla.forward_decompressed(x, causal=True, start_pos=0)
        self.assertEqual(out.shape, (2, 8, self.config.dim))

    def test_causal_masking_invariance(self):
        # Output of first token must not depend on future tokens
        x1 = np.random.randn(1, 4, self.config.dim)
        x2 = x1.copy()
        # Modify future tokens (positions 1, 2, 3)
        x2[:, 1:, :] = np.random.randn(1, 3, self.config.dim)

        out1 = self.mla.forward_decompressed(x1, causal=True)
        out2 = self.mla.forward_decompressed(x2, causal=True)

        np.testing.assert_allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-6)

    def test_exact_mathematical_equivalence(self):
        is_equiv, max_diff, decomp, absorbed = self.mla.verify_equivalence(
            seq_len=12, batch_size=2
        )
        self.assertTrue(is_equiv, f"Equivalence failed: max diff = {max_diff}")
        self.assertLess(max_diff, 1e-6)

    def test_compression_ratio(self):
        # MHA: 2 * 4 heads * (16 + 16) = 256
        # MLA: 32 + 16 = 48
        # Ratio = 256 / 48 = 5.33x
        self.assertGreater(self.config.compression_ratio_vs_mha, 5.0)


class TestMLACLI(unittest.TestCase):
    def test_cli_help(self):
        res = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Multi-Head Latent Attention", res.stdout)

    def test_cli_verify_json(self):
        res = subprocess.run(
            [sys.executable, str(CLI), "verify", "--seq-len", "8", "--batch-size", "1", "--json"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout.strip().split("\n\n")[-1])
        self.assertTrue(data["verified"])
        self.assertLess(data["max_absolute_difference"], 1e-6)

    def test_cli_benchmark(self):
        res = subprocess.run(
            [sys.executable, str(CLI), "benchmark", "--steps", "3"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("KV-Cache Memory Footprint Scaling", res.stdout)

    def test_cli_simulate(self):
        res = subprocess.run(
            [sys.executable, str(CLI), "simulate", "--steps", "3"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Token-by-Token Autoregressive Simulation", res.stdout)

    def test_cli_inspect(self):
        res = subprocess.run(
            [sys.executable, str(CLI), "inspect"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Architecture Specification", res.stdout)


if __name__ == "__main__":
    unittest.main()
