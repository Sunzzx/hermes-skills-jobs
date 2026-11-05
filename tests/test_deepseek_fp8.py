"""Unit and integration tests for DeepSeek-V3 FP8 Mixed-Precision Training & Quantized GEMM Engine."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "fp8-engine"

spec = importlib.util.spec_from_loader("fp8_engine", SourceFileLoader("fp8_engine", str(CLI)))
assert spec is not None and spec.loader is not None
fp8_engine = importlib.util.module_from_spec(spec)
sys.modules["fp8_engine"] = fp8_engine
spec.loader.exec_module(fp8_engine)

E4M3 = fp8_engine.E4M3
E5M2 = fp8_engine.E5M2
DeepSeekFP8Engine = fp8_engine.DeepSeekFP8Engine
compute_sqnr = fp8_engine.compute_sqnr
cosine_similarity = fp8_engine.cosine_similarity
max_relative_error = fp8_engine.max_relative_error


class TestFP8Formats(unittest.TestCase):
    def test_e4m3_specification(self):
        self.assertEqual(E4M3.exp_bits, 4)
        self.assertEqual(E4M3.mant_bits, 3)
        self.assertEqual(E4M3.bias, 7)
        self.assertEqual(E4M3.max_finite, 448.0)
        self.assertEqual(np.nanmax(E4M3.lut), 448.0)
        self.assertTrue(np.isnan(E4M3.lut[0x7F]))
        self.assertTrue(np.isnan(E4M3.lut[0xFF]))
        # Smallest subnormal: 2^-6 * (1/8) = 2^-9 = 0.001953125
        self.assertAlmostEqual(E4M3.lut[0x01], 0.001953125, places=9)
        # Smallest normal: 2^(1-7) * (1 + 0) = 2^-6 = 0.015625
        self.assertAlmostEqual(E4M3.lut[0x08], 0.015625, places=8)

    def test_e5m2_specification(self):
        self.assertEqual(E5M2.exp_bits, 5)
        self.assertEqual(E5M2.mant_bits, 2)
        self.assertEqual(E5M2.bias, 15)
        self.assertEqual(E5M2.max_finite, 57344.0)
        finite_vals = E5M2.lut[np.isfinite(E5M2.lut)]
        self.assertEqual(np.max(finite_vals), 57344.0)
        self.assertTrue(np.isinf(E5M2.lut[0x7C]))
        self.assertTrue(np.isinf(E5M2.lut[0xFC]))
        self.assertTrue(np.isnan(E5M2.lut[0x7D]))
        # Smallest subnormal: 2^-14 * (1/4) = 2^-16 = 1.52587890625e-05
        self.assertAlmostEqual(E5M2.lut[0x01], 1.52587890625e-05, places=10)

    def test_e4m3_encode_decode_roundtrip(self):
        exact_vals = np.array([-448.0, -128.0, -10.0, -1.0, 0.0, 0.5, 1.0, 16.0, 224.0, 448.0])
        enc = E4M3.encode_nearest(exact_vals)
        dec = E4M3.decode(enc)
        np.testing.assert_allclose(exact_vals, dec, atol=1e-5)


class TestDeepSeekTileMicroScaling(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_activation_tile_shapes_and_scales(self):
        M, K = 16, 256
        X = self.rng.randn(M, K)
        q = DeepSeekFP8Engine.quantize_activations_tile128(X, fmt=E4M3)
        self.assertEqual(q.granularity, "tile_1x128")
        self.assertEqual(q.fp8_bytes.shape, (M, K))
        self.assertEqual(q.scales.shape, (M, K // 128))
        self.assertEqual(q.fp8_bytes.dtype, np.uint8)

    def test_weight_block_shapes_and_scales(self):
        K, N = 256, 384
        W = self.rng.randn(K, N)
        q = DeepSeekFP8Engine.quantize_weights_block128x128(W, fmt=E4M3)
        self.assertEqual(q.granularity, "block_128x128")
        self.assertEqual(q.fp8_bytes.shape, (K, N))
        self.assertEqual(q.scales.shape, (K // 128, N // 128))
        self.assertEqual(q.fp8_bytes.dtype, np.uint8)

    def test_outlier_isolation_sqnr_advantage(self):
        M, K = 32, 256
        X = self.rng.randn(M, K)
        # Inject severe outlier into tile 0 (channel 17)
        X[:, 17] *= 40.0

        q_global = DeepSeekFP8Engine.quantize_global(X, fmt=E4M3)
        X_global = DeepSeekFP8Engine.dequantize_global(q_global, fmt=E4M3)
        sqnr_global = compute_sqnr(X, X_global)

        q_tile = DeepSeekFP8Engine.quantize_activations_tile128(X, fmt=E4M3)
        X_tile = DeepSeekFP8Engine.dequantize_activations(q_tile, fmt=E4M3)
        sqnr_tile = compute_sqnr(X, X_tile)

        # Tile quantization must achieve significantly higher SQNR due to outlier confinement
        self.assertGreater(sqnr_tile, sqnr_global + 5.0)
        self.assertGreater(cosine_similarity(X, X_tile), 0.999)


class TestFineGrainedGEMM(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1337)

    def test_gemm_accuracy_against_reference(self):
        M, K, N = 32, 256, 256
        X = self.rng.randn(M, K)
        W = self.rng.randn(K, N) / math.sqrt(K)
        Y_ref = np.matmul(X, W)

        q_act = DeepSeekFP8Engine.quantize_activations_tile128(X, fmt=E4M3)
        q_weight = DeepSeekFP8Engine.quantize_weights_block128x128(W, fmt=E4M3)
        Y_fp8 = DeepSeekFP8Engine.fine_grained_gemm(q_act, q_weight, fmt=E4M3)

        cos_sim = cosine_similarity(Y_ref, Y_fp8)
        sqnr = compute_sqnr(Y_ref, Y_fp8)
        self.assertGreater(cos_sim, 0.999)
        self.assertGreater(sqnr, 28.0)

    def test_gemm_dimension_validation(self):
        M, K_bad = 16, 200  # Not divisible by 128
        X = np.ones((M, K_bad))
        with self.assertRaises(AssertionError):
            DeepSeekFP8Engine.quantize_activations_tile128(X, fmt=E4M3)


class TestStochasticRounding(unittest.TestCase):
    def test_unbiased_expectation(self):
        target_val = 2.45
        x = np.full(10000, target_val)
        enc = E5M2.encode_stochastic(x, seed=42)
        dec = E5M2.decode(enc)
        mean_val = float(np.mean(dec))
        # Expectation should closely match target without systematic bias
        self.assertAlmostEqual(mean_val, target_val, delta=0.03)


class TestMemoryAndScaleOverhead(unittest.TestCase):
    def test_scale_overhead_under_one_percent(self):
        K, N = 1024, 1024
        W = np.zeros((K, N), dtype=np.float32)
        q = DeepSeekFP8Engine.quantize_weights_block128x128(W, fmt=E4M3)
        overhead = q.scale_overhead_pct()
        # 128x128 blocks -> 1 scale (4 bytes) per 16384 bytes -> 0.024%
        self.assertLess(overhead, 0.1)
        self.assertAlmostEqual(overhead, (4.0 / 16384.0) * 100.0, places=3)


class TestCLIExecution(unittest.TestCase):
    def test_cli_subcommands(self):
        for cmd in ["status", "quantize", "gemm", "benchmark", "verify"]:
            res = subprocess.run([sys.executable, str(CLI), cmd], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI command '{cmd}' failed: {res.stderr}")


if __name__ == '__main__':
    unittest.main()
