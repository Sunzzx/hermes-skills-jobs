"""Unit and integration tests for NVIDIA Blackwell NVFP4 Micro-Tensor Engine."""

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

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "nv-fp4-engine"

loader = SourceFileLoader("nv_fp4_engine", str(BIN_PATH))
spec = importlib.util.spec_from_loader("nv_fp4_engine", loader)
assert spec is not None and spec.loader is not None
nv_fp4_engine = importlib.util.module_from_spec(spec)
sys.modules["nv_fp4_engine"] = nv_fp4_engine
loader.exec_module(nv_fp4_engine)

NVFP4Quantizer = nv_fp4_engine.NVFP4Quantizer
NVFP4GEMMEngine = nv_fp4_engine.NVFP4GEMMEngine
MicroScalingConfig = nv_fp4_engine.MicroScalingConfig
FP4_LEVELS = nv_fp4_engine.FP4_LEVELS


class TestNVFP4Engine(unittest.TestCase):
    """Unit tests for NVFP4 mathematical and hardware engine."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.quantizer = NVFP4Quantizer(MicroScalingConfig(block_size=16))

    def test_codebook_properties(self):
        self.assertEqual(len(FP4_LEVELS), 15)
        self.assertAlmostEqual(FP4_LEVELS[-1], 6.0)
        self.assertAlmostEqual(FP4_LEVELS[0], -6.0)
        self.assertIn(0.0, FP4_LEVELS)
        # Verify symmetric levels
        for val in FP4_LEVELS:
            self.assertTrue(np.any(np.isclose(FP4_LEVELS, -val)))

    def test_tile_quantization_bounded_error(self):
        x = self.rng.normal(0.0, 1.0, size=(1024,))
        q, scales, shape = self.quantizer.quantize_tensor(x)
        deq = self.quantizer.dequantize_tensor(q, scales, shape)
        sig = np.mean(x ** 2)
        noise = np.mean((x - deq) ** 2)
        sqnr = 10.0 * np.log10(sig / max(1e-12, noise))
        self.assertGreater(sqnr, 18.0)

    def test_nibble_packing_roundtrip(self):
        indices = self.rng.integers(0, 15, size=256)
        packed = self.quantizer.pack_fp4(indices)
        self.assertEqual(len(packed), 128)
        unpacked = self.quantizer.unpack_fp4(packed, 256)
        np.testing.assert_array_equal(indices, unpacked)

    def test_stochastic_rounding_expectation(self):
        val = 2.4
        runs = 30000
        samples = np.zeros(runs)
        tile = np.array([val], dtype=np.float32)
        for i in range(runs):
            q, scale = self.quantizer.quantize_tile(tile, stochastic=True, rng=self.rng)
            samples[i] = q[0] * scale
        mean_val = float(np.mean(samples))
        self.assertAlmostEqual(mean_val, val, delta=0.03)

    def test_scaled_gemm_execution(self):
        engine = NVFP4GEMMEngine(block_size=16)
        M, K, N = 64, 128, 64
        A = self.rng.normal(0.0, 0.5, size=(M, K)).astype(np.float32)
        B = self.rng.normal(0.0, 0.5, size=(K, N)).astype(np.float32)
        C_fp32 = A @ B
        C_nvfp4, metrics = engine.scaled_gemm(A, B)
        self.assertEqual(C_nvfp4.shape, (M, N))
        rel_err = np.linalg.norm(C_nvfp4 - C_fp32) / np.linalg.norm(C_fp32)
        self.assertLess(rel_err, 0.15)
        self.assertGreater(metrics["compression_ratio"], 2.5)

    def test_cli_subcommands(self):
        res_status = subprocess.run([str(BIN_PATH), "status"], capture_output=True, text=True)
        self.assertEqual(res_status.returncode, 0)
        self.assertIn("NVFP4", res_status.stdout)

        res_verify = subprocess.run([str(BIN_PATH), "verify"], capture_output=True, text=True)
        self.assertEqual(res_verify.returncode, 0)
        self.assertIn("verified successfully", res_verify.stdout)

        res_bench = subprocess.run([str(BIN_PATH), "benchmark"], capture_output=True, text=True)
        self.assertEqual(res_bench.returncode, 0)
        self.assertIn("Benchmark", res_bench.stdout)


if __name__ == "__main__":
    unittest.main()
