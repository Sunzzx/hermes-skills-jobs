#!/usr/bin/env python3
"""Unit tests for Native Sparse Attention (NSA) Engine."""
import unittest
import sys
import os
from pathlib import Path
from importlib.machinery import SourceFileLoader
import numpy as np

BIN_PATH = Path(__file__).resolve().parent.parent / "bin" / "nsa-engine"
nsa_mod = SourceFileLoader("nsa_engine", str(BIN_PATH)).load_module()

class TestNativeSparseAttention(unittest.TestCase):
    def setUp(self):
        self.nsa = nsa_mod.NativeSparseAttention(d_model=32, n_heads=2, block_size=8, top_k_blocks=2, window_size=16)

    def test_forward_shape(self):
        q = np.random.randn(1, 32).astype(np.float32)
        k = np.random.randn(128, 32).astype(np.float32)
        v = np.random.randn(128, 32).astype(np.float32)
        out, stats = self.nsa.forward(q, k, v)
        self.assertEqual(out.shape, (1, 32))
        self.assertIn("compression_ratio", stats)
        self.assertGreater(stats["compression_ratio"], 1.0)

    def test_multi_query(self):
        q = np.random.randn(4, 32).astype(np.float32)
        k = np.random.randn(64, 32).astype(np.float32)
        v = np.random.randn(64, 32).astype(np.float32)
        out, stats = self.nsa.forward(q, k, v)
        self.assertEqual(out.shape, (4, 32))

    def test_simulation(self):
        res = nsa_mod.simulate_inference(seq_len=256, d_model=32, n_heads=2, block_size=8, top_k_blocks=2)
        self.assertIn("memory_savings_pct", res)
        self.assertGreater(res["memory_savings_pct"], 0.0)
        self.assertEqual(res["sequence_length"], 256)

if __name__ == "__main__":
    unittest.main()
