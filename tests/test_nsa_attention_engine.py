"""Unit and integration tests for DeepSeek-V3 Native Sparse Attention (NSA) Engine."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "nsa-attention-engine"

loader = SourceFileLoader("nsa_attention_engine", str(BIN_PATH))
spec = importlib.util.spec_from_loader("nsa_attention_engine", loader)
assert spec is not None and spec.loader is not None
nsa_mod = importlib.util.module_from_spec(spec)
sys.modules["nsa_attention_engine"] = nsa_mod
loader.exec_module(nsa_mod)

NSAConfig = nsa_mod.NSAConfig
NativeSparseAttentionEngine = nsa_mod.NativeSparseAttentionEngine


class TestNativeSparseAttentionEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cfg = NSAConfig(
            seq_len=256,
            d_model=256,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            block_size=32,
            top_k_blocks=2,
            sliding_window_size=64,
            use_causal_mask=True,
        )
        self.engine = NativeSparseAttentionEngine(self.cfg)

    def test_compress_kv_blocks(self):
        K = np.random.randn(256, 2, 64).astype(np.float32)
        V = np.random.randn(256, 2, 64).astype(np.float32)
        k_blocks, v_blocks, num_blocks = self.engine.compress_kv_blocks(K, V)
        self.assertEqual(num_blocks, 8)
        self.assertEqual(k_blocks.shape, (8, 2, 64))
        self.assertEqual(v_blocks.shape, (8, 2, 64))
        # Verify block average
        expected_block0 = np.mean(K[:32, 0, :], axis=0)
        np.testing.assert_allclose(k_blocks[0, 0, :], expected_block0, rtol=1e-5)

    def test_select_topk_blocks(self):
        K = np.random.randn(256, 2, 64).astype(np.float32)
        V = np.random.randn(256, 2, 64).astype(np.float32)
        k_blocks, _, num_blocks = self.engine.compress_kv_blocks(K, V)
        Q = np.random.randn(4, 64).astype(np.float32)
        selected = self.engine.select_topk_blocks(Q, k_blocks, query_idx=100, num_blocks=num_blocks)
        self.assertEqual(selected.shape, (4, 2))
        # Block index must not exceed current block (query_idx=100 -> block 3)
        self.assertTrue((selected <= 3).all())

    def test_forward_shape_and_numerical_stability(self):
        seq_len = 128
        Q = np.random.randn(seq_len, 4, 64).astype(np.float32)
        K = np.random.randn(seq_len, 2, 64).astype(np.float32)
        V = np.random.randn(seq_len, 2, 64).astype(np.float32)
        out, stats = self.engine.forward(Q, K, V)

        self.assertEqual(out.shape, (seq_len, 4, 64))
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())
        self.assertGreater(stats["sparsity_ratio"], 0.0)
        self.assertLessEqual(stats["sparsity_ratio"], 1.0)

    def test_sparsity_scaling(self):
        res = self.engine.benchmark_dense_vs_sparse(seq_len=256)
        self.assertIn("sparsity_ratio", res)
        self.assertIn("flops_reduction", res)
        self.assertGreater(res["flops_reduction"], 0.3)


if __name__ == "__main__":
    unittest.main()
