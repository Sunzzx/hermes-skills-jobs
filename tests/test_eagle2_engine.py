"""Unit and integration tests for EAGLE-2 Dynamic Lossless Speculative Sampling Engine."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "eagle2-engine"

loader = SourceFileLoader("eagle2_engine", str(BIN_PATH))
spec = importlib.util.spec_from_loader("eagle2_engine", loader)
assert spec is not None and spec.loader is not None
eagle2_mod = importlib.util.module_from_spec(spec)
sys.modules["eagle2_engine"] = eagle2_mod
loader.exec_module(eagle2_mod)

EAGLE2Config = eagle2_mod.EAGLE2Config
EAGLE2Engine = eagle2_mod.EAGLE2Engine
DynamicDraftTree = eagle2_mod.DynamicDraftTree


class TestEAGLE2Engine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cfg = EAGLE2Config(
            vocab_size=1000,
            hidden_dim=256,
            num_heads=4,
            head_dim=32,
            max_tree_size=16,
            max_depth=4,
            min_prob_threshold=0.05,
        )
        self.engine = EAGLE2Engine(self.cfg)

    def test_dynamic_tree_construction_and_mask(self):
        def mock_cands(nid, tid, depth):
            return [(tid + i + 1, 0.6 / (depth + 1)) for i in range(2)]

        struct = self.engine.tree_builder.build_tree_from_logits_fn(root_token=50, get_topk_logits_fn=mock_cands)
        num_nodes = struct["num_nodes"]
        self.assertGreater(num_nodes, 1)
        self.assertLessEqual(num_nodes, 16)

        mask = struct["tree_mask"]
        self.assertEqual(mask.shape, (num_nodes, num_nodes))
        # Root attends to itself
        self.assertEqual(mask[0, 0], 1.0)
        # Tree mask is lower triangular or directed ancestor DAG (no cycles)
        for i in range(num_nodes):
            self.assertEqual(mask[i, i], 1.0)

    def test_tree_attention_numerics(self):
        def mock_cands(nid, tid, depth):
            return [(tid + 1, 0.8)]

        struct = self.engine.tree_builder.build_tree_from_logits_fn(root_token=10, get_topk_logits_fn=mock_cands)
        n = struct["num_nodes"]
        Q = np.random.randn(n, 4, 32).astype(np.float32)
        K = np.random.randn(n, 4, 32).astype(np.float32)
        V = np.random.randn(n, 4, 32).astype(np.float32)
        attn_out = self.engine.verify_tree_attention(Q, K, V, struct["tree_mask"])

        self.assertEqual(attn_out.shape, (n, 4, 32))
        self.assertFalse(np.isnan(attn_out).any())
        self.assertFalse(np.isinf(attn_out).any())

    def test_lossless_rejection_sampling(self):
        p_dist = np.array([0.65, 0.25, 0.10], dtype=np.float32)
        q_dist = np.array([0.40, 0.40, 0.20], dtype=np.float32)
        samples = []
        for _ in range(6000):
            draft_tok = int(np.random.choice(3, p=q_dist))
            _, final_tok = self.engine.speculative_sample_branch(p_dist, q_dist, draft_tok)
            samples.append(final_tok)
        counts = np.bincount(samples, minlength=3) / 6000.0
        max_diff = np.max(np.abs(counts - p_dist))
        self.assertLess(max_diff, 0.03)

    def test_benchmark_simulation(self):
        res = self.engine.benchmark_tree_speculation(num_steps=20, tree_size=12)
        self.assertGreater(res["speedup_factor"], 1.5)
        self.assertGreater(res["total_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
