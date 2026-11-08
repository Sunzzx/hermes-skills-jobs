"""Unit and integration tests for MAGIC-Decoding Speculative Tree Generation & KV Saliency Pruning Engine."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "magic-decoding-engine"

loader = SourceFileLoader("magic_decoding_engine", str(BIN_PATH))
spec = importlib.util.spec_from_loader("magic_decoding_engine", loader)
assert spec is not None and spec.loader is not None
magic_mod = importlib.util.module_from_spec(spec)
sys.modules["magic_decoding_engine"] = magic_mod
loader.exec_module(magic_mod)

MAGICConfig = magic_mod.MAGICConfig
MAGICDecodingEngine = magic_mod.MAGICDecodingEngine
MAGICKVCache = magic_mod.MAGICKVCache


class TestMAGICDecodingEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cfg = MAGICConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            kv_budget=64,
            num_sink_tokens=4,
            recent_window=16,
            tree_depth=3,
            tree_branching=2,
        )
        self.engine = MAGICDecodingEngine(self.cfg)

    def test_cache_append_and_invariants(self):
        cache = self.engine.cache
        k = np.random.randn(32, 4, 32).astype(np.float32)
        v = np.random.randn(32, 4, 32).astype(np.float32)
        tokens = list(range(32))
        cache.append(k, v, tokens)

        self.assertEqual(cache.current_len, 32)
        self.assertEqual(len(cache.token_ids), 32)
        self.assertEqual(cache.k_cache.shape, (32, 4, 32))
        self.assertEqual(cache.v_cache.shape, (32, 4, 32))

    def test_saliency_pruning_boundary_and_preservation(self):
        cache = self.engine.cache
        n_tokens = 160
        k = np.random.randn(n_tokens, 4, 32).astype(np.float32)
        v = np.random.randn(n_tokens, 4, 32).astype(np.float32)
        tokens = list(range(n_tokens))
        cache.append(k, v, tokens)

        q = np.random.randn(4, 32).astype(np.float32)
        res = cache.prune(q)

        self.assertLessEqual(cache.current_len, self.cfg.kv_budget)
        self.assertEqual(len(cache.token_ids), cache.current_len)

        for sink in range(self.cfg.num_sink_tokens):
            self.assertIn(sink, cache.token_ids)

        for rec in range(n_tokens - self.cfg.recent_window, n_tokens):
            self.assertIn(rec, cache.token_ids)

        self.assertGreater(res["pruned_tokens"], 0)
        self.assertEqual(res["pruned_tokens"], n_tokens - cache.current_len)
        self.assertGreater(res["compression_ratio"], 1.0)

    def test_candidate_tree_topology(self):
        tree = self.engine.build_candidate_tree(root_token=7, depth=3, branching=2)
        # 1 root + 2 + 4 + 8 = 15 nodes for depth 3 branching 2
        self.assertEqual(len(tree), 15)
        self.assertIsNone(tree[0].parent_idx)
        self.assertEqual(tree[0].depth, 0)

        paths = self.engine.extract_tree_paths(tree)
        self.assertEqual(len(paths), 8)  # 2^3 leaves
        for path in paths:
            self.assertEqual(path[0].token_id, 7)
            self.assertEqual(len(path), 4)  # root + 3 levels

    def test_speculative_tree_verification_flow(self):
        prefix = [10, 20, 30]
        tree = self.engine.build_candidate_tree(root_token=30, depth=3, branching=2)
        res = self.engine.verify_candidate_tree(prefix, tree)

        self.assertIn("accepted_tokens", res)
        self.assertIn("acceptance_rate", res)
        self.assertGreaterEqual(len(res["accepted_tokens"]), 1)
        self.assertLessEqual(len(res["accepted_tokens"]), self.cfg.tree_depth + 1)

    def test_end_to_end_benchmark_numerics(self):
        bench_res = self.engine.benchmark(num_steps=10, prompt_len=64)
        self.assertGreater(bench_res["total_tokens_generated"], 10)
        self.assertGreater(bench_res["tokens_per_step"], 1.0)
        self.assertGreater(bench_res["simulated_throughput_tok_s"], 0.0)
        self.assertLessEqual(bench_res["final_kv_cache_size"], self.cfg.kv_budget)


if __name__ == "__main__":
    unittest.main()
