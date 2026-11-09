"""Unit and integration tests for SparQ-Attention Engine."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "sparq-attention-engine"

loader = SourceFileLoader("sparq_attention_engine", str(BIN_PATH))
spec = importlib.util.spec_from_loader("sparq_attention_engine", loader)
assert spec is not None and spec.loader is not None
sparq_mod = importlib.util.module_from_spec(spec)
sys.modules["sparq_attention_engine"] = sparq_mod
loader.exec_module(sparq_mod)

SparQConfig = sparq_mod.SparQConfig
SparQAttentionEngine = sparq_mod.SparQAttentionEngine
SparQKVCache = sparq_mod.SparQKVCache


class TestSparQAttentionEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cfg = SparQConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            kv_budget=64,
            top_k_tokens=16,
            feature_dim_ratio=0.25,
            num_sink_tokens=4,
            recent_window=16,
        )
        self.engine = SparQAttentionEngine(self.cfg)

    def test_cache_append_and_prune_invariants(self):
        cache = self.engine.cache
        k = np.random.randn(100, 4, 32).astype(np.float32)
        v = np.random.randn(100, 4, 32).astype(np.float32)
        cache.append(k, v, list(range(100)))
        self.assertEqual(cache.seq_len, 100)

        evicted = cache.prune_to_budget()
        self.assertEqual(evicted, 36)
        self.assertEqual(cache.seq_len, 64)

        # Sinks check
        for sink in range(4):
            self.assertIn(sink, cache.token_ids)
        # Recent window check
        self.assertIn(99, cache.token_ids)

    def test_sparsified_attention_correctness(self):
        k = np.random.randn(64, 4, 32).astype(np.float32)
        v = np.random.randn(64, 4, 32).astype(np.float32)
        self.engine.cache.append(k, v, list(range(64)))

        q = np.random.randn(4, 32).astype(np.float32)
        out, stats = self.engine.compute_sparsified_attention(q)

        self.assertEqual(out.shape, (4, 32))
        self.assertFalse(np.isnan(out).any())
        self.assertGreater(stats["bandwidth_saving_ratio"], 0.0)
        self.assertGreaterEqual(stats["theoretical_speedup"], 1.0)
        self.assertLessEqual(stats["sparsity_ratio"], 1.0)

    def test_benchmark_simulation(self):
        res = self.engine.benchmark(num_steps=10, prompt_len=128)
        self.assertEqual(res["num_steps"], 10)
        self.assertGreater(res["tokens_per_second"], 0.0)
        self.assertGreater(res["avg_bandwidth_savings_pct"], 0.0)
        self.assertLessEqual(res["final_cache_size"], self.cfg.kv_budget)


if __name__ == "__main__":
    unittest.main()
