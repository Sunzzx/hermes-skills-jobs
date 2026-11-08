"""Unit and integration tests for TRIAD Speculative Decoding & Chunk KV Pruning Engine."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "triad-engine"

loader = SourceFileLoader("triad_engine", str(BIN_PATH))
spec = importlib.util.spec_from_loader("triad_engine", loader)
assert spec is not None and spec.loader is not None
triad_mod = importlib.util.module_from_spec(spec)
sys.modules["triad_engine"] = triad_mod
loader.exec_module(triad_mod)

TRIADConfig = triad_mod.TRIADConfig
TRIADEngine = triad_mod.TRIADEngine
TRIADKVCache = triad_mod.TRIADKVCache


class TestTRIADEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cfg = TRIADConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            chunk_size=8,
            kv_budget=64,
            num_sink_tokens=4,
            recent_window=16,
            speculative_window=4,
        )
        self.engine = TRIADEngine(self.cfg)

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

    def test_chunk_pruning_boundary_and_preservation(self):
        cache = self.engine.cache
        n_tokens = 160
        k = np.random.randn(n_tokens, 4, 32).astype(np.float32)
        v = np.random.randn(n_tokens, 4, 32).astype(np.float32)
        tokens = list(range(n_tokens))
        cache.append(k, v, tokens)

        q = np.random.randn(4, 32).astype(np.float32)
        res = cache.prune(q)

        # 1. Total length must not exceed budget
        self.assertLessEqual(cache.current_len, self.cfg.kv_budget)
        self.assertEqual(len(cache.token_ids), cache.current_len)

        # 2. Sink tokens (0, 1, 2, 3) must be preserved
        for sink in range(self.cfg.num_sink_tokens):
            self.assertIn(sink, cache.token_ids)

        # 3. Recent window tokens must be preserved
        for rec in range(n_tokens - self.cfg.recent_window, n_tokens):
            self.assertIn(rec, cache.token_ids)

        # 4. Pruning stats correctly reported
        self.assertGreater(res["pruned_tokens"], 0)
        self.assertEqual(res["pruned_tokens"], n_tokens - cache.current_len)

    def test_speculative_verification_flow(self):
        prefix = [10, 20, 30]
        draft = [40, 50, 60, 70]
        res = self.engine.verify_speculative_chunk(prefix, draft)

        self.assertIn("accepted_tokens", res)
        self.assertIn("acceptance_rate", res)
        self.assertGreaterEqual(len(res["accepted_tokens"]), 1)
        self.assertLessEqual(len(res["accepted_tokens"]), len(draft) + 1)

    def test_end_to_end_benchmark_numerics(self):
        bench_res = self.engine.benchmark(num_steps=10, prompt_len=64, gamma=4)
        self.assertGreater(bench_res["total_tokens_generated"], 10)
        self.assertGreater(bench_res["tokens_per_step"], 1.0)
        self.assertGreater(bench_res["simulated_throughput_tok_s"], 0.0)
        self.assertLessEqual(bench_res["final_kv_cache_size"], self.cfg.kv_budget)


if __name__ == '__main__':
    unittest.main()
