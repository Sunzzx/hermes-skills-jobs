#!/usr/bin/env python3
"""Unit tests for Scissorhands KV Cache Pruning Engine."""
import unittest
import sys
import os
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Load bin/scissorhands-kv-engine dynamically
bin_path = Path(__file__).parent.parent / "bin" / "scissorhands-kv-engine"
scissorhands_module = SourceFileLoader(
    "scissorhands_kv_engine",
    str(bin_path)
).load_module()

ScissorhandsConfig = scissorhands_module.ScissorhandsConfig
ScissorhandsKVCacheEngine = scissorhands_module.ScissorhandsKVCacheEngine
PersistenceScorer = scissorhands_module.PersistenceScorer
run_benchmark = scissorhands_module.run_benchmark


class TestScissorhandsKVCacheEngine(unittest.TestCase):
    def setUp(self):
        self.config = ScissorhandsConfig(
            num_heads=4,
            head_dim=32,
            max_capacity=128,
            window_size=16,
            sink_tokens=4,
            decay_factor=0.9,
            importance_threshold=0.05,
            random_seed=42
        )
        self.engine = ScissorhandsKVCacheEngine(self.config)

    def test_persistence_scorer(self):
        scorer = PersistenceScorer(decay_factor=0.9, threshold=0.05)
        history = {10: 1.0, 20: 0.1}
        attn_weights = [0.5, 0.05]
        token_indices = [10, 20]
        updated = scorer.update_persistence(history, attn_weights, token_indices)
        self.assertAlmostEqual(updated[10], 1.0 * 0.9 + 0.5)
        self.assertAlmostEqual(updated[20], 0.1 * 0.9 + 0.05)

        pivotal = scorer.identify_pivotal_tokens(updated, top_k=1)
        self.assertEqual(pivotal, [10])

    def test_prefill_budget_bounding(self):
        # Prefill 512 tokens with capacity budget of 128
        stats = self.engine.prefill(seq_len=512)
        self.assertEqual(stats["total_tokens"], 512)
        self.assertLessEqual(stats["cached_tokens_per_head"], self.config.max_capacity)
        self.assertGreater(stats["memory_savings_pct"], 50.0)
        self.assertGreater(stats["compression_ratio"], 2.0)

    def test_decode_step(self):
        self.engine.prefill(seq_len=256)
        output_repr, step_stats = self.engine.decode_step(step_idx=0)
        self.assertEqual(len(output_repr), self.config.head_dim)
        self.assertIn("step_time_ms", step_stats)
        self.assertIn("total_cached_tokens", step_stats)
        self.assertLessEqual(step_stats["cache_size_per_head"], self.config.max_capacity)

    def test_status(self):
        status = self.engine.get_status()
        self.assertEqual(status["engine"], "Scissorhands-KV-Engine")
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["max_capacity"], self.config.max_capacity)

    def test_benchmark_runner(self):
        bench = run_benchmark(seq_len=256, gen_tokens=4, max_capacity=64, num_heads=2, head_dim=16)
        self.assertEqual(bench["engine"], "Scissorhands-KV-Engine")
        self.assertEqual(bench["context_length"], 256)
        self.assertGreater(bench["throughput_tok_s"], 0)
        self.assertGreater(bench["memory_savings_pct"], 0)


if __name__ == "__main__":
    unittest.main()
