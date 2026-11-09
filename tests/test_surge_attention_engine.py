#!/usr/bin/env python3
"""Unit tests for SurgeAttention Dynamic Head-Varying Sparsification Engine."""
import unittest
import sys
import os
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Load CLI/engine binary directly via SourceFileLoader to handle hyphens cleanly
bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "surge-attention-engine"))
surge_mod = SourceFileLoader("surge_attention_engine", bin_path).load_module()

SurgeAttentionEngine = surge_mod.SurgeAttentionEngine
SurgeAttentionConfig = surge_mod.SurgeAttentionConfig
run_benchmark = surge_mod.run_benchmark


class TestSurgeAttentionEngine(unittest.TestCase):
    def setUp(self):
        self.config = SurgeAttentionConfig(
            embed_dim=2048,
            num_heads=16,
            head_dim=128,
            min_tokens=64,
            max_tokens=512,
            sink_size=4,
            recent_size=32,
            outlier_threshold=2.0
        )
        self.engine = SurgeAttentionEngine(self.config)

    def test_head_allocations_skew(self):
        """Verify head allocations adhere to head-varying non-uniformity."""
        allocs = self.engine.head_allocations
        self.assertEqual(len(allocs), self.config.num_heads)
        # First heads should have minimum budget, last heads maximum budget
        self.assertEqual(allocs[0], self.config.min_tokens)
        self.assertEqual(allocs[-1], self.config.max_tokens)
        self.assertLess(allocs[0], allocs[-1])

    def test_outlier_detection(self):
        """Verify outlier channels are correctly identified by Z-score."""
        # Normal vector with one massive outlier
        vec = [1.0] * 127 + [100.0]
        outliers = self.engine.detect_outlier_channels(vec)
        self.assertIn(127, outliers)

    def test_prefill_and_decode(self):
        """Verify prefill and single-step decoding bandwidth savings."""
        seq_len = 1024
        prefill_stats = self.engine.prefill(seq_len=seq_len)
        self.assertEqual(prefill_stats["cached_tokens"], seq_len)

        outputs, stats = self.engine.decode_step(step_idx=0)
        self.assertEqual(len(outputs), self.config.num_heads)
        self.assertIn("bandwidth_savings_pct", stats)
        self.assertIn("compression_ratio", stats)
        self.assertGreater(stats["bandwidth_savings_pct"], 50.0)
        self.assertGreater(stats["compression_ratio"], 2.0)

    def test_benchmark_runner(self):
        """Verify benchmark execution returns consistent metric structure."""
        res = run_benchmark(seq_len=512, gen_tokens=5)
        self.assertEqual(res["context_length"], 512)
        self.assertEqual(res["generated_tokens"], 5)
        self.assertIn("avg_bandwidth_savings_pct", res)
        self.assertIn("avg_compression_ratio", res)

    def test_status_endpoint(self):
        """Verify status JSON dict returns expected keys."""
        status = self.engine.get_status()
        self.assertEqual(status["engine"], "SurgeAttention")
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["num_heads"], 16)


if __name__ == '__main__':
    unittest.main()
