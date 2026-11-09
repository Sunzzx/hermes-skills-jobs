#!/usr/bin/env python3
"""Unit tests for DuoAttention Head-Specialization KV Cache Engine."""
import unittest
import sys
import os
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Load CLI/engine binary directly via SourceFileLoader
bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "duo-attention-engine"))
duo_mod = SourceFileLoader("duo_attention_engine", bin_path).load_module()

DuoAttentionEngine = duo_mod.DuoAttentionEngine
DuoAttentionConfig = duo_mod.DuoAttentionConfig
run_benchmark = duo_mod.run_benchmark


class TestDuoAttentionEngine(unittest.TestCase):
    def setUp(self):
        self.config = DuoAttentionConfig(
            num_heads=16,
            head_dim=64,
            retrieval_ratio=0.25,
            sink_size=4,
            recent_size=32,
            max_seq_len=4096
        )
        self.engine = DuoAttentionEngine(self.config)

    def test_head_partitioning(self):
        """Verify heads are accurately split into retrieval and streaming sets."""
        retrieval = self.engine.retrieval_head_indices
        streaming = self.engine.streaming_head_indices
        # 16 heads * 0.25 = 4 retrieval heads
        self.assertEqual(len(retrieval), 4)
        self.assertEqual(len(streaming), 12)
        self.assertEqual(len(set(retrieval).intersection(set(streaming))), 0)
        self.assertEqual(len(retrieval) + len(streaming), 16)

    def test_prefill_retention(self):
        """Verify prefill retains full KV for retrieval and bounded KV for streaming."""
        seq_len = 512
        stats = self.engine.prefill(seq_len=seq_len)
        self.assertEqual(self.engine.cached_seq_len, seq_len)
        self.assertGreater(stats["memory_savings_pct"], 60.0)
        self.assertGreater(stats["compression_ratio"], 2.5)

        # Check retrieval heads hold full cache
        for h in self.engine.retrieval_head_indices:
            self.assertEqual(len(self.engine.retrieval_kv_cache[h]), seq_len)

        # Check streaming heads hold only sink + recent
        for h in self.engine.streaming_head_indices:
            self.assertEqual(len(self.engine.streaming_sink_cache[h]), self.config.sink_size)
            self.assertEqual(len(self.engine.streaming_recent_cache[h]), self.config.recent_size)

    def test_decode_step(self):
        """Verify decode step advances token position and rolling buffer correctly."""
        self.engine.prefill(seq_len=256)
        outputs, stats = self.engine.decode_step(step_idx=0)
        self.assertEqual(len(outputs), self.config.num_heads)
        self.assertEqual(self.engine.cached_seq_len, 257)
        self.assertIn("memory_savings_pct", stats)
        self.assertIn("total_dot_products", stats)

    def test_status_endpoint(self):
        """Verify engine status returns expected format and keys."""
        status = self.engine.get_status()
        self.assertEqual(status["engine"], "DuoAttention")
        self.assertEqual(status["status"], "ready")
        self.assertEqual(len(status["retrieval_heads"]), 4)
        self.assertEqual(len(status["streaming_heads"]), 12)

    def test_benchmark_runner(self):
        """Verify benchmark helper returns valid metrics."""
        bench = run_benchmark(seq_len=1024, gen_tokens=4)
        self.assertEqual(bench["context_length"], 1024)
        self.assertEqual(bench["generated_tokens"], 4)
        self.assertGreater(bench["memory_savings_pct"], 60.0)
        self.assertGreater(bench["avg_decode_step_ms"], 0.0)


if __name__ == '__main__':
    unittest.main()
