#!/usr/bin/env python3
"""Unit tests for MagicPIG Multi-Scale Top-K KV Sparsification Engine."""
import unittest
import sys
import os
from pathlib import Path
from importlib.machinery import SourceFileLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
bin_path = REPO_ROOT / "bin" / "magic-kv-engine"

magic_mod = SourceFileLoader("magic_kv_engine", str(bin_path)).load_module()
MagicPIGConfig = magic_mod.MagicPIGConfig
MagicPIGEngine = magic_mod.MagicPIGEngine
MultiScaleHashProjector = magic_mod.MultiScaleHashProjector
run_benchmark = magic_mod.run_benchmark


class TestMagicPIGEngine(unittest.TestCase):
    def setUp(self):
        self.config = MagicPIGConfig(
            num_heads=8,
            head_dim=64,
            bucket_size=16,
            num_hash_functions=6,
            topk_buckets=4,
            sink_tokens=4,
            recent_tokens=32,
            random_seed=123,
        )
        self.engine = MagicPIGEngine(self.config)

    def test_projector_hash(self):
        projector = MultiScaleHashProjector(dim=64, num_hashes=6, seed=123)
        v1 = [0.1] * 64
        v2 = [0.1] * 64
        v_diff = [-0.1] * 64
        h1 = projector.hash_vector(v1)
        h2 = projector.hash_vector(v2)
        h_diff = projector.hash_vector(v_diff)
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h_diff)

    def test_prefill_and_compression(self):
        seq_len = 512
        stats = self.engine.prefill(seq_len=seq_len)
        self.assertIn("prefill_time_ms", stats)
        self.assertEqual(stats["total_tokens"], seq_len)
        self.assertGreater(stats["memory_savings_pct"], 50.0)
        self.assertGreater(stats["compression_ratio"], 2.0)
        self.assertEqual(len(self.engine.sink_cache[0]), self.config.sink_tokens)
        self.assertEqual(len(self.engine.recent_cache[0]), self.config.recent_tokens)
        self.assertEqual(len(self.engine.buckets[0]), seq_len // self.config.bucket_size)

    def test_decode_step(self):
        self.engine.prefill(seq_len=256)
        output_repr, step_stats = self.engine.decode_step(step_idx=0)
        self.assertEqual(len(output_repr), self.config.head_dim)
        self.assertIn("total_sampled_tokens", step_stats)
        self.assertEqual(step_stats["current_pos"], 256)
        self.assertGreater(step_stats["avg_sampled_per_head"], 0.0)

    def test_status_endpoint(self):
        status = self.engine.get_status()
        self.assertEqual(status["engine"], "MagicPIG")
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["num_heads"], self.config.num_heads)

    def test_benchmark_execution(self):
        res = run_benchmark(seq_len=256, gen_tokens=4)
        self.assertEqual(res["engine"], "MagicPIG")
        self.assertGreater(res["throughput_tok_s"], 0.0)
        self.assertEqual(res["generated_tokens"], 4)


if __name__ == '__main__':
    unittest.main()
