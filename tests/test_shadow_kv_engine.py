#!/usr/bin/env python3
"""Unit tests for ShadowKV Offloaded Shadow Summarization Engine."""
import unittest
import sys
import os
from pathlib import Path
from importlib.machinery import SourceFileLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
bin_path = REPO_ROOT / "bin" / "shadow-kv-engine"

shadow_kv_module = SourceFileLoader("shadow_kv_engine", str(bin_path)).load_module()
ShadowKVConfig = shadow_kv_module.ShadowKVConfig
ShadowKVEngine = shadow_kv_module.ShadowKVEngine
benchmark = shadow_kv_module.benchmark


class TestShadowKVEngine(unittest.TestCase):
    def setUp(self):
        self.config = ShadowKVConfig(
            num_heads=8,
            head_dim=64,
            chunk_size=32,
            shadow_dim=8,
            topk_chunks=4,
            recent_tokens=64,
            sink_tokens=4,
        )
        self.engine = ShadowKVEngine(self.config)

    def test_prefill_and_compression(self):
        seq_len = 512
        stats = self.engine.prefill(seq_len=seq_len)
        self.assertIn("prefill_time_ms", stats)
        self.assertEqual(stats["total_tokens"], seq_len)
        self.assertGreater(stats["compression_ratio"], 1.0)
        self.assertEqual(len(self.engine.fast_cache[0]), self.config.sink_tokens + self.config.recent_tokens)

    def test_decode_step(self):
        self.engine.prefill(seq_len=256)
        output_embed, step_stats = self.engine.decode_step(step_idx=0)
        self.assertEqual(len(output_embed), self.config.head_dim)
        self.assertIn("retrieved_chunks", step_stats)
        self.assertEqual(step_stats["avg_retrieved_per_head"], self.config.topk_chunks)

    def test_benchmark_execution(self):
        res = benchmark(seq_len=256, gen_tokens=4)
        self.assertEqual(res["engine"], "ShadowKV")
        self.assertGreater(res["decode_throughput_tok_s"], 0.0)
        self.assertEqual(res["gen_tokens"], 4)


if __name__ == '__main__':
    unittest.main()
