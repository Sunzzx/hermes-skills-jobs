#!/usr/bin/env python3
"""Unit tests for ChunkKV Dynamic KV Cache Sparsifier Engine."""
import unittest
import sys
import os
from pathlib import Path

# Add bin/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
from importlib.machinery import SourceFileLoader

chunk_kv_module = SourceFileLoader(
    "chunk_kv_engine",
    str(Path(__file__).parent.parent / "bin" / "chunk-kv-engine")
).load_module()

ChunkKVConfig = chunk_kv_module.ChunkKVConfig
ChunkKVEngine = chunk_kv_module.ChunkKVEngine
ChunkImportanceScorer = chunk_kv_module.ChunkImportanceScorer
run_benchmark = chunk_kv_module.run_benchmark


class TestChunkKVEngine(unittest.TestCase):
    def setUp(self):
        self.config = ChunkKVConfig(
            num_heads=8,
            head_dim=64,
            chunk_size=32,
            topk_chunks=3,
            sink_tokens=4,
            recent_tokens=32,
            decay_factor=0.9,
            random_seed=123
        )
        self.engine = ChunkKVEngine(self.config)

    def test_chunk_scorer(self):
        scorer = ChunkImportanceScorer(dim=64, decay=0.9)
        q = [0.1] * 64
        c = [0.1] * 64
        score_near = scorer.score_chunk(q, c, chunk_pos=9, total_chunks=10)
        score_far = scorer.score_chunk(q, c, chunk_pos=0, total_chunks=10)
        self.assertGreater(score_near, score_far)

    def test_prefill_and_compression(self):
        stats = self.engine.prefill(seq_len=1024)
        self.assertEqual(stats["total_tokens"], 1024)
        self.assertEqual(stats["num_chunks"], 32)
        self.assertGreater(stats["memory_savings_pct"], 50.0)
        self.assertGreater(stats["compression_ratio"], 1.0)

    def test_decode_step(self):
        self.engine.prefill(seq_len=512)
        output_repr, step_stats = self.engine.decode_step(step_idx=0)
        self.assertEqual(len(output_repr), self.config.head_dim)
        self.assertIn("step_time_ms", step_stats)
        self.assertIn("total_retrieved_tokens", step_stats)
        self.assertGreater(step_stats["total_retrieved_tokens"], 0)

    def test_status(self):
        status = self.engine.get_status()
        self.assertEqual(status["engine"], "ChunkKV")
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["num_heads"], self.config.num_heads)

    def test_benchmark_runner(self):
        bench = run_benchmark(seq_len=256, gen_tokens=2)
        self.assertEqual(bench["engine"], "ChunkKV")
        self.assertEqual(bench["context_length"], 256)
        self.assertGreater(bench["throughput_tok_s"], 0)


if __name__ == "__main__":
    unittest.main()
