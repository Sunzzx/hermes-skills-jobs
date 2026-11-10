#!/usr/bin/env python3
"""Unit tests for LoSparse Sub-Sentence Speculative KV Cache Compactor Engine."""
import unittest
import sys
import os
from pathlib import Path

# Add bin/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
from importlib.machinery import SourceFileLoader

losparse_module = SourceFileLoader(
    "losparse_kv_cache",
    str(Path(__file__).parent.parent / "bin" / "losparse-kv-engine")
).load_module()

LoSparseConfig = losparse_module.LoSparseConfig
LoSparseKVCacheEngine = losparse_module.LoSparseKVCacheEngine
SubSentenceScorer = losparse_module.SubSentenceScorer
run_benchmark = losparse_module.run_benchmark


class TestLoSparseKVCacheEngine(unittest.TestCase):
    def setUp(self):
        self.config = LoSparseConfig(
            num_heads=8,
            head_dim=64,
            sub_sentence_len=16,
            topk_sub_sentences=4,
            sink_tokens=4,
            recent_tokens=32,
            speculative_lookahead=2,
            random_seed=42
        )
        self.engine = LoSparseKVCacheEngine(self.config)

    def test_scorer_relevance(self):
        scorer = SubSentenceScorer(dim=64, relevance_threshold=0.15)
        q = [0.05] * 64
        s = [0.05] * 64
        score_near = scorer.compute_span_relevance(q, s, span_idx=9, total_spans=10)
        score_far = scorer.compute_span_relevance(q, s, span_idx=0, total_spans=10)
        self.assertGreater(score_near, score_far)

    def test_prefill_and_compression(self):
        stats = self.engine.prefill(seq_len=1024)
        self.assertEqual(stats["total_tokens"], 1024)
        self.assertEqual(stats["num_spans"], 64)
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
        self.assertEqual(status["engine"], "LoSparse-KV-Cache")
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["num_heads"], self.config.num_heads)

    def test_benchmark_runner(self):
        bench = run_benchmark(seq_len=256, gen_tokens=2, num_heads=4, head_dim=32)
        self.assertEqual(bench["engine"], "LoSparse-KV-Cache")
        self.assertEqual(bench["context_length"], 256)
        self.assertGreater(bench["throughput_tok_s"], 0)


if __name__ == "__main__":
    unittest.main()
