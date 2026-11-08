import math
import os
import sys
import unittest
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "streaming-llm-engine"))
streaming_llm = SourceFileLoader("streaming_llm_engine", bin_path).load_module()

AttentionSinkKVCache = streaming_llm.AttentionSinkKVCache
StreamingLLMConfig = streaming_llm.StreamingLLMConfig
simulate_streaming = streaming_llm.simulate_streaming


class TestStreamingLLMEngine(unittest.TestCase):
    def test_cache_capacity_bound(self):
        cfg = StreamingLLMConfig(num_sink_tokens=4, local_window_size=16, head_dim=8)
        cache = AttentionSinkKVCache(cfg)

        # Push 100 tokens
        for i in range(100):
            k = [float(i)] * 8
            v = [float(i)] * 8
            cache.update(k, v, pos=i)

        stats = cache.stats()
        self.assertEqual(stats["total_tokens_seen"], 100)
        self.assertEqual(stats["cached_tokens"], 20)  # 4 sinks + 16 window
        self.assertEqual(stats["max_capacity"], 20)
        self.assertEqual(stats["sink_tokens_retained"], 4)

        # Check retained positions: 0, 1, 2, 3 must be preserved as sinks
        positions = stats["retained_indices"]
        self.assertEqual(positions[:4], [0, 1, 2, 3])
        # The remaining 16 positions should be the most recent: 84..99
        self.assertEqual(positions[4:], list(range(100 - 16, 100)))

    def test_attention_computation(self):
        cfg = StreamingLLMConfig(num_sink_tokens=2, local_window_size=4, head_dim=4)
        cache = AttentionSinkKVCache(cfg)

        for i in range(10):
            k = [1.0 if d == (i % 4) else 0.0 for d in range(4)]
            v = [float(i)] * 4
            cache.update(k, v, pos=i)

        q = [1.0, 0.0, 0.0, 0.0]
        context, sink_mass = cache.compute_attention(q)
        self.assertGreater(sink_mass, 0.0)
        self.assertLessEqual(sink_mass, 1.0)
        self.assertEqual(len(context), 4)

    def test_simulate_streaming(self):
        results = simulate_streaming(stream_length=64, sink_tokens=4, window_size=16, head_dim=8)
        self.assertEqual(results["total_tokens_seen"], 64)
        self.assertEqual(results["cached_tokens"], 20)
        self.assertEqual(results["sink_tokens_retained"], 4)
        self.assertEqual(results["compression_ratio"], 3.2)
        self.assertTrue(len(results["sink_attention_trace"]) > 0)


if __name__ == "__main__":
    unittest.main()
