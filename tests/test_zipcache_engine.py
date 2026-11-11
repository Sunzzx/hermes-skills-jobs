#!/usr/bin/env python3
"""Unit tests for ZipCache KV Cache Compression Engine."""
import unittest
import sys
import os
from pathlib import Path
from importlib.machinery import SourceFileLoader
import numpy as np

ENGINE_PATH = Path(__file__).resolve().parent.parent / "bin" / "zipcache-engine"
zipcache_mod = SourceFileLoader("zipcache_engine", str(ENGINE_PATH)).load_module()

class TestZipCacheEngine(unittest.TestCase):
    def setUp(self):
        self.config = zipcache_mod.ZipCacheConfig(num_heads=4, head_dim=32, window_size=32, sink_tokens=2)
        self.engine = zipcache_mod.ZipCacheEngine(self.config)

    def test_bitpacker_2bit(self):
        data = np.array([0, 1, 2, 3, 1, 0, 3, 2], dtype=np.uint8)
        packed = zipcache_mod.BitPacker.pack_2bit(data)
        self.assertEqual(len(packed), 2)
        unpacked = zipcache_mod.BitPacker.unpack_2bit(packed, len(data))
        np.testing.assert_array_equal(data, unpacked)

    def test_bitpacker_4bit(self):
        data = np.array([0, 15, 7, 3, 8, 12], dtype=np.uint8)
        packed = zipcache_mod.BitPacker.pack_4bit(data)
        self.assertEqual(len(packed), 3)
        unpacked = zipcache_mod.BitPacker.unpack_4bit(packed, len(data))
        np.testing.assert_array_equal(data, unpacked)

    def test_sensitivity_analyzer(self):
        tensor = np.zeros((4, 64, 32), dtype=np.float32)
        tensor[:, :, 5] = np.random.randn(4, 64) * 10.0
        sens = zipcache_mod.ChannelSensitivityAnalyzer.identify_sensitive_channels(tensor, ratio=0.25)
        self.assertIn(5, sens)

    def test_prefill_and_decode(self):
        stats = self.engine.prefill(seq_len=128)
        self.assertEqual(stats["total_tokens"], 128)
        self.assertGreater(stats["memory_savings_pct"], 40.0)
        out, step_stats = self.engine.decode_step(0)
        self.assertEqual(len(out), 32)
        self.assertEqual(step_stats["current_tokens"], 129)

if __name__ == "__main__":
    unittest.main()
