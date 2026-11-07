"""Unit and integration tests for CINIC Speculative Kernel Cache."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "cinic-kernel-cache"

loader = SourceFileLoader("cinic_kernel_cache", str(BIN_PATH))
spec = importlib.util.spec_from_loader("cinic_kernel_cache", loader)
assert spec is not None and spec.loader is not None
cinic_mod = importlib.util.module_from_spec(spec)
sys.modules["cinic_kernel_cache"] = cinic_mod
loader.exec_module(cinic_mod)

CacheConfig = cinic_mod.CacheConfig
CINICEngine = cinic_mod.CINICEngine
KernelPageCompressor = cinic_mod.KernelPageCompressor
SpeculativeKernelCache = cinic_mod.SpeculativeKernelCache


class TestCINICEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cfg = CacheConfig(
            max_memory_mb=16.0,
            compression_mode="quant_zlib",
            num_bits=8,
        )
        self.engine = CINICEngine(self.cfg)

    def test_compression_and_decompression_fidelity(self):
        arr = np.random.randn(8, 256, 64).astype(np.float32)
        comp_bytes, scale, ratio = KernelPageCompressor.compress_tensor(arr, num_bits=8, mode="quant_zlib")
        self.assertGreater(ratio, 1.5)
        self.assertLess(len(comp_bytes), arr.nbytes)

        recon = KernelPageCompressor.decompress_tensor(comp_bytes, arr.shape, scale, num_bits=8, mode="quant_zlib")
        self.assertEqual(recon.shape, arr.shape)

        cos_sim = float(
            np.dot(arr.flatten(), recon.flatten())
            / (np.linalg.norm(arr.flatten()) * np.linalg.norm(recon.flatten()))
        )
        self.assertGreater(cos_sim, 0.999)

    def test_cache_put_get_lifecycle(self):
        arr = np.random.randn(4, 64, 32).astype(np.float32)
        meta = self.engine.cache.put_tensor("page_test", arr)
        self.assertEqual(meta.page_id, "page_test")
        self.assertGreater(meta.compression_ratio, 1.0)

        retrieved = self.engine.cache.get_tensor("page_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, arr.shape)

    def test_lru_eviction(self):
        # Configure tiny cache that can only hold ~2 pages
        tiny_cfg = CacheConfig(max_memory_mb=0.05, compression_mode="none")
        tiny_cache = SpeculativeKernelCache(tiny_cfg)
        arr = np.zeros((10, 1024), dtype=np.float32)  # ~40KB per page

        tiny_cache.put_tensor("page_1", arr)
        tiny_cache.put_tensor("page_2", arr)
        tiny_cache.put_tensor("page_3", arr)

        self.assertGreaterEqual(tiny_cache.evictions, 1)
        self.assertIsNone(tiny_cache.get_tensor("page_1"))
        self.assertIsNotNone(tiny_cache.get_tensor("page_3"))

    def test_kv_chunk_orchestration(self):
        k_arr = np.random.randn(4, 128, 64).astype(np.float32)
        v_arr = np.random.randn(4, 128, 64).astype(np.float32)
        meta_k, meta_v = self.engine.cache_kv_chunk(0, 0, k_arr, v_arr)
        self.assertTrue(meta_k.page_id.endswith("_k"))
        self.assertTrue(meta_v.page_id.endswith("_v"))

        rec_k, rec_v = self.engine.retrieve_kv_chunk(0, 0)
        self.assertIsNotNone(rec_k)
        self.assertIsNotNone(rec_v)

    def test_benchmark_simulation(self):
        stats = self.engine.run_benchmark(num_layers=4, num_tokens=256, head_dim=32)
        self.assertEqual(stats["hit_rate"], 1.0)
        self.assertGreater(stats["overall_compression_ratio"], 1.2)


if __name__ == "__main__":
    unittest.main()
