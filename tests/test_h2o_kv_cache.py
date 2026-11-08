import numpy as np
import pytest
import sys
import os

sys.path.insert(0, '/Users/ayu/Developer/hermes-skills-jobs/bin')
from importlib.machinery import SourceFileLoader

h2o = SourceFileLoader('h2o_kv_cache', '/Users/ayu/Developer/hermes-skills-jobs/bin/h2o-kv-cache').load_module()


def test_h2o_compression():
    compressor = h2o.H2OKVCache(heavy_hitter_budget=64, recent_window_budget=32, sink_tokens=8)
    keys = np.random.randn(1, 8, 1024, 64).astype(np.float32)
    values = np.random.randn(1, 8, 1024, 64).astype(np.float32)

    ck, cv, stats = compressor.compress_cache(keys, values)
    assert ck.shape[2] <= compressor.total_budget
    assert cv.shape[2] <= compressor.total_budget
    assert stats["compression_ratio"] > 5.0
    assert stats["memory_saved_pct"] > 80.0


def test_h2o_short_seq():
    compressor = h2o.H2OKVCache(heavy_hitter_budget=64, recent_window_budget=32, sink_tokens=8)
    keys = np.random.randn(1, 4, 50, 64).astype(np.float32)
    values = np.random.randn(1, 4, 50, 64).astype(np.float32)

    ck, cv, stats = compressor.compress_cache(keys, values)
    assert ck.shape[2] == 50
    assert stats["compression_ratio"] == 1.0


def test_h2o_benchmark():
    res = h2o.run_benchmark(seq_len=512, hh_budget=32, recent_budget=16, sink_tokens=4)
    assert res["status"] == "success"
    assert "compression_ratio" in res["benchmark"]
