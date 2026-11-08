import numpy as np
import pytest
import sys
import os

sys.path.insert(0, '/Users/ayu/Developer/hermes-skills-jobs/bin')
from importlib.machinery import SourceFileLoader

snapkv = SourceFileLoader("snapkv_engine", "/Users/ayu/Developer/hermes-skills-jobs/bin/snapkv-engine").load_module()


def test_snapkv_compression():
    compressor = snapkv.SnapKVCompressor(capacity=64, obs_window=16, sink_tokens=8)
    keys = np.random.randn(1, 4, 256, 32).astype(np.float32)
    values = np.random.randn(1, 4, 256, 32).astype(np.float32)

    ck, cv, stats = compressor.compress_cache(keys, values)
    assert ck.shape[0] == 1
    assert ck.shape[1] == 4
    assert ck.shape[2] <= 128
    assert ck.shape[3] == 32
    assert stats["compression_ratio"] > 1.8
    assert stats["memory_saved_pct"] > 40.0


def test_snapkv_short_bypass():
    compressor = snapkv.SnapKVCompressor(capacity=128, obs_window=32)
    keys = np.random.randn(1, 2, 64, 16).astype(np.float32)
    values = np.random.randn(1, 2, 64, 16).astype(np.float32)

    ck, cv, stats = compressor.compress_cache(keys, values)
    assert ck.shape[2] == 64
    assert stats["compression_ratio"] == 1.0
    assert stats["memory_saved_pct"] == 0.0


def test_snapkv_benchmark():
    res = snapkv.run_benchmark(seq_len=512, capacity=64, obs_window=16)
    assert res["status"] == "success"
    assert "compression_ratio" in res["benchmark"]
