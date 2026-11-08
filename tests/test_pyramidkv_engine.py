import numpy as np
import pytest
import sys
import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "pyramidkv-engine"))
pyramidkv = SourceFileLoader("pyramidkv_engine", bin_path).load_module()

PyramidKVEngine = pyramidkv.PyramidKVEngine
PyramidKVConfig = pyramidkv.PyramidKVConfig
run_benchmark = pyramidkv.run_benchmark


def test_pyramidal_budget_distribution():
    config = PyramidKVConfig(num_layers=10, total_budget=600, pyramid_slope=2.0)
    engine = PyramidKVEngine(config)
    budgets = engine.layer_budgets
    assert len(budgets) == 10
    assert sum(budgets) == 600
    # Pyramidal invariant: bottom layer has more budget than top layer
    assert budgets[0] > budgets[-1]
    # Check monotonic descent or stable distribution
    for i in range(len(budgets) - 1):
        assert budgets[i] >= budgets[i + 1]


def test_pyramidkv_layer_compression():
    config = PyramidKVConfig(num_layers=4, total_budget=200, pyramid_slope=1.5, num_sink_tokens=4, recent_window_size=16)
    engine = PyramidKVEngine(config)
    
    # Sequence length 128 > layer budget
    keys = np.random.randn(1, 4, 128, 32).astype(np.float32)
    values = np.random.randn(1, 4, 128, 32).astype(np.float32)
    queries = np.random.randn(1, 4, 128, 32).astype(np.float32)

    for l in range(4):
        ck, cv, stats = engine.compress_layer(l, keys, values, queries=queries)
        assert ck.shape[2] == engine.layer_budgets[l]
        assert cv.shape[2] == engine.layer_budgets[l]
        assert stats["retained_tokens"] == engine.layer_budgets[l]
        assert stats["compression_ratio"] > 1.0
        # Sinks (0..3) must be retained
        assert stats["indices"][:4] == [0, 1, 2, 3]
        # Tail recent window must be retained
        tail_expected = list(range(128 - 16, 128))
        for t in tail_expected:
            assert t in stats["indices"]


def test_pyramidkv_short_sequence_bypass():
    config = PyramidKVConfig(num_layers=4, total_budget=400, pyramid_slope=1.5)
    engine = PyramidKVEngine(config)
    
    # Sequence length 30 <= any layer budget
    keys = np.random.randn(1, 4, 30, 32).astype(np.float32)
    values = np.random.randn(1, 4, 30, 32).astype(np.float32)
    
    ck, cv, stats = engine.compress_layer(0, keys, values)
    assert ck.shape[2] == 30
    assert stats["compression_ratio"] == 1.0
    assert stats["retained_tokens"] == 30


def test_compress_all_layers():
    config = PyramidKVConfig(num_layers=6, total_budget=360, pyramid_slope=2.0)
    engine = PyramidKVEngine(config)

    all_k = [np.random.randn(1, 2, 100, 16).astype(np.float32) for _ in range(6)]
    all_v = [np.random.randn(1, 2, 100, 16).astype(np.float32) for _ in range(6)]
    
    ck_list, cv_list, overall_stats = engine.compress_all_layers(all_k, all_v)
    assert len(ck_list) == 6
    assert overall_stats["total_retained_tokens"] == 360
    assert overall_stats["overall_compression_ratio"] == round(600 / 360, 2)
    assert overall_stats["memory_saved_pct"] > 35.0


def test_benchmark_execution():
    res = run_benchmark(seq_len=256, num_layers=4, total_budget=256, pyramid_slope=2.0)
    assert res["status"] == "success"
    assert "benchmark" in res
    assert res["benchmark"]["pyramidal_skew_ratio"] > 1.0
