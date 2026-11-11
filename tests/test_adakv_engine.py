import pytest
import sys
import os
from importlib.machinery import SourceFileLoader

ADAKV_PATH = "/Users/ayu/Developer/hermes-skills-jobs/bin/adakv-engine"
adakv_mod = SourceFileLoader("adakv_engine", ADAKV_PATH).load_module()


def test_adakv_pooling():
    scores = [0.1, 0.2, 0.9, 0.3, 0.1]
    pooled = adakv_mod.AdaKVKernel.max_pooling_1d(scores, kernel_size=3)
    assert len(pooled) == len(scores)
    assert max(pooled) == 0.9


def test_adakv_head_importance_and_budget_allocation():
    # Retrieval head with sparse high-peak weights
    retrieval_weights = [[0.8, 0.1, 0.05, 0.05], [0.7, 0.2, 0.05, 0.05]]
    # Streaming diffuse head with uniform weights
    diffuse_weights = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    
    score_retrieval = adakv_mod.AdaKVKernel.compute_head_importance_score(retrieval_weights)
    score_diffuse = adakv_mod.AdaKVKernel.compute_head_importance_score(diffuse_weights)
    assert score_retrieval > score_diffuse

    importance_map = {
        (0, 0): score_retrieval,
        (0, 1): score_diffuse
    }
    budgets = adakv_mod.AdaKVKernel.allocate_budgets(importance_map, total_token_budget=100, min_budget_per_head=10)
    assert budgets[(0, 0)] + budgets[(0, 1)] == 100
    assert budgets[(0, 0)] > budgets[(0, 1)]
    assert budgets[(0, 1)] >= 10


def test_adakv_engine_prefill_and_compression():
    cfg = adakv_mod.AdaKVConfig(num_layers=2, num_heads=4, head_dim=32, total_budget_ratio=0.5)
    engine = adakv_mod.AdaKVEngine(cfg)
    res = engine.prefill(seq_len=64)
    
    assert res["status"] == "success"
    assert res["memory_savings_pct"] > 40.0
    assert res["compression_ratio"] > 1.5
    assert engine.cached_seq_len == 64
    assert len(engine.allocated_budgets) == 8


def test_adakv_decode_step():
    cfg = adakv_mod.AdaKVConfig(num_layers=2, num_heads=2, head_dim=16, total_budget_ratio=0.25)
    engine = adakv_mod.AdaKVEngine(cfg)
    engine.prefill(seq_len=32)
    initial_tokens = engine.total_tokens_stored
    
    step_res = engine.decode_step()
    assert step_res["current_seq_len"] == 33
    assert engine.cached_seq_len == 33


def test_adakv_benchmark():
    cfg = adakv_mod.AdaKVConfig(num_layers=2, num_heads=2, head_dim=16, total_budget_ratio=0.5)
    engine = adakv_mod.AdaKVEngine(cfg)
    bench = engine.benchmark(seq_len=32, decode_steps=4)
    
    assert bench["prefill"]["status"] == "success"
    assert bench["decode"]["steps"] == 4
    assert bench["decode"]["final_seq_len"] == 36
