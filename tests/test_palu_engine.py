import pytest
import sys
import os
from importlib.machinery import SourceFileLoader

PALU_PATH = "/Users/ayu/Developer/hermes-skills-jobs/bin/palu-engine"
palu_mod = SourceFileLoader("palu_engine", PALU_PATH).load_module()


def test_palu_projector():
    projector = palu_mod.PaluLowRankProjector(in_features=64, rank=32, seed=42)
    vec = [float(i) for i in range(64)]
    latent = projector.compress(vec)
    assert len(latent) == 32
    recon = projector.reconstruct(latent)
    assert len(recon) == 64


def test_palu_engine_prefill_and_compression():
    cfg = palu_mod.PaluConfig(num_layers=2, num_heads=4, head_dim=32, rank_ratio=0.5)
    assert cfg.target_rank == 16
    engine = palu_mod.PaluEngine(cfg)
    res = engine.prefill(seq_len=64)
    
    assert res["status"] == "success"
    assert res["compression_ratio"] == 2.0
    assert res["memory_savings_pct"] == 50.0
    assert res["palu_cache_bytes"] * 2 == res["full_cache_bytes"]
    assert engine.cached_seq_len == 64


def test_palu_decode_step():
    cfg = palu_mod.PaluConfig(num_layers=2, num_heads=4, head_dim=32, rank_ratio=0.25)
    assert cfg.target_rank == 8
    engine = palu_mod.PaluEngine(cfg)
    engine.prefill(seq_len=16)
    
    step_res = engine.decode_step()
    assert step_res["current_seq_len"] == 17
    assert step_res["cached_rank"] == 8
    assert engine.cached_seq_len == 17


def test_palu_benchmark():
    cfg = palu_mod.PaluConfig(num_layers=2, num_heads=2, head_dim=16, rank_ratio=0.5)
    engine = palu_mod.PaluEngine(cfg)
    bench = engine.benchmark(seq_len=32, decode_steps=4)
    
    assert bench["prefill"]["status"] == "success"
    assert bench["decode"]["steps"] == 4
    assert bench["decode"]["final_seq_len"] == 36
