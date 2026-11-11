import pytest
import sys
import os
from importlib.machinery import SourceFileLoader

FASTGEN_PATH = "/Users/ayu/Developer/hermes-skills-jobs/bin/fastgen-engine"
fastgen_mod = SourceFileLoader("fastgen_engine", FASTGEN_PATH).load_module()

FastGenConfig = fastgen_mod.FastGenConfig
FastGenKernel = fastgen_mod.FastGenKernel
FastGenEngine = fastgen_mod.FastGenEngine
HeadArchetype = fastgen_mod.HeadArchetype


def test_fastgen_classification_sink():
    # Matrix with concentrated sink mass
    matrix = [[0.8, 0.1, 0.05, 0.05]]
    archetype, metrics = FastGenKernel.classify_head(matrix, sink_tokens=1)
    assert archetype == HeadArchetype.SPECIAL_SINK
    assert metrics["sink_mass"] >= 0.35


def test_fastgen_classification_local():
    # Matrix with diagonal locality
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    archetype, metrics = FastGenKernel.classify_head(matrix, sink_tokens=1, locality_window=2)
    assert archetype == HeadArchetype.LOCAL_STREAMING
    assert metrics["local_mass"] >= 0.55


def test_fastgen_engine_prefill_and_decode():
    cfg = FastGenConfig(num_layers=2, num_heads=4, head_dim=32, retrieval_ratio=0.25)
    engine = FastGenEngine(cfg)
    stats = engine.prefill(seq_len=128)

    assert stats["status"] == "success"
    assert stats["total_fastgen_tokens"] < stats["total_uncompressed_tokens"]
    assert stats["memory_savings_pct"] > 0
    assert len(engine.kv_stores) == 8

    decode_info = engine.decode_step()
    assert decode_info["current_seq_len"] == 129


def test_fastgen_benchmark():
    cfg = FastGenConfig(num_layers=2, num_heads=2, head_dim=32)
    engine = FastGenEngine(cfg)
    bench = engine.benchmark(seq_len=64, decode_steps=4)

    assert "prefill" in bench
    assert "decode" in bench
    assert bench["decode"]["steps"] == 4
    assert bench["decode"]["final_seq_len"] == 68
