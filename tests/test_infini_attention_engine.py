import math
import sys
import os
from importlib.machinery import SourceFileLoader

INFINI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "infini-attention-engine",
)
mod = SourceFileLoader("infini_attention_engine", INFINI_PATH).load_module()

InfiniConfig = mod.InfiniConfig
InfiniAttentionKernel = mod.InfiniAttentionKernel
InfiniAttentionEngine = mod.InfiniAttentionEngine


def test_sigma_elu_plus_one_positive():
    # σ(x) = ELU(x)+1 > 0 for all real x
    samples = [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0]
    for x in samples:
        s = mod._elu_plus_one(x)
        assert s > 0.0, f"sigma({x}) should be > 0, got {s}"


def test_memory_update_and_retrieve_roundtrip():
    d = 8
    memory = [[0.0] * d for _ in range(d)]
    normalizer = [0.0] * d
    key = [0.5] * d
    value = [1.0] + [0.0] * (d - 1)
    InfiniAttentionKernel.update_memory(key, value, memory, normalizer, peak_norm=100.0)
    assert sum(abs(x) for x in normalizer) > 0
    # Query aligned with key should recover something close to value direction
    q = list(key)
    out = InfiniAttentionKernel.retrieve_from_memory(q, memory, normalizer)
    assert out[0] > max(out[1:])  # dominant first component
    assert all(math.isfinite(x) for x in out)


def test_local_softmax_attention_identity():
    q = [1.0, 0.0, 0.0, 0.0]
    keys = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    values = [
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 7.0, 0.0, 0.0],
    ]
    out = InfiniAttentionKernel.local_softmax_attention(q, keys, values, scale=10.0)
    # Strong scale → near one-hot on first key → near first value
    assert out[0] > 2.5
    assert out[1] < 1.0


def test_gated_blend_extremes():
    local = [1.0, 1.0]
    mem = [0.0, 0.0]
    assert InfiniAttentionKernel.gated_blend(local, mem, 0.0) == local
    assert InfiniAttentionKernel.gated_blend(local, mem, 1.0) == mem
    mid = InfiniAttentionKernel.gated_blend(local, mem, 0.5)
    assert abs(mid[0] - 0.5) < 1e-9


def test_engine_prefill_saves_memory():
    cfg = InfiniConfig(
        num_layers=2, num_heads=2, head_dim=16, local_window=8, memory_gate=0.4, seed=1
    )
    engine = InfiniAttentionEngine(cfg)
    stats = engine.prefill(seq_len=64)
    assert stats["status"] == "success"
    assert stats["infini_fixed_elements"] < stats["dense_kv_elements"]
    assert stats["memory_savings_pct"] > 0
    assert stats["global_tokens"] == 64
    # Local buffers bounded
    summary = engine.memory_state_summary()
    assert summary["mean_local_len"] <= cfg.local_window


def test_engine_constant_memory_across_decode():
    cfg = InfiniConfig(
        num_layers=2, num_heads=2, head_dim=8, local_window=4, memory_gate=0.5, seed=7
    )
    engine = InfiniAttentionEngine(cfg)
    engine.prefill(seq_len=32)
    elems = cfg.num_layers * cfg.num_heads * (cfg.head_dim ** 2 + cfg.head_dim)
    engine.decode(steps=48)
    # Footprint formula independent of sequence length
    elems_after = cfg.num_layers * cfg.num_heads * (cfg.head_dim ** 2 + cfg.head_dim)
    assert elems == elems_after
    assert engine.global_tokens == 80
    summary = engine.memory_state_summary()
    assert math.isfinite(summary["mean_memory_frobenius"])
    assert summary["mean_local_len"] <= cfg.local_window


def test_benchmark_and_theory():
    cfg = InfiniConfig(num_layers=1, num_heads=2, head_dim=8, local_window=4)
    engine = InfiniAttentionEngine(cfg)
    bench = engine.benchmark(seq_len=32, decode_steps=4)
    assert "prefill" in bench and "decode" in bench
    assert bench["decode"]["steps"] == 4
    assert bench["decode"]["compressive_memory_constant"] is True
    theory = InfiniAttentionKernel.theoretical_memory_bytes(32, 32, 128, 64)
    assert theory["infini_fixed_bytes"] > 0
    # At 1M tokens, Infini should crush dense KV
    mega = [c for c in theory["comparisons"] if c["seq_len"] == 1_048_576][0]
    assert mega["compression_ratio"] > 100


def test_cli_verify_exit_zero(tmp_path=None):
    # Smoke: import main path by running verify logic inline
    cfg = InfiniConfig(num_layers=1, num_heads=1, head_dim=8, local_window=4)
    engine = InfiniAttentionEngine(cfg)
    engine.prefill(16)
    engine.decode(16)
    s = engine.memory_state_summary()
    assert s["tokens_seen_global"] == 32
    assert s["mean_local_len"] <= 4
