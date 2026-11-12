import math
import os
from importlib.machinery import SourceFileLoader

YOCO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "yoco-cache-engine",
)
mod = SourceFileLoader("yoco_cache_engine", YOCO_PATH).load_module()

YocoConfig = mod.YocoConfig
YocoEngine = mod.YocoEngine
SharedGlobalKV = mod.SharedGlobalKV
SelfDecoderLayer = mod.SelfDecoderLayer
CrossDecoderLayer = mod.CrossDecoderLayer
theoretical_memory = mod.theoretical_memory


def test_softmax_sums_to_one():
    w = mod._softmax([1.0, 2.0, 3.0])
    assert abs(sum(w) - 1.0) < 1e-9
    assert all(x > 0 for x in w)


def test_gated_retention_update_and_read():
    d = 4
    state = [[0.0] * d for _ in range(d)]
    key = [1.0, 0.0, 0.0, 0.0]
    value = [2.0, 3.0, 0.0, 0.0]
    mod._gated_retention_update(state, key, value, decay=1.0)
    # S[0] should equal value
    assert state[0] == value
    out = mod._gated_retention_read(state, key)
    assert out == value
    # decay path
    mod._gated_retention_update(state, key, [0.0] * d, decay=0.5)
    assert abs(state[0][0] - 1.0) < 1e-9  # 0.5 * 2.0


def test_shared_kv_append_and_elements():
    kv = SharedGlobalKV(num_heads=2, head_dim=4)
    assert kv.length == 0
    k = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    v = [[0.5] * 4, [0.25] * 4]
    kv.append_token(k, v)
    assert kv.length == 1
    assert kv.element_count() == 2 * 1 * 4 * 2  # H·T·D·2
    kv.append_token(k, v)
    assert kv.length == 2
    assert kv.bytes(2) == kv.element_count() * 2


def test_self_layer_local_window_bounded():
    cfg = YocoConfig(self_layers=1, cross_layers=1, num_heads=2, head_dim=8, local_window=4, seed=1)
    layer = SelfDecoderLayer(cfg, 0, mod.random.Random(1))
    h = [0.1] * cfg.d_model
    for _ in range(20):
        h, _, _ = layer.forward_token(h, emit_global_kv=False)
    for head_k in layer.local_k:
        assert len(head_k) <= cfg.local_window
    assert layer.local_kv_elements() <= cfg.num_heads * cfg.local_window * cfg.head_dim * 2


def test_engine_cache_once_and_cross_zero_kv():
    cfg = YocoConfig(
        self_layers=3, cross_layers=3, num_heads=2, head_dim=16,
        local_window=8, seed=3,
    )
    engine = YocoEngine(cfg)
    stats = engine.prefill(32)
    assert stats["status"] == "success"
    mem = stats["memory"]
    assert mem["shared_kv_len"] == 32
    assert mem["yoco_cross_private_kv_elems"] == 0
    assert mem["kv_cache_once"] is True
    assert mem["memory_savings_pct"] > 0
    # Shared elems = one full layer KV
    expected = cfg.num_heads * 32 * cfg.head_dim * 2
    assert mem["yoco_shared_kv_elems"] == expected


def test_engine_decode_extends_shared_kv_only_once():
    cfg = YocoConfig(self_layers=2, cross_layers=2, num_heads=2, head_dim=8, local_window=4, seed=9)
    engine = YocoEngine(cfg)
    engine.prefill(16)
    engine.decode(24)
    mem = engine.memory_report()
    assert mem["global_tokens"] == 40
    assert mem["shared_kv_len"] == 40  # one entry per token, not per layer
    assert mem["yoco_cross_private_kv_elems"] == 0
    assert engine.last_hidden is not None
    assert all(math.isfinite(x) for x in engine.last_hidden)


def test_theoretical_savings_grow_with_depth():
    shallow = theoretical_memory(2, 2, 8, 64, 128, True, seq_lens=(65536,))
    deep = theoretical_memory(32, 32, 8, 64, 128, True, seq_lens=(65536,))
    s_ratio = shallow["comparisons"][0]["compression_ratio"]
    d_ratio = deep["comparisons"][0]["compression_ratio"]
    assert d_ratio > s_ratio
    # Deep 64-layer model should approach ~64× vs single shared KV (+bounded locals)
    assert d_ratio > 20


def test_benchmark_and_verify_smoke():
    cfg = YocoConfig(self_layers=2, cross_layers=2, num_heads=2, head_dim=8, local_window=4)
    engine = YocoEngine(cfg)
    bench = engine.benchmark(seq_len=24, decode_steps=8)
    assert bench["prefill"]["seq_len"] == 24
    assert bench["decode"]["steps"] == 8
    assert bench["final_memory"]["shared_kv_len"] == 32
    assert bench["theoretical_scaling"]["asymptotic_ratio_approx"] == 4.0
    mega = [c for c in bench["theoretical_scaling"]["comparisons"] if c["seq_len"] == 1_048_576][0]
    assert mega["compression_ratio"] > 1.5


def test_cross_layer_attends_shared_kv():
    cfg = YocoConfig(self_layers=1, cross_layers=1, num_heads=1, head_dim=4, local_window=8, use_retention=False, seed=0)
    engine = YocoEngine(cfg)
    # Manually seed shared KV with a distinctive value
    engine.shared_kv.append_token(
        [[1.0, 0.0, 0.0, 0.0]],
        [[5.0, 0.0, 0.0, 0.0]],
    )
    h = [1.0, 0.0, 0.0, 0.0]
    out = engine.cross_layers[0].forward_token(h, engine.shared_kv)
    assert all(math.isfinite(x) for x in out)
    assert _l2_ok(out)


def _l2_ok(v):
    return math.sqrt(sum(x * x for x in v)) > 0
