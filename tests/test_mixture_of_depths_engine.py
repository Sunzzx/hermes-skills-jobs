"""Unit tests for mixture-of-depths-engine (DeepMind MoD, arXiv:2404.02258)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "mixture-of-depths-engine",
)
mod = SourceFileLoader("mixture_of_depths_engine", MOD_PATH).load_module()

MoDConfig = mod.MoDConfig
MoDEngine = mod.MoDEngine
DepthRouter = mod.DepthRouter
MoDBlock = mod.MoDBlock
theoretical_savings = mod.theoretical_savings
_capacity_k = mod._capacity_k
_topk_indices = mod._topk_indices
_softmax = mod._softmax


def test_softmax_sums_to_one():
    w = _softmax([0.2, 1.5, -0.3])
    assert abs(sum(w) - 1.0) < 1e-9
    assert all(x > 0 for x in w)


def test_capacity_k_bounds():
    assert _capacity_k(0, 0.125) == 0
    assert _capacity_k(1, 0.125) == 1
    assert _capacity_k(8, 0.125) == 1
    assert _capacity_k(16, 0.125) == 2
    assert _capacity_k(100, 1.0) == 100
    assert _capacity_k(10, 0.5) == 5


def test_topk_preserves_order_among_selected():
    scores = [0.1, 5.0, 0.2, 4.0, 0.0]
    idx = _topk_indices(scores, 2)
    assert idx == [1, 3]  # sorted by position among top-2


def test_theoretical_savings_positive_at_low_capacity():
    t = theoretical_savings(12, 2048, 1024, 16, capacity=0.125)
    assert t["savings_pct"] > 50.0
    assert t["selected_per_layer"] == _capacity_k(2048, 0.125)
    assert t["mod_flops"] < t["dense_flops"]


def test_theoretical_full_capacity_near_dense():
    t = theoretical_savings(4, 64, 128, 4, capacity=1.0)
    # router overhead remains, so mod >= dense slightly, savings ~0 or tiny negative
    assert t["selected_per_layer"] == 64
    assert t["compute_ratio"] >= 0.99


def test_router_selects_exact_capacity():
    cfg = MoDConfig(n_layers=1, d_model=32, n_heads=4, capacity=0.25, seed=7)
    engine = MoDEngine(cfg)
    hidden = engine.embed_synthetic(16)
    route = engine.layers[0].router.route(hidden)
    assert route["k"] == 4
    assert len(route["selected"]) == 4
    assert len(route["skipped"]) == 12
    assert set(route["selected"]).isdisjoint(route["skipped"])
    assert set(route["selected"]) | set(route["skipped"]) == set(range(16))


def test_skip_tokens_identity_path():
    """Tokens not selected must remain unchanged through a block."""
    cfg = MoDConfig(n_layers=1, d_model=32, n_heads=4, capacity=0.25, seed=11)
    block = MoDBlock(cfg, 0, mod.random.Random(11))
    hidden = [[float(i + j * 0.01) for j in range(32)] for i in range(12)]
    # freeze copy
    before = [list(h) for h in hidden]
    out = block.forward(hidden)
    skipped = block.last_route["skipped"]
    for idx in skipped:
        assert out[idx] == before[idx]


def test_selected_tokens_change():
    cfg = MoDConfig(n_layers=1, d_model=32, n_heads=4, capacity=0.5, seed=3)
    block = MoDBlock(cfg, 0, mod.random.Random(3))
    hidden = [[mod.random.Random(100 + i).gauss(0, 0.1) for _ in range(32)] for i in range(8)]
    before = [list(h) for h in hidden]
    out = block.forward(hidden)
    selected = block.last_route["selected"]
    assert len(selected) == 4
    # at least one selected token should differ (block is not identity)
    changed = sum(
        1 for idx in selected if any(abs(out[idx][j] - before[idx][j]) > 1e-12 for j in range(32))
    )
    assert changed >= 1


def test_engine_prefill_invariants():
    cfg = MoDConfig(n_layers=4, d_model=64, n_heads=4, capacity=0.125, seed=42)
    engine = MoDEngine(cfg)
    out = engine.prefill(32)
    assert out["status"] == "success"
    assert out["invariants"]["finite_hidden"] is True
    assert out["invariants"]["capacity_respected"] is True
    assert out["total_selected_slots"] == 4 * _capacity_k(32, 0.125)
    assert abs(out["realized_compute_ratio"] - 0.125) < 1e-9
    assert out["theory"]["savings_pct"] > 0
    assert engine.last_hidden is not None
    assert len(engine.last_hidden) == 32
    assert all(math.isfinite(v) for row in engine.last_hidden for v in row)


def test_verify_pass():
    cfg = MoDConfig(n_layers=3, d_model=32, n_heads=4, capacity=0.25, seed=1)
    engine = MoDEngine(cfg)
    report = engine.verify_invariants(seq_len=48)
    assert report["status"] == "pass"
    assert all(report["checks"].values())


def test_higher_capacity_more_compute():
    low = theoretical_savings(8, 256, 256, 8, capacity=0.125)
    high = theoretical_savings(8, 256, 256, 8, capacity=0.5)
    assert high["mod_flops"] > low["mod_flops"]
    assert high["savings_pct"] < low["savings_pct"]


def test_route_map_structure():
    cfg = MoDConfig(n_layers=2, d_model=32, n_heads=4, capacity=0.5, seed=9)
    engine = MoDEngine(cfg)
    rm = engine.route_map(8)
    assert rm["seq_len"] == 8
    assert len(rm["layers"]) == 2
    for layer in rm["layers"]:
        assert len(layer["selected"]) == 4
        assert len(layer["skipped"]) == 4
        assert len(layer["scores"]) == 8


def test_cli_verify_exit_zero():
    rc = mod.main(["--layers", "2", "--d-model", "32", "--heads", "4", "verify", "--seq-len", "24"])
    assert rc == 0
