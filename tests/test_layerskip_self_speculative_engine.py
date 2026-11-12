"""Unit tests for layerskip-self-speculative-engine (Meta FAIR, arXiv:2404.15690)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "layerskip-self-speculative-engine",
)
mod = SourceFileLoader("layerskip_self_speculative_engine", MOD_PATH).load_module()

LayerSkipConfig = mod.LayerSkipConfig
LayerSkipEngine = mod.LayerSkipEngine
theoretical_early_exit = mod.theoretical_early_exit
theoretical_self_speculative = mod.theoretical_self_speculative
_softmax = mod._softmax
_entropy = mod._entropy


def test_softmax_sums_to_one():
    w = _softmax([0.2, 1.5, -0.3, 4.0])
    assert abs(sum(w) - 1.0) < 1e-9
    assert all(x > 0 for x in w)


def test_entropy_uniform_max():
    u = [0.25, 0.25, 0.25, 0.25]
    p = [0.97, 0.01, 0.01, 0.01]
    assert _entropy(u) > _entropy(p)


def test_theoretical_early_exit_saves_when_exiting_early():
    # exit everything at layer 2 of 8
    rates = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # fix: conditional rates — exit at 2 with p=1 after min
    rates = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    t = theoretical_early_exit(8, 256, 8, 4.0, 1000, rates, seq_len=64)
    assert t["layerskip_expected_flops"] < t["dense_flops"]
    assert t["savings_pct"] > 30.0


def test_theoretical_never_exit_near_dense():
    rates = [0.0] * 7 + [1.0]
    t = theoretical_early_exit(8, 128, 4, 4.0, 256, rates, seq_len=32)
    # still pays exit heads each layer → slightly above pure dense block cost ratio ~1
    assert t["compute_ratio"] >= 0.9


def test_self_spec_speedup_scales_with_alpha():
    low = theoretical_self_speculative(12, 4, 256, 8, 4.0, 32000, 4, 0.3, 128)
    high = theoretical_self_speculative(12, 4, 256, 8, 4.0, 32000, 4, 0.9, 128)
    assert high["speedup"] > low["speedup"]
    assert high["gamma_expected_accepted_drafts"] > low["gamma_expected_accepted_drafts"]
    assert high["speedup"] > 1.0


def test_config_clamps_draft_layers():
    cfg = LayerSkipConfig(n_layers=6, draft_layers=100, min_exit_layer=50)
    assert cfg.draft_layers == 5
    assert cfg.min_exit_layer == 5


def test_engine_prefill_invariants():
    cfg = LayerSkipConfig(
        n_layers=6, d_model=64, n_heads=4, vocab_size=128,
        exit_threshold=0.85, min_exit_layer=1, seed=42,
    )
    engine = LayerSkipEngine(cfg)
    out = engine.prefill_early_exit(24)
    assert out["status"] == "success"
    assert out["invariants"]["all_exited"] is True
    assert out["invariants"]["counts_sum"] is True
    assert out["invariants"]["finite_hidden"] is True
    assert abs(sum(out["exit_mass_rates"]) - 1.0) < 1e-9
    assert len(out["exit_layers"]) == 24
    assert all(0 <= e < 6 for e in out["exit_layers"])
    assert out["realized_flops"] <= out["dense_flops"] * 1.01
    assert engine.last_hidden is not None
    assert len(engine.last_hidden) == 24


def test_higher_threshold_exits_deeper_or_equal():
    low_cfg = LayerSkipConfig(
        n_layers=6, d_model=32, n_heads=4, exit_threshold=0.5,
        min_exit_layer=1, seed=7, vocab_size=64,
    )
    high_cfg = LayerSkipConfig(
        n_layers=6, d_model=32, n_heads=4, exit_threshold=0.95,
        min_exit_layer=1, seed=7, vocab_size=64,
    )
    low = LayerSkipEngine(low_cfg).prefill_early_exit(32)
    high = LayerSkipEngine(high_cfg).prefill_early_exit(32)
    assert high["avg_exit_layer"] + 0.75 >= low["avg_exit_layer"]


def test_min_exit_layer_honored():
    cfg = LayerSkipConfig(
        n_layers=8, d_model=32, n_heads=4, min_exit_layer=3,
        exit_threshold=0.01, seed=1, vocab_size=64,
    )
    out = LayerSkipEngine(cfg).prefill_early_exit(20)
    for el in out["exit_layers"]:
        assert el >= 3


def test_self_speculate_emits_exact_count():
    cfg = LayerSkipConfig(
        n_layers=6, d_model=32, n_heads=4, draft_layers=2,
        speculate_k=3, seed=11, vocab_size=64,
    )
    engine = LayerSkipEngine(cfg)
    out = engine.self_speculate(prompt_len=8, gen_tokens=6, acceptance_override=0.75)
    assert out["gen_tokens"] == 6
    assert len(out["generated_ids"]) == 6
    assert out["draft_tokens_total"] >= 1
    assert 0.0 <= out["empirical_acceptance_rate"] <= 1.0
    assert math.isfinite(out["realized_flops"])


def test_verify_pass():
    cfg = LayerSkipConfig(
        n_layers=6, d_model=32, n_heads=4, exit_threshold=0.9,
        min_exit_layer=1, draft_layers=2, speculate_k=3, seed=3, vocab_size=64,
    )
    report = LayerSkipEngine(cfg).verify_invariants(seq_len=24)
    assert report["status"] == "pass", report.get("errors")
    assert all(report["checks"].values())


def test_exit_map_structure():
    cfg = LayerSkipConfig(n_layers=4, d_model=32, n_heads=4, seed=9, vocab_size=32)
    rm = LayerSkipEngine(cfg).exit_map(12)
    assert rm["seq_len"] == 12
    assert len(rm["exit_layers"]) == 12
    assert sum(rm["layer_exit_counts"]) == 12


def test_memory_report_positive():
    cfg = LayerSkipConfig(n_layers=4, d_model=64, n_heads=4, vocab_size=100)
    mem = LayerSkipEngine(cfg).memory_report()
    assert mem["param_count_est"] > 0
    assert mem["param_bytes_fp32"] == mem["param_count_est"] * 4


def test_cli_verify_exit_zero():
    rc = mod.main([
        "--layers", "4", "--d-model", "32", "--heads", "4",
        "--vocab", "64", "--min-exit-layer", "1", "--draft-layers", "2",
        "verify", "--seq-len", "16",
    ])
    assert rc == 0


def test_cli_status_and_speculate():
    assert mod.main(["--layers", "4", "--d-model", "32", "--heads", "4", "status"]) == 0
    rc = mod.main([
        "--layers", "4", "--d-model", "32", "--heads", "4", "--vocab", "32",
        "--draft-layers", "2", "speculate", "--prompt-len", "4", "--gen-tokens", "3",
        "--force-alpha", "0.8",
    ])
    assert rc == 0
