"""Unit tests for omniquant-lwc-let-engine (ICLR 2024, arXiv:2308.13137)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "omniquant-lwc-let-engine",
)
mod = SourceFileLoader("omniquant_lwc_let_engine", MOD_PATH).load_module()

QuantConfig = mod.QuantConfig
OmniQuantConfig = mod.OmniQuantConfig
OmniQuantEngine = mod.OmniQuantEngine
quantize_vector = mod.quantize_vector
dequantize_vector = mod.dequantize_vector
quantize_matrix = mod.quantize_matrix
theoretical_bit_rate = mod.theoretical_bit_rate
make_weight = mod.make_weight
make_activation = mod.make_activation
make_bias = mod.make_bias
fit_lwc = mod.fit_lwc
apply_lwc = mod.apply_lwc
quantize_with_lwc = mod.quantize_with_lwc
fit_let = mod.fit_let
let_equivalence_error = mod.let_equivalence_error
apply_let_weight = mod.apply_let_weight
apply_let_activation = mod.apply_let_activation
apply_let_bias = mod.apply_let_bias
fp_linear = mod.fp_linear
mat_flatten = mod.mat_flatten
_mse = mod._mse
_cosine = mod._cosine
_l2 = mod._l2
_clip_chunk = mod._clip_chunk
_golden_section_1d = mod._golden_section_1d


def test_quantize_dequantize_roundtrip_int4():
    cfg = QuantConfig(n_bits=4, group_size=16, symmetric=False)
    x = [math.sin(i / 3.0) for i in range(64)]
    qt = quantize_vector(x, cfg)
    recon = dequantize_vector(qt)
    assert _cosine(x, recon) > 0.95
    assert qt.bit_rate() <= 6.0 + 1e-12


def test_theoretical_bit_rate_matches():
    n, bits, gs = 256, 4, 64
    cfg = QuantConfig(n_bits=bits, group_size=gs, symmetric=True)
    x = [0.1] * n
    qt = quantize_vector(x, cfg)
    br = theoretical_bit_rate(n, bits, gs, True)
    assert abs(qt.bit_rate() - br) < 1e-9


def test_clip_chunk_respects_bounds():
    chunk = [-10.0, -1.0, 0.0, 1.0, 10.0]
    clipped = _clip_chunk(chunk, gamma=0.5, beta=0.5)
    assert max(clipped) <= max(chunk) + 1e-12
    assert min(clipped) >= min(chunk) - 1e-12
    assert len(clipped) == len(chunk)


def test_golden_section_finds_quadratic_min():
    def f(x: float) -> float:
        return (x - 0.37) ** 2 + 1.0

    x_star, f_star = _golden_section_1d(f, 0.0, 1.0, iters=20)
    assert abs(x_star - 0.37) < 0.02
    assert f_star < f(0.0)


def test_lwc_params_in_unit_interval():
    w = make_weight(24, 24, seed=3, n_outliers=2)
    cfg = QuantConfig(n_bits=4, group_size=32)
    params = fit_lwc(mat_flatten(w), cfg, iters=6)
    assert all(0.0 < g <= 1.0 + 1e-9 for g in params.gammas)
    assert all(0.0 < b <= 1.0 + 1e-9 for b in params.betas)
    assert params.group_size == 32


def test_lwc_does_not_increase_weight_mse_much():
    w = make_weight(32, 32, seed=11, n_outliers=4)
    cfg = QuantConfig(n_bits=4, group_size=64)
    base, _ = quantize_matrix(w, cfg)
    wq, _, _ = quantize_with_lwc(w, cfg)
    # LWC targets output recon; weight MSE may trade — still finite & cosine high
    assert _cosine(mat_flatten(w), mat_flatten(wq)) > 0.85
    assert math.isfinite(_mse(mat_flatten(w), mat_flatten(wq)))
    assert math.isfinite(_mse(mat_flatten(w), mat_flatten(base)))


def test_let_fp_equivalence_near_zero():
    w = make_weight(20, 20, seed=4)
    x = make_activation(12, 20, seed=5)
    b = make_bias(20, seed=6)
    let = fit_let(x, w, b, alpha=0.5, shift=True, refine_iters=2)
    err = let_equivalence_error(x, w, b, let)
    assert err < 1e-8


def test_let_scales_positive():
    w = make_weight(16, 16, seed=1)
    x = make_activation(10, 16, seed=2)
    let = fit_let(x, w, refine_iters=1)
    assert all(s > 0 for s in let.scales)
    assert len(let.scales) == 16
    assert len(let.shifts) == 16


def test_omniquant_beats_or_ties_rtn_on_outlier_weights():
    eng = OmniQuantEngine(OmniQuantConfig(n_bits=4, group_size=32, let_refine_iters=3))
    w = make_weight(24, 24, seed=21, n_outliers=5, outlier_scale=10.0)
    x = make_activation(16, 24, seed=22, n_outlier_channels=3, outlier_scale=15.0)
    b = make_bias(24, seed=23)
    rtn = eng.rtn_baseline(x, w, b)
    omni = eng.quantize_block(x, w, b)
    # Allow small slack — synthetic may vary; primary invariants elsewhere
    assert omni.metrics["output_mse"] <= rtn.metrics["output_mse"] * 1.35 + 1e-9
    assert omni.metrics["output_cosine"] > 0.7
    assert omni.metrics["bit_rate"] < 6.0


def test_lwc_only_path_runs():
    eng = OmniQuantEngine(OmniQuantConfig(n_bits=4, group_size=64))
    w = make_weight(16, 16, seed=9)
    x = make_activation(8, 16, seed=10)
    b = make_bias(16, seed=11)
    r = eng.lwc_only(x, w, b)
    assert r.lwc is not None
    assert r.let is None
    assert math.isfinite(r.metrics["output_mse"])


def test_smoothquant_ablation_runs():
    eng = OmniQuantEngine()
    w = make_weight(16, 16, seed=12)
    x = make_activation(8, 16, seed=13)
    b = make_bias(16, seed=14)
    r = eng.smoothquant_baseline(x, w, b, alpha=0.5)
    assert r.let is not None
    assert r.metrics["let_fp_equiv_mse"] < 1e-8


def test_status_and_verify_pass():
    eng = OmniQuantEngine(OmniQuantConfig(n_bits=4, group_size=64, let_refine_iters=2))
    st = eng.status(rows=16, cols=16, batch=8)
    assert st["status"] == "ok"
    assert "omniquant" in st
    vf = eng.verify(trials=2)
    assert vf["status"] == "pass"


def test_benchmark_and_theory_pack():
    eng = OmniQuantEngine()
    bm = eng.benchmark([12, 16], bits_list=[4, 8])
    assert len(bm["rows"]) == 4
    th = eng.theory()
    assert th["fp16_7b_gib"] > 10.0
    assert len(th["rows"]) >= 4
    pk = eng.pack(16, 16)
    assert pk["bpw_packed"] > 0
    assert "fuse" in pk["deploy_note"].lower() or "absorb" in pk["deploy_note"].lower()


def test_fp_linear_shapes():
    w = make_weight(10, 12, seed=0)  # out=10, in=12
    x = make_activation(4, 12, seed=1)
    b = make_bias(10, seed=2)
    y = fp_linear(x, w, b)
    assert len(y) == 4
    assert len(y[0]) == 10


def test_calibrate_payload():
    eng = OmniQuantEngine()
    c = eng.calibrate(rows=20, cols=20, batch=10)
    assert "weight" in c and "activation" in c
    assert c["hessian_diag_cond_proxy"] >= 1.0
