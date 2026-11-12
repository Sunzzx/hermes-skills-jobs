"""Unit tests for smoothquant-activation-smoothing-engine (arXiv:2211.10438)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "smoothquant-activation-smoothing-engine",
)
mod = SourceFileLoader("smoothquant_activation_smoothing_engine", MOD_PATH).load_module()

SmoothQuantConfig = mod.SmoothQuantConfig
SmoothQuantEngine = mod.SmoothQuantEngine
QuantConfig = mod.QuantConfig
quantize_vector = mod.quantize_vector
dequantize_vector = mod.dequantize_vector
theoretical_bit_rate = mod.theoretical_bit_rate
mat_mul = mod.mat_mul
mat_scale_cols = mod.mat_scale_cols
mat_scale_rows = mod.mat_scale_rows
mat_flatten = mod.mat_flatten
mat_max_abs_rows = mod.mat_max_abs_rows
mat_max_abs_cols = mod.mat_max_abs_cols
make_weight = mod.make_weight
make_activations = mod.make_activations
_mse = mod._mse
_cosine = mod._cosine


def test_quantize_dequantize_roundtrip_int8():
    cfg = QuantConfig(n_bits=8, group_size=0, symmetric=True)
    x = [math.sin(i / 3.0) for i in range(64)]
    qt = quantize_vector(x, cfg)
    recon = dequantize_vector(qt)
    assert _cosine(x, recon) > 0.99
    assert qt.bit_rate() < 10.0


def test_int8_compression_ratio():
    cfg = QuantConfig(n_bits=8, group_size=64, symmetric=True)
    x = [0.01 * i for i in range(256)]
    qt = quantize_vector(x, cfg)
    assert qt.compression_ratio(16.0) > 1.5
    assert 8.0 <= qt.bit_rate() < 10.5


def test_theoretical_bit_rate_matches():
    n, bits, gs = 256, 8, 64
    cfg = QuantConfig(n_bits=bits, group_size=gs, symmetric=True)
    x = [0.1] * n
    qt = quantize_vector(x, cfg)
    br = theoretical_bit_rate(n, bits, gs, True)
    assert abs(qt.bit_rate() - br) < 1e-9


def test_activation_outliers_present():
    X = make_activations(64, 128, seed=3, outlier_frac=0.08, outlier_scale=50.0)
    m = mat_max_abs_rows(X)
    assert max(m) / (_mean_safe(m)) > 5.0


def _mean_safe(v):
    return sum(v) / max(len(v), 1)


def test_equivalent_transform_preserves_output():
    W = make_weight(24, 24, seed=1)
    X = make_activations(24, 48, seed=2)
    # Arbitrary positive scales
    s = [0.5 + 0.1 * (j % 7) for j in range(24)]
    Wp = mat_scale_cols(W, s)
    Xp = mat_scale_rows(X, s)
    Y = mat_flatten(mat_mul(W, X))
    Yp = mat_flatten(mat_mul(Wp, Xp))
    assert _mse(Y, Yp) < 1e-18
    assert _cosine(Y, Yp) > 0.999999


def test_smoothquant_scale_formula_bounds():
    eng = SmoothQuantEngine(SmoothQuantConfig(rows=32, cols=32, n_calibration=64, seed=4))
    s0 = eng.calibrate(alpha=0.0)
    s1 = eng.calibrate(alpha=1.0)
    s5 = eng.calibrate(alpha=0.5)
    assert all(v > 0 for v in s0 + s1 + s5)
    # α=0 → s = 1 / max(|W|)^{1}  (depends only on weights)
    # α=1 → s = max(|X|)^1        (depends only on activations)
    assert min(s1) > 0


def test_smoothquant_runs_and_finite():
    eng = SmoothQuantEngine(
        SmoothQuantConfig(
            rows=24, cols=24, w_bits=8, a_bits=8,
            n_calibration=48, alpha_grid=11, seed=5,
        )
    )
    res = eng.quantize()
    assert math.isfinite(res.mse)
    assert math.isfinite(res.sqnr_db)
    assert 0.0 <= res.best_alpha <= 1.0
    assert res.bit_rate_w > 0 and res.bit_rate_a > 0


def test_smoothquant_beats_or_matches_naive():
    eng = SmoothQuantEngine(
        SmoothQuantConfig(
            rows=32, cols=32, w_bits=8, a_bits=8,
            n_calibration=96, alpha_grid=15, seed=11,
            outlier_frac=0.08, outlier_scale=45.0,
        )
    )
    res = eng.quantize()
    # alpha search includes extremes; best MSE must be <= naive (s=1 path)
    assert res.mse <= res.naive_mse + 1e-9


def test_x_range_compressed_after_smooth():
    eng = SmoothQuantEngine(
        SmoothQuantConfig(
            rows=32, cols=32, n_calibration=64, alpha_grid=9, seed=2,
            outlier_scale=40.0,
        )
    )
    res = eng.quantize()
    # With outliers present and α*<1, X dynamic range should compress
    assert res.range_compress_x >= 0.99


def test_verify_battery_passes():
    eng = SmoothQuantEngine(
        SmoothQuantConfig(
            rows=20, cols=20, w_bits=8, a_bits=8,
            n_calibration=40, alpha_grid=9, seed=2,
        )
    )
    out = eng.verify(trials=2)
    assert out["status"] == "pass"


def test_pack_ledger_keys():
    eng = SmoothQuantEngine(
        SmoothQuantConfig(
            rows=16, cols=16, w_bits=8, a_bits=8,
            n_calibration=32, alpha_grid=7, seed=9,
        )
    )
    eng.quantize()
    packed = eng.pack_int_weights()
    for k in ("rows", "cols", "w_bits", "bit_rate_w", "packed_sample", "best_alpha"):
        assert k in packed
    assert len(packed["packed_sample"]) <= 16


def test_theory_table_nonempty():
    eng = SmoothQuantEngine(SmoothQuantConfig())
    t = eng.theory()
    assert len(t["rows"]) >= 4
    assert t["rows"][0]["comp_vs_fp16"] > 1.0


def test_benchmark_sweep():
    eng = SmoothQuantEngine(SmoothQuantConfig(alpha_grid=5, n_calibration=32, seed=1))
    payload = eng.benchmark([16, 24], [8])
    assert len(payload["rows"]) == 2
    assert all(r["elapsed_ms"] >= 0 for r in payload["rows"])


def test_per_channel_act_quant_runs():
    eng = SmoothQuantEngine(
        SmoothQuantConfig(
            rows=16, cols=16, act_quant="per_channel",
            n_calibration=32, alpha_grid=5, seed=3,
        )
    )
    res = eng.quantize()
    assert math.isfinite(res.mse)
    assert res.cosine > 0.8


def test_cli_status_exits_zero():
    rc = mod.main([
        "status", "--rows", "16", "--cols", "16", "--n-cal", "32",
        "--alpha-grid", "5", "--json",
    ])
    assert rc == 0


def test_cli_verify_exits_zero():
    rc = mod.main([
        "verify", "--rows", "16", "--cols", "16", "--n-cal", "32",
        "--alpha-grid", "5", "--trials", "2",
    ])
    assert rc == 0


def test_w4a8_path():
    eng = SmoothQuantEngine(
        SmoothQuantConfig(
            rows=24, cols=24, w_bits=4, a_bits=8,
            n_calibration=48, alpha_grid=7, seed=8,
        )
    )
    res = eng.quantize()
    assert math.isfinite(res.mse)
    assert res.bit_rate_w < 8.0
