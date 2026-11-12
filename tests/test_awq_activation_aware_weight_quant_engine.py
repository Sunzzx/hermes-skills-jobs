"""Unit tests for awq-activation-aware-weight-quant-engine (arXiv:2306.00978)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "awq-activation-aware-weight-quant-engine",
)
mod = SourceFileLoader("awq_activation_aware_weight_quant_engine", MOD_PATH).load_module()

AWQConfig = mod.AWQConfig
AWQEngine = mod.AWQEngine
QuantConfig = mod.QuantConfig
quantize_vector = mod.quantize_vector
dequantize_vector = mod.dequantize_vector
theoretical_bit_rate = mod.theoretical_bit_rate
activation_saliency = mod.activation_saliency
normalize_saliency = mod.normalize_saliency
mat_mul = mod.mat_mul
mat_scale_cols = mod.mat_scale_cols
mat_scale_rows = mod.mat_scale_rows
mat_flatten = mod.mat_flatten
_mse = mod._mse
_cosine = mod._cosine
make_weight = mod.make_weight
make_activations = mod.make_activations


def test_quantize_dequantize_roundtrip_high_bits():
    cfg = QuantConfig(n_bits=8, group_size=16, symmetric=False)
    x = [math.sin(i / 3.0) for i in range(64)]
    qt = quantize_vector(x, cfg)
    recon = dequantize_vector(qt)
    assert _cosine(x, recon) > 0.99
    assert qt.bit_rate() < 12.0


def test_int4_compression_ratio():
    cfg = QuantConfig(n_bits=4, group_size=32, symmetric=False)
    x = [0.01 * i for i in range(256)]
    qt = quantize_vector(x, cfg)
    assert qt.compression_ratio(16.0) > 2.5
    assert 4.0 <= qt.bit_rate() < 6.5


def test_theoretical_bit_rate_matches():
    n, bits, gs = 256, 4, 32
    cfg = QuantConfig(n_bits=bits, group_size=gs, symmetric=False)
    x = [0.1] * n
    qt = quantize_vector(x, cfg)
    br = theoretical_bit_rate(n, bits, gs, False)
    assert abs(qt.bit_rate() - br) < 1e-9


def test_saliency_positive_and_normalized():
    X = make_activations(32, 64, seed=3)
    s = activation_saliency(X, "mean_abs")
    assert all(v > 0 for v in s)
    sn = normalize_saliency(s)
    g = math.exp(sum(math.log(v) for v in sn) / len(sn))
    assert abs(g - 1.0) < 1e-9


def test_equivalent_transform_preserves_output():
    W = make_weight(16, 16, seed=1)
    X = make_activations(16, 32, seed=2)
    s = normalize_saliency(activation_saliency(X))
    alpha = 0.5
    scales = [v ** alpha for v in s]
    Wp = mat_scale_cols(W, scales)
    Xp = mat_scale_rows(X, scales)
    Y = mat_flatten(mat_mul(W, X))
    Yp = mat_flatten(mat_mul(Wp, Xp))
    assert _mse(Y, Yp) < 1e-18
    assert _cosine(Y, Yp) > 0.999999


def test_awq_runs_and_finite():
    eng = AWQEngine(
        AWQConfig(
            rows=24, cols=24, n_bits=4, group_size=8,
            n_calibration=48, alpha_grid=11, seed=5,
        )
    )
    res = eng.quantize()
    assert math.isfinite(res.mse)
    assert math.isfinite(res.sqnr_db)
    assert 0.0 <= res.best_alpha <= 1.0
    assert res.bit_rate > 0


def test_awq_beats_or_matches_rtn():
    eng = AWQEngine(
        AWQConfig(
            rows=32, cols=32, n_bits=4, group_size=16,
            n_calibration=64, alpha_grid=15, seed=11,
        )
    )
    res = eng.quantize()
    # alpha search includes 0, so best MSE must be <= RTN
    assert res.mse <= res.rtn_mse + 1e-12


def test_verify_battery_passes():
    eng = AWQEngine(
        AWQConfig(
            rows=20, cols=20, n_bits=4, group_size=10,
            n_calibration=40, alpha_grid=9, seed=2,
        )
    )
    out = eng.verify(trials=2)
    assert out["status"] == "pass"


def test_pack_ledger_keys():
    eng = AWQEngine(
        AWQConfig(
            rows=16, cols=16, n_bits=4, group_size=8,
            n_calibration=32, alpha_grid=7, seed=9,
        )
    )
    eng.quantize()
    packed = eng.pack_int_weights()
    for k in ("rows", "cols", "n_bits", "bit_rate", "packed_sample", "best_alpha"):
        assert k in packed
    assert len(packed["packed_sample"]) == 16


def test_theory_table_nonempty():
    eng = AWQEngine(AWQConfig())
    t = eng.theory()
    assert len(t["rows"]) >= 4
    assert t["rows"][0]["comp_vs_fp16"] > 1.0


def test_benchmark_sweep():
    eng = AWQEngine(AWQConfig(alpha_grid=5, n_calibration=32, seed=1))
    payload = eng.benchmark([16, 24], [4, 8])
    assert len(payload["rows"]) == 4
    assert all(r["elapsed_ms"] >= 0 for r in payload["rows"])


def test_cli_status_exits_zero():
    rc = mod.main([
        "status", "--rows", "16", "--cols", "16", "--n-cal", "32",
        "--alpha-grid", "5", "--json",
    ])
    assert rc == 0
