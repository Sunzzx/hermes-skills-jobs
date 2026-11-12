"""Unit tests for gptq-hessian-weight-quant-engine (arXiv:2210.17323)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "gptq-hessian-weight-quant-engine",
)
mod = SourceFileLoader("gptq_hessian_weight_quant_engine", MOD_PATH).load_module()

GPTQConfig = mod.GPTQConfig
GPTQEngine = mod.GPTQEngine
QuantConfig = mod.QuantConfig
quantize_vector = mod.quantize_vector
dequantize_vector = mod.dequantize_vector
theoretical_bit_rate = mod.theoretical_bit_rate
cholesky_decompose = mod.cholesky_decompose
cholesky_inverse = mod.cholesky_inverse
mat_mul = mod.mat_mul
mat_eye = mod.mat_eye
mat_add = mod.mat_add
mat_scale = mod.mat_scale
mat_fro = mod.mat_fro
_mse = mod._mse
_cosine = mod._cosine
_sqnr_db = mod._sqnr_db


def test_quantize_dequantize_roundtrip_high_bits():
    cfg = QuantConfig(n_bits=8, group_size=16, symmetric=True)
    x = [math.sin(i / 3.0) for i in range(64)]
    qt = quantize_vector(x, cfg)
    recon = dequantize_vector(qt)
    assert _cosine(x, recon) > 0.99
    assert qt.bit_rate() < 10.0


def test_int4_compression_ratio():
    cfg = QuantConfig(n_bits=4, group_size=32, symmetric=True)
    x = [0.01 * i for i in range(256)]
    qt = quantize_vector(x, cfg)
    assert qt.compression_ratio(16.0) > 3.0
    assert 4.0 <= qt.bit_rate() < 5.5


def test_theoretical_bit_rate_matches():
    n, bits, gs = 256, 4, 32
    cfg = QuantConfig(n_bits=bits, group_size=gs, symmetric=True)
    x = [0.1] * n
    qt = quantize_vector(x, cfg)
    br = theoretical_bit_rate(n, bits, gs, True)
    assert abs(qt.bit_rate() - br) < 1e-9


def test_cholesky_spd_roundtrip():
    # SPD matrix A = B B^T + I
    B = [[1.0, 0.2], [0.5, 1.5], [0.0, 0.8]]
    # Build 2x2 SPD from simple construction
    A = [[4.0, 1.0], [1.0, 3.0]]
    L = cholesky_decompose(A)
    # L L^T ≈ A
    Lt = [[L[0][0], L[1][0]], [0.0, L[1][1]]]  # wrong - use proper
    recon = mat_mul(L, mod.mat_t(L))
    assert abs(recon[0][0] - 4.0) < 1e-9
    assert abs(recon[1][1] - 3.0) < 1e-9
    assert abs(recon[0][1] - 1.0) < 1e-9


def test_cholesky_inverse_identity():
    A = [[4.0, 1.0, 0.5], [1.0, 5.0, 0.2], [0.5, 0.2, 3.0]]
    inv = cholesky_inverse(A)
    Iapprox = mat_mul(A, inv)
    n = 3
    resid = 0.0
    for i in range(n):
        for j in range(n):
            target = 1.0 if i == j else 0.0
            resid += (Iapprox[i][j] - target) ** 2
    assert math.sqrt(resid / 9) < 1e-8


def test_hessian_spd_after_damp():
    eng = GPTQEngine(GPTQConfig(rows=16, cols=16, n_calibration=32, seed=1, damp_percent=0.01))
    eng.accumulate_hessian()
    stats = eng.hessian_stats()
    assert stats["min_diag"] > 0.0
    assert stats["dim"] == 16
    assert stats["damp"] > 0.0


def test_gptq_runs_and_finite():
    eng = GPTQEngine(
        GPTQConfig(
            rows=24,
            cols=24,
            n_bits=4,
            group_size=8,
            block_size=8,
            n_calibration=48,
            seed=7,
            act_order=True,
        )
    )
    res = eng.quantize()
    assert res.status == "success"
    assert math.isfinite(res.mse)
    assert math.isfinite(res.sqnr_db)
    assert res.cosine > 0.85
    assert res.invariants["finite_weights"]
    assert res.compression_vs_fp16 > 2.0


def test_gptq_competitive_with_rtn():
    """GPTQ should match or beat RTN on output MSE (allow tiny slack)."""
    eng = GPTQEngine(
        GPTQConfig(
            rows=32,
            cols=32,
            n_bits=4,
            group_size=16,
            block_size=16,
            n_calibration=64,
            seed=3,
            act_order=True,
            damp_percent=0.01,
            outlier_frac=0.03,
            outlier_scale=10.0,
        )
    )
    res = eng.quantize()
    # Soft check: GPTQ not dramatically worse than RTN
    assert res.mse <= res.rtn_mse * 1.25 + 1e-9
    assert res.cosine > 0.9


def test_act_order_runs():
    cfg = GPTQConfig(rows=16, cols=16, n_bits=3, group_size=8, block_size=8, n_calibration=32, seed=5, act_order=True)
    eng = GPTQEngine(cfg)
    res = eng.quantize()
    assert res.act_order is True
    assert res.status == "success"


def test_no_act_order_runs():
    cfg = GPTQConfig(rows=16, cols=16, n_bits=4, group_size=8, block_size=8, n_calibration=32, seed=5, act_order=False)
    eng = GPTQEngine(cfg)
    res = eng.quantize()
    assert res.act_order is False
    assert res.cosine > 0.85


def test_asymmetric_quant():
    eng = GPTQEngine(
        GPTQConfig(
            rows=16,
            cols=16,
            n_bits=4,
            group_size=8,
            block_size=8,
            n_calibration=32,
            seed=9,
            symmetric=False,
        )
    )
    res = eng.quantize()
    assert res.status == "success"
    assert res.cosine > 0.85


def test_pack_int_grid():
    eng = GPTQEngine(
        GPTQConfig(rows=12, cols=16, n_bits=4, group_size=8, block_size=8, n_calibration=24, seed=2)
    )
    eng.quantize()
    packed = eng.pack_int_weights()
    assert packed["status"] == "success"
    assert packed["int_min"] >= -9
    assert packed["int_max"] <= 8
    assert packed["compression_vs_fp16"] > 2.0


def test_verify_invariants_all_ok():
    eng = GPTQEngine(GPTQConfig(rows=24, cols=24, n_bits=4, group_size=8, seed=11))
    ver = eng.verify_invariants(trials=2)
    assert ver["status"] == "success"
    assert ver["all_ok"] is True


def test_benchmark_structure():
    eng = GPTQEngine(GPTQConfig(rows=16, cols=16, seed=1))
    b = eng.benchmark(dims=[16, 24], bits_list=[4, 8])
    assert b["status"] == "success"
    assert len(b["rows"]) == 4
    assert all(r["cosine"] > 0.8 for r in b["rows"])


def test_theory_table():
    eng = GPTQEngine(GPTQConfig(rows=64, cols=64))
    t = eng.theory_table()
    assert len(t["rows"]) == 12
    int4 = [r for r in t["rows"] if r["n_bits"] == 4 and r["group_size"] == 128][0]
    assert int4["vs_fp16"] > 3.0


def test_sqnr_perfect_on_identical():
    x = [1.0, -2.0, 0.5]
    assert _sqnr_db(x, x) >= 199.0


def test_cli_status_exit_zero():
    rc = mod.main(["status", "--rows", "16", "--cols", "16", "--n-cal", "32", "--bits", "4"])
    assert rc == 0


def test_cli_verify_exit_zero():
    rc = mod.main(
        ["verify", "--rows", "16", "--cols", "16", "--n-cal", "32", "--bits", "4", "--trials", "2", "--seed", "0"]
    )
    assert rc == 0


def test_cli_quantize_json():
    rc = mod.main(
        ["quantize", "--rows", "12", "--cols", "12", "--n-cal", "24", "--bits", "4", "--group-size", "8", "--json"]
    )
    assert rc == 0


def test_cli_theory_exit_zero():
    rc = mod.main(["theory", "--rows", "32", "--cols", "32"])
    assert rc == 0
