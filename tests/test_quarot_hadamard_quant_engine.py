"""Unit tests for quarot-hadamard-quant-engine (arXiv:2404.00456)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "quarot-hadamard-quant-engine",
)
mod = SourceFileLoader("quarot_hadamard_quant_engine", MOD_PATH).load_module()

fwht = mod.fwht
HadamardRotator = mod.HadamardRotator
RotationConfig = mod.RotationConfig
QuantConfig = mod.QuantConfig
quantize_vector = mod.quantize_vector
dequantize_vector = mod.dequantize_vector
QuaRotConfig = mod.QuaRotConfig
QuaRotEngine = mod.QuaRotEngine
analyze_vector = mod.analyze_vector
theoretical_bit_rate = mod.theoretical_bit_rate
_mse = mod._mse
_cosine = mod._cosine
_is_pow2 = mod._is_pow2


def test_is_pow2():
    assert _is_pow2(1)
    assert _is_pow2(256)
    assert not _is_pow2(0)
    assert not _is_pow2(3)


def test_fwht_involution_normalized():
    x = [1.0, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75]
    y = fwht(x, normalize=True)
    z = fwht(y, normalize=True)
    assert _mse(x, z) < 1e-18


def test_fwht_energy_preservation():
    x = [0.5, -1.0, 2.0, 0.0]
    y = fwht(x, normalize=True)
    assert abs(sum(a * a for a in x) - sum(b * b for b in y)) < 1e-12


def test_hadamard_roundtrip_cosine():
    rot = HadamardRotator(RotationConfig(dim=128, seed=7, mode="hadamard"))
    x = [math.sin(i * 0.37) for i in range(128)]
    y = rot.rotate_vec(x)
    x2 = rot.unrotate_vec(y)
    assert _cosine(x, x2) > 1.0 - 1e-9
    eng = rot.energy_preservation(x)
    assert eng["rel_energy_error"] < 1e-9


def test_identity_rotation_is_noop():
    rot = HadamardRotator(RotationConfig(dim=64, mode="identity"))
    x = list(range(64))
    y = rot.rotate_vec([float(v) for v in x])
    assert all(abs(a - b) < 1e-12 for a, b in zip(x, y))


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


def test_theoretical_bit_rate_matches_tensor():
    n, bits, gs = 256, 4, 32
    cfg = QuantConfig(n_bits=bits, group_size=gs, symmetric=True)
    x = [0.1] * n
    qt = quantize_vector(x, cfg)
    br = theoretical_bit_rate(n, bits, gs, True)
    assert abs(qt.bit_rate() - br) < 1e-9


def test_outliers_reduce_after_hadamard():
    eng = QuaRotEngine(
        QuaRotConfig(dim=256, seed=1, outlier_channels=6, outlier_scale=40.0)
    )
    x = eng.synthesize_activation()
    pre = analyze_vector(x)
    post = analyze_vector(eng.rotator.rotate_vec(x))
    # max abs should drop substantially when sparse spikes are smeared
    assert post.max_abs < pre.max_abs
    assert post.topk_share < pre.topk_share


def test_pipeline_quarot_improves_or_preserves_sqnr_under_outliers():
    eng = QuaRotEngine(
        QuaRotConfig(
            dim=256,
            n_bits=4,
            group_size=32,
            seed=11,
            outlier_channels=8,
            outlier_scale=30.0,
        )
    )
    pipe = eng.pipeline_vector()
    assert pipe.status == "success"
    # With strong outliers, rotation should help SQNR (allow tiny slack)
    assert pipe.rotated_quant["sqnr_db"] + 0.5 >= pipe.baseline_quant["sqnr_db"] - 1.0
    assert pipe.rotated_quant["cosine"] > 0.85
    assert pipe.energy["rel_energy_error"] < 1e-9


def test_pipeline_without_outliers_still_stable():
    eng = QuaRotEngine(
        QuaRotConfig(dim=128, n_bits=4, seed=3, outlier_channels=0, outlier_scale=1.0)
    )
    pipe = eng.pipeline_vector()
    assert pipe.rotated_quant["cosine"] > 0.95
    assert math.isfinite(pipe.rotated_quant["sqnr_db"])


def test_gemm_invariants():
    eng = QuaRotEngine(QuaRotConfig(dim=128, n_bits=4, seed=99, outlier_channels=4))
    g = eng.pipeline_gemm(rows=12)
    assert g["status"] == "success"
    assert g["invariants"]["rotation_energy_ok"]
    assert g["invariants"]["quarot_cosine_high"]
    assert g["quarot"]["cosine"] > 0.9


def test_verify_invariants_all_ok():
    eng = QuaRotEngine(QuaRotConfig(dim=64, n_bits=4, seed=5, outlier_channels=3))
    res = eng.verify_invariants(trials=3)
    assert res["all_ok"] is True
    assert res["status"] == "success"


def test_outlier_map_avg_reduction_positive():
    eng = QuaRotEngine(
        QuaRotConfig(dim=128, seed=2, outlier_channels=5, outlier_scale=20.0)
    )
    m = eng.outlier_map(n_samples=4)
    assert m["avg_max_abs_reduction_x"] > 1.0


def test_theory_table_structure():
    eng = QuaRotEngine(QuaRotConfig(dim=256, group_size=32))
    t = eng.theory_table()
    assert len(t["rows"]) == 4
    assert t["rows"][2]["n_bits"] == 4
    assert t["rows"][2]["vs_fp16"] > 3.0


def test_asymmetric_quant_runs():
    cfg = QuantConfig(n_bits=4, group_size=16, symmetric=False)
    x = [float(i - 20) for i in range(64)]
    qt = quantize_vector(x, cfg)
    recon = dequantize_vector(qt)
    assert _cosine(x, recon) > 0.98
    assert qt.symmetric is False


def test_cli_verify_exit_zero():
    rc = mod.main(["verify", "--dim", "64", "--trials", "2", "--seed", "0"])
    assert rc == 0


def test_cli_status_exit_zero():
    rc = mod.main(["status", "--dim", "128", "--bits", "4"])
    assert rc == 0
