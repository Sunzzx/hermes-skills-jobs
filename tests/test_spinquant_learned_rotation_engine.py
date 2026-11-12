"""Unit tests for spinquant-learned-rotation-engine (arXiv:2405.16406)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "spinquant-learned-rotation-engine",
)
mod = SourceFileLoader("spinquant_learned_rotation_engine", MOD_PATH).load_module()

SpinQuantConfig = mod.SpinQuantConfig
SpinQuantEngine = mod.SpinQuantEngine
QuantConfig = mod.QuantConfig
quantize_vector = mod.quantize_vector
dequantize_vector = mod.dequantize_vector
theoretical_bit_rate = mod.theoretical_bit_rate
mat_mul = mod.mat_mul
mat_t = mod.mat_t
mat_eye = mod.mat_eye
mat_flatten = mod.mat_flatten
mat_orthogonality_error = mod.mat_orthogonality_error
cayley_orthogonal = mod.cayley_orthogonal
random_skew_symmetric = mod.random_skew_symmetric
random_cayley_rotation = mod.random_cayley_rotation
randomized_hadamard = mod.randomized_hadamard
make_weight = mod.make_weight
make_activations = mod.make_activations
_mse = mod._mse
_cosine = mod._cosine


def test_quantize_dequantize_roundtrip_int4():
    cfg = QuantConfig(n_bits=4, group_size=0, symmetric=True)
    x = [math.sin(i / 3.0) for i in range(64)]
    qt = quantize_vector(x, cfg)
    recon = dequantize_vector(qt)
    assert _cosine(x, recon) > 0.95
    assert qt.bit_rate() < 6.0


def test_theoretical_bit_rate_matches():
    n, bits, gs = 256, 4, 64
    cfg = QuantConfig(n_bits=bits, group_size=gs, symmetric=True)
    x = [0.1] * n
    qt = quantize_vector(x, cfg)
    br = theoretical_bit_rate(n, bits, gs, True)
    assert abs(qt.bit_rate() - br) < 1e-9


def test_cayley_produces_orthogonal():
    a = random_skew_symmetric(16, seed=3, scale=0.2)
    r = cayley_orthogonal(a)
    err = mat_orthogonality_error(r)
    assert err < 1e-8


def test_hadamard_orthogonal_pow2():
    r = randomized_hadamard(16, seed=5)
    assert mat_orthogonality_error(r) < 1e-10


def test_rotation_fp_equivalence():
    eng = SpinQuantEngine(SpinQuantConfig(rows=20, cols=20, n_calibration=40, seed=2))
    r = random_cayley_rotation(20, seed=11, scale=0.25)
    err = eng.fp_equivalence_error(r)
    assert err < 1e-18


def test_equivalent_transform_identity_path():
    W = make_weight(16, 16, seed=1)
    X = make_activations(16, 32, seed=2)
    R = mat_eye(16)
    Wp = mat_mul(W, R)
    Xp = mat_mul(mat_t(R), X)
    Y = mat_flatten(mat_mul(W, X))
    Yp = mat_flatten(mat_mul(Wp, Xp))
    assert _mse(Y, Yp) < 1e-18


def test_spinquant_runs_and_finite():
    eng = SpinQuantEngine(
        SpinQuantConfig(
            rows=16, cols=16, w_bits=4, a_bits=8,
            n_calibration=32, steps=6, seed=5,
        )
    )
    res = eng.optimize()
    assert math.isfinite(res.mse)
    assert math.isfinite(res.sqnr_db)
    assert res.orth_error < 1e-5
    assert res.bit_rate_w > 0


def test_spinquant_beats_or_matches_identity():
    eng = SpinQuantEngine(
        SpinQuantConfig(
            rows=20, cols=20, w_bits=4, a_bits=8,
            n_calibration=48, steps=10, seed=11,
            outlier_frac=0.1, outlier_scale=50.0,
        )
    )
    res = eng.optimize()
    assert res.mse <= res.identity_mse + 1e-9


def test_verify_battery_passes():
    eng = SpinQuantEngine(
        SpinQuantConfig(
            rows=16, cols=16, w_bits=4, a_bits=8,
            n_calibration=32, steps=6, seed=2,
        )
    )
    out = eng.verify(trials=2)
    assert out["status"] == "pass"


def test_pack_ledger_keys():
    eng = SpinQuantEngine(
        SpinQuantConfig(
            rows=12, cols=12, w_bits=4, a_bits=8,
            n_calibration=24, steps=4, seed=9,
        )
    )
    eng.optimize()
    packed = eng.pack_int_weights()
    for k in ("rows", "cols", "w_bits", "bit_rate_w", "packed_sample", "orth_error"):
        assert k in packed
    assert len(packed["packed_sample"]) <= 16


def test_theory_table_nonempty():
    eng = SpinQuantEngine(SpinQuantConfig())
    t = eng.theory()
    assert len(t["rows"]) >= 4
    assert t["rows"][0]["comp_vs_fp16"] > 1.0


def test_benchmark_sweep():
    eng = SpinQuantEngine(SpinQuantConfig(steps=4, n_calibration=24, seed=1))
    payload = eng.benchmark([12, 16], [4])
    assert len(payload["rows"]) == 2
    assert all(r["elapsed_ms"] >= 0 for r in payload["rows"])


def test_hadamard_rotation_path():
    eng = SpinQuantEngine(
        SpinQuantConfig(
            rows=16, cols=16, rotation="hadamard",
            n_calibration=32, steps=0, seed=3,
        )
    )
    res = eng.optimize()
    assert math.isfinite(res.mse)
    assert res.rotation_kind == "hadamard"


def test_identity_rotation_path():
    eng = SpinQuantEngine(
        SpinQuantConfig(
            rows=12, cols=12, rotation="identity",
            n_calibration=24, steps=0, seed=4,
        )
    )
    res = eng.optimize()
    assert abs(res.mse - res.identity_mse) < 1e-12


def test_cli_status_exits_zero():
    rc = mod.main([
        "status", "--rows", "12", "--cols", "12", "--n-cal", "24",
        "--steps", "4", "--json",
    ])
    assert rc == 0


def test_cli_verify_exits_zero():
    rc = mod.main([
        "verify", "--rows", "12", "--cols", "12", "--n-cal", "24",
        "--steps", "4", "--trials", "2",
    ])
    assert rc == 0


def test_w4a4_path():
    eng = SpinQuantEngine(
        SpinQuantConfig(
            rows=16, cols=16, w_bits=4, a_bits=4,
            n_calibration=32, steps=6, seed=8,
        )
    )
    res = eng.optimize()
    assert math.isfinite(res.mse)
    assert res.bit_rate_w < 8.0


def test_per_channel_act_quant_runs():
    eng = SpinQuantEngine(
        SpinQuantConfig(
            rows=12, cols=12, act_quant="per_channel",
            n_calibration=24, steps=4, seed=3,
        )
    )
    res = eng.optimize()
    assert math.isfinite(res.mse)
    assert res.cosine > 0.5
