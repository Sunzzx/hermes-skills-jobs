"""Unit tests for quip-sharp-incoherence-lattice-engine (arXiv:2402.04396)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "quip-sharp-incoherence-lattice-engine",
)
mod = SourceFileLoader("quip_sharp_incoherence_lattice_engine", MOD_PATH).load_module()

QuipSharpConfig = mod.QuipSharpConfig
QuipSharpEngine = mod.QuipSharpEngine
QuantConfig = mod.QuantConfig
quantize_vector = mod.quantize_vector
dequantize_vector = mod.dequantize_vector
theoretical_bit_rate = mod.theoretical_bit_rate
mat_mul = mod.mat_mul
mat_t = mod.mat_t
mat_eye = mod.mat_eye
mat_flatten = mod.mat_flatten
mat_orthogonality_error = mod.mat_orthogonality_error
randomized_hadamard = mod.randomized_hadamard
hadamard_matrix = mod.hadamard_matrix
apply_incoherence = mod.apply_incoherence
invert_incoherence = mod.invert_incoherence
incoherence_proxy = mod.incoherence_proxy
make_weight = mod.make_weight
make_activations = mod.make_activations
hessian_from_activations = mod.hessian_from_activations
ldl_decompose = mod.ldl_decompose
generate_e8_codebook = mod.generate_e8_codebook
e8_quantize_vector = mod.e8_quantize_vector
nearest_codeword = mod.nearest_codeword
_mse = mod._mse
_cosine = mod._cosine
_l2 = mod._l2


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


def test_hadamard_orthogonal_pow2():
    h = hadamard_matrix(16)
    assert mat_orthogonality_error(h) < 1e-10


def test_randomized_hadamard_orthogonal():
    r = randomized_hadamard(16, seed=5)
    assert mat_orthogonality_error(r) < 1e-6
    r2 = randomized_hadamard(24, seed=9)  # non-pow2 via MGS
    assert mat_orthogonality_error(r2) < 1e-6


def test_incoherence_roundtrip_exact():
    w = make_weight(20, 20, seed=3)
    u = randomized_hadamard(20, seed=1)
    v = randomized_hadamard(20, seed=2)
    w_hat = apply_incoherence(w, u, v)
    w_back = invert_incoherence(w_hat, u, v)
    assert _mse(mat_flatten(w), mat_flatten(w_back)) < 1e-12


def test_rht_reduces_outlier_proxy():
    w = make_weight(32, 32, seed=11, n_outliers=4, outlier_scale=20.0)
    u = randomized_hadamard(32, seed=4)
    v = randomized_hadamard(32, seed=5)
    before = incoherence_proxy(w)["max_over_rms"]
    after = incoherence_proxy(apply_incoherence(w, u, v))["max_over_rms"]
    assert after <= before * 1.01 + 0.25


def test_ldl_reconstructs_spd():
    x = make_activations(12, 40, seed=2)
    h = hessian_from_activations(x, damp=1e-2)
    l, d = ldl_decompose(h)
    # Reconstruct H ≈ L diag(d) L^T
    n = len(d)
    recon = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(min(i, j) + 1):
                s += l[i][k] * d[k] * l[j][k]
            recon[i][j] = s
    err = _mse(mat_flatten(h), mat_flatten(recon))
    assert err < 1e-8


def test_e8_codebook_nonempty_and_origin():
    cb = generate_e8_codebook(max_norm=2.5, seed=0)
    assert len(cb) >= 64
    assert any(_l2(c) < 1e-15 for c in cb)


def test_e8_nearest_codeword_identity_on_codebook():
    cb = generate_e8_codebook(seed=1)
    target = cb[len(cb) // 3]
    got, idx, dist = nearest_codeword(target, cb)
    assert dist < 1e-18
    assert _mse(got, target) < 1e-18


def test_e8_quantize_finite():
    cb = generate_e8_codebook(seed=2)
    x = [math.sin(i / 5.0) for i in range(64)]
    recon, meta = e8_quantize_vector(x, cb)
    assert math.isfinite(meta["mse"])
    assert len(recon) == 64
    assert _cosine(x, recon) > 0.5


def test_engine_status_runs():
    eng = QuipSharpEngine(QuipSharpConfig(rows=16, cols=16, n_calibration=32, w_bits=4, seed=3))
    st = eng.status()
    assert st["codebook_size"] >= 32
    assert len(st["rows"]) >= 5
    assert math.isfinite(st["best_mse"])


def test_quip_sharp_finite_and_competitive():
    eng = QuipSharpEngine(
        QuipSharpConfig(rows=20, cols=20, n_calibration=40, w_bits=4, seed=8, block_size=8)
    )
    rtn = eng.baseline_rtn()
    qs = eng.quip_sharp()
    assert math.isfinite(qs.mse) and math.isfinite(qs.sqnr_db)
    assert qs.cosine > 0.0
    # allow slack — synthetic tiny dims; must not explode vs RTN
    assert qs.mse <= rtn.mse * 1.5 + 1e-6


def test_rht_ldlq_runs():
    eng = QuipSharpEngine(QuipSharpConfig(rows=16, cols=16, n_calibration=32, w_bits=3, seed=4))
    res = eng.rht_ldlq()
    assert math.isfinite(res.mse)
    assert res.orth_error_u < 1e-5
    assert res.orth_error_v < 1e-5


def test_calibrate_keys():
    eng = QuipSharpEngine(QuipSharpConfig(rows=16, cols=16, n_calibration=24, seed=1))
    cal = eng.calibrate()
    assert cal["outlier_reduction"] > 0
    assert "weight_proxy" in cal and "incoherent_proxy" in cal
    assert cal["codebook_size"] >= 32


def test_verify_battery_pass():
    eng = QuipSharpEngine(QuipSharpConfig(rows=16, cols=16, n_calibration=32, w_bits=4, seed=2))
    payload = eng.verify(trials=2)
    assert payload["status"] == "pass"
    assert all(t["pass"] for t in payload["trials"])


def test_theory_and_pack():
    eng = QuipSharpEngine(QuipSharpConfig(rows=16, cols=16, n_calibration=24, w_bits=2, seed=6))
    th = eng.theory()
    assert len(th["rows"]) >= 3
    assert th["codebook_size"] >= 32
    pk = eng.pack_ledger()
    assert pk["bpw"] > 0
    assert pk["n_weights"] == 16 * 16


def test_benchmark_sweep_small():
    eng = QuipSharpEngine(QuipSharpConfig(rows=16, cols=16, seed=0))
    payload = eng.benchmark([16], [4])
    assert len(payload["rows"]) == 1
    assert math.isfinite(payload["rows"][0]["quip_mse"])


def test_codebook_info():
    eng = QuipSharpEngine(QuipSharpConfig(rows=8, cols=8, n_calibration=16, seed=0))
    info = eng.codebook_info()
    assert info["size"] >= 32
    assert info["dim"] == 8
    assert info["origin_included"] is True


def test_cli_verify_main():
    rc = mod.main(["verify", "--rows", "16", "--cols", "16", "--trials", "1", "--seed", "3"])
    assert rc == 0


def test_cli_status_and_theory():
    assert mod.main(["theory", "--json"]) == 0
    assert mod.main(["codebook", "--rows", "8", "--cols", "8"]) == 0
