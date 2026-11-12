"""Unit tests for qtip-trellis-incoherence-engine (NeurIPS 2024, arXiv:2406.11235)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "qtip-trellis-incoherence-engine",
)
mod = SourceFileLoader("qtip_trellis_incoherence_engine", MOD_PATH).load_module()

QTIPConfig = mod.QTIPConfig
QTIPEngine = mod.QTIPEngine
QuantConfig = mod.QuantConfig
TrellisConfig = mod.TrellisConfig
build_trellis = mod.build_trellis
viterbi_encode = mod.viterbi_encode
tcq_decode = mod.tcq_decode
make_incoherence = mod.make_incoherence
incoherence_proxy = mod.incoherence_proxy
fwht = mod.fwht
ifwht = mod.ifwht
hadamard_matrix = mod.hadamard_matrix
make_weight = mod.make_weight
make_gaussian_vector = mod.make_gaussian_vector
make_activation = mod.make_activation
fp_linear = mod.fp_linear
mat_flatten = mod.mat_flatten
quantize_vector_rtn = mod.quantize_vector_rtn
dequantize_vector_rtn = mod.dequantize_vector_rtn
lloyd_max_levels = mod.lloyd_max_levels
_mse = mod._mse
_cosine = mod._cosine
_l2 = mod._l2
_sqnr_db = mod._sqnr_db
next_pow2 = mod.next_pow2


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(3) == 4
    assert next_pow2(32) == 32
    assert next_pow2(33) == 64


def test_fwht_involutive_up_to_scale():
    v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = fwht(v)
    z = ifwht(y)
    assert _mse(v, z) < 1e-12


def test_hadamard_orthogonal_rows():
    h = hadamard_matrix(8)
    # H H^T = n I
    for i in range(8):
        for j in range(8):
            d = sum(h[i][k] * h[j][k] for k in range(8))
            if i == j:
                assert abs(d - 8.0) < 1e-9
            else:
                assert abs(d) < 1e-9


def test_incoherence_roundtrip_exact():
    v = make_gaussian_vector(40, seed=1)
    inc = make_incoherence(len(v), seed=9)
    y = inc.apply_vector(v)
    x = inc.invert_vector(y, len(v))
    assert _mse(v, x) < 1e-20


def test_incoherence_reduces_outlier_proxy_on_spiky():
    v = make_gaussian_vector(128, seed=2)
    v[0] = 40.0
    v[17] = -35.0
    before = incoherence_proxy(v)
    inc = make_incoherence(len(v), seed=3)
    after = incoherence_proxy(inc.apply_vector(v))
    # max/rms should drop after spreading energy
    assert after["max_over_rms"] < before["max_over_rms"]


def test_rtn_roundtrip_cosine():
    cfg = QuantConfig(n_bits=4, group_size=16, symmetric=True)
    x = [math.sin(i / 3.0) for i in range(64)]
    q = quantize_vector_rtn(x, cfg)
    recon = dequantize_vector_rtn(q)
    assert _cosine(x, recon) > 0.95
    assert q.bit_rate() <= 6.0 + 1e-12


def test_lloyd_levels_sorted_and_cover():
    vals = make_gaussian_vector(200, seed=4)
    cb = lloyd_max_levels(vals, 8, iters=15)
    assert len(cb) == 8
    assert cb == sorted(cb)
    assert min(cb) >= min(vals) - 1e-9
    assert max(cb) <= max(vals) + 1e-9


def test_trellis_graph_sizes():
    cfg = TrellisConfig(n_states=8, rate_bits=2, seed=0)
    cb = build_trellis(cfg)
    assert cb.n_states == 8
    assert cb.n_branches == 4
    assert len(cb.levels) == 8
    assert all(len(row) == 4 for row in cb.levels)
    assert all(0 <= ns < 8 for row in cb.next_state for ns in row)


def test_viterbi_decode_agrees_with_encode():
    x = make_gaussian_vector(64, seed=5)
    cb = build_trellis(TrellisConfig(n_states=8, rate_bits=2, seed=1), calib=x)
    enc = viterbi_encode(x, cb)
    dec = tcq_decode(enc.branches, cb, start_state=enc.states[0])
    assert _mse(enc.recon, dec) < 1e-18
    assert len(enc.branches) == 64
    assert len(enc.states) == 65


def test_viterbi_beats_or_ties_random_path_quality():
    x = make_gaussian_vector(48, seed=6)
    cb = build_trellis(TrellisConfig(n_states=8, rate_bits=2, seed=2), calib=x)
    enc = viterbi_encode(x, cb)
    # random branch path should not beat Viterbi MSE
    import random as _r

    rng = _r.Random(0)
    s = 0
    recon = []
    for _ in range(48):
        b = rng.randrange(cb.n_branches)
        recon.append(cb.levels[s][b])
        s = cb.next_state[s][b]
    assert enc.mse <= _mse(x, recon) + 1e-12


def test_qtip_vector_finite_and_cosine():
    eng = QTIPEngine(QTIPConfig(n_bits=2, n_states=8, group_size=32, refine_iters=3))
    x = make_gaussian_vector(96, seed=8)
    r = eng.quantize_vector(x)
    assert math.isfinite(r["mse"])
    assert r["cosine"] > 0.5
    assert 1.5 <= r["bit_rate"] <= 5.0


def test_qtip_matrix_metrics():
    eng = QTIPEngine(QTIPConfig(n_bits=2, n_states=8, group_size=64, refine_iters=2))
    w = make_weight(12, 16, seed=10, n_outliers=2)
    res = eng.quantize_matrix(w)
    assert res.method == "qtip"
    assert math.isfinite(res.metrics["weight_mse"])
    assert res.metrics["weight_cosine"] > 0.5
    assert res.bit_rate < 5.0


def test_qtip_competitive_with_rtn_on_outliers():
    eng = QTIPEngine(QTIPConfig(n_bits=2, n_states=8, group_size=32, refine_iters=4))
    w = make_weight(16, 16, seed=21, n_outliers=5, outlier_scale=12.0)
    q = eng.quantize_matrix(w)
    r = eng.rtn_baseline(w)
    # Allow slack — synthetic Gaussian+spikes; primary invariants elsewhere
    assert q.metrics["weight_mse"] <= r.metrics["weight_mse"] * 1.6 + 1e-9
    assert q.metrics["weight_cosine"] > 0.45


def test_tcq_only_and_lloyd_baselines_run():
    eng = QTIPEngine(QTIPConfig(n_bits=3, n_states=8, refine_iters=2))
    w = make_weight(10, 12, seed=12)
    t = eng.tcq_only_baseline(w)
    l = eng.lloyd_baseline(w)
    v = eng.vq_baseline(w)
    assert t.method == "tcq-only"
    assert l.method == "lloyd-max"
    assert v.method == "nn-vq"
    assert math.isfinite(t.metrics["weight_mse"])
    assert math.isfinite(l.metrics["weight_mse"])
    assert math.isfinite(v.metrics["weight_mse"])


def test_status_and_verify_pass():
    eng = QTIPEngine(QTIPConfig(n_bits=2, n_states=8, refine_iters=2))
    st = eng.status(rows=12, cols=12)
    assert st["status"] == "ok"
    assert "qtip" in st
    vf = eng.verify(trials=2)
    assert vf["status"] == "pass"


def test_benchmark_theory_pack_ablate():
    eng = QTIPEngine(QTIPConfig(n_bits=2, n_states=8, refine_iters=2))
    bm = eng.benchmark([8, 12], bits_list=[2, 4])
    assert len(bm["rows"]) == 4
    th = eng.theory()
    assert th["fp16_7b_gib"] > 10.0
    assert len(th["rows"]) >= 4
    assert "2406.11235" in th["arxiv"]
    pk = eng.pack(12, 12)
    assert pk["bpw_packed"] > 0
    assert "trellis" in pk["deploy_note"].lower() or "viterbi" in pk["deploy_note"].lower()
    ab = eng.ablate(rows=8, cols=8)
    assert ab["winner"] in {"rtn", "lloyd", "nn-vq", "tcq-only", "qtip"}
    assert len(ab["rows"]) == 5


def test_encode_decode_calibrate_quantize():
    eng = QTIPEngine(QTIPConfig(n_bits=2, n_states=8, refine_iters=2))
    enc = eng.encode(n=64, bits=2)
    assert enc["n"] == 64
    assert math.isfinite(enc["mse"])
    dec = eng.decode_demo(n=32)
    assert dec["encode_decode_mse"] < 1e-18
    cal = eng.calibrate(n=64, seed=1)
    assert cal["incoherence_roundtrip_mse"] < 1e-20
    q = eng.quantize(rows=10, cols=10)
    assert "output_mse_qtip" in q
    assert math.isfinite(q["output_cosine_qtip"])


def test_fp_linear_shapes():
    w = make_weight(10, 12, seed=0)
    x = make_activation(4, 12, seed=1)
    y = fp_linear(x, w)
    assert len(y) == 4
    assert len(y[0]) == 10


def test_sqnr_perfect_is_high():
    v = [1.0, 2.0, 3.0]
    assert _sqnr_db(v, v) >= 90.0
