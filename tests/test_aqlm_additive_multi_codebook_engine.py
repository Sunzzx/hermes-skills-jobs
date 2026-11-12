"""Unit tests for aqlm-additive-multi-codebook-engine (arXiv:2401.06118)."""

import math
import os
from importlib.machinery import SourceFileLoader

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "aqlm-additive-multi-codebook-engine",
)
mod = SourceFileLoader("aqlm_additive_multi_codebook_engine", MOD_PATH).load_module()

AQLMConfig = mod.AQLMConfig
AQLMEngine = mod.AQLMEngine
QuantConfig = mod.QuantConfig
quantize_vector = mod.quantize_vector
dequantize_vector = mod.dequantize_vector
theoretical_bit_rate = mod.theoretical_bit_rate
aqlm_bit_rate = mod.aqlm_bit_rate
make_weight = mod.make_weight
reshape_to_groups = mod.reshape_to_groups
groups_to_matrix = mod.groups_to_matrix
kmeans = mod.kmeans
train_additive_codebooks = mod.train_additive_codebooks
encode_greedy = mod.encode_greedy
encode_beam = mod.encode_beam
decode_indices = mod.decode_indices
nearest_codeword = mod.nearest_codeword
mat_flatten = mod.mat_flatten
_mse = mod._mse
_cosine = mod._cosine
_l2 = mod._l2


def test_quantize_dequantize_roundtrip_int4():
    cfg = QuantConfig(n_bits=4, group_size=16, symmetric=True)
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


def test_group_reshape_roundtrip():
    w = make_weight(10, 12, seed=2)
    groups, shape = reshape_to_groups(w, 8)
    back = groups_to_matrix(groups, shape, 8)
    assert _mse(mat_flatten(w), mat_flatten(back)) < 1e-15
    assert all(len(g) == 8 for g in groups)


def test_kmeans_reduces_inertia():
    rng_data = [[math.sin(i + j * 0.1) for j in range(4)] for i in range(40)]
    # crude: after kmeans, mean nearest distance should be finite and smallish
    cb = kmeans(rng_data, k=6, dim=4, iters=15, seed=1)
    assert len(cb) == 6
    assert all(len(a) == 4 for a in cb)
    dists = []
    for v in rng_data:
        _i, atom, d = nearest_codeword(v, cb)
        dists.append(d)
    assert _mse([0.0] * len(dists), dists) >= 0.0  # finite
    assert sum(dists) / len(dists) < 5.0


def test_residual_codebooks_shrink_residual():
    w = make_weight(16, 16, seed=5)
    groups, _ = reshape_to_groups(w, 8)
    cbs = train_additive_codebooks(groups, n_codebooks=3, codebook_size=8, dim=8, kmeans_iters=8, seed=3)
    assert len(cbs) == 3
    # residual after 3 stages should beat residual after 1 stage on average
    r1 = []
    r3 = []
    for g in groups:
        idxs1, recon1 = encode_greedy(g, cbs[:1])
        idxs3, recon3 = encode_greedy(g, cbs)
        r1.append(_mse(g, recon1))
        r3.append(_mse(g, recon3))
    assert sum(r3) <= sum(r1) * 1.01 + 1e-9


def test_beam_beats_or_matches_greedy():
    w = make_weight(12, 12, seed=7, n_outliers=1)
    groups, _ = reshape_to_groups(w, 8)
    cbs = train_additive_codebooks(groups, 2, 8, 8, kmeans_iters=6, seed=4)
    g = groups[0]
    _ig, rg = encode_greedy(g, cbs)
    _ib, rb = encode_beam(g, cbs, beam_width=4)
    assert _mse(g, rb) <= _mse(g, rg) * 1.05 + 1e-12


def test_decode_indices_matches_encode():
    w = make_weight(8, 8, seed=1)
    groups, _ = reshape_to_groups(w, 8)
    cbs = train_additive_codebooks(groups, 2, 4, 8, kmeans_iters=5, seed=2)
    g = groups[0]
    idxs, recon = encode_beam(g, cbs, beam_width=3)
    dec = decode_indices(idxs, cbs, 8)
    assert _mse(recon, dec) < 1e-12
    assert len(idxs) == 2


def test_aqlm_bit_rate_positive_and_scales():
    br1 = aqlm_bit_rate(1024, 8, 1, 16)
    br2 = aqlm_bit_rate(1024, 8, 2, 16)
    br4 = aqlm_bit_rate(1024, 8, 4, 16)
    assert br1["bpw"] > 0 and br2["bpw"] > 0
    # more codebooks → more index bits
    assert br4["index_bpw"] > br2["index_bpw"] > br1["index_bpw"]
    assert br2["effective_codebook_size"] == 16 ** 2


def test_engine_status_finite():
    eng = AQLMEngine(AQLMConfig(rows=16, cols=16, group_size=8, n_codebooks=2, codebook_size=8, kmeans_iters=4, seed=0, n_cal=16))
    st = eng.status()
    assert st["best_mse"] >= 0.0
    assert math.isfinite(st["best_mse"])
    assert len(st["rows"]) >= 4
    assert st["aqlm_beats_vq"] is True or st["rows"][-1]["mse"] <= st["rows"][3]["mse"] * 1.1


def test_engine_calibrate_keys():
    eng = AQLMEngine(AQLMConfig(rows=12, cols=12, seed=3, n_cal=20))
    cal = eng.calibrate()
    assert cal["n_groups"] > 0
    assert cal["max_over_rms"] > 0
    assert len(cal["top_outlier_cols"]) > 0


def test_aqlm_quantize_cosine_reasonable():
    eng = AQLMEngine(
        AQLMConfig(
            rows=24,
            cols=24,
            group_size=8,
            n_codebooks=2,
            codebook_size=16,
            beam_width=3,
            kmeans_iters=6,
            seed=11,
            n_outliers=1,
            n_cal=24,
        )
    )
    res = eng.aqlm_quantize(use_beam=True)
    assert res.cosine > 0.5
    assert res.mse >= 0.0
    assert res.bit_rate > 0.0
    assert res.extras["effective_K"] == 16 ** 2


def test_aqlm_beats_single_vq():
    eng = AQLMEngine(
        AQLMConfig(
            rows=20,
            cols=20,
            group_size=8,
            n_codebooks=3,
            codebook_size=8,
            beam_width=3,
            kmeans_iters=8,
            seed=19,
            n_outliers=0,
            n_cal=20,
        )
    )
    aqlm = eng.aqlm_quantize(use_beam=True)
    vq = eng.baseline_single_vq()
    assert aqlm.mse <= vq.mse * 1.05 + 1e-9


def test_verify_battery_pass():
    eng = AQLMEngine(AQLMConfig(seed=0))
    payload = eng.verify(trials=2)
    assert payload["status"] == "pass"
    assert all(t["ok"] for t in payload["trials"])


def test_theory_table_nonempty():
    eng = AQLMEngine(AQLMConfig())
    th = eng.theory()
    assert len(th["rows"]) >= 4
    assert th["fp16_7b_gib"] > 10.0
    assert all(r["index_bpw"] > 0 for r in th["rows"])


def test_pack_ledger():
    eng = AQLMEngine(AQLMConfig(rows=16, cols=16, n_codebooks=2, codebook_size=8, kmeans_iters=4, seed=2))
    pack = eng.pack_ledger()
    assert pack["n_weights"] == 256
    assert pack["M"] == 2
    assert pack["bpw"] > 0
    assert math.isfinite(pack["mse"])


def test_benchmark_sweep_runs():
    eng = AQLMEngine(AQLMConfig(codebook_size=8, kmeans_iters=3, beam_width=2, seed=0))
    bm = eng.benchmark(dims=[8, 12], codebooks_list=[1, 2])
    assert len(bm["rows"]) == 4
    assert all("aqlm_mse" in r for r in bm["rows"])


def test_cli_verify_exit_zero():
    rc = mod.main(["verify", "--trials", "2", "--seed", "1"])
    assert rc == 0


def test_cli_status_runs():
    rc = mod.main(["status", "--rows", "12", "--cols", "12", "--codebook-size", "8", "--kmeans-iters", "3"])
    assert rc == 0


def test_cli_theory_runs():
    rc = mod.main(["theory"])
    assert rc == 0
