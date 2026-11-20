"""Unit tests for hyperloglog-plusplus-cardinality: HLL / HLL++ sketches."""

import math
import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/hyperloglog-plusplus-cardinality")
)
eng = SourceFileLoader("hyperloglog_plusplus_cardinality", bin_path).load_module()


def test_hash_stable_and_seeded():
    assert eng.hash64("hermes", 0) == eng.hash64("hermes", 0)
    assert eng.hash64("hermes", 0) != eng.hash64("hermes", 1)
    assert eng.fnv1a64(b"abc") == eng.fnv1a64("abc")


def test_rho_leading_zeros_edges():
    assert eng.rho_leading_zeros(0, 8) == 9
    assert eng.rho_leading_zeros(0b10000000, 8) == 1
    assert eng.rho_leading_zeros(0b00000001, 8) == 8
    assert eng.rho_leading_zeros(0b01000000, 8) == 2


def test_alpha_m_special_and_asym():
    assert abs(eng.alpha_m(16) - 0.673) < 1e-12
    assert abs(eng.alpha_m(32) - 0.697) < 1e-12
    assert abs(eng.alpha_m(64) - 0.709) < 1e-12
    assert 0.70 < eng.alpha_m(1 << 14) < 0.73


def test_relative_std_error_formula():
    m = 1 << 14
    assert abs(eng.relative_std_error(m) - 1.04 / math.sqrt(m)) < 1e-12
    assert eng.relative_std_error(1 << 10) > eng.relative_std_error(1 << 16)


def test_linear_counting_math():
    assert eng.linear_counting(100, 100) == 0.0
    assert abs(eng.linear_counting(100, 99) - (-100 * math.log(0.99))) < 1e-12


def test_empty_hllpp_zero():
    h = eng.HyperLogLog(p=8, seed=1, plusplus=True)
    assert h.estimate() == 0.0
    assert h.zeros() == 256


def test_small_cardinality_linear_path():
    keys = eng.make_keys(40, seed=2)
    h = eng.HyperLogLog(p=8, seed=2, plusplus=True)
    h.add_many(keys)
    err = eng.relative_error(h.estimate(), 40)
    assert err < 0.20


def test_medium_hllpp_within_band():
    n, p = 2500, 11
    h = eng.HyperLogLog(p=p, seed=5, plusplus=True)
    h.add_many(eng.make_keys(n, seed=5))
    err = eng.relative_error(h.estimate(), n)
    rse = eng.relative_std_error(1 << p)
    assert err < 3.5 * rse + 0.05


def test_classic_hll_also_works():
    n = 1500
    h = eng.HyperLogLog(p=10, seed=3, plusplus=False)
    h.add_many(eng.make_keys(n, seed=3))
    assert eng.relative_error(h.estimate(), n) < 0.22


def test_duplicates_do_not_inflate():
    base = eng.make_keys(600, seed=9)
    h = eng.HyperLogLog(p=11, seed=9, plusplus=True)
    h.add_many(base * 8)
    assert eng.relative_error(h.estimate(), 600) < 0.15


def test_merge_union_estimate():
    p = 10
    a = eng.HyperLogLog(p=p, seed=4, plusplus=True)
    b = eng.HyperLogLog(p=p, seed=4, plusplus=True)
    a.add_many(eng.make_keys(400, seed=1, prefix="a") + eng.make_keys(150, seed=3, prefix="o"))
    b.add_many(eng.make_keys(400, seed=2, prefix="b") + eng.make_keys(150, seed=3, prefix="o"))
    a.merge(b)
    truth = 400 + 400 + 150
    assert eng.relative_error(a.estimate(), truth) < 0.20


def test_sparse_to_dense_promotion():
    h = eng.HyperLogLog(p=8, seed=0, plusplus=True, sparse_threshold=16)
    h.add_many(eng.make_keys(60, seed=0))
    h._ensure_dense()
    assert h._dense is not None and len(h._dense) == 256


def test_pack_unpack_sparse():
    v = eng._pack_sparse(42, 9)
    i, r = eng._unpack_sparse(v)
    assert i == 42 and r == 9


def test_pcsa_and_linear_counter():
    pc = eng.PCSA(maps=32, width=32, seed=1)
    pc.add_many(eng.make_keys(800, seed=1))
    assert eng.relative_error(pc.estimate(), 800) < 0.55

    lc = eng.LinearCounter(m=2048, seed=2)
    lc.add_many(eng.make_keys(350, seed=2))
    assert eng.relative_error(lc.estimate(), 350) < 0.15


def test_exact_counter():
    ex = eng.ExactCounter()
    ex.add_many(["x", "y", "x", "z", "y"])
    assert ex.estimate() == 3.0


def test_theory_ledger_shape():
    th = eng.theory_ledger(p=14, n=1_000_000)
    assert th["m"] == 16384
    assert 0.7 < th["rel_stderr_pct"] < 2.0
    assert th["hll_dense_bytes"] == (16384 * 6 + 7) // 8
    assert "add" in th["complexity"]


def test_compare_all_has_winner():
    out = eng.compare_all(n_unique=1000, p=11, seed=7)
    assert out["winner_accuracy"] in ("hll", "hllpp", "linear", "pcsa")
    assert len(out["rows"]) == 5


def test_evaluate_hllpp_ok():
    row = eng.evaluate_sketch("hllpp", n_unique=1200, p=11, seed=8)
    assert row["ok"] is True
    assert row["rel_error"] < 0.20


def test_copy_increments_independently():
    a = eng.HyperLogLog(p=8, seed=1, plusplus=True)
    a.add_many(eng.make_keys(30, seed=1))
    b = a.copy()
    b.add("totally-new-element-99")
    assert b.n_updates == a.n_updates + 1


def test_as_dict_schema():
    h = eng.HyperLogLog(p=9, seed=0, plusplus=True)
    h.add_many(eng.make_keys(100, seed=0))
    d = h.as_dict()
    for k in ("kind", "p", "m", "estimate", "rel_stderr_theory", "mode"):
        assert k in d
    assert d["kind"] == "hllpp"


def test_bias_interpolate_finite():
    assert math.isfinite(eng.interpolate_bias(1000.0, 10))
    assert math.isfinite(eng.interpolate_bias(0.0, 12))


def test_zipf_stream_estimate():
    stream = eng.make_stream(8000, 1000, seed=4)
    h = eng.HyperLogLog(p=11, seed=4, plusplus=True)
    h.add_many(stream)
    assert eng.relative_error(h.estimate(), 1000) < 0.20


def test_selftest_gate_green():
    assert eng.selftest(verbose=False) == 0
