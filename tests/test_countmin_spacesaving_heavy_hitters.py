"""Unit tests for countmin-spacesaving-heavy-hitters streaming heavy-hitter oracle."""

import math
import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/countmin-spacesaving-heavy-hitters")
)
eng = SourceFileLoader("countmin_spacesaving_heavy_hitters", bin_path).load_module()


def test_hash_stable_and_seeded():
    assert eng.hash64("hermes", 0) == eng.hash64("hermes", 0)
    assert eng.hash64("hermes", 0) != eng.hash64("hermes", 1)
    assert eng.fnv1a64(b"abc") == eng.fnv1a64("abc")


def test_cms_width_depth_formulas():
    assert eng.cms_width(0.001) == int(math.ceil(math.e / 0.001))
    assert eng.cms_depth(1e-4) == int(math.ceil(math.log(1e4)))
    assert eng.cms_width(0.01) < eng.cms_width(0.001)


def test_threshold_helpers():
    assert abs(eng.spacesaving_threshold(1000, 10) - 100.0) < 1e-12
    assert abs(eng.misra_gries_threshold(1000, 9) - 100.0) < 1e-12


def test_relative_error():
    assert eng.relative_error(0, 0) == 0.0
    assert abs(eng.relative_error(5, 10) - 0.5) < 1e-12


def test_empty_sketches_zero():
    assert eng.CountMinSketch(depth=3, width=32, seed=0).estimate("x") == 0
    assert eng.CountSketch(depth=3, width=32, seed=0).estimate("x") == 0
    assert eng.SpaceSaving(capacity=5).estimate("x") == 0
    assert eng.MisraGries(capacity=5).estimate("x") == 0


def test_cms_never_underestimates():
    stream = eng.make_zipf_stream(4000, 150, alpha=1.25, seed=3)
    truth = eng.ExactCounter()
    cms = eng.CountMinSketch(depth=5, width=256, seed=3)
    for x in stream:
        truth.add(x)
        cms.add(x)
    assert all(cms.estimate(k) >= c for k, c in truth.items())


def test_conservative_le_classic():
    stream = eng.make_zipf_stream(3000, 100, alpha=1.4, seed=4)
    classic = eng.CountMinSketch(depth=4, width=128, seed=4, conservative=False)
    cons = eng.CountMinSketch(depth=4, width=128, seed=4, conservative=True)
    truth = eng.ExactCounter()
    for x in stream:
        classic.add(x)
        cons.add(x)
        truth.add(x)
    assert all(cons.estimate(k) <= classic.estimate(k) for k, _ in truth.items())
    assert all(cons.estimate(k) >= c for k, c in truth.items())


def test_cms_from_eps_delta_factory():
    cms = eng.CountMinSketch.from_eps_delta(0.01, 0.01, seed=1)
    assert cms.width == eng.cms_width(0.01)
    assert cms.depth == eng.cms_depth(0.01)


def test_count_sketch_heavy_reasonable():
    stream = eng.make_zipf_stream(6000, 300, alpha=1.35, seed=8)
    truth = eng.ExactCounter()
    cs = eng.CountSketch(depth=7, width=1024, seed=8)
    for x in stream:
        truth.add(x)
        cs.add(x)
    heavies = [k for k, _ in truth.topk(8)]
    mae = sum(abs(cs.estimate(k) - truth.estimate(k)) for k in heavies) / len(heavies)
    assert mae < 80.0


def test_space_saving_guarantee():
    stream = eng.make_zipf_stream(5000, 400, alpha=1.3, seed=9)
    truth = eng.ExactCounter()
    cap = 25
    ss = eng.SpaceSaving(capacity=cap)
    for x in stream:
        truth.add(x)
        ss.add(x)
    thr = eng.spacesaving_threshold(ss.total, cap)
    missing = [k for k, c in truth.items() if c > thr and ss.estimate(k) == 0]
    assert missing == []
    assert all(ss.estimate(k) >= truth.estimate(k) for k in ss._counts)


def test_misra_gries_guarantee():
    stream = eng.make_zipf_stream(5000, 400, alpha=1.3, seed=10)
    truth = eng.ExactCounter()
    cap = 25
    mg = eng.MisraGries(capacity=cap)
    for x in stream:
        truth.add(x)
        mg.add(x)
    thr = eng.misra_gries_threshold(mg.total, cap)
    missing = [k for k, c in truth.items() if c > thr and mg.estimate(k) == 0]
    assert missing == []
    assert len(mg._counts) <= cap


def test_cms_merge():
    a = eng.CountMinSketch(depth=4, width=128, seed=2)
    b = eng.CountMinSketch(depth=4, width=128, seed=2)
    a.add_many(["x"] * 10 + ["y"] * 4)
    b.add_many(["x"] * 6 + ["z"] * 3)
    a.merge(b)
    assert a.total == 23
    assert a.estimate("x") >= 16
    assert a.estimate("y") >= 4
    assert a.estimate("z") >= 3


def test_count_sketch_merge():
    a = eng.CountSketch(depth=5, width=256, seed=1)
    b = eng.CountSketch(depth=5, width=256, seed=1)
    a.add_many(["p"] * 20)
    b.add_many(["p"] * 10 + ["q"] * 5)
    a.merge(b)
    assert a.total == 35
    assert abs(a.estimate("p") - 30) <= 8


def test_exact_topk():
    ex = eng.ExactCounter()
    ex.add_many(["a"] * 5 + ["b"] * 3 + ["c"])
    assert ex.topk(2) == [("a", 5), ("b", 3)]
    assert ex.total == 9


def test_space_saving_topk_f1():
    stream = eng.make_zipf_stream(8000, 500, alpha=1.4, seed=11)
    ev = eng.evaluate_stream("space_saving", stream, capacity=40, top_k=10, seed=11)
    assert ev["topk"]["f1"] >= 0.5


def test_cms_topk_f1_with_probe_universe():
    stream = eng.make_zipf_stream(8000, 500, alpha=1.4, seed=12)
    ev = eng.evaluate_stream("cms", stream, depth=5, width=1024, top_k=10, seed=12)
    assert ev["topk"]["f1"] >= 0.5


def test_ss_estimate_range():
    ss = eng.SpaceSaving(capacity=3)
    ss.add_many(["a"] * 10 + ["b"] * 8 + ["c"] * 6 + ["d"] * 5)
    lo, hi = ss.estimate_range("a")
    assert lo <= hi
    if ss.estimate("a"):
        assert hi == ss.estimate("a")


def test_compare_all_has_winner():
    out = eng.compare_all(
        n_events=5000, n_unique=400, alpha=1.3, width=512, capacity=40, top_k=8, seed=0
    )
    assert out["winner_f1"] in {
        "cms",
        "cms_cons",
        "count_sketch",
        "space_saving",
        "misra_gries",
    }
    assert len(out["rows"]) == 6


def test_theory_ledger_keys():
    t = eng.theory_ledger(epsilon=0.01, delta=0.01, capacity=50, n=100000)
    assert t["count_min"]["width"] == eng.cms_width(0.01)
    assert t["space_saving"]["capacity"] == 50
    assert "misra_gries" in t


def test_selftest_gate():
    row = eng.selftest()
    assert row["ok"] is True
    assert row["passed"] == row["total"]
    assert row["total"] >= 30


def test_build_sketch_dispatch():
    assert isinstance(eng.build_sketch("cms"), eng.CountMinSketch)
    assert eng.build_sketch("cms_cons").conservative is True
    assert isinstance(eng.build_sketch("count_sketch"), eng.CountSketch)
    assert isinstance(eng.build_sketch("space_saving", capacity=10), eng.SpaceSaving)
    assert isinstance(eng.build_sketch("misra_gries", capacity=10), eng.MisraGries)
    assert isinstance(eng.build_sketch("exact"), eng.ExactCounter)


def test_uniform_stream_totals():
    uni = eng.make_uniform_stream(500, 40, seed=2)
    ss = eng.SpaceSaving(capacity=15)
    ss.add_many(uni)
    assert ss.total == 500
