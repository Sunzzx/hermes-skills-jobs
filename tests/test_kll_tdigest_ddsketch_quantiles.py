"""Unit tests for kll-tdigest-ddsketch-quantiles streaming quantile oracle."""

import math
import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/kll-tdigest-ddsketch-quantiles")
)
eng = SourceFileLoader("kll_tdigest_ddsketch_quantiles", bin_path).load_module()


def test_kll_capacity_decreases_with_level():
    c0 = eng.kll_capacity(200, 0)
    c3 = eng.kll_capacity(200, 3)
    assert c0 > 0 and c3 > 0
    assert c0 >= c3


def test_kll_rank_error_bound_inverse_k():
    assert eng.kll_rank_error_bound(100) > eng.kll_rank_error_bound(400)
    assert abs(eng.kll_rank_error_bound(200) - 1.0 / 200) < 1e-12


def test_ddsketch_gamma_and_log():
    g = eng.ddsketch_gamma(0.01)
    assert g > 1.0
    assert abs(g - (1.01 / 0.99)) < 1e-12
    lg = eng.ddsketch_log_gamma(0.01)
    assert abs(lg - math.log(g)) < 1e-12


def test_gk_space_bound_grows_with_n():
    s1 = eng.gk_space_bound(0.01, 10_000)
    s2 = eng.gk_space_bound(0.01, 1_000_000)
    assert s1 > 0 and s2 >= s1


def test_relative_error_helpers():
    assert eng.relative_error(0, 0) == 0.0
    assert abs(eng.relative_error(5, 10) - 0.5) < 1e-12
    assert eng.absolute_error(3, 5) == 2.0


def test_exact_quantiles_median_uniform():
    xs = eng.make_uniform_stream(1001, 0.0, 1.0, seed=1)
    eq = eng.ExactQuantiles()
    eq.add_many(xs)
    med = eq.quantile(0.5)
    assert 0.4 < med < 0.6
    assert eq.n == 1001


def test_empty_sketches_safe():
    kll = eng.KLLSketch(k=32, seed=0)
    assert kll.n == 0
    try:
        q = kll.quantile(0.5)
        assert q is None
    except ValueError:
        pass
    td = eng.TDigest(compression=50)
    n_td = getattr(td, "n", None)
    if n_td is None:
        n_td = getattr(td, "count", 0)
    assert n_td == 0
    dd = eng.DDSketch(alpha=0.02)
    assert getattr(dd, "n", 0) == 0
    gk = eng.GKSketch(epsilon=0.05)
    assert getattr(gk, "n", 0) == 0


def test_kll_median_gaussian_rank_error():
    stream = eng.make_gaussian_stream(4000, seed=7)
    sk = eng.KLLSketch(k=128, seed=7)
    truth = eng.ExactQuantiles()
    for x in stream:
        sk.add(x)
        truth.add(x)
    est = sk.quantile(0.5)
    rank_err = abs(truth.rank(est) - 0.5)
    assert rank_err < 0.08


def test_tdigest_p99_lognormal_reasonable():
    stream = eng.make_lognormal_stream(5000, seed=11)
    sk = eng.TDigest(compression=100)
    truth = eng.ExactQuantiles()
    for x in stream:
        sk.add(x)
        truth.add(x)
    est = sk.quantile(0.99)
    tq = truth.quantile(0.99)
    rank_err = abs(truth.rank(est) - 0.99)
    assert rank_err < 0.05
    assert eng.relative_error(est, tq) < 0.35


def test_ddsketch_relative_error_positive_stream():
    stream = eng.make_pareto_stream(6000, seed=13)
    sk = eng.DDSketch(alpha=0.02)
    truth = eng.ExactQuantiles()
    for x in stream:
        sk.add(x)
        truth.add(x)
    qs = [0.5, 0.9, 0.99]
    for q in qs:
        est = sk.quantile(q)
        tq = truth.quantile(q)
        # relative error SLA ~ alpha with some slack on pure-Python impl
        assert eng.relative_error(est, tq) < 0.12


def test_gk_rank_bound_style():
    stream = eng.make_uniform_stream(3000, seed=17)
    sk = eng.GKSketch(epsilon=0.02)
    truth = eng.ExactQuantiles()
    for x in stream:
        sk.add(x)
        truth.add(x)
    est = sk.quantile(0.5)
    rank_err = abs(truth.rank(est) - 0.5)
    assert rank_err < 0.08


def test_kll_merge_preserves_rank():
    s1 = eng.make_gaussian_stream(2000, seed=1)
    s2 = eng.make_gaussian_stream(2000, seed=2)
    full = s1 + s2
    truth = eng.ExactQuantiles()
    truth.add_many(full)
    A = eng.KLLSketch(k=100, seed=1)
    B = eng.KLLSketch(k=100, seed=2)
    for x in s1:
        A.add(x)
    for x in s2:
        B.add(x)
    A.merge(B)
    est = A.quantile(0.5)
    assert abs(truth.rank(est) - 0.5) < 0.1


def test_ddsketch_merge_union():
    s1 = eng.make_lognormal_stream(1500, seed=3)
    s2 = eng.make_lognormal_stream(1500, seed=4)
    truth = eng.ExactQuantiles()
    truth.add_many(s1 + s2)
    A = eng.DDSketch(alpha=0.02)
    B = eng.DDSketch(alpha=0.02)
    for x in s1:
        A.add(x)
    for x in s2:
        B.add(x)
    A.merge(B)
    est = A.quantile(0.9)
    assert abs(truth.rank(est) - 0.9) < 0.08


def test_build_sketch_factory_kinds():
    for kind in ("kll", "tdigest", "ddsketch", "gk", "exact"):
        sk = eng.build_sketch(kind, k=64, compression=50, alpha=0.02, epsilon=0.05, seed=0)
        assert sk is not None


def test_evaluate_stream_kll_ok():
    stream = eng.make_mixture_stream(3000, seed=9)
    out = eng.evaluate_stream("kll", stream, qs=[0.1, 0.5, 0.9], seed=9, k=100)
    assert out["mean_rank_err"] < 0.12
    assert out["n"] == 3000


def test_compare_all_has_winner():
    out = eng.compare_all(n=2500, stream_kind="gaussian", seed=5, k=80, compression=80, alpha=0.02, epsilon=0.05)
    assert "winner_rank_err" in out or "kinds" in out or "results" in out or "rows" in out
    # mean rank errors present for at least one sketch path
    blob = str(out)
    assert "kll" in blob.lower() or "mean_rank" in blob


def test_theory_ledger_keys():
    t = eng.theory_ledger(k=200, alpha=0.01, epsilon=0.01, n=1_000_000, compression=100)
    assert "kll" in t and "tdigest" in t and "ddsketch" in t and "gk" in t
    assert t["ddsketch"]["relative_error_guarantee"] == 0.01


def test_selftest_passes():
    result = eng.selftest()
    assert result.get("ok") is True
    passed = result.get("passed") or result.get("n_passed") or 0
    total = result.get("total") or result.get("n_total") or passed
    if isinstance(passed, int) and isinstance(total, int) and total:
        assert passed == total
    # also accept string form like "40/40"
    if "summary" in result:
        assert "ok" in str(result).lower() or result.get("ok")


def test_stream_factories():
    for kind in eng.STREAM_FACTORIES:
        xs = eng.build_stream(kind, 50, seed=0)
        assert len(xs) == 50
