"""Unit tests for curdkv-value-cur-engine: Value-Guided CUR KV Compression (NeurIPS 2025)."""

import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin/curdkv-value-cur-engine"))
ck = SourceFileLoader("curdkv_value_cur_engine", bin_path).load_module()


def test_corpus_structure():
    corpus = ck.generate_kv_corpus(
        seq_len=256, dim=32, n_needles=8, sink_len=4, local_window=32, seed=1
    )
    assert corpus.n == 256
    assert corpus.dim == 32
    assert corpus.labels[:4] == ["sink"] * 4
    assert corpus.labels[-32:] == ["local"] * 32
    assert sum(1 for L in corpus.labels if L == "needle") == 8
    assert sum(1 for L in corpus.labels if L == "decoy") >= 4


def test_budget_cardinality_all_policies():
    corpus = ck.generate_kv_corpus(seq_len=200, dim=24, n_needles=5, seed=2)
    obs = ck.make_probe_queries(corpus, n_queries=8, seed=0)
    budget = 50
    results = [
        ck.compress_curdkv(corpus.keys, corpus.values, obs, budget, rank=4, seed=0),
        ck.compress_curdkv(corpus.keys, corpus.values, obs, budget, rank=4, seed=0, mode="value"),
        ck.compress_curdkv_hybrid(corpus.keys, corpus.values, obs, budget, rank=4, seed=0),
        ck.compress_attn_score(corpus.keys, obs, budget),
        ck.compress_value_magnitude(corpus.values, budget),
        ck.compress_streaming_llm(corpus.n, budget),
        ck.compress_random(corpus.n, budget, seed=0),
    ]
    for r in results:
        assert len(r.keep_idx) == budget, r.policy
        assert len(set(r.keep_idx)) == budget
        assert min(r.keep_idx) >= 0 and max(r.keep_idx) < corpus.n


def test_curdkv_retains_sinks():
    corpus = ck.generate_kv_corpus(seq_len=300, dim=24, seed=3)
    obs = ck.make_probe_queries(corpus, n_queries=6, seed=1)
    r = ck.compress_curdkv(
        corpus.keys, corpus.values, obs, budget=60, sink_len=4, local_window=24, seed=0
    )
    for i in range(4):
        assert i in r.keep_set


def test_leverage_sum_near_rank():
    corpus = ck.generate_kv_corpus(seq_len=128, dim=20, seed=4)
    obs = ck.make_probe_queries(corpus, n_queries=6, seed=2)
    cols = ck._contribution_columns(corpus.keys, corpus.values, obs)
    rank = 5
    lev = ck.column_leverage_scores(cols, rank=rank, n_iter=10, seed=0)
    assert len(lev) == corpus.n
    assert all(x >= -1e-9 for x in lev)
    assert abs(sum(lev) - rank) < 0.8


def test_full_cache_zero_mse():
    corpus = ck.generate_kv_corpus(seq_len=128, dim=16, n_needles=4, local_window=24, seed=0)
    queries = ck.make_probe_queries(corpus, n_queries=4, seed=1)
    full = ck.CompressResult(list(range(corpus.n)), "full", corpus.n, 0.0)
    rep = ck.evaluate_policy(corpus, full, queries)
    assert rep.output_mse < 1e-8
    assert abs(rep.output_cosine - 1.0) < 1e-6


def test_curdkv_beats_or_matches_random_on_output():
    corpus = ck.generate_kv_corpus(seq_len=280, dim=28, n_needles=8, seed=11)
    obs = ck.make_probe_queries(corpus, n_queries=10, seed=5)
    queries = ck.make_probe_queries(corpus, n_queries=12, seed=6)
    budget = 56
    cur = ck.evaluate_policy(
        corpus,
        ck.compress_curdkv(corpus.keys, corpus.values, obs, budget, rank=5, seed=0),
        queries,
    )
    rnd = ck.evaluate_policy(corpus, ck.compress_random(corpus.n, budget, seed=0), queries)
    # CurDKV should not collapse; beat random on mse or cosine or needle
    assert cur.output_mse >= 0.0
    assert (
        cur.output_mse <= rnd.output_mse * 1.05
        or cur.output_cosine >= rnd.output_cosine - 1e-6
        or cur.needle_recall >= rnd.needle_recall - 1e-9
    )


def test_benchmark_winner_fields():
    out = ck.run_benchmark(seq_len=160, dim=24, budget_ratio=0.3, seed=9, rank=4)
    assert out["budget"] == int(160 * 0.3)
    assert out["winner"] in {r["policy"] for r in out["reports"]}
    assert out["curdkv_output_mse"] >= 0.0
    policies = {r["policy"] for r in out["reports"]}
    assert {"curdkv", "attn_score", "streaming_llm", "random", "value_magnitude"} <= policies


def test_compare_lift_keys():
    out = ck.run_compare(seq_len=180, budget_ratio=0.25, seed=4)
    assert "mse_lift_vs_attn" in out
    assert "needle_lift_vs_attn" in out
    assert "decoy_avoidance_lift" in out


def test_selftest_exit_zero():
    assert ck.cmd_selftest(None) == 0


def test_theory_ledger_monotonic():
    t = ck.theory_ledger(seq_len=4096, dim=64, layers=16, heads=16)
    mibs = [b["kv_mib"] for b in t["budgets"]]
    assert mibs == sorted(mibs, reverse=True)
    assert t["budgets"][0]["budget_ratio"] == 1.0
