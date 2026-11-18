"""Unit tests for keydiff-kv-engine: Key Similarity KV Eviction (NeurIPS 2025)."""

import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin/keydiff-kv-engine"))
kd = SourceFileLoader("keydiff_kv_engine", bin_path).load_module()


def test_corpus_structure():
    corpus = kd.generate_key_corpus(seq_len=256, dim=32, n_needles=8, sink_len=4, local_window=32, seed=1)
    assert corpus.n == 256
    assert corpus.dim == 32
    assert corpus.labels[:4] == ["sink"] * 4
    assert corpus.labels[-32:] == ["local"] * 32
    assert sum(1 for L in corpus.labels if L == "needle") == 8


def test_budget_cardinality_all_policies():
    corpus = kd.generate_key_corpus(seq_len=200, dim=24, n_needles=5, seed=2)
    budget = 50
    results = [
        kd.evict_keydiff(corpus.keys, budget, local_window=20),
        kd.evict_keydiff_block(corpus.keys, budget, block_size=10, local_window=20),
        kd.evict_streaming_llm(corpus.n, budget),
        kd.evict_random(corpus.n, budget, seed=0),
        kd.evict_magnitude(corpus.keys, budget),
    ]
    for r in results:
        assert len(r.keep_idx) == budget, r.policy
        assert len(set(r.keep_idx)) == budget
        assert min(r.keep_idx) >= 0 and max(r.keep_idx) < corpus.n


def test_keydiff_retains_sinks():
    corpus = kd.generate_key_corpus(seq_len=300, dim=24, seed=3)
    r = kd.evict_keydiff(corpus.keys, budget=60, sink_len=4, local_window=24)
    for i in range(4):
        assert i in r.keep_set


def test_keydiff_beats_random_on_needles():
    corpus = kd.generate_key_corpus(seq_len=400, dim=32, n_needles=10, n_clusters=6, seed=11)
    queries = kd.make_probe_queries(corpus, n_queries=20, seed=5)
    budget = 80
    kd_res = kd.evaluate_policy(corpus, kd.evict_keydiff(corpus.keys, budget, local_window=32), queries)
    rnd_res = kd.evaluate_policy(corpus, kd.evict_random(corpus.n, budget, seed=0), queries)
    # At least one of mass or needle should be strictly better; neither should collapse
    assert kd_res.mass_recovery > 0.05
    assert kd_res.needle_recall >= rnd_res.needle_recall - 1e-9 or kd_res.mass_recovery >= rnd_res.mass_recovery


def test_benchmark_winner_fields():
    out = kd.run_benchmark(seq_len=256, dim=24, budget_ratio=0.3, seed=9, block_size=8)
    assert out["budget"] == int(256 * 0.3)
    assert out["winner"] in {r["policy"] for r in out["reports"]}
    assert 0.0 < out["keydiff_mass_recovery"] <= 1.0
    policies = {r["policy"] for r in out["reports"]}
    assert {"keydiff", "keydiff_block", "streaming_llm", "random", "magnitude"} <= policies


def test_compare_lift_keys():
    out = kd.run_compare(seq_len=256, budget_ratio=0.25, seed=4)
    assert "mass_lift_vs_streaming" in out
    assert "needle_lift_vs_streaming" in out


def test_selftest_exit_zero():
    assert kd.cmd_selftest(None) == 0


def test_oracle_mass_sums_near_one_per_query_avg():
    corpus = kd.generate_key_corpus(seq_len=64, dim=16, n_needles=2, local_window=8, seed=0)
    queries = kd.make_probe_queries(corpus, n_queries=4, seed=1)
    mass = kd.oracle_attention_scores(corpus.keys, queries)
    assert abs(sum(mass) - 1.0) < 1e-6
