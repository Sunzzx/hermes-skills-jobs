"""Unit tests for infllm-block-memory-engine: InfLLM training-free block memory."""

import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/infllm-block-memory-engine")
)
eng = SourceFileLoader("infllm_block_memory_engine", bin_path).load_module()


def test_corpus_has_sinks_local_needles():
    corpus = eng.generate_long_context(
        seq_len=512, dim=24, n_needles=6, sink_len=4, local_window=48, block_size=32, seed=1
    )
    assert corpus.n == 512
    assert corpus.labels[:4] == ["sink"] * 4
    assert corpus.labels[-48:] == ["local"] * 48
    assert len(corpus.needle_positions) == 6
    assert all(corpus.labels[p] == "needle" for p in corpus.needle_positions)
    # needles not in sink/local
    for p in corpus.needle_positions:
        assert 4 <= p < 512 - 48


def test_partition_covers_middle_exactly():
    corpus = eng.generate_long_context(seq_len=400, dim=16, block_size=25, seed=2)
    blocks = eng.partition_blocks(corpus)
    covered = []
    for b in blocks:
        covered.extend(range(b.start, b.end))
    expected = list(range(corpus.sink_len, corpus.n - corpus.local_window))
    assert covered == expected
    assert all(b.length <= corpus.block_size for b in blocks)


def test_representatives_selected():
    corpus = eng.generate_long_context(seq_len=320, dim=20, block_size=32, seed=3)
    blocks = eng.build_block_memory(corpus, n_rep=4, seed=0)
    assert blocks
    for b in blocks:
        assert 1 <= len(b.rep_indices_local) <= min(4, b.length)
        assert len(b.rep_keys) == len(b.rep_indices_local)
        assert len(b.centroid_key) == corpus.dim
        assert len(set(b.rep_indices_local)) == len(b.rep_indices_local)


def test_infllm_keeps_sinks_and_local():
    corpus = eng.generate_long_context(seq_len=480, dim=20, block_size=32, seed=4)
    blocks = eng.build_block_memory(corpus, n_rep=3, seed=0)
    queries = eng.make_queries(corpus, n_queries=6, seed=1)
    ws = eng.working_set_infllm(corpus, blocks, queries, top_k=3)
    for i in range(corpus.sink_len):
        assert i in ws.keep_set
    for i in range(corpus.n - corpus.local_window, corpus.n):
        assert i in ws.keep_set
    assert ws.n_blocks_retrieved == min(3, len(blocks))
    assert ws.n_gpu_tokens == len(ws.keep_idx)


def test_dense_zero_mse_full_needles():
    corpus = eng.generate_long_context(seq_len=256, dim=16, n_needles=4, seed=5)
    queries = eng.make_queries(corpus, n_queries=4, seed=2)
    rep = eng.evaluate_working_set(corpus, eng.working_set_dense(corpus), queries)
    assert rep.output_mse < 1e-8
    assert abs(rep.output_cosine - 1.0) < 1e-6
    assert rep.needle_recall == 1.0


def test_infllm_beats_streaming_on_needles():
    corpus = eng.generate_long_context(
        seq_len=640, dim=28, n_needles=6, sink_len=4, local_window=48, block_size=32, seed=11
    )
    blocks = eng.build_block_memory(corpus, n_rep=4, seed=0)
    queries = eng.make_queries(corpus, n_queries=10, seed=7)
    inf = eng.evaluate_working_set(
        corpus, eng.working_set_infllm(corpus, blocks, queries, top_k=4), queries, blocks
    )
    sllm = eng.evaluate_working_set(
        corpus, eng.working_set_streaming_llm(corpus), queries, blocks
    )
    assert inf.needle_recall >= sllm.needle_recall - 1e-9
    # StreamingLLM cannot see middle needles
    assert sllm.needle_recall == 0.0 or inf.needle_recall >= sllm.needle_recall


def test_benchmark_schema_and_policies():
    out = eng.run_benchmark(seq_len=384, dim=20, n_needles=4, block_size=24, top_k=3, seed=9)
    assert out["n_blocks"] > 0
    assert out["winner"] in {r["policy"] for r in out["reports"]}
    policies = {r["policy"] for r in out["reports"]}
    assert {
        "infllm",
        "streaming_llm",
        "random_blocks",
        "h2o_approx",
        "dense",
        "sliding_window",
    } <= policies
    assert 0.0 <= out["infllm_needle_recall"] <= 1.0


def test_compare_lift_keys():
    out = eng.run_compare(seq_len=512, top_k=4, block_size=32, seed=4)
    assert "needle_lift_vs_streaming" in out
    assert "block_hit_rate" in out
    assert "compression_ratio" in out
    assert out["compression_ratio"] >= 1.0


def test_theory_gpu_smaller_than_dense():
    t = eng.theory_ledger(seq_len=128000, top_k=8, local_window=4096)
    assert t["infllm_gpu_kv_gib"] < t["dense_kv_gib"]
    assert t["gpu_compression_vs_dense"] > 1.0
    assert t["n_blocks"] > 0


def test_selftest_exit_zero():
    assert eng.cmd_selftest(None) == 0


def test_score_modes_finite():
    corpus = eng.generate_long_context(seq_len=256, dim=16, block_size=32, seed=8)
    blocks = eng.build_block_memory(corpus, n_rep=3, seed=0)
    q = eng.make_queries(corpus, n_queries=1, seed=0)[0]
    for mode in ("rep_mean", "rep_max", "centroid"):
        for b in blocks:
            s = eng.score_block(q, b, mode=mode)
            assert s == s  # not NaN
            assert abs(s) < 1e6
