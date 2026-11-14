"""Unit tests for radix-attention-cache: Radix Tree KV Cache Sharing Engine."""

import os
import sys
from importlib.machinery import SourceFileLoader
import pytest

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
    "radix-attention-cache",
)
mod = SourceFileLoader("radix_attention_cache", MOD_PATH).load_module()

RadixAttentionCache = mod.RadixAttentionCache
tokenize_synthetic = mod.tokenize_synthetic
run_benchmark = mod.run_benchmark


def test_basic_insertion_and_lookup():
    cache = RadixAttentionCache(page_size=4, max_pages=64)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Miss before insert
    miss_res = cache.lookup(tokens)
    assert miss_res["matched_prefix_tokens"] == 0
    assert miss_res["cache_hit_ratio"] == 0.0

    # Insert
    inserted = cache.insert(tokens, value="doc1")
    assert inserted == len(tokens)

    # Full hit
    hit_res = cache.lookup(tokens)
    assert hit_res["matched_prefix_tokens"] == len(tokens)
    assert hit_res["cache_hit_ratio"] == 1.0
    assert hit_res["new_tokens_to_compute"] == 0


def test_prefix_sharing_and_branch_split():
    cache = RadixAttentionCache(page_size=4, max_pages=64)
    prefix = [10, 20, 30, 40]
    branch_a = prefix + [50, 60]
    branch_b = prefix + [70, 80]

    cache.insert(branch_a, value="A")
    
    # Branch B should hit the common prefix
    lookup_b = cache.lookup(branch_b)
    assert lookup_b["matched_prefix_tokens"] == len(prefix)
    assert lookup_b["new_tokens_to_compute"] == len(branch_b) - len(prefix)

    # Now insert branch B, which causes an edge split
    cache.insert(branch_b, value="B")

    # Both branches should now be retrievable in full
    assert cache.lookup(branch_a)["matched_prefix_tokens"] == len(branch_a)
    assert cache.lookup(branch_b)["matched_prefix_tokens"] == len(branch_b)


def test_lru_eviction():
    # Small memory budget: 4 pages max
    cache = RadixAttentionCache(page_size=2, max_pages=4)
    seq1 = [1, 2, 3, 4]  # 2 pages
    seq2 = [5, 6, 7, 8]  # 2 pages
    seq3 = [9, 10, 11, 12]  # 2 pages -> forces eviction

    cache.insert(seq1, value="seq1")
    cache.insert(seq2, value="seq2")
    assert cache.total_pages_used <= 4

    # Access seq2 to refresh its LRU timestamp
    cache.lookup(seq2)

    # Inserting seq3 will evict seq1
    cache.insert(seq3, value="seq3")
    assert cache.total_pages_used <= 4
    assert cache.evicted_pages > 0

    # seq2 should still match, seq1 should have been evicted
    assert cache.lookup(seq2)["matched_prefix_tokens"] == len(seq2)
    assert cache.lookup(seq1)["matched_prefix_tokens"] < len(seq1)


def test_benchmark_run():
    bench = run_benchmark(page_size=8, max_pages=128)
    assert bench["system_tokens_count"] > 0
    assert bench["turn1_lookup"]["cache_hit_ratio"] > 0.0
    assert bench["turn2_lookup"]["cache_hit_ratio"] > bench["turn1_lookup"]["cache_hit_ratio"]
    assert bench["final_cache_stats"]["hits"] == 3
    assert bench["final_cache_stats"]["misses"] == 0
