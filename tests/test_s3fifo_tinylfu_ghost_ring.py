"""Unit tests for s3fifo-tinylfu-ghost-ring cache eviction oracle."""

import os
from collections import Counter
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/s3fifo-tinylfu-ghost-ring")
)
eng = SourceFileLoader("s3fifo_tinylfu_ghost_ring", bin_path).load_module()


def test_fnv_deterministic():
    assert eng._fnv1a64("x", 1) == eng._fnv1a64("x", 1)
    assert eng._fnv1a64("x", 1) != eng._fnv1a64("y", 1)


def test_count_min_sketch_ordering():
    cms = eng.CountMinSketch(width=256, depth=4, age_after=100000)
    for _ in range(100):
        cms.add("hot")
    for _ in range(2):
        cms.add("cold")
    assert cms.estimate("hot") >= 80
    assert cms.estimate("hot") > cms.estimate("cold")
    assert cms.estimate("absent") == 0


def test_cms_aging():
    cms = eng.CountMinSketch(width=64, depth=3, age_after=10)
    for _ in range(25):
        cms.add("a")
    assert cms.ages >= 2


def test_bloom_doorkeeper():
    b = eng.BloomDoorkeeper(n_bits=1024, n_hashes=4)
    assert b.add("k") is True
    assert b.might_contain("k") is True
    assert b.add("k") is False


def test_lru_eviction_order():
    c = eng.LRUCache(3)
    for k in (1, 2, 3):
        c.put(k)
    c.get(1)
    c.put(4)
    assert c.get(2) is False
    assert c.get(1) is True
    assert c.size() == 3


def test_fifo_ignores_recency():
    c = eng.FIFOCache(2)
    c.put("a")
    c.put("b")
    c.get("a")
    c.put("c")
    assert c.get("a") is False
    assert c.get("b") is True


def test_clock_prefers_referenced():
    c = eng.ClockCache(4)
    for k in (1, 2, 3, 4):
        c.put(k)
    for i in range(4):
        c._ref[i] = 0
    c._hand = 0
    c.get(1)
    c.put(5)
    assert c.get(1) is True
    assert c.get(5) is True


def test_s3fifo_scan_resistance():
    c = eng.S3FIFOCache(20, small_ratio=0.1)
    ws = list(range(10))
    for _ in range(5):
        for k in ws:
            c.access(k)
    for k in range(100, 200):
        c.access(k)
    retained = sum(1 for k in ws if c.get(k))
    assert retained >= 5


def test_s3fifo_ghost_readmit():
    c = eng.S3FIFOCache(8, small_ratio=0.25)
    for k in range(30):
        c.access(k)
    assert len(c.ghost) > 0
    gk = next(iter(c.ghost.keys()))
    before = c.stats.ghost_hits
    c.access(gk)
    assert c.stats.ghost_hits == before + 1
    assert gk in c.main or gk in c.small


def test_wtinylfu_keeps_hot():
    c = eng.WTinyLFUCache(32, window_ratio=0.1, use_doorkeeper=False)
    for _ in range(40):
        c.access(0)
    for k in range(1, 100):
        c.access(k)
    assert c.get(0) is True


def test_hybrid_capacity_and_zipf():
    c = eng.S3FIFOTinyLFUHybrid(64)
    keys = eng.generate_workload("zipf", 400, 4000, seed=9, alpha=1.25)
    res = eng.run_workload(c, keys)
    assert 0.0 <= res["hit_rate"] <= 1.0
    assert c.size() <= 64
    inv = c.inventory()
    assert inv["small"] + inv["main"] == c.size()


def test_arc_capacity():
    c = eng.ARCCache(16)
    for k in range(50):
        c.access(k)
    assert c.size() <= 16


def test_zipf_concentration():
    z = eng.sample_zipf(100, 1.5, 4000, __import__("random").Random(2))
    top10 = sum(v for _, v in Counter(z).most_common(10))
    assert top10 > 4000 * 0.4


def test_workload_kinds():
    for kind in ("zipf", "ir", "scan", "loop", "hotspot", "mixed"):
        keys = eng.generate_workload(kind, 50, 200, seed=1)
        assert len(keys) == 200
        assert all(0 <= k < 50 or kind == "loop" for k in keys) or kind == "loop"


def test_compare_policies_ranking():
    res = eng.compare_policies(capacity=48, n_keys=400, n_ops=4000, workload="zipf", seed=4)
    assert res["winner"] in eng.POLICY_CTORS
    assert len(res["ranking"]) == len(eng.POLICY_CTORS)
    rates = [r["hit_rate"] for r in res["ranking"]]
    assert rates == sorted(rates, reverse=True)


def test_scan_s3fifo_competitive_with_lru():
    keys = eng.generate_workload("scan", 400, 6000, seed=12, alpha=1.1)
    lru = eng.run_workload(eng.LRUCache(40), keys)
    s3 = eng.run_workload(eng.S3FIFOCache(40), list(keys))
    assert s3["hit_rate"] + 1e-9 >= lru["hit_rate"] - 0.08


def test_theory_ledger_keys():
    th = eng.theory_ledger(128)
    for k in ("s3fifo", "w_tinylfu", "hybrid_s3fifo_tinylfu", "arc", "cms_error"):
        assert k in th


def test_capacity_rejected():
    try:
        eng.LRUCache(1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_byte_hit_rate():
    c = eng.LRUCache(4)
    c.access("a", 10)
    c.access("a", 10)
    assert abs(c.stats.byte_hit_rate() - 0.5) < 1e-9


def test_selftest_passes():
    summary = eng.run_selftest(verbose=False)
    assert summary["ok"] is True
    assert summary["checks"] >= 18


def test_make_policy_unknown():
    try:
        eng.make_policy("nope", 8)
        assert False
    except ValueError:
        pass


def test_hybrid_hot_retention_after_cold_flood():
    c = eng.S3FIFOTinyLFUHybrid(30, small_ratio=0.2)
    for _ in range(25):
        for k in range(10):
            c.access(k)
    for k in range(1000, 1150):
        c.access(k)
    retained = sum(1 for k in range(10) if c.get(k))
    assert retained >= 3
