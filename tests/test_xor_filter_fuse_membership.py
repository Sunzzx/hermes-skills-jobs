"""Unit tests for xor-filter-fuse-membership: XOR + Binary Fuse filters."""

import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/xor-filter-fuse-membership")
)
eng = SourceFileLoader("xor_filter_fuse_membership", bin_path).load_module()


def test_theory_bloom_optimal_bpk():
    th = eng.theory_ledger(n=10_000, fpr=0.01)
    assert 9.0 < th["bloom"]["bits_per_key_optimal"] < 10.5
    assert abs(th["xor8"]["bits_per_key"] - 9.84) < 0.2
    assert th["fuse8"]["bits_per_key"] < th["xor8"]["bits_per_key"]


def test_fnv_and_splitmix_stable():
    assert eng.fnv1a64("hermes", 0) == eng.fnv1a64("hermes", 0)
    assert eng.splitmix64(123) != eng.splitmix64(124)
    assert eng.fingerprint8(0xABC) == (0xABC & 0xFF)


def test_empty_xor_rejects():
    xf = eng.XorFilter([], fingerprint_bits=8)
    assert xf.might_contain("anything") is False
    assert xf.stats.success is True


def test_xor8_perfect_recall_and_fpr_band():
    keys = eng.make_keys(120, seed=3)
    xf = eng.XorFilter(keys, fingerprint_bits=8, seed=11)
    assert xf.stats.success
    assert eng.measure_recall(xf, keys) == 1.0
    neg = eng.make_negatives(3000, seed=19)
    fpr = eng.measure_fpr(xf, neg)
    assert 0.0 <= fpr <= 0.03
    assert 8.0 <= xf.bits_per_key() <= 12.0


def test_xor16_tighter_than_xor8():
    keys = eng.make_keys(100, seed=4)
    neg = eng.make_negatives(5000, seed=20)
    x8 = eng.XorFilter(keys, fingerprint_bits=8, seed=2)
    x16 = eng.XorFilter(keys, fingerprint_bits=16, seed=2)
    assert eng.measure_recall(x16, keys) == 1.0
    assert eng.measure_fpr(x16, neg) <= eng.measure_fpr(x8, neg) + 0.01


def test_binary_fuse8_recall():
    keys = eng.make_keys(200, seed=5)
    bf = eng.BinaryFuseFilter(keys, fingerprint_bits=8, seed=8)
    assert bf.stats.success
    assert eng.measure_recall(bf, keys) == 1.0
    fpr = eng.measure_fpr(bf, eng.make_negatives(2500, seed=21))
    assert fpr <= 0.03


def test_binary_fuse16_recall():
    keys = eng.make_keys(80, seed=6)
    bf = eng.BinaryFuseFilter(keys, fingerprint_bits=16, seed=9)
    assert eng.measure_recall(bf, keys) == 1.0


def test_bloom_recall_and_theory_fpr():
    keys = eng.make_keys(150, seed=7)
    bloom = eng.BloomFilter(n=len(keys), fpr=0.01, seed=1)
    bloom.add_many(keys)
    assert eng.measure_recall(bloom, keys) == 1.0
    assert bloom.theory_fpr() < 0.05
    assert bloom.as_dict()["kind"] == "bloom"


def test_blocked_bloom_recall():
    keys = eng.make_keys(150, seed=8)
    bb = eng.BlockedBloomFilter(n=len(keys), fpr=0.01, seed=1)
    bb.add_many(keys)
    assert eng.measure_recall(bb, keys) == 1.0
    assert bb.as_dict()["kind"] == "blocked-bloom"


def test_build_filter_dispatch():
    keys = eng.make_keys(40, seed=9)
    for kind in ("bloom", "blocked-bloom", "xor8", "xor16", "fuse8", "fuse16"):
        f = eng.build_filter(kind, keys, seed=3)
        assert eng.measure_recall(f, keys) == 1.0


def test_evaluate_ok():
    out = eng.evaluate("xor8", n=300, neg=1500, seed=12)
    assert out["ok"] is True
    assert out["recall"] == 1.0
    assert out["empirical_fpr"] < 0.03


def test_compare_all_ranks():
    out = eng.compare_all(n=500, neg=2000, seed=14)
    assert out["winner_bits"] is not None
    assert len(out["ranking"]) >= 4
    assert all(r.get("ok") for r in out["rows"] if "error" not in r)


def test_xor16_beats_bloom_space_at_low_fpr():
    n = 600
    keys = eng.make_keys(n, seed=30)
    bloom = eng.BloomFilter(n=n, fpr=1 / 65536, seed=2)
    bloom.add_many(keys)
    xor16 = eng.XorFilter(keys, fingerprint_bits=16, seed=2)
    assert xor16.bits_per_key() < bloom.bits_per_key()


def test_large_peel_xor_and_fuse():
    keys = eng.make_keys(1200, seed=40)
    xf = eng.XorFilter(keys, fingerprint_bits=8, seed=6, max_attempts=48)
    assert xf.stats.success
    assert eng.measure_recall(xf, keys) == 1.0
    bf = eng.BinaryFuseFilter(keys, fingerprint_bits=8, seed=6, max_attempts=48)
    assert bf.stats.success
    assert eng.measure_recall(bf, keys) == 1.0


def test_selftest_gate():
    rc = eng.selftest(verbose=False)
    assert rc == 0


def test_main_theory_cli():
    rc = eng.main(["theory", "--n", "1000", "--fpr", "0.01"])
    assert rc == 0


def test_main_status_cli():
    rc = eng.main(["status"])
    assert rc == 0
