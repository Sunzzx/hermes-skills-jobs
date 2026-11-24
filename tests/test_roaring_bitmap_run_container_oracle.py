"""Unit tests for roaring-bitmap-run-container-oracle."""

import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/roaring-bitmap-run-container-oracle")
)
eng = SourceFileLoader("roaring_bitmap_run_container_oracle", bin_path).load_module()


def test_high_low_roundtrip():
    for x in [0, 1, 65535, 65536, 123456789, 0xFFFFFFFF]:
        assert eng.combine(*eng.high_low(x)) == (x & 0xFFFFFFFF)


def test_popcount_and_ith_bit():
    assert eng.popcount64(0) == 0
    assert eng.popcount64(0b1011) == 3
    assert eng.ith_bit_in_word(0b1010, 0) == 1
    assert eng.ith_bit_in_word(0b1010, 1) == 3


def test_array_container_sorted_unique_contains():
    ac = eng.ArrayContainer([5, 1, 3, 3, 9])
    assert ac.vals == [1, 3, 5, 9]
    assert ac.cardinality() == 4
    assert ac.contains(5) and not ac.contains(4)
    ac2 = ac.clone()
    ac2.add(2)
    assert ac.cardinality() == 4
    assert ac2.cardinality() == 5


def test_array_promotes_to_bitmap_past_threshold():
    c: eng.Container = eng.ArrayContainer()
    for i in range(eng.ARRAY_MAX + 1):
        c = c.add(i)
    assert isinstance(c, eng.BitmapContainer)
    assert c.cardinality() == eng.ARRAY_MAX + 1


def test_bitmap_iter_and_to_array():
    bm = eng.BitmapContainer()
    for i in [0, 63, 64, 1000, 65535]:
        bm.add_inplace(i)
    assert list(bm.iter_values()) == [0, 63, 64, 1000, 65535]
    arr = bm.to_array()
    assert arr.vals == [0, 63, 64, 1000, 65535]


def test_run_container_from_contiguous():
    run = eng.RunContainer.from_sorted(list(range(100, 200)))
    assert run.n_runs() == 1
    assert run.cardinality() == 100
    assert run.contains(150) and not run.contains(99)
    assert run.serialized_bytes() < eng.array_bytes(100)


def test_optimize_prefers_run_for_solid_range():
    solid = eng.optimize_container(eng.ArrayContainer(list(range(0, 3000))))
    assert isinstance(solid, eng.RunContainer)
    assert solid.serialized_bytes() < eng.array_bytes(3000)


def test_optimize_prefers_array_for_tiny_sparse():
    tiny = eng.optimize_container(eng.ArrayContainer([1, 100, 5000]))
    assert isinstance(tiny, eng.ArrayContainer)


def test_roaring_empty_and_idempotent_add():
    rb = eng.RoaringBitmap()
    assert rb.cardinality() == 0
    rb.add(42)
    rb.add(42)
    assert rb.cardinality() == 1
    assert 42 in rb and 43 not in rb


def test_roaring_cross_chunk():
    rb = eng.RoaringBitmap()
    keys = [10, eng.CHUNK + 5, 2 * eng.CHUNK + 7]
    rb.add_many(keys)
    assert rb.cardinality() == 3
    assert rb.n_containers() == 3
    assert all(rb.contains(k) for k in keys)
    rb.remove(eng.CHUNK + 5)
    assert rb.cardinality() == 2
    assert not rb.contains(eng.CHUNK + 5)


def test_union_intersection_xor_difference_equiv_set():
    a = eng.make_uniform_keys(300, universe=8000, seed=1)
    b = eng.make_uniform_keys(300, universe=8000, seed=2)
    ra, rb = eng.build_roaring(a), eng.build_roaring(b)
    assert ra.union(rb).to_set() == set(a) | set(b)
    assert ra.intersection(rb).to_set() == set(a) & set(b)
    assert ra.symmetric_difference(rb).to_set() == set(a) ^ set(b)
    assert ra.difference(rb).to_set() == set(a) - set(b)


def test_rank_select_inversion():
    keys = [2, 5, 9, eng.CHUNK + 3, eng.CHUNK + 10]
    rb = eng.build_roaring(keys)
    xs = sorted(set(keys))
    for i, x in enumerate(xs):
        assert rb.rank(x) == i + 1
        assert rb.select(i) == x
    assert rb.rank(1) == 0


def test_clustered_and_sparse_workloads():
    ck = eng.make_clustered_keys(400, seed=9)
    rc = eng.build_roaring(ck)
    assert rc.cardinality() == len(set(ck))
    assert all(rc.contains(k) for k in ck)

    sk = eng.make_sparse_keys(80, seed=3)
    rs = eng.build_roaring(sk)
    assert rs.n_containers() >= 40
    assert rs.container_kinds()["array"] >= 40


def test_dense_chunk_uses_bitmap_or_run():
    keys = list(range(0, 5000))
    rb = eng.build_roaring(keys)
    kinds = rb.container_kinds()
    assert kinds["bitmap"] + kinds["run"] >= 1
    assert rb.cardinality() == 5000


def test_range_space_beats_naive():
    keys = eng.make_range_keys(0, 15000)
    rb = eng.build_roaring(keys)
    naive = 4 * len(keys)
    assert rb.serialized_bytes() < naive / 10


def test_pure_rle_and_dense_bitset_baselines():
    pr = eng.PureRLE()
    pr.build_from(range(0, 500))
    assert len(pr.runs) == 1
    assert pr.contains(100) and not pr.contains(501)

    db = eng.DenseBitset(2048)
    db.add_many([1, 2, 100])
    assert db.cardinality() == 3
    assert db.contains(100) and not db.contains(99)


def test_theory_ledger_shape():
    th = eng.theory_ledger(card=4096, n_runs=1, density=0.05)
    assert th["array_max_threshold"] == eng.ARRAY_MAX
    assert th["single_chunk"]["prefer"] in ("array", "run", "bitmap")
    assert "multi_chunk_model" in th


def test_clone_independence():
    a = eng.build_roaring([1, 2, 3])
    b = a.clone()
    b.add(99)
    assert a.cardinality() == 3 and b.cardinality() == 4


def test_selftest_passes():
    assert eng.selftest(verbose=False) == 0


def test_cli_status_and_build():
    assert eng.main(["status"]) == 0
    assert eng.main(["build", "--workload", "range", "--n", "500", "--seed", "1"]) == 0
    assert eng.main(["op", "--op", "and", "--n", "200", "--seed", "2"]) == 0
    assert eng.main(["rank", "--n", "300", "--seed", "3"]) == 0
