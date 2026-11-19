"""Unit tests for hotstuff-chained-bft: HotStuff Chained BFT consensus engine."""

import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/hotstuff-chained-bft")
)
eng = SourceFileLoader("hotstuff_chained_bft", bin_path).load_module()


def test_max_faults_and_quorum():
    assert eng.max_faults(4) == 1
    assert eng.max_faults(7) == 2
    assert eng.quorum_size(4) == 3
    assert eng.quorum_size(7) == 5
    assert 3 * eng.max_faults(4) + 1 == 4
    assert 3 * eng.max_faults(7) + 1 == 7


def test_digest_deterministic():
    assert eng.digest({"a": 1},) == eng.digest({"a": 1})
    assert eng.digest({"a": 1}) != eng.digest({"a": 2})


def test_kv_state_machine():
    sm = eng.KVStateMachine()
    sm.apply(1, {"op": "SET", "key": "a", "val": 10})
    assert sm.apply(2, {"op": "INCR", "key": "a", "val": 2}) == 12
    assert sm.apply(3, {"op": "CAS", "key": "a", "expected": 12, "val": 0}) is True
    assert sm.apply(4, {"op": "CAS", "key": "a", "expected": 12, "val": 1}) is False
    assert sm.apply(5, {"op": "APPEND", "key": "s", "val": "hi"}) == "hi"
    assert sm.apply(6, {"op": "GET", "key": "s"}) == "hi"


def test_happy_path_agreement():
    out = eng.run_happy_path(n=4, n_ops=8, seed=21)
    assert out["agreement"] is True
    assert out["validity"] is True
    assert out["stores_agree"] is True
    assert out["committed_ops"] >= 5


def test_n7_f2_happy():
    out = eng.run_happy_path(n=7, n_ops=5, seed=22)
    assert out["agreement"] is True
    assert out["f"] == 2
    assert out["stores_agree"] is True


def test_leader_crash_progress():
    out = eng.run_leader_crash(n=4, seed=23)
    assert out["agreement"] is True
    assert out["committed_ops"] >= 2
    assert out["stores_agree"] is True


def test_byzantine_fork_safety():
    out = eng.run_byzantine_fork(n=4, seed=24)
    assert out["agreement"] is True
    assert out["validity"] is True
    assert out["stores_agree"] is True


def test_lossy_network_agreement():
    out = eng.run_lossy(n=4, drop=0.08, seed=25)
    assert out["agreement"] is True
    assert out["committed_ops"] >= 3


def test_chaos_combined():
    out = eng.run_chaos(n=4, drop=0.05, seed=26)
    assert out["agreement"] is True
    assert out["stores_agree"] is True


def test_theory_ledger_linearity():
    t = eng.theory_ledger(n=4)
    assert t["hotstuff_msgs_per_block"] == 12
    assert t["pbft_msgs_per_op"] > t["hotstuff_msgs_per_block"]
    assert t["linearity_ratio_pbft_over_hs"] >= 2.0
    assert t["n_equals_3f1"] is True


def test_compare_pbft():
    c = eng.compare_pbft(n=7)
    assert c["hotstuff"]["complexity"] == "O(n)"
    assert c["pbft"]["complexity"] == "O(n²)"
    assert c["msg_savings_factor"] > 1


def test_safe_node_rules():
    r = eng.HotStuffReplica(0, 4)
    parent_qc = eng.QC(view=1, height=1, block_hash="parenthash0000001", voters=(0, 1, 2))
    pb = eng.Block(view=1, height=1, parent_id=eng.GENESIS_HASH, cmd=None, justify=None)
    pb.block_hash = "parenthash0000001"
    r.blocks["parenthash0000001"] = pb
    r.locked_qc = eng.QC(view=1, height=1, block_hash="parenthash0000001", voters=(0, 1, 2))
    good = eng.Block(
        view=2, height=2, parent_id="parenthash0000001", cmd=None, justify=parent_qc
    )
    r.blocks[good.block_hash] = good
    assert r.safe_node(good) is True
    high_j = eng.QC(view=5, height=1, block_hash=eng.GENESIS_HASH, voters=(0, 1, 2))
    unlocked = eng.Block(view=6, height=2, parent_id=eng.GENESIS_HASH, cmd=None, justify=high_j)
    assert r.safe_node(unlocked) is True


def test_bench_smoke():
    out = eng.run_bench(n=4, ops=12, seed=3)
    assert out["agreement"] is True
    assert out["msgs_per_op"] < 50
    assert out["stores_agree"] is True


def test_selftest_gate():
    assert eng.selftest(verbose=False) == 0


def test_block_hash_stable():
    c = eng.Command(0, 1, {"op": "SET", "key": "x", "val": 1})
    b1 = eng.Block(view=1, height=1, parent_id=eng.GENESIS_HASH, cmd=c, justify=None)
    b2 = eng.Block(view=1, height=1, parent_id=eng.GENESIS_HASH, cmd=c, justify=None)
    assert b1.block_hash == b2.block_hash
    assert len(b1.block_hash) == 16
