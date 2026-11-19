"""Unit tests for pbft-byzantine-quorum: Practical Byzantine Fault Tolerance engine."""

import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/pbft-byzantine-quorum")
)
eng = SourceFileLoader("pbft_byzantine_quorum", bin_path).load_module()


def test_max_faults_and_quorum():
    assert eng.max_faults(4) == 1
    assert eng.max_faults(7) == 2
    assert eng.pbft_quorum(4) == 3
    assert eng.pbft_quorum(7) == 5
    assert 2 * eng.pbft_quorum(7) - 7 == 3  # f+1


def test_digest_deterministic():
    assert eng.digest({"a": 1}, 2) == eng.digest({"a": 1}, 2)
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
    assert out["results_non_null"] == 8
    assert out["executed"] >= 8


def test_n7_f2_happy():
    out = eng.run_happy_path(n=7, n_ops=5, seed=22)
    assert out["agreement"] is True
    assert out["results_non_null"] == 5
    assert out["f"] == 2


def test_primary_crash_view_change():
    out = eng.run_primary_crash(n=4, seed=23)
    assert out["agreement"] is True
    assert out["post_view_result"] is not None
    assert len(set(out["honest_x_values"])) == 1


def test_byzantine_equivocation_safety():
    out = eng.run_byzantine_equivocation(n=4, seed=24)
    assert out["agreement"] is True
    assert out["validity"] is True
    assert len(set(out["honest_final"])) == 1
    assert out["results_ok"] >= 6


def test_lossy_network_agreement():
    out = eng.run_lossy_network(n=4, drop=0.08, seed=25)
    assert out["agreement"] is True
    assert out["committed_ops"] >= 5


def test_checkpoint_watermark():
    c = eng.PBFTCluster(n=4, checkpoint_interval=3, seed=26)
    for i in range(6):
        assert c.request({"op": "SET", "key": "c", "val": i}) is not None
    ck = min(r.last_stable_checkpoint for r in c.replicas.values())
    assert ck >= 3


def test_theory_ledger():
    t = eng.theory_ledger(n=4)
    assert t["f"] == 1
    assert t["quorum_2f1"] == 3
    assert t["normal_case_msgs_per_request"] > 0


def test_selftest_gate():
    assert eng.selftest(verbose=False) == 0


def test_bench_small():
    out = eng.run_bench(n=4, ops=8, seed=27)
    assert out["agreement"] is True
    assert out["ok"] == 8
    assert out["final_counter"] == 8
