"""Unit tests for swim-gossip-membership: SWIM gossip failure detector."""

import os
from importlib.machinery import SourceFileLoader

bin_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../bin/swim-gossip-membership")
)
eng = SourceFileLoader("swim_gossip_membership", bin_path).load_module()


def test_theory_linearity():
    t = eng.theory_ledger(n=32, k=3)
    assert t["swim_msgs_per_period_happy"] < t["all_to_all_heartbeat_msgs"]
    assert t["complexity"]["ratio_mesh_vs_swim"] > 5
    t128 = eng.theory_ledger(n=128)
    assert t128["complexity"]["ratio_mesh_vs_swim"] > 20


def test_membership_update_key():
    u = eng.MembershipUpdate(3, eng.MemberState.DEAD, 4, from_node=1, hop=2)
    assert u.key() == (3, "dead", 4)
    assert u.as_dict()["incarnation"] == 4


def test_incarnation_self_refutation():
    nd = eng.SwimNode(0, 4, seed=1)
    eng._init_node_maps(nd)
    nd._enqueue_update(
        eng.MembershipUpdate(0, eng.MemberState.SUSPECT, 0, from_node=1), now=1.0
    )
    assert nd.incarnation >= 1
    assert nd.view[0].state == eng.MemberState.ALIVE
    assert nd.refutations >= 1


def test_update_precedence():
    nd = eng.SwimNode(1, 4, seed=2)
    eng._init_node_maps(nd)
    nd._enqueue_update(
        eng.MembershipUpdate(2, eng.MemberState.ALIVE, 1, from_node=0), 1.0
    )
    nd._enqueue_update(
        eng.MembershipUpdate(2, eng.MemberState.SUSPECT, 1, from_node=0), 2.0
    )
    assert nd.view[2].state == eng.MemberState.SUSPECT
    # same incarnation cannot downgrade suspect → alive
    nd._enqueue_update(
        eng.MembershipUpdate(2, eng.MemberState.ALIVE, 1, from_node=0), 3.0
    )
    assert nd.view[2].state == eng.MemberState.SUSPECT
    # higher incarnation wins
    nd._enqueue_update(
        eng.MembershipUpdate(2, eng.MemberState.ALIVE, 2, from_node=2), 4.0
    )
    assert nd.view[2].state == eng.MemberState.ALIVE
    assert nd.view[2].incarnation == 2


def test_partition_drops_cross_traffic():
    net = eng.LossyNetwork(4, partitions=[{0, 1}, {2, 3}], seed=3)
    net.send(eng.Message(eng.MsgType.PING, 0, 2, seq=1))
    assert net.dropped == 1
    assert net.sent == 1
    net.send(eng.Message(eng.MsgType.PING, 0, 1, seq=2))
    assert net.sent == 2
    assert len(net.queue) == 1  # intra-partition delivered to queue


def test_happy_path_no_false_positives():
    out = eng.run_happy(n=6, ticks=100, seed=101)
    assert out["ok"] is True
    assert out["false_positives"] == 0
    assert out["agreement"]["mean_accuracy"] >= 0.99


def test_single_crash_detection():
    out = eng.run_single_crash(n=8, ticks=200, seed=102, crash_at=25, victim=2)
    assert out["ok"] is True
    assert out["detected_fraction"] >= 0.8
    assert out["true_detections"] >= 1
    assert out["detection_latency"]["mean"] is not None
    assert out["detection_latency"]["mean"] > 0


def test_multi_crash():
    out = eng.run_multi_crash(n=10, victims=[1, 4, 7], ticks=240, seed=103)
    assert out["ok"] is True
    assert out["mean_detected_fraction"] >= 0.7


def test_lossy_network_detection():
    out = eng.run_lossy(n=8, drop=0.1, ticks=280, seed=104)
    assert out["ok"] is True
    assert out["detected_fraction"] >= 0.6


def test_false_suspicion_refute():
    out = eng.run_false_suspicion_refute(n=8, ticks=150, seed=105)
    assert out["ok"] is True
    assert out["dead_marks"] <= 2


def test_partition_scenario():
    out = eng.run_partition(n=10, ticks=160, seed=106)
    assert out["ok"] is True
    assert out["cross_detected"] >= 3


def test_chaos_combined():
    out = eng.run_chaos(n=10, drop=0.05, ticks=260, seed=107)
    assert out["ok"] is True
    assert out["true_detections"] >= 1


def test_compare_ledger():
    cmp_ = eng.compare_ledger(n=16)
    assert cmp_["ok"] is True
    assert cmp_["theory"]["swim_msgs_per_period_happy"] > 0


def test_bench_runs():
    b = eng.bench(n=12, ticks=80, seed=108)
    assert b["net"]["sent"] > 0
    assert b["wall_ms"] > 0


def test_lonely_node_noop():
    lonely = eng.SwimNode(0, 1, seed=0)
    eng._init_node_maps(lonely)
    net = eng.LossyNetwork(1, seed=0)
    lonely.start_protocol_period(net)
    assert lonely.pending_ping is None


def test_selftest_exit_zero():
    assert eng.selftest() == 0


def test_engine_metadata():
    assert eng.ENGINE_NAME == "swim-gossip-membership"
    assert eng.ENGINE_VERSION == "1.0.0"
