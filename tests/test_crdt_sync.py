"""
Comprehensive Unit and Integration Test Suite for CRDT & Causal Vector Clock Engine.
"""

from importlib.machinery import SourceFileLoader
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "crdt-sync"

loader = SourceFileLoader("crdt_sync", str(CLI))
mod = types.ModuleType(loader.name)
loader.exec_module(mod)

VectorClock = mod.VectorClock
ClockRelation = mod.ClockRelation
GCounter = mod.GCounter
PNCounter = mod.PNCounter
LWWElementSet = mod.LWWElementSet
ORSet = mod.ORSet
RGAText = mod.RGAText
AtomId = mod.AtomId
Atom = mod.Atom
CollaborativeDoc = mod.CollaborativeDoc
ChaosNetworkSimulator = mod.ChaosNetworkSimulator


class TestVectorClock(unittest.TestCase):
    def test_vector_clock_basic(self):
        vc = VectorClock()
        self.assertEqual(vc.tick("A"), 1)
        self.assertEqual(vc.tick("A"), 2)
        self.assertEqual(vc.tick("B"), 1)
        self.assertEqual(vc.get("A"), 2)
        self.assertEqual(vc.get("B"), 1)
        self.assertEqual(vc.get("C"), 0)

    def test_vector_clock_merge_and_domination(self):
        v1 = VectorClock({"A": 3, "B": 1})
        v2 = VectorClock({"A": 1, "B": 4, "C": 2})
        v_join = v1.merge(v2)
        self.assertEqual(v_join.clocks, {"A": 3, "B": 4, "C": 2})
        self.assertTrue(v_join.dominates(v1))
        self.assertTrue(v_join.dominates(v2))
        self.assertFalse(v1.dominates(v2))
        self.assertEqual(v1.relation(v2), ClockRelation.CONCURRENT)

    def test_vector_clock_serialization(self):
        v1 = VectorClock({"A": 5, "B": 2})
        data = v1.to_dict()
        v2 = VectorClock.from_dict(data)
        self.assertEqual(v1, v2)


class TestCounters(unittest.TestCase):
    def test_g_counter(self):
        g1 = GCounter()
        g2 = GCounter()
        g1.inc("A", 5)
        g2.inc("B", 8)
        g_merged = g1.merge(g2)
        self.assertEqual(g_merged.value, 13)
        # Idempotence
        self.assertEqual(g_merged.merge(g1).value, 13)

    def test_pn_counter(self):
        p1 = PNCounter()
        p2 = PNCounter()
        p1.inc("A", 15)
        p1.dec("A", 4)
        p2.inc("B", 10)
        p2.dec("B", 12)
        p_merged = p1.merge(p2)
        self.assertEqual(p_merged.value, (15 - 4) + (10 - 12))
        self.assertEqual(p_merged.value, 9)

        # Serialization round-trip
        data = p_merged.to_dict()
        p_restored = PNCounter.from_dict(data)
        self.assertEqual(p_restored.value, 9)


class TestSets(unittest.TestCase):
    def test_lww_element_set(self):
        s1 = LWWElementSet()
        s2 = LWWElementSet()
        t0 = 100.0
        s1.add("x", "A", timestamp=t0)
        s2.remove("x", "B", timestamp=t0 + 1.0)
        merged = s1.merge(s2)
        self.assertFalse(merged.contains("x"))

        # Re-add with higher timestamp
        merged.add("x", "A", timestamp=t0 + 2.0)
        self.assertTrue(merged.contains("x"))

    def test_or_set_concurrent_add_remove(self):
        or1 = ORSet()
        or2 = ORSet()
        or1.add("item1", "nodeA")
        # or2 concurrently adds item1
        or2.add("item1", "nodeB")
        # or1 removes item1 (observes only nodeA's tag)
        or1.remove("item1")
        self.assertFalse(or1.contains("item1"))

        # Merge: unobserved add from nodeB must survive!
        merged = or1.merge(or2)
        self.assertTrue(merged.contains("item1"))

        # Now remove item1 after merge
        merged.remove("item1")
        self.assertFalse(merged.contains("item1"))


class TestRGAText(unittest.TestCase):
    def test_rga_basic_editing(self):
        rga = RGAText(node_id="A")
        rga.insert_string(0, "HELLO")
        self.assertEqual(rga.text, "HELLO")
        rga.delete_at(1)  # Delete 'E'
        self.assertEqual(rga.text, "HLLO")
        rga.insert_at(1, "A")
        self.assertEqual(rga.text, "HALLO")

    def test_rga_concurrent_inserts_convergence(self):
        rga1 = RGAText(node_id="node1")
        rga2 = RGAText(node_id="node2")
        rga1.insert_string(0, "ABC")
        rga2.merge(rga1)

        # Concurrent insert at position 1
        rga1.insert_at(1, "X")
        rga2.insert_at(1, "Y")

        rga1.merge(rga2)
        rga2.merge(rga1)
        self.assertEqual(rga1.text, rga2.text)
        self.assertIn("X", rga1.text)
        self.assertIn("Y", rga1.text)

    def test_rga_serialization(self):
        rga = RGAText(node_id="node1")
        rga.insert_string(0, "CRDT_SERIALIZE")
        rga.delete_at(4)
        data = rga.to_dict()
        restored = RGAText.from_dict(data)
        self.assertEqual(rga.text, restored.text)
        self.assertEqual(len(rga.atoms), len(restored.atoms))


class TestCollaborativeDoc(unittest.TestCase):
    def test_collaborative_document(self):
        docA = CollaborativeDoc(doc_id="doc1", node_id="A")
        docB = CollaborativeDoc(doc_id="doc1", node_id="B")

        docA.get_counter("likes").inc("A", 5)
        docB.get_counter("likes").dec("B", 1)
        docA.get_set("tags").add("distributed", "A")
        docB.get_set("tags").add("crdt", "B")
        docA.get_text("body").insert_string(0, "Title: Hermes")
        docA.set_register("status", "draft")
        docB.set_register("status", "published", timestamp=time.time() + 10.0)

        docA.merge(docB)
        docB.merge(docA)

        snapA = docA.snapshot()
        snapB = docB.snapshot()
        self.assertEqual(snapA["counters"], snapB["counters"])
        self.assertEqual(snapA["counters"]["likes"], 4)
        self.assertEqual(snapA["sets"]["tags"], ["crdt", "distributed"])
        self.assertEqual(snapA["registers"]["status"], "published")
        self.assertEqual(snapA["texts"]["body"], "Title: Hermes")


class TestChaosNetwork(unittest.TestCase):
    def test_split_brain_partition_and_heal(self):
        sim = ChaosNetworkSimulator(["P1", "P2", "P3"])
        sim.nodes["P1"].insert_string(0, "START")
        sim.sync_all()

        # Partition P1 away from P2, P3
        sim.set_partitions([["P1"], ["P2", "P3"]])
        sim.nodes["P1"].insert_string(5, "_ONE")
        sim.nodes["P2"].insert_string(0, "ZERO_")

        # Heal
        sim.heal_partitions()
        sim.sync_all()
        self.assertTrue(sim.assert_strong_eventual_consistency())
        self.assertEqual(sim.nodes["P1"].text, sim.nodes["P2"].text)


class TestCLIExecution(unittest.TestCase):
    def test_cli_subcommands(self):
        # 1. test
        res = subprocess.run([sys.executable, str(CLI), "test"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("0 Failed", res.stdout)

        # 2. bench
        res_bench = subprocess.run([sys.executable, str(CLI), "bench", "--ops", "200", "--peers", "3"], capture_output=True, text=True)
        self.assertEqual(res_bench.returncode, 0)
        self.assertIn("Microbenchmark completed", res_bench.stdout)

        # 3. sim
        res_sim = subprocess.run([sys.executable, str(CLI), "sim", "--steps", "3"], capture_output=True, text=True)
        self.assertEqual(res_sim.returncode, 0)
        self.assertIn("Final Reconciled Text", res_sim.stdout)

        # 4. chaos
        res_chaos = subprocess.run([sys.executable, str(CLI), "chaos", "--peers", "4"], capture_output=True, text=True)
        self.assertEqual(res_chaos.returncode, 0)
        self.assertIn("Fully Reconciled Text", res_chaos.stdout)


if __name__ == '__main__':
    unittest.main()
