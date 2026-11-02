"""Comprehensive Unit and Integration Test Suite for Lehman-Yao B-Link Tree Engine."""

from importlib.machinery import SourceFileLoader
import math
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "blink-tree"

loader = SourceFileLoader("blink_tree_engine", str(BIN_PATH))
blink_mod = types.ModuleType(loader.name)
loader.exec_module(blink_mod)

BLinkNode = blink_mod.BLinkNode
BLinkTree = blink_mod.BLinkTree
BLinkEngine = blink_mod.BLinkEngine
WriteAheadLog = blink_mod.WriteAheadLog
WALRecord = blink_mod.WALRecord
POS_INF = blink_mod.POS_INF
key_gt = blink_mod.key_gt
key_ge = blink_mod.key_ge
key_lt = blink_mod.key_lt
key_le = blink_mod.key_le
key_eq = blink_mod.key_eq
run_concurrency_benchmark = blink_mod.run_concurrency_benchmark
run_verification_suite = blink_mod.run_verification_suite


class TestKeyComparators(unittest.TestCase):
    def test_pos_inf_properties(self):
        self.assertTrue(key_gt(POS_INF, 1000))
        self.assertTrue(key_gt(POS_INF, "zzzz"))
        self.assertFalse(key_gt(1000, POS_INF))
        self.assertTrue(key_le(1000, POS_INF))
        self.assertTrue(key_le(POS_INF, POS_INF))
        self.assertTrue(key_ge(POS_INF, POS_INF))
        self.assertTrue(key_eq(POS_INF, POS_INF))
        self.assertFalse(key_eq(POS_INF, 100))
        self.assertEqual(str(POS_INF), "+∞")


class TestBLinkNode(unittest.TestCase):
    def test_node_creation_and_serialization(self):
        node = BLinkNode(node_id=1, level=0, high_key=POS_INF, keys=[10, 20], values=["a", "b"])
        self.assertTrue(node.is_leaf)
        self.assertEqual(node.high_key, POS_INF)

        d = node.to_dict()
        self.assertEqual(d["node_id"], 1)
        self.assertEqual(d["high_key"], "+inf")
        self.assertEqual(d["keys"], [10, 20])

        restored = BLinkNode.from_dict(d)
        self.assertEqual(restored.node_id, 1)
        self.assertEqual(restored.high_key, POS_INF)
        self.assertEqual(restored.keys, [10, 20])
        self.assertEqual(restored.values, ["a", "b"])


class TestWriteAheadLog(unittest.TestCase):
    def test_wal_append_replay_and_truncate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "test.wal"
            wal = WriteAheadLog(wal_path)

            lsn1 = wal.append(blink_mod.OP_PUT, "k1", "v1")
            lsn2 = wal.append(blink_mod.OP_PUT, "k2", "v2")
            lsn3 = wal.append(blink_mod.OP_DELETE, "k1", None)
            self.assertEqual(lsn1, 1)
            self.assertEqual(lsn2, 2)
            self.assertEqual(lsn3, 3)
            wal.close()

            # Reopen and replay
            wal2 = WriteAheadLog(wal_path)
            records = wal2.replay()
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0].key, "k1")
            self.assertEqual(records[0].value, "v1")
            self.assertEqual(records[1].key, "k2")
            self.assertEqual(records[2].op_type, blink_mod.OP_DELETE)

            # Truncate
            wal2.truncate()
            records_after = wal2.replay()
            self.assertEqual(len(records_after), 0)
            wal2.close()

    def test_wal_crc32_corruption_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = Path(tmpdir) / "corrupt.wal"
            wal = WriteAheadLog(wal_path)
            wal.append(blink_mod.OP_PUT, "k1", "v1")
            wal.append(blink_mod.OP_PUT, "k2", "v2")
            wal.close()

            # Corrupt byte in second record
            with open(wal_path, "r+b") as f:
                f.seek(-5, os.SEEK_END)
                f.write(b"\xFF")

            wal_recover = WriteAheadLog(wal_path)
            records = wal_recover.replay()
            # Should recover first valid record and halt before corrupted record
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].key, "k1")
            wal_recover.close()


class TestBLinkTreeCore(unittest.TestCase):
    def test_crud_operations(self):
        tree = BLinkTree(fanout=3)
        self.assertIsNone(tree.get("missing"))
        self.assertFalse("missing" in tree)

        tree.put("apple", 100)
        tree.put("banana", 200)
        tree.put("cherry", 300)

        self.assertEqual(tree.get("apple"), 100)
        self.assertEqual(tree.get("banana"), 200)
        self.assertEqual(tree.get("cherry"), 300)
        self.assertTrue("banana" in tree)

        # Update
        tree.put("banana", 999)
        self.assertEqual(tree.get("banana"), 999)

        # Delete
        deleted = tree.delete("banana")
        self.assertTrue(deleted)
        self.assertIsNone(tree.get("banana"))
        self.assertFalse("banana" in tree)
        self.assertFalse(tree.delete("nonexistent"))

    def test_invariants_under_heavy_splits(self):
        # Minimal fanout B=2 forces frequent splits at leaf and internal levels
        tree = BLinkTree(fanout=2)
        keys = list(range(0, 300, 2))  # 150 even numbers
        for k in keys:
            tree.put(k, f"val_{k}")

        # Verify all keys retrieve correctly
        for k in keys:
            self.assertEqual(tree.get(k), f"val_{k}")
            self.assertIsNone(tree.get(k + 1))

        # Verify mathematical structural invariants
        inv = tree.verify_invariants()
        self.assertTrue(inv["valid"], f"Invariants failed: {inv['errors']}")
        self.assertGreater(inv["depth"], 2)

    def test_horizontal_range_scans(self):
        tree = BLinkTree(fanout=3)
        for i in range(100):
            tree.put(i, i * 10)

        # Full scan
        full = tree.scan()
        self.assertEqual(len(full), 100)
        self.assertEqual(full[0], (0, 0))
        self.assertEqual(full[-1], (99, 990))

        # Sub-range scan
        sub = tree.scan(start_key=25, end_key=75)
        expected = [(i, i * 10) for i in range(25, 76)]
        self.assertEqual(sub, expected)

        # Scan with limit
        lim = tree.scan(start_key=10, limit=5)
        self.assertEqual(len(lim), 5)
        self.assertEqual(lim, [(i, i * 10) for i in range(10, 15)])

    def test_snapshot_save_and_load(self):
        tree = BLinkTree(fanout=3)
        for i in range(50):
            tree.put(f"item_{i:03d}", i)

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "snapshot.json"
            tree.save_snapshot(snap_path)
            self.assertTrue(snap_path.exists())

            restored = BLinkTree.load_snapshot(snap_path)
            self.assertEqual(restored.fanout, 3)
            for i in range(50):
                self.assertEqual(restored.get(f"item_{i:03d}"), i)
            self.assertTrue(restored.verify_invariants()["valid"])

    def test_stats_and_ascii_visualizer(self):
        tree = BLinkTree(fanout=2)
        for i in range(20):
            tree.put(i, f"val_{i}")
        stats = tree.get_stats()
        self.assertGreater(stats["depth"], 1)
        self.assertEqual(stats["total_keys"], 20)
        self.assertGreater(stats["splits_occurred"], 0)

        ascii_tree = tree.render_ascii_tree()
        self.assertIn("Lehman-Yao B-Link Tree", ascii_tree)
        self.assertIn("Level 0", ascii_tree)
        self.assertIn("link", ascii_tree)


class TestBLinkEngineAndConcurrency(unittest.TestCase):
    def test_engine_persistence_and_wal_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = BLinkEngine(data_dir=tmpdir, fanout=3)
            for i in range(40):
                engine.put(f"key_{i}", f"val_{i}")
            engine.delete("key_5")
            engine.close()

            # Reopen and check crash recovery
            engine2 = BLinkEngine(data_dir=tmpdir, fanout=3)
            self.assertEqual(engine2.get("key_0"), "val_0")
            self.assertEqual(engine2.get("key_39"), "val_39")
            self.assertIsNone(engine2.get("key_5"))
            engine2.checkpoint()
            engine2.close()

    def test_multi_threaded_concurrency_stress(self):
        res = run_concurrency_benchmark(
            num_threads=6,
            num_operations=2400,
            read_ratio=0.75,
            fanout=3,
        )
        self.assertTrue(res["invariants_valid"])
        self.assertGreater(res["throughput_ops_sec"], 1000)
        self.assertGreater(res["tree_splits"], 0)

    def test_verification_suite(self):
        self.assertTrue(run_verification_suite())


if __name__ == '__main__':
    unittest.main()
