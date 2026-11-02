"""Comprehensive Unit and Integration Test Suite for Kademlia DHT Engine."""

from importlib.machinery import SourceFileLoader
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "kademlia"

loader = SourceFileLoader("kademlia_engine", str(BIN_PATH))
kad_mod = types.ModuleType(loader.name)
loader.exec_module(kad_mod)

NodeID = kad_mod.NodeID
Contact = kad_mod.Contact
KBucket = kad_mod.KBucket
RoutingTable = kad_mod.RoutingTable
RPCMethod = kad_mod.RPCMethod
RPCMessage = kad_mod.RPCMessage
StoredValue = kad_mod.StoredValue
KademliaNode = kad_mod.KademliaNode
KademliaNetwork = kad_mod.KademliaNetwork
KEY_BITS = kad_mod.KEY_BITS
MAX_ID = kad_mod.MAX_ID


class TestNodeID(unittest.TestCase):
    def test_node_id_creation_and_bounds(self):
        id0 = NodeID(0)
        id_max = NodeID(MAX_ID)
        self.assertEqual(id0.value, 0)
        self.assertEqual(id_max.value, MAX_ID)
        with self.assertRaises(ValueError):
            NodeID(-1)
        with self.assertRaises(ValueError):
            NodeID(MAX_ID + 1)

    def test_sha1_hashing_and_hex_roundtrip(self):
        nid = NodeID.from_key("test-key-alpha")
        hex_str = nid.to_hex()
        self.assertEqual(len(hex_str), 40)
        reconstructed = NodeID(hex_str)
        self.assertEqual(nid, reconstructed)
        self.assertEqual(hash(nid), hash(reconstructed))

    def test_xor_distance_and_identity(self):
        n1 = NodeID.random()
        n2 = NodeID.random()
        self.assertEqual(n1.distance(n1), 0)
        self.assertEqual(n1.distance(n2), n2.distance(n1))
        self.assertGreaterEqual(n1.distance(n2), 0)

    def test_unidirectional_and_triangle_equality(self):
        x = NodeID.random()
        y = NodeID.random()
        z = NodeID.random()
        d_xz = x.distance(z)
        d_xy = x.distance(y)
        d_yz = y.distance(z)
        # XOR metric property: d(x, z) = d(x, y) ^ d(y, z)
        self.assertEqual(d_xz, d_xy ^ d_yz)
        # Standard metric triangle inequality: d(x, z) <= d(x, y) + d(y, z)
        self.assertLessEqual(d_xz, d_xy + d_yz)

    def test_prefix_len_and_bit_manipulation(self):
        val_a = (0b11110000) << (KEY_BITS - 8)
        val_b = (0b11111111) << (KEY_BITS - 8)
        na = NodeID(val_a)
        nb = NodeID(val_b)
        self.assertEqual(na.prefix_len(nb), 4)
        self.assertEqual(na.get_bit(0), 1)
        self.assertEqual(na.get_bit(1), 1)
        self.assertEqual(na.get_bit(2), 1)
        self.assertEqual(na.get_bit(3), 1)
        self.assertEqual(na.get_bit(4), 0)
        with self.assertRaises(IndexError):
            na.get_bit(160)


class TestContact(unittest.TestCase):
    def test_contact_touch_and_failure(self):
        c = Contact(NodeID.random(), ip="192.168.1.1", port=9000)
        c.record_failure()
        self.assertEqual(c.failed_pings, 1)
        c.touch()
        self.assertEqual(c.failed_pings, 0)

    def test_contact_dict_serialization(self):
        nid = NodeID.random()
        c = Contact(nid, ip="10.0.0.1", port=8080, rtt_ms=4.5)
        d = c.to_dict()
        c2 = Contact.from_dict(d)
        self.assertEqual(c.node_id, c2.node_id)
        self.assertEqual(c.ip, c2.ip)
        self.assertEqual(c.port, c2.port)
        self.assertAlmostEqual(c.rtt_ms, c2.rtt_ms)


class TestKBucket(unittest.TestCase):
    def test_kbucket_lru_insertion_and_ordering(self):
        b = KBucket(0, MAX_ID, k=3)
        c1 = Contact(NodeID.random())
        c2 = Contact(NodeID.random())
        c3 = Contact(NodeID.random())

        self.assertTrue(b.add_contact(c1))
        self.assertTrue(b.add_contact(c2))
        self.assertTrue(b.add_contact(c3))
        self.assertEqual(len(b), 3)
        self.assertTrue(b.is_full())

        # Order is [c1, c2, c3]. Touching c1 moves it to tail -> [c2, c3, c1]
        b.add_contact(c1)
        self.assertEqual(b.contacts[-1].node_id, c1.node_id)
        self.assertEqual(b.contacts[0].node_id, c2.node_id)

    def test_kbucket_replacement_cache_promotion(self):
        b = KBucket(0, MAX_ID, k=2)
        c1 = Contact(NodeID.random())
        c2 = Contact(NodeID.random())
        c3 = Contact(NodeID.random())

        b.add_contact(c1)
        b.add_contact(c2)
        # Head is alive -> c3 placed in replacement cache
        b.add_contact(c3, ping_callback=lambda c: True)
        self.assertEqual(len(b.contacts), 2)
        self.assertEqual(len(b.replacement_cache), 1)
        self.assertEqual(b.replacement_cache[0].node_id, c3.node_id)

        # Removing contact promotes c3 into active contacts
        removed = b.remove_contact(c1.node_id)
        self.assertEqual(removed.node_id, c1.node_id)
        self.assertEqual(len(b.contacts), 2)
        self.assertTrue(b.contains(c3.node_id))
        self.assertEqual(len(b.replacement_cache), 0)


class TestRoutingTable(unittest.TestCase):
    def test_bucket_index_calculation(self):
        local_id = NodeID(0)
        rt = RoutingTable(local_id, k=20)
        # Distance 1 -> bit_length = 1 -> bucket 0
        self.assertEqual(rt.get_bucket_index(NodeID(1)), 0)
        # Distance 2 -> bit_length = 2 -> bucket 1
        self.assertEqual(rt.get_bucket_index(NodeID(2)), 1)
        # Distance 4 -> bit_length = 3 -> bucket 2
        self.assertEqual(rt.get_bucket_index(NodeID(4)), 2)

    def test_find_closest_contacts_ordering(self):
        local_id = NodeID.random()
        rt = RoutingTable(local_id, k=5)
        contacts = [Contact(NodeID.random()) for _ in range(15)]
        for c in contacts:
            rt.add_contact(c)

        target = NodeID.random()
        closest = rt.find_closest_contacts(target, count=5)
        self.assertEqual(len(closest), 5)
        # Check strict distance sorting
        distances = [target.distance(c.node_id) for c in closest]
        self.assertEqual(distances, sorted(distances))


class TestKademliaNodeAndNetwork(unittest.TestCase):
    def setUp(self):
        self.net = KademliaNetwork.create_cluster(num_nodes=25, k=8, alpha=3, seed=999)
        self.nodes = list(self.net.nodes.values())

    def test_rpc_ping_pong(self):
        n1 = self.nodes[0]
        n2 = self.nodes[1]
        msg = RPCMessage(
            message_id="ping_1",
            method=RPCMethod.PING,
            sender=n1.contact,
        )
        resp = n2.handle_rpc(msg)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.method, RPCMethod.PONG)
        self.assertEqual(resp.payload.get("status"), "OK")

    def test_iterative_node_lookup_convergence(self):
        origin = self.nodes[0]
        target = self.nodes[-1].node_id
        closest = origin.iterative_find_node(target)
        self.assertGreater(len(closest), 0)
        # Target node itself must be found in cluster
        self.assertEqual(closest[0].node_id, target)

    def test_iterative_store_and_find_value(self):
        publisher = self.nodes[0]
        inquirer = self.nodes[10]
        key = "config.distributed.cluster_name"
        value = {"name": "Hermes-Titan", "nodes": 25}

        # Store key
        replicated_count = publisher.iterative_store(key, value, replication=5)
        self.assertGreaterEqual(replicated_count, 3)

        # Query from distant node
        found_val, path = inquirer.iterative_find_value(key)
        self.assertEqual(found_val, value)
        self.assertGreater(len(path), 0)

    def test_churn_resilience(self):
        publisher = self.nodes[0]
        key = "resilience.test.payload"
        value = "critical_invariant_data"
        publisher.iterative_store(key, value, replication=6)

        # Drop 25% of nodes offline
        drop_count = int(len(self.nodes) * 0.25)
        for node in self.nodes[2:2 + drop_count]:
            node.is_online = False

        # Query from an active node
        active_inquirer = self.nodes[1]
        found_val, _ = active_inquirer.iterative_find_value(key)
        self.assertEqual(found_val, value)


class TestCLIExecution(unittest.TestCase):
    def test_cli_subcommand_test(self):
        res = subprocess.run([sys.executable, str(BIN_PATH), "test"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("All 6 Kademlia Formal Mathematical Invariants Verified Successfully!", res.stdout)

    def test_cli_subcommand_table(self):
        res = subprocess.run([sys.executable, str(BIN_PATH), "table"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Kademlia 160-Bit Routing Table Visualization", res.stdout)

    def test_cli_subcommand_bench(self):
        res = subprocess.run([sys.executable, str(BIN_PATH), "bench"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Scaling characteristics confirm sub-linear logarithmic convergence", res.stdout)


if __name__ == '__main__':
    unittest.main()
