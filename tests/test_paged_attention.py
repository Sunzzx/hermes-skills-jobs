"""Comprehensive Unit and Integration Test Suite for PagedAttention Engine."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from importlib.machinery import SourceFileLoader
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "paged-attention"

# Import paged_attention module directly
sys.path.insert(0, str(ROOT / "bin"))
paged_module = SourceFileLoader("paged_attention", str(CLI)).load_module()

PhysicalBlockPool = paged_module.PhysicalBlockPool
BlockAllocator = paged_module.BlockAllocator
Sequence = paged_module.Sequence
RadixPrefixCache = paged_module.RadixPrefixCache
PagedAttentionEngine = paged_module.PagedAttentionEngine


class TestPagedAttentionEngine(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1337)

    def test_allocator_lifecycle(self):
        alloc = BlockAllocator(num_blocks=4)
        self.assertEqual(alloc.free_blocks_count, 4)
        self.assertEqual(alloc.allocated_blocks_count, 0)

        b0 = alloc.allocate()
        b1 = alloc.allocate()
        self.assertEqual(b0, 0)
        self.assertEqual(b1, 1)
        self.assertEqual(alloc.ref_counts[0], 1)
        self.assertEqual(alloc.ref_counts[1], 1)
        self.assertEqual(alloc.allocated_blocks_count, 2)

        alloc.retain(b0)
        self.assertEqual(alloc.ref_counts[b0], 2)

        alloc.release(b0)
        self.assertEqual(alloc.ref_counts[b0], 1)
        self.assertEqual(alloc.free_blocks_count, 2)

        alloc.release(b0)
        self.assertEqual(alloc.ref_counts[b0], 0)
        self.assertEqual(alloc.free_blocks_count, 3)
        self.assertIn(b0, alloc.free_list)

        # Exhaustion test
        b2 = alloc.allocate()
        b3 = alloc.allocate()
        b4 = alloc.allocate()
        with self.assertRaises(MemoryError):
            alloc.allocate()

    def test_copy_on_write_isolation(self):
        engine = PagedAttentionEngine(num_blocks=32, block_size=16, num_heads=2, head_dim=32)
        seq1 = engine.register_sequence("seq1", list(range(20)))  # 2 blocks (16 + 4)
        self.assertEqual(len(seq1.block_table), 2)

        # Fill KV data
        K_data = self.rng.randn(2, 20, 32)
        V_data = self.rng.randn(2, 20, 32)
        for t in range(20):
            engine.write_kv("seq1", t, K_data[:, t, :], V_data[:, t, :])

        # Fork
        seq2 = engine.fork_sequence("seq1", "seq2")
        self.assertEqual(seq1.block_table, seq2.block_table)
        self.assertEqual(engine.allocator.ref_counts[seq1.block_table[0]], 2)
        self.assertEqual(engine.allocator.ref_counts[seq1.block_table[1]], 2)

        # Append to seq2 (slot 5 in block 1 -> triggers CoW!)
        cows_start = engine.allocator.cow_counter
        k_new = self.rng.randn(2, 32)
        v_new = self.rng.randn(2, 32)
        engine.append_and_write_token("seq2", 500, k_new, v_new)

        self.assertEqual(engine.allocator.cow_counter, cows_start + 1)
        # seq1 and seq2 should now have different physical IDs for block 1
        self.assertEqual(seq1.block_table[0], seq2.block_table[0])
        self.assertNotEqual(seq1.block_table[1], seq2.block_table[1])
        self.assertEqual(engine.allocator.ref_counts[seq1.block_table[1]], 1)
        self.assertEqual(engine.allocator.ref_counts[seq2.block_table[1]], 1)

        # Verify seq1 KV is completely intact
        q = self.rng.randn(2, 32)
        out1 = engine.forward_decode("seq1", q)
        out1_ref = PagedAttentionEngine.naive_dense_attention(q, K_data, V_data)
        np.testing.assert_allclose(out1, out1_ref, atol=1e-12)

    def test_prefix_caching(self):
        cache = RadixPrefixCache(block_size=4)
        chunk = (10, 20, 30, 40)
        cache.register_chunk(chunk, block_id=7)
        self.assertEqual(cache.get_cached_block(chunk), 7)
        self.assertIsNone(cache.get_cached_block((10, 20, 30, 41)))

        cache.invalidate_block(7)
        self.assertIsNone(cache.get_cached_block(chunk))

    def test_mathematical_precision_various_shapes(self):
        test_params = [
            # (H, d, B, L)
            (1, 32, 16, 16),
            (2, 64, 16, 33),
            (4, 64, 8, 80),
            (8, 64, 32, 127),
            (2, 128, 16, 255),
        ]

        for H, d, B, L in test_params:
            engine = PagedAttentionEngine(num_blocks=64, block_size=B, num_heads=H, head_dim=d)
            seq_id = f"test_{H}_{d}_{B}_{L}"
            engine.register_sequence(seq_id, list(range(L)))

            K = self.rng.randn(H, L, d)
            V = self.rng.randn(H, L, d)
            q = self.rng.randn(H, d)

            for t in range(L):
                engine.write_kv(seq_id, t, K[:, t, :], V[:, t, :])

            out_paged = engine.forward_decode(seq_id, q)
            out_dense = PagedAttentionEngine.naive_dense_attention(q, K, V)

            max_err = np.max(np.abs(out_paged - out_dense))
            self.assertLess(
                max_err, 1e-11,
                f"Failed precision on H={H}, d={d}, B={B}, L={L}. Max diff: {max_err}",
            )

    def test_cli_execution(self):
        commands = [
            ["verify"],
            ["demo"],
            ["status"],
            ["bench", "--heads", "2", "--dim", "32", "--block-size", "16", "--batch-size", "2"],
        ]

        for cmd_args in commands:
            res = subprocess.run(
                [sys.executable, str(CLI), *cmd_args],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res.returncode, 0,
                f"CLI failed for {cmd_args}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
