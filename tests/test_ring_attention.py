"""Comprehensive Unit and Integration Test Suite for RingAttention & Blockwise Parallel Context Engine."""

from __future__ import annotations

import subprocess
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "ring-attention"

sys.path.insert(0, str(ROOT / "bin"))
ring_module = SourceFileLoader("ring_attention", str(CLI)).load_module()

RingCommunicator = ring_module.RingCommunicator
OnlineSoftmaxAccumulator = ring_module.OnlineSoftmaxAccumulator
BlockwiseRingAttention = ring_module.BlockwiseRingAttention
MultiHeadRingAttention = ring_module.MultiHeadRingAttention
MonolithicAttention = ring_module.MonolithicAttention


class TestRingPrimitives(unittest.TestCase):
    def test_ring_communicator_topology(self):
        num_ranks = 4
        comm = RingCommunicator(num_ranks)
        payloads = [{"rank": r, "data": np.array([r * 10.0])} for r in range(num_ranks)]
        received = comm.step_ring_p2p(payloads)

        # Each rank p should have received from (p - 1) % num_ranks
        for p in range(num_ranks):
            expected_sender = (p - 1) % num_ranks
            self.assertEqual(received[p]["rank"], expected_sender)
            self.assertEqual(float(received[p]["data"][0]), expected_sender * 10.0)

        self.assertEqual(comm.stats.p2p_messages, 4)
        self.assertEqual(comm.stats.total_hops, 4)

    def test_online_softmax_accumulator_equivalence(self):
        rng = np.random.RandomState(42)
        num_rows = 8
        total_cols = 32
        head_dim = 4
        block_size = 8
        num_blocks = total_cols // block_size

        # Full logits and full V
        full_logits = rng.randn(num_rows, total_cols)
        full_V = rng.randn(total_cols, head_dim)

        # Standard offline softmax
        m_offline = np.max(full_logits, axis=-1, keepdims=True)
        exp_offline = np.exp(full_logits - m_offline)
        l_offline = np.sum(exp_offline, axis=-1, keepdims=True)
        P_offline = exp_offline / l_offline
        O_offline = P_offline @ full_V

        # Online softmax with blockwise updates
        acc = OnlineSoftmaxAccumulator(num_rows, head_dim)
        for b in range(num_blocks):
            block_logits = full_logits[:, b * block_size : (b + 1) * block_size]
            block_V = full_V[b * block_size : (b + 1) * block_size, :]
            acc.update(block_logits, block_V)

        O_online = acc.finalize()

        np.testing.assert_allclose(O_online, O_offline, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(acc.m, m_offline, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(acc.l, l_offline, rtol=1e-12, atol=1e-12)


class TestBlockwiseRingAttentionKernel(unittest.TestCase):
    def test_single_head_causal_equivalence(self):
        rng = np.random.RandomState(42)
        seq_len = 16
        num_ranks = 4
        block_len = seq_len // num_ranks
        head_dim = 8
        scale = 1.0 / np.sqrt(head_dim)

        Q = rng.randn(seq_len, head_dim)
        K = rng.randn(seq_len, head_dim)
        V = rng.randn(seq_len, head_dim)

        # Monolithic single-head causal attention
        scores = (Q @ K.T) * scale
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores[mask] = -np.inf
        m = np.max(scores, axis=-1, keepdims=True)
        P_mat = np.exp(scores - m) / np.sum(np.exp(scores - m), axis=-1, keepdims=True)
        O_ref = P_mat @ V

        # Ring Attention single-head causal
        kernel = BlockwiseRingAttention(num_ranks=num_ranks, causal=True)
        Q_shards = [Q[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]
        K_shards = [K[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]
        V_shards = [V[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]

        O_shards, _ = kernel.forward(Q_shards, K_shards, V_shards)
        O_ring = np.concatenate(O_shards, axis=0)

        np.testing.assert_allclose(O_ring, O_ref, rtol=1e-12, atol=1e-12)

    def test_single_head_non_causal_equivalence(self):
        rng = np.random.RandomState(42)
        seq_len = 16
        num_ranks = 4
        block_len = seq_len // num_ranks
        head_dim = 8
        scale = 1.0 / np.sqrt(head_dim)

        Q = rng.randn(seq_len, head_dim)
        K = rng.randn(seq_len, head_dim)
        V = rng.randn(seq_len, head_dim)

        scores = (Q @ K.T) * scale
        m = np.max(scores, axis=-1, keepdims=True)
        P_mat = np.exp(scores - m) / np.sum(np.exp(scores - m), axis=-1, keepdims=True)
        O_ref = P_mat @ V

        kernel = BlockwiseRingAttention(num_ranks=num_ranks, causal=False)
        Q_shards = [Q[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]
        K_shards = [K[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]
        V_shards = [V[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]

        O_shards, _ = kernel.forward(Q_shards, K_shards, V_shards)
        O_ring = np.concatenate(O_shards, axis=0)

        np.testing.assert_allclose(O_ring, O_ref, rtol=1e-12, atol=1e-12)


class TestMultiHeadRingAttentionEquivalence(unittest.TestCase):
    def test_causal_multi_head_forward_and_backward(self):
        rng = np.random.RandomState(123)
        seq_len = 32
        num_ranks = 4
        hidden_dim = 16
        num_heads = 2
        block_len = seq_len // num_ranks

        X = rng.randn(seq_len, hidden_dim)
        grad_Y = rng.randn(seq_len, hidden_dim)

        ring_model = MultiHeadRingAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_ranks=num_ranks,
            causal=True,
            rng=rng,
        )

        mono_model = MonolithicAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            causal=True,
            W_q=ring_model.W_q,
            W_k=ring_model.W_k,
            W_v=ring_model.W_v,
            W_o=ring_model.W_o,
        )

        # Forward
        X_shards = [X[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]
        grad_Y_shards = [grad_Y[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]

        Y_shards, ring_cache = ring_model.forward(X_shards)
        Y_ring = np.concatenate(Y_shards, axis=0)
        Y_mono, mono_cache = mono_model.forward(X)

        np.testing.assert_allclose(Y_ring, Y_mono, rtol=1e-10, atol=1e-10)

        # Backward
        grad_X_shards = ring_model.backward(grad_Y_shards, ring_cache)
        grad_X_ring = np.concatenate(grad_X_shards, axis=0)
        grad_X_mono = mono_model.backward(grad_Y, mono_cache)

        np.testing.assert_allclose(grad_X_ring, grad_X_mono, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ring_model.grad_W_q, mono_model.grad_W_q, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ring_model.grad_W_k, mono_model.grad_W_k, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ring_model.grad_W_v, mono_model.grad_W_v, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ring_model.grad_W_o, mono_model.grad_W_o, rtol=1e-10, atol=1e-10)

    def test_non_causal_multi_head_forward_and_backward(self):
        rng = np.random.RandomState(456)
        seq_len = 24
        num_ranks = 3
        hidden_dim = 12
        num_heads = 3
        block_len = seq_len // num_ranks

        X = rng.randn(seq_len, hidden_dim)
        grad_Y = rng.randn(seq_len, hidden_dim)

        ring_model = MultiHeadRingAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_ranks=num_ranks,
            causal=False,
            rng=rng,
        )

        mono_model = MonolithicAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            causal=False,
            W_q=ring_model.W_q,
            W_k=ring_model.W_k,
            W_v=ring_model.W_v,
            W_o=ring_model.W_o,
        )

        X_shards = [X[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]
        grad_Y_shards = [grad_Y[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]

        Y_shards, ring_cache = ring_model.forward(X_shards)
        Y_ring = np.concatenate(Y_shards, axis=0)
        Y_mono, mono_cache = mono_model.forward(X)

        np.testing.assert_allclose(Y_ring, Y_mono, rtol=1e-10, atol=1e-10)

        grad_X_shards = ring_model.backward(grad_Y_shards, ring_cache)
        grad_X_ring = np.concatenate(grad_X_shards, axis=0)
        grad_X_mono = mono_model.backward(grad_Y, mono_cache)

        np.testing.assert_allclose(grad_X_ring, grad_X_mono, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ring_model.grad_W_q, mono_model.grad_W_q, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ring_model.grad_W_k, mono_model.grad_W_k, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ring_model.grad_W_v, mono_model.grad_W_v, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ring_model.grad_W_o, mono_model.grad_W_o, rtol=1e-10, atol=1e-10)


class TestRingAttentionCLI(unittest.TestCase):
    def test_cli_verify(self):
        res = subprocess.run(
            [sys.executable, str(CLI), "verify", "--ranks", "3", "--seq-len", "24", "--hidden", "12", "--heads", "3"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("VERIFICATION SUCCESSFUL", res.stdout)

    def test_cli_simulate(self):
        res = subprocess.run(
            [sys.executable, str(CLI), "simulate", "--ranks", "4"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Schedule Matrix", res.stdout)
        self.assertIn("Efficiency Metrics", res.stdout)

    def test_cli_benchmark(self):
        res = subprocess.run(
            [sys.executable, str(CLI), "benchmark", "--ranks", "2", "--hidden", "16", "--heads", "2", "--steps", "1"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Scalability & Context Length Benchmark", res.stdout)

    def test_cli_inspect(self):
        res = subprocess.run(
            [sys.executable, str(CLI), "inspect"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Online Softmax Recurrence", res.stdout)


if __name__ == "__main__":
    unittest.main()
