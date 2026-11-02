"""Comprehensive Unit and Integration Test Suite for Megatron-LM 1D Tensor Parallelism Engine."""

from __future__ import annotations

import subprocess
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "tensor-parallel"

sys.path.insert(0, str(ROOT / "bin"))
tp_module = SourceFileLoader("tensor_parallel", str(CLI)).load_module()

TensorParallelCommunicator = tp_module.TensorParallelCommunicator
ColumnParallelLinear = tp_module.ColumnParallelLinear
RowParallelLinear = tp_module.RowParallelLinear
ParallelMLP = tp_module.ParallelMLP
ParallelSelfAttention = tp_module.ParallelSelfAttention
VocabParallelCrossEntropy = tp_module.VocabParallelCrossEntropy
MonolithicTransformerBlock = tp_module.MonolithicTransformerBlock
gelu = tp_module.gelu
dgelu = tp_module.dgelu
softmax = tp_module.softmax


class TestTensorParallelCommunicator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_invalid_world_size(self):
        with self.assertRaises(ValueError):
            TensorParallelCommunicator(world_size=3)
        with self.assertRaises(ValueError):
            TensorParallelCommunicator(world_size=0)

    def test_all_reduce_sum(self):
        comm = TensorParallelCommunicator(world_size=4)
        tensors = [self.rng.randn(2, 4) for _ in range(4)]
        ref = sum(tensors)
        out = comm.all_reduce(tensors, op="sum")
        self.assertEqual(len(out), 4)
        for r in out:
            np.testing.assert_allclose(r, ref, atol=1e-14)
        self.assertEqual(comm.telemetry.all_reduce_calls, 1)
        self.assertGreater(comm.telemetry.all_reduce_bytes, 0)

    def test_all_reduce_max(self):
        comm = TensorParallelCommunicator(world_size=2)
        t1 = np.array([[1.0, 5.0], [3.0, 2.0]])
        t2 = np.array([[4.0, 2.0], [0.0, 7.0]])
        ref = np.maximum(t1, t2)
        out = comm.all_reduce([t1, t2], op="max")
        for r in out:
            np.testing.assert_allclose(r, ref, atol=1e-14)

    def test_all_gather(self):
        comm = TensorParallelCommunicator(world_size=2)
        t1 = np.array([[1.0, 2.0]])
        t2 = np.array([[3.0, 4.0]])
        gathered = comm.all_gather([t1, t2], axis=-1)
        ref = np.array([[1.0, 2.0, 3.0, 4.0]])
        for g in gathered:
            np.testing.assert_allclose(g, ref, atol=1e-14)
        self.assertEqual(comm.telemetry.all_gather_calls, 1)

    def test_reduce_scatter(self):
        comm = TensorParallelCommunicator(world_size=2)
        t1 = np.array([[1.0, 2.0]])
        t2 = np.array([[3.0, 5.0]])
        scattered = comm.reduce_scatter([t1, t2], axis=-1)
        self.assertEqual(len(scattered), 2)
        np.testing.assert_allclose(scattered[0], np.array([[4.0]]), atol=1e-14)
        np.testing.assert_allclose(scattered[1], np.array([[7.0]]), atol=1e-14)


class TestColumnParallelLinear(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(101)

    def test_forward_backward_ungathered(self):
        tp = 2
        B, S, Din, Dout = 2, 3, 16, 32
        layer = ColumnParallelLinear(Din, Dout, world_size=tp, gather_output=False, rng=self.rng)
        W_ref = self.rng.randn(Din, Dout)
        b_ref = self.rng.randn(Dout)
        layer.init_from_dense(W_ref, b_ref)

        X = self.rng.randn(B, S, Din)
        dY = self.rng.randn(B, S, Dout)
        dY_splits = np.split(dY, tp, axis=-1)

        Y_ref = MonolithicTransformerBlock.linear_forward(X, W_ref, b_ref)
        dW_ref, db_ref, dX_ref = MonolithicTransformerBlock.linear_backward(dY, X, W_ref)

        Y_tp = layer.forward(X)
        Y_tp_cat = np.concatenate(Y_tp, axis=-1)
        dW_tp, db_tp, dX_tp = layer.backward(dY_splits, X)
        dW_tp_cat = np.concatenate(dW_tp, axis=1)
        db_tp_cat = np.concatenate(db_tp, axis=0)

        np.testing.assert_allclose(Y_tp_cat, Y_ref, atol=1e-13)
        np.testing.assert_allclose(dW_tp_cat, dW_ref, atol=1e-13)
        np.testing.assert_allclose(db_tp_cat, db_ref, atol=1e-13)
        np.testing.assert_allclose(dX_tp[0], dX_ref, atol=1e-13)

    def test_forward_gathered(self):
        tp = 2
        Din, Dout = 8, 16
        layer = ColumnParallelLinear(Din, Dout, world_size=tp, gather_output=True, rng=self.rng)
        W_ref = self.rng.randn(Din, Dout)
        b_ref = self.rng.randn(Dout)
        layer.init_from_dense(W_ref, b_ref)

        X = self.rng.randn(2, 4, Din)
        Y_ref = MonolithicTransformerBlock.linear_forward(X, W_ref, b_ref)
        Y_tp = layer.forward(X)
        for y in Y_tp:
            np.testing.assert_allclose(y, Y_ref, atol=1e-13)


class TestRowParallelLinear(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(202)

    def test_forward_backward(self):
        tp = 2
        B, S, Din, Dout = 2, 3, 32, 16
        layer = RowParallelLinear(Din, Dout, world_size=tp, input_is_parallel=True, rng=self.rng)
        W_ref = self.rng.randn(Din, Dout)
        b_ref = self.rng.randn(Dout)
        layer.init_from_dense(W_ref, b_ref)

        X = self.rng.randn(B, S, Din)
        X_splits = np.split(X, tp, axis=-1)
        dY = self.rng.randn(B, S, Dout)
        dY_list = [np.copy(dY) for _ in range(tp)]

        Y_ref = MonolithicTransformerBlock.linear_forward(X, W_ref, b_ref)
        dW_ref, db_ref, dX_ref = MonolithicTransformerBlock.linear_backward(dY, X, W_ref)

        Y_tp = layer.forward(X_splits)
        dW_tp, db_tp, dX_tp = layer.backward(dY_list, X_splits)
        dW_tp_cat = np.concatenate(dW_tp, axis=0)
        dX_tp_cat = np.concatenate(dX_tp, axis=-1)

        np.testing.assert_allclose(Y_tp[0], Y_ref, atol=1e-13)
        np.testing.assert_allclose(dW_tp_cat, dW_ref, atol=1e-13)
        np.testing.assert_allclose(db_tp, db_ref, atol=1e-13)
        np.testing.assert_allclose(dX_tp_cat, dX_ref, atol=1e-13)


class TestParallelMLPAndAttention(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(303)

    def test_mlp_forward_backward(self):
        tp = 2
        B, S, D, FFN = 2, 4, 16, 64
        mlp = ParallelMLP(D, FFN, world_size=tp, rng=self.rng)
        W1 = self.rng.randn(D, FFN)
        b1 = self.rng.randn(FFN)
        W2 = self.rng.randn(FFN, D)
        b2 = self.rng.randn(D)
        mlp.init_from_dense(W1, b1, W2, b2)

        X = self.rng.randn(B, S, D)
        dY = self.rng.randn(B, S, D)
        dY_list = [np.copy(dY) for _ in range(tp)]

        Y_ref, H_ref, A_ref = MonolithicTransformerBlock.mlp_forward(X, W1, b1, W2, b2)
        dW2_ref, db2_ref, dA_ref = MonolithicTransformerBlock.linear_backward(dY, A_ref, W2)
        dH_ref = dA_ref * dgelu(H_ref)
        dW1_ref, db1_ref, dX_ref = MonolithicTransformerBlock.linear_backward(dH_ref, X, W1)

        Y_tp, H_tp, A_tp = mlp.forward(X)
        dW1_tp, db1_tp, dW2_tp, db2_tp, dX_tp = mlp.backward(dY_list, X, H_tp, A_tp)

        np.testing.assert_allclose(Y_tp[0], Y_ref, atol=1e-13)
        np.testing.assert_allclose(dX_tp[0], dX_ref, atol=1e-13)

    def test_attention_forward(self):
        tp = 2
        B, S, D, heads = 2, 4, 16, 4
        attn = ParallelSelfAttention(D, heads, world_size=tp, rng=self.rng)
        W_qkv = self.rng.randn(D, 3 * D)
        b_qkv = self.rng.randn(3 * D)
        W_out = self.rng.randn(D, D)
        b_out = self.rng.randn(D)
        attn.init_from_dense(W_qkv, b_qkv, W_out, b_out)

        X = self.rng.randn(B, S, D)
        mask = np.triu(np.full((S, S), -1e9), k=1)

        Y_ref, _ = MonolithicTransformerBlock.attention_forward(X, W_qkv, b_qkv, W_out, b_out, heads, mask)
        Y_tp, _ = attn.forward(X, causal_mask=mask)

        np.testing.assert_allclose(Y_tp[0], Y_ref, atol=1e-13)


class TestVocabParallelCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(404)

    def test_vocab_cross_entropy_exact(self):
        tp = 4
        N_tokens, V = 12, 32
        vocab_ce = VocabParallelCrossEntropy(vocab_size=V, world_size=tp)
        logits = self.rng.randn(N_tokens, V)
        targets = self.rng.randint(0, V, size=(N_tokens,))

        loss_ref, dlogits_ref = MonolithicTransformerBlock.cross_entropy(logits, targets)
        logits_splits = np.split(logits, tp, axis=-1)
        loss_tp, dlogits_splits, _ = vocab_ce.forward(logits_splits, targets)
        dlogits_tp = np.concatenate(dlogits_splits, axis=-1)

        self.assertAlmostEqual(loss_ref, loss_tp, places=12)
        np.testing.assert_allclose(dlogits_tp, dlogits_ref, atol=1e-14)


class TestCLIIntegration(unittest.TestCase):
    def test_cli_status(self):
        res = subprocess.run([sys.executable, str(CLI), "status"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Megatron-LM 1D Tensor Parallel Engine Status", res.stdout)

    def test_cli_verify(self):
        res = subprocess.run([sys.executable, str(CLI), "verify", "--tp", "2"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("All mathematical equivalence invariants PASSED!", res.stdout)

    def test_cli_bench(self):
        res = subprocess.run([sys.executable, str(CLI), "bench"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Benchmark complete", res.stdout)

    def test_cli_demo(self):
        res = subprocess.run([sys.executable, str(CLI), "demo", "--tp", "2"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Megatron-LM 1D Tensor Parallelism Structural Diagram", res.stdout)


if __name__ == "__main__":
    unittest.main()
