"""Comprehensive Unit and Integration Test Suite for ZeRO Optimization Engine."""

from __future__ import annotations

import subprocess
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "zero-optimizer"

sys.path.insert(0, str(ROOT / "bin"))
zero_module = SourceFileLoader("zero_optimizer", str(CLI)).load_module()

ZeROCommunicator = zero_module.ZeROCommunicator
MonolithicAdamW = zero_module.MonolithicAdamW
ZeROStage1Optimizer = zero_module.ZeROStage1Optimizer
ZeROStage2Optimizer = zero_module.ZeROStage2Optimizer
ZeROStage3Optimizer = zero_module.ZeROStage3Optimizer
ZeROOffloadOptimizer = zero_module.ZeROOffloadOptimizer
AdamWConfig = zero_module.AdamWConfig
SyntheticMLP = zero_module.SyntheticMLP


class TestZeROCommunicator(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)

    def test_invalid_world_size(self):
        with self.assertRaises(ValueError):
            ZeROCommunicator(world_size=0)
        with self.assertRaises(ValueError):
            ZeROCommunicator(world_size=-2)

    def test_all_reduce_sum(self):
        comm = ZeROCommunicator(world_size=4)
        tensors = [self.rng.randn(3, 5) for _ in range(4)]
        ref = sum(tensors)
        out = comm.all_reduce(tensors, op="sum")
        self.assertEqual(len(out), 4)
        for r in out:
            np.testing.assert_allclose(r, ref, atol=1e-14)
        self.assertEqual(comm.telemetry.all_reduce_calls, 1)
        self.assertGreater(comm.telemetry.all_reduce_bytes, 0)

    def test_all_reduce_mean(self):
        comm = ZeROCommunicator(world_size=2)
        t1 = np.array([[2.0, 4.0]])
        t2 = np.array([[6.0, 8.0]])
        ref = np.array([[4.0, 6.0]])
        out = comm.all_reduce([t1, t2], op="mean")
        for r in out:
            np.testing.assert_allclose(r, ref, atol=1e-14)

    def test_reduce_scatter(self):
        comm = ZeROCommunicator(world_size=2)
        t1 = np.array([1.0, 2.0, 3.0, 4.0])
        t2 = np.array([5.0, 6.0, 7.0, 8.0])
        shards = comm.reduce_scatter([t1, t2], op="sum")
        self.assertEqual(len(shards), 2)
        np.testing.assert_allclose(shards[0], np.array([6.0, 8.0]), atol=1e-14)
        np.testing.assert_allclose(shards[1], np.array([10.0, 12.0]), atol=1e-14)
        self.assertEqual(comm.telemetry.reduce_scatter_calls, 1)

    def test_all_gather(self):
        comm = ZeROCommunicator(world_size=2)
        s1 = np.array([1.0, 2.0])
        s2 = np.array([3.0, 4.0])
        gathered = comm.all_gather([s1, s2], original_shape=(2, 2))
        ref = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(gathered), 2)
        for g in gathered:
            np.testing.assert_allclose(g, ref, atol=1e-14)
        self.assertEqual(comm.telemetry.all_gather_calls, 1)


class TestMonolithicAdamW(unittest.TestCase):
    def test_single_step(self):
        p = np.array([1.0, 2.0], dtype=np.float64)
        g = np.array([0.1, -0.2], dtype=np.float64)
        opt = MonolithicAdamW([p], AdamWConfig(lr=0.1, weight_decay=0.0))
        opt.step([g])
        self.assertEqual(opt.step_num, 1)
        self.assertFalse(np.allclose(p, [1.0, 2.0]))


class TestZeROStagesEquivalence(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(123)
        self.din, self.dhidden, self.dout = 10, 16, 4
        self.batch_size = 6
        self.world_size = 4
        self.steps = 5

        self.W1 = self.rng.randn(self.din, self.dhidden) * 0.1
        self.b1 = self.rng.randn(self.dhidden) * 0.1
        self.W2 = self.rng.randn(self.dhidden, self.dout) * 0.1
        self.b2 = self.rng.randn(self.dout) * 0.1

        self.rank_X = [self.rng.randn(self.batch_size, self.din) for _ in range(self.world_size)]
        self.rank_Y = [self.rng.randn(self.batch_size, self.dout) for _ in range(self.world_size)]

    def _run_mono(self):
        m = SyntheticMLP(self.din, self.dhidden, self.dout)
        m.W1[:] = np.copy(self.W1)
        m.b1[:] = np.copy(self.b1)
        m.W2[:] = np.copy(self.W2)
        m.b2[:] = np.copy(self.b2)
        opt = MonolithicAdamW(m.params, AdamWConfig(lr=0.01, weight_decay=0.01))
        for _ in range(self.steps):
            accum_grads = [np.zeros_like(p) for p in m.params]
            for r in range(self.world_size):
                pred, a1, z1 = m.forward(self.rank_X[r])
                _, g = m.backward(self.rank_X[r], self.rank_Y[r], pred, a1, z1)
                for i in range(len(accum_grads)):
                    accum_grads[i] += g[i] / self.world_size
            opt.step(accum_grads)
        return m.params

    def test_zero_stage1_equivalence(self):
        mono_params = self._run_mono()
        comm = ZeROCommunicator(self.world_size)
        models = [SyntheticMLP(self.din, self.dhidden, self.dout) for _ in range(self.world_size)]
        opts = []
        for r in range(self.world_size):
            models[r].W1[:] = np.copy(self.W1)
            models[r].b1[:] = np.copy(self.b1)
            models[r].W2[:] = np.copy(self.W2)
            models[r].b2[:] = np.copy(self.b2)
            opts.append(ZeROStage1Optimizer(r, self.world_size, models[r].params, AdamWConfig(lr=0.01, weight_decay=0.01), comm))

        for _ in range(self.steps):
            step_grads = []
            for r in range(self.world_size):
                pred, a1, z1 = models[r].forward(self.rank_X[r])
                _, g = models[r].backward(self.rank_X[r], self.rank_Y[r], pred, a1, z1)
                step_grads.append(g)
            reduced_grads = []
            for p_idx in range(len(models[0].params)):
                all_r_grads = [step_grads[r][p_idx] for r in range(self.world_size)]
                reduced_grads.append(comm.all_reduce(all_r_grads, op="mean")[0])
            ZeROStage1Optimizer.step_all(opts, reduced_grads)

        for p_mono, p_s1 in zip(mono_params, models[0].params):
            np.testing.assert_allclose(p_s1, p_mono, atol=1e-12)

    def test_zero_stage2_equivalence(self):
        mono_params = self._run_mono()
        comm = ZeROCommunicator(self.world_size)
        models = [SyntheticMLP(self.din, self.dhidden, self.dout) for _ in range(self.world_size)]
        opts = []
        for r in range(self.world_size):
            models[r].W1[:] = np.copy(self.W1)
            models[r].b1[:] = np.copy(self.b1)
            models[r].W2[:] = np.copy(self.W2)
            models[r].b2[:] = np.copy(self.b2)
            opts.append(ZeROStage2Optimizer(r, self.world_size, models[r].params, AdamWConfig(lr=0.01, weight_decay=0.01), comm))

        for _ in range(self.steps):
            step_grads = []
            for r in range(self.world_size):
                pred, a1, z1 = models[r].forward(self.rank_X[r])
                _, g = models[r].backward(self.rank_X[r], self.rank_Y[r], pred, a1, z1)
                step_grads.append(g)
            ZeROStage2Optimizer.step_all(opts, step_grads)

        for p_mono, p_s2 in zip(mono_params, models[0].params):
            np.testing.assert_allclose(p_s2, p_mono, atol=1e-12)

    def test_zero_stage3_equivalence(self):
        mono_params = self._run_mono()
        comm = ZeROCommunicator(self.world_size)
        shapes = [self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape]
        init_p = [self.W1, self.b1, self.W2, self.b2]
        opts = [
            ZeROStage3Optimizer(r, self.world_size, shapes, init_p, AdamWConfig(lr=0.01, weight_decay=0.01), comm)
            for r in range(self.world_size)
        ]

        for _ in range(self.steps):
            step_grads = []
            for r in range(self.world_size):
                w1, b1, w2, b2 = opts[r].fetch_full_parameters(opts)
                z1 = self.rank_X[r] @ w1 + b1
                a1 = np.maximum(0.0, z1)
                pred = a1 @ w2 + b2
                B = self.rank_X[r].shape[0]
                dpred = (2.0 / (B * pred.shape[1])) * (pred - self.rank_Y[r])
                dW2 = a1.T @ dpred
                db2 = np.sum(dpred, axis=0)
                dA1 = dpred @ w2.T
                dZ1 = dA1 * (z1 > 0).astype(np.float64)
                dW1 = self.rank_X[r].T @ dZ1
                db1 = np.sum(dZ1, axis=0)
                step_grads.append([dW1, db1, dW2, db2])
            ZeROStage3Optimizer.step_all(opts, step_grads)

        s3_final = opts[0].fetch_full_parameters(opts)
        for p_mono, p_s3 in zip(mono_params, s3_final):
            np.testing.assert_allclose(p_s3, p_mono, atol=1e-12)

    def test_zero_offload_equivalence(self):
        mono_params = self._run_mono()
        model = SyntheticMLP(self.din, self.dhidden, self.dout)
        model.W1[:] = np.copy(self.W1)
        model.b1[:] = np.copy(self.b1)
        model.W2[:] = np.copy(self.W2)
        model.b2[:] = np.copy(self.b2)
        offload = ZeROOffloadOptimizer(model.params, AdamWConfig(lr=0.01, weight_decay=0.01))

        for _ in range(self.steps):
            accum_grads = [np.zeros_like(p) for p in model.params]
            for r in range(self.world_size):
                pred, a1, z1 = model.forward(self.rank_X[r])
                _, g = model.backward(self.rank_X[r], self.rank_Y[r], pred, a1, z1)
                for i in range(len(accum_grads)):
                    accum_grads[i] += g[i] / self.world_size
            offload.step(accum_grads)

        for p_mono, p_off in zip(mono_params, model.params):
            np.testing.assert_allclose(p_off, p_mono, atol=1e-12)


class TestZeroCLI(unittest.TestCase):
    def test_cli_help(self):
        res = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("zero-optimizer", res.stdout)

    def test_cli_status(self):
        res = subprocess.run([sys.executable, str(CLI), "status"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ZeRO: Zero Redundancy Optimizer", res.stdout)

    def test_cli_bench(self):
        res = subprocess.run([sys.executable, str(CLI), "bench", "--world-size", "8"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("LLaMA-3.1-405B", res.stdout)

    def test_cli_verify(self):
        res = subprocess.run([sys.executable, str(CLI), "verify", "--world-size", "2", "--steps", "3"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("All ZeRO mathematical equivalence invariants PASSED", res.stdout)


if __name__ == "__main__":
    unittest.main()
