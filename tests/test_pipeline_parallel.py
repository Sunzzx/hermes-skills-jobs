"""Comprehensive Unit and Integration Test Suite for Megatron-LM 1F1B Pipeline Parallelism Engine."""

from __future__ import annotations

import subprocess
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "pipeline-parallel"

sys.path.insert(0, str(ROOT / "bin"))
pp_module = SourceFileLoader("pipeline_parallel", str(CLI)).load_module()

gelu = pp_module.gelu
dgelu = pp_module.dgelu
PipelineLayer = pp_module.PipelineLayer
PipelineStage = pp_module.PipelineStage
P2PCommunicator = pp_module.P2PCommunicator
PipelineSchedule = pp_module.PipelineSchedule
PipelineEngine = pp_module.PipelineEngine
MonolithicModel = pp_module.MonolithicModel
build_pipeline_and_monolithic = pp_module.build_pipeline_and_monolithic


class TestPipelinePrimitives(unittest.TestCase):
    def test_gelu_and_dgelu(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        y = gelu(x)
        dy = dgelu(x)
        self.assertAlmostEqual(float(y[2]), 0.0, places=7)
        self.assertAlmostEqual(float(dy[2]), 0.5, places=7)
        self.assertTrue(np.all(dy > -0.2))

    def test_pipeline_layer_forward_backward(self):
        rng = np.random.RandomState(42)
        layer = PipelineLayer(din=8, dout=4, name="l1", use_gelu=True, rng=rng)
        x = rng.randn(2, 8)
        out, cache = layer.forward(x)
        self.assertEqual(out.shape, (2, 4))

        grad_out = np.ones((2, 4), dtype=np.float64)
        grad_in = layer.backward(grad_out, cache)
        self.assertEqual(grad_in.shape, (2, 8))
        self.assertEqual(layer.grad_W.shape, (8, 4))
        self.assertEqual(layer.grad_b.shape, (4,))


class TestP2PCommunicator(unittest.TestCase):
    def test_p2p_send_recv(self):
        comm = P2PCommunicator(num_stages=3)
        t0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        comm.send_forward(from_stage=0, mb_id=0, tensor=t0)
        recv_t0 = comm.recv_forward(to_stage=1, mb_id=0)
        np.testing.assert_allclose(t0, recv_t0)

        t_back = np.array([[5.0, 6.0]])
        comm.send_backward(from_stage=2, mb_id=0, tensor=t_back)
        recv_b = comm.recv_backward(to_stage=1, mb_id=0)
        np.testing.assert_allclose(t_back, recv_b)

        self.assertEqual(comm.telemetry.p2p_send_calls, 2)
        self.assertEqual(comm.telemetry.p2p_recv_calls, 2)
        self.assertGreater(comm.telemetry.p2p_bytes_transferred, 0)

    def test_recv_missing_raises(self):
        comm = P2PCommunicator(num_stages=2)
        with self.assertRaises(RuntimeError):
            comm.recv_forward(to_stage=1, mb_id=99)
        with self.assertRaises(RuntimeError):
            comm.recv_backward(to_stage=0, mb_id=99)


class TestPipelineSchedule(unittest.TestCase):
    def test_bubble_ratio_formulas(self):
        # P=4, M=16 -> (4-1)/(16+4-1) = 3/19
        b_g = PipelineSchedule.bubble_ratio_gpipe(4, 16)
        self.assertAlmostEqual(b_g, 3.0 / 19.0, places=7)

        b_1f1b = PipelineSchedule.bubble_ratio_1f1b(4, 16)
        self.assertAlmostEqual(b_1f1b, 3.0 / 19.0, places=7)

        b_v2 = PipelineSchedule.bubble_ratio_interleaved(4, 16, v=2)
        self.assertAlmostEqual(b_v2, 3.0 / (2 * 16 + 3), places=7)

    def test_1f1b_action_generation(self):
        actions = PipelineSchedule.generate_1f1b_actions(num_stages=4, num_microbatches=8)
        self.assertEqual(len(actions), 4)
        for p in range(4):
            total_f = sum(1 for op, _ in actions[p] if op == "F")
            total_b = sum(1 for op, _ in actions[p] if op == "B")
            self.assertEqual(total_f, 8)
            self.assertEqual(total_b, 8)
            self.assertEqual(len(actions[p]), 16)

        # Stage 0 warmup has 3 forwards (P - 1 - 0)
        self.assertEqual([op for op, _ in actions[0][:3]], ["F", "F", "F"])
        # Stage 3 warmup has 0 forwards, so action #1 is B0 after F0
        self.assertEqual(actions[3][0], ("F", 0))
        self.assertEqual(actions[3][1], ("B", 0))

    def test_gpipe_action_generation(self):
        actions = PipelineSchedule.generate_gpipe_actions(num_stages=4, num_microbatches=6)
        for p in range(4):
            ops = [op for op, _ in actions[p]]
            self.assertEqual(ops, ["F"] * 6 + ["B"] * 6)


class TestPipelineEngineEquivalence(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.B = 32
        self.M = 8
        self.dims = [16, 24, 20, 16, 8]

    def test_2_stage_pipeline_equivalence(self):
        stages, mono_model, all_layers = build_pipeline_and_monolithic(self.dims, num_stages=2, rng_seed=42)
        saved_W = [l.W.copy() for l in all_layers]
        saved_b = [l.b.copy() for l in all_layers]

        X = self.rng.randn(self.B, self.dims[0])
        Y = self.rng.randn(self.B, self.dims[-1])

        mono_loss, mono_gw, mono_gb = mono_model.forward_backward(X, Y)

        for l, w, b in zip(all_layers, saved_W, saved_b):
            l.W[:] = w
            l.b[:] = b

        engine = PipelineEngine(stages, schedule_type="1f1b")
        pipe_loss, telemetry = engine.forward_backward_epoch(X, Y, self.M)

        self.assertAlmostEqual(mono_loss, pipe_loss, places=12)
        for layer, mgw, mgb in zip(all_layers, mono_gw, mono_gb):
            np.testing.assert_allclose(layer.grad_W, mgw, atol=1e-12)
            np.testing.assert_allclose(layer.grad_b, mgb, atol=1e-12)

        # Check peak activation bounds: stage 0 <= 2, stage 1 <= 1
        peak_acts = telemetry["peak_activations_per_stage"]
        self.assertLessEqual(peak_acts[0], 2)
        self.assertLessEqual(peak_acts[1], 1)

    def test_4_stage_pipeline_equivalence(self):
        stages, mono_model, all_layers = build_pipeline_and_monolithic(self.dims, num_stages=4, rng_seed=99)
        saved_W = [l.W.copy() for l in all_layers]
        saved_b = [l.b.copy() for l in all_layers]

        X = self.rng.randn(self.B, self.dims[0])
        Y = self.rng.randn(self.B, self.dims[-1])

        mono_loss, mono_gw, mono_gb = mono_model.forward_backward(X, Y)

        for l, w, b in zip(all_layers, saved_W, saved_b):
            l.W[:] = w
            l.b[:] = b

        engine = PipelineEngine(stages, schedule_type="1f1b")
        pipe_loss, telemetry = engine.forward_backward_epoch(X, Y, self.M)

        self.assertAlmostEqual(mono_loss, pipe_loss, places=12)
        for layer, mgw, mgb in zip(all_layers, mono_gw, mono_gb):
            np.testing.assert_allclose(layer.grad_W, mgw, atol=1e-12)
            np.testing.assert_allclose(layer.grad_b, mgb, atol=1e-12)

        # 1F1B memory bounds
        peak_acts = telemetry["peak_activations_per_stage"]
        for p in range(4):
            self.assertLessEqual(peak_acts[p], 4 - p)

    def test_gpipe_schedule_equivalence(self):
        stages, mono_model, all_layers = build_pipeline_and_monolithic(self.dims, num_stages=4, rng_seed=77)
        X = self.rng.randn(self.B, self.dims[0])
        Y = self.rng.randn(self.B, self.dims[-1])

        mono_loss, mono_gw, mono_gb = mono_model.forward_backward(X, Y)

        engine = PipelineEngine(stages, schedule_type="gpipe")
        gpipe_loss, telemetry = engine.forward_backward_epoch(X, Y, self.M)

        self.assertAlmostEqual(mono_loss, gpipe_loss, places=12)
        for layer, mgw, mgb in zip(all_layers, mono_gw, mono_gb):
            np.testing.assert_allclose(layer.grad_W, mgw, atol=1e-12)
            np.testing.assert_allclose(layer.grad_b, mgb, atol=1e-12)

        # Under GPipe, stage 0 must hold all M microbatches
        peak_acts = telemetry["peak_activations_per_stage"]
        self.assertEqual(peak_acts[0], self.M)


class TestPipelineCLI(unittest.TestCase):
    def test_cli_help(self):
        res = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Megatron-LM 1F1B Pipeline Parallelism", res.stdout)

    def test_cli_status(self):
        res = subprocess.run([sys.executable, str(CLI), "status"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("MEGATRON-LM 1F1B PIPELINE PARALLELISM", res.stdout)

    def test_cli_schedule(self):
        res = subprocess.run([sys.executable, str(CLI), "schedule", "-p", "4", "-m", "8"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Schedule Matrix [1F1B]", res.stdout)
        self.assertIn("Schedule Matrix [GPIPE]", res.stdout)

    def test_cli_bench(self):
        res = subprocess.run([sys.executable, str(CLI), "bench", "-p", "8", "-m", "32"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("LLaMA-3.1-405B", res.stdout)

    def test_cli_verify(self):
        res = subprocess.run([sys.executable, str(CLI), "verify", "-p", "4", "-m", "8"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ALL PIPELINE PARALLELISM MATHEMATICAL INVARIANTS PASSED", res.stdout)


if __name__ == "__main__":
    unittest.main()
