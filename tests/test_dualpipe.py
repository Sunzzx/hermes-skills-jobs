"""Comprehensive Unit and Integration Test Suite for DualPipe Bidirectional Pipeline Parallelism Engine."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import os
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "dualpipe"

# Dynamic import of dualpipe module
spec = importlib.util.spec_from_loader("dualpipe", SourceFileLoader("dualpipe", str(CLI)))
assert spec is not None and spec.loader is not None
dualpipe = importlib.util.module_from_spec(spec)
sys.modules["dualpipe"] = dualpipe
spec.loader.exec_module(dualpipe)

DifferentiableLayer = dualpipe.DifferentiableLayer
DualPipeStage = dualpipe.DualPipeStage
DualPipeCommunicator = dualpipe.DualPipeCommunicator
DualPipeSchedule = dualpipe.DualPipeSchedule
DualPipeEngine = dualpipe.DualPipeEngine
MonolithicReferenceModel = dualpipe.MonolithicReferenceModel
gelu = dualpipe.gelu
dgelu = dualpipe.dgelu


class TestMathematicalActivations(unittest.TestCase):
    """Tests GELU activation function and analytical derivative."""

    def test_gelu_values(self):
        x = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
        out = gelu(x)
        self.assertAlmostEqual(float(out[1]), 0.0, places=7)
        self.assertLess(float(out[0]), 0.0)
        self.assertGreater(float(out[2]), 1.9)

    def test_dgelu_numerical_gradient(self):
        eps = 1e-6
        x = np.linspace(-3.0, 3.0, 25, dtype=np.float64)
        analytical = dgelu(x)
        numerical = (gelu(x + eps) - gelu(x - eps)) / (2.0 * eps)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5, rtol=1e-5)


class TestDifferentiableLayer(unittest.TestCase):
    """Tests layer forward pass and split B_X / B_W equivalence."""

    def test_split_backward_matches_unified(self):
        rng = np.random.RandomState(42)
        din, dout = 16, 16
        batch_size = 8

        l_unified = DifferentiableLayer(din, dout, use_gelu=True, rng=rng)
        l_split = DifferentiableLayer(din, dout, use_gelu=True, rng=rng)
        l_split.W = l_unified.W.copy()
        l_split.b = l_unified.b.copy()

        x = rng.randn(batch_size, din).astype(np.float64)
        grad_out = rng.randn(batch_size, dout).astype(np.float64)

        # Unified backward
        act_u, cache_u = l_unified.forward(x)
        grad_in_u = l_unified.backward(grad_out, cache_u)

        # Split backward
        act_s, cache_s = l_split.forward(x)
        grad_in_s, bw_ctx = l_split.backward_input(grad_out, cache_s)
        l_split.backward_weight(bw_ctx)

        np.testing.assert_allclose(act_u, act_s, atol=1e-12)
        np.testing.assert_allclose(grad_in_u, grad_in_s, atol=1e-12)
        np.testing.assert_allclose(l_unified.grad_W, l_split.grad_W, atol=1e-12)
        np.testing.assert_allclose(l_unified.grad_b, l_split.grad_b, atol=1e-12)


class TestDualPipeSchedule(unittest.TestCase):
    """Tests theoretical bubble formulas and memory bounds."""

    def test_bubble_ratio_formulas(self):
        P = 8
        M = 32
        gpipe_b = DualPipeSchedule.bubble_ratio_gpipe(P, M)
        f1b_b = DualPipeSchedule.bubble_ratio_1f1b(P, M)
        interleaved_b = DualPipeSchedule.bubble_ratio_interleaved(P, M, v=2)
        dualpipe_b = DualPipeSchedule.bubble_ratio_dualpipe(P, M)

        self.assertAlmostEqual(gpipe_b, 7.0 / 39.0, places=6)
        self.assertAlmostEqual(f1b_b, 7.0 / 39.0, places=6)
        self.assertAlmostEqual(interleaved_b, 7.0 / 71.0, places=6)
        self.assertAlmostEqual(dualpipe_b, 7.0 / 64.0, places=6)
        self.assertLess(dualpipe_b, f1b_b)

    def test_ascii_gantt_generation(self):
        chart = DualPipeSchedule.generate_ascii_gantt(num_stages=4, num_microbatches=8)
        self.assertIn("DualPipe Bidirectional Gantt Chart", chart)
        self.assertIn("Rank  0", chart)
        self.assertIn("Rank  3", chart)


class TestDualPipeEngineEquivalence(unittest.TestCase):
    """Tests end-to-end mathematical gradient and loss equivalence against monolithic baseline."""

    def test_full_pipeline_gradient_equivalence(self):
        P = 4
        M = 8
        B = 32
        din, dout = 16, 16
        rng = np.random.RandomState(1337)

        all_layers = [
            DifferentiableLayer(din, dout, name=f"layer_{i}", use_gelu=True, rng=rng)
            for i in range(2 * P)
        ]
        saved_W = [l.W.copy() for l in all_layers]
        saved_b = [l.b.copy() for l in all_layers]

        X = rng.randn(B, din).astype(np.float64)
        Y = rng.randn(B, dout).astype(np.float64)

        # 1. Monolithic run
        mono = MonolithicReferenceModel(all_layers)
        mono_loss = mono.forward_backward_epoch(X, Y, M)
        mono_grad_W = [l.grad_W.copy() for l in all_layers]
        mono_grad_b = [l.grad_b.copy() for l in all_layers]

        # 2. Reset gradients
        for l in all_layers:
            l.zero_grad()

        # 3. DualPipe run
        stages = []
        for p in range(P):
            c0 = all_layers[p]
            c1 = all_layers[2 * P - 1 - p]
            stages.append(DualPipeStage(p, P, [c0], [c1]))

        engine = DualPipeEngine(stages)
        dualpipe_loss, telemetry = engine.forward_backward_epoch(X, Y, M)

        # Assert loss match
        self.assertAlmostEqual(mono_loss, dualpipe_loss, places=12)

        # Assert all parameter gradients match to 1e-12
        for idx, layer in enumerate(all_layers):
            np.testing.assert_allclose(layer.grad_W, mono_grad_W[idx], atol=1e-12)
            np.testing.assert_allclose(layer.grad_b, mono_grad_b[idx], atol=1e-12)

        # Overlap telemetry checks
        comm = telemetry["comm"]
        self.assertGreater(comm.bytes_transferred, 0)
        self.assertGreater(comm.p2p_send_calls, 0)
        self.assertGreaterEqual(comm.hidden_comm_time_ms, 0.0)


class TestDualPipeCLI(unittest.TestCase):
    """Tests the CLI entrypoint commands."""

    def test_cli_help(self):
        res = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("DualPipe Bidirectional Pipeline Parallelism Engine", res.stdout)

    def test_cli_status(self):
        res = subprocess.run([sys.executable, str(CLI), "status"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("DualPipe: Bidirectional Pipeline Parallelism", res.stdout)
        self.assertIn("Theoretical Bubble", res.stdout)

    def test_cli_schedule(self):
        res = subprocess.run([sys.executable, str(CLI), "schedule", "-p", "4", "-m", "8"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Gantt Chart", res.stdout)

    def test_cli_bench(self):
        res = subprocess.run([sys.executable, str(CLI), "bench", "-p", "4", "-m", "16"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Performance & Bubble Benchmark", res.stdout)

    def test_cli_verify(self):
        res = subprocess.run([sys.executable, str(CLI), "verify", "-p", "2", "-m", "4", "-b", "16"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ALL DUALPIPE MATHEMATICAL INVARIANTS RIGOROUSLY PASSED", res.stdout)


if __name__ == "__main__":
    unittest.main()
