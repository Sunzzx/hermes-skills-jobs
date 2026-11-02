"""
Comprehensive Unit and Integration Test Suite for Sparse MoE Routing Engine.
"""

from importlib.machinery import SourceFileLoader
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "sparse-moe"

loader = SourceFileLoader("sparse_moe", str(BIN_PATH))
moe_module = types.ModuleType(loader.name)
loader.exec_module(moe_module)

MathOps = moe_module.MathOps
MoEConfig = moe_module.MoEConfig
TopKGatingRouter = moe_module.TopKGatingRouter
SwiGLUExpert = moe_module.SwiGLUExpert
SparseMoELayer = moe_module.SparseMoELayer
SyntheticDataGenerator = moe_module.SyntheticDataGenerator


class TestMathOps(unittest.TestCase):
    def test_dot_product(self):
        u = [1.0, 2.0, 3.0]
        v = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(MathOps.dot(u, v), 32.0)

    def test_norm_and_normalize(self):
        u = [3.0, 4.0]
        self.assertAlmostEqual(MathOps.norm(u), 5.0)
        normed = MathOps.normalize(u)
        self.assertAlmostEqual(normed[0], 0.6)
        self.assertAlmostEqual(normed[1], 0.8)
        self.assertAlmostEqual(MathOps.norm(normed), 1.0)

    def test_softplus(self):
        self.assertAlmostEqual(MathOps.softplus(0.0), math.log(2.0), places=5)
        self.assertAlmostEqual(MathOps.softplus(50.0), 50.0, places=4)
        self.assertGreater(MathOps.softplus(-50.0), 0.0)

    def test_silu(self):
        self.assertAlmostEqual(MathOps.silu(0.0), 0.0)
        self.assertAlmostEqual(MathOps.silu(20.0), 20.0, places=4)
        self.assertAlmostEqual(MathOps.silu(-20.0), 0.0, places=4)

    def test_softmax_sum_and_monotonicity(self):
        logits = [2.0, 1.0, 0.1, -1.0]
        probs = MathOps.softmax(logits)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])
        self.assertGreater(probs[2], probs[3])

    def test_top_k(self):
        vals = [10.0, 50.0, 20.0, 40.0, 30.0]
        indices, top_vals = MathOps.top_k(vals, 3)
        self.assertEqual(indices, [1, 3, 4])
        self.assertEqual(top_vals, [50.0, 40.0, 30.0])

    def test_gini_coefficient(self):
        self.assertAlmostEqual(MathOps.gini([10.0, 10.0, 10.0, 10.0]), 0.0, places=4)
        self.assertGreater(MathOps.gini([100.0, 0.0, 0.0, 0.0]), 0.6)

    def test_entropy(self):
        probs = [0.25, 0.25, 0.25, 0.25]
        self.assertAlmostEqual(MathOps.entropy(probs), 2.0, places=5)


class TestTopKGatingRouter(unittest.TestCase):
    def setUp(self):
        self.config = MoEConfig(
            d_model=32,
            d_ffn=64,
            num_experts=8,
            top_k=2,
            capacity_factor=1.25,
            aux_loss_weight=0.01,
            z_loss_weight=0.001,
            router_noise=0.05,
            seed=42,
        )
        self.router = TopKGatingRouter(self.config)

    def test_router_output_structure(self):
        tokens = [[0.1] * 32 for _ in range(16)]
        res = self.router.forward(tokens)
        self.assertEqual(res.total_tokens, 16)
        self.assertEqual(len(res.token_routes), 16)
        for routes in res.token_routes:
            self.assertLessEqual(len(routes), 2)
            if routes:
                weights = [w for _, w in routes]
                self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_router_losses_positive(self):
        tokens = SyntheticDataGenerator.generate_clustered_tokens(32, 32, num_clusters=4)
        res = self.router.forward(tokens, training=True)
        self.assertGreater(res.aux_loss, 0.0)
        self.assertGreater(res.z_loss, 0.0)
        self.assertFalse(math.isnan(res.aux_loss))
        self.assertFalse(math.isnan(res.z_loss))

    def test_capacity_overflow_dropping(self):
        tight_cfg = MoEConfig(
            d_model=16,
            num_experts=4,
            top_k=2,
            capacity_factor=0.3,
            seed=123,
        )
        tight_router = TopKGatingRouter(tight_cfg)
        tokens = [[1.0] * 16 for _ in range(40)]
        res = tight_router.forward(tokens)
        self.assertGreater(res.dropped_tokens, 0)
        overflow_sum = sum(res.expert_overflow_counts.values())
        self.assertGreater(overflow_sum, 0)


class TestSwiGLUExpert(unittest.TestCase):
    def test_expert_forward(self):
        expert = SwiGLUExpert(expert_id=1, d_model=16, d_ffn=32, seed=99)
        x = [0.5] * 16
        out = expert.forward(x)
        self.assertEqual(len(out), 16)
        self.assertEqual(expert.tokens_processed, 1)
        self.assertGreater(expert.activation_energy, 0.0)


class TestSparseMoELayer(unittest.TestCase):
    def test_layer_residual_shape(self):
        cfg = MoEConfig(
            d_model=16,
            d_ffn=32,
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            seed=77,
        )
        layer = SparseMoELayer(cfg)
        tokens = [[0.2] * 16 for _ in range(8)]
        outputs, res = layer.forward(tokens)
        self.assertEqual(len(outputs), 8)
        self.assertEqual(len(outputs[0]), 16)
        for out in outputs:
            self.assertFalse(any(math.isnan(v) for v in out))

    def test_shared_expert_inclusion(self):
        cfg = MoEConfig(d_model=16, d_ffn=32, num_experts=4, num_shared_experts=2)
        layer = SparseMoELayer(cfg)
        self.assertEqual(len(layer.shared_experts), 2)
        tokens = [[0.1] * 16 for _ in range(4)]
        outputs, _ = layer.forward(tokens)
        self.assertEqual(len(outputs), 4)


class TestCLIExecution(unittest.TestCase):
    def test_cli_verify(self):
        cmd = [sys.executable, str(BIN_PATH), "verify"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("All 5/5 Mathematical Invariant Tests Verified!", res.stdout)

    def test_cli_route_json(self):
        cmd = [
            sys.executable,
            str(BIN_PATH),
            "route",
            "--tokens", "16",
            "--d-model", "16",
            "--d-ffn", "32",
            "--num-experts", "4",
            "--top-k", "2",
            "--json",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("aux_loss", res.stdout)

    def test_cli_benchmark(self):
        cmd = [
            sys.executable,
            str(BIN_PATH),
            "benchmark",
            "--tokens", "32",
            "--d-model", "16",
            "--d-ffn", "32",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Throughput", res.stdout)

    def test_cli_balance(self):
        cmd = [
            sys.executable,
            str(BIN_PATH),
            "balance",
            "--tokens", "32",
            "--d-model", "16",
            "--num-experts", "4",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Gini Index", res.stdout)


if __name__ == "__main__":
    unittest.main()
