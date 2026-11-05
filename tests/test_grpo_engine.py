"""Unit and integration tests for DeepSeek-R1 Group Relative Policy Optimization (GRPO) Engine."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "grpo-engine"

spec = importlib.util.spec_from_loader("grpo_engine", SourceFileLoader("grpo_engine", str(CLI)))
assert spec is not None and spec.loader is not None
grpo_engine = importlib.util.module_from_spec(spec)
sys.modules["grpo_engine"] = grpo_engine
spec.loader.exec_module(grpo_engine)

compute_group_advantages = grpo_engine.compute_group_advantages
compute_schulman_kl = grpo_engine.compute_schulman_kl
compute_clipped_surrogate_loss = grpo_engine.compute_clipped_surrogate_loss
RuleBasedRewardVerifier = grpo_engine.RuleBasedRewardVerifier
GRPOToyPolicy = grpo_engine.GRPOToyPolicy
GRPORuntime = grpo_engine.GRPORuntime


class TestGroupAdvantages(unittest.TestCase):
    def test_group_advantages_zero_sum(self):
        r = [1.0, 3.0, 5.0, 7.0, 9.0]
        adv = compute_group_advantages(r)
        self.assertEqual(len(adv), 5)
        self.assertAlmostEqual(float(np.sum(adv)), 0.0, places=9)

    def test_group_advantages_unit_std(self):
        r = [10.0, 20.0, 30.0, 40.0, 50.0]
        adv = compute_group_advantages(r)
        self.assertAlmostEqual(float(np.var(adv)), 1.0, places=5)

    def test_identical_rewards_zero_advantages(self):
        r = [42.0, 42.0, 42.0, 42.0]
        adv = compute_group_advantages(r)
        np.testing.assert_array_equal(adv, np.zeros(4))

    def test_empty_rewards(self):
        adv = compute_group_advantages([])
        self.assertEqual(len(adv), 0)

    def test_negative_rewards(self):
        r = [-10.0, -5.0, 0.0, 5.0, 10.0]
        adv = compute_group_advantages(r)
        self.assertAlmostEqual(float(np.sum(adv)), 0.0, places=9)
        self.assertAlmostEqual(float(np.var(adv)), 1.0, places=5)


class TestSchulmanKLDivergence(unittest.TestCase):
    def test_kl_zero_at_identity(self):
        logp = np.array([-0.5, -1.2, -3.4, -0.01])
        kl = compute_schulman_kl(logp, logp)
        np.testing.assert_allclose(kl, np.zeros_like(logp), atol=1e-12)

    def test_kl_strictly_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(5):
            p = rng.uniform(-10, 0, size=50)
            ref = rng.uniform(-10, 0, size=50)
            kl = compute_schulman_kl(p, ref)
            self.assertTrue(np.all(kl >= 0.0))

    def test_kl_monotonic_with_divergence(self):
        ref = np.zeros(1)
        kl_small = compute_schulman_kl(np.array([0.1]), ref)[0]
        kl_large = compute_schulman_kl(np.array([1.0]), ref)[0]
        self.assertGreater(kl_large, kl_small)


class TestSurrogateLossAndClipping(unittest.TestCase):
    def test_surrogate_positive_advantage_clipped(self):
        # ratio = exp(0.5) = 1.6487 > 1.2
        diff = np.array([0.5])
        loss, frac = compute_clipped_surrogate_loss(diff, np.zeros(1), advantage=2.0, epsilon=0.2)
        self.assertAlmostEqual(loss, -2.4, places=4)
        self.assertEqual(frac, 1.0)

    def test_surrogate_negative_advantage_pessimistic(self):
        # ratio = exp(-0.5) = 0.6065 < 0.8
        # min(0.6065 * -2, 0.8 * -2) = min(-1.213, -1.6) = -1.6
        diff = np.array([-0.5])
        loss, frac = compute_clipped_surrogate_loss(diff, np.zeros(1), advantage=-2.0, epsilon=0.2)
        self.assertAlmostEqual(loss, 1.6, places=4)
        self.assertEqual(frac, 1.0)

    def test_surrogate_unclipped_regime(self):
        # ratio = exp(0.05) = 1.0513 in [0.8, 1.2]
        diff = np.array([0.05])
        loss, frac = compute_clipped_surrogate_loss(diff, np.zeros(1), advantage=1.5, epsilon=0.2)
        expected = -float(np.exp(0.05) * 1.5)
        self.assertAlmostEqual(loss, expected, places=4)
        self.assertEqual(frac, 0.0)


class TestRuleBasedRewardVerifier(unittest.TestCase):
    def setUp(self):
        self.v = RuleBasedRewardVerifier()

    def test_format_valid(self):
        res = "<think>\nLet's break this down into algebraic subproblems.\n</think>\n\boxed{24}"
        score = self.v.verify_format(res)
        self.assertEqual(score, 1.0)

    def test_format_missing_tags(self):
        res = "Direct answer without tags: 24"
        self.assertEqual(self.v.verify_format(res), 0.0)

    def test_format_unclosed_tags(self):
        res = "<think>unclosed tags here\boxed{24}"
        self.assertEqual(self.v.verify_format(res), 0.0)

    def test_countdown_valid(self):
        ok, msg = self.v.verify_countdown(24, [4, 6, 8, 2], "\boxed{6 * (8 - 4) = 24}")
        self.assertTrue(ok)

    def test_countdown_wrong_target(self):
        ok, msg = self.v.verify_countdown(24, [4, 6, 8, 2], "\boxed{6 + 8 = 14}")
        self.assertFalse(ok)

    def test_countdown_illegal_numbers(self):
        ok, msg = self.v.verify_countdown(24, [4, 6, 8, 2], "\boxed{12 * 2 = 24}")
        self.assertFalse(ok)

    def test_countdown_reused_number(self):
        ok, msg = self.v.verify_countdown(24, [4, 6, 8, 2], "\boxed{4 * 4 = 16}")
        self.assertFalse(ok)

    def test_boxed_math_exact(self):
        ok, _ = self.v.verify_boxed_math(42, "\boxed{42}")
        self.assertTrue(ok)

    def test_boxed_math_float(self):
        ok, _ = self.v.verify_boxed_math(2.718, "\boxed{2.718}")
        self.assertTrue(ok)

    def test_boxed_math_mismatch(self):
        ok, _ = self.v.verify_boxed_math(100, "\boxed{99}")
        self.assertFalse(ok)


class TestGRPORuntimeAndTraining(unittest.TestCase):
    def test_sample_reasoning_group(self):
        runtime = GRPORuntime(group_size=6, seed=42)
        prob = {"type": "countdown", "target": 24, "numbers": [4, 6, 8, 2], "prompt": "Make 24"}
        group = runtime.sample_reasoning_group(prob)
        self.assertEqual(len(group), 6)
        self.assertAlmostEqual(sum(s.advantage for s in group), 0.0, places=5)
        for s in group:
            self.assertTrue(np.isfinite(s.total_reward))
            self.assertTrue(np.isfinite(s.surrogate_loss))
            self.assertGreaterEqual(s.kl_divergence, 0.0)

    def test_execute_grpo_step(self):
        runtime = GRPORuntime(group_size=4, seed=42)
        prob = {"type": "countdown", "target": 24, "numbers": [4, 6, 8, 2], "prompt": "Make 24"}
        metrics = runtime.execute_grpo_step(prob, lr=1e-3)
        self.assertIn("mean_reward", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("format_score", metrics)
        self.assertIn("surrogate_loss", metrics)
        self.assertIn("kl_loss", metrics)
        self.assertEqual(metrics["group_size"], 4)


class TestCLIIntegration(unittest.TestCase):
    def test_cli_status(self):
        res = subprocess.run([str(CLI), "status"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("DeepSeek-R1 Group Relative Policy Optimization", res.stdout)

    def test_cli_verify(self):
        res = subprocess.run([str(CLI), "verify"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("All 18/18 GRPO Invariant & Numerical Tests PASSED", res.stdout)

    def test_cli_benchmark(self):
        res = subprocess.run([str(CLI), "benchmark"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("DeepSeek-R1 GRPO vs. Standard PPO", res.stdout)


if __name__ == "__main__":
    unittest.main()
