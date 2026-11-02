"""Comprehensive Unit and Integration Test Suite for Process Reward Model (PRM) Engine."""

from importlib.machinery import SourceFileLoader
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "bin" / "prm-engine"

loader = SourceFileLoader("prm_engine", str(BIN_PATH))
prm_mod = types.ModuleType(loader.name)
loader.exec_module(prm_mod)

StepType = prm_mod.StepType
StepClassification = prm_mod.StepClassification
ReasoningStep = prm_mod.ReasoningStep
ReasoningTrace = prm_mod.ReasoningTrace
StepParser = prm_mod.StepParser
MathSymbolicEvaluator = prm_mod.MathSymbolicEvaluator
PRMScorer = prm_mod.PRMScorer
MonteCarloSuperviser = prm_mod.MonteCarloSuperviser
PRMBeamSearch = prm_mod.PRMBeamSearch
SelfConsistencyEngine = prm_mod.SelfConsistencyEngine


class TestStepParser(unittest.TestCase):
    def test_numbered_step_parsing(self):
        text = (
            "Step 1: First identify knowns.\n"
            "Step 2: Calculate intermediate sum 10 + 20 = 30.\n"
            "Step 3: Multiply by 2: 30 * 2 = 60.\n"
            "Step 4: Therefore, the final answer is 60."
        )
        steps = StepParser.parse(text)
        self.assertEqual(len(steps), 4)
        self.assertIn("First identify knowns", steps[0])
        self.assertIn("30 * 2 = 60", steps[2])

    def test_delimiter_token_parsing(self):
        text = "State initial premise. ---step--- Compute sub-product. ---step--- Final result."
        steps = StepParser.parse(text)
        self.assertEqual(len(steps), 3)

    def test_paragraph_chunking(self):
        text = "First paragraph premise.\n\nSecond paragraph derivation.\n\nThird paragraph conclusion."
        steps = StepParser.parse(text)
        self.assertEqual(len(steps), 3)

    def test_answer_extraction(self):
        self.assertEqual(StepParser.extract_final_answer("The final answer is 42."), "42")
        self.assertEqual(StepParser.extract_final_answer("Therefore, we get \\boxed{15/4}"), "15/4")
        self.assertEqual(StepParser.extract_final_answer("Answer: $120.00"), "$120.00")


class TestMathSymbolicEvaluator(unittest.TestCase):
    def test_safe_arithmetic_eval(self):
        self.assertEqual(MathSymbolicEvaluator.safe_eval_expr("14 + 26", {}), 40.0)
        self.assertEqual(MathSymbolicEvaluator.safe_eval_expr("5 * 8 - 12", {}), 28.0)
        self.assertEqual(MathSymbolicEvaluator.safe_eval_expr("2 ** 5", {}), 32.0)
        self.assertEqual(MathSymbolicEvaluator.safe_eval_expr("100 / 4", {}), 25.0)

    def test_chained_equation_verification(self):
        checked, faults = MathSymbolicEvaluator.verify_step_equations(
            "Let F = m * a = 20 * 5 = 100", {"m": 20.0, "a": 5.0}
        )
        self.assertTrue(all(v for _, v, _ in checked))
        self.assertEqual(len(faults), 0)

    def test_arithmetic_fault_detection(self):
        checked, faults = MathSymbolicEvaluator.verify_step_equations(
            "Calculate total: 15 * 4 = 55", {}
        )
        self.assertGreaterEqual(len(faults), 1)
        self.assertIn("Arithmetic mismatch", faults[0])

    def test_currency_and_unit_cleaning(self):
        checked, faults = MathSymbolicEvaluator.verify_step_equations(
            "Total cost is 40 * $3 = $120 dollars", {}
        )
        self.assertTrue(any(v for _, v, _ in checked))
        self.assertEqual(len(faults), 0)


class TestPRMScorer(unittest.TestCase):
    def test_sound_trace_properties(self):
        text = (
            "Step 1: Given a rectangle with width w = 8 and length l = 12.\n"
            "Step 2: Area is calculated as length * width: 12 * 8 = 96.\n"
            "Step 3: Perimeter is 2 * (12 + 8) = 2 * 20 = 40.\n"
            "Step 4: Therefore, the final answer is 96 square units."
        )
        trace = PRMScorer.evaluate_trace(text, "Find area and perimeter")
        self.assertTrue(trace.is_sound)
        self.assertIsNone(trace.first_error_step)
        self.assertGreaterEqual(trace.min_score, 0.70)
        self.assertLessEqual(trace.product_score, trace.min_score)
        self.assertLessEqual(trace.min_score, trace.mean_score)

    def test_flawed_step_localization(self):
        text = (
            "Step 1: Start with 50 items.\n"
            "Step 2: Add 30 items: 50 + 30 = 80 items.\n"
            "Step 3: Sell 25 items: 80 - 25 = 45 items.\n"
            "Step 4: Multiply by 2: 45 * 2 = 90 items."
        )
        trace = PRMScorer.evaluate_trace(text, "Track inventory")
        self.assertFalse(trace.is_sound)
        # Step 3 had 80 - 25 = 45 (which is 55, not 45!)
        self.assertEqual(trace.first_error_step, 3)
        self.assertLess(trace.steps[2].score, 0.40)
        self.assertEqual(trace.steps[2].classification, StepClassification.NEGATIVE)

    def test_step_entropy_and_margin(self):
        step = PRMScorer.score_step(
            step_text="Compute 5 * 5 = 25",
            step_num=1,
            cumulative_prefix="5*5=25",
            problem="Math",
            is_final=False,
            state_vars={},
        )
        self.assertGreater(step.score, 0.70)
        self.assertGreater(step.margin, 0.0)
        self.assertGreaterEqual(step.entropy, 0.0)


class TestMonteCarloSuperviser(unittest.TestCase):
    def test_monte_carlo_value_rollouts(self):
        text = (
            "Step 1: Compute 10 + 10 = 20.\n"
            "Step 2: Compute 20 * 3 = 50.\n"
            "Step 3: Final Answer is 50."
        )
        trace = PRMScorer.evaluate_trace(text, "Math")
        mc_results = MonteCarloSuperviser.simulate_rollouts(trace, "60", num_rollouts=30, seed=42)
        self.assertEqual(len(mc_results), 3)
        # Step 2 has error (20*3=60!=50)
        v1, _ = mc_results[0]
        v2, adv2 = mc_results[1]
        self.assertGreater(v1, v2)
        self.assertLess(adv2, 0.0)


class TestPRMBeamSearch(unittest.TestCase):
    def test_beam_search_with_pruning(self):
        def mock_generator(cum: str, depth: int):
            if depth == 1:
                return ["2 + 2 = 4", "2 + 2 = 9"]  # Second is flawed
            elif depth == 2:
                return ["4 * 5 = 20", "4 * 5 = 22"]  # Second is flawed
            return ["Final answer: 20"]

        searcher = PRMBeamSearch(beam_width=2, max_depth=3, prune_threshold=0.35)
        result = searcher.search("Compute (2+2)*5", mock_generator)
        self.assertGreater(result["total_pruned"], 0)
        self.assertGreaterEqual(result["best_bottleneck_score"], 0.70)
        self.assertEqual(len(result["trajectory"]), 3)


class TestSelfConsistencyEngine(unittest.TestCase):
    def test_prm_overrides_corrupt_majority(self):
        t_good = PRMScorer.evaluate_trace("Step 1: 6 * 7 = 42.\nFinal Answer: 42.", "Multiplication")
        t_bad1 = PRMScorer.evaluate_trace("Step 1: 6 * 7 = 40.\nFinal Answer: 40.", "Multiplication")
        t_bad2 = PRMScorer.evaluate_trace("Step 1: 6 * 7 = 40.\nFinal Answer: 40.", "Multiplication")
        res = SelfConsistencyEngine.ensemble_traces([t_good, t_bad1, t_bad2])
        self.assertEqual(res["standard_majority_winner"], "40")
        self.assertEqual(res["prm_min_winner"], "42")
        self.assertFalse(res["consensus_aligned"])


class TestCLIIntegration(unittest.TestCase):
    def test_cli_subcommand_test(self):
        res = subprocess.run([sys.executable, str(BIN_PATH), "test"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("All 7 Formal PRM Mathematical & Algorithmic Invariants Verified Successfully!", res.stdout)

    def test_cli_subcommand_bench(self):
        res = subprocess.run([sys.executable, str(BIN_PATH), "bench"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("PRM Verification Engine passed all reasoning benchmark suites successfully.", res.stdout)

    def test_cli_subcommand_verify_json(self):
        res = subprocess.run(
            [sys.executable, str(BIN_PATH), "verify", "--text", "Step 1: 5+5=10\nStep 2: 10*2=20", "--json"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertTrue(data["is_sound"])
        self.assertEqual(len(data["steps"]), 2)


if __name__ == '__main__':
    unittest.main()
