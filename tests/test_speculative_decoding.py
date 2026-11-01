"""
Comprehensive Unit and Integration Test Suite for Speculative Decoding & Medusa Tree Attention Engine.
"""

import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "bin" / "spec-decode"
sys.path.insert(0, str(ROOT / "bin"))

from importlib.machinery import SourceFileLoader
import types
loader = SourceFileLoader("spec_decode", str(CLI))
mod = types.ModuleType(loader.name)
loader.exec_module(mod)

SpeculativeSampler = mod.SpeculativeSampler
MedusaTreeEngine = mod.MedusaTreeEngine
TreeNode = mod.TreeNode
SpeculativeKVCache = mod.SpeculativeKVCache
RooflineProfiler = mod.RooflineProfiler
SpeculativeSimulationEngine = mod.SpeculativeSimulationEngine


class TestSpeculativeSampler(unittest.TestCase):
    """Test mathematical validity of Speculative Rejection Sampling."""

    def test_distribution_equivalence_zero_shift(self):
        """Statistically proves zero distributional shift between direct and speculative sampling."""
        res = SpeculativeSampler.test_distribution_equivalence(
            vocab_size=6,
            num_trials=15000,
            seed=123,
        )
        self.assertTrue(res["statistically_identical"])
        self.assertLess(res["tvd_spec_vs_target"], 0.03)
        self.assertLess(res["kl_divergence"], 0.01)
        self.assertGreater(res["acceptance_rate"], 0.5)

    def test_verify_chain_all_accepted(self):
        """Tests chain verification when target perfectly matches draft candidates."""
        rng = np.random.RandomState(42)
        K = 3
        V = 10
        draft_tokens = [2, 5, 8]
        target_probs = np.zeros((K + 1, V))
        draft_probs = np.zeros((K, V))

        for k in range(K):
            tok = draft_tokens[k]
            target_probs[k, tok] = 1.0
            draft_probs[k, tok] = 1.0
        target_probs[K, 1] = 1.0  # Bonus token

        accepted, num_accepted, mean_alpha = SpeculativeSampler.verify_chain(
            target_probs, draft_probs, draft_tokens, rng=rng
        )
        self.assertEqual(num_accepted, 3)
        self.assertEqual(len(accepted), 4)  # 3 accepted + 1 bonus token
        self.assertEqual(accepted[:3], draft_tokens)
        self.assertEqual(accepted[3], 1)
        self.assertAlmostEqual(mean_alpha, 1.0)

    def test_verify_chain_rejection_at_first_step(self):
        """Tests immediate rejection when candidate probability is zero in target."""
        rng = np.random.RandomState(42)
        K = 3
        V = 5
        draft_tokens = [0, 1, 2]
        target_probs = np.zeros((K + 1, V))
        draft_probs = np.zeros((K, V))

        # Draft proposes tok 0 with prob 1.0, but target assigns 0 prob to tok 0
        draft_probs[0, 0] = 1.0
        target_probs[0, 4] = 1.0  # Target strictly wants tok 4

        accepted, num_accepted, mean_alpha = SpeculativeSampler.verify_chain(
            target_probs, draft_probs, draft_tokens, rng=rng
        )
        self.assertEqual(num_accepted, 0)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0], 4)  # Resampled from residual


class TestMedusaTreeEngine(unittest.TestCase):
    """Test Medusa candidate tree construction, attention mask, and branch verification."""

    def test_candidate_tree_construction(self):
        """Tests Cartesian tree node topology and path generation."""
        head_branching = [2, 2]
        preds = [
            [(10, 0.6), (11, 0.4)],
            [(20, 0.7), (21, 0.3)],
        ]
        nodes, paths = MedusaTreeEngine.construct_candidate_tree(head_branching, preds)

        # Root (1) + Level 1 (2) + Level 2 (2*2 = 4) = 7 nodes total
        self.assertEqual(len(nodes), 7)
        self.assertEqual(len(paths), 4)  # 4 distinct Cartesian paths
        for p in paths:
            self.assertEqual(len(p), 3)  # [root, level_1, level_2]
            self.assertEqual(p[0], 0)

    def test_tree_attention_mask_causality_and_isolation(self):
        """Verifies 2D Tree Attention Causal Mask properties and sibling isolation."""
        head_branching = [2, 2]
        preds = [
            [(10, 0.6), (11, 0.4)],
            [(20, 0.7), (21, 0.3)],
        ]
        nodes, paths = MedusaTreeEngine.construct_candidate_tree(head_branching, preds)
        prefix_len = 2
        mask = MedusaTreeEngine.build_tree_attention_mask(nodes, prefix_length=prefix_len)

        total_len = prefix_len + (len(nodes) - 1)  # 2 + 6 = 8
        self.assertEqual(mask.shape, (total_len, total_len))

        # 1. Prefix causality: lower triangular
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])
        self.assertTrue(mask[1, 0])
        self.assertTrue(mask[1, 1])

        # 2. Candidate tokens attend to prefix tokens
        for i in range(prefix_len, total_len):
            self.assertTrue(np.all(mask[i, :prefix_len]))

        # 3. Siblings cannot attend to each other
        # Node 1 and Node 2 are children of root
        # Row for Node 1: prefix_len + 0 = 2; Row for Node 2: prefix_len + 1 = 3
        self.assertFalse(mask[2, 3])  # Node 1 does not attend to Node 2
        self.assertFalse(mask[3, 2])  # Node 2 does not attend to Node 1

    def test_verify_tree_branches_greedy(self):
        """Verifies greedy branch selection across candidate tree."""
        head_branching = [2, 2]
        preds = [
            [(10, 0.6), (11, 0.4)],
            [(20, 0.7), (21, 0.3)],
        ]
        nodes, paths = MedusaTreeEngine.construct_candidate_tree(head_branching, preds)

        target_logits = {
            0: np.zeros(50),
            1: np.zeros(50),
            2: np.zeros(50),
            3: np.zeros(50),
        }
        # Set target logits so that branch [0 -> node 1 (tok 10) -> node 3 (tok 20)] is accepted
        target_logits[0][10] = 10.0
        target_logits[1][20] = 10.0
        target_logits[3][45] = 10.0  # Bonus token from leaf

        best_tokens, best_path, num_accepted = MedusaTreeEngine.verify_tree_branches(
            nodes, paths, target_logits, greedy=True
        )
        self.assertEqual(num_accepted, 2)
        self.assertEqual(best_path, [0, 1, 3])
        self.assertEqual(best_tokens, [10, 20, 45])


class TestSpeculativeKVCache(unittest.TestCase):
    """Test Dynamic Key-Value state management and rollback."""

    def test_cache_append_and_rollback(self):
        cache = SpeculativeKVCache(num_heads=2, head_dim=4)
        k1 = np.ones((2, 3, 4), dtype=np.float32)
        v1 = np.ones((2, 3, 4), dtype=np.float32)
        cache.append(k1, v1)
        self.assertEqual(cache.seq_len, 3)

        # Checkpoint before speculative expansion
        cache.checkpoint()
        self.assertEqual(cache.checkpoint_len, 3)

        # Append speculative branch tokens
        k_spec = np.zeros((2, 4, 4), dtype=np.float32)
        v_spec = np.zeros((2, 4, 4), dtype=np.float32)
        cache.append(k_spec, v_spec)
        self.assertEqual(cache.seq_len, 7)

        # Rollback on branch prune
        cache.rollback()
        self.assertEqual(cache.seq_len, 3)
        self.assertEqual(cache.memory_bytes, 2 * (2 * 3 * 4 * 4))


class TestRooflineProfiler(unittest.TestCase):
    """Test Roofline memory bandwidth model and speedup metrics."""

    def test_roofline_analysis(self):
        prof = RooflineProfiler.analyze(
            param_count=7e9,
            dtype_bytes=2,
            memory_bandwidth_gbs=300.0,
            tokens_per_step=3.0,
            speculative_overhead=0.1,
        )
        self.assertAlmostEqual(prof["theoretical_speedup"], 3.0 / 1.1, places=2)
        self.assertAlmostEqual(prof["dram_bandwidth_savings_percent"], 66.67, places=1)
        self.assertGreater(prof["spec_arithmetic_intensity_flops_per_byte"], prof["ar_arithmetic_intensity_flops_per_byte"])


class TestCLIExecution(unittest.TestCase):
    """Test CLI subcommands via subprocess."""

    def test_cli_help(self):
        p = subprocess.run([str(CLI), "--help"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("Speculative Decoding", p.stdout)

    def test_cli_verify(self):
        p = subprocess.run([str(CLI), "verify", "--trials", "1000"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("MATHEMATICAL THEOREM CONFIRMED", p.stdout)

    def test_cli_tree(self):
        p = subprocess.run([str(CLI), "tree", "--branching", "2,2"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("Medusa Candidate Speculation Tree", p.stdout)
        self.assertIn("2D Tree Causal Attention Adjacency Matrix", p.stdout)

    def test_cli_bench(self):
        p = subprocess.run([str(CLI), "bench", "--tokens", "30"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("Medusa Tree Attention", p.stdout)

    def test_cli_profile(self):
        p = subprocess.run([str(CLI), "profile"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("Roofline Memory Bandwidth", p.stdout)


if __name__ == "__main__":
    unittest.main()
