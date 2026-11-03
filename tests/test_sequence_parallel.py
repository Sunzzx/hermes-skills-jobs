"""Comprehensive Unit and Integration Test Suite for DeepSpeed Ulysses & Sequence Parallelism Engine."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

# Direct import of engine components from bin
repo_dir = Path(__file__).resolve().parent.parent
bin_path = repo_dir / 'bin' / 'sequence-parallel'

from importlib.machinery import SourceFileLoader
loader = SourceFileLoader("sequence_parallel", str(bin_path))
spec = importlib.util.spec_from_loader("sequence_parallel", loader)
assert spec is not None
sp_module = importlib.util.module_from_spec(spec)
sys.modules["sequence_parallel"] = sp_module
loader.exec_module(sp_module)

SequenceParallelConfig = sp_module.SequenceParallelConfig
SequenceCommunicator = sp_module.SequenceCommunicator
MonolithicAttention = sp_module.MonolithicAttention
DeepSpeedUlyssesAttention = sp_module.DeepSpeedUlyssesAttention
MegatronSequenceParallelAttention = sp_module.MegatronSequenceParallelAttention
SequenceParallelTransformerBlock = sp_module.SequenceParallelTransformerBlock
compute_memory_profile = sp_module.compute_memory_profile
compute_comm_volume = sp_module.compute_comm_volume


class TestSequenceParallelEngine(unittest.TestCase):
    def setUp(self):
        self.cfg = SequenceParallelConfig(
            batch_size=2,
            seq_len=64,
            hidden_dim=128,
            num_heads=8,
            world_size=4,
            causal=False,
            dtype="float64",
        )
        self.comm = SequenceCommunicator(world_size=self.cfg.world_size)
        rng = np.random.RandomState(42)
        self.x_full = rng.randn(self.cfg.batch_size, self.cfg.seq_len, self.cfg.hidden_dim)
        self.x_shards = np.split(self.x_full, self.cfg.world_size, axis=1)

    def test_ulysses_numerical_equivalence(self):
        mono = MonolithicAttention(self.cfg, seed=123)
        y_mono = mono.forward(self.x_full)

        weights = (mono.W_q, mono.W_k, mono.W_v, mono.W_o)
        ulysses = DeepSpeedUlyssesAttention(self.cfg, self.comm, weights=weights)
        y_ulysses_shards = ulysses.forward(self.x_shards)
        y_ulysses = np.concatenate(y_ulysses_shards, axis=1)

        max_diff = float(np.max(np.abs(y_mono - y_ulysses)))
        self.assertLess(max_diff, 1e-11, f"Ulysses numerical error too high: {max_diff}")

    def test_causal_masking_equivalence(self):
        cfg_causal = SequenceParallelConfig(
            batch_size=2,
            seq_len=64,
            hidden_dim=128,
            num_heads=8,
            world_size=4,
            causal=True,
        )
        mono = MonolithicAttention(cfg_causal, seed=456)
        y_mono = mono.forward(self.x_full)

        weights = (mono.W_q, mono.W_k, mono.W_v, mono.W_o)
        ulysses = DeepSpeedUlyssesAttention(cfg_causal, self.comm, weights=weights)
        y_ulysses_causal_shards = ulysses.forward(self.x_shards)
        y_ulysses_causal = np.concatenate(y_ulysses_causal_shards, axis=1)

        max_diff = float(np.max(np.abs(y_mono - y_ulysses_causal)))
        self.assertLess(max_diff, 1e-11, f"Causal Ulysses numerical error too high: {max_diff}")

    def test_megatron_sp_equivalence(self):
        mono = MonolithicAttention(self.cfg, seed=789)
        y_mono = mono.forward(self.x_full)

        weights = (mono.W_q, mono.W_k, mono.W_v, mono.W_o)
        megatron_sp = MegatronSequenceParallelAttention(self.cfg, self.comm, weights=weights)
        y_meg_shards = megatron_sp.forward(self.x_shards)
        y_meg = np.concatenate(y_meg_shards, axis=1)

        max_diff = float(np.max(np.abs(y_mono - y_meg)))
        self.assertLess(max_diff, 1e-11, f"Megatron-SP numerical error too high: {max_diff}")

    def test_transformer_block_execution(self):
        block = SequenceParallelTransformerBlock(self.cfg, self.comm, seed=101)
        out_shards = block.forward(self.x_shards)
        self.assertEqual(len(out_shards), self.cfg.world_size)
        self.assertEqual(out_shards[0].shape, (self.cfg.batch_size, self.cfg.local_seq_len, self.cfg.hidden_dim))

    def test_communication_stats_tracking(self):
        self.comm.reset_stats()
        mono = MonolithicAttention(self.cfg, seed=123)
        ulysses = DeepSpeedUlyssesAttention(self.cfg, self.comm, weights=(mono.W_q, mono.W_k, mono.W_v, mono.W_o))
        _ = ulysses.forward(self.x_shards)

        self.assertEqual(self.comm.stats.all_to_all_calls, 4)  # Q, K, V seq->head + Out head->seq
        self.assertGreater(self.comm.stats.bytes_transferred, 0)
        self.assertGreater(self.comm.stats.messages_sent, 0)

    def test_memory_profile_analytical(self):
        prof = compute_memory_profile(seq_len=65536, world_size=8)
        self.assertIn("ulysses_peak_act_gb", prof)
        self.assertLess(prof["ulysses_peak_act_gb"], prof["monolithic_peak_act_gb"])

    def test_cli_execution_commands(self):
        res_verify = subprocess.run([sys.executable, str(bin_path), 'verify'], capture_output=True, text=True)
        self.assertEqual(res_verify.returncode, 0)
        self.assertIn('All Sequence Parallelism verifications passed', res_verify.stdout)

        res_bench = subprocess.run([sys.executable, str(bin_path), 'bench'], capture_output=True, text=True)
        self.assertEqual(res_bench.returncode, 0)
        self.assertIn('DeepSpeed Ulysses Memory & Scaling Benchmark', res_bench.stdout)

        res_comp = subprocess.run([sys.executable, str(bin_path), 'compare'], capture_output=True, text=True)
        self.assertEqual(res_comp.returncode, 0)
        self.assertIn('Sequence Parallelism Architecture Comparison', res_comp.stdout)

        res_sim = subprocess.run([sys.executable, str(bin_path), 'simulate'], capture_output=True, text=True)
        self.assertEqual(res_sim.returncode, 0)
        self.assertIn('All-to-All Communication Schedule', res_sim.stdout)


if __name__ == '__main__':
    unittest.main()
