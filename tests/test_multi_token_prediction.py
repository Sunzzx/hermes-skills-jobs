"""Comprehensive Unit and Integration Test Suite for Multi-Token Prediction (MTP) Engine."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import unittest

import numpy as np

# Dynamically load mtp-engine from bin/
BIN_PATH = Path(__file__).resolve().parent.parent / "bin" / "mtp-engine"
loader = SourceFileLoader("mtp_engine", str(BIN_PATH))
spec = importlib.util.spec_from_loader("mtp_engine", loader)
mtp_mod = importlib.util.module_from_spec(spec)
sys.modules["mtp_engine"] = mtp_mod
loader.exec_module(mtp_mod)

RMSNorm = mtp_mod.RMSNorm
SwiGLUFFN = mtp_mod.SwiGLUFFN
CausalSelfAttention = mtp_mod.CausalSelfAttention
MTPModule = mtp_mod.MTPModule
MTPConfig = mtp_mod.MTPConfig
DeepSeekMTPModel = mtp_mod.DeepSeekMTPModel


class TestMTPComponents(unittest.TestCase):
    """Unit tests for individual components of the MTP engine."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_rmsnorm_unit_variance(self):
        dim = 64
        norm = RMSNorm(dim)
        x = self.rng.normal(0.0, 3.0, size=(2, 8, dim))
        out = norm.forward(x)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_allclose(rms, np.ones_like(rms), atol=1e-5)

    def test_rmsnorm_gradient_finite_difference(self):
        dim = 32
        norm = RMSNorm(dim)
        x = self.rng.normal(0.0, 1.0, size=(1, 4, dim))
        grad_out = self.rng.normal(0.0, 1.0, size=(1, 4, dim))

        norm.forward(x)
        grad_analytic = norm.backward(grad_out)

        eps = 1e-5
        grad_numeric = np.zeros_like(x)
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_plus = x.copy()
                x_plus[0, i, j] += eps
                x_minus = x.copy()
                x_minus[0, i, j] -= eps
                y_plus = norm.forward(x_plus)
                y_minus = norm.forward(x_minus)
                grad_numeric[0, i, j] = np.sum((y_plus - y_minus) / (2 * eps) * grad_out)

        np.testing.assert_allclose(grad_analytic, grad_numeric, atol=1e-4)

    def test_swiglu_ffn_shape_and_backprop(self):
        dim, hidden_dim = 32, 64
        ffn = SwiGLUFFN(dim, hidden_dim, rng=self.rng)
        x = self.rng.normal(0.0, 1.0, size=(2, 6, dim))
        out = ffn.forward(x)
        self.assertEqual(out.shape, (2, 6, dim))

        grad_out = np.ones_like(out)
        grad_x = ffn.backward(grad_out)
        self.assertEqual(grad_x.shape, x.shape)
        self.assertEqual(ffn.grad_w_down.shape, (hidden_dim, dim))
        self.assertEqual(ffn.grad_w_up.shape, (dim, hidden_dim))
        self.assertEqual(ffn.grad_w_gate.shape, (dim, hidden_dim))
        self.assertTrue(np.all(np.isfinite(grad_x)))

    def test_causal_attention_masking(self):
        dim, heads = 32, 4
        attn = CausalSelfAttention(dim, heads, rng=self.rng)
        seq_len = 10
        x = self.rng.normal(0.0, 1.0, size=(1, seq_len, dim))
        out1 = attn.forward(x)

        # Modify position 7 in the input sequence
        x_mod = x.copy()
        x_mod[0, 7, :] += 10.0
        out2 = attn.forward(x_mod)

        # Positions 0 to 6 must be strictly identical (causality check)
        np.testing.assert_allclose(out1[0, :7, :], out2[0, :7, :], atol=1e-12)
        # Position 7 and subsequent positions should differ
        self.assertFalse(np.allclose(out1[0, 7, :], out2[0, 7, :]))

    def test_mtp_module_weight_tying(self):
        dim, heads, vocab_size = 32, 2, 100
        shared_head = self.rng.normal(0.0, 0.1, size=(vocab_size, dim))
        mtp = MTPModule(depth=1, dim=dim, num_heads=heads, shared_unembed_weight=shared_head, rng=self.rng)

        # Ensure pointer identity for shared unembedding matrix
        self.assertIs(mtp.shared_unembed, shared_head)

        h_prev = self.rng.normal(0.0, 1.0, size=(2, 5, dim))
        emb_k = self.rng.normal(0.0, 1.0, size=(2, 5, dim))
        h_k, logits = mtp.forward(h_prev, emb_k)

        self.assertEqual(h_k.shape, (2, 5, dim))
        self.assertEqual(logits.shape, (2, 5, vocab_size))


class TestDeepSeekMTPModel(unittest.TestCase):
    """Integration tests for DeepSeek-V3 MTP model training & speculative generation."""

    def setUp(self):
        self.config = MTPConfig(
            vocab_size=200,
            dim=48,
            num_heads=3,
            num_base_layers=2,
            mtp_depth=2,
            seed=42,
        )
        self.model = DeepSeekMTPModel(self.config)

    def test_forward_base_shape(self):
        token_ids = np.array([[5, 12, 99, 4], [10, 20, 30, 40]])
        h_main, logits_main = self.model.forward_base(token_ids)
        self.assertEqual(h_main.shape, (2, 4, self.config.dim))
        self.assertEqual(logits_main.shape, (2, 4, self.config.vocab_size))

    def test_joint_loss_computation(self):
        seq_len = 16
        token_ids = np.random.randint(0, self.config.vocab_size, size=(2, seq_len))
        losses = self.model.compute_joint_loss(token_ids)

        self.assertIn("loss_main", losses)
        self.assertIn("loss_mtp_avg", losses)
        self.assertIn("loss_joint", losses)
        self.assertEqual(len(losses["mtp_individual"]), self.config.mtp_depth)
        self.assertGreater(losses["loss_joint"], 0.0)
        self.assertGreaterEqual(losses["loss_main"], 0.0)

    def test_speculative_generation_length_and_tokens(self):
        prompt = [1, 5, 23, 88]
        max_new = 10
        gen_tokens, stats = self.model.speculative_generate(prompt, max_new_tokens=max_new)

        self.assertEqual(len(gen_tokens), len(prompt) + max_new)
        self.assertEqual(stats["total_generated"], max_new)
        self.assertGreaterEqual(stats["base_steps"], 1)
        self.assertLessEqual(stats["base_steps"], max_new)  # Speedup implies fewer or equal steps
        self.assertGreaterEqual(stats["tokens_per_step"], 1.0)

    def test_standard_autoregressive_generation(self):
        prompt = [10, 20, 30]
        max_new = 5
        gen_tokens, steps = self.model.standard_autoregressive_generate(prompt, max_new_tokens=max_new)
        self.assertEqual(len(gen_tokens), len(prompt) + max_new)
        self.assertEqual(steps, max_new)


class TestMTPCLIIntegration(unittest.TestCase):
    """Tests verifying CLI subcommands of mtp-engine."""

    def test_cli_verify_exit_code(self):
        res = mtp_mod.cmd_verify(None)
        self.assertEqual(res, 0)


if __name__ == "__main__":
    unittest.main()
