"""Unit and integration tests for SimPO Simple Preference Optimization Engine."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest

# Dynamically import bin/simpo-engine as module
ROOT = Path(__file__).resolve().parents[1]
BIN_PATH = ROOT / "bin" / "simpo-engine"

spec = importlib.util.spec_from_loader("simpo_engine", SourceFileLoader("simpo_engine", str(BIN_PATH)))
assert spec is not None and spec.loader is not None
simpo_engine = importlib.util.module_from_spec(spec)
sys.modules["simpo_engine"] = simpo_engine
spec.loader.exec_module(simpo_engine)

SimPO = simpo_engine.SimPO
DPO = simpo_engine.DPO
IPO = simpo_engine.IPO
ORPO = simpo_engine.ORPO
CausalLM = simpo_engine.CausalLM
SimpleTokenizer = simpo_engine.SimpleTokenizer
PreferencePair = simpo_engine.PreferencePair
PreferenceTrainer = simpo_engine.PreferenceTrainer
AdamW = simpo_engine.AdamW
sigmoid = simpo_engine.sigmoid
log_sigmoid = simpo_engine.log_sigmoid
softmax = simpo_engine.softmax
log_softmax = simpo_engine.log_softmax
generate_synthetic_preferences = simpo_engine.generate_synthetic_preferences
generate_length_bias_stress_dataset = simpo_engine.generate_length_bias_stress_dataset


class TestMathematicalPrimitives:
    """Test numerical stability and mathematical axioms of primitive functions."""

    def test_sigmoid_numerical_stability(self):
        extreme_inputs = np.array([-1000.0, -50.0, -10.0, 0.0, 10.0, 50.0, 1000.0])
        outputs = sigmoid(extreme_inputs)
        assert not np.any(np.isnan(outputs))
        assert not np.any(np.isinf(outputs))
        assert outputs[0] == pytest.approx(0.0, abs=1e-7)
        assert outputs[3] == pytest.approx(0.5, abs=1e-7)
        assert outputs[-1] == pytest.approx(1.0, abs=1e-7)

    def test_log_sigmoid_numerical_stability(self):
        extreme_inputs = np.array([-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0])
        outputs = log_sigmoid(extreme_inputs)
        assert not np.any(np.isnan(outputs))
        assert not np.any(np.isinf(outputs))
        assert outputs[3] == pytest.approx(-math.log(2.0), abs=1e-6)
        assert outputs[-1] == pytest.approx(0.0, abs=1e-6)
        assert outputs[0] < -30.0

    def test_softmax_probability_axioms(self):
        x = np.random.randn(4, 10) * 5.0
        probs = softmax(x, axis=-1)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        sums = np.sum(probs, axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_log_softmax_consistency(self):
        x = np.random.randn(3, 8) * 3.0
        log_p = log_softmax(x, axis=-1)
        probs = softmax(x, axis=-1)
        assert np.allclose(log_p, np.log(probs + 1e-12), atol=1e-5)


class TestSimPOLossFormulation:
    """Verify SimPO loss mathematical properties, margins, and analytical gradients."""

    def test_margin_boundary_value(self):
        simpo = SimPO(beta=2.0, gamma=0.8)
        # When r_w - r_l == gamma, diff = 0, so loss = -log(sigmoid(0)) = log(2)
        chosen_len = 10
        rejected_len = 10
        # r_w - r_l = (beta / 10) * (logp_w - logp_l) = 0.8 => logp_w - logp_l = 4.0
        loss, rw, rl, diff = simpo.compute_loss(0.0, chosen_len, -4.0, rejected_len)
        assert abs(diff - simpo.gamma) < 1e-6
        assert abs(loss - math.log(2.0)) < 1e-6

    def test_loss_monotonicity(self):
        simpo = SimPO(beta=2.0, gamma=0.5)
        chosen_len = 12
        rejected_len = 15
        rej_logp = -25.0

        losses = []
        for chosen_logp in np.linspace(-30.0, -5.0, 10):
            l, _, _, _ = simpo.compute_loss(chosen_logp, chosen_len, rej_logp, rejected_len)
            losses.append(l)

        # Increasing chosen logp must strictly decrease loss
        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1]

    def test_analytical_gradient_finite_difference(self):
        simpo = SimPO(beta=2.5, gamma=0.5)
        c_logp = -18.5
        c_len = 20
        r_logp = -22.3
        r_len = 24

        d_chosen, d_rejected = simpo.compute_grad_weights(c_logp, c_len, r_logp, r_len)

        eps = 1e-6
        l_c_plus, _, _, _ = simpo.compute_loss(c_logp + eps, c_len, r_logp, r_len)
        l_c_minus, _, _, _ = simpo.compute_loss(c_logp - eps, c_len, r_logp, r_len)
        num_d_chosen = (l_c_plus - l_c_minus) / (2.0 * eps)

        l_r_plus, _, _, _ = simpo.compute_loss(c_logp, c_len, r_logp + eps, r_len)
        l_r_minus, _, _, _ = simpo.compute_loss(c_logp, c_len, r_logp - eps, r_len)
        num_d_rejected = (l_r_plus - l_r_minus) / (2.0 * eps)

        assert abs(d_chosen - num_d_chosen) < 1e-5
        assert abs(d_rejected - num_d_rejected) < 1e-5

    def test_length_normalization_gradient_scaling(self):
        simpo = SimPO(beta=2.0, gamma=0.5)
        # A 50-token response should receive 5x smaller token gradient multiplier than a 10-token response
        d_c_10, _ = simpo.compute_grad_weights(-10.0, 10, -10.0, 10)
        d_c_50, _ = simpo.compute_grad_weights(-50.0, 50, -50.0, 50)
        ratio = d_c_10 / d_c_50
        assert abs(ratio - 5.0) < 1e-6


class TestTokenizerAndCausalLM:
    """Verify model architecture, causal attention, and parameter updates."""

    def test_tokenizer_encode_decode(self):
        tok = SimpleTokenizer(vocab_size=128)
        text = "Hello SimPO!"
        encoded = tok.encode(text)
        assert encoded[0] == tok.bos_token_id
        assert encoded[-1] == tok.eos_token_id
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_causal_masking_invariant(self):
        # Tokens at future positions must not influence earlier position logits
        model = CausalLM(vocab_size=64, d_model=32, d_ff=64, seed=42)
        seq_a = [1, 10, 20, 30, 2]
        seq_b = [1, 10, 20, 45, 55, 2]  # Identical prefix [1, 10, 20]

        logits_a, _ = model.forward(seq_a)
        logits_b, _ = model.forward(seq_b)

        # Logits at position 0, 1, 2 must be bit-level identical
        assert np.allclose(logits_a[:3], logits_b[:3], atol=1e-7)

    def test_model_copy(self):
        model = CausalLM(vocab_size=64, d_model=32, seed=42)
        copy_m = model.copy()
        for p1, p2 in zip(model.params, copy_m.params):
            assert np.array_equal(p1, p2)
            p2 += 1.0
            assert not np.array_equal(p1, p2)


class TestOptimizerAdamW:
    """Test AdamW weight decay, momentum, and clipping."""

    def test_adamw_gradient_clipping(self):
        param = np.zeros((10, 10))
        opt = AdamW([param], lr=1e-2, max_grad_norm=1.0)
        huge_grad = np.ones((10, 10)) * 100.0  # norm = 1000
        gnorm = opt.step([huge_grad])
        assert gnorm == pytest.approx(1000.0, rel=1e-4)
        # Param update should be bounded
        assert np.max(np.abs(param)) < 0.1


class TestPreferenceTrainer:
    """Test end-to-end training and optimization progress."""

    def test_training_step_convergence(self):
        tokenizer = SimpleTokenizer(vocab_size=128)
        model = CausalLM(vocab_size=128, d_model=32, seed=42)
        simpo = SimPO(beta=2.0, gamma=0.5)
        opt = AdamW(model.params, lr=5e-3)
        trainer = PreferenceTrainer(model, tokenizer=tokenizer)

        pair = PreferencePair(
            prompt="Solve equation",
            chosen="x = 42",
            rejected="error 000",
            prompt_ids=tokenizer.encode("Solve equation"),
            chosen_ids=tokenizer.encode("x = 42"),
            rejected_ids=tokenizer.encode("error 000"),
        )

        initial_metrics = trainer.train_step_simpo(pair, simpo, opt)
        # Train for multiple steps on this pair
        last_metrics = initial_metrics
        for _ in range(12):
            last_metrics = trainer.train_step_simpo(pair, simpo, opt)

        # Margin should have expanded and loss decreased
        assert last_metrics.loss < initial_metrics.loss
        assert last_metrics.reward_margin > initial_metrics.reward_margin
        assert last_metrics.accuracy == 1.0

    def test_evaluate_dataset(self):
        tokenizer = SimpleTokenizer(vocab_size=128)
        model = CausalLM(vocab_size=128, d_model=32, seed=42)
        simpo = SimPO(beta=2.0, gamma=0.5)
        trainer = PreferenceTrainer(model, tokenizer=tokenizer)
        pairs = generate_synthetic_preferences(tokenizer, count=5, seed=42)

        eval_res = trainer.evaluate_dataset(pairs, simpo)
        assert "eval_loss" in eval_res
        assert "eval_accuracy" in eval_res
        assert "eval_reward_margin" in eval_res
        assert eval_res["eval_accuracy"] >= 0.0


class TestLengthBiasStressSuite:
    """Verify that length normalization mitigates verbosity bias."""

    def test_stress_dataset_length_disparity(self):
        tokenizer = SimpleTokenizer(vocab_size=64)
        pairs = generate_length_bias_stress_dataset(tokenizer, count=5)
        for p in pairs:
            # Rejected response must be significantly longer due to padded fluff
            assert p.rejected_len > p.chosen_len * 3


class TestCLIIntegration:
    """Verify CLI commands run successfully with proper exit codes."""

    def test_cli_help(self):
        res = subprocess.run([str(BIN_PATH), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "SimPO: Simple Preference Optimization Engine" in res.stdout

    def test_cli_status(self):
        res = subprocess.run([str(BIN_PATH), "status"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "SimPO Solution" in res.stdout
        assert "Memory & Computational Complexity Analysis" in res.stdout

    def test_cli_verify(self):
        res = subprocess.run([str(BIN_PATH), "verify"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "All mathematical and gradient verification invariants PASSED!" in res.stdout

    def test_cli_train_mini(self):
        res = subprocess.run(
            [str(BIN_PATH), "train", "--epochs", "2", "--samples", "4", "--d-model", "24"],
            capture_output=True, text=True
        )
        assert res.returncode == 0
        assert "Post-Training Validation Results" in res.stdout

    def test_cli_compare_mini(self):
        res = subprocess.run(
            [str(BIN_PATH), "compare", "--steps", "5", "--samples", "4", "--d-model", "24"],
            capture_output=True, text=True
        )
        assert res.returncode == 0
        assert "Multi-Algorithm Alignment Benchmark Comparison" in res.stdout

    def test_cli_benchmark_mini(self):
        res = subprocess.run(
            [str(BIN_PATH), "benchmark", "--iterations", "5", "--d-model", "24"],
            capture_output=True, text=True
        )
        assert res.returncode == 0
        assert "SimPO Computational Throughput Benchmark" in res.stdout
