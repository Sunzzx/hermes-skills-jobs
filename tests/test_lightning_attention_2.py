import json
import math
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

# Add bin directory to sys.path
BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
sys.path.insert(0, str(BIN_DIR))

from importlib.machinery import SourceFileLoader

lightning_mod = SourceFileLoader("lightning_attention_2", str(BIN_DIR / "lightning-attention-2")).load_module()
LightningAttention2Engine = lightning_mod.LightningAttention2Engine
LightningAttentionConfig = lightning_mod.LightningAttentionConfig
verify_equivalence = lightning_mod.verify_equivalence
run_benchmark = lightning_mod.run_benchmark


def test_numerical_equivalence_basic():
    """Test exact mathematical equivalence between exact, recurrent, and chunked modes."""
    res = verify_equivalence(seq_len=128, block_size=32)
    assert res["passed"] is True
    assert res["diff_recurrent"] < 1e-4
    assert res["diff_chunked"] < 1e-4


def test_numerical_equivalence_non_power_of_two():
    """Test arbitrary sequence length with block padding."""
    res = verify_equivalence(seq_len=173, block_size=64)
    assert res["passed"] is True
    assert res["diff_chunked"] < 1e-4


def test_decay_factors():
    """Verify that decay matrices follow strictly positive exponential decay."""
    cfg = LightningAttentionConfig(d_model=64, n_heads=2, d_head=32, block_size=16)
    engine = LightningAttention2Engine(cfg)
    decay = engine.get_decay_matrix(16, 0)
    assert decay.shape == (16, 16)
    assert np.all(decay >= 0.0)
    assert np.allclose(np.diag(decay), 1.0)
    # Strict upper triangle must be 0
    assert np.all(np.triu(decay, k=1) == 0.0)


def test_recurrent_causality():
    """Assert future tokens do not affect past recurrent state outputs."""
    b, h, d_head = 1, 2, 16
    cfg = LightningAttentionConfig(d_model=h * d_head, n_heads=h, d_head=d_head, block_size=16)
    engine = LightningAttention2Engine(cfg)

    np.random.seed(123)
    q = np.random.randn(b, h, 32, d_head).astype(np.float32)
    k = np.random.randn(b, h, 32, d_head).astype(np.float32)
    v = np.random.randn(b, h, 32, d_head).astype(np.float32)

    out1, _ = engine.forward_recurrent(q, k, v)

    # Perturb future tokens beyond index 15
    k_perturbed = k.copy()
    k_perturbed[:, :, 20:, :] += 10.0
    out2, _ = engine.forward_recurrent(q, k_perturbed, v)

    # Past tokens <= 19 must remain identical
    np.testing.assert_allclose(out1[:, :, :20, :], out2[:, :, :20, :], atol=1e-5)


def test_cli_execution():
    """Assert CLI binary executes cleanly and outputs valid JSON telemetry."""
    cmd = [str(BIN_DIR / "lightning-attention-2"), "--test", "--seq-len", "64", "--block-size", "32", "--json"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    assert data["passed"] is True
    assert data["seq_len"] == 64
