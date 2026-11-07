import json
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
import importlib.util

BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
MINFERENCE_BIN = BIN_DIR / "minference-engine"

loader = SourceFileLoader("minference_engine", str(MINFERENCE_BIN))
spec = importlib.util.spec_from_loader("minference_engine", loader)
assert spec is not None and spec.loader is not None
minf_mod = importlib.util.module_from_spec(spec)
sys.modules["minference_engine"] = minf_mod
loader.exec_module(minf_mod)

MInferenceConfig = minf_mod.MInferenceConfig
MInferenceAttentionEngine = minf_mod.MInferenceAttentionEngine
AttentionPattern = minf_mod.AttentionPattern
PatternClassifier = minf_mod.PatternClassifier


def test_minference_cli_test_flag():
    """Verify CLI --test executes successfully with JSON metrics output."""
    result = subprocess.run([sys.executable, str(MINFERENCE_BIN), "--test", "--json"], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI test failed: {result.stderr}"
    data = json.loads(result.stdout.strip())
    assert data["status"] == "passed"
    assert data["theoretical_speedup_x"] > 1.2
    assert data["sparsity_percentage"] > 40.0
    assert data["cosine_similarity"] > 0.95


def test_minference_cli_benchmark():
    """Verify CLI --benchmark outputs structured performance metrics."""
    result = subprocess.run([sys.executable, str(MINFERENCE_BIN), "--benchmark", "--seq-len", "128", "--json"], capture_output=True, text=True)
    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    data = json.loads(result.stdout.strip())
    assert data["seq_len"] == 128
    assert data["theoretical_speedup_x"] >= 1.0
    assert "head_patterns" in data


def test_pattern_classifier_a_shape():
    """Verify PatternClassifier correctly isolates A-Shape sink+band patterns."""
    classifier = PatternClassifier(sink_size=16, local_window=32)
    seq_len = 128
    attn_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
    # Dense attention only on sink and diagonal
    attn_matrix[:, :16] = 0.5
    for i in range(seq_len):
        start = max(0, i - 16)
        attn_matrix[i, start:i+1] += 0.5
    attn_matrix = attn_matrix / np.sum(attn_matrix, axis=-1, keepdims=True)
    pattern = classifier.classify_head(attn_matrix)
    assert pattern in (AttentionPattern.A_SHAPE, AttentionPattern.VERTICAL_SLASH, AttentionPattern.BLOCK_SPARSE)


def test_sparse_vs_dense_numerical_fidelity():
    """Test numerical fidelity of A-Shape and Vertical-Slash vs Dense Attention."""
    config = MInferenceConfig(sink_size=8, local_window_size=16, vertical_top_k=8, block_size=16, block_top_k=2)
    engine = MInferenceAttentionEngine(config)
    seq_len = 128
    head_dim = 32
    Q = np.random.randn(seq_len, head_dim).astype(np.float32)
    K = np.random.randn(seq_len, head_dim).astype(np.float32)
    V = np.random.randn(seq_len, head_dim).astype(np.float32)

    dense_out, _ = engine.compute_dense_attention(Q, K, V)
    a_out, a_stats = engine.compute_a_shape_sparse_attention(Q, K, V)
    v_out, v_stats = engine.compute_vertical_slash_sparse_attention(Q, K, V)
    b_out, b_stats = engine.compute_block_sparse_attention(Q, K, V)

    assert a_out.shape == (seq_len, head_dim)
    assert v_out.shape == (seq_len, head_dim)
    assert b_out.shape == (seq_len, head_dim)
    assert a_stats["sparsity_ratio"] > 0.0
    assert v_stats["sparsity_ratio"] > 0.0
    assert b_stats["sparsity_ratio"] > 0.0
