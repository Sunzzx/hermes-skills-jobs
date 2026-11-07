import json
import math
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
import importlib.util

BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
LOOKAHEAD_BIN = BIN_DIR / "lookahead-decoding-engine"

loader = SourceFileLoader("lookahead_decoding_engine", str(LOOKAHEAD_BIN))
spec = importlib.util.spec_from_loader("lookahead_decoding_engine", loader)
assert spec is not None and spec.loader is not None
lookahead_mod = importlib.util.module_from_spec(spec)
sys.modules["lookahead_decoding_engine"] = lookahead_mod
loader.exec_module(lookahead_mod)

LookaheadConfig = lookahead_mod.LookaheadConfig
LookaheadBranchPool = lookahead_mod.LookaheadBranchPool
LookaheadVerificationEngine = lookahead_mod.LookaheadVerificationEngine
LookaheadDecodingEngine = lookahead_mod.LookaheadDecodingEngine


def test_lookahead_cli_test_flag():
    result = subprocess.run([sys.executable, str(LOOKAHEAD_BIN), "--test", "--json"], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI test failed: {result.stderr}"
    data = json.loads(result.stdout.strip())
    assert data["status"] == "passed"
    assert data["effective_speedup_x"] >= 1.0
    assert data["lossless_equivalence"] is True


def test_lookahead_cli_benchmark():
    result = subprocess.run([sys.executable, str(LOOKAHEAD_BIN), "--benchmark", "--json", "--tokens", "25"], capture_output=True, text=True)
    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    data = json.loads(result.stdout.strip())
    assert data["num_tokens"] == 25
    assert data["baseline_steps"] == 25
    assert data["lookahead_steps"] <= 25
    assert data["effective_speedup_x"] >= 1.0
    assert data["lossless_equivalence"] is True


def test_lookahead_branch_pool_generation():
    pool = LookaheadBranchPool(window_size=5, step_depth=4)
    prompt = [10, 20, 30, 40, 50, 10, 20, 30]
    pool.update_cache(prompt)
    candidates = pool.generate_candidates(prompt)
    assert len(candidates) == 5
    for cand in candidates:
        assert len(cand) == 4


def test_lookahead_2d_attention_mask():
    config = LookaheadConfig(lookahead_window=3, lookahead_steps=2)
    engine = LookaheadDecodingEngine(config)
    prefix_len = 5
    candidates = [[1, 2], [3, 4], [5, 6]]
    mask = engine.verifier.build_lookahead_mask(prefix_len, candidates)
    total_tokens = prefix_len + sum(len(c) for c in candidates)  # 5 + 6 = 11
    assert mask.shape == (total_tokens, total_tokens)
    # Prefix tokens attend causally to prefix
    for i in range(prefix_len):
        for j in range(prefix_len):
            if j <= i:
                assert mask[i, j] == 0.0
            else:
                assert mask[i, j] <= -1e8
    # Branch tokens attend to prefix
    for row in range(prefix_len, total_tokens):
        assert np.all(mask[row, :prefix_len] == 0.0)
