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
MEDUSA_BIN = BIN_DIR / "medusa-speculative-engine"

loader = SourceFileLoader("medusa_speculative_engine", str(MEDUSA_BIN))
spec = importlib.util.spec_from_loader("medusa_speculative_engine", loader)
assert spec is not None and spec.loader is not None
medusa_mod = importlib.util.module_from_spec(spec)
sys.modules["medusa_speculative_engine"] = medusa_mod
loader.exec_module(medusa_mod)

MedusaConfig = medusa_mod.MedusaConfig
MedusaDraftTreeBuilder = medusa_mod.MedusaDraftTreeBuilder
MedusaVerifier = medusa_mod.MedusaVerifier
MedusaEngine = medusa_mod.MedusaEngine


def test_medusa_cli_test_flag():
    result = subprocess.run([sys.executable, str(MEDUSA_BIN), "--test", "--json"], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI test failed: {result.stderr}"
    data = json.loads(result.stdout.strip())
    assert data["status"] == "passed"
    assert data["tree_size"] > 1
    assert data["accepted_count"] >= 1
    assert data["speedup_factor"] > 0
    assert 0.0 < data["mask_sparsity"] < 1.0


def test_medusa_cli_benchmark():
    result = subprocess.run([sys.executable, str(MEDUSA_BIN), "--benchmark", "--json", "--steps", "20"], capture_output=True, text=True)
    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    data = json.loads(result.stdout.strip())
    assert data["num_iterations"] == 20
    assert data["mean_accepted_tokens_per_step"] > 1.0
    assert data["effective_speedup"] > 1.0
    assert data["throughput_tokens_per_sec"] > 0


def test_tree_attention_mask_properties():
    cfg = MedusaConfig(num_heads=3, topk=2)
    builder = MedusaDraftTreeBuilder(cfg)
    
    # Mock head logits
    np.random.seed(123)
    head_logits = [np.random.randn(cfg.vocab_size) for _ in range(cfg.num_heads)]
    
    tree_tokens, parents_with_idx, tree_mask = builder.build_tree(root_token=10, head_logits=head_logits)
    n = len(tree_tokens)
    
    assert n > 1
    assert tree_mask.shape == (n, n)
    # Root attends to itself
    assert tree_mask[0, 0] == 1.0
    # All diagonal entries are 1.0
    for i in range(n):
        assert tree_mask[i, i] == 1.0
    # Mask should be causal/directed without cycles
    for i in range(n):
        for j in range(n):
            if tree_mask[i, j] == 1.0 and i != j:
                # j must precede i in topological order
                assert j < i


def test_medusa_verifier_greedy_acceptance():
    tokens = [10, 20, 21, 30]
    parents = [-1, 0, 0, 1]
    
    # Target predictions: node 0 predicts 20, node 1 predicts 30, node 2 predicts 99, node 3 predicts 40
    target_logits = np.zeros((4, 100), dtype=np.float32)
    target_logits[0, 20] = 10.0
    target_logits[1, 30] = 10.0
    target_logits[2, 99] = 10.0
    target_logits[3, 40] = 10.0
    
    accepted, count = MedusaVerifier.verify_tree(tokens, parents, target_logits)
    # Path 0 -> 1 (tok 20) -> 3 (tok 30) matches perfectly, plus bonus at node 3 (tok 40)
    assert count == 3
    assert accepted == [20, 30, 40]


def test_medusa_engine_speedup_bounds():
    cfg = MedusaConfig(num_heads=4, topk=3)
    engine = MedusaEngine(cfg)
    
    # Perfect draft scenario
    res_perfect = engine.simulate_step(root_token=50, draft_head_probs=1.0, seed=1)
    assert res_perfect["accepted_count"] >= 4
    assert res_perfect["speedup_factor"] >= 3.0
    
    # Poor draft scenario
    res_poor = engine.simulate_step(root_token=50, draft_head_probs=0.0, seed=2)
    assert res_poor["accepted_count"] >= 1
    assert res_poor["speedup_factor"] >= 0.8
