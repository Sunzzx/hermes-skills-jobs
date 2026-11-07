import json
import math
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
sys.path.insert(0, str(BIN_DIR))

quest_mod = SourceFileLoader("quest_engine", str(BIN_DIR / "quest-attention-engine")).load_module()
QuestAttentionEngine = quest_mod.QuestAttentionEngine
QuestConfig = quest_mod.QuestConfig
QuestPage = quest_mod.QuestPage
verify_quest_engine = quest_mod.verify_quest_engine
run_benchmark = quest_mod.run_benchmark


def test_bounding_box_guarantee():
    # Test that estimated upper bound is strictly greater than or equal to true dot products.
    np.random.seed(101)
    keys = np.random.randn(4, 16, 32).astype(np.float32)
    values = np.random.randn(4, 16, 32).astype(np.float32)
    page = QuestPage(0, keys, values)
    query = np.random.randn(4, 32).astype(np.float32)

    ub = page.estimate_upper_bound(query)
    dots = np.matmul(page.keys, query[:, :, None])[:, :, 0]
    max_dots = np.max(dots, axis=1)

    assert ub.shape == (4,)
    assert np.all(ub >= max_dots - 1e-5)


def test_paged_kv_partitioning():
    # Test cache paging partitions keys and values cleanly into correct page counts.
    cfg = QuestConfig(d_model=64, n_heads=2, d_head=32, page_size=16)
    engine = QuestAttentionEngine(cfg)
    keys = np.random.randn(2, 60, 32).astype(np.float32)
    values = np.random.randn(2, 60, 32).astype(np.float32)
    pages = engine.build_paged_kv_cache(keys, values)
    assert len(pages) == 4  # ceil(60/16) = 4
    assert pages[0].keys.shape == (2, 16, 32)
    assert pages[3].keys.shape == (2, 12, 32)


def test_quest_verification_fidelity():
    # Verify Quest attention retains >95% attention mass and near-1.0 cosine similarity.
    res = verify_quest_engine(seq_len=256, page_size=16, top_k_pages=6)
    assert res["passed"] is True
    assert res["bound_violations"] == 0
    assert res["cosine_similarity"] >= 0.95
    assert res["captured_attention_mass"] >= 0.90
    assert res["memory_bandwidth_savings_pct"] > 50.0


def test_benchmark_execution():
    # Test benchmarking routine completes across sequence lengths.
    results = run_benchmark()
    assert len(results) == 5
    assert all(r["retained_pages"] <= 8 for r in results)
    assert results[-1]["kv_cache_transfer_reduction_pct"] > 90.0


def test_cli_execution():
    # Verify CLI executes and outputs valid JSON telemetry.
    cmd = [sys.executable, str(BIN_DIR / "quest-attention-engine"), "--test", "--json"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    assert data["passed"] is True
    assert data["bound_violations"] == 0
