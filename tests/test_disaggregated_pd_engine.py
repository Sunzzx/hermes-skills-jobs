"""Unit and integration tests for Disaggregated Prefill-Decode Engine."""

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

ROOT = Path(__file__).resolve().parents[1]
BIN_PATH = ROOT / "bin" / "disaggregated-pd-engine"

spec = importlib.util.spec_from_loader(
    "disaggregated_pd_engine", SourceFileLoader(
        "disaggregated_pd_engine", str(BIN_PATH)
    )
)
assert spec is not None and spec.loader is not None
disagg_mod = importlib.util.module_from_spec(spec)
sys.modules["disaggregated_pd_engine"] = disagg_mod
spec.loader.exec_module(disagg_mod)

ServingConfig = disagg_mod.ServingConfig
MemoryManager = disagg_mod.MemoryManager
ChunkedPrefillNode = disagg_mod.ChunkedPrefillNode
DecodeNode = disagg_mod.DecodeNode
DisaggregatedPDServingSystem = disagg_mod.DisaggregatedPDServingSystem


class TestDisaggregatedPDEngine:
    def test_memory_manager_allocation_and_free(self):
        mgr = MemoryManager(num_blocks=10, block_size=16, num_heads=4, head_dim=32)
        assert len(mgr.free_blocks) == 10
        b1 = mgr.allocate()
        b2 = mgr.allocate()
        assert b1 != b2
        assert len(mgr.free_blocks) == 8
        assert len(mgr.allocated_blocks) == 2
        mgr.free(b1)
        assert len(mgr.free_blocks) == 9
        assert b1 in mgr.free_blocks

    def test_chunked_prefill_computation(self):
        cfg = ServingConfig(chunk_size=128, num_heads=2, head_dim=16)
        node = ChunkedPrefillNode(node_id=0, config=cfg)
        np.random.seed(42)
        chunk1 = np.random.randn(128, 2, 16).astype(np.float32)
        k_empty = np.zeros((0, 2, 16), dtype=np.float32)
        v_empty = np.zeros((0, 2, 16), dtype=np.float32)

        out1, k1, v1 = node.process_chunk(chunk1, k_empty, v_empty, causal_offset=0)
        assert out1.shape == (128, 2, 16)
        assert k1.shape == (128, 2, 16)
        assert v1.shape == (128, 2, 16)

        chunk2 = np.random.randn(64, 2, 16).astype(np.float32)
        out2, k2, v2 = node.process_chunk(chunk2, k1, v1, causal_offset=128)
        assert out2.shape == (64, 2, 16)
        assert k2.shape == (192, 2, 16)
        assert v2.shape == (192, 2, 16)

    def test_decode_node_paged_reception_and_step(self):
        cfg = ServingConfig(page_block_size=16, num_heads=2, head_dim=16)
        node = DecodeNode(node_id=0, config=cfg)
        np.random.seed(42)
        prompt_len = 35  # requires 3 blocks (16 + 16 + 3)
        k_tensor = np.random.randn(prompt_len, 2, 16).astype(np.float32)
        v_tensor = np.random.randn(prompt_len, 2, 16).astype(np.float32)

        node.receive_kv_cache("test-req-1", k_tensor, v_tensor)
        assert len(node.block_tables["test-req-1"]) == 3
        assert node.mem_mgr.allocated_blocks[node.block_tables["test-req-1"][0]].num_filled == 16
        assert node.mem_mgr.allocated_blocks[node.block_tables["test-req-1"][2]].num_filled == 3

        new_token = np.random.randn(1, 2, 16).astype(np.float32)
        out_ctx = node.decode_step("test-req-1", new_token)
        assert out_ctx.shape == (1, 2, 16)
        assert node.mem_mgr.allocated_blocks[node.block_tables["test-req-1"][2]].num_filled == 4

    def test_end_to_end_system_serve(self):
        cfg = ServingConfig(chunk_size=128, page_block_size=16, num_heads=2, head_dim=16)
        sys = DisaggregatedPDServingSystem(cfg)
        prompt = np.random.randn(300, 2, 16).astype(np.float32)
        res = sys.serve_request("test-req-e2e", prompt, decode_steps=5)

        assert res["chunks"] == 3
        assert res["prompt_length"] == 300
        assert len(res["decode_latencies_ms"]) == 5
        assert res["ttft_ms"] > 0
        assert res["mean_tpot_ms"] > 0

    def test_cli_subcommands(self):
        cmd_status = subprocess.run([str(BIN_PATH), "status"], capture_output=True, text=True)
        assert cmd_status.returncode == 0
        assert "Disaggregated Prefill-Decode" in cmd_status.stdout

        cmd_verify = subprocess.run([str(BIN_PATH), "verify"], capture_output=True, text=True)
        assert cmd_verify.returncode == 0
        assert "All 4 mathematical invariants verified" in cmd_verify.stdout

        cmd_bench = subprocess.run(
            [str(BIN_PATH), "benchmark", "--prompt-len", "256", "--chunk-size", "128", "--decode-steps", "3"],
            capture_output=True, text=True
        )
        assert cmd_bench.returncode == 0
        assert "Benchmark executed successfully" in cmd_bench.stdout
