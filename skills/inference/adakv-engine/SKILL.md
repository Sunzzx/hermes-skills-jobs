---
name: adakv-engine
description: AdaKV Adaptive Budget Allocation for KV Cache Compression (NeurIPS 2024)
category: inference
version: 1.0.0
tags: [llm, inference, kv-cache, adakv, adaptive-budget, neurips-2024, memory-efficiency]
cli: adakv-engine
---

# AdaKV: Adaptive Budget Allocation for KV Cache Compression

## Overview

**AdaKV** (NeurIPS 2024 / arXiv:2407.11550) introduces dynamic, cross-layer and cross-head adaptive budget allocation for Large Language Model (LLM) Key-Value cache compression. Rather than enforcing fixed, uniform token budgets across all attention heads (as in standard SnapKV or H2O), AdaKV evaluates head importance patterns during the observation window and adaptively allocates more token budget to pivotal retrieval heads while aggressively pruning redundant streaming heads.

### Key Capabilities:
1. **Dynamic Layer & Head Budget Allocation**: Uses prompt-specific attention entropy and variance scoring to solve an adaptive budget distribution across all $(L, H)$ heads.
2. **Pivotal Head Retention**: Preserves long-range associative recall by granting higher capacity to critical retrieval heads.
3. **50% - 75% Memory Reduction**: Halves KV memory bandwidth and storage footprints while maintaining generation quality.
4. **High-Throughput Streaming Decoding**: Low-latency autoregressive token eviction preserving attention sinks and dynamic working sets.

---

## CLI Usage

```bash
# View theoretical budget allocation and savings model
adakv-engine status --budget-ratio 0.5 --num-layers 32 --num-heads 32

# Run prefill and token-by-token generation benchmark
adakv-engine benchmark --seq-len 1024 --decode-steps 32 --budget-ratio 0.5
```

---

## Python API

```python
from importlib.machinery import SourceFileLoader
import os

bin_path = os.path.expanduser("~/.local/bin/adakv-engine")
adakv_mod = SourceFileLoader("adakv_engine", bin_path).load_module()

config = adakv_mod.AdaKVConfig(
    num_layers=4,
    num_heads=8,
    head_dim=64,
    total_budget_ratio=0.5  # Retain ~50% average budget across heads
)
engine = adakv_mod.AdaKVEngine(config)

# Prefill prompt tokens
stats = engine.prefill(seq_len=512)
print(f"Memory Savings: {stats['memory_savings_pct']}% | Compression Ratio: {stats['compression_ratio']}x")

# Autoregressive decoding step
step_info = engine.decode_step()
print(f"Step Latency: {step_info['step_latency_us']} us")
```
