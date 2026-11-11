---
name: nsa-engine
description: "Native Sparse Attention (NSA): Multi-scale hierarchical dynamic KV attention engine combining compressed tokens, selected fine-grained blocks, and sliding window for long context LLMs."
version: 1.0.0
category: inference
tags:
  - kv-cache
  - sparse-attention
  - deepseek-v3
  - nsa
  - llm-inference
---

# Native Sparse Attention (NSA) Engine

Hardware-aligned native sparse attention engine designed for long-context inference scaling, based on the **Native Sparse Attention (NSA)** paradigm (DeepSeek-V3 / DSA architecture).

NSA replaces dense KV computation with a 3-way multi-scale hierarchical routing mechanism:
1. **Compressed Tokens (Coarse-Grained Pooling):** Stride-based block averaging reduces context footprint while preserving global semantic anchors.
2. **Selected Fine-Grained Tokens:** Dynamic top-k block scoring routes attention strictly to the highest-affinity token clusters.
3. **Sliding Window:** Preserves exact continuous local attention for prompt tail and immediate transitions.

## CLI Usage

```bash
# Run default simulation
nsa-engine

# Benchmark performance across long context (4096 tokens)
nsa-engine bench --seq-len 4096 --d-model 64 --block-size 16 --top-k 4

# Inspect memory and attention routing breakdown
nsa-engine inspect --seq-len 4096
```
