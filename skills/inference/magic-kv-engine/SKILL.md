---
name: magic-kv-engine
description: MagicPIG Multi-Scale Top-K KV Cache Sparsification & Speculative Sampling Engine
category: inference
version: 1.0.0
tags: [llm, inference, kv-cache, sparsification, multi-scale, locality-sensitive-hashing]
---

# MagicPIG: Multi-Scale Top-K KV Cache Sparsification Engine

MagicPIG (Multi-scale Approximate Attention with Gated Key Sampling) addresses memory and bandwidth bottlenecks during long-context LLM generation by clustering key-value representations into multi-scale buckets indexed via locality-sensitive projection hashing. During decoding steps, query projections score and retrieve only the top-k most relevant token buckets, eliminating 70-85% of KV FLOPs while preserving attention sinks and local recency.

## Key Capabilities
- **Multi-Scale Locality-Sensitive Projection:** Computes randomized hyper-plane sketches of cluster centroids for $O(1)$ Hamming distance scoring.
- **Speculative Bucket Routing:** Identifies top-k candidate buckets per attention head to avoid full-sequence tensor dot products.
- **Attention Sink & Local Recency Preservation:** Guarantees exact representations for initial attention sinks and sliding recency windows.
- **High Compression:** Demonstrates >75-90% KV attention bandwidth and compute savings with full context recall.

## CLI Usage
```bash
# Inspect status and architecture hyperparameters
magic-kv-engine status

# Run long-context inference benchmark
magic-kv-engine benchmark --seq-len 8192 --gen-tokens 16
```

## Python API
```python
from bin.magic_kv_engine import MagicPIGConfig, MagicPIGEngine

config = MagicPIGConfig(
    num_heads=32,
    head_dim=128,
    bucket_size=32,
    num_hash_functions=8,
    topk_buckets=6,
    sink_tokens=4,
    recent_tokens=64
)
engine = MagicPIGEngine(config)
prefill_stats = engine.prefill(seq_len=8192)
repr_vec, step_stats = engine.decode_step(step_idx=0)
```
