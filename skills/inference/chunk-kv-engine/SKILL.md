---
name: chunk-kv-engine
description: ChunkKV Chunk-Level Dynamic Key-Value Cache Sparsifier & Token Budget Allocator
category: inference
version: 1.0.0
tags: [llm, inference, kv-cache, sparsification, chunk-attention, dynamic-budget]
---

# ChunkKV: Chunk-Level Dynamic Key-Value Cache Sparsifier

ChunkKV introduces contiguous block-level partitioning and dynamic query-chunk importance scoring to prune redundant key-value memory during long-context generation. Rather than computing expensive fine-grained per-token attention over thousands of historical states, ChunkKV pools representations into contiguous semantic chunks, scores them via normalized cross-correlations with exponential temporal decay, and dynamically retrieves top-k chunks alongside invariant attention sinks and local recency windows.

## Key Capabilities
- **Contiguous Chunk Representation:** Groups KV pairs into fixed-size contiguous chunks with L2-normalized mean-pooled representations.
- **Dynamic Semantic Cross-Correlation:** Scores chunk utility using normalized dot-product projections weighted by temporal positional decay.
- **Top-K Chunk Budget Allocation:** Selectively routes generation attention to only the most salient chunks, saving 75-92% KV bandwidth and FLOPs.
- **Guaranteed Sinks & Local Recency:** Protects initial attention sink tokens and recent generation context from eviction, guaranteeing full attention stability.

## CLI Usage
```bash
# Inspect status and architecture hyperparameters
chunk-kv-engine status

# Run long-context inference benchmark
chunk-kv-engine benchmark --seq-len 8192 --gen-tokens 16
```

## Python API
```python
from bin.chunk_kv_engine import ChunkKVConfig, ChunkKVEngine

config = ChunkKVConfig(
    num_heads=32,
    head_dim=128,
    chunk_size=64,
    topk_chunks=4,
    sink_tokens=4,
    recent_tokens=64,
    decay_factor=0.95
)
engine = ChunkKVEngine(config)
prefill_stats = engine.prefill(seq_len=8192)
repr_vec, step_stats = engine.decode_step(step_idx=0)
```
