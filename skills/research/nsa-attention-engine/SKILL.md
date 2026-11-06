---
name: nsa-attention-engine
description: "DeepSeek-V3 Native Sparse Attention (NSA) engine featuring multi-scale compressed token attention, dynamic top-k coarse block routing, and sliding-window local context."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Frontier-AI, DeepSeek-V3, SparseAttention, NativeSparseAttention, AttentionKernels, NSA, LongContext]
---

# DeepSeek-V3 Native Sparse Attention (NSA) Engine

## Overview
DeepSeek-V3 introduces **Native Sparse Attention (NSA)** to compress memory footprint and reduce quadratic attention FLOPs in extreme context windows (128K+ tokens). NSA replaces uniform dense self-attention with a multi-scale hierarchical routing mechanism:
1. **Compressed Token Attention**: Coarse-grained key/value representations pooled into fixed-size blocks (e.g. block size 32 or 64) via average pooling.
2. **Selected Block Attention**: Token queries dynamically attend to the top-$k$ most relevant coarse blocks using coarse affinity scoring.
3. **Sliding Window Attention**: Maintains full precision, dense local attention within a local temporal window (e.g. 128 to 256 tokens) to capture high-frequency syntactic structure.

## Key Features
- **Multi-Scale KV Compression**: Deterministic block-level key/value pooling reducing coarse memory traffic by $O(B)$.
- **Top-K Coarse Block Routing**: Causally masked block selection per attention head, compatible with Grouped Query Attention (GQA).
- **Exact Causal Fusion**: Combines selected block indices with local sliding window tokens into a unified, normalized softmax attention distribution.
- **Sparsity & Speedup Profiling**: Built-in CLI benchmarks measuring attention sparsity percentage, FLOPs reduction, and wall-clock execution time.

## CLI Reference

### Run Benchmark Simulation
```bash
nsa-attention-engine benchmark --seq-len 1024 --num-heads 16 --num-kv-heads 4 --block-size 64 --top-k 4 --window 128
```

### Numerical Property Verification
```bash
nsa-attention-engine verify --seq-len 512
```

## Python API Integration
```python
import numpy as np
from nsa_attention_engine import NSAConfig, NativeSparseAttentionEngine

cfg = NSAConfig(
    seq_len=2048,
    num_heads=16,
    num_kv_heads=4,
    head_dim=64,
    block_size=64,
    top_k_blocks=4,
    sliding_window_size=256,
)
engine = NativeSparseAttentionEngine(cfg)

Q = np.random.randn(2048, 16, 64).astype(np.float32)
K = np.random.randn(2048, 4, 64).astype(np.float32)
V = np.random.randn(2048, 4, 64).astype(np.float32)

output, stats = engine.forward(Q, K, V)
print(f"Sparsity: {stats['sparsity_ratio']*100:.2f}%, FLOPs Reduction: {stats['theoretical_speedup']:.2f}x")
```
