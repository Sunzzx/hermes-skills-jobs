---
name: triad-engine
description: "TRIAD asynchronous speculative decoding and dynamic chunk-level KV cache pruning engine for high-throughput, low-latency LLM inference."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Frontier-AI, SpeculativeDecoding, KVCache, Pruning, TRIAD, InferenceEngine, LLM]
---

# TRIAD: Asynchronous Speculative Decoding & Dynamic Chunk KV Pruning Engine

## Overview
**TRIAD** (Towards Robust and Efficient Inference with Asynchronous Speculation and Pruning, 2024) unifies speculative generation with hierarchical memory pruning:
1. **Asynchronous Verification**: Decouples draft token expansion from target verification passes via sliding execution queues, overlapping pipeline latency.
2. **Dynamic Chunk-Level KV Pruning**: Partitions intermediate context into coherent token chunks (blocks) and scores relevance using normalized query-key dot products.
3. **Anchor & Local Conservation**: Invariantly preserves attention sinks (critical for softmax denominator stability) and the sliding local attention window, while dynamically evicting redundant middle-context chunks.
4. **Lossless Multi-Candidate Acceptance**: Evaluates candidate streams via rejection sampling with residual distribution fallback, preventing distribution drift.

## Theoretical Foundations
- **Attention Sink Conservation**: Maintains $\tau_{sink}$ tokens at sequence start to absorb massive residual attention scores.
- **Chunk Score Function**:
  $$S(C_j) = \frac{1}{|C_j|} \sum_{t \in C_j} \frac{\mathbf{q}^T \mathbf{k}_t}{\|\mathbf{q}\| \|\mathbf{k}_t\|}$$
  Chunks with lower cumulative relevance are evicted when cache capacity hits $B_{kv}$.
- **Speculative Speedup**: Increases token emission rate from $1.0$ to $(1 + \alpha \cdot \gamma)$ tokens per step where $\gamma$ is speculation depth and $\alpha$ is candidate acceptance rate.

## CLI Commands

### Benchmark Speculative Throughput
```bash
triad-engine benchmark --steps 50 --prompt-len 128 --gamma 4 --kv-budget 128
```

### Formally Verify Pruning Invariants
```bash
triad-engine verify --budget 64
```

## Python API Integration
```python
from triad_engine import TRIADConfig, TRIADEngine

config = TRIADConfig(kv_budget=128, chunk_size=8, speculative_window=4)
engine = TRIADEngine(config)

# Run full benchmark
results = engine.benchmark(num_steps=50, prompt_len=128, gamma=4)
print(f"Tokens Per Step: {results['tokens_per_step']}")
print(f"Effective Speedup: {results['effective_speedup']}x")
```
