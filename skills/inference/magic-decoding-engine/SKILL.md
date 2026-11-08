---
name: magic-decoding-engine
description: "MAGIC-Decoding memory-augmented speculative tree generation and saliency-guided KV cache pruning engine for high-throughput LLM inference."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Frontier-AI, SpeculativeDecoding, KVCache, MAGIC, TreeDecoding, InferenceEngine, LLM]
---

# MAGIC-Decoding: Memory-Augmented Generative Inference & Speculative Verification

## Overview
**MAGIC-Decoding** (Memory-Augmented Generative Inference via Speculative Candidate Trees & KV Saliency Pruning, 2024) unifies multi-branch speculative decoding with adaptive KV memory compression:

1. **Speculative Candidate Tree Generation**: Expands multi-branch draft token trees with topology tracking (parent-child indices and path cumulative probabilities), maximizing path diversity against the target LLM distribution.
2. **Saliency-Guided Adaptive KV Pruning**: Scores intermediate tokens via a hybrid of query-key cosine similarity and key-vector L2 magnitude, retaining only high-saliency middle context under a fixed budget.
3. **Attention Sink & Local Window Conservation**: Invariantly preserves attention sinks (critical for softmax denominator stability) and the sliding local attention window for syntactic coherence.
4. **Exact Unbiased Tree Verification**: Traverses the highest-probability candidate path using Leviathan-style rejection sampling with residual distribution fallback, preventing distribution drift.

## Theoretical Foundations
- **Attention Sink Conservation**: Maintains $\tau_{sink}$ tokens at sequence start to absorb residual attention mass.
- **Saliency Score Function**:
  $$S_i = \alpha \cdot \mathrm{CosSim}(\mathbf{q}, \mathbf{k}_i) + (1-\alpha) \cdot \frac{\|\mathbf{k}_i\|_2}{\max_j \|\mathbf{k}_j\|_2}$$
  Tokens with lower saliency in the middle context are evicted when cache capacity hits $B_{kv}$.
- **Tree Speculative Speedup**: Increases token emission rate from $1.0$ to $(1 + \alpha \cdot d)$ tokens per step where $d$ is tree depth and $\alpha$ is path acceptance rate.

## CLI Commands

```bash
# Formal invariant verification (sinks + recent window + budget)
magic-decoding-engine verify --budget 64

# Throughput + KV pruning benchmark
magic-decoding-engine benchmark --steps 50 --prompt-len 128 --depth 4 --branching 2 --kv-budget 128
```

## Integration
- Executable: `bin/magic-decoding-engine` (also installed to `~/.local/bin/magic-decoding-engine`)
- Tests: `tests/test_magic_decoding_engine.py`
- Category: `inference`
