---
name: sparq-attention-engine
description: "SparQ-Attention bandwidth-efficient LLM decoding engine via top-k feature and sequence dual sparsification."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Frontier-AI, SparQ, Sparsification, KVCache, InferenceEngine, MemoryBandwidth, LLM]
---

# SparQ-Attention: Bandwidth-Efficient LLM Inference via Adaptive Feature & Sequence Sparsification

## Overview
**SparQ-Attention** (Ribera et al., 2024) is a state-of-the-art inference acceleration technique that alleviates the primary performance bottleneck of autoregressive LLM decoding: **memory bandwidth saturation during attention evaluation**.

Rather than transferring the full Key-Value cache across memory channels on every decoding step, SparQ employs a dual-stage adaptive sparsification hierarchy:

1. **Feature-Dimension Pre-Scoring (Component 1)**: Identifies the top-$r$ largest magnitude components of the incoming query vector $\mathbf{q}$. It evaluates approximate query-key inner products reading only the corresponding $r$ feature dimensions of Key cache $\mathbf{K}[:, \text{top}_r]$, slashing initial memory read bandwidth by up to 75%.
2. **Sequence-Level Top-$k$ Selection (Component 2)**: Uses the approximate attention scores to identify the top-$k$ most salient context tokens across the sequence length.
3. **Attention Sink & Recent Window Invariants**: Automatically and unconditionally preserves attention sinks (initial $\tau_{sink}$ tokens, maintaining softmax denominator stability) and the sliding local attention window (preserving recent syntactic coherence).
4. **Exact Full-Dimensional Attention**: Executes exact, full-dimensional attention pooling strictly over the gathered tokens (invariant sinks + recent tokens + top-$k$ candidates).

## Algorithmic Formulation

### 1. Approximate Feature Scoring
Let $\mathbf{q} \in \mathbb{R}^d$ and $\mathbf{K} \in \mathbb{R}^{L \times d}$. Let $\Omega_r = \operatorname{argtopr}(|\mathbf{q}|)$ be the indices of the $r$ largest absolute components of $\mathbf{q}$.
$$\tilde{\mathbf{S}} = \frac{d}{r \sqrt{d}} \sum_{j \in \Omega_r} \mathbf{K}[:, j] \cdot \mathbf{q}[j]$$

### 2. Sequence Candidate Gathering
$$\mathcal{I} = \mathcal{I}_{sink} \cup \mathcal{I}_{recent} \cup \operatorname{argtopk}_{i \notin \mathcal{I}_{sink} \cup \mathcal{I}_{recent}}(\tilde{\mathbf{S}}_i)$$

### 3. Exact Attention Pooling
$$\mathbf{A}_{\mathcal{I}} = \operatorname{softmax}\left(\frac{\mathbf{K}[\mathcal{I}] \mathbf{q}}{\sqrt{d}}\right), \quad \mathbf{o} = \mathbf{A}_{\mathcal{I}} \mathbf{V}[\mathcal{I}]$$

## Theoretical Memory Bandwidth Gains
- **Dense Attention Transfer**: $2 L d \times 4$ bytes per head.
- **SparQ Transfer**: $(L r + 2 |\mathcal{I}| d) \times 4$ bytes per head.
- **Effective Bandwidth Savings**: $\sim 40\% - 70\%$ reduction in memory bus saturation during long-context generation without loss in generation perplexity.

## CLI Commands

```bash
# Formally verify invariants (sinks + recent window + budget conservation)
sparq-attention-engine verify --budget 64

# Run decoding throughput and memory bandwidth savings benchmark
sparq-attention-engine benchmark --steps 50 --prompt-len 256 --budget 128 --top-k 32 --top-r-ratio 0.25
```

## Integration

SparQ-Attention integrates directly into Hermes inference pipelines, serving as an adaptive memory-bandwidth optimizer for high-concurrency decoding swarms.
