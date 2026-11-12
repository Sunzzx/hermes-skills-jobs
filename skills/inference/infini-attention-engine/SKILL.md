---
name: infini-attention-engine
description: Infini-attention compressive infinite-context engine (Google arXiv:2404.07143).
category: inference
tags:
  - inference
  - infinite-context
  - compressive-memory
  - linear-attention
  - infini-attention
  - google-2024
  - long-context
---

# Infini-Attention: Leave No Context Behind (Google, arXiv:2404.07143)

Infini-attention is a compressive infinite-context architecture that binds **local sliding-window softmax attention** with a **fixed-size linear-attention memory**. Long-range context is folded into an O(d_k × d_v) matrix and normalizer vector, so RAM stays constant while sequence length grows without bound.

## Theoretical Foundations

Standard dense KV caches grow as O(L · n_layers · n_heads · d). Infini-attention instead maintains:

1. **Local window (W tokens):** classic scaled-dot-product attention over the most recent W keys/values — preserves sharp local syntax and recency bias.
2. **Compressive memory M ∈ R^{d_k×d_v} and normalizer z ∈ R^{d_k}:** updated with the delta rule  
   `M ← M + σ(K)ᵀV`, `z ← z + σ(K)` where `σ(x) = ELU(x) + 1` (strictly positive kernel).
3. **Memory retrieval:** `A_mem = σ(Q) M / (σ(Q)·z + ε)`.
4. **Gated blend:** `O = (1 − g) · A_local + g · A_mem` with configurable/learned gate g.

Evicted local tokens are absorbed into M, so nothing is hard-dropped — only compressed.

## CLI Usage

```bash
# Theoretical fixed-memory vs dense KV at 1K…10M tokens
infini-attention-engine status --num-layers 32 --num-heads 32 --head-dim 128 --local-window 64

# Prefill + decode microbenchmark
infini-attention-engine benchmark --seq-len 256 --decode-steps 16 --num-layers 2 --num-heads 4

# Invariant verification (constant footprint, bounded window, finite memory)
infini-attention-engine verify --seq-len 128 --local-window 16
```

## Architecture

- **InfiniAttentionKernel**: pure-Python ELU+1 kernel, delta memory update, local softmax, gated blend, theoretical byte scaling.
- **InfiniAttentionEngine**: multi-layer multi-head streaming engine with causal retrieve-then-update, ring-buffer eviction into compressive store, and peak-norm stabilization for long runs.
- **CLI**: `status` / `benchmark` / `verify` — JSON stdout, zero external deps.

## When to Use

- Modeling or prototyping infinite / multi-million-token context without KV RAM blow-up.
- Comparing compressive memory vs sliding-window / StreamingLLM / H2O eviction policies.
- Teaching linear-attention + local-softmax hybrid designs.

Don't use for: training full Transformer weights (this is an inference/memory engine, not a trainer).
