---
name: yoco-cache-engine
description: YOCO decoder-decoder single shared KV cache (Meta FAIR arXiv:2405.05254).
category: inference
tags:
  - inference
  - yoco
  - kv-cache
  - decoder-decoder
  - meta-fair
  - long-context
  - gated-retention
---

# YOCO: You Only Cache Once (Meta FAIR, arXiv:2405.05254)

YOCO is a **decoder-decoder** LLM architecture that emits a **single shared global KV cache once** from an early self-decoder stack, then lets every late cross-decoder layer read that same store. Cross-half layers carry **zero private KV**, so peak KV RAM drops by roughly 2× at a 50/50 split (and ~L× asymptotically vs depth when locals are bounded).

## Theoretical Foundations

Standard causal decoders cache K/V on **every** layer: memory scales as `O(L · H · D · T)`.

YOCO splits depth into:

1. **Self-decoder (`L_s` layers):** efficient causal attention (sliding-window softmax + optional gated retention). The **final** self layer emits global `(K, V)` **once** into a shared store of length `T`.
2. **Cross-decoder (`L_c` layers):** each token forms queries only and **cross-attends** to the shared global KV. No per-layer KV appends.

Effective cache:

- Shared global KV: `O(H · D · T)` (one layer equivalent)
- Self local windows: `O(L_s · H · D · W)` (bounded)
- Gated retention states: `O(L_s · H · D²)` (sequence-independent)
- Cross private KV: **0**

At long `T`, compression ratio approaches `L_s + L_c` vs a naive full stack.

## CLI Usage

```bash
# 32+32 layer LLaMA-class theoretical scaling
yoco-cache-engine status --self-layers 16 --cross-layers 16 --num-heads 32 --head-dim 128

# Prefill + decode microbenchmark
yoco-cache-engine benchmark --seq-len 128 --decode-steps 16 --self-layers 2 --cross-layers 2

# Invariant verification (shared-once, cross zero-KV, savings > 0)
yoco-cache-engine verify --seq-len 64 --decode-steps 32

# Live memory timeline while streaming tokens
yoco-cache-engine simulate --tokens 48
```

## Architecture

- **SelfDecoderLayer**: sliding-window causal softmax, gated retention `S ← γS + k⊗v`, final-layer global KV emit.
- **SharedGlobalKV**: single `H × T × D` key/value store consumed by all cross layers.
- **CrossDecoderLayer**: query-only cross-attention over shared KV; zero private cache growth.
- **YocoEngine**: prefill/decode orchestration, byte-level memory accounting vs standard decoder, theory tables.
- **CLI**: `status` / `benchmark` / `verify` / `simulate` — JSON stdout, zero external deps.

## When to Use

- Prototyping half-depth KV budgets without dropping global token-token connectivity.
- Comparing decoder-decoder YOCO vs pure eviction (H2O, StreamingLLM) or compressive memory (Infini-attention).
- Teaching gated retention + cross-attention hybrid designs.

Don't use for: training full YOCO weights end-to-end (this is an inference/memory architecture engine, not a trainer).
