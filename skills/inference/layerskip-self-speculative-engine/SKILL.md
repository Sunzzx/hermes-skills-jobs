---
name: layerskip-self-speculative-engine
description: Meta LayerSkip early-exit + self-speculative decoding (arXiv:2404.15690).
category: inference
tags:
  - inference
  - layerskip
  - early-exit
  - self-speculative-decoding
  - meta-fair
  - speculative-decoding
  - dynamic-depth
  - transformer
---

# LayerSkip Self-Speculative Engine (Meta FAIR, arXiv:2404.15690)

LayerSkip enables **two complementary speedups** on a single shared transformer:

1. **Early-exit inference** — intermediate classifiers halt "easy" tokens before
   full depth, so average FLOPs scale with *exit layer*, not `L`.
2. **Self-speculative decoding** — the first `E` layers draft multi-token
   candidates; the remaining `L-E` layers verify them. No auxiliary draft
   model (unlike EAGLE/Medusa); weights are shared.

Layer-dropout during training is what makes early exits reliable; this engine
models that prior and exposes exact FLOP / acceptance ledgers.

## Theoretical Foundations

Dense decode cost per token:

```
FLOPs_dense ≈ L · (Attn(kv) + MLP) + Head
```

Early-exit with per-layer conditional exit probability `p_i`:

```
FLOPs_EE ≈ Σ_i  S_i · (Attn + MLP + ExitHead)
S_0 = 1,  S_{i+1} = S_i · (1 - p_i)
```

Self-speculation with draft depth `E`, window `k`, acceptance `α`:

```
γ = Σ_{i=1..k} α^i          # expected accepted draft tokens
tokens/round ≈ γ + 1
speedup ≈ FLOPs_dense / (round_cost / tokens_per_round)
```

## CLI Usage

```bash
# Theory tables (early-exit + self-spec across α)
layerskip-self-speculative-engine status --layers 12 --d-model 1024 --heads 16

# Early-exit prefill microbenchmark
layerskip-self-speculative-engine benchmark --seq-lens 16 32 64

# Invariant verification
layerskip-self-speculative-engine verify --seq-len 32

# Growing-sequence exit timeline
layerskip-self-speculative-engine simulate --tokens 64 --steps 4

# Per-token exit layer map
layerskip-self-speculative-engine exit-map --seq-len 16

# Self-speculative rollout (optional fixed acceptance α)
layerskip-self-speculative-engine speculate --prompt-len 16 --gen-tokens 8 --force-alpha 0.8
```

## Architecture

- **ExitHead**: per-layer vocab projection + confidence head (CE-style).
- **TransformerBlock**: pre-norm MHA + GELU MLP + KV cache + exit head.
- **LayerSkipEngine**: early-exit prefill, self-speculative decode, FLOP ledger.
- **CLI**: `status` / `benchmark` / `verify` / `simulate` / `exit-map` / `speculate`.

## When to Use

- Prototyping early-exit policies and confidence thresholds without full training.
- Comparing self-speculation (shared weights) vs separate draft models (EAGLE).
- Teaching dynamic-depth inference and speculative acceptance math.

Don't use for: training LayerSkip LLMs end-to-end or replacing production
serving runtimes (vLLM/SGLang) — this is an architecture + ledger engine.
