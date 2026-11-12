---
name: mixture-of-depths-engine
description: DeepMind MoD expert-choice dynamic token compute router (arXiv:2404.02258).
category: inference
tags:
  - inference
  - mixture-of-depths
  - dynamic-compute
  - expert-choice
  - deepmind
  - sparse-compute
  - transformer
---

# Mixture-of-Depths Engine (Google DeepMind, arXiv:2404.02258)

Mixture-of-Depths (MoD) **dynamically allocates transformer FLOPs across tokens**.
Each layer runs an **expert-choice router** that admits only a fixed capacity
fraction of positions (e.g. 12.5%) into the full self-attention + MLP block.
Unselected tokens take a pure **residual identity** path — zero block compute.

Unlike Mixture-of-Experts (which sparsifies *experts* while every token still
pays a block), MoD sparsifies *depth participation* per token, cutting total
compute roughly proportional to capacity `C`.

## Theoretical Foundations

Dense stack cost over `L` layers and `T` tokens:

```
FLOPs_dense ≈ L · (Attn(T) + MLP(T))
```

MoD with capacity `C` (expert-choice top-`k`, `k = ⌈C·T⌉`):

```
FLOPs_MoD ≈ L · (Router(T) + Attn(k) + MLP(k))
```

Asymptotic compute ratio → `C` (plus tiny router overhead). At `C = 1/8` this
is ~8× fewer block FLOPs with a static, balanced budget — no token-choice load
imbalance.

Routing is **expert-choice** (capacity selects tokens), not token-choice
(tokens select experts). That guarantees exact per-layer FLOP caps.

## CLI Usage

```bash
# Theory tables across sequence lengths
mixture-of-depths-engine status --layers 12 --d-model 1024 --heads 16 --capacity 0.125

# Prefill microbenchmark
mixture-of-depths-engine benchmark --seq-lens 32 64 128 --layers 4 --d-model 128 --heads 4

# Invariant verification (capacity exact, skip identity, finite states)
mixture-of-depths-engine verify --seq-len 64

# Growing-sequence compute timeline
mixture-of-depths-engine simulate --tokens 64 --steps 4

# Dump per-layer selected/skipped index maps + scores
mixture-of-depths-engine route --seq-len 16
```

## Architecture

- **DepthRouter**: RMSNorm → linear logit → temperature; top-`k` expert-choice.
- **MoDBlock**: gather selected → pre-norm causal MHA → MLP → scatter; skipped = identity.
- **MoDEngine**: stacked blocks, FLOP ledger vs dense, memory/param report, verify suite.
- **CLI**: `status` / `benchmark` / `verify` / `simulate` / `route` — JSON stdout, zero deps.

## When to Use

- Prototyping static-budget sparse depth compute without MoE expert sharding.
- Comparing FLOP savings of MoD vs dense / MoE / early-exit schemes.
- Teaching expert-choice routing and residual skip identity paths.

Don't use for: training full MoD LLMs end-to-end (this is an inference/compute
architecture engine and ledger, not a trainer).
