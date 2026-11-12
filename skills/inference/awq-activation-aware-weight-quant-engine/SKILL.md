---
name: awq-activation-aware-weight-quant-engine
description: AWQ activation-aware INT4 weight quant (MLSys 2024, arXiv:2306.00978).
category: inference
tags:
  - inference
  - awq
  - activation-aware
  - quantization
  - int4
  - llm-serving
  - post-training
  - alpha-search
---

# AWQ Activation-aware Weight Quant Engine (arXiv:2306.00978)

AWQ protects **salient weight channels** — those multiplying large-magnitude
activations — before group-wise low-bit quantization.

## Equivalent transform

Per input channel j, measure saliency $s_j = \mathbb{E}[|X_j|]$ and apply:

```
W'_{:,j} = W_{:,j} · s_j^{α}
X'_j     = X_j     / s_j^{α}
```

so `W' X' = W X` exactly in floating point. Then **group asymmetric RTN**
quantizes `W'`. Grid-search `α ∈ [0, 1]` to minimize output MSE:

```
α* = argmin_α  || W X − dequant(Q(W')) X' ||_F²
```

No Hessian inverse (cf. GPTQ). Online cost is one saliency pass + a cheap
alpha grid — the default INT4 path in vLLM, AutoAWQ, and many GGUF exporters.

## CLI Usage

```bash
# Theory ledger + live AWQ vs RTN demo
awq-activation-aware-weight-quant-engine status --rows 64 --cols 64 --bits 4

# Activation saliency only
awq-activation-aware-weight-quant-engine calibrate --cols 64 --n-cal 256

# Alpha grid search curve
awq-activation-aware-weight-quant-engine search --rows 64 --cols 64 --alpha-grid 20

# Full layer quant at α*
awq-activation-aware-weight-quant-engine quantize --rows 64 --cols 64 --bits 4 --group-size 32

# Dim/bit sweep
awq-activation-aware-weight-quant-engine benchmark --dims 32 48 64 --bits-list 3 4 8

# Invariant battery (finite MSE, beats RTN, cosine, alpha range)
awq-activation-aware-weight-quant-engine verify --trials 3

# Integer pack ledger
awq-activation-aware-weight-quant-engine pack --rows 32 --cols 32 --bits 4

# Compression theory table (incl. 7B footprint estimate)
awq-activation-aware-weight-quant-engine theory
```

## Architecture

- **Saliency**: mean-abs / RMS / max per input channel, geometric-mean normalized.
- **Scale transform**: column-scale W, row-scale X (exact fp equivalence).
- **Group RTN**: asymmetric (default) or symmetric INT2–INT8 with bit-rate ledger.
- **Alpha search**: uniform grid on [0,1]; pick min output-space MSE.
- **AWQEngine**: synthetic heavy-tailed X + boosted salient W channels; MSE/SQNR/cosine vs RTN (α=0).

Complementary to GPTQ (Hessian OBS residual redistribution) and QuaRot
(randomized Hadamard rotation for W+A INT4).
