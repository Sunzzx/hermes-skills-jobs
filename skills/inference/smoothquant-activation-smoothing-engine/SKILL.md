---
name: smoothquant-activation-smoothing-engine
description: SmoothQuant W8A8 activation smoothing PTQ (ICML 2023, arXiv:2211.10438).
category: inference
tags:
  - inference
  - smoothquant
  - activation-smoothing
  - quantization
  - w8a8
  - int8
  - llm-serving
  - post-training
  - outlier-migration
---

# SmoothQuant Activation-Smoothing Engine (arXiv:2211.10438)

SmoothQuant migrates **activation outliers into weights** so both fit INT8
round-to-nearest — enabling efficient W8A8 GEMM without fine-tuning.

## Equivalent transform

Per input channel j, measure $m_j^X = \max|X_j|$ and $m_j^W = \max|W_{:,j}|$:

```
s_j = (m_j^X)^α  /  (m_j^W)^{1-α}

X'_j     = X_j     / s_j
W'_{:,j} = W_{:,j} * s_j
```

so `W' X' = W X` exactly in floating point. Then **symmetric RTN** quantizes
both `W'` (per-row/group) and `X'` (per-tensor or per-channel). Grid-search
`α ∈ [0, 1]` to minimize joint W8A8 output MSE:

```
α* = argmin_α  || W X − dequant(Q(W')) dequant(Q(X')) ||_F²
```

α≈0.5–0.75 is robust across OPT/BLOOM/LLaMA. Online cost is one calibration
pass for channel maxes; scales fold into prior LayerNorm.

## CLI Usage

```bash
# Theory ledger + live SmoothQuant vs naive W8A8 demo
smoothquant-activation-smoothing-engine status --rows 64 --cols 64

# Channel max stats + s_j at fixed α
smoothquant-activation-smoothing-engine calibrate --cols 64 --n-cal 256 --alpha 0.5

# Alpha grid search curve
smoothquant-activation-smoothing-engine search --rows 64 --cols 64 --alpha-grid 21

# Full W8A8 layer quant at α*
smoothquant-activation-smoothing-engine quantize --rows 64 --cols 64 --w-bits 8 --a-bits 8

# Dim/bit sweep (W4A8, W8A8, …)
smoothquant-activation-smoothing-engine benchmark --dims 32 48 64 --bits-list 4 8

# Invariant battery (fp equivalence, beats naive, cosine, alpha range)
smoothquant-activation-smoothing-engine verify --trials 3

# Integer pack ledger
smoothquant-activation-smoothing-engine pack --rows 32 --cols 32

# Compression / 7B footprint theory table
smoothquant-activation-smoothing-engine theory
```

## Architecture

- **Outlier-aware X**: synthetic calibration with rare channel spikes (the SmoothQuant pathology).
- **Scale formula**: `s_j = max(|X_j|)^α / max(|W_j|)^{1-α}` (max / RMS / p99 act stats).
- **Exact smooth**: column-scale W, row-scale X (fp identity preserved).
- **Joint W+A RTN**: symmetric INT4/6/8 weights + per-tensor or per-channel activations.
- **Alpha search**: uniform grid on [0,1]; min output-space MSE vs naive (unsmoothed) W8A8.
- **Ablations**: W-only / A-only MSE, dynamic-range compression ratios, bit-rate ledger.

Complementary to AWQ (salient-weight protection for INT4-W), GPTQ (Hessian OBS),
and QuaRot (randomized Hadamard for W+A INT4).
