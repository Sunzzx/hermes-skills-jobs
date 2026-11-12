---
name: omniquant-lwc-let-engine
description: OmniQuant LWC+LET PTQ calibration (ICLR 2024, arXiv:2308.13137).
category: inference
tags:
  - inference
  - omniquant
  - lwc
  - let
  - learnable-clipping
  - equivalent-transformation
  - weight-only
  - llm-serving
  - post-training
  - calibration
---

# OmniQuant LWC+LET Engine (arXiv:2308.13137)

OmniQuant is omnidirectional PTQ calibration for LLMs. Two absorbable transforms:

```
LWC:  W̃ = clip_γβ(W) then asymmetric group RTN
LET:  X'=(X−δ)⊘s ,  W'=W⊙s ,  b'=b+Wδ   (exact FP equivalence)
```

- **LWC** learns per-group (γ, β) clip hinges via golden-section search on recon MSE.
- **LET** generalizes SmoothQuant with *learned* channel scales/shifts + block loss.
- After fuse into adjacent LayerNorm/Linear: **zero runtime overhead** vs plain RTN.
- Ablations: RTN4, SmoothQuant-α, LWC-only, full OmniQuant.

## CLI Usage

```bash
# Ablation ledger: RTN / SmoothQuant / LWC / OmniQuant
omniquant-lwc-let-engine status --rows 32 --cols 32

# Weight/activation outlier + Hessian cond proxy
omniquant-lwc-let-engine calibrate --cols 64 --batch 32

# Fit LWC γ/β only
omniquant-lwc-let-engine lwc --rows 32 --iters 8

# Fit LET scales/shifts + FP equivalence check
omniquant-lwc-let-engine let --alpha 0.5 --refine-iters 5

# Full OmniQuant vs RTN
omniquant-lwc-let-engine quantize --rows 32 --cols 32

# SmoothQuant-α ablation
omniquant-lwc-let-engine smoothquant --alpha 0.5

# Dim × bit sweep
omniquant-lwc-let-engine benchmark --dims 16 24 32 --bits-list 2 3 4 8

# Invariant battery
omniquant-lwc-let-engine verify --trials 3

# Pack + 7B footprint theory
omniquant-lwc-let-engine pack --rows 32 --cols 32
omniquant-lwc-let-engine theory
```

## Architecture

- Pure Python, zero third-party deps (Hermes evolution standard).
- Group asymmetric RTN baseline + LWC clip search + LET closed-form/refine.
- Block output MSE/SQNR/cosine ledgers; LET FP equivalence must be ~0.
- Bit-rate = index bits + fp16 scale/zero; LWC/LET params fuse away at export.

## When to Use

- W4/W3/W2 weight-only PTQ research needing learnable clip + channel transforms.
- Comparing OmniQuant vs SmoothQuant-α vs GPTQ/AWQ/QuaRot scalar paths.
- Teaching absorbable equivalent transformations for quantization-friendly weights.

## Verification

```bash
python -m pytest tests/test_omniquant_lwc_let_engine.py -q
omniquant-lwc-let-engine verify --trials 3
```
