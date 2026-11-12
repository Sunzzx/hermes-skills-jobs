---
name: spinquant-learned-rotation-engine
description: SpinQuant learned orthogonal rotation PTQ (Meta FAIR, arXiv:2405.16406).
category: inference
tags:
  - inference
  - spinquant
  - learned-rotation
  - cayley
  - hadamard
  - quantization
  - w4a8
  - llm-serving
  - post-training
  - stiefel
---

# SpinQuant Learned Rotation Engine (arXiv:2405.16406)

SpinQuant learns an **orthogonal rotation** R on the Stiefel manifold so that
after the equivalent transform both weights and activations become far more
quantization-friendly than identity or random Hadamard (QuaRot).

## Equivalent transform

```
W' = W R
X' = R^T X
```

so `W' X' = W X` exactly in floating point (`R^T R = I`). Then group RTN
quantizes `W'` (INT2/3/4/8) and `X'` (INT4/8). SpinQuant optimizes R to
minimize output-space MSE:

```
R* = argmin_{R^T R = I}  || W X − dequant(Q(W R)) dequant(Q(R^T X)) ||_F²
```

Cayley parameterization: skew-symmetric A gives always-orthogonal

```
R = (I − A)(I + A)^{−1}
```

Rotations fuse into adjacent LayerNorm / linear weights at deploy — **zero
runtime overhead**. Complements QuaRot (random Hadamard), GPTQ/AWQ
(weight-only), and SmoothQuant (diagonal smooth).

## CLI Usage

```bash
# Theory ledger + live SpinQuant vs identity / Hadamard demo
spinquant-learned-rotation-engine status --rows 32 --cols 32

# Channel outlier stats
spinquant-learned-rotation-engine calibrate --cols 64 --n-cal 256

# Probe a rotation (fp equivalence + orth error + quant MSE)
spinquant-learned-rotation-engine rotate --rotation cayley --cols 32

# Cayley gradient optimize R*
spinquant-learned-rotation-engine optimize --rows 32 --cols 32 --steps 25 --w-bits 4

# Full quant path (alias of optimize)
spinquant-learned-rotation-engine quantize --rows 32 --cols 32 --w-bits 4 --a-bits 8

# Dim/bit sweep
spinquant-learned-rotation-engine benchmark --dims 16 24 32 --bits-list 4 8

# Invariant battery (fp equiv, orth, beats identity, cosine)
spinquant-learned-rotation-engine verify --trials 3

# Integer pack ledger
spinquant-learned-rotation-engine pack --rows 32 --cols 32

# Compression / 7B footprint theory table
spinquant-learned-rotation-engine theory
```

## Architecture

- **Cayley Stiefel params**: skew A → orthogonal R; finite-diff GD on upper triangle.
- **Givens product / randomized Hadamard / identity** ablation baselines.
- **Exact equivalent transform**: W'=WR, X'=R^T X (fp identity preserved).
- **Joint W+A RTN**: symmetric INT2–8 weights + per-tensor or per-channel activations.
- **Loss**: output-space MSE after quant vs identity and QuaRot-style Hadamard.
- **Fusible**: R absorbed into adjacent linears → deploy cost = plain W4A8 GEMM.
