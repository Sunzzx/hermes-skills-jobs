---
name: gptq-hessian-weight-quant-engine
description: GPTQ Hessian OBS one-shot weight quant (ICLR 2023, arXiv:2210.17323).
category: inference
tags:
  - inference
  - gptq
  - hessian
  - quantization
  - int4
  - optimal-brain-surgeon
  - post-training
  - llm-serving
---

# GPTQ Hessian Weight Quant Engine (arXiv:2210.17323)

GPTQ performs **layer-wise post-training weight quantization** that minimizes
expected squared output error under calibration activations X:

```
min_{W_q}  || W X − W_q X ||_F²
```

via a second-order Optimal Brain Surgeon (OBS) approximation. The quadratic
Hessian is

```
H = 2 X Xᵀ
```

dampened as `H ← H + λ · mean(diag(H)) · I` for numerical stability.

## Algorithm

1. **Calibrate** Hessian from activation matrix X (in_features × n_cal).
2. **Invert** H via Cholesky (SPD) with Gauss–Jordan fallback.
3. **Act-order** (optional): permute columns by descending |H_ii|.
4. **Block column quantize**: round-to-nearest per group scale (INT2/3/4/8).
5. **OBS residual update**: push quantization error into remaining free weights
   using the corresponding row of H⁻¹:
   ```
   δ_{k>j} -= ((w_j − ŵ_j) / H⁻¹_{jj}) · H⁻¹_{jk}
   ```
6. **Pack** integer grid + fp16 group scales; report bit-rate & compression.

Complementary to QuaRot (rotation-based W+A INT4): GPTQ is the classic
**weight-only** one-shot OBS quantizer used in production LLM serving.

## CLI Usage

```bash
# Theory ledger + live GPTQ vs RTN demo
gptq-hessian-weight-quant-engine status --rows 64 --cols 64 --bits 4

# Accumulate calibration Hessian only
gptq-hessian-weight-quant-engine calibrate --cols 64 --n-cal 256

# Full layer quant + metrics
gptq-hessian-weight-quant-engine quantize --rows 64 --cols 64 --bits 4 --group-size 128

# Dim/bit sweep
gptq-hessian-weight-quant-engine benchmark --dims 32 48 64 --bits-list 3 4 8

# Invariant battery (Cholesky inverse residual, finite weights, cosine)
gptq-hessian-weight-quant-engine verify --trials 3

# Integer pack ledger
gptq-hessian-weight-quant-engine pack --rows 32 --cols 32 --bits 4

# Compression theory table (incl. 7B footprint estimate)
gptq-hessian-weight-quant-engine theory
```

## Architecture

- **Dense linear algebra**: GEMM, Cholesky, Cholesky-inverse, Gauss–Jordan.
- **RTN group quantizer**: symmetric/asymmetric INT2–INT8 with bit-rate ledger.
- **GPTQEngine**: synthetic heavy-tailed W & X, Hessian accumulate, act-order
  permutation, block OBS quant, output-space MSE/SQNR vs RTN baseline.
- **CLI**: `status` / `calibrate` / `quantize` / `benchmark` / `verify` /
  `pack` / `theory`.

## Paper

Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers*, ICLR 2023 (arXiv:2210.17323).
