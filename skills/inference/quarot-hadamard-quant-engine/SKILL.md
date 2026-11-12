---
name: quarot-hadamard-quant-engine
description: QuaRot outlier-free INT4 via randomized Hadamard rotation (arXiv:2404.00456).
category: inference
tags:
  - inference
  - quarot
  - hadamard
  - quantization
  - int4
  - outliers
  - fwht
  - llm-serving
---

# QuaRot Hadamard Quant Engine (arXiv:2404.00456)

QuaRot removes the activation-outlier barrier to **weight+activation INT4** by
multiplying features and weights with an orthogonal **randomized Hadamard**

```
Q = n^{-1/2} · H · diag(s)      s_i ∈ {±1}
```

Outlier mass is incoherently smeared across channels (kurtosis collapses),
L2 energy is exactly preserved, and round-to-nearest group quant becomes
near-lossless — no learned smoothing scales required (cf. SmoothQuant).

Online cost is **O(n log n)** via the Fast Walsh–Hadamard Transform (FWHT),
fusible into RMSNorm / RoPE.

## Theoretical Foundations

Orthogonal invariance:

```
||Qx||₂ = ||x||₂
Q⁻¹ = Qᵀ = diag(s) · H / √n
```

RTN group quant bit rate (symmetric, fp16 scales):

```
bit_rate = n_bits + 16 / group_size
comp_fp16 ≈ 16 / bit_rate          # ≈ 3.8× at INT4 / gs=32
```

Hadamard butterflies (length n = 2^k):

```
for h = 1..n/2:
  (a, b) ← (a+b, a-b)
scale by n^{-1/2}
```

## CLI Usage

```bash
# Theory ledger + live outlier-smear demo
quarot-hadamard-quant-engine status --dim 256 --bits 4

# Rotation energy + kurtosis delta
quarot-hadamard-quant-engine rotate --dim 256

# Baseline RTN metrics (no rotation)
quarot-hadamard-quant-engine quantize --bits 4 --group-size 32

# Full rotate → INT quant → unrotate pipeline
quarot-hadamard-quant-engine pipeline --dim 256 --bits 4

# Dim/bit sweep + fake-quant GEMM
quarot-hadamard-quant-engine benchmark --dims 64 128 256 512 --bits-list 4 8

# Invariant battery (FWHT involution, energy, GEMM cosine)
quarot-hadamard-quant-engine verify --trials 5

# Multi-sample pre/post outlier table
quarot-hadamard-quant-engine outlier-map --samples 8
```

## Architecture

- **fwht**: normalized Fast Walsh–Hadamard (involutory orthonormal).
- **HadamardRotator**: Q = H·diag(s), vector/matrix row & column rotates.
- **RTN Quantizer**: symmetric/asymmetric group INT2–INT8 with bit-rate ledger.
- **QuaRotEngine**: synthetic heavy-tailed activations, end-to-end pipeline,
  fake-quant GEMM (W' = WQ, x' = Qᵀx), outlier maps, theory tables.
- **CLI**: `status` / `rotate` / `quantize` / `pipeline` / `benchmark` /
  `verify` / `outlier-map`.

## Paper

Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*,
NeurIPS 2024 (arXiv:2404.00456).
