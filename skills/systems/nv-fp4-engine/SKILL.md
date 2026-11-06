---
name: nv-fp4-engine
description: "NVIDIA Blackwell NVFP4 (FP4-E2M1) Micro-Tensor Float Quantization & Scaled GEMM Engine with two-level microscaling and stochastic rounding."
version: "1.0.0"
category: systems
tags:
  - nvfp4
  - fp4
  - blackwell
  - microscaling
  - quantization
  - gemm
  - frontier-ai
  - tensor-cores
cli: nv-fp4-engine
---

# NVIDIA Blackwell NVFP4 Micro-Tensor Float Quantization & Scaled GEMM Engine

`nv-fp4-engine` is a production-grade, mathematically verified reference implementation of **NVIDIA Blackwell NVFP4 (FP4-E2M1)** micro-tensor float quantization and scaled GEMM architecture (as specified in OCP Microscaling Formats and NVIDIA B200 Tensor Core architecture).

---

## Theoretical Foundations

### 1. The Inference Compute & Memory Wall in Modern LLMs
While FP8 mixed-precision training and serving (OCP E4M3 / E5M2) achieved significant adoption in Hopper (H100) and DeepSeek-V3, frontier reasoning models (o1, DeepSeek-R1) and massive MoE clusters require higher compute density and memory bandwidth. Standard 4-bit integer quantization (INT4) suffers from severe dynamic range limitations and asymmetric zero-point calculation overhead.

### 2. NVFP4 (FP4-E2M1) Micro-Tensor Specification
NVFP4 adopts an optimized floating-point representation with:
- **1 sign bit**, **2 exponent bits** (bias = 1), and **1 mantissa bit**.
- 15 unique symmetric signed values + zero:
  $$\mathcal{S} = \{-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$$
- Smallest subnormal value: $0.5$.
- Maximum finite magnitude: $6.0$.

### 3. Two-Level Hierarchical Micro-Scaling
Rather than applying a single coarse scale factor per tensor (which suffers from catastrophic outlier underflow), NVFP4 partitions weights and activations into micro-vectors of size $B = 16$ or $32$ elements:

$$\alpha_{\text{tile}} = \frac{\max_{i \in \text{tile}} |x_i|}{6.0}$$

Each micro-vector scale $\alpha_{\text{tile}}$ is quantized into an 8-bit scale factor (FP8 E4M3 / E8M0), confining activation spikes to individual tiles without degrading signal-to-quantization noise ratio (SQNR > 20 dB).

### 4. Tiled Scaled GEMM with Scale Folding
During matrix multiplication $Y = X \cdot W$, 4-bit dot products are accumulated in packed tensor core registers, while the two tile scale factors are folded at block boundaries in FP32:

$$Y_{i,j} = \sum_{k=1}^{K/B} \left( \sum_{b=1}^{B} \hat{X}_{i, (k-1)B + b} \cdot \hat{W}_{(k-1)B + b, j} \right) \cdot \alpha_{X, i, k} \cdot \alpha_{W, k, j}$$

This yields **75% memory compression vs FP16** and **up to 2.5x-3.0x higher arithmetic throughput**.

---

## CLI Reference

```bash
# Inspect NVFP4 format specifications and codebook levels
nv-fp4-engine status

# Run mathematical and architectural invariant tests (5 proofs)
nv-fp4-engine verify

# Quantize tensor with micro-scaling and optional activation outlier channels
nv-fp4-engine quantize --rows 64 --cols 256 --block-size 16 --outliers

# Execute tiled scaled GEMM comparing NVFP4 to FP32 reference baseline
nv-fp4-engine gemm --m 128 --k 256 --n 128 --block-size 16

# Benchmark multi-shape throughput, memory reduction, and Frobenius error
nv-fp4-engine benchmark
```
