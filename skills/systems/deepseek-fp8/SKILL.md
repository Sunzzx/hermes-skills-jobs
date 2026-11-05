---
name: deepseek-fp8
description: "DeepSeek-V3 fine-grained FP8 mixed-precision training: E4M3/E5M2 formats, 1x128 activation tile micro-scaling, 128x128 weight block scaling, outlier isolation, and high-precision chunked GEMM."
version: "1.0.0"
category: systems
tags:
  - deepseek-v3
  - fp8
  - mixed-precision
  - quantization
  - microscaling
  - gemm
  - transformer-training
cli: fp8-engine
---

# DeepSeek-V3 FP8 Mixed-Precision Training & Quantized GEMM Engine

`fp8-engine` is a production-grade, mathematically verified implementation of the **Fine-Grained FP8 Mixed-Precision Training Framework** and **Quantized Matrix Multiplication (GEMM) Engine** introduced in **DeepSeek-V3**.

---

## Theoretical Architecture & DeepSeek-V3 Innovations

FP8 low-precision training delivers massive speedups on modern tensor cores (e.g., NVIDIA Hopper H100/H800/B200) by cutting memory bandwidth and storage by $4\times$ compared to FP32 ($2\times$ compared to BF16/FP16). However, naive per-tensor FP8 quantization encounters severe accuracy degradation due to activation outliers (channels whose values exceed typical activations by $10\times-100\times$).

DeepSeek-V3 solves this through **Fine-Grained Micro-Scaling** and **Format Specialization**:

### 1. Dual FP8 Formats (OCP / IEEE 754-2019)
- **FP8 E4M3 (1S, 4E, 3M, Bias=7, Max Finite=448.0)**:
  Higher numerical precision with 3 mantissa bits ($8$ intervals per bin). Dedicated to **activations and weights in the forward pass**.
- **FP8 E5M2 (1S, 5E, 2M, Bias=15, Max Finite=57344.0)**:
  Wider dynamic range with 5 exponent bits. Dedicated to **gradients ($B_X$, $B_W$) in the backward pass** to prevent gradient underflow/overflow.

### 2. Fine-Grained Tile & Block Micro-Scaling
Rather than computing a single scale factor for an entire matrix, DeepSeek-V3 partitions tensors into local micro-tiles:
- **Activations**: Quantized at **$1 \times 128$ tile granularity** (1 token across 128 channel dimensions):
  $$S_X[m, g] = \frac{\max_{c \in [0, 128)} |X_{m, g \cdot 128 + c}|}{\text{max\_finite}(\text{E4M3})} = \frac{\max |X_{m, g}|}{448.0}$$
  $$X^{(\text{fp8})}_{m, g \cdot 128 + c} = \text{quantize}_{\text{E4M3}}\left(\frac{X_{m, g \cdot 128 + c}}{S_X[m, g]}\right)$$
- **Weights**: Quantized at **$128 \times 128$ block granularity** (2D parameter sub-blocks):
  $$S_W[u, v] = \frac{\max |W_{u, v}|}{448.0}$$
  $$W^{(\text{fp8})}_{u, v} = \text{quantize}_{\text{E4M3}}\left(\frac{W_{u, v}}{S_W[u, v]}\right)$$
- **Outlier Isolation**: When an activation outlier emerges, its large scale factor $S_X$ is confined strictly to its own 128-element tile. The remaining tiles preserve full dynamic range, boosting Signal-to-Quantization-Noise Ratio (SQNR) by **$+10\text{ to }+15\text{ dB}$** over global scaling.

### 3. High-Precision Chunked GEMM Accumulation
Hardware tensor cores compute inner dot-products in FP32 accumulation registers. For output tile $Y_{m, n}$:
$$Y_{m, n} = \sum_{g=0}^{(K/128)-1} \left( S_X[m, g] \cdot S_W[g, n // 128] \right) \cdot \sum_{c=0}^{127} X^{(\text{fp8})}_{m, g \cdot 128 + c} \cdot W^{(\text{fp8})}_{g \cdot 128 + c, n}$$
Scale factors $S_X \cdot S_W$ are applied only at $128$-element chunk boundaries, preserving numerical fidelity while maximizing tensor core throughput.

### 4. Stochastic Rounding for Gradient Updates
To eliminate systematic gradient quantization bias during backward propagation, gradients are quantized using stochastic rounding:
$$\text{round}_{\text{stoch}}(x) = \begin{cases} \lfloor x \rfloor + 1 & \text{with probability } x - \lfloor x \rfloor \\ \lfloor x \rfloor & \text{otherwise} \end{cases}$$
This ensures $\mathbb{E}[\hat{G}] = G$, preserving convergence stability over millions of gradient steps.

---

## Scale Storage & Memory Overhead

| Representation | Bits/Param | Memory (4K x 4K Weight) | Compression Ratio | Micro-Scale Overhead |
| :--- | :---: | :---: | :---: | :---: |
| **FP32** | 32 | 64.0 MB | 1.00x | 0.00% |
| **BF16 / FP16** | 16 | 32.0 MB | 2.00x | 0.00% |
| **DeepSeek FP8 (E4M3)** | 8 | 16.0 MB | **4.00x** | **0.024%** (1 scale per 16,384 bytes) |

---

## CLI Commands & Usage

The CLI tool `fp8-engine` is globally installed into `~/.local/bin/fp8-engine`.

```bash
# 1. Inspect FP8 specifications, bit layouts, and memory footprints
fp8-engine status

# 2. Quantize activations and demonstrate outlier resistance (1x128 Tile vs Global)
fp8-engine quantize --batch-size 4 --seq-len 16 --dim 256 --outlier-mult 50.0

# 3. Execute fine-grained FP8 GEMM against FP32 reference with fidelity metrics
fp8-engine gemm --batch-size 4 --seq-len 16 --dim 256 --out_dim 256

# 4. Benchmark memory footprint, SQNR, and compression across model dimensions
fp8-engine benchmark

# 5. Run exhaustive mathematical and numerical invariant verification
fp8-engine verify
```

---

## Python SDK Programmatic API

```python
from fp8_engine import DeepSeekFP8Engine, E4M3, E5M2, compute_sqnr, cosine_similarity
import numpy as np

# Activations: (Batch*Seq, Hidden) where Hidden % 128 == 0
X = np.random.randn(64, 4096)
W = np.random.randn(4096, 4096) / np.sqrt(4096)

# Quantize activations (1x128 tile) and weights (128x128 block)
q_act = DeepSeekFP8Engine.quantize_activations_tile128(X, fmt=E4M3)
q_weight = DeepSeekFP8Engine.quantize_weights_block128x128(W, fmt=E4M3)

# Execute fine-grained quantized GEMM
Y_fp8 = DeepSeekFP8Engine.fine_grained_gemm(q_act, q_weight, fmt=E4M3)

# Reference FP32
Y_ref = np.matmul(X, W)
print("Cosine Similarity:", cosine_similarity(Y_ref, Y_fp8))
print("SQNR (dB):       ", compute_sqnr(Y_ref, Y_fp8))
```
