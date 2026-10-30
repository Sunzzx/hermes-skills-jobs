---
name: bitnet-b158
description: "1.58-bit ternary quantization kernel (weights in {-1, 0, +1}) with activation absmax scaling, memory-efficient packed ternary GEMM, and perplexity evaluation."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Frontier-AI, BitNet, Quantization, TernaryWeights, InferenceEngine]
---

# BitNet b1.58 Ternary Quantization & Inference Engine

## Theoretical Foundations
BitNet b1.58 replaces standard FP16/BF16 matrix multiplications with addition-only ternary dot products where weights $W \in \{-1, 0, +1\}$.

Weight quantization uses absolute mean scaling:
$$\gamma = rac{1}{nm} \sum_{i,j} |W_{i,j}|, \quad 	ilde{W} = 	ext{RoundClip}\left(rac{W}{\gamma}, -1, 1ight)$$

## CLI Usage
```bash
bitnet bench 2048   # Benchmark ternary GEMM against FP32 baseline
```
