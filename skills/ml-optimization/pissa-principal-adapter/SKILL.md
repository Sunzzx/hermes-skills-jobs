---
name: pissa-principal-adapter
description: "Principal Singular values and Singular vectors Adaptation (NeurIPS 2024 / arXiv:2404.05971) for parameter-efficient fine-tuning (PEFT) with zero inference latency."
category: ml-optimization
tags:
  - peft
  - lora
  - svd
  - fine-tuning
  - llm
cli: pissa-principal-adapter
---

# PiSSA: Principal Singular values and Singular vectors Adaptation

PiSSA (NeurIPS 2024) is a next-generation Parameter-Efficient Fine-Tuning (PEFT) framework that initializes low-rank adapters directly from the principal singular components of the pre-trained weight matrices, freezing the residual non-principal components.

## Theoretical Formulation

Given a pre-trained weight matrix $W \in \mathbb{R}^{m \times n}$, PiSSA computes the thin Singular Value Decomposition (SVD):

$$W = U \Sigma V^T = \sum_{i=1}^{\min(m, n)} \sigma_i u_i v_i^T$$

PiSSA slices the top-$r$ principal components into trainable adapters:
- Trainable adapter $A = U_r \Sigma_r^{1/2} \in \mathbb{R}^{m \times r}$
- Trainable adapter $B = \Sigma_r^{1/2} V_r^T \in \mathbb{R}^{r \times n}$
- Frozen residual $W_{\text{res}} = W - A B \in \mathbb{R}^{m \times n}$

## Key Advantages over Standard LoRA

1. **Zero-Discard Principal Initialization**: Unlike LoRA (which initializes $B=0$ and $A \sim \mathcal{N}(0, \sigma^2)$, completely ignoring the base weight distribution), PiSSA immediately captures the primary data manifold.
2. **Faster Convergence**: Learns up to 10x faster to reach identical loss compared to standard LoRA in fine-tuning runs.
3. **Zero Inference Overhead**: At inference time, $W_{\text{final}} = W_{\text{res}} + A B$ can be pre-merged with zero additional inference latency.

## Usage CLI

```bash
# Run head-to-head convergence benchmark against standard LoRA
pissa-principal-adapter bench --steps 35

# SVD decomposition and energy fraction inspection
pissa-principal-adapter decompose --rows 32 --cols 32 --rank 8 --json

# Compute parameter reduction statistics on 7B/8B architectures
pissa-principal-adapter stats --rank 16
```
