---
name: tensor-parallel
description: "Megatron-LM 1D Tensor Parallelism engine: column/row parallel linear, multi-head attention, parallel MLP, and zero-gather vocab cross entropy."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  category: systems
  type: cli
  command: tensor-parallel
  hermes:
    tags: [TensorParallelism, MegatronLM, DistributedTraining, DistributedInference, SPMD]
---

# ⚡ Megatron-LM 1D Tensor Parallelism Engine

## Overview
A mathematically rigorous, production-grade implementation of Megatron-LM 1D Tensor Parallelism (Shoeybi et al., 2019; Narayanan et al., 2021). The engine partitions multi-head self-attention, feed-forward networks, and vocabulary cross-entropy loss across distributed ranks while maintaining bitwise / machine-epsilon equivalence to monolithic Transformer layers.

## Key Architectural Primitives

### 1. ColumnParallelLinear
Weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ is split along the column axis:
$$W = \begin{bmatrix} W_0 & W_1 & \cdots & W_{k-1} \end{bmatrix}, \quad W_i \in \mathbb{R}^{d_{\text{in}} \times (d_{\text{out}}/k)}$$

- **Forward**: $Y_i = X W_i + b_i$
  - `gather_output=False`: Output remains partitioned $(B, S, d_{\text{out}}/k)$ for feeding directly into downstream RowParallel layers without inter-device communication.
  - `gather_output=True`: An All-Gather operation concatenates outputs along the last dimension.
- **Backward**: 
  - Weight gradient: $\nabla W_i = X_i^T \nabla Y_i$
  - Input gradient: $\nabla X = \sum_{i=0}^{k-1} \nabla Y_i W_i^T$ via **All-Reduce (sum)**.

### 2. RowParallelLinear
Weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ is split along the row axis:
$$W = \begin{bmatrix} W_0 \\ W_1 \\ \vdots \\ W_{k-1} \end{bmatrix}, \quad W_i \in \mathbb{R}^{(d_{\text{in}}/k) \times d_{\text{out}}}$$

- **Forward**: Input $X$ is split along hidden dimension $[X_0, X_1, \dots, X_{k-1}]$. Each rank computes partial product $Z_i = X_i W_i$. An **All-Reduce (sum)** computes:
  $$Y = \left(\sum_{i=0}^{k-1} X_i W_i\right) + b$$
- **Backward**:
  - Weight gradient: $\nabla W_i = X_i^T \nabla Y$
  - Input gradient: $\nabla X_i = \nabla Y W_i^T$ (natively partitioned along hidden dimension).

### 3. Parallel MLP (Feed-Forward Network)
Connects `ColumnParallelLinear` $(h \to 4h/k)$ directly to `RowParallelLinear` $(4h/k \to h)$:
- **GeLU Activation**: Evaluated elementwise purely on local rank buffers with **zero communication**!
- **Total Communication**: Exactly **ONE All-Reduce** in forward pass, and **ONE All-Reduce** in backward pass.

### 4. Parallel Multi-Head Self-Attention
- **QKV Projection**: ColumnParallelLinear shards attention heads evenly across ranks ($h/k$ heads per rank).
- **Scaled Dot-Product Attention**: Fully computed locally on each rank's subset of heads with zero communication.
- **Out Projection**: RowParallelLinear combines local head projections with **ONE All-Reduce**.

### 5. Vocabulary-Parallel Cross-Entropy
Vocabulary $V$ is sharded across ranks ($V/k$ tokens per rank):
1. **Local Maxima**: $m_i = \max_j Z_{i, j}$
2. **Global Maxima**: $m = \text{AllReduce}(\max)(m_i)$
3. **Local Exponentials**: $s_i = \sum_j \exp(Z_{i, j} - m)$
4. **Global LogSumExp**: $\text{LSE} = m + \ln\left(\text{AllReduce}(\text{sum})(s_i)\right)$
5. **Loss & Gradients**: Evaluated locally in-place without ever gathering full vocabulary logits onto any single GPU!

---

## CLI Reference

The engine is installed as `tensor-parallel` in the system path:

```bash
# Verify mathematical equivalence vs monolithic ground truth across TP degrees
tensor-parallel verify --tp 2
tensor-parallel verify --tp 4

# Benchmark memory reduction and interconnect communications
tensor-parallel bench --hidden-dim 2048 --heads 16 --seq-len 1024 --batch-size 4

# View ASCII tensor sharding and Ring communication diagram
tensor-parallel demo --tp 2

# Inspect engine status, supported modules, and numerical tolerances
tensor-parallel status
```
