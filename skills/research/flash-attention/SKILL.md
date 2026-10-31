---
name: flash-attention
description: "FlashAttention-2 and online softmax IO-aware SRAM tiling engine with O(N) memory complexity, causal masking, exact backward pass without N x N storage, and FlashDecoding split-KV."
version: 1.0.0
category: research
tags:
  - Frontier-AI
  - FlashAttention
  - OnlineSoftmax
  - IO-Aware
  - Transformers
  - DeepLearning
---

# ⚡ FlashAttention-2 & Online Softmax IO-Aware Tiling Engine

A mathematically rigorous, production-grade implementation of **FlashAttention-2**, **Online Softmax**, and **FlashDecoding Split-KV** in NumPy.

---

## 📐 Theoretical Foundations & Memory Hierarchy

### 1. The GPU Memory Hierarchy Problem
In modern accelerator architectures (NVIDIA Hopper/Blackwell, Apple Silicon unified memory), raw tensor compute capability (FLOP/s) has grown by orders of magnitude faster than High-Bandwidth Memory (HBM) bandwidth (TB/s).

Standard Multi-Head Attention materializes intermediate matrices directly into DRAM:
$$egin{aligned}
S &= rac{Q K^T}{\sqrt{d}} \in \mathbb{R}^{B \times H \times N \times N} \quad &(\text{DRAM Write}) \\nP &= \text{softmax}(S) \in \mathbb{R}^{B \times H \times N \times N} \quad &(\text{DRAM Read } S, \text{ DRAM Write } P) \\nO &= P V \in \mathbb{R}^{B \times H \times N \times d} \quad &(\text{DRAM Read } P, V, \text{ DRAM Write } O)
\end{aligned}$$

For a sequence length of $N = 16,384$ and $d = 64$:
- Standard attention requires storing an $N \times N$ float32 matrix $\approx 1.07\text{ GB}$ per head per batch.
- Total DRAM memory traffic scales as $\Theta(N d + N^2)$ words.

### 2. IO-Aware SRAM Tiling Theorem (Dao et al.)
By tiling inputs into fast on-chip SRAM blocks of size $B_r \times d$ (for Query) and $B_c \times d$ (for Key/Value), FlashAttention fuses matrix multiplication and softmax without ever writing the $N \times N$ matrix to DRAM.

- **Peak Working Memory**: Strict $\Theta(N)$ instead of $\Theta(N^2)$.
- **DRAM IO Bandwidth Complexity**:
$$\text{IO}_{\text{Flash}} = \Theta\left(\frac{N^2 d^2}{M}\right) \text{ bytes}$$
where $M$ is the on-chip SRAM capacity, reducing DRAM memory traffic by a factor of $\Theta(M / d)$ (typically $5\times$ to $10\times$ reduction).

---

## 🧠 Algorithmic Innovations

### 1. Online Softmax with Dynamic Rescaling
Softmax over row vector $x$ is defined as:
$$m = \max_j x_j, \quad l = \sum_j e^{x_j - m}, \quad \text{softmax}(x)_j = \frac{e^{x_j - m}}{l}$$

When processing inputs block-by-block, FlashAttention dynamically rescales the running accumulator $O_i$ as new block maximums are encountered:
$$\begin{aligned}
m_{\text{new}} &= \max(m_i, \; m_{ij}) \\n\alpha &= \exp(m_i - m_{\text{new}}) \\n\tilde{P} &= \exp(S_{ij} - m_{\text{new}}) \\nl_{\text{new}} &= \alpha \cdot l_i + \sum_{\text{cols}} \tilde{P} \\nO_i &= \alpha \cdot O_i + \tilde{P} V_j \\nm_i &= m_{\text{new}}, \quad l_i = l_{\text{new}}
\end{aligned}$$
At the end of all Key-Value tiles, the block output is normalized in one step:
$$O_i^{\text{final}} = \frac{O_i}{l_i}, \quad L_i = m_i + \log(l_i)$$

### 2. Block-Level Causal Pruning
For autoregressive decoding ($Q$ attends only to past $K$), entire Key-Value tiles lying strictly in the future are skipped with zero FLOPs and zero DRAM transfers:
$$\text{If } k_{\text{start}} > q_{\text{end}} - 1 + (N_{\text{kv}} - N_q) \implies \text{Skip block entirely}$$
Diagonal tiles apply intra-block causal triangular masking.

### 3. Exact Backward Pass without Storing Attention Weights
Instead of caching $P \in \mathbb{R}^{B \times H \times N \times N}$ in DRAM during forward pass, FlashAttention-2 stores only the compact logsumexp vector $L \in \mathbb{R}^{B \times H \times N \times 1}$.

During backward pass, in fast SRAM:
1. Recomputes $S_{ij} = (Q_i K_j^T) / \sqrt{d}$.
2. Recomputes $P_{ij} = \exp(S_{ij} - L_i)$.
3. Precomputes row-wise dot product $D_i = \sum_d (dO_i \odot O_i)$.
4. Calculates gradient $dS_{ij} = P_{ij}(dO_i V_j^T - D_i)$.
5. Accumulates $dV_j += P_{ij}^T dO_i$, $dK_j += dS_{ij}^T Q_i / \sqrt{d}$, and $dQ_i += dS_{ij} K_j / \sqrt{d}$.

### 4. FlashDecoding for Long Contexts ($N_q = 1$)
Standard FlashAttention parallelizes across batch and head dimensions. During autoregressive generation ($N_q = 1$), batch and heads are small, leaving GPU streaming multiprocessors underutilized.

**FlashDecoding** partitions the sequence dimension $N_{\text{kv}}$ into $S$ parallel chunks:
1. Each chunk computes local partial attention $\tilde{O}_s, m_s, l_s$ in parallel.
2. A parallel reduction tree aggregates partial outputs across splits using logsumexp fusion:
$$m^* = \max_s m_s, \quad l^* = \sum_s e^{m_s - m^*} l_s, \quad O^* = \frac{1}{l^*} \sum_s e^{m_s - m^*} \tilde{O}_s$$
Achieving up to $8\times$ lower generation latency on long context windows.

---

## 🛠️ CLI Quickstart

The engine is accessible globally as `flash-attn`:

### 1. Mathematical & Numerical Parity Verification
```bash
flash-attn verify
```
Validates across 5 rigorous checkpoints:
- Non-causal forward pass output error $< 10^{-10}$
- Causal forward pass output error with block pruning $< 10^{-10}$
- Exact backward pass gradient parity ($dQ, dK, dV$) without saving attention matrix $< 10^{-10}$
- FlashDecoding split-KV long-context parity ($N=2048$, 8 splits) $< 10^{-10}$
- Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) parity $< 10^{-10}$

### 2. Interactive Architecture & IO Walkthrough
```bash
flash-attn demo
```
Displays the SRAM tiling grid, active computed vs skipped causal blocks, and analytical DRAM bandwidth comparisons across sequence lengths from 512 to 16,384.

### 3. Execution Latency & Memory Benchmark
```bash
flash-attn bench
```
Compares Standard Attention vs FlashAttention-2 across sequence lengths $N \in [256, 2048]$, measuring latency, peak memory, and numerical error.

### 4. FlashDecoding Split-KV Long-Context Evaluation
```bash
flash-attn decode
```
Benchmarks autoregressive single-token decoding across context lengths up to 8,192 tokens with multi-split parallel reductions.

---

## 💻 Python API Reference

```python
import numpy as np
from bin.flash_attn import FlashAttentionEngine

# 1. Forward Pass
B, H, N, d = 2, 8, 1024, 64
Q = np.random.randn(B, H, N, d)
K = np.random.randn(B, H, N, d)
V = np.random.randn(B, H, N, d)

# O(N) memory execution
O, L = FlashAttentionEngine.forward(Q, K, V, causal=True, Br=64, Bc=64)

# 2. Exact Backward Pass without N x N matrix
dO = np.random.randn(B, H, N, d)
dQ, dK, dV = FlashAttentionEngine.backward(dO, Q, K, V, O, L, causal=True)

# 3. FlashDecoding (Autoregressive Token Generation)
Q_step = np.random.randn(B, H, 1, d)
O_step = FlashAttentionEngine.flash_decoding(Q_step, K, V, num_splits=8)
```

---

## 🧪 Test Suite

Run the comprehensive automated unit tests:
```bash
python3 -m unittest tests/test_flash_attention.py
```
