---
name: sequence-parallel
description: "DeepSpeed Ulysses & Sequence Parallelism Engine: All-to-All collective context scaling, zero-communication MLP/RMSNorm paths, and hybrid USP 2D parallelism."
category: systems
type: skill+tool
version: 1.0.0
author: Hermes Evolution & Autonomous Syndicate
cli: sequence-parallel
tags:
  - deepspeed-ulysses
  - sequence-parallelism
  - all-to-all
  - distributed-training
  - context-scaling
  - megatron-sp
  - hybrid-usp
---

# DeepSpeed Ulysses & Sequence Parallelism Engine

A mathematically rigorous, production-grade distributed context scaling framework implementing **DeepSpeed Ulysses** (Jacobs et al., 2023) and **Megatron-LM Sequence Parallelism** (Korthikanti et al., 2023).

Partitions ultra-long context sequences across distributed ranks without memory bottlenecks by dynamically transposing between sequence-sharded and head-sharded representations via dual All-to-All collective communication.

## Theoretical Foundations & Core Primitives

### 1. Dual All-to-All Collective Transposition
Given an input sequence $X \in \mathbb{R}^{B 	imes S 	imes D}$ partitioned across $P$ ranks along the sequence dimension, rank $p$ holds:
$$X_p \in \mathbb{R}^{B 	imes (S/P) 	imes D}$$

#### Step 1: Sequence-to-Head Transposition (All-to-All 1)
Local linear projections produce full heads $H$ over local sequence tokens:
$$Q_p, K_p, V_p = X_p W_{Q,K,V} \in \mathbb{R}^{B 	imes (S/P) 	imes H 	imes d_h}$$

Slicing heads into $P$ partitions of size $H/P$ and executing an All-to-All collective yields the complete sequence $S$ across assigned local heads:
$$	ext{All-to-All}_1: (B, S/P, H, D) \longrightarrow (B, S, H/P, D)$$

#### Step 2: Full-Sequence Attention Computation
Each rank computes standard scaled dot-product attention over the full sequence $S$ for its local heads:
$$	ext{Attn}_p = 	ext{Softmax}\left(rac{Q_p K_p^T}{\sqrt{d_h}}ight) V_p \in \mathbb{R}^{B 	imes S 	imes (H/P) 	imes d_h}$$

#### Step 3: Head-to-Sequence Transposition (All-to-All 2)
A reverse All-to-All collective transposes full sequence length back to sequence-sharded representations:
$$	ext{All-to-All}_2: (B, S, H/P, D) \longrightarrow (B, S/P, H, D)$$

### 2. Zero-Communication Feed-Forward & Normalization
Because RMSNorm and MLP activations (SwiGLU, GeLU) operate point-wise along token vectors independently, the sequence-sharded state $(B, S/P, D)$ executes with **zero inter-GPU communication**:
- RMSNorm: 0 bytes communicated
- Residual Connections: 0 bytes communicated
- MLP Projection & Gating: 0 bytes communicated

### 3. Comparison with RingAttention and Megatron-SP
| Metric / Feature | Megatron-SP | DeepSpeed Ulysses | RingAttention | Hybrid USP (2D) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Collectives** | AllGather + RedScat | 2x All-to-All | P2P Ring (P hops) | All-to-All + Ring |
| **Communication Steps** | 2 steps | 2 steps | P steps | 2 intra + P inter |
| **Message Size** | $B \cdot S \cdot D / P$ | $B \cdot S \cdot D / P^2$ | $B \cdot (S/P) \cdot D / P$ | Hierarchical |
| **Ideal Hardware** | Intra-node NVLink | Intra-node NVLink | Inter-node / WAN | NVLink + InfiniBand |
| **Attention Kernel** | Standard / Flash | Standard / Flash | Online Softmax | Standard / Blockwise |

## CLI Usage & Verification

```bash
# Run full mathematical verification against monolithic attention
sequence-parallel verify

# Benchmark activation memory scaling and communication volume up to 1M context
sequence-parallel bench

# Compare parallelism paradigms (Megatron-SP vs Ulysses vs Ring vs USP)
sequence-parallel compare

# Inspect mathematical derivations and complexity bounds
sequence-parallel inspect

# Simulate step-by-step All-to-All tensor schedule across ranks
sequence-parallel simulate --world-size 4 --seq-len 32 --heads 8
```
