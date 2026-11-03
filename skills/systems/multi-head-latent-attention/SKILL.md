---
name: multi-head-latent-attention
description: "Multi-Head Latent Attention (MLA) & KV-Cache Compression Engine: DeepSeek-V2/V3 low-rank joint compression, decoupled RoPE, and matrix absorption."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags:
      - multi-head-latent-attention
      - mla
      - deepseek-v2
      - deepseek-v3
      - kv-cache-compression
      - decoupled-rope
      - matrix-absorption
      - transformer-inference
---

# Multi-Head Latent Attention (MLA) & KV-Cache Compression Engine

A mathematically rigorous, production-grade implementation of **Multi-Head Latent Attention (MLA)**, introduced by DeepSeek-AI (DeepSeek-V2, DeepSeek-V3, DeepSeek-R1). MLA fundamentally resolves the key-value (KV) memory bottleneck in long-context transformer inference through low-rank joint compression, decoupled Rotary Position Embeddings (RoPE), and dynamic matrix absorption during autoregressive decoding.

---

## Theoretical Foundations

### 1. The Inference Memory Wall in Frontier LLMs
Standard Multi-Head Attention (MHA) caches full Key and Value tensors for every historical token. For a model with $n_h$ attention heads, head dimension $d_h$, sequence length $S$, and batch size $B$:
$$\text{Cache}_{\text{MHA}} = 2 \cdot B \cdot S \cdot n_h \cdot d_h \cdot \text{bytes}\quad \left[\mathcal{O}(S)\right]$$

For DeepSeek-V2 scale ($n_h = 128$, $d_h = 128$):
$$\text{Cache per Token} = 2 \times 128 \times 128 \times 2\text{ bytes} = 65,536\text{ bytes (64 KB/token)}$$
At context length $S = 128,000$, a single sequence demands **8.39 GB of high-bandwidth memory (HBM)** solely for KV cache, severely constraining concurrent generation throughput and batch capacity.

While Grouped-Query Attention (GQA) reduces cache size by sharing key-value heads across groups, it inherently degrades multi-head representational capacity.

---

## MLA Architecture & Formulation

```
                Input h_t in R^d
             ┌─────────┴─────────┐
             │                   │
       [Query Down-Proj]   [KV Down-Proj] W_DKV
         W_DQ in R^{d_c' x d}    │ in R^{d_c x d}
             │                   ▼
        RMSNorm(c_t^Q)     RMSNorm(c_t^{KV}) in R^{d_c}
        ┌────┴────┐         (Stored in KV Cache)
        │         │              │
   [W_UQ]      [W_QR]            ├───► Decoupled RoPE: W_KR @ h_t -> k_t^R in R^{d_R}
   Content     Rotary            │
   Query       Query             ├─── Prefill: Uncompressed (W_UK, W_UV)
   q_t^C       q_t^R             └─── Decode:  Matrix Absorption into Query & Value
```

### 2. Low-Rank Key-Value Joint Compression
MLA projects hidden state $\mathbf{h}_t \in \mathbb{R}^d$ into a low-rank latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ where $d_c \ll n_h \cdot d_h$:
$$\mathbf{c}_t^{KV} = \text{RMSNorm}(W_{DKV} \mathbf{h}_t) \in \mathbb{R}^{d_c}$$

In standard training and prompt prefill, content keys and values are up-projected:
$$\mathbf{k}_{t, i}^C = W_{UK, i} \mathbf{c}_t^{KV} \in \mathbb{R}^{d_{hc}},\quad \mathbf{v}_{t, i}^C = W_{UV, i} \mathbf{c}_t^{KV} \in \mathbb{R}^{d_v}$$
where $i \in \{1, \dots, n_h\}$ indexes attention heads.

### 3. Decoupled Rotary Position Embedding (RoPE)
Because low-rank linear compression breaks rotary positional invariance ($R_m W \neq W R_m$), positional information cannot be baked directly into $\mathbf{c}_t^{KV}$. MLA solves this by isolating positional rotation into a dedicated shared rotary key:
$$\mathbf{k}_t^R = \text{RoPE}(W_{KR} \mathbf{h}_t) \in \mathbb{R}^{d_R}$$

The total KV cache per token consists solely of:
$$\mathbf{cache}_t = \left[ \mathbf{c}_t^{KV} \; ; \; \mathbf{k}_t^R \right] \in \mathbb{R}^{d_c + d_R}$$
For DeepSeek-V2 ($d_c = 512, d_R = 64$):
$$\text{Cache per Token (MLA)} = (512 + 64) \times 2\text{ bytes} = 1,152\text{ bytes (1.12 KB)}$$
$$\text{Memory Reduction} = \frac{65,536}{1,152} = 56.9\times \text{ smaller (98.24% reduction!)}$$

### 4. Matrix Absorption (Weight Folding during Decoding)
During autoregressive generation, MLA completely avoids decompressing historical keys and values into $\mathbb{R}^{n_h \cdot d_h}$ tensors via matrix absorption.

#### A. Key Content Absorption into Query
Using the associative property of matrix multiplication:
$$\mathbf{q}_{t, i}^C \cdot \mathbf{k}_{s, i}^C = (\mathbf{q}_{t, i}^C)^T (W_{UK, i} \mathbf{c}_s^{KV}) = (W_{UK, i}^T \mathbf{q}_{t, i}^C)^T \mathbf{c}_s^{KV} = (\tilde{\mathbf{q}}_{t, i}^C)^T \mathbf{c}_s^{KV}$$

The query token $\mathbf{q}_{t, i}^C$ is pre-multiplied by $W_{UK, i}^T$ once at the start of the step. The attention score is then computed directly against the compact compressed latent history $\mathbf{c}_s^{KV}$!

#### B. Value Matrix Absorption
Similarly, the Value projection $W_{UV, i}$ is factored out of the sequence summation:
$$\mathbf{u}_{t, i} = \sum_{s=1}^t \alpha_{t, s, i} (W_{UV, i} \mathbf{c}_s^{KV}) = W_{UV, i} \left( \sum_{s=1}^t \alpha_{t, s, i} \mathbf{c}_s^{KV} \right)$$

The attention weights $\alpha$ are accumulated over $d_c$-dimensional latent vectors first; $W_{UV, i}$ is applied only once per head after summation.

---

## Architecture Comparison Matrix

| Architecture | KV Cache Elements / Token | DeepSeek-V2 Cache / Tok | Multi-Head Capacity | Matrix Absorption |
| :--- | :--- | :--- | :--- | :--- |
| **MHA** | $2 \cdot n_h \cdot d_h$ | 32,768 elems (65.5 KB) | Full ($n_h$ heads) | No |
| **MQA** | $2 \cdot 1 \cdot d_h$ | 256 elems (512 B) | Degraded (1 head) | No |
| **GQA** | $2 \cdot n_{kv} \cdot d_h$ | 2,048 elems (4.1 KB) | Sub-sampled ($g$ groups) | No |
| **MLA** | $d_c + d_R$ | **576 elems (1.15 KB)** | **Full ($n_h$ heads)** | **Yes (Zero runtime decompression)** |

---

## CLI Reference (`mla-engine`)

### 1. Mathematical Equivalence Verification
Verifies machine-epsilon equivalence ($< 10^{-6}$) between uncompressed full forward and matrix-absorbed generation:
```bash
mla-engine verify --dim 512 --heads 8 --q-rank 128 --kv-rank 128 --seq-len 32
```

### 2. Memory Scaling & Latency Benchmark
Benchmarks memory savings across context windows up to 131,072 tokens:
```bash
mla-engine benchmark --dim 512 --heads 8 --kv-rank 128 --steps 50
```

### 3. Step-by-Step Autoregressive Simulation
Visualizes cache allocation and matrix absorption dynamics token by token:
```bash
mla-engine simulate --steps 10
```

### 4. Architectural Specification & Mathematical Inspection
```bash
mla-engine inspect
```
