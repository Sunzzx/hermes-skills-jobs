---
name: gated-deltanet
description: "Gated DeltaNet parallel delta-rule linear attention: Householder fast-weight updates, forget gates, chunkwise scan, associative memory probes."
version: "1.0.0"
category: research
tags:
  - gated-deltanet
  - delta-rule
  - linear-attention
  - fast-weight-programmers
  - mamba2
  - frontier-ai
cli: deltanet
---

# Gated DeltaNet & Parallel Delta Rule Engine

`deltanet` is a production-grade, mathematically verified implementation of **Gated Delta Networks** — the linear-time sequence layer that upgrades cumulative outer-product state (RetNet / linear attention / Mamba-2 SSD) with an **error-correcting delta rule** and a **multiplicative forget gate**.

---

## Theoretical Core

### 1. Classical Delta Rule (Fast Weight Programmer)

Maintain a fast-weight matrix $W_t \in \mathbb{R}^{d_v \times d_k}$. On token $t$ with key $k_t$, value $v_t$, and write strength $\beta_t \in [0,1]$:

$$
u_t = W_{t-1} k_t, \qquad
W_t = W_{t-1} + \beta_t (v_t - u_t) k_t^\top
$$

Equivalently, a rank-1 Householder-style right action:

$$
W_t = W_{t-1}\bigl(I - \beta_t k_t k_t^\top\bigr) + \beta_t v_t k_t^\top
$$

When $\|k_t\|_2 = 1$ and $\beta_t = 1$, a single write yields perfect post-readout $W_t k_t = v_t$.

### 2. Gated Decay

Before the delta write, apply a multiplicative forget gate $\alpha_t \in (0,1]$:

$$
W \leftarrow \alpha_t W, \qquad
W \leftarrow W + \beta_t (v_t - W k_t) k_t^\top
$$

Combined right-acting form:

$$
W_t = W_{t-1}\cdot\underbrace{\alpha_t\bigl(I - \beta_t k_t k_t^\top\bigr)}_{A_t} + \beta_t v_t k_t^\top
$$

### 3. Query-Key Separation

Writes use $k_t$; readouts use query $q_t$:

$$
y_t = W_t q_t
$$

Keys are $\ell_2$-normalized per head for stable Householder geometry.

### 4. Parallel / Chunkwise Scan

The recurrence expands into a product of $A_t$ matrices plus rank-1 residual terms. Chunkwise fused scans carry $W$ across chunk boundaries and match the sequential engine to machine precision — the algebraic backbone for future WY / UT transform GPU kernels.

### 5. Contrast to Causal Linear Attention

Pure linear attention accumulates $S_t = \alpha_t S_{t-1} + v_t k_t^\top$ without subtracting the current recall $W k_t$. DeltaNet's residual $(v - Wk)$ is an online least-squares correction, giving stronger associative memory and reducing key interference.

---

## CLI

Global binary: `deltanet` (`~/.local/bin/deltanet`).

```bash
# Mathematical invariant suite (seq↔par, causality, memory, gates)
deltanet verify

# Kernel latency: sequential / parallel / chunk / linear-attn baseline
deltanet benchmark --seq-len 256 --dim 64 --repeats 5

# Parameter, state footprint, FLOP model
deltanet inspect --dim 256 --heads 8 --seq-len 2048

# Short forward demo
deltanet simulate --seq-len 32 --dim 16

# Associative memory write/recall probe
deltanet probe --dim 32 --pairs 16 --beta 1.0 --alpha 1.0
```

---

## Mathematical Invariants Verified

- **Seq ↔ Parallel equivalence**: $\|Y_{\mathrm{seq}} - Y_{\mathrm{par}}\| / \|Y_{\mathrm{seq}}\| < 10^{-10}$
- **Chunk-fused equivalence**: identical outputs for arbitrary chunk sizes
- **Unit-key perfect write**: $\beta=1,\alpha=1,\|k\|=1 \Rightarrow y_{\mathrm{post}}=v$
- **Zero-β pure decay**: $W_t = \alpha_t W_{t-1}$
- **Causal past invariance**: future value mutations leave $y_{<t}$ unchanged
- **Associative memory**: mean cosine $> 0.85$ on 8 random unit-key pairs
- **Gate ranges**: $\beta\in[0,1]$, $\alpha\in(0,1]$, keys unit-norm
- **Delta ≠ linear attention**: relative separation $> 10^{-3}$ on shared inputs

---

## When to Use

- Prototyping linear-time alternatives to softmax attention
- Validating delta-rule / fast-weight update algebra before CUDA work
- Comparing Mamba-2 SSD, RetNet retention, and delta-rule memory
- Teaching Householder fast-weight programmers and gated scans

Don't use for: production GPU training kernels (this is a NumPy reference oracle).
