---
name: mamba-ssm
description: "Frontier AI Selective State Space Model (Mamba) with Blelloch associative parallel scan, continuous ZOH discretization, and O(1) inference."
version: 1.0.0
category: research
tags:
  - Frontier-AI
  - Mamba
  - State-Space-Models
  - Blelloch-Scan
  - Deep-Learning
  - HiPPO
---

# ⚡ Mamba-SSM: Selective State Space Model Engine

A mathematical and high-performance implementation of **Selective State Space Models (Mamba / S4)** from scratch with **Blelloch Associative Parallel Prefix Scan** in NumPy.

## 📐 Mathematical Formulation

### 1. Continuous-Time State Space System
$$\begin{aligned}
h'(t) &= A h(t) + B(t) x(t) \\
y(t) &= C(t) h(t) + D x(t)
\end{aligned}$$

### 2. Zero-Order Hold (ZOH) Discretization
For time step parameter $\Delta$:
$$\begin{aligned}
\bar{A} &= \exp(\Delta A) \\
\bar{B} &= (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot (\Delta B) = A^{-1}(\exp(\Delta A) - I) \cdot B
\end{aligned}$$

### 3. Selective Mechanism
Parameters are dynamically projected from the input sequence $x_t$:
$$\begin{aligned}
\Delta_t &= \text{softplus}(\text{Linear}_{\Delta}(x_t) + \text{bias}_\Delta) \\
B_t &= \text{Linear}_B(x_t) \\
C_t &= \text{Linear}_C(x_t)
\end{aligned}$$

### 4. Blelloch Parallel Associative Scan
The recurrent equation $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ is computed in parallel across the sequence dimension using the associative binary operator:
$$(a_j, b_j) \bullet (a_i, b_i) = (a_j a_i, \; a_j b_i + b_j)$$
yielding an algorithmic depth of $O(\log L)$ rather than $O(L)$.

---

## 🛠️ CLI Quickstart

The tool is accessible globally as `mamba-ssm`:

### Run Mathematical Verification
```bash
mamba-ssm verify
```
Validates:
1. Exact parity ($< 10^{-10}$ error) between parallel associative scan and sequential recurrence across sequence lengths $L \in [8, 1024]$.
2. Continuous-to-discrete Zero-Order Hold (ZOH) analytical convergence.
3. HiPPO-LegS transition matrix Hurwitz stability (all eigenvalues have strictly negative real parts).
4. Step-by-step $O(1)$ token inference state preservation.

### Benchmark Latency & Throughput
```bash
mamba-ssm benchmark
```

### Run Forward Pass & State Trajectory
```bash
mamba-ssm run --seq-len 128 --dim 32 --state 16
```

### Inspect Architecture Specs
```bash
mamba-ssm info
```
