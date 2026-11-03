---
name: pipeline-parallel
description: "Megatron-LM 1F1B and GPipe pipeline parallelism: activation stashing, bubble minimization, and machine-epsilon gradient equivalence."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  category: systems
  type: cli
  command: pipeline-parallel
  hermes:
    tags: [PipelineParallelism, 1F1B, GPipe, MegatronLM, DistributedTraining, ActivationStashing, BubbleMinimization]
---

# ⚡ Megatron-LM 1F1B Pipeline Parallelism Engine

## Overview
A mathematically rigorous, production-grade implementation of Megatron-LM (Narayanan et al., 2021) and PipeDream (Harlap et al., 2018) **1F1B (One-Forward-One-Backward)** Pipeline Parallelism. The engine partitions deep neural network layers across sequential pipeline stages, coordinates micro-batch execution, stashes intermediate activations with optimal memory bounds, and minimizes pipeline idle bubble fractions while guaranteeing bitwise / machine-epsilon numerical equivalence to monolithic model execution.

---

## Theoretical Foundations & Mathematical Formulations

### 1. The Pipeline Bubble Problem
When a model with $L$ layers is partitioned across $P$ pipeline stages and executed with micro-batching across $M$ micro-batches ($M \ge P$):

Let $t_F$ be the forward pass execution time of one micro-batch on one stage, and $t_B$ be the backward execution time (typically $t_B \approx 2 t_F$). In a normalized unit where $t_F = t_B = 1$:

- **Ideal Execution Time (Zero Bubble)**: $T_{\text{ideal}} = 2M$
- **Startup Bubble**: Incurred while Stage $P-1$ waits for Stage 0 to propagate the first micro-batch: $(P - 1) t_F$
- **Drain Bubble**: Incurred while Stage 0 waits for the final backward gradients from Stage $P-1$: $(P - 1) t_B$
- **Total Idle Bubble Time**: $T_{\text{bubble}} = (P - 1)(t_F + t_B) = 2(P - 1)$
- **Bubble Fraction**: 
  $$F_{\text{bubble}} = \frac{T_{\text{bubble}}}{T_{\text{total}}} = \frac{(P - 1)(t_F + t_B)}{M(t_F + t_B) + (P - 1)(t_F + t_B)} = \frac{P - 1}{M + P - 1}$$

For $M \gg P$ (e.g., $P = 8, M = 64$), the bubble fraction drops to $\frac{7}{71} \approx 9.86\%$.

### 2. 1F1B (One-Forward-One-Backward) Steady State
In classical GPipe (Huang et al., 2019), all $M$ forward micro-batches are executed across all stages before any backward micro-batch begins. This forces Stage 0 to keep all $M$ intermediate micro-batch activations stored in memory, creating an $O(M)$ memory explosion.

In **1F1B**, each stage transitions into a steady state where it alternates executing **one forward micro-batch** and **one backward micro-batch**:

1. **Warmup Phase**:
   Stage $p \in \{0, \dots, P-1\}$ executes $W_p = \min(P - 1 - p, M)$ forward micro-batches.
   - Stage 0 executes $P - 1$ warmup forwards.
   - Stage $P-1$ executes 0 warmup forwards (it immediately runs backward $B_0$ after forward $F_0$).

2. **1F1B Steady State Phase**:
   Each stage alternates:
   $$\text{Step } i: \quad F_{W_p + i} \longrightarrow B_i \quad \text{for } i \in [0, M - W_p - 1]$$
   Running a backward pass immediately frees the stashed activation of micro-batch $i$ before running the next forward pass.

3. **Cooldown Phase**:
   Each stage finishes its remaining $W_p$ backward micro-batches without launching new forwards.

### 3. Activation Stashing & Strict Memory Bounds
Under 1F1B, the number of activations stashed simultaneously on Stage $p$ is strictly bounded by:
$$\text{Peak Stashed Micro-batches}(p) \le P - p \le P$$

| Schedule | Stage 0 Peak Activations | Stage $P-1$ Peak Activations | Memory Scaling |
| :--- | :--- | :--- | :--- |
| **GPipe** | $M$ micro-batches | $M$ micro-batches | $O(M)$ (Unbounded) |
| **1F1B** | $P$ micro-batches | $1$ micro-batch | $O(P)$ (Strictly Bounded) |

When $M = 32$ and $P = 8$, 1F1B achieves a **4x to 8x reduction** in activation memory footprint compared to GPipe.

### 4. Interleaved 1F1B Schedule
In Interleaved 1F1B (Megatron-LM), each physical device hosts $v$ virtual stages (e.g., $v=2$ means Device 0 holds Stage 0 and Stage $P$). Micro-batches cycle through virtual stages, shrinking the bubble fraction by a factor of $v$:
$$F_{\text{bubble}}^{\text{interleaved}} = \frac{P - 1}{v \cdot M + P - 1}$$

---

## CLI Quick Reference

```bash
# 1. View pipeline architecture specifications and bubble formulations
pipeline-parallel status

# 2. Visualize the 1F1B and GPipe execution timelines & matrix
pipeline-parallel schedule --stages 4 --microbatches 8

# 3. Benchmark memory savings across frontier LLMs (LLaMA-3.1 8B, 70B, 405B)
pipeline-parallel bench --stages 8 --microbatches 32

# 4. Run formal mathematical equivalence verification against Monolithic model
pipeline-parallel verify --stages 4 --microbatches 8
```

---

## Mathematical Invariant Verification

To ensure 100% production fidelity, the engine verifies four key invariants on every run:
1. **Loss Equivalence**: $\left|\mathcal{L}_{\text{1F1B}} - \mathcal{L}_{\text{mono}}\right| < 10^{-12}$
2. **Parameter Gradient Equivalence**: $\max_l \left\| \nabla W_{l,\text{1F1B}} - \nabla W_{l,\text{mono}} \right\|_\infty < 10^{-12}$
3. **Stash Bound Invariant**: $\text{PeakAct}(p) \le P - p$ for every stage $p \in [0, P-1]$
4. **GPipe Invariant**: Bitwise loss match with $O(M)$ activation footprint.
