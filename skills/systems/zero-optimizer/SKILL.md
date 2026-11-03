---
name: zero-optimizer
description: "ZeRO: Zero Redundancy Optimizer for distributed deep learning (Stage 1 P_os, Stage 2 P_os+g, Stage 3 P_os+g+p, and ZeRO-Offload) with machine-epsilon AdamW equivalence."
category: systems
status: production
tags:
  - distributed-training
  - zero-redundancy-optimizer
  - deepspeed
  - memory-optimization
  - model-parallelism
  - adamw
  - spmd
---

# ZeRO: Zero Redundancy Optimizer Execution Engine

A mathematically verified, production-grade implementation of the **ZeRO (Zero Redundancy Optimizer)** paradigm (Rajbhandari et al., DeepSpeed / Microsoft Research 2020) for large-scale distributed deep learning.

ZeRO eliminates the massive memory redundancies inherent in data-parallel training (where parameters, gradients, and optimizer states are traditionally replicated across all workers) while retaining the simplicity of Data Parallelism (DP) and maintaining **exact mathematical equivalence** to monolithic FP32 AdamW.

---

## Theoretical Foundations & Mathematical Formulations

In standard mixed-precision training (FP16/BF16 weights and activations with FP32 master weights and AdamW states), memory is dominated by model states:

$$\text{Memory}_{\text{DP}} = \underbrace{2\Psi}_{\text{FP16 weights}} + \underbrace{2\Psi}_{\text{FP16 gradients}} + \underbrace{12\Psi}_{\text{FP32 AdamW: master}(4\Psi) + m(4\Psi) + v(4\Psi)} = 16\Psi \text{ bytes}$$

where $\Psi$ is the total parameter count.

### 1. ZeRO-Stage 1: Optimizer State Partitioning ($P_{\text{os}}$)
- **Partitioning:** The $12\Psi$ bytes of optimizer state ($m_t, v_t$, master weights) are divided equally across $N_d$ data-parallel ranks.
- **Memory per GPU:**
  $$\text{Memory}_{\text{ZeRO-1}} = 2\Psi + 2\Psi + \frac{12\Psi}{N_d} = 4\Psi + \frac{12\Psi}{N_d} \text{ bytes}$$
- **Communication:** Exactly identical to standard DP ($2\Psi$ All-Reduce via Reduce-Scatter + All-Gather).
- **Benefit:** 4x memory reduction on large clusters ($N_d \ge 64$).

### 2. ZeRO-Stage 2: Gradient Partitioning ($P_{\text{os}+\text{g}}$)
- **Partitioning:** Gradients are reduce-scattered during the backward pass such that rank $r$ only retains the gradient partition corresponding to its owned parameter partition.
- **Memory per GPU:**
  $$\text{Memory}_{\text{ZeRO-2}} = 2\Psi + \frac{2\Psi}{N_d} + \frac{12\Psi}{N_d} = 2\Psi + \frac{14\Psi}{N_d} \text{ bytes}$$
- **Communication:** $2\Psi$ communication volume ($1\times$ Reduce-Scatter + $1\times$ All-Gather parameters).
- **Benefit:** 8x memory reduction with 0% extra communication overhead.

### 3. ZeRO-Stage 3: Full Parameter Partitioning ($P_{\text{os}+\text{g}+\text{p}}$)
- **Partitioning:** Parameters $\Psi$ are partitioned across all $N_d$ ranks persistently. During forward and backward passes, full weights are dynamically gathered layer-by-layer via All-Gather and discarded immediately after computation.
- **Memory per GPU:**
  $$\text{Memory}_{\text{ZeRO-3}} = \frac{16\Psi}{N_d} \text{ bytes}$$
- **Communication:** $3\Psi$ communication volume ($1.5\times$ standard DP).
- **Benefit:** Linear memory reduction proportional to cluster size $N_d$, enabling trillion-parameter models on standard clusters.

### 4. ZeRO-Offload
- Exploits host CPU RAM and PCIe interconnect by offloading FP32 master weights, momentum, and variance to CPU, executing the optimizer step on host CPU while GPU handles forward/backward compute.

---

## CLI Quick Reference

```bash
# 1. Verify mathematical equivalence against Monolithic FP32 AdamW across all 4 stages
zero-optimizer verify --world-size 4 --steps 10

# 2. Benchmark memory footprint and cluster communication scaling for frontier models (1.5B to 405B)
zero-optimizer bench --world-size 8

# 3. Interactive memory architecture visualization
zero-optimizer demo --world-size 4

# 4. Engine status and supported collective primitives
zero-optimizer status
```

---

## Python API Integration

```python
from zero_optimizer import (
    ZeROCommunicator,
    ZeROStage1Optimizer,
    ZeROStage2Optimizer,
    ZeROStage3Optimizer,
    ZeROOffloadOptimizer,
    AdamWConfig,
)

# Initialize collective communicator across data-parallel ranks
comm = ZeROCommunicator(world_size=8)

# Configure ZeRO-Stage 2 optimizer
opt = ZeROStage2Optimizer(
    rank=0,
    world_size=8,
    params=model.parameters(),
    config=AdamWConfig(lr=1e-3, weight_decay=0.01),
    comm=comm,
)
```
