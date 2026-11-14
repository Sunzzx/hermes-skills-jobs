---
name: vera-vector-random-adaptation-engine
description: VeRA vector-based random matrix adaptation PEFT (ICLR 2024).
category: research
tags:
  - research
  - vera
  - lora
  - peft
  - fine-tuning
  - efficiency
---

# 🔬 VeRA: Vector-based Random Matrix Adaptation Engine

Production-grade implementation and analytical framework for **VeRA** (ICLR 2024, arXiv:2310.11454) — *"VeRA: Vector-based Random Matrix Adaptation"* by Dawid J. Kopiczko, Tijmen Blankevoort, and Jakub M. Tomczak (Qualcomm AI Research).

## Theoretical Overview

Standard Low-Rank Adaptation (LoRA) adapts linear layers $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ via trainable low-rank matrices $B \in \mathbb{R}^{d_{out} \times r}$ and $A \in \mathbb{R}^{r \times d_{in}}$:
$$\Delta W = \frac{\alpha}{r} B A$$
For deep networks (such as 32-layer LLaMA models), LoRA requires tens of millions of trainable parameters ($2 \cdot d \cdot r \cdot L$).

### Key VeRA Innovation:
1. **Frozen Shared Matrices:** Generates a single pair of random matrices $\hat{A} \in \mathbb{R}^{r \times d_{max, in}}$ and $\hat{B} \in \mathbb{R}^{d_{max, out} \times r}$ initialized once with a fixed PRNG seed and frozen across **ALL** layers of the network.
2. **Trainable Scaling Vectors:** Each adapted layer trains only two lightweight 1D diagonal scaling vectors: $d \in \mathbb{R}^{d_{out}}$ and $b \in \mathbb{R}^r$:
$$\Delta W^{(l)} = \frac{\alpha}{r} \operatorname{diag}(d^{(l)}) \hat{B} \operatorname{diag}(b^{(l)}) \hat{A}$$
3. **Extreme Efficiency:** Trainable parameters drop from $O(L \cdot (d_{out} + d_{in}) \cdot r)$ to $O(L \cdot (d_{out} + r))$, achieving **10x to 100x parameter reduction** compared to LoRA while retaining downstream accuracy.

## Parameter Comparison (LLaMA-7B, Rank=16)

- **Full Model Weights:** 6,476,005,376 params (12.06 GiB FP16)
- **LoRA Trainable Params:** 39,976,960 (76.25 MiB)
- **DoRA Trainable Params:** 41,336,832 (78.84 MiB)
- **VeRA Trainable Params:** 1,363,456 (2.60 MiB)
- **Compression Factor:** **29.32x fewer parameters** than LoRA; **4,749x parameter efficiency** vs Full Fine-Tuning.

## CLI Commands

```bash
# Status & basic PEFT ledger summary
./bin/vera-vector-random-adaptation-engine status --rank 16

# Compare parameter footprints across model architectures (llama-7b, llama-3-8b, llama-70b)
./bin/vera-vector-random-adaptation-engine compare --arch llama-7b --rank 16

# Run synthetic adaptation optimization
./bin/vera-vector-random-adaptation-engine simulate --in-features 128 --out-features 128 --rank 8 --steps 50
```
