---
name: galore-gradient-lowrank-projector
description: GaLore low-rank gradient projection training (ICML 2024).
category: research
tags:
  - research
  - galore
  - low-rank
  - gradient-projection
  - memory-efficient
  - adamw
  - lora
  - llm-training
  - subspace-iteration
  - icml-2024
---

# GaLore Gradient Low-Rank Projector (arXiv:2403.03507)

GaLore (ICML 2024 Oral — Zhao et al.) keeps **full-parameter** weights while
projecting only the *gradient* into a slowly refreshed rank-`r` subspace so
Adam/Adafactor moments shrink from `O(mn)` to `O(r · min(m,n))`:

```
SVD / subspace-iter on G ∈ R^{m×n}
  std side:  m≤n → R = PᵀG (r×n)     else R = GQ (m×r)
Adam ρ(R) on the core only
ΔW = P·ρ(R)  or  ρ(R)·Qᵀ
refresh P/Q every update_proj_gap steps
```

Unlike LoRA (`W = W0_frozen + BA`), GaLore does **not** reparameterize `W` or
freeze a base — full-rank learning dynamics, low-rank optimizer state.

## CLI Usage

```bash
# Health check + memory ledger
galore-gradient-lowrank-projector status --rows 64 --cols 64 --rank 8

# Lemma 3.1: stable-rank collapse under G=A−BWC
galore-gradient-lowrank-projector calibrate --rows 32 --cols 48 --steps 40

# One-shot project + reconstruct
galore-gradient-lowrank-projector project --rows 32 --cols 40 --rank 4

# Single GaLore-AdamW step
galore-gradient-lowrank-projector step --rows 24 --cols 32 --rank 4

# Train linear task (galore | full | lora)
galore-gradient-lowrank-projector train --method galore --steps 60 --rank 4
galore-gradient-lowrank-projector train --method full --steps 60
galore-gradient-lowrank-projector train --method lora --steps 60 --rank 4

# Ablation ranking
galore-gradient-lowrank-projector ablate --rows 20 --cols 28 --rank 4

# Dim × rank sweep
galore-gradient-lowrank-projector benchmark --dims 16 24 32 --ranks 2 4 8

# Invariant battery
galore-gradient-lowrank-projector verify --trials 3

# Pack + LLaMA-7B theory
galore-gradient-lowrank-projector pack --rows 64 --cols 64 --rank 8
galore-gradient-lowrank-projector theory --rank 128
```

## Architecture

- Pure Python, zero third-party deps (Hermes evolution standard).
- Subspace-iteration thin SVD (top-`r` left/right singular vectors).
- One-sided projector (`std` / `reverse` / `left` / `right`).
- GaLore-AdamW, full AdamW, LoRA-AdamW baselines on a linear regression task.
- Synthetic PSD dynamics proving gradient stable-rank collapse (paper Eq. 6).
- Optimizer-state memory theory for LLaMA-7B-shaped layers.

## When to Use

- Memory-efficient full-parameter pre-training / fine-tuning research.
- Comparing GaLore vs LoRA vs full Adam under matched rank budgets.
- Teaching low-rank gradient structure and optimizer-state compression.

## Verification

```bash
python -m pytest tests/test_galore_gradient_lowrank_projector.py -q
galore-gradient-lowrank-projector verify --trials 3
```
