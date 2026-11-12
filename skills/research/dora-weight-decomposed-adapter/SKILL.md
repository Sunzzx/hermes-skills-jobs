---
name: dora-weight-decomposed-adapter
description: DoRA weight-decomposed LoRA PEFT (ICML 2024).
category: research
tags:
  - research
  - dora
  - lora
  - peft
  - weight-decomposition
  - low-rank
  - fine-tuning
  - magnitude-direction
  - icml-2024
  - nvidia
---

# DoRA Weight-Decomposed Adapter (arXiv:2402.09353)

DoRA (ICML 2024 — Liu et al., NVIDIA) closes the gap between LoRA and
full fine-tuning by decomposing each weight into **magnitude** and
**direction**, then applying LoRA only on the directional component:

```
W0 = m ⊙ (V / ||V||_c)     # column-wise unit direction
V  = W0 + BA               # LoRA on direction (W0 frozen)
W' = m ⊙ (V / ||V||_c)     # trainable magnitude m re-composes
```

Unlike vanilla LoRA (`W = W0 + BA`), DoRA learns column magnitudes
independently of the low-rank angular update — matching FT dynamics
while adding only `n_cols` extra parameters over LoRA.

## CLI Usage

```bash
# Health check + PEFT ledger
dora-weight-decomposed-adapter status --rows 64 --cols 64 --rank 8

# Magnitude / direction decomposition round-trip
dora-weight-decomposed-adapter decompose --rows 32 --cols 40

# Gradient energy split (mag vs direction-tangent)
dora-weight-decomposed-adapter analyze --rows 24 --cols 32

# Single optimizer step (dora | lora | full)
dora-weight-decomposed-adapter step --method dora --rows 24 --cols 32 --rank 4

# Train linear task
dora-weight-decomposed-adapter train --method dora --steps 60 --rank 4
dora-weight-decomposed-adapter train --method lora --steps 60 --rank 4
dora-weight-decomposed-adapter train --method full --steps 60

# Ablation ranking
dora-weight-decomposed-adapter ablate --rows 20 --cols 28 --rank 4

# Dim × rank sweep
dora-weight-decomposed-adapter benchmark --dims 16 24 32 --ranks 2 4 8

# Invariant battery
dora-weight-decomposed-adapter verify --trials 3

# Pack + LLaMA-7B PEFT theory
dora-weight-decomposed-adapter pack --rows 64 --cols 64 --rank 8
dora-weight-decomposed-adapter theory --rank 16
```

## Architecture

- Pure Python, zero third-party deps (Hermes evolution standard).
- Column-wise magnitude/direction decomposition with renorm guards.
- Analytic sphere-projected gradients for DoRA magnitude + LoRA factors.
- DoRA / LoRA / full-FT AdamW trainers on linear regression probes.
- Gradient energy split (magnitude-aligned vs direction-tangent).
- PEFT param ledger + LLaMA-7B-shaped adapter footprint theory.

## When to Use

- PEFT research comparing DoRA vs LoRA vs full fine-tuning.
- Teaching weight magnitude/direction decomposition.
- Estimating adapter parameter overhead before large-model runs.

## Verification

```bash
python -m pytest tests/test_dora_weight_decomposed_adapter.py -q
dora-weight-decomposed-adapter verify --trials 3
```
