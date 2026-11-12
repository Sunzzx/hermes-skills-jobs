---
name: aqlm-additive-multi-codebook-engine
description: AQLM multi-codebook additive VQ PTQ (NeurIPS 2024, arXiv:2401.06118).
category: inference
tags:
  - inference
  - aqlm
  - additive-quantization
  - multi-codebook
  - residual-vq
  - beam-search
  - weight-only
  - llm-serving
  - post-training
  - extreme-compression
---

# AQLM Additive Multi-Codebook Engine (arXiv:2401.06118)

AQLM is extreme weight-only PTQ (often 2–3 bpw). Core move:

```
w ≈ Σ_{m=1}^{M} C_m[i_m]     # additive multi-codebook VQ on d-dim groups
```

- **Effective codebook size** `K^M` while index storage is only `M·log2(K)` bits per group.
- **Residual multi-stage K-means** trains codebooks; **beam search** encodes at PTQ time.
- Codebooks shared per layer → amortized codebook overhead → 0 at 7B scale.
- Complements QuIP# (lattice + RHT), GPTQ/AWQ (scalar OBS/activation scales), SmoothQuant.

## CLI Usage

```bash
# Ablation ledger: RTN2/3/4 / VQ-M1 / AQLM-greedy / AQLM-beam
aqlm-additive-multi-codebook-engine status --rows 32 --cols 32

# Weight outlier + group stats + Hessian cond proxy
aqlm-additive-multi-codebook-engine calibrate --cols 64 --n-cal 256

# Train residual multi-codebooks
aqlm-additive-multi-codebook-engine train --n-codebooks 4 --codebook-size 16

# Encode one group (beam or --greedy)
aqlm-additive-multi-codebook-engine encode --beam-width 4

# Full AQLM quantize vs RTN4
aqlm-additive-multi-codebook-engine quantize --rows 32 --cols 32 --n-codebooks 2

# Codebook norms / effective K^M
aqlm-additive-multi-codebook-engine codebook

# Dim × M sweep
aqlm-additive-multi-codebook-engine benchmark --dims 16 24 32 --codebooks-list 1 2 4

# Invariant battery
aqlm-additive-multi-codebook-engine verify --trials 3

# Pack + 7B footprint theory
aqlm-additive-multi-codebook-engine pack --rows 32 --cols 32
aqlm-additive-multi-codebook-engine theory
```

## Architecture

- Pure Python, zero third-party deps (Hermes evolution standard).
- Group reshape → residual Lloyd K-means stages → greedy/beam additive encode.
- RTN scalar + single-codebook VQ ablations for fair ledgers.
- Bit-rate = index bits + fp16 codebook atoms amortized over layer weights.
- Output-side `||(W−Wq)X||` MSE alongside weight MSE/SQNR/cosine.

## When to Use

- Extreme weight-only PTQ research (2–4 bpw) and teaching additive VQ vs GPTQ/QuIP#.
- Comparing multi-codebook residual VQ vs lattice (E8) and scalar grids.
- Compression design reviews needing effective-K vs storage tradeoff tables.

## Verification

```bash
python -m pytest tests/test_aqlm_additive_multi_codebook_engine.py -q
aqlm-additive-multi-codebook-engine verify --trials 3
```
