---
name: qtip-trellis-incoherence-engine
description: QTIP trellis+incoherence PTQ (NeurIPS 2024, arXiv:2406.11235).
category: inference
tags:
  - inference
  - qtip
  - trellis
  - tcq
  - viterbi
  - incoherence
  - hadamard
  - weight-only
  - llm-serving
  - post-training
  - quantization
---

# QTIP Trellis + Incoherence Engine (arXiv:2406.11235)

QTIP (Cornell RelaxML, NeurIPS 2024) keeps QuIP#-style Randomized Hadamard
incoherence, then replaces lattice/NN vector lookup with **trellis-coded
quantization** (TCQ) decoded by Viterbi:

```
incoherence:  ŵ = H(s ⊙ w) / √n     (Gaussianizes outliers)
TCQ encode:   min-MSE path on Ungerboeck rate-R trellis
bitstream:    branch choices (R bits/dim) + fp16 group scales
decode:       FSM walk — linear in length, no exponential codebook
```

- **Incoherence** spreads spiky weight energy so coordinates look i.i.d. Gaussian.
- **TCQ + Viterbi** finds the global min-MSE reconstruction path under a finite-state codebook of size `n_states · 2^R` (linear, not exponential in dim).
- Ablations: RTN, Lloyd-Max, NN-VQ, TCQ-only (no RHT), full QTIP.

## CLI Usage

```bash
# QTIP vs RTN health check
qtip-trellis-incoherence-engine status --rows 32 --cols 32

# Incoherence kurtosis / outlier proxies
qtip-trellis-incoherence-engine calibrate --n 256

# 1-D encode ledger
qtip-trellis-incoherence-engine encode --n 128 --bits 2

# Viterbi path decode agreement
qtip-trellis-incoherence-engine decode --n 64

# Full matrix QTIP vs RTN + output MSE
qtip-trellis-incoherence-engine quantize --rows 24 --cols 24

# Ablation ranking
qtip-trellis-incoherence-engine ablate --rows 20 --cols 20

# Dim × bit sweep
qtip-trellis-incoherence-engine benchmark --dims 16 24 32 --bits-list 2 3 4

# Invariant battery
qtip-trellis-incoherence-engine verify --trials 3

# Pack + 7B footprint theory
qtip-trellis-incoherence-engine pack --rows 32 --cols 32
qtip-trellis-incoherence-engine theory
```

## Architecture

- Pure Python, zero third-party deps (Hermes evolution standard).
- FWHT + random-sign incoherence (exact orthogonal round-trip).
- Ungerboeck trellis graph, Lloyd-refined Gaussian edge labels, Viterbi DP.
- Group RMS scales; bit-rate = rate bits + fp16 meta (+ shared trellis codebook).

## When to Use

- 2–4 bpw weight-only PTQ research beyond QuIP# E8 / AQLM additive VQ.
- Teaching trellis-coded quantization and incoherence processing for LLMs.
- Comparing TCQ vs RTN / Lloyd / unstructured VQ under matched bit budgets.

## Verification

```bash
python -m pytest tests/test_qtip_trellis_incoherence_engine.py -q
qtip-trellis-incoherence-engine verify --trials 3
```
