---
name: quip-sharp-incoherence-lattice-engine
description: QuIP# RHT incoherence + E8 lattice PTQ (ICML 2024, arXiv:2402.04396).
category: inference
tags:
  - inference
  - quip-sharp
  - incoherence
  - hadamard
  - e8-lattice
  - ldlq
  - quantization
  - weight-only
  - llm-serving
  - post-training
---

# QuIP# Incoherence Lattice Engine (arXiv:2402.04396)

QuIP# is weight-only PTQ for extreme compression (≤4 bpw). Three moves:

1. **Randomized Hadamard Transform (RHT)** incoherence: `Ŵ = U W Vᵀ` smears
   channel outliers so weights look i.i.d. sub-Gaussian (ball-shaped).
2. **E8-family lattice VQ** on 8-dim blocks — optimal unit-ball packing in 8D,
   better than scalar grids on incoherent weights (E8P proxy codebook).
3. **BlockLDLQ** — LDL/Hessian adaptive rounding (QuIP lineage) polishes
   residuals; factors fuse at deploy (near weight-only lookup + optional FWHT).

Complements QuaRot/SpinQuant (joint W/A rotations), GPTQ/AWQ (weight OBS/activation
scale), and SmoothQuant (diagonal smooth).

## CLI Usage

```bash
# Ablation ledger: RTN / LDLQ / RHT / E8 / QuIP#
quip-sharp-incoherence-lattice-engine status --rows 32 --cols 32

# Outlier proxies before/after RHT + Hessian cond
quip-sharp-incoherence-lattice-engine calibrate --cols 64 --n-cal 256

# RHT-only incoherence report
quip-sharp-incoherence-lattice-engine incohere --rows 32 --cols 32

# BlockLDLQ path (no lattice)
quip-sharp-incoherence-lattice-engine ldlq --rows 32 --w-bits 4

# Full QuIP# (RHT + E8 + LDLQ polish)
quip-sharp-incoherence-lattice-engine quantize --rows 32 --cols 32 --w-bits 2

# E8 codebook stats
quip-sharp-incoherence-lattice-engine codebook

# Dim/bit sweep
quip-sharp-incoherence-lattice-engine benchmark --dims 16 24 32 --bits-list 2 3 4

# Invariant battery (orth, roundtrip, incoherence, competitive)
quip-sharp-incoherence-lattice-engine verify --trials 3

# Pack + 7B footprint theory
quip-sharp-incoherence-lattice-engine pack --rows 32 --cols 32
quip-sharp-incoherence-lattice-engine theory
```

## Architecture

- Pure Python, zero third-party deps (Hermes evolution standard).
- Sylvester Hadamard + sign-flip RHT; non-pow2 via modified Gram-Schmidt.
- Damped Hessian `H = 2XXᵀ + λI` → LDL / BlockLDLQ residual redistribution.
- E8 shells: origin, integer (±2,0⁷)/(±1,±1,0⁶), half-integer even-parity coset.
- Ablations expose where each QuIP# ingredient pays off.

## When to Use

- Extreme weight-only PTQ research (2–4 bpw) and teaching QuIP vs GPTQ/AWQ.
- Comparing incoherence (RHT) vs learned/random rotations (SpinQuant/QuaRot).
- Lattice/vector quantization baselines for LLM compression design reviews.

## Verification

```bash
pytest tests/test_quip_sharp_incoherence_lattice_engine.py -q
quip-sharp-incoherence-lattice-engine verify --trials 3
```
