---
name: xlstm-extended-memory
description: "xLSTM sLSTM/mLSTM extended memory (Beck 2024)."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Frontier-AI, xLSTM, sLSTM, mLSTM, SequenceModels, LSTM]
    related_skills: [mamba-ssm, gated-deltanet, titans-neural-memory]
---

# xLSTM Extended Long Short-Term Memory Engine

Production-grade pure-Python reference for **xLSTM** (Beck et al., arXiv:2405.04517): exponential-gated **sLSTM** with normalizer + head-local memory mixing, matrix-memory **mLSTM** with outer-product KV covariance updates, residual RMSNorm stacks, and probe tasks (copy, delayed parity, associative recall, majority) versus classic LSTM.

## When to Use

- Comparing revision-capable exponential gates vs sigmoid LSTM gates
- Teaching matrix memory / Fast Weight Programmer updates (mLSTM ≈ linear attention cell)
- Benchmarking associative recall and parity state-tracking capacity of sLSTM vs mLSTM vs LSTM
- Estimating param/FLOP ledgers before a CUDA/mLSTM kernel port

Don't use for: production training at LLM scale (this is a stdlib math reference), tokenizer pipelines, or replacing Transformer serving stacks.

## Theoretical Foundations

**Classic LSTM limit.** Sigmoid gates \(i,f \in (0,1)\) only decay/blend the scalar cell; they cannot freely *revise* stored magnitudes after a forget.

**sLSTM.** Exponential input/forget gates with stabilizer \(m_t\) (log-sum-exp style):

\[
m_t=\max(\log f_t+m_{t-1},\log i_t),\quad
i' = e^{\log i_t-m_t},\quad
f' = e^{\log f_t+m_{t-1}-m_t}
\]

\[
c_t=f'\odot c_{t-1}+i'\odot z_t,\quad
n_t=f'\odot n_{t-1}+i',\quad
h_t=o_t\odot\frac{c_t}{n_t+\varepsilon}
\]

Head-local recurrent mixing matrix \(R\) mixes \(h\to\) gate pre-activations (state tracking).

**mLSTM.** Matrix covariance memory (parallelizable — no hidden→gate mix):

\[
C_t=f'C_{t-1}+i'(vk^\top),\quad
n_t=f'n_{t-1}+i'k,\quad
h \propto \frac{C_t q}{n_t^\top q}
\]

**xLSTM block.** Residual pre-RMSNorm stacks alternating sLSTM (nonlinear tracking) and mLSTM (associative storage).

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="xlstm-extended-memory selftest", timeout=60)
terminal(command="xlstm-extended-memory compare --task parity --steps 80", timeout=120)
terminal(command="xlstm-extended-memory compare --task assoc --length 8 --steps 100", timeout=120)
terminal(command="xlstm-extended-memory simulate --cell mlstm --task assoc", timeout=30)
terminal(command="xlstm-extended-memory bench --dim 32 --seq 64", timeout=60)
terminal(command="xlstm-extended-memory theory --dim 64 --seq 1024", timeout=15)
terminal(command="xlstm-extended-memory revision --rank", timeout=30)
terminal(command="xlstm-extended-memory explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (≥18 invariant checks) |
| `simulate --cell C --task T` | One forward + state norms JSON |
| `compare --task T` | Rank lstm/slstm/mlstm (readout SGD probes) |
| `bench` | tok/s + ms/step microbench |
| `theory` | Params, FLOPs, complexity ledger |
| `revision --rank` | Exp-gate revision + mLSTM rank demo |
| `explain` / `status` | Walkthrough / catalog |

Cells: `lstm`, `slstm`, `mlstm`.

Tasks: `copy`, `parity`, `assoc`, `majority`.

## Procedure

1. `selftest` — must be green before citing numbers.
2. `theory --dim 64` — read param ratios and \(O(\cdot)\) notes.
3. `compare --task parity` then `compare --task assoc` — inspect winner + eval_acc.
4. `simulate --cell slstm` / `mlstm` — confirm normalizer / Frobenius state growth.
5. `revision --rank` — show exponential revision vs sigmoid and covariance rank proxy.

## Pitfalls

- Probe `compare` trains a **linear readout on frozen random cells** to isolate memory architecture cheaply — not full BPTT accuracy claims.
- Exponential gates **require** the \(m_t\) stabilizer; removing it NaNs on moderate logits.
- mLSTM \(C_t\) is \(d_m\times d_m\) — state bytes grow quadratically; theory ledger shows this.
- Associative recall vocab/pairs must satisfy `val ≥ n_pairs` in the generator — don't set vocab < 2·pairs.
- Head count for sLSTM must divide `d_hid`.

## Verification

- `xlstm-extended-memory selftest` exits 0 with ≥18 checks
- `pytest tests/test_xlstm_extended_memory.py -q` green
- `theory` JSON includes sLSTM stabilizer and mLSTM covariance equations
