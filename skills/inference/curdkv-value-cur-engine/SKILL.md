---
name: curdkv-value-cur-engine
description: "CurDKV value-guided CUR leverage KV compression (NeurIPS 2025)."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Frontier-AI, KV-Cache, CurDKV, CUR, LeverageScores, ValueGuided, NeurIPS2025, Inference]
    related_skills: [keydiff-kv-engine, snapkv-engine, h2o-kv-cache, streaming-llm-engine, gearkv-engine]
---

# CurDKV: Value-Guided KV Compression via Approximated CUR (NeurIPS 2025)

Training-free KV cache compression that ranks tokens by **value-centric CUR leverage scores**, not query-key attention mass alone. Preserves the dominant subspace of the attention output $Y=\mathrm{softmax}(QK^\top)V$ under a hard token budget, and avoids wasting slots on high-attention near-zero value "decoys".

## When to Use

- Long-context decode under a hard KV memory budget where SnapKV/H2O keep decoys
- You need end-to-end **output reconstruction** (MSE / cosine on $Y$), not just attention-mass recovery
- Prefill-time token selection that must stay compatible with dense attention over a subset
- Benchmarking value-aware vs score-only eviction on synthetic multi-cluster + needle corpora

Don't use for: full model serving stacks (this is the selection-policy reference), or weight quantization.

## Theoretical Foundations

Attention-score heuristics assume high $A_{qi}$ implies importance. Counterexample: decoy tokens with large $\|k\|$ but $\|v\|\approx 0$ absorb mass yet contribute nothing to $Y$.

**CUR column selection.** For matrix $M$ with columns $m_j$, top-$r$ right singular vectors $V_r$ induce statistical leverage scores

$$\ell_j = \|e_j^\top V_r\|^2 = \sum_{s=1}^{r} (V_r)_{js}^2$$

Sampling / top-$k$ by $\ell_j$ yields near-optimal column subsets for range approximation (Drineas, Mahoney & Muthukrishnan; Mahoney & Drineas).

**CurDKV contribution matrix.** Form columns

$$c_j = \big(\alpha\sqrt{\bar m_j}+(1-\alpha)\big)\cdot \phi(\|v_j\|)\, v_j$$

with mean probe mass $\bar m_j$ and value-norm gate $\phi$. Power / subspace iteration approximates $V_r$ of $C$ without a full SVD ($O(T d r\,n_{\mathrm{iter}})$). Retain attention-sink prefix + local window anchors, fill remaining budget by leverage rank (optional hybrid blend with mass).

Oracle metrics: mean $\|Y - \hat Y\|_2^2$, cosine$(Y,\hat Y)$, needle recall, decoy retention rate.

## Prerequisites

- Python 3.10+ stdlib only (no numpy/torch required)

## How to Run

```bash
terminal(command="curdkv-value-cur-engine selftest", timeout=60)
terminal(command="curdkv-value-cur-engine bench --seq-len 384 --budget-ratio 0.25", timeout=120)
terminal(command="curdkv-value-cur-engine compare --seq-len 320 --budget-ratio 0.22", timeout=120)
terminal(command="curdkv-value-cur-engine compress --policy curdkv --seq-len 256 --budget 64 --json", timeout=60)
terminal(command="curdkv-value-cur-engine leverage --seq-len 200 --rank 6 --json", timeout=60)
terminal(command="curdkv-value-cur-engine theory --seq-len 8192 --json", timeout=30)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `curdkv-value-cur-engine selftest` | CI invariant gate (7 groups) |
| `curdkv-value-cur-engine bench` | Multi-policy MSE/cosine/needle/decoy table |
| `curdkv-value-cur-engine compare` | Lift vs attn-score + StreamingLLM |
| `curdkv-value-cur-engine compress --policy curdkv_hybrid` | Hybrid leverage+mass path |
| `curdkv-value-cur-engine leverage` | Top leverage tokens + label means |
| `curdkv-value-cur-engine theory` | 7B-class FP16 KV footprint ledger |

## Procedure

1. Build or load $K,V\in\mathbb{R}^{T\times d}$ (synthetic generator included: sinks, clusters, needles, decoys).
2. Collect a small probe/obs query set (prefill window or synthetic retrieval probes).
3. Choose budget $B=\lfloor\rho T\rfloor$ (typical $\rho\in[0.10,0.35]$) and CUR rank $r\in[4,16]$.
4. Run `compress_curdkv` (contribution) or `curdkv_value` / `curdkv_hybrid`.
5. Verify output MSE ≤ attn-score baseline and decoy_rate ≤ attn-score on the structured corpus.
6. Emit keep indices into the serving runtime's block table / page allocator.

Completion criterion: `selftest` exits 0; `compare` reports non-negative `mse_lift_vs_attn` or positive `decoy_avoidance_lift` with stable needle recall.

## Pitfalls

- Pure-Python subspace iteration is $O(T d r n_{\mathrm{iter}})$ — fine to ~2–4K tokens; shard heads or sketch further above that.
- If sink+local ≥ budget, leverage picks are skipped (anchors only).
- Leverage sum is renormalized to ≈ rank; do not treat absolute $\ell_j$ as probabilities without dividing by $\sum\ell$.
- Output MSE is a proxy, not task EM; validate on LongBench / RULER before production.
- Decoy advantage appears when the corpus contains high-key / low-value tokens; uniform random corpora understate the gap.

## Verification

```bash
curdkv-value-cur-engine selftest
pytest tests/test_curdkv_value_cur_engine.py -q
```
