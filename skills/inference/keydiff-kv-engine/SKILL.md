---
name: keydiff-kv-engine
description: "KeyDiff key-similarity KV eviction without attention scores (NeurIPS 2025)."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Frontier-AI, KV-Cache, KeyDiff, Eviction, FlashAttention, NeurIPS2025, Inference]
    related_skills: [h2o-kv-cache, streaming-llm-engine, snapkv-engine, quest-attention-engine]
---

# KeyDiff: Key Similarity-Based KV Cache Eviction (NeurIPS 2025)

Training-free KV cache eviction that ranks tokens by **geometric distinctiveness of key vectors**, not by attention scores. Compatible with FlashAttention and other fused kernels that never materialize the full attention matrix.

## When to Use

- Long-context decode under a hard KV memory budget
- Needles / rare entities get dropped by StreamingLLM-style sink+recent windows
- You cannot afford attention-score accumulation (H2O) or must stay FlashAttention-native
- Benchmarking eviction policies on synthetic multi-cluster key corpora

Don't use for: full model serving stacks (this is the eviction policy reference), or weight quantization.

## Theoretical Foundations

KeyDiff observes that **geometrically distinctive keys** (high minimum distance to the retained set) correlate with high attention mass. Under budget $B$:

1. **Pin anchors**: attention-sink prefix + recent local window.
2. **Farthest-point sampling (Gonzalez)**: repeatedly add $\arg\max_i \min_{j \in S} d(k_i, k_j)$ with $d = 1 - \cos$ or $\ell_2$.
3. **Block variant**: FPS on block mean-reps for $O((n/B_{\mathrm{blk}})^2)$ scaling.

Oracle evaluation uses probe queries biased toward needles and cluster centroids; metrics are attention-mass recovery and needle recall vs StreamingLLM / random / magnitude baselines.

## Prerequisites

- Python 3.10+ stdlib only (no numpy/torch required)

## How to Run

```bash
terminal(command="keydiff-kv-engine selftest", timeout=60)
terminal(command="keydiff-kv-engine bench --seq-len 1024 --budget-ratio 0.25", timeout=120)
terminal(command="keydiff-kv-engine compare --seq-len 768 --budget-ratio 0.2", timeout=120)
terminal(command="keydiff-kv-engine evict --policy keydiff --seq-len 512 --budget 128 --json", timeout=60)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `keydiff-kv-engine selftest` | CI invariant gate |
| `keydiff-kv-engine bench` | Multi-policy table (mass%, needle%, diversity, latency) |
| `keydiff-kv-engine compare` | Lift vs StreamingLLM |
| `keydiff-kv-engine evict --policy keydiff_block` | Block-compressed FPS path |

## Procedure

1. Build or load key matrix $K \in \mathbb{R}^{n \times d}$ (synthetic generator included).
2. Choose budget $B = \lfloor \rho n \rfloor$ (typical $\rho \in [0.15, 0.35]$).
3. Run `evict_keydiff` (exact) or `evict_keydiff_block` (long-context).
4. Verify mass recovery and needle recall beat StreamingLLM on the structured corpus.
5. Emit keep indices into the serving runtime's block table / page allocator.

Completion criterion: `selftest` exits 0 and `compare` reports non-negative mass lift with needle recall ≥ StreamingLLM.

## Pitfalls

- Pure-Python FPS is $O(nB d)$ — use `--policy keydiff_block` above ~4K tokens.
- If local_window + sink_len ≥ budget, diversity picks are skipped (anchors only).
- Cosine metric assumes roughly normalized keys; unnormalized caches prefer `--metric l2`.
- Oracle mass is a proxy, not task EM; validate on LongBench / Math500 before production.

## Verification

```bash
keydiff-kv-engine selftest
pytest tests/test_keydiff_kv_engine.py -q
```

Expect: exact budget cardinality, sink retention, KeyDiff ≥ random on needle/mass for the synthetic multi-cluster corpus.
