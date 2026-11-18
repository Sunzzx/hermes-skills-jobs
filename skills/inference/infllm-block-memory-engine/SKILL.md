---
name: infllm-block-memory-engine
description: "InfLLM training-free block KV memory retrieval (arXiv:2402.04617)."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Frontier-AI, KV-Cache, InfLLM, BlockMemory, LongContext, Retrieval, Inference]
    related_skills: [streaming-llm-engine, h2o-kv-cache, snapkv-engine, keydiff-kv-engine, curdkv-value-cur-engine]
---

# InfLLM: Training-Free Block Memory for Infinite Context (arXiv:2402.04617)

Long-context engine that **offloads** past KV into fixed blocks and **retrieves** only the blocks whose representative tokens match the current query. Unlike StreamingLLM/H2O (permanent eviction) or dense GPU caches, InfLLM keeps full history on host memory and pulls a sparse working set per decode step.

## When to Use

- Context lengths beyond GPU KV capacity (32K–1M+) where eviction would drop distant needles
- You need **block-level retrieval** with representative-token sketches, not token-level eviction ranks
- Benchmarking training-free long-context memory vs StreamingLLM, sliding window, H2O, random blocks
- Planning host/GPU KV footprints for 7B-class GQA models under InfLLM budgets

Don't use for: learned retrieval modules (this is training-free), weight quantization, or full multi-tenant serving stacks.

## Theoretical Foundations

**Partition.** Past tokens outside the attention-sink prefix and local window are split into blocks of length $L$.

**Representatives.** Each block keeps $r$ summary tokens chosen by intra-block self-attention centrality mixed with key-space diversity (farthest-from-chosen). Block sketch $= \{k_{i_1},\ldots,k_{i_r}\}$.

**Retrieval.** For query $q$, block score is mean (or max) cosine similarity against representatives (centroid fallback). Load top-$k$ blocks from external memory into the GPU working set:

$$\mathcal{W}(q) = \mathrm{Sinks} \cup \mathrm{Local} \cup \bigcup_{b \in \mathrm{TopK}(q)} \mathrm{Tokens}(b)$$

**Oracle metrics.** Distant needle recall, needle-block hit-rate, attention-output MSE/cosine vs dense cache, GPU token count, compression ratio.

## Prerequisites

- Python 3.10+ stdlib only (no numpy/torch required)

## How to Run

```bash
terminal(command="infllm-block-memory-engine selftest", timeout=60)
terminal(command="infllm-block-memory-engine bench --seq-len 768 --top-k 4", timeout=120)
terminal(command="infllm-block-memory-engine compare --seq-len 768 --json", timeout=120)
terminal(command="infllm-block-memory-engine retrieve --seq-len 512 --top-k 3", timeout=60)
terminal(command="infllm-block-memory-engine simulate --steps 8 --top-k 3", timeout=60)
terminal(command="infllm-block-memory-engine theory --seq-len 128000 --json", timeout=30)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `infllm-block-memory-engine selftest` | CI invariant gate (8 groups) |
| `infllm-block-memory-engine bench` | Multi-policy needle/MSE/GPU-token table |
| `infllm-block-memory-engine compare` | Lift vs StreamingLLM / random / H2O |
| `infllm-block-memory-engine retrieve` | Dump top blocks + needle alignment |
| `infllm-block-memory-engine simulate` | Stepwise decode block-pull trace |
| `infllm-block-memory-engine theory` | 7B-class dense vs InfLLM KV ledger |

## Procedure

1. Build or stream $K,V$ for the prompt; pin sinks (first $s$ tokens) and a local window of size $w$.
2. Partition the middle span into blocks of size $L$; select $r$ representatives per block; offload full block KV to host.
3. At each decode step, score blocks by representative cosine; DMA top-$k$ blocks into GPU KV pages.
4. Run dense attention over $\mathcal{W}(q)$ only.
5. Verify distant needle recall ≫ StreamingLLM and block_hit_rate > random-block baseline at matched GPU budget.

Completion criterion: `selftest` exits 0; `compare` reports non-negative `needle_lift_vs_streaming` with `compression_ratio ≥ 1`.

## Pitfalls

- If top-$k \cdot L$ is tiny vs the number of needle-bearing blocks, recall collapses — raise $k$ or shrink $L$.
- Representative count $r=1$ under-sketches multi-topic blocks; $r\in[4,8]$ is the usual band.
- Scoring uses key sketches only; value-decoy blocks can still win if keys align — pair with value-aware filters (CurDKV) if needed.
- Pure-Python attention over the working set is $O(|\mathcal{W}|d)$ per query — fine for reference; production needs paged kernels.
- Synthetic needles are key-aligned probes; validate on RULER / InfiniteBench before shipping.

## Verification

```bash
infllm-block-memory-engine selftest
pytest tests/test_infllm_block_memory_engine.py -q
```
