---
name: kll-tdigest-ddsketch-quantiles
description: "Use when estimating streaming quantiles/CDFs. KLL, T-Digest, DDSketch, GK sketches."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Quantiles, KLL, TDigest, DDSketch, GreenwaldKhanna, Streaming, Sketches, Systems]
    related_skills: [countmin-spacesaving-heavy-hitters, hyperloglog-plusplus-cardinality, xor-filter-fuse-membership]
---

# KLL + T-Digest + DDSketch Streaming Quantile Oracle

Production-grade pure-Python **one-pass quantile / CDF** estimators: **KLL** (Karnin, Lang, Liberty FOCS 2016), **T-Digest** (Dunning & Ertl 2019), **DDSketch** (Masson et al. VLDB 2019), and **Greenwald–Khanna** (SIGMOD 2001) — plus exact sorted oracle, mergeable shards, multi-distribution benches, and theory ledgers. Completes the streaming sketch triad with HyperLogLog++ (*how many?*) and Count-Min/Space-Saving (*who is hot?*) for *where is the distribution?*.

## When to Use

- Estimating p50/p95/p99/p999 latency or score quantiles on streams too large for exact sorts
- Merging per-shard latency histograms without shipping raw samples (KLL / DDSketch / T-Digest merge)
- Choosing sketch knobs (KLL `k`, DDSketch `α`, T-Digest compression, GK `ε`) before a Flink/Druid/Datadog port
- Teaching rank-error vs relative-error SLAs and deterministic GK bounds

Don't use for: distinct counts (HLL++), heavy-hitter identity (CMS/Space-Saving), or membership (XOR/Fuse/Bloom).

## Theoretical Foundations

**KLL.** Hierarchical compactors; level-\(i\) capacity \(\sim c\cdot k\cdot 2^{-i/3}\); random half-compaction. Rank error \(\Theta(1/k)\); space \(O(k\log\log n)\) (FOCS 2016).

**T-Digest.** Weighted centroids with scale-function capacity denser near the median; excellent extreme tails (p99/p999); mergeable via centroid union + recluster.

**DDSketch.** Positive values on a \(\gamma\)-log lattice with \(\gamma=(1+\alpha)/(1-\alpha)\). Relative-error guarantee \(|\hat q - q|/q \le \alpha\); fully mergeable bucket-wise SUM (VLDB 2019).

**Greenwald–Khanna.** Deterministic \(\varepsilon\)-approximate summary of \((v,g,\Delta)\) tuples; size \(O((1/\varepsilon)\log(\varepsilon n))\).

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="kll-tdigest-ddsketch-quantiles selftest", timeout=120)
terminal(command="kll-tdigest-ddsketch-quantiles estimate --kind kll --n 8000 --stream gaussian", timeout=60)
terminal(command="kll-tdigest-ddsketch-quantiles estimate --kind tdigest --n 8000 --stream lognormal", timeout=60)
terminal(command="kll-tdigest-ddsketch-quantiles estimate --kind ddsketch --n 8000 --stream pareto --alpha 0.01", timeout=60)
terminal(command="kll-tdigest-ddsketch-quantiles merge --kind kll --n1 4000 --n2 4000", timeout=60)
terminal(command="kll-tdigest-ddsketch-quantiles compare --n 10000 --stream mixture", timeout=90)
terminal(command="kll-tdigest-ddsketch-quantiles bench --kind ddsketch --n 20000 --rounds 3", timeout=90)
terminal(command="kll-tdigest-ddsketch-quantiles theory --k 200 --alpha 0.01 --epsilon 0.01", timeout=15)
terminal(command="kll-tdigest-ddsketch-quantiles explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (40+ invariant checks) |
| `estimate` | Stream → quantile MAE/MRE + mean rank error |
| `stream` | Ingest + sketch stats + quantile JSON |
| `merge` | Two-shard union rank-error vs exact |
| `compare` | Rank exact / kll / tdigest / ddsketch / gk |
| `bench` | updates/sec + avg rank-error microbench |
| `theory` | k/α/ε → capacity, γ, space bounds |
| `explain` | Human protocol walkthrough |
| `status` | Kind + command catalog |

Kinds: `kll`, `tdigest`, `ddsketch`, `gk`, `exact`.  
Streams: `uniform`, `gaussian`, `lognormal`, `pareto`, `mixture`.

## Procedure

1. Pick SLA: absolute rank error → KLL `k` or GK `ε`; relative error on positives (latency ms) → DDSketch `α`; tail monitoring → T-Digest compression.
2. `theory` to size memory before production.
3. `compare --stream mixture` (or your production-like law) and check `mean_rank_err` / p99 rank error.
4. Prefer mergeable kinds (`kll`, `ddsketch`, `tdigest`) when sharding.

## Pitfalls

- Classic DDSketch is for **positive** values; signed streams need shift or KLL/T-Digest.
- Very small `k` / high `α` inflates tail rank error — always validate with `estimate` on a holdout exact oracle.
- GK is slower and larger at tiny `ε`; use as theory baseline, not default production path.
- Pure Python: microbench numbers are relative, not competitive with C++/Java sketch libs.

## References

- Karnin, Lang, Liberty — Optimal Quantile Approximation in Streams, FOCS 2016
- Dunning & Ertl — Computing Extremely Accurate Quantiles Using t-Digests, 2019
- Masson, Rim, Nguyen — DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees, VLDB 2019
- Greenwald & Khanna — Space-Efficient Online Computation of Quantile Summaries, SIGMOD 2001
