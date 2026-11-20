---
name: hyperloglog-plusplus-cardinality
description: "HyperLogLog++ streaming cardinality sketches (Flajolet / Google)."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Cardinality, HyperLogLog, HLL++, Streaming, Sketches, Systems]
    related_skills: [xor-filter-fuse-membership, s3fifo-tinylfu-ghost-ring, lsm-tree]
---

# HyperLogLog++ Cardinality Sketch Oracle

Production-grade pure-Python **streaming distinct-count** estimator: classic **HyperLogLog** (Flajolet et al. 2007), **HyperLogLog++** (Heule / Google 2013) with sparse registers + bias correction + linear-counting fallback, **PCSA/Flajolet–Martin** bitmaps, and standalone **Linear Counting** — with mergeable union, relative-error theory \(1.04/\sqrt{m}\), and JSON ledgers.

## When to Use

- Estimating unique users, IPs, session IDs, or query keys when the full set does not fit in RAM
- Merging per-shard distinct counts without shipping raw identifiers (register-wise MAX)
- Choosing precision `p` (memory vs error) before a production Redis/BigQuery/Druid port
- Teaching why HLL beats bitmap counters at scale and when linear counting still wins for tiny \(n\)

Don't use for: exact sets that must never err, heavy-hitter / top-k frequency (use Count-Min / Space-Saving), or membership tests (use Bloom / XOR / Fuse filters).

## Theoretical Foundations

**HyperLogLog.** Hash to 64 bits; top \(p\) bits index one of \(m=2^p\) registers; remaining bits yield rank \(\rho=1+\mathrm{clz}\). Update \(M[j]\leftarrow\max(M[j],\rho)\). Harmonic estimator:

\[
E=\alpha_m\, m^2\Big/\sum_{j=0}^{m-1} 2^{-M[j]},\qquad
\alpha_\infty=\tfrac{1}{2\ln 2},\quad
\sigma\approx 1.04/\sqrt{m}
\]

**HLL++.** Sparse list of \((j,\rho)\) while tiny; **linear counting** \(-m\ln(V/m)\) for \(E\lesssim 2.5m\); empirical **bias correction** in the mid window; 64-bit hash removes the classic 32-bit large-range patch for practical \(n\).

**Linear Counting (Whang).** Bitmap occupancy estimator — optimal only while many zero bits remain.

**PCSA.** Stochastic averaging of FM least-zero-bit ranks; pedagogical baseline with weaker constants.

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="hyperloglog-plusplus-cardinality selftest", timeout=120)
terminal(command="hyperloglog-plusplus-cardinality estimate --n 50000 --p 14 --kind hllpp", timeout=60)
terminal(command="hyperloglog-plusplus-cardinality merge --n1 5000 --n2 5000 --overlap 1000", timeout=60)
terminal(command="hyperloglog-plusplus-cardinality compare --n 8000 --p 12", timeout=90)
terminal(command="hyperloglog-plusplus-cardinality bench --n 30000 --p 14", timeout=90)
terminal(command="hyperloglog-plusplus-cardinality theory --p 14 --n 1000000", timeout=15)
terminal(command="hyperloglog-plusplus-cardinality explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (≥25 invariant checks) |
| `estimate` | Synthetic uniques → estimate + rel error |
| `stream` | Zipf multiset stream cardinality |
| `merge` | Two-sketch union (register max) demo |
| `compare` | Rank exact / linear / hll / hllpp / pcsa |
| `bench` | Error + updates/sec microbench |
| `theory` | \(m\), RSE, bytes, switch thresholds |
| `explain` | Human protocol walkthrough |
| `status` | Filter + command catalog |

Kinds: `hllpp` (default), `hll`, `linear`, `pcsa`, `exact`.

## Procedure

1. Pick target relative error → `theory --p P` (p=14 → ~1.6% 1σ, ~12 KiB dense).
2. `compare --n N --p P` → confirm HLL++ wins accuracy/bytes vs PCSA/linear for your N.
3. `estimate` / `stream` on a sample of real keys (hash stability is seed-scoped).
4. Shard with one sketch per partition; `merge` via register-wise max for global distincts.
5. Keep `selftest` green before citing error bands in a design doc.

## Pitfalls

- **Precision p is memory** — dense sketch is \(\lceil 6m/8\rceil\) bytes; do not over-allocate p=18 on tiny hosts.
- **Bias tables here are shape-faithful**, not a byte-for-byte dump of Google's multi-KB constants — validate empirical error on your key domain before production SLOs.
- **Merge requires equal p** — never mix precisions without rehashing into a common sketch.
- HLL answers *cardinality*, not *membership* and not *frequency*.
- Sparse mode only helps while \(n\) is small; large streams promote to dense automatically.
