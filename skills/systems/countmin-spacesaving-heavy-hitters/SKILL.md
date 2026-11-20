---
name: countmin-spacesaving-heavy-hitters
description: "Count-Min + Space-Saving streaming heavy-hitter oracle."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [HeavyHitters, CountMin, SpaceSaving, MisraGries, Streaming, Sketches, Systems]
    related_skills: [hyperloglog-plusplus-cardinality, xor-filter-fuse-membership, s3fifo-tinylfu-ghost-ring]
---

# Count-Min + Space-Saving Heavy-Hitter Stream Oracle

Production-grade pure-Python **one-pass frequency / heavy-hitter** estimators: **Count-Min Sketch** (Cormode & Muthukrishnan 2005) with optional **conservative updates**, **Count-Sketch** (Charikar et al. 2002), **Space-Saving** (Metwally et al. 2005), and **Misra–Gries** (1982) — plus mergeable CMS shards, Zipf benches, top-k precision/recall, and theory ledgers. Pairs with HyperLogLog++ (*how many uniques?*) for *who is hot?*.

## When to Use

- Estimating per-key frequencies or top-k / φ-heavy hitters on streams too large for exact maps
- Merging per-shard frequency sketches without shipping raw event logs (CMS counter-wise SUM)
- Choosing CMS (ε, δ) or Space-Saving capacity k before a Redis/Druid/Flink port
- Teaching one-sided CMS error vs deterministic Space-Saving / Misra–Gries guarantees

Don't use for: exact distinct counts (use HLL++), membership tests (XOR/Fuse/Bloom), or quantile/CDF estimation (T-Digest / DDSketch).

## Theoretical Foundations

**Count-Min.** \(d\) hash rows × \(w\) counters; query \(=\min_r T[r,h_r(x)]\). With \(w=\lceil e/\varepsilon\rceil\), \(d=\lceil\ln(1/\delta)\rceil\):

\[
\hat f(x) \ge f(x),\qquad \Pr\big[\hat f(x) > f(x)+\varepsilon\|f\|_1\big] \le \delta
\]

**Conservative CMS.** Only counters equal to the current min are raised — same worst-case bound, tighter practice.

**Count-Sketch.** Signed \(\pm 1\) row hashes; median estimator; unbiased, strong on \(\ell_2\) heavies.

**Space-Saving.** \(k\) tracked counters; on miss replace min \(m\) with \(\mathrm{count}=m+c\), error floor \(\varepsilon_i=m\). Every item with \(f > n/k\) is tracked; overestimate \(\le n/k\).

**Misra–Gries.** On full table, decrement all counters. Survivors include every item with \(f > n/(k+1)\).

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="countmin-spacesaving-heavy-hitters selftest", timeout=120)
terminal(command="countmin-spacesaving-heavy-hitters estimate --kind cms --events 20000 --unique 2000", timeout=60)
terminal(command="countmin-spacesaving-heavy-hitters topk --kind space_saving --capacity 50 --top-k 15", timeout=60)
terminal(command="countmin-spacesaving-heavy-hitters merge --kind cms --n1 5000 --n2 5000", timeout=60)
terminal(command="countmin-spacesaving-heavy-hitters compare --events 20000 --capacity 64", timeout=90)
terminal(command="countmin-spacesaving-heavy-hitters bench --kind cms_cons --rounds 3", timeout=90)
terminal(command="countmin-spacesaving-heavy-hitters theory --epsilon 0.001 --delta 1e-4", timeout=15)
terminal(command="countmin-spacesaving-heavy-hitters explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (40+ invariant checks) |
| `estimate` | Zipf stream → point MAE/MRE + top-k F1 |
| `stream` | Ingest + sketch stats JSON |
| `topk` | Top-k precision / recall / Jaccard |
| `merge` | Two-shard CMS / Count-Sketch union |
| `compare` | Rank exact / cms / cms_cons / cs / ss / mg |
| `bench` | updates/sec + avg F1 microbench |
| `theory` | ε,δ → width/depth/bytes; SS/MG thresholds |
| `explain` | Human protocol walkthrough |
| `status` | Kind + command catalog |

Kinds: `cms`, `cms_cons`, `count_sketch`, `space_saving`, `misra_gries`, `exact`.

## Procedure

1. Pick SLA: additive \(\varepsilon\|f\|_1\) → `theory --epsilon E --delta D`; or deterministic top-k → Space-Saving capacity \(k\).
2. `compare` on a Zipf sample matching production skew (\(\alpha\)).
3. For sharded logs: one CMS per partition, `merge` via counter SUM; probe candidates then rank.
4. For tiny RAM top-k dashboards: Space-Saving; use `estimate_range` lower bounds for φ-guarantees.
5. Keep `selftest` green before citing error bands in a design doc.

## Pitfalls

- **CMS needs a candidate set to surface top-k** — pure CMS has no key list; pair with a Space-Saving side table or known probe universe.
- **CMS never underestimates, Count-Sketch can** — do not treat CS estimates as upper bounds.
- **Merge requires identical (depth, width, seed, conservative)** — never mix shapes.
- **Space-Saving overestimates** tracked keys; use lower bound \(c-\varepsilon_i\) for safe φ thresholds.
- Misra–Gries counts are **lower** bounds after decrements — not interchangeable with Space-Saving totals.

## Verification

- `selftest` exits 0 with `passed == total`
- `topk --kind space_saving` F1 ≥ 0.4 on default Zipf
- `estimate --kind cms` reports `underestimate_keys == 0`
- pytest: `tests/test_countmin_spacesaving_heavy_hitters.py`
