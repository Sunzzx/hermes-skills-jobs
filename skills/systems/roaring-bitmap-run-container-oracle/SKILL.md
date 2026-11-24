---
name: roaring-bitmap-run-container-oracle
description: "Use when compressing sparse bitsets/postings. Roaring array/bitmap/run containers."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [RoaringBitmap, Bitsets, RunLength, Containers, InvertedIndex, Systems, Compression]
    related_skills: [xor-filter-fuse-membership, hyperloglog-plusplus-cardinality, s3fifo-tinylfu-ghost-ring]
---

# Roaring Bitmap + Run-Length Container Bitset Oracle

Production-grade pure-Python **Roaring Bitmaps** (Chambi, Lemire, Kaser SPE 2016): partition the 32-bit universe into 2^16 chunks, store each chunk as an **Array** (sorted uint16, card ≤ 4096), **Bitmap** (1024×uint64 = 8192 B), or **Run** (RLE start/length) container — whichever serializes smallest. Complements XOR/Fuse (*approximate membership*), HLL++ (*how many distinct?*), and S3-FIFO (*what stays hot?*) with *exact compressed set algebra* for inverted indexes and bitmap columns.

## When to Use

- Choosing container thresholds before a Druid / Lucene / Spark / ClickHouse roaring port
- Estimating bits/value for sparse postings vs dense bitsets vs pure RLE
- Validating OR/AND/XOR/ANDNOT and rank/select against a Python-set oracle
- Teaching why mixed-density ID spaces beat one-size-fits-all bitmaps

Don't use for: approximate membership (XOR/Fuse/Bloom), distinct counts (HLL++), or streaming quantiles (KLL/T-Digest).

## Theoretical Foundations

**Partition.** For \(x \in [0,2^{32})\): high = \(x \gg 16\), low = \(x \bmod 2^{16}\). Only non-empty highs are stored.

**Array container.** Sorted unique lows; binary-search membership; ~16 bits/value + header; promotes to bitmap when card > 4096.

**Bitmap container.** Fixed 8192-byte bitset over the chunk; excellent at medium-high density; demotes to array when card ≤ 4096.

**Run container.** Non-overlapping `(start, length)` runs; wins on long contiguous ranges (often ≪ array and ≪ bitmap).

**Algebra.** Chunk-wise merge with cardinality-aware re-optimization after OR/AND/XOR/ANDNOT.

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="roaring-bitmap-run-container-oracle selftest", timeout=120)
terminal(command="roaring-bitmap-run-container-oracle build --workload clustered --n 8000", timeout=60)
terminal(command="roaring-bitmap-run-container-oracle op --op and --workload uniform --n 2000", timeout=60)
terminal(command="roaring-bitmap-run-container-oracle rank --workload sparse --n 1000", timeout=60)
terminal(command="roaring-bitmap-run-container-oracle compare --workload range --n 20000", timeout=90)
terminal(command="roaring-bitmap-run-container-oracle bench --workload dense --n 12000 --rounds 3", timeout=90)
terminal(command="roaring-bitmap-run-container-oracle theory --card 4096 --n-runs 1 --density 0.01", timeout=15)
terminal(command="roaring-bitmap-run-container-oracle explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (40+ invariant checks) |
| `build` | Workload → roaring stats + bits/value |
| `query` | Membership + rank probe |
| `op` | or/and/xor/andnot vs Python set |
| `rank` | Rank/select inversion check |
| `compare` | Roaring vs naive set / pure RLE / dense bitset |
| `bench` | Build/query/union microbench |
| `theory` | Threshold + bits/value ledger |
| `explain` | Human protocol walkthrough |
| `status` | Kind + command catalog |

Workloads: `uniform`, `clustered`, `dense`, `sparse`, `range`.

## Procedure

1. Characterize density: sparse multi-chunk → expect many **array** containers; solid ranges → **run**; hot chunks → **bitmap**.
2. `theory --card N --n-runs R` to size memory before a production port.
3. `compare --workload` matching production shape; confirm `membership_ok` and bits/value vs naive/RLE.
4. `op --op and/or` on sharded key sets to validate algebra before wiring inverted-index merges.

## Pitfalls

- Classic roaring targets **32-bit** keys; 64-bit IDs need Roaring64 / hierarchical maps.
- Pure-Python microbench is relative — CRoaring/Java roaring will be far faster; trust correctness + space model here.
- Random inserts that thrash array↔bitmap conversions: batch-add then optimize, or rebuild containers periodically.
- Dense bitset baseline only meaningful when the universe fits in memory; roaring shines when U ≫ |S|.

## References

- Chambi, Lemire, Kaser, Godin — Better bitmap performance with Roaring bitmaps, SPE 2016 (arXiv:1402.6407)
- Lemire et al. — Roaring bitmaps: Implementation of an optimized software library, SPE 2018
- Wu, Otoo, Shoshani — Optimizing bitmap indices with efficient compression, TODS 2006
