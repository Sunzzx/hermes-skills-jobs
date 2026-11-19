---
name: s3fifo-tinylfu-ghost-ring
description: "S3-FIFO + TinyLFU hybrid cache with CMS ghost rings."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Caching, S3FIFO, TinyLFU, CountMinSketch, Eviction, Systems]
    related_skills: [lsm-tree, blink-tree, lockfree-concurrency]
---

# S3-FIFO + Window-TinyLFU Ghost Ring Cache Oracle

Production-grade pure-Python reference for modern cache admission and eviction: **S3-FIFO** three-ring queues (Yang et al. 2023), **Window-TinyLFU** with Count-Min Sketch frequency admission (Einziger / Caffeine), a **hybrid** that keeps S3-FIFO rings and adds TinyLFU duels on cold small-evictions, plus LRU / FIFO / CLOCK / ARC baselines under Zipf, scan, hotspot, and mixed workloads.

## When to Use

- Choosing a cache policy for KV proxies, CDN edges, or API gateways under Zipf + scan mixes
- Teaching scan pollution (why LRU dies on sequential reads) and ghost-list reuse detection
- Benchmarking hit rate / ops-sec of S3-FIFO vs W-TinyLFU vs ARC before a production port
- Validating Count-Min Sketch aging and Bloom doorkeeper first-sighting behavior

Don't use for: distributed coherent caches (needs invalidation protocol), sized multi-object packing (this is key-count capacity), or persistence (no WAL).

## Theoretical Foundations

**S3-FIFO.** Capacity split ≈10% *small* + 90% *main*; *ghost* holds keys only (~1× capacity). Inserts enter small. On small eviction: freq bit set → promote to main; else drop data and remember key in ghost. Ghost hit re-admits into main. Main uses 1-bit CLOCK reinsertion. One-hit wonders never pollute main → scan resistance with O(1) ops.

**W-TinyLFU.** Window LRU (~1%) + SLRU main (probation / protected). Admission: `CMS(candidate) ≥ CMS(victim)`. Bloom doorkeeper skips CMS bump on first sighting. CMS: depth-4 Count-Min, width ≈ 4C, periodic right-shift aging.

**Hybrid (`s3fifo-tinylfu`).** S3-FIFO rings + CMS duel on cold small-eviction path; ghost ring preserved. Target: scan-proof structure with frequency-aware tail filtering.

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="s3fifo-tinylfu-ghost-ring selftest", timeout=120)
terminal(command="s3fifo-tinylfu-ghost-ring compare --capacity 256 --ops 30000 --workload zipf", timeout=90)
terminal(command="s3fifo-tinylfu-ghost-ring simulate --policy s3fifo-tinylfu --workload scan --capacity 128", timeout=60)
terminal(command="s3fifo-tinylfu-ghost-ring bench --capacity 512 --ops 80000", timeout=120)
terminal(command="s3fifo-tinylfu-ghost-ring theory --capacity 1024", timeout=15)
terminal(command="s3fifo-tinylfu-ghost-ring explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (≥18 invariant checks) |
| `compare` | Rank all policies on one workload |
| `simulate --policy P` | Single-policy run + inventory JSON |
| `bench` | Hit-rate + throughput microbench |
| `theory` | Caps, CMS ε/δ, complexity ledger |
| `explain` | Human protocol walkthrough |
| `status` | Policy + workload catalog |

Policies: `lru`, `fifo`, `clock`, `s3fifo`, `w-tinylfu`, `arc`, `s3fifo-tinylfu`.

Workloads: `zipf`, `ir`, `scan`, `loop`, `hotspot`, `mixed`.

## Procedure

1. Pick capacity C and workload (start `zipf` α=1.2, then `scan`).
2. `compare` → read ranking hit_rate and winner.
3. `simulate --policy s3fifo-tinylfu` and inspect `inventory` (small/main/ghost, cms_ages).
4. Confirm `selftest` green before relying on numbers in a design doc.

## Pitfalls

- Capacity is **entry count**, not bytes — pass uniform `size` only for byte-hit demos.
- CMS **over-estimates** only; never under-estimates (Cormode bound).
- Doorkeeper true-negatives are exact; false-positives may bump CMS one extra time.
- ARC `p` adaptation is sensitive to ghost length; short runs look noisy.
- Hybrid admission rejects can be 0 on tiny C with mild Zipf — use cold floods in tests.

## Verification

```bash
terminal(command="s3fifo-tinylfu-ghost-ring selftest", timeout=120)
terminal(command="pytest -q tests/test_s3fifo_tinylfu_ghost_ring.py", timeout=120)
```

Expect selftest `checks ≥ 18` and pytest all green. Compare on `scan` should show `s3fifo` hit rate near or above `lru`.

## References

- Yang et al., *FIFO queues are all you need for cache eviction* (S3-FIFO), 2023
- Einziger, Scalosub, Friedman, *TinyLFU: A Highly Efficient Cache Admission Policy*, TOCS 2017
- Cormode & Muthukrishnan, Count-Min Sketch
- Megiddo & Modha, *ARC*, FAST 2003
- Ben Manes, Caffeine W-TinyLFU design notes
