---
name: xor-filter-fuse-membership
description: "XOR + Binary Fuse static set filters (Graf & Lemire)."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Filters, XORFilter, BinaryFuse, BloomFilter, Membership, Systems]
    related_skills: [s3fifo-tinylfu-ghost-ring, lsm-tree, hnsw-vector-search]
---

# XOR + Binary Fuse Filter Membership Oracle

Production-grade pure-Python reference for **static approximate set membership**: classic **Bloom** and **blocked Bloom** baselines, **XOR filters** (Graf & Lemire SEA 2020), and **Binary Fuse filters** (2022) with 8/16-bit fingerprints, hypergraph peeling construction, empirical FPR ledgers, and bits/key theory.

## When to Use

- Choosing a frozen membership structure for URL allow-lists, dictionary blocks, package name sets, or LSM bloom replacements
- Teaching why XOR/Fuse beat Bloom on query cost (3 XORs vs k bit probes) and on space at low target FPR
- Benchmarking bits/key and empirical FPR before a production C++/Rust port
- Validating peel construction success rates under ~1.23n (XOR) and ~1.125n (Fuse) array factors

Don't use for: dynamic inserts/deletes (rebuild required), adversarial unkeyed sets without a secret seed, or exact sets that must never false-positive (use perfect hashing / sorted arrays).

## Theoretical Foundations

**Bloom (1970).** \(m = -n \ln p / (\ln 2)^2\) bits, \(k = (m/n)\ln 2\) hashes. Query touches \(k\) random bits. Optimal bits/key \(\approx 1.44\log_2(1/p)\).

**XOR filter (Graf & Lemire 2020).** Array of \(\approx 1.23n\) fingerprints in 3 segments. Each key is a 3-uniform hyperedge. Peel degree-1 vertices; assign the lone slot so \(\mathrm{fp}(k)=t[h_0]\oplus t[h_1]\oplus t[h_2]\). Query is three loads + two XORs. Theory FPR \(=2^{-f}\) for \(f\)-bit fingerprints (~1/256 for 8-bit).

**Binary Fuse (2022).** Same peel algebra with consecutive segment windows → cache-local probes and array factor \(\rightarrow\sim 1.125n\).

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="xor-filter-fuse-membership selftest", timeout=120)
terminal(command="xor-filter-fuse-membership compare --n 3000 --neg 8000", timeout=90)
terminal(command="xor-filter-fuse-membership build --kind fuse8 --n 5000", timeout=60)
terminal(command="xor-filter-fuse-membership bench --kind xor16 --n 8000", timeout=90)
terminal(command="xor-filter-fuse-membership theory --n 100000 --fpr 0.001", timeout=15)
terminal(command="xor-filter-fuse-membership explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (≥18 invariant checks) |
| `build --kind K` | Construct filter + metadata JSON |
| `query --key K` | Demo membership probe |
| `compare` | Rank bloom/blocked/xor/fuse on one workload |
| `bench` | FPR + query ns/op microbench |
| `theory` | Bits/key + complexity ledger |
| `explain` | Human protocol walkthrough |
| `status` | Filter + command catalog |

Kinds: `bloom`, `blocked-bloom`, `xor8`, `xor16`, `fuse8`, `fuse16`.

## Procedure

1. Size the set \(n\) and pick target FPR (start `0.01`, then `1/65536`).
2. `theory` → read Bloom vs XOR8/Fuse8 vs XOR16/Fuse16 bits/key.
3. `compare --n N` → confirm ranking and empirical FPR.
4. `build --kind fuse8` (or `xor16` for tight FPR) and verify `recall_check: 1.0`.
5. Keep `selftest` green before citing numbers in a design doc.

## Pitfalls

- **Unique keys required** for peel construction — duplicates leave residual cycles; dedupe first.
- Peel may retry seeds (up to `max_attempts`); very dense factors (<1.18 for XOR) raise failure rate.
- Fingerprint FPR is \(2^{-f}\) *only under ideal hash uniformity* — measure empirical FPR on your key domain.
- Filters are **static** — any insert requires full rebuild.
- Blocked Bloom trades a little FPR for cache locality; do not expect identical FPR to classic Bloom at equal bits.

## Verification

```bash
terminal(command="xor-filter-fuse-membership selftest", timeout=120)
terminal(command="pytest tests/test_xor_filter_fuse_membership.py -q", timeout=180)
```

Expect selftest all-pass exit 0 and pytest green. `compare` should report a `winner_bits` among fuse/xor/bloom rows with `ok: true`.
