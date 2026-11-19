---
name: hotstuff-chained-bft
description: "HotStuff chained BFT with linear QC pipeline."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [DistributedSystems, HotStuff, BFT, Consensus, QuorumCertificate, PODC2019]
    related_skills: [pbft-byzantine-quorum, raft-consensus, crdt-sync]
---

# HotStuff Chained BFT Consensus Engine

Yin et al. HotStuff (PODC 2019 / arXiv:1803.05069) reference for `n = 3f+1` replicas: chained pipeline with generic Quorum Certificates, 3-chain commit, locked-QC safe-node voting, linear O(n) leader-aggregated messaging, pacemaker NEW-VIEW recovery, and a linearizable KV state machine — pure-Python discrete-event simulator with crash / Byzantine-fork / lossy fault models.

Complements classic PBFT (`pbft-byzantine-quorum`) with linearity and responsiveness; crash-only clusters still prefer Raft (`raft-consensus`).

## When to Use

- Teaching or verifying HotStuff 3-chain commit and locked-QC safety
- Comparing PBFT O(n²) three-phase broadcasts vs HotStuff O(n) pipelined QCs
- Chaos-testing leader crash, Byzantine fork attempts, and lossy networks
- Estimating linear message cost before a production BFT / LibraBFT-style port

Don't use for: production multi-node deployment (needs threshold signatures, WAL, timers), proof-of-stake finality gadgets alone (see Casper FFG), or crash-only clusters (prefer Raft).

## Theoretical Foundations

**Fault bound.** `n = 3f + 1`, quorum size `2f+1`. Any two quorums intersect in ≥ `f+1` replicas (hence ≥ 1 honest).

**Chained phase (view v, leader ℓ = v mod n).**
1. Replicas send `NEW-VIEW` with highest QC to ℓ (O(n))
2. Leader proposes block `b` extending `highQC`, embedding justify QC
3. Replicas vote iff `safe-node(b)`; votes return to leader (O(n))
4. Leader forms `QC(b)` from `2f+1` votes; next proposal carries it

**3-chain commit.** Commit `b0` when there exist `b1, b2` with consecutive views and nested QCs: `b0 ← b1 ← b2`, `b1.justify = QC(b0)`, `b2.justify = QC(b1)`.

**safe-node.** Vote if `b` extends `lockedQC` **or** `b.justify.view > lockedQC.view`.

**Linearity.** Happy-path ≈ `3n` messages/block vs PBFT ≈ `2n²+n`.

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="hotstuff-chained-bft selftest", timeout=120)
terminal(command="hotstuff-chained-bft simulate --scenario happy --ops 12", timeout=60)
terminal(command="hotstuff-chained-bft simulate --scenario leader_crash", timeout=60)
terminal(command="hotstuff-chained-bft simulate --scenario byzantine_fork", timeout=60)
terminal(command="hotstuff-chained-bft chaos --drop 0.05", timeout=90)
terminal(command="hotstuff-chained-bft bench --n 4 --ops 40", timeout=60)
terminal(command="hotstuff-chained-bft compare --n 7", timeout=15)
terminal(command="hotstuff-chained-bft theory --n 7", timeout=15)
terminal(command="hotstuff-chained-bft explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (37 invariant checks) |
| `simulate --scenario happy` | Honest n=4/7 agreement + validity |
| `simulate --scenario leader_crash` | Pacemaker after leader crash |
| `simulate --scenario byzantine_fork` | Byzantine double-propose safety |
| `simulate --scenario lossy` | Drop/delay network stress |
| `chaos` | Byzantine + crash + loss combined |
| `bench` | Throughput microbench |
| `compare` | HotStuff O(n) vs PBFT O(n²) ledger |
| `theory` | Quorum + message-complexity ledger |
| `explain` | Human protocol walkthrough |

## Procedure

1. Choose `n=3f+1` (4→f=1, 7→f=2); construct `HotStuffCluster`.
2. Drive `propose_once` / `run_ops`; each view forms one QC and advances the chain.
3. After ≥3 consecutive honest views, assert `check_agreement()` and `stores_agree()`.
4. Inject crash/Byzantine/lossy faults; re-check honest prefix agreement.

Completion criterion: `selftest` exits 0; happy/crash/byz/lossy report `agreement=true`.

## Pitfalls

- Tail blocks need 2 extra pipeline rounds before 3-chain commit fires — always flush with empty proposes.
- Byzantine leaders may propose forks; honest `safe-node` rejects them, but view still advances.
- This simulator abstracts threshold signatures as voter-set QCs; real HotStuff uses BLS/threshold multi-sig.
- `n` must be ≥4 and satisfy `n=3f+1` for the textbook quorum proofs.

## Verification

```bash
terminal(command="hotstuff-chained-bft selftest", timeout=120)
terminal(command="pytest tests/test_hotstuff_chained_bft.py -q", timeout=120)
```

Expect: selftest 37/37; pytest all passed; compare shows PBFT/HotStuff msg ratio ≥2× at n=4.
