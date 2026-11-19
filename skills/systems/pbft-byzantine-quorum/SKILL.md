---
name: pbft-byzantine-quorum
description: "PBFT Byzantine consensus with view-change & quorum certs."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [DistributedSystems, PBFT, ByzantineFaultTolerance, Consensus, Quorum]
    related_skills: [raft-consensus, hotstuff-chained-bft, zksnark-prover, crdt-sync]
---

# Practical Byzantine Fault Tolerance (PBFT) Quorum Consensus Engine

Castro & Liskov PBFT (OSDI 1999 / TOCS 2002) reference for `n = 3f+1` replicas: three-phase agreement (PRE-PREPARE → PREPARE → COMMIT), `2f+1` certificates, view-change with prepared-certificate recovery, checkpoints, and a linearizable KV state machine — pure-Python discrete-event simulator with crash / mute / equivocation fault models.

Complements crash-fault Raft (`raft-consensus`) with full Byzantine safety under quorum intersection.

## When to Use

- Teaching or verifying PBFT normal-case and view-change safety invariants
- Comparing crash-fault (Raft) vs Byzantine (PBFT) quorum requirements (`n≥2f+1` vs `n≥3f+1`)
- Chaos-testing primary crash, prepare equivocation, and lossy networks against agreement
- Estimating O(n²) normal-case message cost before a production BFT port

Don't use for: production multi-node deployment (needs real MACs/TLS, WAL, timers), proof-of-stake BFT (Tendermint/HotStuff), or crash-only clusters (prefer Raft).

## Theoretical Foundations

**Fault bound.** `n = 3f + 1` so any two quorums of size `2f+1` intersect in ≥ `f+1` replicas (hence ≥ 1 honest).

**Normal case (view v, primary p = v mod n).**
1. Client → primary: `REQUEST`
2. Primary assigns seq `n`, broadcasts `PRE-PREPARE(v,n,d)` with `d = H(m)`
3. Backups accept (auth, in-view, fresh `(v,n)`), broadcast `PREPARE(v,n,d)`
4. `prepared(v,n,d)` when pre-prepare + prepares from `2f+1` distinct replicas
5. Broadcast `COMMIT`; on `2f+1` commits → `committed-local`; execute in order
6. Client accepts after `f+1` matching `REPLY`s

**View change.** On timeout, `VIEW-CHANGE(v+1)` carries prepared certificates; new primary gathers `2f+1` and issues `NEW-VIEW` with set `O` re-proposing highest prepared request per sequence (or NOP).

**Checkpoints.** Every `k` seqs, `CHECKPOINT` digests; stable watermark at `2f+1` matching digests enables log GC.

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="pbft-byzantine-quorum selftest", timeout=120)
terminal(command="pbft-byzantine-quorum simulate --scenario happy --ops 12", timeout=60)
terminal(command="pbft-byzantine-quorum simulate --scenario primary_crash", timeout=60)
terminal(command="pbft-byzantine-quorum simulate --scenario equivocation", timeout=60)
terminal(command="pbft-byzantine-quorum chaos --drop 0.05", timeout=90)
terminal(command="pbft-byzantine-quorum bench --n 4 --ops 40", timeout=60)
terminal(command="pbft-byzantine-quorum theory --n 7", timeout=15)
terminal(command="pbft-byzantine-quorum explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (26 invariant checks) |
| `simulate --scenario happy` | Honest n=4/7 agreement + validity |
| `simulate --scenario primary_crash` | View-change after primary crash |
| `simulate --scenario equivocation` | Byzantine double-prepare safety |
| `simulate --scenario lossy` | Drop/delay network stress |
| `chaos` | Crash + equivocator + loss combined |
| `bench` | Throughput microbench |
| `theory` | Quorum + O(n²) message ledger |
| `explain` | Human protocol walkthrough |

## Procedure

1. Choose `n=3f+1` (4→f=1, 7→f=2); construct `PBFTCluster`.
2. Submit client commands via primary; pump discrete-event inbox to completion.
3. Assert `check_agreement()` across honest replicas and `check_validity(submitted)`.
4. For liveness under faults: arm crash on primary, broadcast view-changes, install NEW-VIEW, resume.
5. For Byzantine safety: mark one backup `EQUIVOCATE`; confirm honest stores still converge.

Completion criterion: `selftest` exits 0 (26/26); `simulate --scenario happy` reports `agreement=true`.

## Pitfalls

- `n < 3f+1` cannot guarantee quorum intersection with one honest replica — engine rejects `n < 4`.
- Client must retry with a fresh timestamp after timeout; identical `(client, ts)` is a retransmission.
- View-change without prepared certs inserts NOPs for gaps — seq holes are filled, not skipped.
- Pure-Python flood is O(n²) msgs/request; production uses authenticators and often collector trees.
- Equivocation is detected locally on conflicting pre-prepares; safety still holds via certificates even if undetected at a single node.

## Verification

```bash
terminal(command="pbft-byzantine-quorum selftest", timeout=120)
terminal(command="pytest tests/test_pbft_byzantine_quorum.py -q", timeout=180)
```

Expect selftest `26/26`, pytest all green, happy-path `agreement=true`.
