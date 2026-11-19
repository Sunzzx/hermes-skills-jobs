---
name: swim-gossip-membership
description: "SWIM gossip failure detector and membership protocol."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [DistributedSystems, SWIM, Gossip, FailureDetector, Membership, DSN2002, Epidemic]
    related_skills: [kademlia-dht, raft-consensus, pbft-byzantine-quorum, hotstuff-chained-bft, crdt-sync]
---

# SWIM Gossip Failure Detector & Membership Engine

Das, Gupta, Motivala SWIM (DSN 2002) reference: scalable weakly-consistent infection-style process group membership with direct + indirect ping failure detection, suspicion/confirm sub-protocol, incarnation-number refutation, and epidemic piggyback dissemination — pure-Python discrete-event simulator with crash / mute / lossy / partition fault models.

Complements consensus engines (Raft / PBFT / HotStuff): SWIM answers *who is alive*, not *what is decided*. Production cousins: HashiCorp memberlist, Serf, Consul.

## When to Use

- Teaching SWIM vs all-to-all heartbeat message complexity
- Estimating failure-detection latency under loss, mute storms, partitions
- Modeling Serf/Consul/memberlist-style gossip before a Go/Rust port
- Comparing infection-style disseminate cost to mesh heartbeats at n=32..128

Don't use for: strong consistency / linearizability (use Raft/HotStuff), Byzantine membership authenticity (needs signed digests), or production multi-node deploy (needs UDP, timers, Lifeguard extensions).

## Theoretical Foundations

**Probe period T.** Each node round-robins a direct `PING`; expects `ACK` within `ping_timeout`.

**Indirect probe.** On timeout, ask `k` random peers (`PING-REQ`) to ping the target. Still silent → `SUSPECT`.

**Suspicion → confirm.** After `suspect_timeout` without higher-incarnation `ALIVE` refute → `DEAD` (confirm) and infect-gossip.

**Incarnation.** Node hearing itself `SUSPECT`/`DEAD` increments incarnation and gossips `ALIVE` (false-positive refute).

**Infection dissemination.** Each membership delta retransmitted ≈ `log₂(n)+c` times via probe piggyback + gossip fanout.

**Complexity.** Happy path `O(n)` msgs/period cluster-wide vs mesh heartbeat `O(n²)`. At n=128, theory ratio >20×.

## Prerequisites

- Python 3.10+ stdlib only (no NumPy)

## How to Run

```bash
terminal(command="swim-gossip-membership selftest", timeout=120)
terminal(command="swim-gossip-membership simulate --scenario happy --n 8", timeout=60)
terminal(command="swim-gossip-membership simulate --scenario single_crash --victim 3", timeout=60)
terminal(command="swim-gossip-membership simulate --scenario multi_crash --n 12", timeout=90)
terminal(command="swim-gossip-membership simulate --scenario lossy --drop 0.1", timeout=90)
terminal(command="swim-gossip-membership simulate --scenario false_suspicion_refute", timeout=60)
terminal(command="swim-gossip-membership simulate --scenario partition --n 10", timeout=60)
terminal(command="swim-gossip-membership chaos --n 12 --drop 0.08", timeout=90)
terminal(command="swim-gossip-membership bench --n 24", timeout=60)
terminal(command="swim-gossip-membership theory --n 128", timeout=15)
terminal(command="swim-gossip-membership compare --n 32", timeout=90)
terminal(command="swim-gossip-membership explain", timeout=15)
```

## Quick Reference

| Command | Purpose |
|---|---|
| `selftest` | CI gate (33 invariant checks) |
| `simulate --scenario happy` | Full-alive, zero false positives |
| `simulate --scenario single_crash` | Detection latency + fraction |
| `simulate --scenario multi_crash` | Concurrent multi-failure |
| `simulate --scenario lossy` | Drop-p stress + still detect |
| `simulate --scenario false_suspicion_refute` | Transient mute → refute, not DEAD |
| `simulate --scenario partition` | Cross-partition suspect/dead |
| `chaos` | Crash + mute + loss combined |
| `bench` | Wall-clock ticks/sec microbench |
| `theory` | O(n) vs O(n²) message ledger |
| `compare` | Theory + empirical happy/crash |
| `explain` | Human protocol walkthrough |

## Procedure

1. Choose cluster size `n` and indirect fanout `k` (default 3).
2. Run `selftest` — completion: exit 0, 33/33.
3. Run `simulate --scenario single_crash` — completion: `ok=true`, `detected_fraction≥0.8`.
4. Run `theory --n 128` — completion: `ratio_mesh_vs_swim > 20`.
5. Optionally `chaos` before trusting latency numbers under loss.

## Pitfalls

- SWIM is weakly consistent: two live nodes may briefly disagree on SUSPECT vs ALIVE.
- High `drop_p` + short `suspect_timeout` raises false positives; prefer refute path.
- Partitions make each side declare the other DEAD — heal requires re-join/ALIVE gossip.
- Not a replacement for consensus; pair with Raft for state machine safety.

## Verification

```bash
terminal(command="swim-gossip-membership selftest", timeout=120)
terminal(command="python3 -m pytest tests/test_swim_gossip_membership.py -q", timeout=180)
```

Completion criteria: selftest exit 0; pytest all passed; theory ratio mesh/SWIM > 5 at n=32.
