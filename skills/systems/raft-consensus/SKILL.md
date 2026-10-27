---
name: raft-consensus
description: "Distributed Raft Consensus Protocol engine: leader election, log replication, commit safety invariants, split-brain chaos testing, and linearizable KV store."
version: 1.0.0
category: systems
tags:
  - DistributedSystems
  - Raft
  - Consensus
  - FaultTolerance
  - StateMachines
  - FormalVerification
---

# ⚡ Distributed Raft Consensus Engine

A rigorous, production-grade implementation of the **Raft Distributed Consensus Protocol** (*Ongaro & Ousterhout, Stanford University 2014*) written from scratch with zero external dependencies.

Includes discrete-event async network simulation, network partition chaos testing, monotonic term management, election timers, and formal invariant validation.

---

## 📐 Theoretical Foundations & Core Invariants

Raft guarantees safety and linearizability under all non-Byzantine network conditions (delays, partitions, packet drops, packet reordering):

### 1. The 5 Fundamental Raft Safety Invariants
1. **Election Safety (§5.2)**: At most one leader can be elected in any given term.
2. **Leader Append-Only (§5.3)**: A leader never overwrites or truncates its log entries; it only appends new entries.
3. **Log Matching Invariant (§5.3)**: If two logs contain an entry with the same index and term, then the logs are identical in all entries up through the given index.
4. **Leader Completeness (§5.4.1)**: If a log entry is committed in a given term, then that entry will be present in the logs of the leaders for all higher-numbered terms.
5. **State Machine Safety (§5.4.3)**: If a server has applied a log entry at a given index to its state machine, no other server will ever apply a different log entry for the same index.

### 2. RPC Protocols

#### RequestVote RPC (§5.2, §5.4.1)
Invoked by candidates to gather votes during elections.
$$\text{RequestVoteArgs}: (\text{term}, \text{candidateId}, \text{lastLogIndex}, \text{lastLogTerm})$$
$$\text{RequestVoteReply}: (\text{term}, \text{voteGranted})$$

**Voting Criterion**: A voter grants its vote if and only if:
1. $\text{args.term} \ge \text{currentTerm}$
2. Voter has not voted for another candidate in this term (`votedFor` is None or `candidateId`)
3. Candidate log is at least as up-to-date as receiver log:
   $$\text{lastLogTerm}_c > \text{lastLogTerm}_r \quad \lor \quad (\text{lastLogTerm}_c = \text{lastLogTerm}_r \land \text{lastLogIndex}_c \ge \text{lastLogIndex}_r)$$

#### AppendEntries RPC (§5.3)
Invoked by leader to replicate log entries and serve as periodic heartbeats.
$$\text{AppendEntriesArgs}: (\text{term}, \text{leaderId}, \text{prevLogIndex}, \text{prevLogTerm}, \text{entries}, \text{leaderCommit})$$
$$\text{AppendEntriesReply}: (\text{term}, \text{success}, \text{matchIndex}, \text{conflictIndex})$$

**Commit Rule**: Leader advances `commitIndex` to $N$ if:
$$N > \text{commitIndex} \land \sum_{i} \mathbb{I}(\text{matchIndex}[i] \ge N) > \frac{n}{2} \land \text{log}[N].\text{term} = \text{currentTerm}$$

---

## 🛠️ CLI Quickstart

The tool is accessible globally as `raft-consensus`:

### 1. Run Formal Invariant & Safety Verification Suite
```bash
raft-consensus verify
```
Validates 5 rigorous distributed test cases:
- **[1/5] Election Safety**: Proof of single leader election per term.
- **[2/5] Log Matching**: Exact entry consistency & linearizable state machine hashes.
- **[3/5] Leader Crash & Failover**: Active leader crash, follower timeout, and higher-term election.
- **[4/5] Split-Brain Quorum Safety**: Strict partition into minority `{n1, n2}` and majority `{n3, n4, n5}`. Proves minority rejects uncommitted writes while majority retains progress.
- **[5/5] Partition Healing & Convergence**: Resolves network split, steps down stale leaders, reconciles logs, and verifies identical state hash across all 5 nodes.

### 2. Run Interactive Live Cluster Demo
```bash
raft-consensus demo
```
Launches a 5-node cluster, tracks election phases, submits state machine transactions (`SET`, `INCR`, `CAS`), replicates logs, and audits cluster-wide state convergence.

### 3. Run Split-Brain Network Partition Chaos Test
```bash
raft-consensus chaos
```
Simulates dynamic network failure scenarios, auditing cluster behavior under split-brain and post-healing convergence.

### 4. Run Consensus Throughput Benchmark
```bash
raft-consensus benchmark --proposals 500
```
Measures proposal commit rate, packet routing, and replication latency.

### 5. Inspect Formal Specifications
```bash
raft-consensus info
```
