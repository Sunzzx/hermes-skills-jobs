---
name: crdt-sync
description: "Conflict-Free Replicated Data Types (CRDT) & Causal Vector Clock Engine for real-time collaborative text editing, state-based counters, sets, and partition-tolerant synchronization."
category: systems
tags: ["systems", "distributed-systems", "crdt", "vector-clocks", "eventual-consistency", "collaborative-editing", "rga", "consensus"]
cli: crdt-sync
---

# ⚡ Conflict-Free Replicated Data Types (CRDT) & Causal Vector Clock Engine

A mathematically rigorous, production-grade distributed state synchronization engine implementing State-based (CvRDT) and Operation-based (CmRDT) structures, Causal Vector Clocks, Replicated Growable Array (RGA) for collaborative text editing, Observed-Removed Sets (ORSet), PN-Counters, and a Chaos Network Partition Simulator verifying Strong Eventual Consistency (SEC).

---

## 🧠 Theoretical Foundations

### 1. Strong Eventual Consistency (SEC) & Semilattices
Under the CAP and PACELC theorems, systems prioritizing availability and partition tolerance cannot rely on synchronous consensus (such as Raft or Paxos) for high-frequency interactive mutations. Conflict-Free Replicated Data Types (CRDTs) provide **Strong Eventual Consistency (SEC)**: any two replicas that have received the same set of updates are guaranteed to be in the exact same state, irrespective of the order in which updates arrived or network latency.

Mathematically, a State-based CRDT (CvRDT) forms a **bounded join-semilattice** $(\mathcal{S}, \sqcup, \le)$ where:
1. **Partial Order ($\le$):** Captures information subsumption ($a \le b$ denotes that state $b$ subsumes state $a$).
2. **Join Operator ($\sqcup$):** Computes the least upper bound (supremum), satisfying:
   - **Commutativity:** $a \sqcup b = b \sqcup a$
   - **Associativity:** $(a \sqcup b) \sqcup c = a \sqcup (b \sqcup c)$
   - **Idempotence:** $a \sqcup a = a$
3. **Monotonic Growth:** $a \le a \sqcup b$ for all states $a, b \in \mathcal{S}$.

### 2. Causal Vector Clocks
Every actor maintains a logical vector $\mathbf{V} \in \mathbb{N}^k$. For two events with vector clocks $V_1, V_2$:
$$\begin{aligned}
V_1 \le V_2 &\iff \forall k: V_1[k] \le V_2[k] \quad (V_1 \text{ causally precedes } V_2) \\n\text{Concurrent}(V_1, V_2) &\iff \neg(V_1 \le V_2) \land \neg(V_2 \le V_1) \quad (V_1 \parallel V_2)
\end{aligned}$$

When merging states from replica $R$ with clock $V_R$ into local clock $V_L$:
$$V_{\text{merged}}[k] = \max(V_L[k], V_R[k])$$

### 3. Replicated Growable Array (RGA) Sequence CRDT
Used in real-time collaborative editors (Figma, Notion, Google Docs style P2P systems). Each inserted atom is assigned an immutable global identifier:
$$\text{AtomId} = (\text{seq}, \text{node\_id})$$

When two peers concurrently insert atoms $A_1, A_2$ after the same causal origin $L$, deterministic linearization is guaranteed by sorting concurrent siblings by descending priority $(\text{seq}, \text{node\_id})$. Multi-hop descendants are preserved in causal subtrees, ensuring universal convergence across network partitions.

---

## 💻 Global CLI Usage (`crdt-sync`)

### 1. Formal Mathematical Invariant Verification Suite
Verify join-semilattice properties, causal vector clock relations, PN-counter monotonicity, and 50-step randomized split-brain network fuzzing:
```bash
crdt-sync test
```

### 2. Interactive Multi-Peer Collaborative Simulation
Simulate real-time multi-peer collaborative editing with keystroke interleaving, causal vector clock snapshots, and conflict reconciliation:
```bash
crdt-sync sim --peers 3 --steps 5
```

### 3. Chaos Network Partition Simulation
Simulate split-brain network partitions with simulated packet drops, reordering, and post-healing universal gossip convergence:
```bash
crdt-sync chaos --peers 5
```

### 4. High-Throughput Microbenchmark
Benchmark mutation throughput and full mesh N-way state convergence:
```bash
crdt-sync bench --ops 2000 --peers 4
```

---

## 🐍 Python API Integration

```python
from crdt_sync import RGAText, PNCounter, ORSet, VectorClock, CollaborativeDoc

# 1. Collaborative Document with Mixed CRDT Types
doc1 = CollaborativeDoc(doc_id="spec-101", node_id="clientA")
doc2 = CollaborativeDoc(doc_id="spec-101", node_id="clientB")

# 2. Concurrent Mutations
doc1.get_text("notes").insert_string(0, "Hermes Autonomous OS")
doc2.get_text("notes").insert_string(0, "✨ ")
doc1.get_counter("views").inc("clientA", 10)
doc2.get_counter("views").inc("clientB", 5)

# 3. Distributed Peer Synchronization (Join-Semilattice Merge)
doc1.merge(doc2)
doc2.merge(doc1)

# Both replicas hold identical state
assert doc1.snapshot() == doc2.snapshot()
print(doc1.snapshot())
```
