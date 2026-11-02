---
name: blink-tree
description: "Lehman-Yao B-Link Tree & High-Concurrency Latch-Free Storage Engine: high-key invariants, horizontal right-link pointers, lock-free search traversal, atomic two-phase node splits, and WAL durability."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [DistributedSystems, StorageEngines, LehmanYao, BLinkTree, Concurrency, DatabaseInternals]
---

# Lehman-Yao B-Link Tree Storage Engine

## Overview
A mathematically rigorous, production-grade implementation of the landmark Lehman-Yao B-Link Tree algorithm (*Lehman & Yao, ACM TODS 1981: "Efficient Locking for Concurrent Operations on B-Trees"*).

The Lehman-Yao architecture eliminates the primary concurrency bottleneck in high-throughput database storage engines: lock-coupling (crabbing) during tree traversal. By augmenting standard B+ trees with **high-key invariants** and **horizontal right-link pointers**, readers can traverse down and across the tree completely latch-free, navigating through concurrent node splits without blocking or restarting from root.

## Architectural Foundations

### 1. High-Key Invariant
Every node (both internal routing nodes and leaf data nodes) maintains an explicit `high_key` representing the supremum of all keys in its subtree:
$$\forall k \in \text{subtree}(u), \quad k \le u.\text{high\_key}$$
The rightmost node at any level has $u.\text{high\_key} = +\infty$.

### 2. Horizontal Right-Link Pointers
Every node $u$ contains a horizontal pointer $u.\text{link\_ptr}$ to its immediate right sibling at the same level:
$$u.\text{link\_ptr} = \text{sibling}_{\text{right}}(u)$$
When a node splits due to insertion overflow, the new right half is spliced into the linked list at the same level before the parent pointer is installed.

### 3. Latch-Free Search & Concurrent Split Traversal
A reader traversing node $u$ looking for key $k$:
- Checks if $k > u.\text{high\_key}$.
- If true, a concurrent split has relocated $k$ into a newly spawned right sibling. The reader immediately moves right via $u.\text{link\_ptr}$ at the same level.
- Readers acquire **zero read latches**, completely eliminating lock contention during search and range scans.

### 4. Atomic Two-Phase Split Protocol
1. Allocate right sibling $v$.
2. Migrate upper half of keys and values/children to $v$.
3. Set $v.\text{high\_key} = u.\text{high\_key}$ and $v.\text{link\_ptr} = u.\text{link\_ptr}$.
4. Update $u.\text{high\_key} = \text{split\_key}$ and $u.\text{link\_ptr} = v$.
5. Release node lock and propagate $(\text{split\_key}, v)$ up to parent asynchronously.

### 5. Write-Ahead Log (WAL) Durability
- Monotonic Log Sequence Numbers (LSN).
- 32-bit CRC polynomials for frame validation.
- Deterministic crash recovery replay on restart.

---

## Global CLI: `blink-tree`

The engine provides an interactive command-line interface and diagnostic inspector:

```bash
# Display interactive ASCII tree visualization & horizontal split demonstration
blink-tree demo

# Execute comprehensive 6-stage verification suite
blink-tree verify

# Run multi-threaded concurrency benchmark (e.g. 8 threads, 10,000 ops, 85% reads)
blink-tree benchmark --threads 8 --ops 10000 --read-ratio 0.85

# Persistent Key-Value Store operations with WAL
blink-tree --db ~/.blink_data put user:1001 '{"name":"Ayu","role":"Architect"}'
blink-tree --db ~/.blink_data get user:1001
blink-tree --db ~/.blink_data scan --start user:1000 --end user:1010
blink-tree --db ~/.blink_data tree
blink-tree --db ~/.blink_data checkpoint
```

---

## Python API Integration

```python
from blink_tree import BLinkTree, BLinkEngine

# In-memory high-concurrency tree
tree = BLinkTree(fanout=4)
tree.put("txn_001", {"amount": 250.0, "status": "committed"})
tree.put("txn_002", {"amount": 890.5, "status": "pending"})

# Latch-free point query
val = tree.get("txn_001")

# Direct leaf range scan across link pointers
results = tree.scan(start_key="txn_000", end_key="txn_999")

# Persistent engine with WAL
engine = BLinkEngine(data_dir="./db_data", fanout=4)
engine.put("metric:cpu", 42.1)
engine.checkpoint()
engine.close()
```
