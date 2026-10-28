---
name: lockfree-concurrency
description: "Lock-Free Concurrency & Chase-Lev Work-Stealing Runtime: Atomic CAS primitives, Michael-Scott non-blocking queues, Treiber stacks, Disruptor ring buffers, Epoch-Based Reclamation (EBR), and Blumofe-Leiserson randomized work-stealing schedulers."
version: 1.0.0
category: systems
tags:
  - Hardcore-Systems
  - Concurrency
  - Lock-Free
  - Work-Stealing
  - Chase-Lev
  - Michael-Scott
  - Disruptor
  - Epoch-Based-Reclamation
  - Linearizability
---

# ⚡ Lock-Free Concurrency & Chase-Lev Work-Stealing Runtime

A production-grade, mathematically rigorous **Lock-Free Concurrency & Work-Stealing Runtime** engineered from first principles. Built without external dependencies, it provides formal non-blocking synchronization primitives, cache-friendly data structures, epoch-based memory reclamation, and an optimal Blumofe-Leiserson work-stealing task scheduler.

---

## 📐 Mathematical Formulation & Systems Architecture

### 1. Progress Guarantee Hierarchy
In distributed and concurrent systems, operations are categorized by formal progress guarantees:

$$\text{Wait-Freedom} \subset \text{Lock-Freedom} \subset \text{Obstruction-Freedom}$$

1. **Wait-Freedom (Herlihy 1991):** Every thread is guaranteed to complete its operation in a bounded number of execution steps regardless of peer thread preemption, scheduling delays, or contention.
2. **Lock-Freedom:** At least one active thread is guaranteed to make global progress in a finite number of steps. While individual threads may experience temporary starvation, the system as a whole never deadlocks or livelocks.
3. **Obstruction-Freedom (Herlihy et al. 2003):** A thread is guaranteed to make progress if all other concurrent threads are suspended or execute without interference.

---

### 2. The ABA Hazard & Memory Reclamation

#### The Classic ABA Problem
Consider memory location $M$ holding pointer $A$:
1. Thread $T_1$ reads $A$ from $M$ and pauses before CAS.
2. Thread $T_2$ pops $A$, frees memory at address $M_A$, allocates object $B$, and deallocates it.
3. Thread $T_2$ allocates new node $C$ at recycled address $M_A$ and stores $A$ back into $M$.
4. Thread $T_1$ wakes up and executes $\text{CAS}(M, A, \text{new})$. CAS evaluates $M == A$ (pointer addresses match) and succeeds, despite the fact that the underlying data structure state and links were mutated.

#### Mitigation 1: Tagged Pointers / Versioned Atomic Reference
Couples the address pointer $P$ with a monotonic 64-bit sequence counter $V$ in a 128-bit double-word atomic container:
$$\text{CAS2}((P_{\text{old}}, V_{\text{old}}) \longrightarrow (P_{\text{new}}, V_{\text{old}} + 1))$$
Since $V$ increments monotonically on every modification, identical pointer reuse is rejected ($V_{\text{current}} \neq V_{\text{old}}$).

#### Mitigation 2: Epoch-Based Reclamation (EBR, Fraser 2004)
Eliminates memory leaks and use-after-free hazards without garbage collection pauses:
- Global epoch $e \in \{0, 1, 2\}$ advances when all active threads have exited epoch $e$.
- Threads publish their active epoch upon entering a critical region: $\text{ThreadEpoch}[i] \leftarrow e_{\text{global}}$.
- Retired nodes are appended to limbo list $L[e_{\text{global}}]$.
- When global epoch increments to $e'$, memory from epoch $(e' + 1) \pmod 3$ is guaranteed to have zero outstanding thread references and is reclaimed safely.

---

### 3. Michael-Scott Non-Blocking FIFO Queue (PODC 1996)

Maintains a sentinel node with atomic `head` and `tail` pointers. Resolves lagging tail pointers via cooperative **helping mechanisms**:

#### Enqueue Algorithm (Linearization Point: CAS of `tail.next`):
1. Read current `tail` and its successor `tail.next`.
2. If `tail.next == NULL`:
   - Attempt $\text{CAS}(\text{tail.next}, \text{NULL}, \text{newNode})$.
   - If successful, cooperatively swing $\text{CAS}(\text{tail}, \text{tail}, \text{newNode})$ and return.
3. If `tail.next != NULL` (lagging tail):
   - Help advance $\text{CAS}(\text{tail}, \text{tail}, \text{tail.next})$ and retry with backoff.

#### Dequeue Algorithm (Linearization Point: CAS of `head`):
1. Read `head`, `tail`, and `head.next`.
2. If `head == tail` and `head.next == NULL`, queue is empty.
3. If `head == tail` and `head.next != NULL`, tail is lagging; help advance tail.
4. Otherwise, extract value from `head.next` and attempt $\text{CAS}(\text{head}, \text{head}, \text{head.next})$.

---

### 4. Chase-Lev Dynamic Work-Stealing Deque (SPAA 2005 / PPoPP 2013)

A circular dynamic array deque optimized for multi-threaded divide-and-conquer parallelism:
- **Bottom End (LIFO):** Exclusively accessed by the owner worker thread via fast-path `push()` and `pop()`. Provides maximum cache locality.
- **Top End (FIFO):** Accessed concurrently by idle thief threads via `steal()`. Snatching tasks from the top extracts the coarsest-grained subproblems from the top of the computation DAG, minimizing total steal volume.

#### Work Stealing Optimality Theorem (Blumofe & Leiserson 1999):
For any computation DAG with total work $T_1$ (sequential execution time) and critical-path length $T_\infty$ (span), a randomized work-stealing scheduler with $P$ processors completes execution in expected time:
$$\mathbb{E}[T_P] \le \frac{T_1}{P} + \mathcal{O}(T_\infty)$$
and incurs at most $\mathcal{O}(P \cdot T_\infty)$ steal operations.

---

## 🛠️ CLI Quickstart & Usage Reference

The CLI is installed globally as `lockfree-rt`:

### 1. Run Formal Linearizability & Concurrency Verification
```bash
lockfree-rt verify
```
Executes high-contention stress tests across 8 threads, validating Michael-Scott queues, Treiber stacks, ABA tagged pointers, Chase-Lev deques, and Epoch-Based Reclamation.

### 2. Multi-Threaded Contention Micro-Benchmark
```bash
lockfree-rt benchmark
```
Compares throughput (ops/sec), latency, and lock contention between coarse-grained mutex locking and non-blocking lock-free implementations.

### 3. Parallel Work-Stealing Task Scheduler
```bash
lockfree-rt schedule --workers 4 --n 24
```
Executes parallel divide-and-conquer computation, displaying task distribution, steal efficiency, and load imbalance Gini coefficient.

### 4. Live Interactive Terminal Visualizer
```bash
lockfree-rt simulate --duration 6 --workers 4
```
Streams an ANSI dashboard showing real-time worker states, queue depths, dynamic steal affinity matrix, and load balancing telemetry.

### 5. Architectural & Mathematical Theory
```bash
lockfree-rt info
```

---

## 🔬 Production Verification Suite

- **Zero External Dependencies:** Implemented in pure Python using hardware memory models and atomic abstractions.
- **Deterministic Linearizability:** Rigorously checked under heavy multi-threaded interleaving with zero lost updates or race conditions.
- **Gini Imbalance Metric:** Real-time computation of Lorenz curve and Gini coefficient ($0.000 =$ perfectly balanced) to verify dynamic load convergence.
