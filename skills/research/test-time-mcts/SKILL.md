---
name: test-time-mcts
description: "Test-Time Compute MCTS & Process Reward Model (PRM) reasoning engine with PUCT exploration, step verification, and ANSI tree visualization."
version: 1.0.0
category: research
tags:
  - Frontier-AI
  - Test-Time-Compute
  - MCTS
  - PRM
  - Reasoning
  - Search
  - PUCT
---

# 🧠 Test-Time Compute MCTS & Process Reward Model (PRM) Engine

A rigorous, zero-dependency implementation of **Test-Time Search** inspired by frontier reasoning paradigms (OpenAI o1/o3, DeepSeek-R1). Features **Monte Carlo Tree Search (MCTS)** with step-level **Process Reward Models (PRMs)**, **Polynomial Upper Confidence Trees (PUCT)** exploration, and self-correcting backtracking.

---

## 📐 Mathematical Formulation

### 1. Test-Time Compute Scaling Law
Reasoning performance scales smoothly with test-time inference compute budget $C$ according to an exponential return curve:
$$\mathbb{P}(\text{Success}) = 1 - \exp(-k \cdot C^\alpha)$$
where $C$ represents simulation budget/rollouts, and $\alpha$ represents search efficiency.

### 2. Polynomial Upper Confidence Trees (PUCT)
During tree traversal, each node selects child action $a^*$ maximizing PUCT value:
$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
where:
- $Q(s, a) = \frac{W(s, a)}{N(s, a)}$ is the expected step-supervised value.
- $P(s, a)$ is the prior probability from the step policy.
- $N(s, a)$ is the visit count.
- $c_{\text{puct}}$ controls the exploration-exploitation balance.

### 3. Step-Level Process Supervision (PRM vs ORM)
- **Outcome Supervision (ORM)** evaluates only final answer correctness $r \in \{0, 1\}$, suffering from severe credit assignment problems in multi-step deductions ($L \ge 10$).
- **Process Supervision (PRM)** scores each individual step $r_t = \text{PRM}(s_t, a_t) \in [0, 1]$.

### 4. Backtracking & Branch Pruning
When the PRM confidence drops below a threshold $\tau_{\text{backtrack}}$:
$$\text{PRM}(s_t, a_t) < \tau_{\text{backtrack}} \implies \text{Prune branch and trigger early self-correction}$$
This eliminates combinatorial search explosions over flawed reasoning prefixes.

---

## 🛠️ CLI Quickstart

The CLI is installed globally as `test-time-mcts`:

### 1. Run Verification Test Suite
```bash
test-time-mcts verify
```
Validates PUCT formula, Game of 24 fractional solver, PRM fallacy detection, and comparative advantage over beam search.

### 2. Solve Problems with ANSI Search Tree
```bash
# Countdown / Game of 24 (Fractional reasoning)
test-time-mcts solve --problem game24 --numbers 3,8,8,3 --tree

# Olympiad Chinese Remainder Theorem system
test-time-mcts solve --problem olympiad

# Cryptarithmetic constraint deduction (SEND + MORE = MONEY)
test-time-mcts solve --problem cryptarithm
```

### 3. Compare Search Paradigms (MCTS vs Beam vs Greedy)
```bash
test-time-mcts compare --numbers 3,8,8,3
```

### 4. Run Benchmark Suite
```bash
test-time-mcts benchmark
```

### 5. Architectural & Theoretical Reference
```bash
test-time-mcts info
```
