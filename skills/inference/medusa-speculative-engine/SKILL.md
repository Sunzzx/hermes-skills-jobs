---
name: medusa-speculative-engine
description: "Medusa-2 multi-head self-speculative decoding engine with parallel draft heads, tree attention verification, and 2.5x-3.5x lossless inference acceleration."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [LLM, Inference, SpeculativeDecoding, Medusa, TreeAttention, Acceleration]
---

# Medusa-2 Self-Speculative Multi-Head Tree Decoding Engine

## Overview
Medusa-2 is an ultra-efficient speculative decoding architecture that accelerates LLM inference without requiring an auxiliary draft model. By augmenting the base transformer with lightweight multi-layer perceptron (MLP) draft heads atop the final hidden states, Medusa concurrently predicts multiple forward tokens ($t+1, t+2, \dots, t+K$).

These multi-head candidate predictions are assembled into a candidate draft tree and verified in a single forward pass of the original model via Tree Attention masking, yielding a 2.5x to 3.5x lossless speedup over autoregressive decoding.

## Theoretical Architecture

### 1. Multi-Head Parallel Drafting
Instead of sequential drafting, Medusa branches $K$ heads off the frozen base model's representations:
$$\mathbf{H}_t = \text{TransformerBlock}_{L}(\mathbf{X}_{1:t})$$
$$\mathbf{p}_{t+k} = \text{Softmax}(\text{MedusaHead}_k(\mathbf{H}_t)), \quad k \in \{1, 2, \dots, K\}$$

### 2. Speculative Tree Construction & Tree Attention
Given top-$p$ candidates across draft heads, Medusa forms a tree DAG of candidate paths. The causal attention mask is extended into an ancestor-directed tree attention mask:
$$\mathbf{M}_{i, j} = \begin{cases} 1, & \text{if node } j \text{ is an ancestor of node } i \text{ or } j = i \\ 0, & \text{otherwise} \end{cases}$$

This allows the base model to verify dozens of candidate sequences in parallel with less than 15% attention computational overhead.

### 3. Verification & Acceptance Rate
During tree verification, the base model evaluates candidate branches simultaneously. The longest valid sequence of matching tokens is accepted alongside one ground-truth speculative bonus token, guaranteeing mathematically lossless sample equivalence with greedy or nucleus decoding.

## CLI Usage & Commands

### Run Self-Test & Validation
```bash
medusa-speculative-engine --test --json
```

### Run Multi-Step Speedup & Acceptance Benchmark
```bash
medusa-speculative-engine --benchmark --heads 4 --topk 3 --steps 200
```

### Standalone Python Integration
```python
from medusa_speculative_engine import MedusaConfig, MedusaEngine

cfg = MedusaConfig(vocab_size=32000, num_heads=4, topk=3)
engine = MedusaEngine(cfg)
result = engine.simulate_step(root_token=42)
print(f"Accepted: {result['accepted_count']} tokens | Speedup: {result['speedup_factor']}x")
```
