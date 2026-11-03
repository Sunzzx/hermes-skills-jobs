---
name: ring-attention
description: "RingAttention & Blockwise Parallel Context Engine: infinite-context transformer scaling, overlapping P2P ring communication, blockwise online softmax, and machine-epsilon equivalence."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags:
      - ring-attention
      - sequence-parallelism
      - infinite-context
      - blockwise-transformers
      - online-softmax
      - distributed-training
      - p2p-communication
---

# RingAttention & Blockwise Parallel Context Engine

A mathematically rigorous, production-grade implementation of RingAttention (Liu et al., 2023; Hao et al., 2023) for infinite-context transformer training and inference. Overcomes standard quadratic $\mathcal{O}(S^2)$ memory limits by distributing sequence tokens across a ring of $P$ workers, overlapping blockwise online softmax computations with peer-to-peer tensor communication, and maintaining machine-epsilon mathematical equivalence to monolithic multi-head self-attention.

## Theoretical Foundations

Standard transformer self-attention calculates an $S \times S$ attention matrix between all queries and keys. For sequence length $S$:
$$\text{Memory}_{\text{Monolithic}} = \mathcal{O}(S^2)$$
At $S = 1,000,000$ tokens, storing attention scores requires over 8 Terabytes per attention head in FP64/FP32, causing immediate Out-Of-Memory (OOM) failures on any single accelerator.

### RingAttention Architecture

RingAttention partitions the sequence length $S$ into $P$ blocks of length $B = S / P$:
- Worker $p \in \{0, \dots, P-1\}$ stores Query block $Q_p$, Key block $K_p$, and Value block $V_p$.
- In each step $s \in \{0, \dots, P-1\}$, worker $p$ computes attention between its local Query $Q_p$ and the current Key-Value block $(K_j, V_j)$ where $j = (p - s) \bmod P$.
- Concurrently, worker $p$ transmits $(K_j, V_j)$ to worker $(p + 1) \bmod P$ and receives $(K_{j-1}, V_{j-1})$ from $(p - 1) \bmod P$ across the circular ring.

$$\text{Memory}_{\text{RingAttention}} = \mathcal{O}\left(\frac{S}{P} \cdot d\right)$$

### Online Softmax Recurrence

To compute exact softmax attention without materializing the full score matrix, each worker maintains running state $(m_i, l_i, O_i)$ across ring steps for row $i$:
$$m_{new} = \max(m_{old}, \text{rowmax}(S_{ij}))$$
$$\alpha = \exp(m_{old} - m_{new})$$
$$P_{ij} = \exp(S_{ij} - m_{new})$$
$$l_{new} = \alpha \cdot l_{old} + \sum_c P_{ij}[c]$$
$$O_{new} = \alpha \cdot O_{old} + P_{ij} V_j$$

At the conclusion of $P$ ring steps, the exact normalized output is:
$$O_{final} = \frac{O_{new}}{l_{new}}$$

### Causal Triangular Schedule

In autoregressive language models, tokens cannot attend to future tokens. RingAttention prunes all blocks where $j > p$ without performing compute or transmitting redundant gradients:
- **$j < p$**: Fully computed dense attention block.
- **$j == p$**: Diagonal block with local causal triangular mask applied ($c > r \to -\infty$).
- **$j > p$**: Safely skipped future block (zero FLOPs, zero state update).

## Communication Schedule Matrix

```
Rank  │ Step 0    │ Step 1    │ Step 2    │ Step 3
──────┼───────────┼───────────┼───────────┼───────────
p = 0 │ KV_0 [◿]  │  [SKIP]   │  [SKIP]   │  [SKIP]
p = 1 │ KV_1 [◿]  │ KV_0 [■]  │  [SKIP]   │  [SKIP]
p = 2 │ KV_2 [◿]  │ KV_1 [■]  │ KV_0 [■]  │  [SKIP]
p = 3 │ KV_3 [◿]  │ KV_2 [■]  │ KV_1 [■]  │ KV_0 [■]
```

## CLI Reference

```bash
# Verify forward output and backward gradients vs monolithic reference
ring-attention verify --ranks 4 --seq-len 128 --hidden 32 --heads 4

# Verify bidirectional (encoder / non-causal) mode
ring-attention verify --ranks 4 --seq-len 128 --hidden 32 --heads 4 --no-causal

# Visualize step-by-step ring communication schedule & triangular pruning
ring-attention simulate --ranks 4

# Benchmark execution throughput and memory scaling
ring-attention benchmark --ranks 4 --hidden 32 --heads 4 --steps 3

# Display mathematical derivations and architecture specification
ring-attention inspect
```

## Python SDK Example

```python
import numpy as np
from importlib.machinery import SourceFileLoader
from pathlib import Path

# Load RingAttention module
CLI = Path.home() / ".local" / "bin" / "ring-attention"
ring_mod = SourceFileLoader("ring_attention", str(CLI)).load_module()

# Initialize MultiHeadRingAttention with 4 ranks across 1024 tokens
hidden_dim = 64
num_heads = 4
num_ranks = 4
seq_len = 1024
block_len = seq_len // num_ranks

model = ring_mod.MultiHeadRingAttention(
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_ranks=num_ranks,
    causal=True,
)

# Shard input sequence across ranks
X_full = np.random.randn(seq_len, hidden_dim)
X_shards = [X_full[p * block_len : (p + 1) * block_len] for p in range(num_ranks)]

# Forward pass
Y_shards, cache = model.forward(X_shards)
print(f"Rank 0 Output Shape: {Y_shards[0].shape}")

# Backward pass with incoming loss gradients
grad_Y_shards = [np.ones_like(y) for y in Y_shards]
grad_X_shards = model.backward(grad_Y_shards, cache)
print(f"Rank 0 Input Gradient Shape: {grad_X_shards[0].shape}")
```

## Key References
- Liu et al. (2023): *Ring Attention with Blockwise Transformers for Near-Infinite Context* (UC Berkeley)
- Hao et al. (2023): *Scaling Large Language Model Context to Millions of Tokens with RingAttention*
- Dao et al. (2022, 2023): *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
