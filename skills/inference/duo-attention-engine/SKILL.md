---
name: duo-attention-engine
description: DuoAttention Dual-Path Retrieval and Streaming Head Specialization Engine
category: inference
version: 1.0.0
tags: [llm, inference, kv-cache, duo-attention, head-specialization, streaming-attention, long-context, memory-efficiency]
cli: duo-attention-engine
---

# DuoAttention: Dual-Path Head Specialization KV Cache Engine

## Overview

**DuoAttention** eliminates up to 75-85% of KV cache memory footprints and memory bandwidth bottlenecks during long-context LLM inference without fine-tuning or perplexity degradation. 

DuoAttention leverages the key empirical finding that **only a small subset of attention heads (~20-25%) are "Retrieval Heads"** requiring full context history. The remaining ~75-80% are **"Streaming Heads"**, whose attention is strictly concentrated on **attention sinks** and a compact **rolling local context window**.

### Core Capabilities:
1. **Dual-Path Head Partitioning**: Distinguishes Retrieval Heads from Streaming Heads across all transformer layers.
2. **Retrieval Head Full-Context Preservation**: Guarantees full key-value memory for heads responsible for long-distance factual recall and needle-in-a-haystack retrieval.
3. **Streaming Head Sink & Sliding Window**: Restricts streaming heads to constant-sized initial attention sinks ($N=4$) and local sliding windows ($W=128$), bounding their memory footprint to $O(1)$.
4. **Linear Memory & Bandwidth Scaling**: Decouples 75%+ of the KV cache from sequence length, delivering 3x-6x decoding acceleration on extreme context lengths (up to 32k-128k).

---

## CLI Usage

```bash
# Inspect engine architecture and head partitioning status
duo-attention-engine status

# Run memory reduction and latency benchmark
duo-attention-engine benchmark --seq-len 8192 --gen-tokens 32

# Simulate step-by-step decoding with live savings updates
duo-attention-engine simulate --seq-len 4096 --steps 10
```

---

## Python API

```python
from importlib.machinery import SourceFileLoader
import os

# Load engine directly
bin_path = os.path.expanduser("~/.local/bin/duo-attention-engine")
duo_mod = SourceFileLoader("duo_attention_engine", bin_path).load_module()

config = duo_mod.DuoAttentionConfig(
    num_heads=32,
    head_dim=128,
    retrieval_ratio=0.25,
    sink_size=4,
    recent_size=128
)
engine = duo_mod.DuoAttentionEngine(config)

# Prefill prompt
stats = engine.prefill(seq_len=8192)
print(f"Memory Savings: {stats['memory_savings_pct']}% | Compression: {stats['compression_ratio']}x")

# Autoregressive generation step
outputs, step_stats = engine.decode_step(step_idx=0)
```
