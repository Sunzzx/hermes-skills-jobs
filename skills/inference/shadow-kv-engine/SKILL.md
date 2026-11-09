---
name: shadow-kv-engine
description: ShadowKV High-Throughput Long-Context LLM Inference Engine via Offloaded Shadow Summarization
category: inference
version: 1.0.0
tags: [llm, inference, kv-cache, offloading, memory-efficiency, long-context]
---

# ShadowKV: High-Throughput Long-Context LLM Inference Engine

ShadowKV addresses KV cache memory bottlenecks in extreme long-context LLM inference by maintaining a low-rank compact "shadow" sketch in high-speed GPU memory while offloading high-precision full KV tensors to host DRAM or NVMe storage.

## Key Capabilities
- **Hierarchical Storage Offloading:** Stores voluminous full-precision key-value tensors in host memory / NVMe.
- **Low-Rank Shadow Sketches:** Fast VRAM tracks compact sketches per chunk to evaluate global attention logits without full transfer.
- **Dynamic Top-k Chunk Retrieval:** Decode steps retrieve only the top-k essential chunks on-demand.
- **Sink & Sliding Window Preservation:** Always retains initial attention sinks and recent local context in fast memory.

## CLI Usage
```bash
# Check status and architecture features
shadow-kv-engine status

# Run performance benchmark with long sequence context
shadow-kv-engine benchmark --seq-len 8192 --gen-tokens 16
```

## Python API
```python
from bin.shadow_kv_engine import ShadowKVConfig, ShadowKVEngine

config = ShadowKVConfig(
    num_heads=32,
    head_dim=128,
    chunk_size=64,
    shadow_dim=16,
    topk_chunks=8,
    recent_tokens=128
)
engine = ShadowKVEngine(config)
prefill_stats = engine.prefill(seq_len=8192)
embed, step_stats = engine.decode_step(step_idx=0)
```
