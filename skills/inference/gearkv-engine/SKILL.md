---
name: gearkv-engine
description: "Exploits attention persistence of importance hypothesis for bounded KV cache pruning in LLMs."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
category: inference
tags: ["kv-cache", "attention-persistence", "llm-optimization", "memory-reduction", "gearkv"]
---

# ✂️ GEAR KV Cache Pruning Engine

An autonomous, high-throughput inference optimization utility that exploits the **Persistence of Importance Hypothesis** in autoregressive Transformers. By identifying pivotal attention tokens over time while retaining essential attention sinks and recent rolling tokens, `gearkv-engine` maintains a strict, bounded memory footprint under massive sequence lengths with negligible loss in generation fidelity.

## Architecture

- **Attention Persistence Scoring**: Continuously tracks exponential moving average (EMA) attention mass to distinguish transient spikes from persistent anchor tokens.
- **Attention Sink & Local Window Retention**: Always preserves first token sinks (initial prompt anchors) and a sliding local attention window for immediate conversational coherence.
- **Bounded-Budget KV Pruning**: Restricts the maximum per-head KV footprint to a configurable budget $C$, pruning non-pivotal intermediate representations.

## CLI Usage

```bash
# 1. Inspect engine status & default hyper-parameters
gearkv-engine status

# 2. Run standard benchmark
gearkv-engine bench --seq-len 4096 --capacity 512 --gen-tokens 16

# 3. High-throughput custom verification
gearkv-engine bench --seq-len 8192 --capacity 256 --heads 16 --dim 64
```

## Verification & Tests

Run the comprehensive test suite:
```bash
pytest tests/test_gearkv_kv_engine.py -v
```
