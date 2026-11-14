---
name: sarathi-chunked-prefill
description: "Sarathi-Serve: Chunked-Prefill & Stall-Free Piggyback Batching Engine (OSDI 2024) for high-throughput LLM serving with near-zero ITL jitter."
category: inference
tags:
  - sarathi-serve
  - chunked-prefill
  - piggyback-batching
  - osdi2024
  - inference-engine
  - llm-serving
---

# Sarathi-Serve: Chunked-Prefill & Stall-Free Piggyback Batching Engine (OSDI 2024)

Sarathi-Serve addresses the fundamental tension in generative LLM serving between throughput and latency by introducing **Chunked Prefills** and **Piggyback Batching**.

## Theoretical Foundations

In standard iteration-level batching (e.g. Orca, vLLM v1):
1. **Prefill Head-of-Line Blocking**: Long prompts take hundreds of milliseconds to compute, stalling active decodes and causing massive spikes in **Inter-Token Latency (ITL)** and **Time-Between-Tokens (TBT)**.
2. **Resource Underutilization**: Prefill iterations are strictly compute-bound (saturating tensor cores), whereas decode iterations are strictly memory-bandwidth-bound (KV cache lookups). Running them disjointly wastes GPU compute or memory bandwidth.

### Sarathi Innovations
- **Uniform Iteration Budgets**: Prompts are split into fixed chunk sizes (e.g. 512 tokens). Every iteration completes within a predictable compute envelope (e.g., 20–30 ms).
- **Stall-Free Piggybacking**: Decode tokens piggyback onto chunked prefill iterations. Compute units process prefill GEMMs while memory bandwidth loads decode model weights and KV caches in parallel.
- **Roofline Balance**: Dynamically tunes chunk size based on GPU TFLOPS and memory bandwidth to hit the Pareto-optimal operating point.

## CLI Usage

```bash
# Run discrete-event simulation across variable prompt/output lengths
sarathi-chunked-prefill simulate --requests 20 --chunk-size 512

# Compare standard monolithic prefill vs Sarathi piggybacking
sarathi-chunked-prefill compare --requests 20

# Calculate optimal chunk size to meet target ITL latency SLA
sarathi-chunked-prefill plan --prompt-len 4096 --active-decodes 32 --target-itl-ms 25.0
```
