---
name: disaggregated-prefill-decode
description: Disaggregated Prefill-Decode serving architecture with chunked prefill scheduling and zero-copy RDMA KV transfer.
tags:
  - systems
  - llm-serving
  - inference-optimization
  - disaggregated-serving
  - vllm
  - mooncake
---

# Disaggregated Prefill-Decode (Chunked Prefill & KV Transfer) Engine

## 1. Problem: The Unified Serving Interference Bottleneck

In standard unified Large Language Model (LLM) serving architectures (e.g. baseline vLLM, HuggingFace TGI), both the **Prefill** (prompt evaluation) phase and the **Decode** (autoregressive token generation) phase are executed on identical GPU compute nodes:

1. **Computational Divergence**:
   - **Prefill Phase**: Matrix-Matrix multiplication (GEMM) with high arithmetic intensity, completely saturating Tensor Cores and compute bandwidth.
   - **Decode Phase**: Matrix-Vector multiplication (GEMV) with low arithmetic intensity, strictly bounded by High Bandwidth Memory (HBM) bandwidth.
2. **Head-of-Line Blocking & TPOT Jitter**:
   - In unified clusters, when a long prompt request (e.g., 8k-32k tokens) arrives, it claims compute engines for multiple hundred milliseconds.
   - Concurrently running autoregressive decode streams suffer extreme Time-Per-Output-Token (TPOT) spikes (often 5x to 10x SLA degradation).
3. **Decoupled Resource Needs**:
   - Prefill scaling requires compute-dense accelerators (e.g., H100 SXM5 / B200).
   - Decode scaling requires high memory capacity and memory-bandwidth-dense instances.

---

## 2. Core Architecture: Disaggregated Serving (Splitwise / Mooncake / DistServe)

Disaggregated serving physically segregates prefill and decode execution into dedicated node pools connected by a high-throughput network fabric (RDMA / InfiniBand / PCIe Gen5):

```
                    [ Client Requests (Prompts) ]
                                 │
                                 ▼
                ┌─────────────────────────────────┐
                │  Global Distributed Dispatcher   │
                └──────────────┬──────────────────┘
                               │
               Chunked Prompts │ (Parallel GEMM)
                               ▼
        ┌─────────────────────────────────────────────────┐
        │             PREFILL POOL (Worker 0..N)          │
        │  • High-throughput Chunked Prefill (GEMM)       │
        │  • Bounds burst execution duration              │
        │  • Generates paged KV cache tensors             │
        └──────────────────────┬──────────────────────────┘
                               │
             Zero-Copy RDMA    │ High-Speed Network
             KV-Cache Transfer │ (InfiniBand / RoCE v2)
                               ▼
        ┌─────────────────────────────────────────────────┐
        │             DECODE POOL (Worker 0..M)           │
        │  • Ingests paged physical block tables          │
        │  • Low-latency memory-bandwidth GEMV execution  │
        │  • Strict, jitter-free TPOT SLA preservation    │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼ Tokens
                   [ Streamed Responses ]
```

### Key Operational Principles

1. **Chunked Prefill Scheduling (Sarathi-Serve Paradigm)**:
   - Long input sequences of length $L$ are chunked into uniform slices of size $C$ (e.g., 256 or 512 tokens):
     $$\text{Chunks} = \left\lceil \frac{L}{C} \right\rceil$$
   - Bounds maximum prefill compute burst times, ensuring stable execution scheduling.

2. **Paged Physical KV Block Tables**:
   - Key and Value tensors are partitioned into fixed block pages (e.g., 16 tokens/block).
   - Virtual request positions are mapped dynamically to physical memory pages, eliminating external memory fragmentation.

3. **Zero-Copy RDMA Migration**:
   - As soon as the final prompt chunk is processed by the prefill node, the logical block table and physical pages are pushed directly to the target decode node's HBM via RDMA without CPU staging.

---

## 3. CLI & Execution Commands

Hermes provides a production-grade Python/NumPy CLI engine located at `bin/disaggregated-pd-engine`:

```bash
# 1. Inspect architectural status & invariants
./bin/disaggregated-pd-engine status

# 2. Run mathematical & architectural verification test suite
./bin/disaggregated-pd-engine verify

# 3. Compare latency & jitter profiles (Unified vs Disaggregated)
./bin/disaggregated-pd-engine compare

# 4. Benchmark chunked prefill, RDMA transfer, and decode TPOT
./bin/disaggregated-pd-engine benchmark --prompt-len 2048 --chunk-size 512 --decode-steps 10
```

---

## 4. Benchmark Performance Summary

| Metric | Unified Serving | Disaggregated Serving |
| :--- | :--- | :--- |
| **Prefill Interference** | Severe (5x-10x TPOT spikes) | **Zero (Compute decoupled from BW)** |
| **TPOT Predictability** | High variance, long-tail P99 | **Strict deterministic P99 bounds** |
| **Resource Scaling** | Monolithic GPU allocation | **Independent Prefill/Decode scaling** |
| **KV Cache Mobility** | Bound to local GPU memory | **Fast RDMA paged block transfer** |
