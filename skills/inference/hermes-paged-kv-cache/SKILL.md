---
name: hermes-paged-kv-cache
description: "Virtual memory paging manager for LLM KV caches. Eliminates memory fragmentation by mapping non-contiguous physical memory blocks to logical token sequences with copy-on-write fork support."
version: 1.0.0
platforms: [macos, linux]
metadata:
  hermes:
    tags: [PagedAttention, KVCache, vLLM, MemoryManagement, Inference]
---

# PagedAttention Virtual Memory & Block-Table Allocator

## Overview
PagedAttention breaks the continuous memory allocation barrier for large language model KV caches, reducing waste from fragmentation to near 0%.

## CLI Interface
```bash
paged-kv-allocator   # Run allocation and benchmark
```
