---
name: speculative-decoding-verifier
description: "Draft-target verification engine for speculative sampling in autoregressive LLMs, implementing unbiased token acceptance/rejection and distribution recovery."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [SpeculativeDecoding, LLMInference, LogitVerification, SamplingAlgorithms]
---

# ⚡ Speculative Decoding Verification Engine

Evaluates draft model token candidates against target model logit distributions with mathematically unbiased acceptance/rejection criterion:

$$\alpha = \min\left(1, \frac{p(x)}{q(x)}\right)$$

## CLI Usage
```bash
spec-verify
```
