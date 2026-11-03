---
name: multi-token-prediction
description: "DeepSeek-V3 Multi-Token Prediction (MTP) architecture: sequential MTP modules, dual representation fusion, shared unembedding projection, and zero-overhead speculative drafting."
version: "1.0.0"
category: systems
tags:
  - deepseek-v3
  - multi-token-prediction
  - mtp
  - speculative-decoding
  - transformer-inference
  - acceleration
cli: mtp-engine
---

# Multi-Token Prediction (MTP) & Speculative Drafting Engine (DeepSeek-V3)

`mtp-engine` is a production-grade, mathematically verified implementation of the **Multi-Token Prediction (MTP)** architecture introduced in **DeepSeek-V3**.

---

## Theoretical Core: Multi-Token Prediction (MTP)

Traditional autoregressive Large Language Models predict exactly one token at a time: given context $t_{1:i}$, the model computes logits for next token $t_{i+1}$. This sequential generation bottleneck bounds inference speed to hardware memory bandwidth.

**DeepSeek-V3 Multi-Token Prediction** extends this paradigm by training $D$ sequential prediction modules to predict $D$ additional future tokens ($t_{i+2}, \dots, t_{i+D+1}$) concurrently with the main token:

1. **Sequential MTP Depth**: $D$ lightweight transformer modules stacked sequentially.
2. **Dual Representation Fusion**: Each depth $k$ fuses the previous depth's representation $h_i^{k-1}$ with the token embedding of ground truth token $t_{i+k-1}$ using dual RMSNorm:
   $$h'_{k, i} = M_k \left[ \text{RMSNorm}(h_i^{k-1}) \;;\; \text{RMSNorm}(\text{Emb}(t_{i+k-1})) \right]$$
   where $M_k \in \mathbb{R}^{d \times 2d}$ projects the fused dimension back to $d$.
3. **Transformer Block**: $h'_{k, i}$ is passed through Pre-RMSNorm Causal Multi-Head Self-Attention and Pre-RMSNorm SwiGLU FFN with residual connections:
   $$h''_k = h'_k + \text{Attention}(\text{RMSNorm}(h'_k))$$
   $$h_k = h''_k + \text{SwiGLU}(\text{RMSNorm}(h''_k))$$
4. **Shared Unembedding Projection**: Logits for future token $t_{i+k+1}$ are computed by sharing the base model's unembedding weight matrix $W_{\text{head}} \in \mathbb{R}^{V \times d}$ with a dedicated output RMSNorm:
   $$\text{logits}_{k, i} = \text{RMSNorm}(h_k) W_{\text{head}}^T$$
   This completely eliminates parameter duplication, adding 0 extra parameters for vocabulary projections.

---

## Zero-Overhead Speculative Drafting at Inference

During inference, MTP modules serve as an embedded **zero-overhead draft model** for speculative decoding:

1. **Single-Pass Drafting**: Given verified prompt $t_{1:i}$, base model emits verified token $t_{i+1}$.
2. **MTP Heads Rollout**: MTP module 1 feeds $(h_i^0, t_{i+1})$ to draft $\hat{t}_{i+2}$; MTP module 2 feeds $(h_i^1, \hat{t}_{i+2})$ to draft $\hat{t}_{i+3}$.
3. **Parallel Verification**: In the next forward step, the base model evaluates all candidate drafted tokens simultaneously in a single forward pass.
4. **Zero Distributional Shift**: Tokens are accepted only if they match the base model's greedy/sampling distribution, guaranteeing strictly zero degradation in generation quality.

---

## CLI Commands & Usage

The CLI tool `mtp-engine` is globally installed into `~/.local/bin/mtp-engine`.

```bash
# 1. Inspect MTP architecture parameters and memory footprint
mtp-engine status --vocab 1000 --dim 128 --heads 4 --depth 2

# 2. Simulate joint training step with multi-token loss
mtp-engine train-step --seq-len 32 --batch-size 4 --lambda-mtp 0.3

# 3. Benchmark MTP speculative decoding speedup vs autoregressive baseline
mtp-engine benchmark --prompt-len 16 --gen-tokens 24 --depth 2

# 4. Run full mathematical invariant test suite
mtp-engine verify
```

---

## Mathematical Invariants Verified

- **Unit Variance**: $\mathbb{E}[\text{RMSNorm}(x)^2] = 1.0 \pm 10^{-6}$
- **Unembedding Weight Tying**: $\text{id}(M_k.\text{shared\_unembed}) \equiv \text{id}(\text{Model}.W_{\text{head}})$
- **Causal Mask Invariance**: Future token mutations do not affect past sequence outputs ($||\nabla_{\text{future}} y_{\text{past}}|| = 0$)
- **Speculative Consistency**: Speculative MTP decoding produces identical token distributions to standard autoregressive rollout.
