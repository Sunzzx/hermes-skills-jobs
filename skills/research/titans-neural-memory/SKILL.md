---
name: titans-neural-memory
description: "Titans Test-Time Neural Long-Term Memory (LTM) Architecture: Online gradient descent memorization, surprise-gated momentum updates, persistent prefix priors, and MAC/MAG sliding-window attention fusion for infinite-context retrieval."
version: 1.0.0
category: research
type: skill+tool
author: Hermes Evolution & Autonomous Syndicate
cli: titans-ltm
tags:
  - FrontierAI2
  - Titans
  - NeuralMemory
  - TestTimeLearning
  - LongTermMemory
  - SurpriseGating
  - LongContext
  - AssociativeRecall
---

# ğŸ§¢F—Fç3¢ÆV&æ–ærFòÖVÖ÷&—¦RBFW7BF–ÖR„æWW&ÂÆöærÕFW&ÒÖVÖ÷'’&6†—FV7GW&R ¤&öGV7F–öâÖw&FRÂÖF†VÖF–6ÆÇ’&–v÷&÷W2–×ÆVÖVçFF–öâöb¥EF—Fç2¢¢†2–çG&öGV6VB–â¥F—Fç3¢ÆV&æ–ærFòÖVÖ÷&—¦RBFW7BF–ÖR†'’&V‡&÷W¢WBÂâÂvöövÆR&W6V&6‚ÂFV2##B’à ¥7FæF&BG&ç6f÷&ÖW"ÖöFVÇ27VffW"g&öÒgVæFÖVçFÂF–6†÷Fö×“ £â¢¤ÖöFVÂvV–v‡G2EÅÇF†WFB¢£¢7FF–2gFW"G&–æ–æs²–æ6&ÆRöbÆV&æ–ær÷"FF–ærG–æÖ–6ÆÇ’Fò'VçF–ÖRf7G2à£"â¢¤GFVçF–öâµbÔ66†R¢£¢W†VÖW&ÂÂVæ&÷VæFVBDò„â’BÖVÖ÷'’w&÷wF‚æBDò„åã"’F6ö×WFR6ö×ÆW†—G“²7W66WF–&ÆRFò6öçFW‡BF–ÇWF–öâæBGFVçF–öâG&šY‚‚•][œÈœ™XZÜÈ\È˜\œšY\ˆH[›ÙXÚ[™ÈH
Š“™]\˜[Û™ËU\›HY[[ÜH
JJŠˆ[Ù[H]
›X\›œÈ]\İ[YJˆšXHÛ›[™HÜ˜YY[\ØÙ[ˆY[[ÜH\È\˜[Y]\š^™Y\ÈHY\ÙZYÚX]š^ÜˆS]šX]H	IÚXÚ\È[˜[ZXØ[H\]YÚÙ[‹XK]ÚÙ[ˆ˜\ÙYÛˆH
Šœİ\œš\ÙHY]šXÊŠ‹Y\]™HXØ^K[™[ÛY[[K‚‚‹KKB‚ˆÈÈ<'ääˆX][X]XØ[›İ[™][ÛœÈ	ˆY[[ÜH[˜[ZXÜÂ‚ˆÈÈÈKˆ™]\˜[Û™ËU\›HY[[ÜH
JH™]šY]˜[	ˆ™YXİ[Ûˆ\œ›Ü‚•HY[[ÜH[Ù[H	Wİ[ˆX]˜ÔŸWÙŞİ˜[H[Y\ÈŞÚÙ^__IX\È]Y\KÚÙ^H™XİÜœÈÈ™]šY]™Y˜[YH™XİÜœÎ‚‰	]İŸWİHWŞİL_H×İ	‚•Ú[ˆ[ˆ\ÜÛØÚX][Ûˆ	
×İ—İ
I\œš]™\ËH[Ù[Ø[İ[]\ÈH
Šœ™YXİ[Ûˆ\œ›Ü—
ˆ[™
ŠœØØ[\ˆİ\œš\ÙW
‚‰	$e_t = v_t - \\hat{v}_t = v_t - M_{t-1} k_t$$
[$mathcal{S}_t = \\frac{1}{2} \\|e_t\\|^2_2$$

The gradient of surprise with respect to memory matrix $M$ is:
$$\\nabla_M \\mathcal{S}_t = - (v_t - M_{t-1} k_t) k_t^T = - e_t \\otimes k_t$$

### 2. Online Test-Time Memory Update with Adaptive Momentum & Forgetting
To prevent catastrophic interference while consolidating novel knowledge, Titans introduces adaptive gating:
- **Forgetting Gate (Weight Decay)**: $\\alpha_t = \\sigma(x_t W_\\alpha + b_\\alpha) \\in [0, 1]$
- **Adaptive Learning Rate**: $\\eta_t = \\sigma(x_t W_\\eta + b_\\eta) \\cdot \\eta_{base}$
- **Momentum Factor**: $\\beta_t = \\sigma(x_t W_\\beta + b_\\beta) \\in [0, 1]$

The memory state updates via momentum gradient descent:
$$V_t = \\beta_t V_{t-1} + \\eta_t (e_t \\otimes k_t)$$
$$M_t = (1 - \\alpha_t) M_{t-1} + V_t$

When the model already recognizes an association ($e_t \\approx 0$), surprise vanishes and memory remains unchanged. When encountering novel or surprising patterns ($e_t \\gg 0$), the memory rapidly updates its weights online.

### 3. Memory Branch Fusion Architectures
- **MAG (Memory As Gating)**: Computes dynamic gating between Sliding-Window STM and Neural LTM:
  $$g_t = \\sigma([y_{stm}; y_{ltm}] W_g + b_g)$$
  $$y_t = (1 - g_t) \\odot y_{stm} + g_t \\odot y_{ltm}$$j- **MAC (Memory As Context)**: Memory readouts are prepended as contextual prefix tokens to attention key-values.
- **MAL (Memory As Layer)**: Sequential composition where LTM acts as an adaptive non-linear feature layer prior to attention.

---

## ğŸ›‰€ CLI Reference (`titans-ltm`)

The executable CLI tool is installed in `~/.local/bin/titans-ltm`:

```bash
# 1. Run complete mathematical and gradient verification suite
titans-ltm verify

# 2. Benchmark throughput and memory complexity scaling
titans-ltm bench --quick

# 3. Test associative Needle-in-a-Haystack recall over long sequences
titans-ltm needle --length 1000

# 4. Inspect model dimensions, parameter footprint, and memory matrix
titans-ltm inspect
```

---

## ğŸ–¿ Empirical Benchmarks & Scaling Advantages

- **Bounded $O(W)$ Attention Memory**: Fixed-size sliding-window cache eliminates memory blowup.
- **Constant $O(1)$ Neural Memory Footprint**: Fixed weight matrix $M$ retains associative associations across unbounded context.
- **Exponential Loss Decay**: Demonstrates >$10^4\\times$ loss reduction on repeated token patterns.
- **Zero-Drift Retrieval**: Needle-in-a-Haystack associations retained in matrix weights without attention sink dilution.
