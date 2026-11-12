---
title: "Chapter 3: Optimizers"
description: "How mainstream optimizers evolved and what state they keep inside, how to analyze optimizer memory cost, and the foundation for ZeRO memory optimization and mixed-precision training"
tags: ["Optimizer", "Adam", "AdamW", "LAMB", "Memory analysis", "Mixed precision"]
sourceHash: "258c0e689a8b163e"
---

## About this chapter

The optimizer is the steering wheel of training—gradients only tell you which way to go; how to step and how far is all on the optimizer. This chapter skips the math derivations and sticks to the two things an AI Infra engineer most needs: **what the optimizer stores internally** (that decides memory use), and **what special demands show up in distributed settings** (large-batch convergence). Those two points are the direct prerequisites for ZeRO memory optimization and mixed-precision training later.

---

## Chapter structure

### 3.1 Optimizer principles and memory cost analysis

One article covers everything in this chapter. The through-line:

- **Where the optimizer sits in the training loop**: the three steps behind `optimizer.step()`, and why "internal state that stays resident in GPU memory" is the root of the memory cost
- **Evolution line (SGD → Momentum → Adam → AdamW)**: how many extra state tensors each step adds, what problem it solves, and why AdamW became the default for large-model training
- **Memory cost analysis (focus)**: bytes-per-parameter comparison, a full memory ledger for a 7B model under AdamW + mixed precision, the derivation that optimizer state is about 75\% of static memory, and why you need FP32 master weights
- **Large-batch optimizers**: the convergence pain when data parallelism inflates the batch, and the per-layer adaptive scaling idea behind LARS/LAMB plus when they apply
- **Hands-on lab**: Count elements in a PyTorch optimizer `state` and verify empirically that "Adam stores 2 tensors per parameter, SGD stores 0"

---

---

## Learning goals

- Explain what each step from SGD → Momentum → Adam → AdamW solves and what extra state it stores
- Hand-calculate optimizer-state memory for a model of any size under AdamW + mixed precision, and explain why it is about 75\% of static memory
- Explain why mixed-precision training still needs FP32 master weights in the optimizer
- Answer "batch size grew 10×—how do I set the learning rate?", and know when LAMB/LARS are actually worth using
