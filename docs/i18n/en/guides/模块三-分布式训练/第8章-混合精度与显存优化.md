---
title: "Chapter 8: Mixed precision and memory optimization"
description: "FP16/BF16/FP8 mixed-precision training, gradient accumulation, Activation Checkpointing, and what each costs in compute and memory"
tags: ["Mixed precision", "BF16", "FP8", "Activation Checkpointing", "Gradient accumulation", "Memory optimization"]
sourceHash: "1f11d3e160eedb8c"
---

## About this chapter

Parallelism gets the model onto many GPUs, but there is still plenty of memory to save on each card. This chapter covers three techniques that sit outside the parallelism axes and show up in almost every training run: **mixed precision** (lower bit width for speed and memory), **gradient accumulation** (simulate a large batch with little memory), and **Activation Checkpointing** (trade compute for memory). These are the basics of pushing the hardware hard.


---

## Sections

### 1. Mixed-precision training (focus of this chapter)

- **Data type comparison**:

  | Type | Bits | Exponent | Mantissa | Range | Typical use |
  |------|------|----------|----------|-------|-------------|
  | FP32 | 32 | 8 | 23 | Large | Master weight, optimizer |
  | FP16 | 16 | 5 | 10 | Small, easy to overflow | Older approach |
  | BF16 | 16 | 8 | 7 | Same as FP32 | Default for large-model training |
  | FP8 (E4M3) | 8 | 4 | 3 | Very small | Forward pass on H100+ |

- **Why mixed precision saves memory and speeds things up**: lower bit width → roughly half the memory + Tensor Core throughput doubles
- **BF16 mixed-precision flow**: forward/backward in BF16, parameter updates with FP32 Master Weight (ties back to optimizer state in Chapter 3)
- **FP16 vs BF16**: FP16 has a small range and overflows easily; BF16 matches FP32's range but has less precision—why large models usually pick BF16
- **Loss Scaling**: the gradient scaling trick FP16 training needs (static / dynamic); BF16 usually skips it because the range is large enough
- **FP8 training**: H100 Transformer Engine, forward in FP8 with per-tensor dynamic scaling, backward gradients in BF16; where it fits and the accuracy risks

### 2. Gradient Accumulation

- **Goal**: simulate a larger Effective Batch Size under a memory budget
- **Mechanism**: run forward+backward for $K$ steps while accumulating gradients, then update parameters once every $K$ steps
- **Effective Batch Size** = per-GPU batch × world_size × accumulation_steps
- **With PP**: PP micro-batches already give a similar effect; gradient accumulation amplifies it further
- **With DDP**: use `no_sync()` during accumulation to skip AllReduce on intermediate steps and save communication
- **Caveat**: divide the loss by $K$ (or equivalently divide the gradients by $K$)

### 3. Activation Checkpointing (activation recomputation)

- **Problem**: activations saved in the forward pass grow linearly with depth and batch size; in large-model training, activation memory can exceed parameter memory
- **Core idea**: keep only some layers' activations (checkpoints); recompute the rest with a forward pass during backward
- **Full Checkpointing**: keep only each Transformer Block's input; recompute the whole Block on backward
  - Memory savings: activation memory drops from $O(L)$ to $O(\sqrt{L})$ (one checkpoint every $\sqrt{L}$ layers)
  - Compute cost: about 33% more forward compute
- **Selective Checkpointing**: recompute only ops that are cheap but produce large activations (e.g. Attention softmax outputs); keep activations for expensive ops
- **With parallelism**: PP naturally stores activations only per Stage; under TP, SP already cuts activation redundancy (links to Chapters 6/7)

### 4. Memory optimization landscape and how it connects

This section maps every "save memory" lever in the series into one view:

- **Sharding** (multi-GPU): ZeRO/FSDP shards state, TP shards weights, PP shards layers, SP/CP shards activations → Chapters 5–9
- **Lower precision**: mixed precision (Section 1 of this chapter)
- **Time for space**: gradient accumulation, Activation Checkpointing (Sections 2–3 of this chapter)
- **Offload**: ZeRO-Offload/Infinity (Chapter 5)
- A table of "four memory components × which technique acts on each," tying back to the memory ledger in Chapter 1

---

## Learning goals

- Explain the full BF16 mixed-precision training flow (including Master Weight) and why large models prefer BF16
- Say what problem Loss Scaling solves and why BF16 usually does not need it
- Compute Effective Batch Size with gradient accumulation and explain what `no_sync()` does
- Analyze Activation Checkpointing's memory savings ($O(\sqrt{L})$) and compute cost (~33%)
- Sketch the "four memory components × optimization techniques" mapping table
