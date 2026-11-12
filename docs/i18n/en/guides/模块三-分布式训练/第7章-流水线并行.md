---
title: "Chapter 7: Pipeline parallelism (PP)"
description: "What causes the pipeline bubble, and how GPipe, 1F1B, and Interleaved scheduling reduce it—with quantitative analysis"
tags: ["Pipeline parallelism", "PP", "GPipe", "1F1B", "Interleaved", "Pipeline Bubble"]
sourceHash: "1f32a6994bb64113"
---

## About this chapter

Pipeline parallelism (PP) assigns different layers of the model to different devices and is the main way to scale across machines. This chapter focuses on PP's central problem—the pipeline bubble—and on how three generations of scheduling cut that bubble step by step.

---

## Sections

### 1. The basic idea of pipeline parallelism

- **Split across layers**: divide the model's $L$ layers evenly across $P$ stages; each stage owns $\frac{L}{P}$ layers
- **Analogy**: like a factory assembly line—each station does one step, and the product moves through them in order
- **Core problem (bubble)**: in a naive setup, one micro-batch must pass through every stage in series → many GPUs sit idle waiting
- **Bubble ratio** (naive PP): $\frac{P-1}{P}$ (with $P$ stages, $P-1$ time steps are wasted)

### 2. GPipe: micro-batch pipelining

- **Core idea**: split one mini-batch into $M$ micro-batches and inject them into the pipeline in sequence
- **Execution pattern**: finish forward for all micro-batches, then do all backwards together (F-F-F-...-B-B-B-...)
- **Bubble ratio**: $\frac{P-1}{M+P-1}$; larger $M$ means a smaller bubble
- **Downsides**:
  - backward cannot start until forward is done → you must keep activations for every micro-batch → high memory use
  - very large $M$ shrinks the bubble but blows up memory

### 3. 1F1B (One Forward One Backward)

- **Core idea**: alternate forward and backward so activations can be freed early
- **Execution pattern**: Warmup (fill the pipeline) → Steady (each F is followed immediately by a B) → Cooldown (drain the pipeline)
- **Bubble ratio**: same as GPipe, $\frac{P-1}{M+P-1}$
- **Key win**: peak activation memory drops from $M$ copies to $P$ copies (at most $P$ micro-batches' activations are live at once)
- **Megatron-LM's default schedule**

### 4. Interleaved 1F1B (virtual stages)

- **Core idea**: each device owns not one contiguous $\frac{L}{P}$ block of layers, but several non-contiguous segments (virtual stages)
- **Design**: each device takes $v$ virtual stages (each virtual stage has $\frac{L}{P \cdot v}$ layers)
- **Bubble ratio**: $\frac{P-1}{M \cdot v + P - 1}$, equivalent to multiplying $M$ by $v$
- **Cost**: communication frequency rises by $v\times$ (every boundary between virtual stages needs a cross-device activation transfer)
- **When it fits**: when inter-node bandwidth is plentiful, trade extra communication for a smaller bubble

### 5. Engineering challenges for PP

- **Load balance**: different layers can cost different amounts of compute (Embedding vs Transformer); stages need careful assignment
- **Embedding and Loss**: Embedding sits at the front of the model and Loss at the end; they usually go on the first and last stages
- **Cross-node activation transfer**: data moved between stages = $b \times s \times h$ (batch × seq_len × hidden); needs InfiniBand-class bandwidth
- **Working with Activation Checkpointing**: PP pairs naturally with recomputation—recompute on backward instead of storing activations

### 6. Bubble ratio comparison and visualization

| Schedule | Bubble ratio | Peak activation memory | Communication frequency |
|---------|------------|------------|---------|
| Naive PP | $(P-1)/P$ | 1 copy | low |
| GPipe | $(P-1)/(M+P-1)$ | $M$ copies | low |
| 1F1B | $(P-1)/(M+P-1)$ | $P$ copies | low |
| Interleaved 1F1B | $(P-1)/(Mv+P-1)$ | $P$ copies | high ($v\times$) |

### 7. Hands-on exercises

- Draw the timelines for GPipe and 1F1B with 4 stages and 8 micro-batches
- Compute bubble ratios for different $M$, $P$, $v$ combinations
- Work out a layer assignment for a 64-layer model with PP=4

---

## Learning goals

- Explain what causes the pipeline bubble and the formulas that quantify it
- Draw timelines for GPipe, 1F1B, and Interleaved 1F1B
- Compute bubble ratio and peak activation memory for a given config
- Analyze the Interleaved trade-off of communication for a smaller bubble
