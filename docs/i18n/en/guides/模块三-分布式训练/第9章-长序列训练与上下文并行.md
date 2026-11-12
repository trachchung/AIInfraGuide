---
title: "Chapter 9: Long-sequence training and context parallelism"
description: "Breaking the sequence-length wall—Attention's O(s²) problem, and a comparison of Ring Attention, DeepSpeed-Ulysses, and Megatron Context Parallel"
tags: ["Long-sequence training", "Context Parallel", "Ring Attention", "Ulysses", "FlashAttention"]
sourceHash: "156ec276609e75bd"
---

## About this chapter

As long documents, long context, and long chains of thought become common, sequence length has grown from 2K to 128K and even 1M. Attention compute and activation memory scale as $O(s^2)$ with sequence length, so a single GPU cannot hold it—which led to **Context Parallelism (CP)**, sharding along the sequence dimension. This chapter stands on its own and clarifies how the three main approaches differ in what they shard. One distinction to keep straight: sequence parallelism (SP) in the previous chapter shards activations outside Attention; this chapter shards **Attention compute itself**.

---

## Sections

### 1. The sequence-length wall

- Attention's $O(s^2)$ compute and memory: double $s$, and the cost grows 4×
- How KV Cache and activation memory grow with $s$
- Why TP/PP/DP cannot fix "one sequence is too long"

### 2. FlashAttention recap (prerequisite)

- Tiled compute + online softmax, cutting $O(s^2)$ memory down to $O(s)$
- Why most CP schemes are built on top of FlashAttention

### 3. Ring Attention

- Shard Q/K/V across GPUs along the sequence dimension
- Pass KV shards in a ring; each GPU computes local Attention block by block and merges with online softmax
- Overlap compute with KV transfers
- Load imbalance under causal masking, and Zigzag/Striped fixes

### 4. DeepSpeed-Ulysses

- Shard along the Head dimension → All-to-All rearrange into sequence-dimension shards
- Communication volume: independent of sequence length, tied to the Head dimension
- Trade-offs vs Ring Attention: communication pattern vs Head-count limits

### 5. Megatron Context Parallel

- Shard along the sequence dimension, with FlashAttention tiling
- Working with TP/SP: how Megatron runs TP + CP together
- How communication cost scales with sequence length

### 6. Comparing the three approaches

| Approach | Shard dimension | Comm primitive | Limits | Best fit |
|------|---------|---------|------|---------|
| Ring Attention | Sequence | P2P ring of KV | Load imbalance under causal mask | Ultra-long sequences, large KV |
| Ulysses | Head→sequence | All-to-All | Head count must be ≥ CP degree | Models with many heads |
| Megatron CP | Sequence | P2P/AllGather | Tied to Megatron | Megatron ecosystem |

### 7. Hands-on

- Run a single-GPU long sequence with FlashAttention and watch memory vs $s$
- Implement a simplified Ring Attention: KV ring pass + online softmax merge

---

## Learning goals

- Explain Attention's $O(s^2)$ problem and why you need to shard along the sequence
- Distinguish SP (non-Attention activations) from CP (Attention compute) by shard dimension
- Compare Ring Attention, Ulysses, and Megatron CP on shard dimension and communication pattern
- Describe the load-balance problem causal masking creates for Ring Attention and the ideas behind the fixes
