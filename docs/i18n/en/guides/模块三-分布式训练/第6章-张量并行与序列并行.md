---
title: "Chapter 6: Tensor parallelism (TP) and sequence parallelism (SP)"
description: "Megatron-LM's tensor-parallel design (Column/Row Parallel Linear), where communication is inserted, and how sequence parallelism cuts activation memory"
tags: ["Tensor parallelism", "Sequence parallelism", "Megatron-LM", "TP", "SP", "NVLink"]
sourceHash: "0182a70c58739e00"
---

## About this chapter

When even a single Transformer layer's parameters or activations exceed one GPU's memory, you have to split the **intra-layer** matrix math across GPUs—that is tensor parallelism (TP). Sequence parallelism (SP) then cleans up the activation redundancy that TP leaves in the regions it does not cover.

---

## Sections

### 1. Why you need tensor parallelism

- **Limits of data parallelism**: DP/FSDP only shard redundant copies of training state; if a single layer's forward already exceeds one GPU's compute or memory (e.g. FFN intermediate activations), DP cannot help
- **Where TP sits**: split one matmul $Y = XA$ across GPUs so each holds only $\frac{1}{N}$ of the weights and the matching slice of activations
- **Bandwidth constraint**: TP needs communication on every layer and is very latency-sensitive → usually kept inside one NVLink-connected node (≤ 8 GPUs)

### 2. Column Parallel Linear (split by columns)

- **How it splits**: weight $A$ is column-split into $[A_1, A_2, ..., A_N]$; each GPU holds $A_i$
- **Compute**: $Y_i = X \cdot A_i$ (each GPU multiplies the full input $X$ by its own column shard)
- **Output**: each GPU gets a different column shard $Y_i$ of $Y$
- **Communication**: input $X$ must be Broadcast / copied to every GPU (or every GPU already has the full input)
- **Where it is used**: the first linear layer of the FFN, Attention QKV projections

### 3. Row Parallel Linear (split by rows)

- **How it splits**: weight $A$ is row-split; each GPU holds $A_i$ (a row-direction shard)
- **Input requirement**: input $X$ must also be column-sharded into $X_i$ (so it lines up with Column Parallel's outputs)
- **Compute**: $Y_i = X_i \cdot A_i$ (each GPU computes a partial result)
- **Communication**: the per-GPU $Y_i$ values are **AllReduce**d (sum) into the full output $Y$
- **Where it is used**: the second linear layer of the FFN, Attention output projection

### 4. TP layout for a Transformer block

- **Attention layer**: QKV uses Column Parallel (multi-head splits naturally; each GPU owns some Heads); output projection uses Row Parallel
- **FFN layer**: first linear is Column Parallel (split the intermediate dim); second linear is Row Parallel (merge the output)
- **Where communication lands**: each Transformer block's forward needs 2 AllReduces (Attention output + FFN output)
- **Backward**: forward AllReduce maps to backward AllReduce; communication volume is symmetric
- **Per-layer volume**: forward $2 \times 2bsh$ (2 AllReduces, each moving $2bsh$ of data)

### 5. Sequence parallelism (SP)

- **Problem**: LayerNorm and Dropout are outside the TP split, so their inputs/outputs are full on every GPU → redundant activations
- **Fix**: in non-TP regions (LayerNorm, Dropout), shard activations along the sequence dimension
- **Communication change**: AllReduce in the TP region becomes ReduceScatter (when entering a non-TP region) + AllGather (when returning to a TP region)
- **Gain**: activation memory in non-TP regions drops from full $bsh$ to $\frac{bsh}{N}$; total activation memory approaches linear scaling
- **Communication change**: total volume stays the same (AllReduce = ReduceScatter + AllGather), but activation memory drops a lot

### 6. TP under GQA/MQA

- **Problem**: GQA's KV Head count may be smaller than the TP degree (e.g. KV Head = 4, TP = 8)
- **Strategies**:
  - replicate KV Heads (every TP rank keeps a full copy of the KV Heads)
  - have some ranks share KV Heads (a grouping strategy)
- **Impact**: replicating KV Heads makes TP's parameter split uneven, but compute can still run in parallel

### 7. Hands-on exercises

- Draw the split of one Transformer block under TP=4 (label every communication step)
- Read the Megatron-LM `ColumnParallelLinear` and `RowParallelLinear` source

---

## Learning goals

- Derive input/output shapes and communication points for Column Parallel and Row Parallel
- Draw the full TP split of a Transformer block (Attention + FFN)
- Explain how sequence parallelism turns AllReduce into ReduceScatter + AllGather to save activation memory
- Analyze how to handle too few KV Heads when applying TP to a GQA model
