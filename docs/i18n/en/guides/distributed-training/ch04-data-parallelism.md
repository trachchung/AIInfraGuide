---
title: "Chapter 4: Data parallelism (DP, DDP, FSDP)"
description: "How three generations of data parallelism evolved, how they work in practice, and what really differs between AllReduce gradient sync and parameter sharding"
tags: ["Data parallelism", "DDP", "FSDP", "AllReduce", "ReduceScatter"]
sourceHash: "fdc6b4e5232dfb8e"
---

## About this chapter

Data parallelism is the most basic and most widely used parallel strategy. This chapter follows a "why did it evolve" line and walks through what each generation—DP → DDP → FSDP—fixes and what cost it adds. There are two companion articles:

- **4.1 Data parallelism in depth**: the principles piece—math foundations, the three-generation story, the Bucket mechanism, Ring AllReduce communication volume, a memory ledger, and how to choose.
- **4.2 PyTorch data parallelism from principles to practice**: the hands-on piece—PyTorch communication basics, full runnable DDP/FSDP code, multi-node torchrun launch, and common pitfalls.

Read 4.1 first to lock in the principles, then use 4.2 to put them into practice.

---

## Sections

### 1. DataParallel (DP) — one process, many threads

- **How it works**: one process owns the model; on each forward it Scatters the model to multiple GPUs and Gathers outputs back to the primary GPU
- **Fatal flaws**: Python GIL, uneven memory and compute on the primary GPU, communication over PCIe instead of NVLink
- **Where it fits**: quick prototypes only; do not use it in production

### 2. DistributedDataParallel (DDP) — the production default

- **Core mechanism**:
  - one independent process per GPU, each holding a full model replica
  - `init_process_group` builds the process group
  - `DistributedSampler` keeps each process's data from overlapping
  - gradients sync with AllReduce during backward
- **Bucket mechanism and compute–communication overlap**:
  - parameter gradients are grouped into Buckets in reverse-compute order
  - each Bucket launches AllReduce as soon as it is ready, without waiting for the full backward pass
  - communication overlaps with backward compute on later Buckets
- **Communication volume**: AllReduce volume per step = $2\Psi$ (total send+recv per GPU under Ring AllReduce)
- **Usage notes**: `DistributedSampler` + `set_epoch()`, log/save only on rank 0, access raw parameters via `model.module`
- **Limit**: every GPU stores the full model → parameters + gradients + optimizer state must fit in one GPU's memory

### 3. FullyShardedDataParallel (FSDP) — memory-optimized data parallelism

- **Core idea**: do not keep a full redundant copy of every state on each GPU; shard it, and AllGather on demand when you need to compute
- **Workflow**:
  1. Initial state: each GPU holds only $\frac{1}{N}$ of parameters / gradients / optimizer state
  2. Forward: AllGather to rebuild full parameters → compute → free non-local shards
  3. Backward: AllGather parameters → compute gradients → ReduceScatter sharded gradients → free full parameters
  4. Update: each GPU updates only the shard it owns
- **Sharding strategy choices**:
  - `FULL_SHARD` (ZeRO-3): shard parameters + gradients + optimizer state; lowest memory
  - `SHARD_GRAD_OP` (ZeRO-2): shard gradients + optimizer state; leave parameters unsharded
  - `HYBRID_SHARD`: full shard within a node + data parallelism across nodes; recommended for multi-node
  - `NO_SHARD`: equivalent to DDP; for debugging
- **Communication volume**: per step = AllGather (forward) + AllGather + ReduceScatter (backward) = $3\Psi$, 50% more than DDP
- **Engineering knobs**: `auto_wrap_policy` (auto-shard by module size or type), `MixedPrecision`, DCP save/load
- **FSDP2 briefly**: per-parameter sharding, built on DTensor, combines better with TP

### 4. DDP vs FSDP selection guide

- Comparison table: memory use, communication volume, code complexity, suitable scale
- Decision rule: if it fits on one GPU → DDP (simple and efficient); if not → FSDP; for large multi-node models → FSDP + `HYBRID_SHARD`

### 5. Hands-on exercises

Companion code and full implementations are in **4.2 PyTorch data parallelism from principles to practice**:

- Exercise A: turn a single-GPU script into a DDP version (30 minutes)
- Exercise B: on 2–4 GPUs, compare peak memory and training throughput for DDP vs FSDP
- Exercise C: measure how different `sharding_strategy` values affect memory and speed

---

## Learning goals

- Explain what each step DP → DDP → FSDP fixed and what cost it introduced
- Convert a single-GPU script to multi-GPU training with PyTorch DDP (including Sampler, set_epoch, and model saving)
- Configure FSDP sharding strategy and auto-wrap policy
- Compute per-step communication volume for DDP and FSDP and explain the difference
