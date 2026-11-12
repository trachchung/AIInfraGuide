---
title: "Chapter 2: Collective communication primitives"
description: "The physical foundation of distributed training—semantics of AllReduce/ReduceScatter/AllGather/All-to-All and other collectives, how the Ring algorithm works, and quantitative communication-volume analysis"
tags: ["Collective communication", "AllReduce", "Ring AllReduce", "NCCL", "Communication volume analysis"]
sourceHash: "52e8a069756a7f48"
---

## About this chapter

Every parallelism strategy boils down to one thing: **how GPUs exchange data efficiently**. Names like AllReduce, ReduceScatter, and AllGather show up in every later chapter—DDP syncs gradients with AllReduce, ZeRO-2 uses ReduceScatter, TP uses AllReduce, MoE uses All-to-All. This chapter lays out that communication alphabet once, and gives you a way to quantify communication volume, so you can compute the communication cost of any strategy yourself later.

> This chapter is prerequisite knowledge for reading the communication cost of every parallelism strategy that follows. By the end you should be able to sketch the "input → output" semantics of each primitive from memory.

---

## Chapter structure

### 1. Why cover communication primitives on their own

- The essence of a parallelism strategy = sharding + communication; sharding decides memory, communication decides speed
- A "communication op vs parallelism strategy" lookup table (panorama first, details later)
- Where communication happens: NVLink (intra-node) vs InfiniBand/RoCE (inter-node) vs PCIe, with bandwidth differing by an order of magnitude

### 2. Point-to-point communication

- `send` / `recv`: the most basic two-process communication; PP stages pass activations this way
- Sync vs async (`isend` / `irecv`) and deadlock risk

### 3. Collective primitives in detail

For each one: an "input → output" semantics diagram + typical uses:

- **Broadcast**: Copy one rank's data to every rank (e.g. sync parameters at init)
- **Scatter / Gather**: Distribute / collect (older DP setups, data sharding)
- **Reduce**: Reduce data from all ranks (sum / max) onto one rank
- **AllReduce**: Broadcast the reduced result to every rank (DDP gradient sync, TP output merge)—focus of this chapter
- **ReduceScatter**: After reduce, each rank keeps only its shard (ZeRO-2/3, FSDP backward)
- **AllGather**: Concatenate every rank's shard into the full tensor and give it to every rank (ZeRO-3 forward, FSDP parameter reassembly)
- **All-to-All**: Every rank sends different data to every other rank (MoE Expert Parallel, Ulysses sequence parallelism)
- **Key identity**: `AllReduce = ReduceScatter + AllGather` (central throughout ZeRO/SP)

### 4. Ring AllReduce algorithm (focus of this chapter)

- Problem with a naive AllReduce: a central-node bandwidth bottleneck
- Two phases of Ring: Reduce-Scatter phase + AllGather phase
- **Communication volume derivation**: each GPU sends/receives $2\frac{N-1}{N}\Psi \approx 2\Psi$ (independent of GPU count—that is the point of Ring)
- Bandwidth optimality: why Ring approaches the bandwidth limit on large messages
- Other topologies briefly: Tree AllReduce (low latency on small messages), Double Binary Tree, and NCCL's automatic choice

### 5. Quantifying communication volume

- Unified notation: parameter count $\Psi$, batch $b$, sequence $s$, hidden dim $h$
- Three communication-volume formulas: AllReduce $2\Psi$, ReduceScatter / AllGather $\Psi$ each, and All-to-All
- Communication time estimate: $T \approx \alpha + \frac{\text{data volume}}{\text{bandwidth}}$ (latency term $\alpha$ + bandwidth term)
- Basics of compute/communication overlap: why you can "compute while transferring"

### 6. Hands-on labs

- Run all 7 collective primitives with `torch.distributed`, print each rank's inputs and outputs to verify semantics
- Use `NCCL_DEBUG=INFO` to see which algorithm NCCL picked
- Measure AllReduce bandwidth across message sizes, and compare NVLink vs PCIe

---

## Learning goals

- Sketch the "input → output" semantics of all 7 collective primitives
- Derive that Ring AllReduce moves $2\Psi$ per GPU and does not grow with GPU count
- Recite the identity $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$ and explain what it means for ZeRO
- Estimate one communication's cost with $T \approx \alpha + \text{data volume}/\text{bandwidth}$
