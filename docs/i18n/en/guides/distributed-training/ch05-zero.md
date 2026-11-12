---
title: "Chapter 5: The ZeRO memory-optimization series"
description: "How ZeRO-1/2/3 shard state and what they cost in communication, plus how ZeRO-Offload and Infinity move state onto heterogeneous memory"
tags: ["ZeRO", "DeepSpeed", "Memory optimization", "Offload", "Communication analysis"]
sourceHash: "a041cbca1820d56b"
---

## About this chapter

ZeRO (Zero Redundancy Optimizer) is the core technique behind DeepSpeed. This chapter starts from "where is the redundancy in training state," walks stage by stage through how sharding removes it, and looks at each stage's communication cost and when to use it.

---

## Sections

### 1. ZeRO's core insight

- **Redundancy analysis**: under DDP every GPU redundantly stores the full optimizer state ($12\Psi$), gradients ($2\Psi$), and parameters ($2\Psi$); on an $N$-GPU cluster, $(N-1)/N$ of that storage is waste
- **Design idea**: a "shard–aggregate" pattern—each GPU normally holds only $\frac{1}{N}$, fetches the full data over the network when needed, then discards it
- **Three-stage progression**: from the easiest piece to shard (optimizer state) to the hardest (parameters), cutting redundancy step by step

### 2. ZeRO-1: shard optimizer state

- **What is sharded**: Adam's FP32 parameter copy + first-order momentum + second-order momentum ($12\Psi$ Bytes total)
- **Communication pattern**: gradients still need AllReduce (same as DDP); after the parameter update you need AllGather to sync parameters
- **Memory savings**: per GPU drops from $16\Psi$ to $4\Psi + \frac{12\Psi}{N}$
- **Communication volume**: same as DDP ($2\Psi$) + an extra AllGather of parameters
- **When it fits**: when optimizer state dominates memory (Adam is ~75%); a little extra communication buys a large memory cut

### 3. ZeRO-2: also shard gradients

- **What is sharded**: on top of ZeRO-1, gradients are stored sharded too
- **Communication pattern**: backward uses ReduceScatter (instead of AllReduce); each GPU keeps only the aggregated gradient for its own shard
- **Memory savings**: per GPU drops to $2\Psi + \frac{14\Psi}{N}$
- **Communication volume**: ReduceScatter = $\Psi$ (half of AllReduce's $2\Psi$, but you lose the AllGather that AllReduce implies)
- **When it fits**: medium-to-large models where sharding optimizer state + gradients still leaves parameters fitting on one GPU

### 4. ZeRO-3: shard parameters too

- **What is sharded**: parameters, gradients, and optimizer state are all sharded
- **Communication pattern**:
  - Forward: AllGather to rebuild the current layer's parameters → compute → free
  - Backward: AllGather parameters → compute gradients → ReduceScatter gradients → free parameters
- **Memory savings**: per GPU drops to $\frac{16\Psi}{N}$ (ideal linear scaling)
- **Communication volume**: $3\Psi$ (forward AllGather $\Psi$ + backward AllGather $\Psi$ + ReduceScatter $\Psi$), 50% more than DDP
- **When it fits**: the parameters themselves do not fit on one GPU, so you must shard

### 5. ZeRO-Offload and ZeRO-Infinity

- **ZeRO-Offload**: offload optimizer state and gradient computation to the CPU; the GPU only does forward/backward
  - When it fits: training large models on few GPUs (1–4)
  - Cost: CPU–GPU transfer bandwidth becomes the bottleneck (PCIe 4.0: ~32 GB/s)
- **ZeRO-Infinity**: on top of Offload, also uses NVMe SSD storage
  - When it fits: training extremely large models (trillion parameters) on limited GPUs
  - Key techniques: chunked prefetch, overlap of compute with I/O

### 6. ZeRO selection guide

| Stage | Memory savings | Communication cost | When it fits |
|------|---------|---------|---------|
| ZeRO-1 | optimizer state $\frac{12\Psi}{N}$ | close to DDP | parameters + gradients fit on one GPU |
| ZeRO-2 | + gradients | slightly above DDP | parameters fit on one GPU |
| ZeRO-3 | everything | 50% more than DDP | parameters do not fit either |
| Offload | uses CPU memory | + PCIe transfers | few GPUs, large model |

### 7. How ZeRO relates to FSDP

- FSDP `FULL_SHARD` ≈ ZeRO-3, `SHARD_GRAD_OP` ≈ ZeRO-2
- ZeRO is the algorithm / paper-level idea; FSDP is the native PyTorch implementation; DeepSpeed is Microsoft's separate implementation
- FSDP vs DeepSpeed: prefer FSDP inside the PyTorch ecosystem; use DeepSpeed when you need Offload/Infinity or already have a DeepSpeed config

---

## Learning goals

- Sketch, for each ZeRO-1/2/3 stage, what is sharded and how communication works
- Compute per-GPU memory formulas and per-step communication volume for each stage
- Explain ZeRO-Offload's CPU offload mechanism and its performance bottleneck
- Pick the right ZeRO stage from model size and hardware
