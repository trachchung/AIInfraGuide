---
title: "Chapter 1: Distributed training overview"
description: "From model-scale bottlenecks, hand-calculate the training memory ledger, survey the five parallelism strategies, then land on rank/torchrun/NCCL setup and your first DDP script"
tags: ["Distributed training", "Memory analysis", "Parallelism strategies", "3D parallelism", "Communication topology", "torchrun", "NCCL", "DDP", "rank"]
sourceHash: "8a55f9a66a117430"
---

## About this chapter

This chapter is the spine of the distributed training module. It builds a three-layer mental model: **why** you need distributed training → how much **GPU memory** training actually needs → which parallelism strategies exist and **how they split work**. Then it drops into the engineering foundation: how GPUs form a process group, how to launch with one torchrun command, and how to get a first DDP training script running. After this chapter you have a map for what every later chapter is solving, and you can run multi-GPU training yourself.

> This chapter has two parts: the first half is the conceptual overview (memory ledger + parallelism panorama), the second half is the engineering foundation (process model + torchrun + NCCL + a minimal DDP script). They map to the two articles below.

---

## Chapter structure

### Part 1: Conceptual overview (article 1.1)

#### 1. Why distributed training

- **Memory bottleneck**: Using the LLaMA family as an example (7B → 70B → 405B), show how model scale and memory demand grow. A single 80GB H100 cannot hold a full training state once you go past 7B
- **Time bottleneck**: Training a 70B model on 1T tokens takes years on one GPU; multi-GPU parallelism can cut wall time nearly linearly
- **Core takeaway**: Distributed training solves two problems—"won't fit" (memory) and "won't finish" (time)

#### 2. Training-state memory analysis (focus of this chapter)

- **Four memory components**:
  - Parameter memory (FP16: $2\Psi$ Bytes)
  - Gradient memory (FP16: $2\Psi$ Bytes)
  - Optimizer-state memory (Adam: FP32 parameter copy $4\Psi$ + first moment $4\Psi$ + second moment $4\Psi$ = $12\Psi$ Bytes)
  - Activation memory (depends on batch size, sequence length, and model width)
- **Worked example**: Full memory ledger for BF16 mixed-precision training of a 7B model
  - Static memory = $(2 + 2 + 12) \times 7 \times 10^9$ Bytes ≈ 112 GB
  - Activation memory estimate (seq_len=2048, batch=1 as an example)
- **Memory formula summary**: 16–18 Bytes per parameter (mixed precision + Adam) + activations

#### 3. Parallelism strategy panorama

- **Data parallelism (DP/DDP)**: Full model on every GPU, shard the data, AllReduce to sync gradients—solves "won't finish"
- **ZeRO / FSDP**: Shard optimizer state / gradients / parameters, gather on demand—solves "won't fit"
- **Tensor parallelism (TP)**: Split matrix ops inside a layer across GPUs—solves "a single layer is too big"
- **Pipeline parallelism (PP)**: Put different layers on different GPUs—solves "too many layers"
- **Sequence parallelism (SP) / context parallelism (CP)**: Split along the sequence dimension—solves "sequence too long"

#### 4. How to choose a parallelism strategy

- **Communication bandwidth sets the scope**: TP stays inside a node linked by NVLink (needs high bandwidth and low latency); PP can cross nodes (only ships activations); DP is the least demanding on bandwidth
- **Decision tree**: Fits on one GPU → DDP; does not fit → FSDP/ZeRO; a single layer does not fit → TP; too many layers → PP; combine them → 3D parallelism
- **How communication topology maps to hardware**: How NVLink (900 GB/s) vs InfiniBand (400 Gbps) vs Ethernet shapes strategy choice

### Part 2: Environment setup and distributed launch (article 1.2)

#### 5. Process groups and core concepts

- Three scales—single GPU, single-node multi-GPU, multi-node multi-GPU—and the bandwidth tiers (NVLink vs InfiniBand)
- Process model: why it is "one process per GPU" (bypass the GIL, own the CUDA context)
- Core concepts: `rank` / `local_rank` / `world_size`, and why `set_device` must use `local_rank`
- Process group `init_process_group` and backend choice (nccl / gloo)
- Subgroups with `new_group`: paving the way for 3D parallelism

#### 6. How torchrun launches jobs

- Why you need a dedicated launcher; the evolution from `torch.distributed.launch` → `torchrun`
- Environment variables torchrun injects (RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT)
- Single-node multi-GPU launch, multi-node multi-GPU launch, and Rendezvous (c10d elastic rendezvous)
- Elasticity and fault tolerance in torchrun (see Chapter 14)

#### 7. NCCL environment variables and communication debugging

- `NCCL_DEBUG=INFO` as the first debugging move
- `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` to pin network interfaces
- `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` as binary-search debug switches
- `CUDA_VISIBLE_DEVICES` and device binding

#### 8. Write your first multi-GPU training script

- Five-part skeleton of a minimal DDP script: init → set_device → wrap with DDP → training loop (with Sampler) → destroy
- A full runnable template with a section-by-section breakdown
- The engineering habit of letting only rank 0 do side work; `DistributedSampler` + `set_epoch`
- Common pitfalls the first time you run multi-GPU

---

## Learning goals

- Calculate BF16 mixed-precision training memory for a model of any size (given parameter count)
- Draw a comparison table of the five parallelism strategies: what they shard × how they communicate
- Pick a suitable parallelism mix from model scale and cluster topology
- Distinguish rank / local_rank / world_size accurately, and write torchrun launch commands for single-node and multi-node jobs
- Independently write and run a minimal DDP training script with torchrun
- Name the first thing to do when multi-GPU communication breaks (`NCCL_DEBUG=INFO`) and the usual follow-up checks
