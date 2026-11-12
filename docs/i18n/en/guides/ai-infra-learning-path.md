---
title: "AI Infra learning path"
description: "From prerequisites and CUDA kernel optimization through distributed training to inference deployment: knowledge points, recommended resources, and measurable checkpoints for each layer of AI Infra"
tags: ["AI Infra", "Learning path", "CUDA", "Distributed training", "Inference optimization"]
sourceHash: "eec736e8e747b86c"
---

AI Infra (artificial intelligence infrastructure) is what makes large models trainable, fast, and affordable on GPU clusters. This article walks the AI Infra learning path layer by layer—from foundations to inference deployment—and for each layer lists the knowledge you need, recommended materials, and measurable checkpoints.

The articles on this site follow this path. Under each layer's "Recommended materials," **site articles** are linked so you can jump straight in.

<!-- more -->

## 📑 Table of contents

- [Overview: three-layer architecture](#️-overview-three-layer-architecture)
- [Layer 0: Prerequisites](#-layer-0-prerequisites)
- [Layer 1: CUDA programming and operator optimization](#-layer-1-cuda-programming-and-operator-optimization)
- [Layer 2: Distributed training](#️-layer-2-distributed-training)
- [Layer 3: Inference and deployment](#-layer-3-inference-and-deployment)
- [Getting started as a newcomer](#-getting-started-as-a-newcomer)
- [References](#-references)

---

## 🗺️ Overview: three-layer architecture

At its core, AI Infra is **"releasing hardware compute with systems engineering"**. Bottom-up, we can split it into three core layers plus a prerequisite layer:

| Layer | Name | Core focus | Matching module on this site |
|------|------|-----------|-------------|
| Layer 0 | Prerequisites | Programming languages, math foundations, Transformer architecture, PyTorch, GPU hardware, communication topology | [Module 1: Prerequisites](https://caomaolufei.github.io/AIInfraGuide/prerequisites/) |
| Layer 1 | CUDA programming and operator optimization | GPU architecture, memory hierarchy, kernel writing, FlashAttention, AI compilers | [Module 2: CUDA programming and operator optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/) |
| Layer 2 | Distributed training | Data parallelism, ZeRO, 3D parallelism, context parallelism, MoE parallelism, mixed precision | [Module 3: Distributed training](https://caomaolufei.github.io/AIInfraGuide/distributed/) |
| Layer 3 | Inference and deployment | KV Cache, PagedAttention, quantization, Speculative Decoding, PD disaggregation, distributed inference | [Module 4: Inference optimization](https://caomaolufei.github.io/AIInfraGuide/inference/) |

Every optimization is a trade-off in the impossible triangle of **"compute, communication, and memory"**: ZeRO trades communication for memory; rematerialization (Activation Checkpointing) trades compute for memory; quantization trades precision for memory and bandwidth. While learning, keep asking: **what does this technique give up, and what does it buy?**

---

## 📖 Layer 0: Prerequisites

AI Infra is not a field you start from a blank slate—it sits on programming skill, math intuition, and model understanding. If the next three layers are the building, this layer is the foundation: a shaky foundation wobbles more the higher you go. The good news is you do not need to be an expert in every direction; "good enough" is enough to keep climbing.

### 0.1 Knowledge points

**Programming languages**

- **Python**: the lingua franca of the AI ecosystem. You need fluency, not just "I can write some." Focus on: OOP, decorators, generators, multiprocessing/multithreading, performance profiling
- **C/C++**: the host language for CUDA. You do not need template metaprogramming mastery, but you do need pointers, memory management, and the compile/link process. Reading C++ project code and writing and compiling simple C++ functions is enough
- **Linux basics**: CLI, shell scripts, process management, environment variables. Almost all AI Infra development and deployment happens on Linux

**Math foundations**

- **Linear algebra**: matrix multiply, transpose, blocked matrix ops, basic eigenvalue ideas. Understanding "why Transformer's core ops are all matrix multiply" needs this. You do not need theorem proofs, but you need intuition for dimension changes—seeing (B, S, H) x (H, V) and immediately knowing the result is (B, S, V)
- **Basic probability and statistics**: distributions, expectation, variance, the probabilistic reading of Softmax, what cross-entropy loss means. Correctness arguments for quantization and Speculative Decoding both lean on probability
- **Calculus** (awareness is enough): chain rule, what a gradient means. Understanding why backprop works and why gradients overflow in mixed precision needs this

**Transformer architecture**

This is the "shared base" of the large-model era. Every later layer optimizes around it—CUDA optimizes its operators, distributed training shards its parameters, inference speeds up its generation. You must understand:

- **Self-Attention**: what Q, K, V mean and how they are computed (QK^T → scale → softmax → PV); why Attention complexity is O(N^2)—the prerequisite for understanding FlashAttention later
- **Feed-forward network (FFN)**: the original Transformer uses two linear layers + activation; mainstream models like LLaMA switch to SwiGLU (three weight matrices). FFN holds most of the model parameters
- **Positional encoding**: Sinusoidal, RoPE, and why Transformers need extra position information
- **Normalization**: LayerNorm vs RMSNorm, Pre-Norm vs Post-Norm, and why large models generally use Pre-Norm
- **Full forward pass**: starting from token embedding, walk data through one Transformer Block and state the input/output dims at each step. The original paper is Post-Norm (Attention → Add & Norm → FFN → Add & Norm); mainstream LLMs are Pre-Norm (Norm → Attention → Add → Norm → FFN → Add)

**PyTorch**

- **Core ideas**: Tensor ops, autograd, how Module / Parameter are organized
- **Training loop**: the standard DataLoader → forward → loss → backward → optimizer.step flow
- **Save and load**: state_dict, checkpoints
- **Basic debugging**: `torch.cuda.memory_summary()` for memory use, `torch.profiler` for light performance analysis

**Communication topology**

Distributed training and multi-GPU inference both need efficient GPU-to-GPU and node-to-node communication. This knowledge is the prerequisite for understanding parallel strategies later—if you do not know NVLink and PCIe differ by an order of magnitude in bandwidth, you cannot understand "why tensor parallelism cannot cross machines."

- **Intra-node communication**: NVLink / NVSwitch bandwidth and topology. NVLink is a dedicated high-speed path between GPUs in the same chassis—think of it as a private conveyor belt inside a factory campus
- **Inter-node communication**: InfiniBand (IB), RoCE. IB is more like a high-speed rail between cities: lower bandwidth than NVLink, but it can connect distant nodes
- **Collective primitives**: meaning and cost of AllReduce, AllGather, ReduceScatter—the most frequent ops in distributed training; their communication-volume formulas are the basis for analyzing parallel strategy cost
- **NCCL**: basic use and tuning of NVIDIA's collective communication library

### 0.2 Recommended materials

**Site articles:** [Programming language basics](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/第1章-编程语言基础/) · [Math foundations](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/第2章-数学基础/) · [Transformer architecture quick start](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/transformer架构快速入门/) (and the [Chapter 3 series](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/第3章-transformer架构详解/)) · [PyTorch framework quick start](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/pyroch/pytorch框架入门/) · [GPU basics](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics/) · [Cluster communication networks and NCCL](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/collective-communication-primer/)

| Type | Resource | Notes |
|------|------|------|
| Paper | Attention Is All You Need | Original Transformer paper; required reading |
| Tutorial | The Illustrated Transformer (Jay Alammar) | Illustrated Transformer intro; good for intuition |
| Walkthrough | Linlang Amu: illustrated LLM inference | Detailed LLM architecture; best read alongside the paper |
| Tutorial | Official PyTorch tutorial (60 Minute Blitz) | Fast PyTorch onboarding |
| Video | 3Blue1Brown: Essence of linear algebra | Geometric intuition for linear algebra; more efficient than textbooks |
| Course | MIT The Missing Semester | Shell, Vim, Git, debugging, profiling—CLI fundamentals |
| Video | Andrej Karpathy: Let's build GPT from scratch | Hand-write a GPT and walk every Transformer module |
| Official docs | NVIDIA NCCL docs | Collective primitives and multi-GPU programming |
| Tutorial | NVIDIA Deep Learning Performance Guide | Methodology for finding hardware bottlenecks |

### 0.3 Checkpoints

You do not need to go deep on this layer, but you must clear these "good enough" bars:

- **Whiteboard a Transformer**: without notes, draw a full Pre-Norm Transformer Decoder Block (RMSNorm → Masked Self-Attention → residual add → RMSNorm → FFN → residual add), label input/output dims at each step (e.g. (B, S, D) → (B, S, D)), and explain how Q, K, V are obtained from the input via linear projections
- **Dimension derivation**: given a 7B config (hidden_dim=4096, num_heads=32, num_layers=32, FFN intermediate dim 11008 with SwiGLU, vocab_size=32000), hand-estimate total parameters (Attention weights + FFN weights + Embedding) within 20% (reference answer ~6.7B)
- **PyTorch training script**: independently write a full PyTorch training loop (no Trainer class) with DataLoader, forward, loss, backward, optimizer step, LR schedule, checkpoint save, and run it on GPU
- **C++ read/write basics**: read the host side of a simple CUDA kernel (malloc, memcpy, kernel launch, free) and understand data movement between CPU and GPU
- **Day-to-day Linux**: on a server, independently: SSH login, tmux sessions, conda/pip envs, check GPUs (`nvidia-smi`), manage code with git, write simple bash scripts to submit jobs in batch
- **Topology awareness**: on an 8-GPU machine, run `nvidia-smi topo -m`, read NV12, SYS, NODE and friends in the matrix, and say which pairs use NVLink vs PCIe
- **Collective intuition**: sketch data-flow diagrams for AllReduce, AllGather, and ReduceScatter, state each op's communication-volume formula (e.g. Ring AllReduce ≈ 2(N-1)/N × data size), and explain why the NVLink vs IB bandwidth gap decides which parallel strategies can cross machines

---

## 💻 Layer 1: CUDA programming and operator optimization

This layer bridges hardware and software: it turns high-level math into machine instructions the GPU can run efficiently. Before you write kernels, you must understand GPU hardware architecture—like writing a work manual for factory workers only after you know how many workers there are, how the shop floor is laid out, and where the shelves sit.

### 1.1 Knowledge points

**GPU hardware architecture**

Understanding GPU hardware is the prerequisite for writing good CUDA kernels. Picture a GPU as a **super factory with thousands of simple workers**—each worker (CUDA Core) only does basic add/mul, but there are so many of them working at once that throughput dwarfs a small shop with a few senior engineers (CPU cores).

- **GPU core architecture**: SM (Streaming Multiprocessor), Tensor Core, CUDA Core—how they differ and cooperate
- **Mainstream GPU specs**: A100 / H100 / H200 / B200 compute, memory bandwidth, HBM capacity, and new low-precision formats each generation (Hopper FP8, Blackwell FP4)
- **Memory Wall**: why memory bandwidth often hurts more than compute—like a chef who is blazing fast with knife and heat, but the ingredient conveyor is too slow, so most of the time is spent waiting for food to arrive
- **Memory hierarchy**: registers > shared memory > L1/L2 Cache > HBM > host memory. The core of kernel writing is maximizing reuse in this hierarchy and cutting accesses to slow storage

**CUDA programming basics**

- Programming model: Grid / Block / Thread hierarchy, thread index calculation
- Memory model: global, shared, registers, constant—traits and usage
- Key ideas: Warp (32 threads as the minimum scheduling unit; the GPU issues instructions per Warp, like an army moving by squad), Bank Conflict, Coalesced Access, Occupancy
- Core intuition: "memory access pattern decides runtime speed"

**Common operator implementation and optimization**

- Reduce: multiple parallel reduction implementations and optimizations (Warp Shuffle, multi-level reduce)
- GEMM: blocking, vectorization, Shared Memory Tiling, using Tensor Cores
- Softmax: Online normalizer calculation, efficient Softmax kernels
- Operator fusion: merge many small ops into one kernel to cut global memory traffic

**Attention operators**

- FlashAttention V1/V2: memory-aware exact Attention via tiling to cut HBM traffic—like solving a big tabletop puzzle in small chunks you bring to hand, instead of dumping every piece onto the table at once
- FlashAttention-3: further utilization gains on Hopper
- Flash-Decoding / FlashDecoding++: Attention acceleration for the Decode phase
- FlashInfer: customizable Attention engine for Serving, with composable formats and heterogeneous KV storage
- Paged-KV Attention kernels: vLLM's early `paged_attention` CUDA kernel, and how vLLM V1 reads paged KV Cache directly via backends like FlashAttention / FlashInfer

**AI compilers**

- Triton: OpenAI's open-source GPU language that lowers the bar for writing efficient operators
- TVM / XLA: graph optimization and codegen
- `torch.compile`: PyTorch 2.x compile mode; understand Graph Breaks and performance gains

### 1.2 Recommended materials

**Site articles:** [CUDA programming quick start](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/cuda编程入门指南/) · [CUDA programming intro (Chapter 1)](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第1章-cuda编程入门/) · [CUDA performance optimization basics (Chapter 2)](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第2章-cuda性能优化基础/) · [Reduce operator optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/31-cuda-reduce算子优化/) · [GEMM operator performance optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化/) · [Naive Softmax optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/51-cuda-softmax朴素实现优化/) · [Online Softmax](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现/) · [FlashAttention V1](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解/) · [FlashAttention V2](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解/) · [Triton programming model](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/71-triton编程框架/)

| Type | Resource | Notes |
|------|------|------|
| Official docs | NVIDIA GPU architecture whitepapers (Ampere / Hopper / Blackwell) | SM, Tensor Core, HBM design |
| Intro tutorial | Xiaoxiao Jiang: minimal CUDA programming intro | Zero-to-CUDA for beginners |
| Official docs | NVIDIA CUDA Programming Guide | Authoritative CUDA reference |
| Reduce | PeakCrosser: CUDA Reduce operator optimization | Thorough Reduce implementation and optimization summary |
| Practice platform | [CUDA hand-written operators · interview practice](https://kkcocoa.github.io/cuda-operator-interview/) | Graded hand-written practice covering memory access, Reduce, Softmax, Attention, GEMM, with hints, checklists, and an in-browser editor |
| GEMM | Meng Yuan: from zero to CUDA GEMM optimization | GEMM tutorial from basic blocking to extreme tuning |
| GEMM | MegEngine Bot: ultimate CUDA matmul optimization guide | Systematic GEMM optimization reference |
| Softmax | Online normalizer calculation for softmax | Original Online Softmax paper (NVIDIA); algorithmic base of FlashAttention |
| Softmax | OneFlow: how to implement an efficient Softmax CUDA kernel | Softmax kernel engineering practice |
| Operator fusion | Cheng Cheng: how OneFlow became the world's fastest DL framework | Operator fusion ideas and methods |
| Attention | FlashAttention V1 Paper | Milestone memory-aware Attention paper |
| Attention | FlashAttention V2 Paper | Better parallelism and tiling |
| Attention | FlashAttention-3 Paper | Further optimization on Hopper |
| Attention | Flash-Decoding tech report (Stanford CRFM) | Decode-phase Attention acceleration |
| Attention | FlashInfer Paper + Repo | Composable Attention engine |
| Walkthrough | Meng Yuan: illustrated FlashAttention V1/V2 series | Illustrated intro for beginners |
| Walkthrough | Fang Jiarui: understanding the PagedAttention CUDA implementation | Illustrated walkthrough of the vLLM PagedAttention kernel |
| Compiler | Official Triton tutorials | A new paradigm for GPU programming |
| Compiler | PyTorch profiling torch.compile | Catch Graph Breaks, compile gains and costs |
| Tool | Nsight Systems User Guide | CPU–GPU interaction analysis; check whether the host is the bottleneck |
| Tool | Nsight Compute Profiling Guide | Kernel-level drill-down for SM / Memory / Tensor Core bottlenecks |

### 1.3 Checkpoints

Hands-on work is the only standard for this layer; talking about it does not count:

- **Hardware-parameter intuition**: for an H100, without looking things up, state the order of magnitude of HBM capacity (80GB), HBM bandwidth (~3.35TB/s), L2 size (50MB), shared-memory ceiling (228KB/SM), and explain why "memory bandwidth" usually becomes the bottleneck before "compute"
- **Bandwidth estimate**: given an AllReduce of e.g. 2GB of gradients, estimate the theoretical time difference on NVLink 4.0 (450GB/s unidirectional per GPU, 900GB/s bidirectional) vs PCIe Gen5 x16 (64GB/s unidirectional)
- **Reduce trilogy**: start from naive global-memory atomic add and write a Reduce Sum kernel; then shared memory + tree reduction to kill atomics; finally Warp Shuffle to drop shared memory; run all three in Nsight Compute and explain where each step actually saves work
- **Bank Conflict intuition**: hand-write a 32x32 matrix-transpose kernel with 32-way bank conflict, then add a padding column to remove it, and use Nsight Compute's Shared Memory panel to verify conflicts drop from dozens to 0
- **GEMM tiling**: implement Shared Memory Tiling GEMM on 1024x1024 and hit ≥50% of cuBLAS—this is where you really feel "why access pattern decides everything"
- **FlashAttention whiteboard derivation**: without the paper, sketch FlashAttention tiling—for V1, outer loop over KV blocks, inner over Q blocks (V2 swaps to outer Q, inner KV); each tile does QK^T → scale → mask → softmax → PV in SRAM with online softmax to avoid two passes; the key is explaining why extra memory falls from O(N^2) to O(N) (no materialized N×N matrix), and why HBM traffic falls from Θ(N^2) to Θ(N^2·d^2/M) (M = SRAM size; d^2 ≪ M, still grows with N^2 but with a much smaller constant)
- **Triton hands-on**: implement a fused Softmax kernel in Triton (follow the official tutorial), compare correctness and performance with native PyTorch, and feel how Triton's block-level model differs from CUDA's thread-level model
- **Profiling in practice**: capture one training-iteration trace with Nsight Systems and say whether GPU idle gaps come from CPU preprocessing, communication waits, or kernel launch overhead; open a kernel report in Nsight Compute and read the SOL (Speed of Light) panel to judge memory-bound vs compute-bound

---

## 🏋️ Layer 2: Distributed training

When parameters outgrow single-GPU memory, distributed training is the only path—and the core job of large-model training teams. Training a hundred-billion-parameter model is like **hand-copying a tens-of-thousands-page encyclopedia**—one person would never finish. Data parallelism photocopies the book so each person copies different chapters and then merges; tensor parallelism splits each page into columns so each person copies only their columns; pipeline parallelism has the first person pass chapter one to the second and immediately start the next batch. How you split, pass, and merge is the core problem.

### 2.1 Knowledge points

**Model architecture evolution (distributed view)**

Layer 0 already covered standard Transformer structure. Here the focus is variants that directly change sharding strategy and communication cost:

- Attention variants: MHA → MQA → GQA → MLA. MQA/GQA shrink KV Cache by sharing KV heads; MLA (DeepSeek V2) compresses further with low-rank projection—these variants directly affect TP sharding and inference memory planning
- FFN variants: Mixture of Experts (MoE / DeepSeekMoE). Sparse activation turns parallel strategy from "shard the matrix" into "place the experts," adding Expert Parallelism as a new axis

**Optimizers**

The optimizer is the "steering wheel" of training—after gradients are computed, how parameters update depends entirely on it. Understanding optimizer state is the direct prerequisite for ZeRO memory optimization and mixed precision: ZeRO-1 shards optimizer state, and the FP32 master weights in mixed precision live in the optimizer too.

- **SGD**: the simplest optimizer; only one gradient per parameter, no extra state. Like stepping in the steepest direction—simple, but easy to bounce in a valley
- **Momentum SGD**: adds momentum (first moment) on top of SGD, so each parameter stores a momentum buffer. Like giving a downhill ball inertia so small bumps do not stop it
- **Adam / AdamW**: the de facto standard for large-model training. Each parameter keeps **two states**—first moment (EMA of gradients) and second moment (EMA of squared gradients), remembering both "which way" and "how bumpy the road is." AdamW fixes Adam's weight-decay implementation and is the mainstream choice today
- **Optimizer-state memory cost**: the key to understanding ZeRO. For AdamW + mixed precision, each parameter needs: FP32 parameter copy (4B) + FP32 first moment (4B) + FP32 second moment (4B) = 12 bytes/parameter. Optimizer state for a 7B model is ~84GB, far above the ~14GB of FP16 parameters—hence ZeRO starts by carving up optimizer state
- **Large-batch optimizers (LAMB / LARS)**: data parallelism scales effective batch size linearly. When batch size grows from hundreds to tens of thousands, plain Adam's learning rate is hard to tune and training tends to diverge. LAMB (Layer-wise Adaptive Moments) scales each layer's parameters with an independent trust-region factor so large-batch training stays stable

**Data parallelism**

- DP (DataParallel): basic data parallelism, single process, multi-GPU
- DDP (DistributedDataParallel): multi-process data parallelism; understand AllReduce gradient sync
- FSDP (Fully Sharded Data Parallel): PyTorch-native ZeRO-3

**Model parallelism (3D parallelism)**

- Tensor parallelism (TP): shard matmul along a chosen dim across GPUs; communication-heavy, usually single-node
- Pipeline parallelism (PP): place different layers on different GPUs and pass data like a pipeline. The core issue is pipeline bubble; GPipe, 1F1B, Interleaved 1F1B all compress bubbles
- Sequence parallelism (SP): shard along the sequence dim; pairs with TP to cut activation memory

**Long-sequence and MoE parallelism**

- Context parallelism (CP): Attention compute and activation memory grow with sequence length squared; Ring Attention, DeepSpeed-Ulysses, Megatron Context Parallel split one long sequence across GPUs
- Expert parallelism (EP): MoE places different experts on different GPUs; tokens go to the right expert via All-to-All from the Router; load balance is the hard part. Real training often combines EP with DP, TP, and PP

**Memory optimization**

- ZeRO family (DeepSpeed): shard training state across GPUs—like roommates each storing only their own furniture and borrowing when needed, freeing space:
  - ZeRO-1: shard optimizer state
  - ZeRO-2: shard optimizer state + gradients
  - ZeRO-3: shard optimizer state + gradients + parameters (trade communication for memory)
- Mixed precision: FP16 / BF16 / FP8 training to cut memory and compute. Like using scratch paper (low precision) for speed day to day, and formal answer sheets (high precision) only on the critical steps
- Gradient accumulation: simulate larger effective batch size under limited memory
- Activation Checkpointing (rematerialization): trade compute for memory—keep only some activations and recompute when needed. Like not keeping every scrap of exam draft on the desk; remember key intermediates and re-derive when required

**Training frameworks**

- Megatron-LM: reference implementation of TP and PP
- DeepSpeed: core ZeRO implementation plus a rich training-optimization toolkit
- PyTorch FSDP: native distributed training path

### 2.2 Recommended materials

**Site articles:** [Distributed training overview](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/11-分布式训练总论/) · [Environment setup and distributed launch](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/12-环境搭建与分布式启动/) · [Collective communication primitives in depth](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/21-集合通信原语详解/) · [Optimizers and memory-cost analysis](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/31-优化器原理与显存开销分析/) · [Data parallelism in depth](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/41-数据并行详解/) · [PyTorch data parallelism from principles to practice](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/42-pytorch-数据并行从原理到实战/)

| Type | Resource | Notes |
|------|------|------|
| Paper | DeepSeek V2 technical report | MLA attention |
| Paper | DeepSeekMoE Paper | MoE architecture design |
| Tutorial | Mixture of Experts (MoE) explained | MoE intro |
| Walkthrough | Su Jianlin: from MHA, MQA, GQA to MLA | Attention-variant evolution |
| Paper | Megatron-LM Paper | Milestone paper on TP and PP |
| Paper | ZeRO Paper (DeepSpeed) | Core memory-optimization method |
| Paper | Ring Attention Paper | Context parallelism by circulating KV blocks |
| Paper | DeepSpeed-Ulysses Paper | Context parallelism via All-to-All between sequence and head dims |
| Docs | Official DeepSpeed docs | ZeRO config and usage |
| Docs | PyTorch DDP / FSDP tutorials | Native distributed training intro |

### 2.3 Checkpoints

The core of this layer is **being able to do the math and run the code**:

- **Memory ledger**: for a 7B model (e.g. LLaMA-2-7B), without notes, estimate ~14GB for FP16 parameters and ~84GB for Adam optimizer state (FP32 param copy + first/second moments at 28GB each), then decide whether an 80GB card can hold the full training state or whether you must use ZeRO
- **ZeRO breakdown**: if asked "what's the real difference between ZeRO-2 and ZeRO-3," answer in one breath: ZeRO-2 ReduceScatters gradients in backward (each GPU keeps only its shard), then AllGathers after the update; parameters stay fully replicated, communication volume 2Ψ same as DDP; ZeRO-3 also shards parameters, so forward/backward AllGather params and discard after use, communication 3Ψ (~1.5× DDP), but per-GPU model-state memory drops to 1/N
- **DDP conversion**: take a single-GPU PyTorch training script and turn it into working multi-GPU DDP within 30 minutes—`init_process_group`, `DistributedSampler`, model wrap, gradient sync—without leaning too hard on docs
- **3D parallel topology**: for a 64-GPU cluster (8 nodes × 8 GPUs), design TP=8 (intra-node), PP=4 (inter-node), DP=2, draw the topology with NVLink vs IB labeled, and explain why TP cannot cross machines (bandwidth is not enough)
- **Mixed-precision rationale**: answer "BF16 and FP16 are both 16-bit—why do large models prefer BF16?"—BF16 has a wider exponent (8 bits vs 5), dynamic range close to FP32, less overflow/underflow, and usually needs no Loss Scaling

---

## 🚀 Layer 3: Inference and deployment

Training is only the first step of a long march. Getting the model to serve users fast and cheap is what industry cares about most. If training is "teaching the model knowledge," inference is "sending it into the exam hall"—what matters most is **answer speed** and **how many examinees you serve at once**, and jotting computed intermediates on scratch paper (KV Cache) avoids recomputation and speeds things up a lot.

### 3.1 LLM inference basics

**Knowledge points**

- Two phases of LLM inference: Prefill (process the input) and Decode (generate token by token)
- KV Cache: the "memory assassin" of autoregressive generation; understand its lifecycle and fragmentation. KV Cache is like exam scratch paper—write down intermediates already computed so each new token need not start from scratch, but too much scratch paper fills the whole desk
- Key metrics: TTFT (time to first token), TPOT (time per output token), throughput (token/s), P50/P95 tail latency

**Recommended materials**

**Site articles:** [LLM inference basics](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础/) · [vLLM quick start](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/12-vllm快速入门/) · [From Transformer to LLM autoregressive generation](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解/)

| Type | Resource | Notes |
|------|------|------|
| Walkthrough | Linlang Amu: illustrated LLM inference—KV Cache | KV Cache explained in depth |
| Survey | Towards Efficient Generative LLM Serving: A Survey | CMU survey of LLM inference (algorithms + systems) |

**Checkpoints**

Put yourself in the shoes of an engineer who just got an inference requirement and answer:

- **Two-phase intuition**: when a colleague asks "why is Prefill fast but Decode slow," answer immediately: Prefill processes the whole prompt at once—large matrices, high compute utilization (compute bound); Decode emits one token per step—matrices collapse to vectors, most time waiting on memory to move KV Cache (memory bound)
- **KV Cache arithmetic**: for LLaMA-2-7B (32 layers, 32 heads, head_dim=128), context 4096, batch_size=16, FP16 storage, hand-compute KV Cache total = 2 × 32 × 32 × 128 × 4096 × 16 × 2B ≈ 32GB, then judge how much of an 80GB card is left for weights and temporary buffers
- **Pipeline breakdown**: split one inference request end to end into tokenize → prefill (GEMM-heavy) → decode loop (token-by-token, memory bound) → sampling (Top-p/Top-k) → detokenize, and point out which stage is most likely the bottleneck under high concurrency (usually decode-phase KV Cache bandwidth)

### 3.2 Inference engines

**Knowledge points**

- PagedAttention: vLLM's virtual-memory paging idea for managing KV Cache and fixing fragmentation—like an OS cutting memory into fixed-size pages so you no longer need one huge contiguous block
- Continuous Batching: dynamic batching that admits requests as they arrive, vs traditional static batching. Static is like a tour bus that leaves only when full, so early arrivals wait; Continuous Batching is more like ride-sharing—join as you arrive, and when someone gets off a new passenger fills the seat
- Prefix Cache / RadixAttention: reuse computed KV Cache for repeated prefixes
- Chunked Prefill: process long prompts in chunks to reduce prefill interference with decode

**Mainstream inference frameworks**

| Framework | Core features | Fit |
|------|---------|---------|
| vLLM | PagedAttention, Continuous Batching, Prefix Cache | General inference serving; active community |
| SGLang | RadixAttention, cFSM structured-output acceleration | Complex agents, multi-turn generation, structured output |
| TensorRT-LLM | Paged KV, Inflight Batching, deep hardware optimization | Peak performance; NVIDIA ecosystem |

**Recommended materials**

**Site articles:** [PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) · [Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) · [Prefix Cache and RadixAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) · [Chunked Prefill and unified scheduling](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) · [Attention backends and graph optimization](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) · [nano-vllm source walkthrough (Chapter 3)](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/第3章-nano-vllm源码解读/)

| Type | Resource | Notes |
|------|------|------|
| Paper | vLLM Paper (PagedAttention) | Milestone inference-engine paper |
| Source | vLLM GitHub Repo | Full set of production features |
| Docs | Official vLLM docs | Deployment and configuration |
| Walkthrough | Meng Yuan: vLLM source series (architecture / scheduler / BlockManager) | Deep dive into vLLM internals |
| Paper | SGLang Paper | RadixAttention + frontend language |
| Source | SGLang GitHub Repo | Engineering implementation |
| Docs | Official TensorRT-LLM docs | Production-oriented |
| Paper | Orca Paper | Original Continuous Batching paper |
| Walkthrough | Chi Guodong Bu Tu Guodong Pi: Continuous Batching | Chinese Continuous Batching walkthrough |
| Walkthrough | DefTruth: illustrated vLLM Prefix Cache | Long-form Prefix Cache explanation |

**Checkpoints**

Inference engines are where engineering lands; checkpoints emphasize "can use, can choose, can debug":

- **End-to-end deploy**: take a 7B model, deploy OpenAI-compatible APIs with both vLLM and SGLang, and with the same load script (fixed concurrency, input/output lengths) produce a comparison table of TTFT, TPOT, and throughput
- **Continuous Batching principle**: explain to a non-technical colleague—Static Batching waits for a whole batch to finish before taking new requests, so short requests are dragged by long ones; Continuous Batching lets finished requests leave and new ones join anytime, and GPU utilization can climb from 30% to 80%+
- **Full KV Cache management path**: from allocation, use, and fragmentation, through PagedAttention's virtual/physical page mapping, to how Prefix Cache reuses existing KV blocks via hash match—draw the full data flow
- **Selection decision**: when the boss asks "which inference framework should we use," give a structured answer—throughput and community → vLLM; multi-turn / Agent / structured output → SGLang (RadixAttention + cFSM); absolute lowest latency and willingness to invest in adaptation → TensorRT-LLM

### 3.3 Quantization

**Knowledge points**

- W8A8 (SmoothQuant): move activation outlier pain onto weights; engineering-friendly. Quantization is like compressing a high-res photo into a thumbnail—fewer bits for weights saves memory and bandwidth, at some cost to detail (precision)
- Weight-only INT4 (GPTQ / AWQ): quantize only weights to 3/4-bit to cut memory and bandwidth; at inference, fused kernels like Marlin dequantize on-chip
- KV Cache quantization (KIVI / FP8 KV Cache): KIVI does 2-bit quantization channel-wise on Key and token-wise on Value; FP8 KV Cache is the more common engineering option in engines like vLLM, with clear wins for long context
- FP8 quantization: native Tensor Core support on Hopper and Ada; a balance point between precision and performance
- NVFP4 / MXFP4: 4-bit float formats native to Blackwell, with micro-block scales (one Scale shared by every 16 or 32 elements) to control error
- Quantization decision tree:

```
Goal: save memory? save bandwidth? raise throughput?
├─ General, engineering-friendly → W8A8 (SmoothQuant); prefer FP8 on Hopper+
├─ Save more memory/bandwidth → INT4 weight-only (AWQ/GPTQ); on Blackwell consider NVFP4
└─ Long context / high concurrency → KV Cache quantization (FP8 KV / KIVI)
```

**Recommended materials**

**Site articles:** [Quantization basics](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) · [W8A8 and SmoothQuant](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) · [Weight-only INT4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) · [KV Cache quantization](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) · [FP8 and NVFP4/MXFP4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) · [Quantization selection and vLLM practice](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/)

| Type | Resource | Notes |
|------|------|------|
| Paper | SmoothQuant Paper | Classic W8A8 scheme |
| Paper | GPTQ Paper | Weight-only PTQ 3/4-bit |
| Paper | AWQ Paper | 4-bit quantization guided by activation distributions |
| Paper | KIVI Paper | 2-bit KV Cache quantization |
| Paper | Marlin Paper (FP16xINT4) | High-performance INT4 matmul kernel |
| Code | TensorRT-LLM quantization toolchain | FP8/INT4/AWQ/SmoothQuant integration |
| Code | vLLM quantization support | GPTQ/AWQ/FP8/INT8 entry points |

**Checkpoints**

Quantization looks simple and tunes like mysticism; the checkpoint is understanding trade-offs, not memorizing recipes:

- **Scheme choice**: the team must ship a 70B model on only two A100 80GB cards. Quick judgment: FP16 weights ≈140GB; two cards =160GB total, only ~20GB left for KV Cache and buffers after weights, so concurrency stalls; W8A8 compresses weights to ~70GB, leaving ample KV Cache under two-card TP; INT4 weight-only ≈35GB fits on one card, but you must assess quality loss and kernel efficiency. Also A100 Tensor Cores do not support FP8, so FP8 is off the table on this machine
- **Hands-on validation**: load FP16 and AWQ-INT4 of the same model in vLLM or TensorRT-LLM, run the same benchmark prompts, report throughput comparison plus a few human-compared generations as a reusable eval write-up
- **Counter-intuition**: when someone says "lower bit-width is always faster," explain why INT4 can be slower than INT8 in some cases—small-batch Decode is memory bound, so W4A16 moves half the weight bytes and is usually faster; once batch grows into compute bound, W4A16 must dequantize back to FP16 then use FP16 Tensor Cores, while W8A8 runs INT8 Tensor Cores directly (~2× FP16 compute), so INT8 wins
- **Incident triage**: after quantization, quality drops sharply—list a triage checklist: some layers have huge activation outliers (need per-channel or SmoothQuant), calibration data is unrepresentative, some special structures (e.g. MoE gates) dislike low-bit quantization

### 3.4 Speculative Decoding

**Knowledge points**

- Speculative Sampling: classic frame—small model (Draft) "guesses" several tokens in batch; large model (Target) verifies in one pass with unbiased distribution. Like having an intern draft quickly and a senior editor review in one go: correct guesses ship, wrong ones get fixed on the spot—faster than the editor writing every word from scratch
- Medusa: no external Draft model; multiple Decoding Heads predict multi-token candidates and verify in parallel
- EAGLE-2 / EAGLE-3: train a light Draft head on Target hidden states. EAGLE-2 builds a candidate tree dynamically from Draft confidence; EAGLE-3 fuses multi-layer features and simulates inference during training, raising acceptance further
- MTP (Multi-Token Prediction): DeepSeek-V3's training-time multi-token prediction module; usable directly as Draft at inference
- N-gram / Suffix: no Draft model; match candidates from the prompt or already-generated text—good for code editing, RAG, and other high-repetition settings
- Block Verification: lift token-level verify to block-level joint verify for more speedup

**Recommended materials**

**Site articles:** [Speculative decoding core principles](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) · [Draft models and draft-free methods](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) · [Medusa, EAGLE, and MTP](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) · [Gains, boundaries, and limits](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) · [vLLM speculative decoding in practice](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/)

| Type | Resource | Notes |
|------|------|------|
| Paper | Speculative Decoding (Leviathan et al.) / Speculative Sampling (Chen et al.) | Two classic papers that independently proposed Draft + Verify |
| Paper | Medusa Paper + Repo | Multi-head decoding without a Draft model |
| Paper | EAGLE-2 Paper | Dynamic Draft Tree |
| Paper | EAGLE-3 Paper | Multi-layer feature fusion and train-time testing |
| Paper | Block Verification Paper | Block-level joint verification |
| Code | Speculative support in vLLM / TensorRT-LLM | Production references |

**Checkpoints**

The essence of Speculative Decoding is "space for time, parallelism for serial work"; checkpoints focus on correctness and gain boundaries:

- **Correctness guarantee**: if an interviewer asks whether Speculative Decoding changes the model's output distribution, answer: no—rejection sampling keeps accepted tokens strictly on the target distribution. Correct Draft guesses are used as-is; wrong ones are corrected by sampling from the target–draft probability gap, mathematically equivalent to sampling from the target alone
- **Measured gains**: with vLLM speculative decoding, benchmark code generation (high acceptance) vs open chat (low acceptance), and explain why the former speeds up a lot (code tokens are predictable; Draft hits often) while the latter gains little or goes negative
- **Where it does not pay**: list at least three weak cases—high-temperature sampling crashes Draft hit rate; large batch sizes already leave little room and the extra Draft forward steals compute; Draft far weaker than Target so acceptance rate < 50%
- **Engineering coupling risk**: point out conflicts with other optimizations—stacking with quantization can drop Draft precision and further cut acceptance; stacking with Continuous Batching means different requests have different speculative lengths and scheduling gets harder

### 3.5 System architecture: Prefill/Decode disaggregation

**Knowledge points**

- Core problem: mixing Prefill and Decode in one batch couples resources and interferes, blowing up tail latency. Like one chef cooking both fast stir-fries (Decode, low latency) and slow braises (Prefill, heavy compute)—they drag each other down. Disaggregation splits the quick-service and slow-braise stations with dedicated cooks
- DistServe (OSDI'24): systematically argues for and implements Prefill/Decode disaggregation, scheduling around goodput
- Splitwise (ISCA'24): assign Prefill and Decode to different GPU pools to optimize throughput and cost
- TaiChi (2025): unify aggregation and disaggregation for optimal goodput under different SLO mixes
- Goodput: effective throughput that meets SLO, distinct from raw QPS

**Recommended materials**

| Type | Resource | Notes |
|------|------|------|
| Paper | DistServe (OSDI'24) | Systematic case for Prefill/Decode disaggregation |
| Paper | Splitwise (ISCA'24) | Phase Splitting design |
| Paper | TaiChi (2025) | Unified aggregation / disaggregation framework |
| Tutorial | Disaggregated Inference: 18 Months Later | Why disaggregation became the default |
| Tutorial | MLC microserving | Programmable API for cross-engine orchestration |

**Checkpoints**

Prefill/Decode disaggregation is a systems-level optimization; checkpoints focus on "why split" and "what new problems appear after splitting":

- **Interference quantification**: on a mixed-batching inference service, craft a scenario—a few ultra-long-prompt prefill requests arrive with many short decode requests—and show with metrics that decode P95 TPOT is slowed 3–5× by prefill; that is the motive for disaggregation
- **Goodput concept**: when the boss says "QPS is high, why are users still complaining," explain goodput—the share of requests that meet SLO (e.g. TTFT < 500ms and TPOT < 50ms) is the real quality metric; raw QPS is not user experience
- **Resource ratio derivation**: given a workload (avg prompt 2000 tokens, avg output 500), estimate Prefill vs Decode compute ratio and derive a sensible Prefill GPU pool : Decode GPU pool ratio (e.g. 1:3 or 1:4)
- **Risk list**: list new problems disaggregation introduces—network bandwidth pressure moving KV Cache from Prefill to Decode nodes (for LLaMA-2-7B, one 2048-token sequence's KV ≈1GB; even 200Gb/s IB needs ~40ms in theory), scheduler queue complexity, idle waste when Prefill/Decode load is unbalanced

### 3.6 Performance analysis and Benchmark

**Knowledge points**

- Core metrics: QPS, TTFT, TPOT, token/s, P50/P95 tail latency
- Profiling toolchain:
  - `torch.profiler`: official PyTorch profiler for ops and shapes
  - Nsight Systems: full CPU–GPU interaction traces
  - Nsight Compute: kernel-level drill-down
- Load tools: `vllm bench serve` (vLLM's built-in online serving bench), GenAI-Perf / AIPerf (NVIDIA LLM load tools; AIPerf documents migration from GenAI-Perf), Triton Perf Analyzer
- Authoritative benchmark: MLPerf Inference (Datacenter), unified rules and definitions
- Regression gate: every change ships the same metrics comparison table; regressions block merge

**Recommended materials**

| Type | Resource | Notes |
|------|------|------|
| Tool | torch.profiler docs | Starting point for PyTorch performance analysis |
| Tool | GenAI-Perf / AIPerf | One-stop TTFT/TPOT/token throughput load tests |
| Tool | Triton Inference Server Quickstart | Containerized online-inference baseline |
| Benchmark | MLPerf Inference (Datacenter) | Authoritative benchmark entry |
| Walkthrough | MLPerf Inference v5.0 LLM task notes | Trends in low-latency LLM benchmarks |

**Checkpoints**

Performance analysis is not "knowing the tools"; the core is a **repeatable, comparable, traceable** engineering system:

- **Full metric set**: every performance report must include at least six metrics—QPS, TTFT (P50/P95), TPOT (P50/P95), end-to-end throughput (token/s), peak GPU memory, GPU utilization; missing any one can hide a bottleneck
- **Reproducible Benchmark**: model, batch size, input/output lengths, concurrency, hardware SKU, driver version all live in one config file; anyone with that file can reproduce your numbers, and two runs differ by only one variable
- **Regression localization**: after a commit, TPOT P95 regresses 15%—use `git bisect` to narrow to the commit, then compare Nsight Systems traces before/after: slower kernel, extra sync, or changed batch scheduling?
- **Ship gate**: define simple enforceable performance gates for the team—e.g. "TPOT P95 regression >5% blocks merge," "memory growth >10% needs an attached analysis"—and wire them into CI

### 3.7 Distributed inference and production deployment

**Knowledge points**

- Distributed inference: when one GPU cannot hold the model or throughput is short, spread with TP, PP, DP, and EP across multi-GPU / multi-node. Large MoE models often combine "Attention on DP, experts on EP"; multi-node deploy can use orchestrators like Ray
- Production deployment: containers and Kubernetes, Prometheus metrics and observability, autoscaling by queue length or KV Cache occupancy, Prefix-Cache-aware load balancing, capacity planning
- On-device inference: model export, on-device runtimes (e.g. ExecuTorch, llama.cpp), heterogeneous CPU/GPU/NPU execution, plus quantization and memory optimization

**Recommended materials**

**Site articles:** [On-device inference basics: from model export to heterogeneous hardware](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第9章-端侧推理/91-端侧推理基础/)

| Type | Resource | Notes |
|------|------|------|
| Docs | Official vLLM docs (Parallelism and Scaling) | TP/PP/DP/EP config and multi-node deploy |
| Docs | Official ExecuTorch docs | PyTorch's official on-device inference runtime |

**Checkpoints**

- **Parallelism choice**: for a 70B Dense model on one 8×H100 machine, explain how TP=8 vs TP=4 + DP=2 differ in single-request latency, total throughput, and KV Cache capacity
- **Autoscaling signals**: explain why LLM services should not scale only on GPU utilization, and should combine queued requests, KV Cache usage, TTFT, and related signals

### 3.8 Optimization selection decision tree

Inference optimization is not "pile on every technique and call it done"—**find the bottleneck first, then treat the symptom**. Like seeing a doctor: they do not prescribe on sight; they ask where it hurts, run tests, then write a prescription. The decision tree below is the "intake flow" for inference optimization—start from the symptom and narrow to a concrete plan.

**Step 1: Confirm the main symptom**

After you get a performance issue, first run a baseline with load tools (`vllm bench serve`, GenAI-Perf / AIPerf, or a custom script), see which metric misses the mark, then match:

```
What is the symptom?
│
├─ [A] TTFT too high (users wait too long for the first token)
├─ [B] TPOT too high (tokens trickle out; feels stuttery)
├─ [C] Not enough memory (OOM or cannot raise concurrency)
└─ [D] Tail latency P95/P99 out of control (average is fine, spikes are bad)
```

**[A] TTFT too high—first token never arrives**

TTFT is dominated by Prefill. Prefill processes the whole prompt at once and is typically compute-bound. Diagnosis path:

| Sub-symptom | Diagnosis | Recommended plan |
|--------|---------|---------|
| Very long prompt (thousands to tens of thousands of tokens) | Check whether prefill time grows linearly or super-linearly with prompt length | Chunked Prefill to avoid one long prefill monopolizing the GPU; upgrade to more efficient GEMM kernels |
| CPU or scheduling is the bottleneck | Capture a Nsight Systems trace; see if GPU idle gaps come from the host | Hunt tokenizer / preprocessing CPU cost; consider a more mature scheduling engine |
| Many requests share the same prefix (e.g. system prompt) | Measure prompt-prefix duplication rate | Enable Prefix Cache (vLLM) or RadixAttention (SGLang) to reuse computed KV Cache |

**[B] TPOT too high—generation feels like squeezing toothpaste**

Decode emits one token per step; matrices collapse to vector ops and are typically memory-bound. Three optimization directions:

| Sub-symptom | Diagnosis | Recommended plan |
|--------|---------|---------|
| KV Cache movement is the bandwidth bottleneck | In Nsight Compute, check whether Attention kernel memory throughput nears HBM bandwidth | FlashAttention / FlashInfer to cut useless memory traffic; improve KV Cache layout |
| Fundamental limit of serial token generation | Single-request TPOT already near the theoretical floor, users still want more | Speculative Decoding (EAGLE-3 / MTP / N-gram)—"guess + verify" to break serial bottlenecks |
| Concurrency too low; GPU compute underfed | GPU utilization <60%, each batch has few requests | Continuous Batching + higher concurrency to grow batch size and compute density |

**[C] Not enough memory—OOM or concurrency capped**

Memory is the scarcest resource in inference, split between model weights and KV Cache. Choose by which dominates:

| Memory hog | Diagnosis | Recommended plan |
|---------|---------|---------|
| KV Cache (long context or high concurrency) | Compare weight memory vs KV Cache memory (see the 3.1 method) | PagedAttention for fragmentation; KV Cache quantization (FP8 KV, KIVI 2-bit) to compress storage |
| Model weights themselves | FP16 weights already exceed one GPU | INT4 weight-only (AWQ/GPTQ) can compress weights to ~1/4; W8A8 (SmoothQuant or FP8) to ~1/2, with better compute efficiency at large batch |

**[D] Tail latency P95/P99 out of control—average OK, spikes bad**

This is the hardest case: often not one slow stage, but system-level contention and bad scheduling:

| Sub-symptom | Diagnosis | Recommended plan |
|--------|---------|---------|
| Prefill and Decode requests interfere | Under mixed load, compare pure Decode vs mixed TPOT P95; >2× gap confirms interference | Prefill/Decode disaggregated deploy (DistServe / Splitwise) with separate GPU pools |
| Very strict SLO; need globally optimal scheduling | After plain disaggregation some requests still time out, or requests have differentiated SLOs | TaiChi-style unified scheduling that dynamically allocates Prefill/Decode by SLO priority |

**Practical usage tips**

In practice, problems often have more than one symptom—TTFT and memory may both miss. Suggested order:

1. **Fix OOM first** (memory), or the service cannot even start
2. **Then optimize TTFT**, because first-token latency dominates user feel
3. **Then improve TPOT and throughput**, usually trading latency vs throughput
4. **Finally handle tail latency**, which needs systems-architecture changes and the most investment

---

## 🧭 Getting started as a newcomer

### Recommended learning path

Facing such a large stack, **clear the foundations of each layer in the shortest time, then go deep in one direction for your target role**—you do not need to master every layer before moving up:

**Foundation stage (0–3 months)**

1. Clear every Layer-0 checkpoint: programming languages, math, Transformer architecture, PyTorch training loop, communication topology basics
2. Learn CUDA programming basics; write simple Reduce / GEMM kernels
3. Try distributing training to two GPUs with PyTorch DDP and watch memory and communication change

**Deep dive (3–6 months)**

1. Closely read four milestone papers against code:
   - Megatron-LM (TP and PP)
   - ZeRO (DeepSpeed core)
   - FlashAttention (memory-aware algorithm)
   - vLLM (PagedAttention, KV Cache)
2. Read through a small inference engine (e.g. nano-vllm; see this site's [source walkthrough](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/第3章-nano-vllm源码解读/)) and connect scheduling, KV Cache management, and model execution
3. Contribute to open source (vLLM, DeepSpeed, SGLang)—operator optimizations or feature modules
4. Master inference optimizations such as quantization and Speculative Decoding

**Engineering practice (6+ months)**

1. Deploy 10B / 100B-parameter models on GPU clusters and optimize end-to-end performance
2. Build a full performance-analysis and regression system
3. Study frontier system architectures such as Prefill/Decode disaggregation
4. Track the latest iterations (FP8/FP4 low precision, RDMA network tuning, new-architecture bring-up, etc.)

**Direction tip**: large-scale pretraining roles concentrate in the few teams with 10k-GPU clusters, and seats are limited; inference optimization, operator development, and training/inference framework roles have broader demand. If job hunting is the goal, prioritize shipping showable projects in those directions.

### Core mental model

While learning AI Infra, keep one trade-off frame in mind:

| Optimization | What you give up | What you gain |
|---------|-----------|-----------|
| ZeRO | Communication bandwidth | Memory |
| Activation Checkpointing | Compute time | Memory |
| Quantization | Precision | Memory + bandwidth + throughput |
| Speculative Decoding | Extra compute (Draft forward + rejected-token verify) | Decode latency |
| Prefill/Decode disaggregation | System complexity + KV migration cost | Tail latency + goodput |
| FlashAttention | Implementation complexity | Memory + speed |

---

## 📚 References

### Papers

- **Attention Is All You Need** (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **Megatron-LM** (Shoeybi et al., 2019): [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2019): [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
- **FlashAttention V1** (Dao et al., 2022): [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- **FlashAttention V2** (Dao, 2023): [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- **FlashAttention-3** (Shah et al., 2024): [https://arxiv.org/abs/2407.08691](https://arxiv.org/abs/2407.08691)
- **Flash-Decoding** (Stanford CRFM, 2023): [https://crfm.stanford.edu/2023/10/12/flashdecoding.html](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- **FlashInfer** (Ye et al., 2025): [https://arxiv.org/abs/2501.01005](https://arxiv.org/abs/2501.01005)
- **vLLM / PagedAttention** (Kwon et al., 2023): [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **SGLang** (Zheng et al., 2023): [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- **Orca** (Yu et al., 2022): [https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)
- **DistServe** (Zhong et al., OSDI'24): [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)
- **Splitwise** (Patel et al., ISCA 2024): [https://arxiv.org/abs/2311.18677](https://arxiv.org/abs/2311.18677)
- **SmoothQuant** (Xiao et al., 2022): [https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)
- **GPTQ** (Frantar et al., 2022): [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ** (Lin et al., 2023): [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- **KIVI** (Liu et al., 2024): [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)
- **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2022): [https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
- **Accelerating LLM Decoding with Speculative Sampling** (Chen et al., 2023): [https://arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318)
- **Medusa** (Cai et al., 2024): [https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- **EAGLE-2** (Li et al., 2024): [https://arxiv.org/abs/2406.16858](https://arxiv.org/abs/2406.16858)
- **EAGLE-3** (Li et al., 2025): [https://arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840)
- **Block Verification Accelerates Speculative Decoding** (Sun et al., 2024): [https://arxiv.org/abs/2403.10444](https://arxiv.org/abs/2403.10444)
- **Ring Attention** (Liu et al., 2023): [https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)
- **DeepSpeed Ulysses** (Jacobs et al., 2023): [https://arxiv.org/abs/2309.14509](https://arxiv.org/abs/2309.14509)
- **Online normalizer calculation for softmax** (Milakov & Gimelshein, 2018): [https://arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)
- **DeepSeek V2 technical report**: [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- **DeepSeekMoE**: [https://arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)
- **DeepSeek V3 technical report** (includes MTP, FP8 training): [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- **Towards Efficient Generative LLM Serving: A Survey** (CMU): [https://arxiv.org/abs/2312.15234](https://arxiv.org/abs/2312.15234)

### Tutorials and videos

- **The Illustrated Transformer** (Jay Alammar): [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- **3Blue1Brown: Essence of linear algebra**: [https://www.3blue1brown.com/topics/linear-algebra](https://www.3blue1brown.com/topics/linear-algebra)
- **Andrej Karpathy: Let's build GPT from scratch**: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **MIT The Missing Semester**: [https://missing.csail.mit.edu/](https://missing.csail.mit.edu/)

### Open-source projects and tools

- **vLLM** GitHub: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **SGLang** GitHub: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
- **DeepSpeed** GitHub: [https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- **Megatron-LM** GitHub: [https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- **FlashAttention** GitHub: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **FlashInfer** GitHub: [https://github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **TensorRT-LLM** GitHub: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **Triton** GitHub: [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)
- **AIPerf** GitHub: [https://github.com/ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf)

### Official documentation

- **NVIDIA CUDA Programming Guide**: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **NVIDIA NCCL docs**: [https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- **Nsight Systems User Guide**: [https://docs.nvidia.com/nsight-systems/UserGuide/](https://docs.nvidia.com/nsight-systems/UserGuide/)
- **Nsight Compute**: [https://docs.nvidia.com/nsight-compute/ProfilingGuide/](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- **NVIDIA Deep Learning Performance Guide**: [https://docs.nvidia.com/deeplearning/performance/](https://docs.nvidia.com/deeplearning/performance/)
- **Official DeepSpeed docs**: [https://www.deepspeed.ai/](https://www.deepspeed.ai/)
- **Official PyTorch tutorial (60 Minute Blitz)**: [https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- **PyTorch DDP tutorial**: [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- **PyTorch FSDP tutorial**: [https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- **Official vLLM docs**: [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **Official TensorRT-LLM docs**: [https://nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
- **Official Triton tutorials**: [https://triton-lang.org/main/getting-started/tutorials/](https://triton-lang.org/main/getting-started/tutorials/)
- **MLPerf Inference**: [https://mlcommons.org/benchmarks/inference-datacenter/](https://mlcommons.org/benchmarks/inference-datacenter/)
- **GenAI-Perf**: [https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/)
