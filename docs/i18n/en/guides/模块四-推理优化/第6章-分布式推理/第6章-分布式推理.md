---
title: "Chapter 6: Distributed inference and large-model deployment"
description: "Storage and communication costs of TP, PP, DP, and EP for inference, plus single-node and multi-node vLLM deploys, MoE DP+EP, DCP, and controlled experiments to pick a layout"
tags: ["Distributed inference", "Tensor parallelism", "Pipeline parallelism", "Expert Parallel", "Ray", "vLLM"]
sourceHash: "610ad9898850821c"
---

## About this chapter

When a model no longer fits on one GPU, or one replica cannot push enough throughput, inference has to split across cards and machines. Inference and training share the same parallelism vocabulary, but the goals differ: training cares about gradient sync and convergence; inference cares whether weights and the KV cache fit, and about TTFT, TPOT, and throughput.

This chapter goes from "how to slice" to "how to deploy and how to verify":

- **Choosing a strategy**: use a memory ledger to tell "does not fit" from "not fast enough," compare what TP, PP, DP, and EP each slice and what they have to sync, and give an order for picking parallel degrees.
- **TP and PP**: how much data and latency TP's two AllReduces per layer cost, KV head replication under GQA, and the need for NVLink; what PP passes between stages, why several batches must stay in flight, and how to combine TP×PP.
- **DP and EP**: multi-replica Dense models; for MoE, attention DP + expert EP, two All-to-Alls, idle sync, load imbalance and EPLB, plus DeepSeek-V3's production layout.
- **Deployment**: Ray and multi-process multi-node runtimes, rank layout, starting multi-node MoE DP + EP, InfiniBand and GPUDirect RDMA settings; then compare three layouts on one 8-GPU machine, and cover DCP for cutting long-context KV replication.

> ⏰ **Currency:** The principles do not depend on any framework. vLLM parameters, defaults, logs, and metrics here follow the v0.32.0 docs and source (released 2026-10-18); DeepEP numbers are from v1.2.1. Distributed knobs have moved a lot in recent releases; when you change versions, check the official docs.

## Sections

- [**6.1 Inference parallelism overview: not fitting and not being fast enough are different problems**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/61-推理并行策略总览/): memory ledger, storage and communication comparison across the four splits, order for choosing parallel degrees, and an estimation script
- [**6.2 Tensor-parallel inference: slice one layer into pieces, sync twice per layer**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/62-张量并行推理/): column and row splits, AllReduce volume and latency model, KV head replication, NVLink dependence, and Custom AllReduce
- [**6.3 Pipeline-parallel inference: split by layer, keep the pipe full with multiple batches**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/63-流水线并行推理/): inter-stage traffic, Decode batch queues, uneven layer splits, single-request latency, and TP×PP combinations
- [**6.4 Data parallelism and expert parallelism: more replicas, and sending tokens to experts**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行/): Dense DP, MoE Dispatch/Combine, attention DP + expert EP, EPLB, DeepEP, and DeepSeek-V3 deployment
- [**6.5 Multi-node deployment: spread one service across machines with Ray or multi-process**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/65-多节点部署/): Ray clusters and Placement Groups, multi-process `--nnodes`, multi-node DP + EP, NCCL network config and debugging
- [**6.6 Distributed vLLM in practice: same 8 GPUs, three splits side by side**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/66-vllm分布式部署实战/): memory estimates, log checks, and benchmarks for TP8, TP4×PP2, and TP4×DP2, plus DCP
