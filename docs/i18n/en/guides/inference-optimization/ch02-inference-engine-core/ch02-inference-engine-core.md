---
title: "Chapter 2: Core inference engine techniques"
description: "PagedAttention, Continuous Batching, Prefix Cache, Chunked Prefill, Attention backends, and graph optimization—the five core techniques in modern inference engines"
tags: ["PagedAttention", "Continuous Batching", "Prefix Cache", "Chunked Prefill", "FlashAttention", "CUDA Graph"]
sourceHash: "99bbf47949796d3f"
---

## 📖 About this chapter

The throughput of a modern LLM inference engine comes from several core techniques working together. This chapter goes through each one and ties it back to how vLLM implements it.

**PagedAttention** borrows the idea of virtual memory paging from operating systems. A block table maps logical blocks to physical blocks, which cuts both internal and external fragmentation in the KV cache and raises GPU memory utilization a lot.

**Continuous Batching** fixes the problem in Static Batching where short requests wait on long ones. Scheduling moves down to each iteration step (iteration-level scheduling): requests join as they arrive and leave as they finish, so the batch no longer carries idle slots.

**Prefix Cache / RadixAttention** targets workloads where many requests share the same system prompt. vLLM's Prefix Cache reuses existing KV blocks through chained block hashes; SGLang's RadixAttention does longest-prefix matching on a radix tree and is stronger when the sharing forms a tree. In vLLM V1 the bookkeeping cost of prefix caching is pushed very low, and it is on by default.

**Chunked Prefill and unified token-budget scheduling** split a long prompt into chunks so a heavy prefill interferes less with decode requests. vLLM V1 goes further: it drops the Prefill/Decode split and uses one token-budget scheduler that holds both kinds of requests at once.

**Attention backends and graph optimization** cover vLLM's pluggable Attention backends (FlashAttention 2/3/4, FlashInfer, FlashMLA, Triton) and how they serve mixed Prefill/Decode batches, plus how CUDA Graph and torch.compile remove per-kernel launch overhead and ease the CPU bottleneck in decode.

> ⏰ **Currency note**: the principles of each technique follow the corresponding papers; vLLM's default parameters, the source walk-through of scheduling and KV cache management, and the CUDA Graph modes are based on v0.32.0 (released 2026-10-18).

## 📑 Sections

- [**2.1 PagedAttention: manage the KV cache the way an OS manages memory**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/): the two kinds of fragmentation under contiguous allocation, Block Table mapping, kernel addressing, CoW and V1 read-only sharing, preemption and recomputation, `KVCacheManager` and `BlockPool` source
- [**2.2 Continuous Batching: dynamic batching so the GPU stops idling**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/): idle slots in Static Batching, iteration-level scheduling, how it works with PagedAttention, V1 Scheduler queues and preemption
- [**2.3 Prefix Cache and RadixAttention: reuse shared-prefix KV**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/): chained block hashes, LRU and reference-count eviction, SGLang's Radix Tree, vLLM APC source
- [**2.4 Chunked Prefill and unified token-budget scheduling**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/): TPOT spikes from long Prefill, chunking, packing under a token budget, tuning `max_num_batched_tokens`
- [**2.5 Attention backends and graph optimization**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/): pluggable backends and defaults, mixed batches, CUDA Graph and padding, Piecewise and `FULL_AND_PIECEWISE`, torch.compile graph cuts
