---
title: "AIInfraGuide article update log"
description: "A record of every content update to the AIInfraGuide knowledge base, so readers can track what changed"
tags: ["Announcement", "Update log"]
sourceHash: "01537d99535d6a9a"
---

This page keeps a running record of AIInfraGuide knowledge-base content updates, newest first, so you can see what changed.

---

## 2026-11-07

### New content

**Module 4 · Inference optimization** Chapter 6 "Distributed inference and large-model deployment" is live. It goes from the four parallel cuts used in inference to single-node and multi-node vLLM deploy plus side-by-side experiments. Config is based on vLLM v0.32.0, with 10 new hand-drawn-style diagrams:

- **Article:** [6.1 Inference parallelism overview: not fitting in memory and not being fast enough are two different problems](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/61-推理并行策略总览/) | Use a memory ledger to separate "does not fit" from "not fast enough", compare what TP, PP, DP, and EP cut and what they sync, give an order for choosing parallel degrees, and a runnable estimate script

- **Article:** [6.2 Tensor-parallel inference: split one layer into pieces, sync twice per layer](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/62-张量并行推理/) | Derive per-layer AllReduce volume and latency, analyze KV head replication under GQA, the NVLink vs cross-node bandwidth gap, how TP degree affects latency and throughput, and vLLM's Custom AllReduce

- **Article:** [6.3 Pipeline-parallel inference: cut by layer, fill the pipeline with multiple batches](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/63-流水线并行推理/) | Clarify what moves between stages and how much, why Decode needs multiple in-flight batches, how uneven layer splits are handled, and compare TP×PP mixes with a cost model

- **Article:** [6.4 Data parallel and expert parallel: more replicas, and sending tokens to experts](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行/) | From multi-replica Dense models to MoE attention DP + expert EP, covering two All-to-Alls, idle sync, load-imbalance simulation and EPLB, DeepEP, and DeepSeek-V3 deploy plans

- **Article:** [6.5 Multi-node deploy: spread one service across machines with Ray or multiprocessing](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/65-多节点部署/) | Ray cluster setup and rank layout, multi-process `--nnodes` usage, MoE multi-node DP + EP launch, InfiniBand and GPUDirect RDMA config, plus a communication self-check script and troubleshooting table

- **Article:** [6.6 vLLM distributed deploy in practice: same 8 GPUs, three cuts measured together](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/66-vllm分布式部署实战/) | On one 8-GPU machine compare Llama-3.1-70B TP8, TP4×PP2, and TP4×DP2: estimate KV capacity before launch, check logs and metrics after, sweep concurrency with `vllm bench serve`; also cover using DCP to remove long-context KV replication

### Changes

- **Article:** [Chapter 6 Distributed inference and large-model deployment](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/第6章-分布式推理/) | Chapter intro rewritten to match the actual content, with an information-freshness note and links to each section

---

## 2026-11-01

### Changes

**Module 4 · Inference optimization** Chapter 3 "Reading the nano-vllm source" — 6 articles revised: source walkthrough aligned to the nano-vllm main commit from 2026-04-26, 3 new diagrams, all 8 existing Mermaid diagrams redrawn as hand-drawn-style SVG, and some inaccurate wording fixed:

- **Article:** [3.1 Project overview and four-layer architecture](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/31-项目全景与四层架构/) | Project info table updates line counts, versions and deps, Star count; four-layer architecture and request-lifecycle diagrams become SVG; notes that Prefill and Decode are not mixed in one batch

- **Article:** [3.2 Config, entry points, and one full generate path](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/32-配置入口与推理主循环/) | Adds constraints for `tensor_parallel_size`, `ignore_eos`, and similar params; notes nano-vllm does not support greedy sampling, and that the README example skips a chat template; notes `step` returns `num_tokens` with sign distinguishing phases

- **Article:** [3.3 Model skeleton and core ops](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/33-模型与核心算子/) | Tensor-parallel split diagram becomes SVG; new Sampler section; explains where the 2 all_reduces per layer come from, and how KV head count limits TP degree

- **Article:** [3.4 KV Cache and BlockManager](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/34-kvcache与blockmanager/) | Prefix-cache section split into hash chain, registration, and match; new prefix-hit allocation diagram, match flowchart becomes SVG; notes the last block is not looked up, the difference between hit-in-use and free blocks, and reuse truncation from whole-block granularity

- **Article:** [3.5 Scheduler and continuous batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/35-scheduler与连续批处理/) | Sequence-state and scheduling-loop diagrams become SVG; explains the three `break`s in the prefill branch one by one, head-of-line blocking from strict FCFS; notes preemption uses recompute, and when `while ... else` runs

- **Article:** [3.6 ModelRunner, CUDA Graph, and performance benchmarks](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/36-modelrunner引擎循环与性能/) | New `slot_mapping` compute diagram and CUDA Graph tier diagram; `run` flowchart and full-chapter call-chain diagram become SVG; notes 6-D KV Cache layout, tier caps vs `max_num_seqs`; performance section adds test conditions and where conclusions apply

**Module 4 · Inference optimization** Chapter 4 "Quantization" — 6 articles revised: 7 new diagrams, 4 existing flat illustrations redrawn in hand-drawn style, existing Mermaid diagrams all become SVG, and some wording fixed against vLLM v0.32.0 source:

- **Article:** [4.1 Quantization basics](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | New four-granularity diagram; redrawn symmetric/asymmetric number line; adds 4-bit symmetric encoding range; PyTorch quantization references switched to torchao docs

- **Article:** [4.2 W8A8 and SmoothQuant](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | Redrawn paired-scale diagram; offline quantization flowchart becomes SVG

- **Article:** [4.3 Weight-only INT4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | New W4A16 pack and on-chip dequant diagram, GPTQ/AWQ comparison diagram; Marlin pipeline diagram becomes SVG

- **Article:** [4.4 KV Cache quantization](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | Redrawn KIVI grouping diagram; notes Value residual region is quantized out token-by-token under a sliding window

- **Article:** [4.5 FP8 and NVFP4/MXFP4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | New FP8 vs INT8 scale-distribution comparison; redrawn FP4 microblock scale diagram; online FP8 path adds `fp8_per_channel`

- **Article:** [4.6 Quantization selection and vLLM practice](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | Selection flowchart becomes SVG; fixes online FP8 load path (v0.32.0 builds on meta first and converts layer by layer; it does not load full 16-bit weights at once)

**Module 4 · Inference optimization** Chapter 5 "Speculative Decoding" revised: 1 new diagram, all 4 existing Mermaid diagrams redrawn as hand-drawn-style SVG, and some wording fixed against vLLM v0.32.0 source:

- **Article:** [5.1 Speculative decoding core ideas](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | Draft + Verify one-round flowchart becomes SVG, with accept/correct/reject positions marked for γ = 4; new residual-sampling distribution-preserving diagram, using section 7 numbers to show the two paths sum to p

- **Article:** [5.2 Draft models and draft-free methods](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | N-gram lookup flowchart becomes SVG, marking vLLM's longest-then-earliest match rule; fixes where `ngram_gpu` applies (both V1 and V2 Model Runner); notes default when Suffix has no candidate cap

- **Article:** [5.3 Medusa, EAGLE, and MTP](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | Draft Tree diagram becomes SVG, same tree as section 8 experiments, with tree Mask and depth-numbered Position IDs; notes Block acceptance only in V2 Model Runner; fixes MTP conditions that avoid version mismatch

- **Article:** [5.4 Benefit boundaries and limits](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | Scheduling-loop diagram becomes SVG; adds an example of different step lengths for requests in the same batch; fixes Adaptive Verification support range (not only DSpark) and the N-gram/Suffix benefit description under high QPS

- **Article:** [5.5 vLLM speculative decoding in practice](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | Per-position accept metrics use the actual exposed `..._per_pos_total` name and explain the counters; adds `--speculative-config` short form `-sc`

---

## 2026-10-29

### Changes

**Module 4 · Inference optimization** Chapter 2 "Core inference-engine techniques" fully revised: 5 articles gain 9 new diagrams, default params and source walkthroughs aligned to vLLM v0.32.0, and some outdated or inaccurate wording fixed:

- **Article:** [2.1 PagedAttention: manage the KV Cache the way an OS manages memory](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) | New Block Table mapping diagram and fragmentation comparison; `gpu_memory_utilization` default updated to 0.92; notes V1 preemption only recomputes and no longer uses Copy-on-Write; fixes `free_blocks` free-queue enqueue logic

- **Article:** [2.2 Continuous Batching: dynamic batching so the GPU stops idling](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) | New Static vs Continuous Batching timeline comparison; replaces uncited utilization numbers with idle-slot examples and Orca paper data; preemption pseudocode becomes the retry loop from the source

- **Article:** [2.3 Prefix Cache and RadixAttention: reuse shared-prefix KV](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) | New chained block-hash diagram and Radix Tree example; adds `cache_salt`, hash seeds, and RadixAttention refcounting, leaf-first LRU, and scheduling by cache hit

- **Article:** [2.4 Chunked Prefill and unified Token Budget scheduling](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) | New chunked timeline and Token-budget bin-packing diagrams; notes vLLM chunk length is set by remaining budget; adds `long_prefill_token_threshold` and official `max_num_batched_tokens` tuning advice

- **Article:** [2.5 Attention backends and graph optimization](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) | New Kernel launch vs CUDA Graph comparison and Piecewise CUDA Graph diagram; adds default backends by GPU architecture; fixes CUDA Graph padding rules and V1 default `FULL_AND_PIECEWISE` mode

---

## 2026-10-26

### Changes

**Module 4 · Inference optimization** Chapter 4 "Quantization" — 6 articles revised, framework content aligned to vLLM v0.32.0, each article gains an information-freshness note, and chapter structure reorganized:

- **Article:** [4.1 Quantization basics: Scale, granularity, and PTQ/QAT](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | Rewrites quantize/dequantize with a "ruler with limited ticks" metaphor; bit-width, matrix axes, and granularity become separate sections; calibration gets its own chapter covering range choice and static vs dynamic calibration

- **Article:** [4.2 W8A8 quantization: how SmoothQuant handles activation outliers](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | Equivalent scaling gets formulas and an intuitive single-dot-product derivation; the smoothing-coefficient section becomes how to choose migration strength α; adds offline flow, benefits and fit conditions, plus a hands-on dot-product example

- **Article:** [4.3 Weight-only INT4: GPTQ, AWQ, and Marlin](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | Adds GPTQ cost and effect, AWQ's idea of scaling instead of mixed precision; new vLLM kernel selection, quantization-tool choice, and debug order

- **Article:** [4.4 KV Cache quantization: KIVI and FP8 cache](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | New KV footprint estimate; FP8 KV Cache section rewritten for v0.32.0 covering format choice, Scale calibration, skipping sensitive layers, and storage vs compute precision; notes the old `calculate_kv_scales` param was removed

- **Article:** [4.5 FP8 and NVFP4/MXFP4: how low-bit floats work](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | Adds E2M1's 8 magnitudes, microblock shape notation and Scale's own format; compares NVFP4 two-level Scale vs MXFP4 one-level Scale; new usage in vLLM v0.32.0

- **Article:** [4.6 Quantization selection and vLLM practice](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | Separates quantization method, checkpoint format, and Kernel as three layers; explains vLLM quantization entry points; experiment baseline updated to v0.32.0

**Module 4 · Inference optimization** Chapter 5 "Speculative Decoding" — 5 articles revised, config baseline upgraded from vLLM v0.31.0 to v0.32.0, with new MTP content:

- **Article:** [5.1 Speculative decoding core ideas: draft a stretch, then let the big model verify](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | Rewrites one full Draft + Verify round; adds verification when "Draft gives tokens only, not probabilities"

- **Article:** [5.2 Draft models and draft-free methods: cheaper ways to guess better](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | New sections "matching the Target matters more than a high own score" and parallel Draft; unifies definitions for accept rate, mean accept length, and candidate coverage

- **Article:** [5.3 Medusa, EAGLE, and MTP: reuse big-model info for better drafts](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | New section 7 on MTP covering the shift from pretrain objective to inference speedup, real results, shape in vLLM, and exploring one forward for a whole draft block; EAGLE-2 notes confidence is only used to allocate budget

- **Article:** [5.4 Benefit boundaries of speculative decoding: high accept rate can still be slower](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | Adds speedup conditions and diminishing returns of speculation length; new load-adaptive dynamic speculation length; structured output enters the stop-condition discussion

- **Article:** [5.5 vLLM speculative decoding in practice: measure config, accept rate, and performance together](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | Baseline upgraded to vLLM v0.32.0; new fourth EAGLE-3 experiment group and MTP attach conditions; collection script switches to per-request accept metrics, plus common-symptom debug order and next-round tuning directions

---

## 2026-10-12

### New content

**Module 4 · Inference optimization** Chapter 3 "Reading the nano-vllm source" is live. Using the ~1200-line minimal inference engine [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) as the spine, it walks module by module through how continuous batching, PagedAttention, FlashAttention, tensor parallelism, and CUDA Graph are organized in real code:

- **Article:** [3.1 Project overview and four-layer architecture](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/31-项目全景与四层架构/) | See nano-vllm's role and directory layout; one four-layer architecture diagram covers the full data path from user request to token output

- **Article:** [3.2 Config, entry points, and one full generate path](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/32-配置入口与推理主循环/) | Starting from example.py, break down Config/SamplingParams/LLM/LLMEngine and see how the generate main loop drives add_request, step, and postprocess

- **Article:** [3.3 Model skeleton and core ops](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/33-模型与核心算子/) | Layer-by-layer Qwen3: RMSNorm, RoPE, SwiGLU, parallel Linear and tensor-parallel splits, vocab-parallel Embedding/LMHead, and Attention's KV write plus the two FlashAttention paths

- **Article:** [3.4 KV Cache and BlockManager](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/34-kvcache与blockmanager/) | Blocked KV management: Block structure, PagedAttention's block idea, xxhash prefix cache, block alloc/free, and Sequence state machine plus block_table

- **Article:** [3.5 Scheduler and continuous batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/35-scheduler与连续批处理/) | Scheduling core: Waiting/Running dual queues, Prefill/Decode split scheduling, Token Budget, chunked prefill, and preemption plus recompute when memory is short

- **Article:** [3.6 ModelRunner, CUDA Graph, and performance benchmarks](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/36-modelrunner引擎循环与性能/) | Sequence tensorization, global Context injection, KV Cache memory splits, CUDA Graph capture and replay, plus official benchmarks vs vLLM

### Changes

- The old *vLLM quick start* moves under Chapter 1 as [1.2 vLLM quick start](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/12-vllm快速入门/), as a hands-on entry to inference basics.

---

## 2026-10-08

### New content

**Module 4 · Inference optimization** Chapter 4 "Quantization" — 6 articles live:

- **Article:** [4.1 Quantization basics: Scale, granularity, and PTQ/QAT](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | Understand quantize/dequantize from numeric mapping; bit-width, Scale, zero-point, granularity, plus calibration and PTQ vs QAT

- **Article:** [4.2 W8A8 quantization: how SmoothQuant handles activation outliers](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | From Activation Outliers derive SmoothQuant equivalent scaling; smoothing coefficients, INT8 GEMM, and the offline deploy flow

- **Article:** [4.3 Weight-only INT4: GPTQ, AWQ, and Marlin](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | GPTQ error compensation, AWQ activation-aware scaling, and how Marlin turns INT4 storage into inference gains

- **Article:** [4.4 KV Cache quantization: KIVI and FP8 cache](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | Streaming constraints of KV Cache quantization, KIVI's different Key/Value grouping, and vLLM FP8 KV Cache calibration plus benefit bounds

- **Article:** [4.5 FP8 and NVFP4/MXFP4: how low-bit floats work](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | Integer vs float quantization; FP8 range and precision; NVFP4/MXFP4 microblock scaling and hardware run conditions

- **Article:** [4.6 Quantization selection and vLLM practice](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | Pick quantization candidates by bottleneck; relate vLLM quantization formats to Kernels; run FP16 vs AWQ INT4 throughput and generation-quality comparisons

---

## 2026-10-07

### New content

**Module 4 · Inference optimization** Chapter 5 "Speculative Decoding" — 5 articles live:

- **Article:** [5.1 Speculative decoding core ideas: draft a stretch, then let the big model verify](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | From the autoregressive serial bottleneck derive Draft + Verify, accept probability, and the residual distribution; prove the final output follows the Target distribution

- **Article:** [5.2 Draft models and draft-free methods: cheaper ways to guess better](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | Compare an independent small model, N-gram Prompt Lookup, and Suffix Decoding; Tokenizer compatibility, history match, candidate coverage, and accept-rate tradeoffs

- **Article:** [5.3 Medusa and EAGLE: reuse big-model info for better drafts](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | Medusa multi-head prediction, EAGLE feature-level drafts, EAGLE-2 dynamic trees and EAGLE-3 training improvements; Tree Attention and path commit

- **Article:** [5.4 Benefit boundaries of speculative decoding: high accept rate can still be slower](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | Build a performance ledger from mean commit length and full-round time; shared constraints from batching, long context, quantization, scheduling, and streaming latency

- **Article:** [5.5 vLLM speculative decoding in practice: measure config, accept rate, and performance together](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | Against vLLM v0.31.0, start plain Decode, N-gram, and independent Draft in turn; collect accept metrics, quality output, and latency/throughput

---

## 2026-09-15

### New content

**Module 4 · Inference optimization** adds Chapter 12 "On-device inference":

- **Article:** [12.1 On-device inference basics: from model export to heterogeneous hardware](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-端侧推理/121-端侧推理基础/) | Build a systematic view of on-device inference hardware, model formats, runtimes, quantization, memory, and Benchmark, and walk a minimal deploy path with ExecuTorch/XNNPACK

---

## 2026-07-09

### New content

**Module 4 · Inference optimization** Chapters 1 and 2 articles live:

- **Article:** [1.1 LLM inference basics](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础/) | One article on the underlying logic of LLM inference: Prefill/Decode phases, the KV Cache memory ledger, metrics such as TTFT/TPOT, and using Roofline to explain why Decode is Memory Bound

- **Article:** [2.1 PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) | Deep dive into vLLM's core technique PagedAttention: borrow virtual-memory paging, use a Block Table to remove KV Cache fragmentation, and support prefix sharing plus Copy-on-Write

- **Article:** [2.2 Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) | Deep dive into Continuous Batching: Iteration-level Scheduling lets requests join as they arrive and leave when done, pulling GPU utilization from about 30% to 80%+

- **Article:** [2.3 Prefix Cache and RadixAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) | Deep dive into prefix cache: how vLLM's hash-based automatic prefix cache reuses shared-prefix KV blocks, and how SGLang's RadixAttention uses a Radix Tree for more efficient prefix sharing

- **Article:** [2.4 Chunked Prefill and unified scheduling](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) | Deep dive into Chunked Prefill: chunk long-prompt Prefill so it stops interfering with Decode; how vLLM V1's unified Token-budget scheduler flattens the Prefill/Decode boundary

- **Article:** [2.5 Attention backends and graph optimization](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) | Deep dive into vLLM's pluggable Attention backends (FlashAttention/FlashInfer/FlashMLA/Triton) and graph opts: CUDA Graph and torch.compile cut Decode-phase CPU launch overhead

**Module 3 · Distributed training** collective communication, optimizer, and data-parallel chapter articles live:

- **Article:** [2.1 Collective communication primitives in depth](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/21-集合通信原语详解/) | AllReduce/ReduceScatter/AllGather/All-to-All input/output semantics, Ring AllReduce algorithm, and how to quantify one communication's cost

- **Article:** [3.1 Optimizers: from SGD to AdamW and memory-cost analysis](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/31-优化器原理与显存开销分析/) | Optimizer evolution from SGD to AdamW, optimizer-state memory cost and why FP32 master weights matter, plus LARS/LAMB for large-batch settings

- **Article:** [4.1 Data parallelism in depth](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/41-数据并行详解/) | Evolution of DP/DDP/FSDP, gradient sync, and math for communication volume / memory ledgers; the real difference between AllReduce and parameter sharding

- **Article:** [4.2 PyTorch data parallelism from principle to practice](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/42-pytorch-数据并行从原理到实战/) | Full PyTorch data-parallel stack: communication primitives, DDP, FSDP parameter sharding, and multi-node multi-GPU practice with complete code samples

---

## 2026-07-07

### New content

- **Article:**  [1.1 Distributed training overview: the memory ledger and five parallel strategies](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/11-分布式训练总论/) | Hand-compute the training memory ledger; overview of five parallel strategies and how to choose

- **Article:**  [1.2 Environment setup and distributed launch](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/12-环境搭建与分布式启动/) | rank/local_rank/world_size, torchrun launch and rendezvous, NCCL debugging, minimal DDP script

---

## 2026-07-06

### New content

Updated the distributed-training tutorial outline:
| Chapter | Main content |
|------|----------|
| Chapter 1 Distributed training overview | Why distributed training, training memory ledger, five parallel strategies, environment setup and DDP launch |
| Chapter 2 Collective communication primitives | AllReduce/ReduceScatter/AllGather/All-to-All semantics, Ring algorithm, communication-volume analysis |
| Chapter 3 Optimizers | SGD/Adam/AdamW evolution, optimizer-state memory cost, large-batch optimizers LAMB/LARS |
| Chapter 4 Data parallelism | DataParallel, DistributedDataParallel, FSDP |
| Chapter 5 ZeRO series | ZeRO-1/2/3 sharding and communication cost, ZeRO-Offload/Infinity heterogeneous memory offload |
| Chapter 6 Tensor parallel and sequence parallel | Megatron Column/Row Parallel Linear, where communication is inserted, sequence-parallel activation memory opts |
| Chapter 7 Pipeline parallel | Bubble problem, GPipe, 1F1B, Interleaved schedule |
| Chapter 8 Mixed precision and memory opts | FP16/BF16/FP8 mixed precision, gradient accumulation, Activation Checkpointing |
| Chapter 9 Long-sequence training and context parallel | Attention's O(s²) problem, Ring Attention, DeepSpeed-Ulysses, Megatron Context Parallel |
| Chapter 10 MoE parallel | Router mechanism, Expert Parallelism All-to-All, EP×DP×TP×PP mixes and load balance |
| Chapter 11 3D parallel and hybrid strategies | TP×PP×DP×EP×CP topology design, communication-domain mapping, rank layout, cluster selection |


---

## 2026-06-25

### New content

- **Article:**  [6.1 FlashAttention V1 in depth](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解) | Deep dive into FlashAttention V1 principles and implementation: how Tiling + Online Softmax cuts Attention's extra memory from $O(N^2)$ to $O(N)$, sharply reduces HBM traffic, and speeds up without accuracy loss

- **Article:**  [6.2 FlashAttention V2 in depth](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解) | This this article deep-dives FlashAttention V2's core changes vs V1: swap inner/outer loop order, improve work distribution inside the thread block, and cut non-matmul ops

---

## 2026-06-06

### New content

- **Article:**  [5.2 CUDA Online Softmax implementation and optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现) | From algorithm derivation to implementing and optimizing an Online Softmax CUDA Kernel — core prerequisite knowledge for understanding FlashAttention

---

## 2026-06-05

### New content

- **Article:**  [5.1 CUDA Softmax naive implementation and optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/51-cuda-softmax朴素实现优化) | From a naive implementation, gradually introduce Safe Softmax, block-level parallelism, Warp Shuffle, vectorized loads, and related opts

---

## 2026-06-02

### New content

- **Article:** [4.1 CUDA GEMM operator performance optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化) | From a naive implementation through Block-Warp-Thread three-level tiling, vectorized loads, Bank Conflict removal, double buffering, and more — a full methodology for CUDA GEMM optimization

---

## 2026-05-18

### New content

- **Article:** [3.1 CUDA Reduce operator optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/31-cuda-reduce算子优化) | Reduce is one of the most basic GPU ops and one of the best for learning parallel thinking. From the simplest implementation, this article gradually adds Warp-level primitives, vectorized loads, multi-element processing, and more — each step with performance comparisons and explanations so you really understand how to write a fast Kernel.

---

## 2026-05-16

### New content

- **Module 3 · Distributed training** adds [Chapter 2: Optimizers](/AIInfraGuide/distributed/模块三-分布式训练/第2章-优化器) | Follow SGD → Momentum → Adam → AdamW evolution and internal state makeup; hand-compute optimizer memory cost (AdamW is 75% of training memory); know when large-batch optimizers LAMB/LARS fit; foundation for later ZeRO memory opts and mixed-precision training.

### Structure changes

- **Module 3 · Distributed training** chapters renumbered: old Chapters 2–7 become 3–8; new Chapter 2 "Optimizers" sits between the overview and data parallelism
- Synced the AI Infra learning path and the Module 3 chapter list in the README

---

## 2026-05-13

### New content

- **Article:** [2.4 Synchronization and atomic operations](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/24-同步与原子操作) | Correct sync is the base of bug-free parallel programs. This article covers CUDA block sync `__syncthreads()`, Warp-level sync, Memory Fence, and when to use atomics, their cost, and how to optimize — so you can write high-performance parallel code without giving up correctness.

---

## 2026-05-12

### New content

- **Article:** [2.2 Memory-access optimization](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/22-内存访问优化) | Memory-access efficiency is the biggest lever on CUDA Kernel performance. Coalesced Access principles and checks, shared-memory Bank Conflict causes and Padding fixes, and vectorized loads (float4/int4) to raise bandwidth use.

- **Article:** [2.3 Occupancy and resource allocation](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/23-occupancy与资源分配) | Occupancy measures active Warps on an SM vs the theoretical max — a core CUDA Kernel tuning metric. Definition, how to compute it, three limiters (registers/shared memory/Block size), and why higher Occupancy is not always better — the real goal is balance between latency hiding and resource use.

---

## 2026-05-11

### New content

- **Article:** [2.1 Warp and the execution model](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型) | Deep dive into the GPU's core execution unit — the Warp

---
## 2026-05-09

### New content

- **Article:** [CUDA memory model](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/13-cuda内存模型) | Global memory, shared memory, registers, constant memory, and unified memory — traits, fit cases, and optimization tips

- **Article:** [First useful Kernel: vector add](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/14-第一个实用kernel) | A full vector-add case that walks the whole CUDA programming path

---
## 2026-05-08

### New content

- **Article:** [1.1 CUDA development environment setup](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/11-cuda开发环境搭建)
- **Article:** [1.2 CUDA programming model](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型)

---
## 2026-05-07

### New content

- **Article:** [✅ CUDA programming quick-start guide 🔥](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/cuda编程入门指南/)


---
## 2026-04-28

### New content

- **Article:** [🔥 Cluster communication networks and NCCL: the communication backbone of distributed training](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/collective-communication-primer/)

---
## 2026-04-27

### New content

- **Article:** [5.1 NVIDIA GPU architecture evolution: from Volta to Blackwell](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpunvidia-gpu-evolution/)

---
## 2026-04-26

### New content

- **Article:** [🔥 GPU basics: from hardware architecture to AI compute](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics/)

---
## 2026-04-25

### New content

- **Article:** [🔥 PyTorch framework quick start](/AIInfraGuide/prerequisites/模块一-前置知识/pyroch/pytorch框架入门)

---
## 2026-04-24

### New content

- **Article:** [3.9 Tokenization and word embeddings](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/39-tokenization与词嵌入)


---
## 2026-04-23

### New content

- **Article:** [3.8 From Transformer to LLM autoregressive generation in depth](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解)

---

## 2026-04-21

### New content

- **Article:** [3.6 LayerNorm and residual connections in depth](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/36-layernorm与残差连接深入理解)
- **Article:** [3.7 Transformer Decoder Block full breakdown](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/37-transformer-decoder-block完整解析)

---

## 2026-04-20

### New content

- **Article:** [3.4 Transformer feed-forward network (FFN) in depth](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/34-transformer前馈网络ffn深入理解)
- **Article:** [3.5 Transformer positional encoding in depth](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/35-transformer位置编码深入理解)

---

## 2026-04-18

### New content

- **Article:** [3.3 Self-Attention mechanism in depth](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/33-Self-Attention机制深入理解)

---

## 2026-04-17

### New content

- **Interview section**: **180+** real interview sessions covering **40+** companies, organized by tier. [Browse online →](https://caomaolufei.github.io/AIInfraGuide/interview)
- Companies include ByteDance, Alibaba, Tencent, Baidu, Kuaishou, Meituan, Ant Group, NVIDIA, MiniMax, NIO, XPeng, Li Auto, and more.
- **Community entry**: new [WeChat discussion group]((/AIInfraGuide/about/author)), welcome to join

### Content improvements

- Updated the README and added a project intro image

---

## 2026-04-16

### Major update

- **Knowledge-base framework overhaul**: finer four learning modules, page layout adjusted
  - **[Module 1 · Prerequisites](/AIInfraGuide/prerequisites)** (6 chapters): [Programming language basics](/AIInfraGuide/prerequisites/模块一-前置知识/第1章-编程语言基础), [Math basics](/AIInfraGuide/prerequisites/模块一-前置知识/第2章-数学基础), [Transformer architecture in depth](/AIInfraGuide/prerequisites/模块一-前置知识/第3章-Transformer架构详解), [PyTorch framework](/AIInfraGuide/prerequisites/模块一-前置知识/第4章-PyTorch框架), [GPU hardware overview](/AIInfraGuide/prerequisites/模块一-前置知识/第5章-GPU硬件概论), [Collective communication basics](/AIInfraGuide/prerequisites/模块一-前置知识/第6章-集合通信基础)
  - **[Module 2 · CUDA programming and operator optimization](/AIInfraGuide/cuda)** (8 chapters): [CUDA intro](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第1章-CUDA编程入门), [Performance optimization basics](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第2章-CUDA性能优化基础), Reduce practice, GEMM practice, Softmax practice, [Attention operators](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第6章-Attention算子), [AI compilers](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第7章-AI编译器), [Performance analysis toolchain](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第8章-性能分析工具链)
  - **[Module 3 · Distributed training](/AIInfraGuide/distributed)** (7 chapters): [Distributed training overview](/AIInfraGuide/distributed/模块三-分布式训练/第1章-分布式训练总论), [Data parallelism](/AIInfraGuide/distributed/模块三-分布式训练/第2章-数据并行), [ZeRO series](/AIInfraGuide/distributed/模块三-分布式训练/第3章-ZeRO系列), [Tensor parallel and sequence parallel](/AIInfraGuide/distributed/模块三-分布式训练/第4章-张量并行与序列并行), [Pipeline parallel](/AIInfraGuide/distributed/模块三-分布式训练/第5章-流水线并行), [3D parallel and hybrid training strategies](/AIInfraGuide/distributed/模块三-分布式训练/第6章-其他显存优化技术), [Training framework practice](/AIInfraGuide/distributed/模块三-分布式训练/第7章-训练框架实战)
  - **[Module 4 · Inference optimization](/AIInfraGuide/inference)** (8 chapters): Inference basics, [Core inference-engine techniques](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术), Inference frameworks, [Quantization](/AIInfraGuide/inference/模块四-推理优化/第4章-量化), [Speculative Decoding](/AIInfraGuide/inference/模块四-推理优化/第5章-Speculative-Decoding), [PD disaggregation](/AIInfraGuide/inference/模块四-推理优化/第6章-PD解耦架构), [Performance analysis and Benchmark](/AIInfraGuide/inference/模块四-推理优化/第7章-性能分析与Benchmark), [Selection and end-to-end practice](/AIInfraGuide/inference/模块四-推理优化/第8章-推理优化选型与端到端实战)

### New content

- **Transformer series**:
  - [Transformer architecture intro](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/Transformer架构快速入门)
  - [3.1 Why AI Infra engineers must understand Transformer](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/31-ai-infra工程师为什么必须懂transformer)
  - [3.2 Transformer overview and code](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/32-transformer全貌及代码实现)
- **Course material**: 150+ detailed sub-articles live, covering all chapters of the four modules
- **Blog**: new [About the author](/AIInfraGuide/about/author) page

---

## 2026-04-15

### Project launch

- AIInfraGuide knowledge base officially live
- Published [AI Infra learning path](/AIInfraGuide/guides/AI Infra学习路线) overview
- Published [Understanding AI Infra from scratch](/AIInfraGuide/blog/ai-infra-introduction) starter guide

---

## 2026-04-14

### Project init

- Website framework built (Astro + Starlight)
- Repo initialized, first commit published
