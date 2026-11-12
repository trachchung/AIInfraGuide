---
title: "Chapter 3: Reading the nano-vllm source"
description: "A ~1200-line minimal inference engine as the through-line: continuous batching, PagedAttention, FlashAttention, tensor parallelism, and CUDA Graph as they appear in real code"
tags: ["nano-vllm", "Source reading", "Inference engine", "PagedAttention", "Continuous batching", "CUDA Graph"]
sourceHash: "96bb8e19046ee354"
---

## 📖 About this chapter

Chapter 2 took the core inference techniques apart one by one: how PagedAttention manages GPU memory, how Continuous Batching fills the GPU, how Prefix Cache reuses prefixes, how Chunked Prefill unifies scheduling. How those pieces form one pipeline inside a real engine is the question this chapter answers.

The method is reading code. The sample is [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a minimal LLM inference engine open-sourced by Xingkai Yu (GitHub ID `GeeeekExplorer`; his profile lists DeepSeek and Nanjing University). The project is about 1200 lines of Python and covers the full path—config, scheduling, KV management, model forward, sampling—plus prefix caching, tensor parallelism, `torch.compile`, and CUDA Graph. Its relationship to vLLM is a bit like nanoGPT to a large training framework: small enough to read in one sitting, while keeping the backbone of a production system.

By the end you should be able to walk the source tree and say what each layer does from a string request in, through the first output token, to the finished generation—and map those pieces back to vLLM's design.

> ⏰ **Currency note**: the code analysis is based on the nano-vllm `main` branch commit from 2026-04-26 (`pyproject.toml` version 0.2.0; as of 2026-11-01 the repo had about 16.2k stars, MIT license). The example model is `Qwen/Qwen3-0.6B`. The source is still moving; if a key identifier differs, trust the latest repo version.

## 📑 Sections

- **3.1 Project overview and four-layer architecture**: where the project sits, the directory layout, and the layering "interface → engine → scheduler → execution" with the end-to-end data flow
- **3.2 Config, entry points, and one full generate path**: `Config` / `SamplingParams`, how `LLM` relates to `LLMEngine`, the README usage example, and the `generate → add_request → step → postprocess` main loop
- **3.3 Model skeleton and core operators**: vocabulary-parallel Embedding/LMHead, RMSNorm, RoPE, SwiGLU, the parallel Linear family and tensor-parallel weight sharding, Attention's KV write path and the two FlashAttention paths, temperature sampling in the Sampler, and how Qwen3 is assembled
- **3.4 KV Cache and BlockManager**: the `Block` structure, PagedAttention's blocking idea, hash-based prefix caching, block allocate/free, and the `Sequence` lifecycle
- **3.5 Scheduler and continuous batching**: Waiting/Running queues, Prefill/Decode scheduling decisions, Token Budget, preemption and recomputation
- **3.6 ModelRunner, CUDA Graph, and benchmarks**: preparing input tensors, injecting global Context, allocating KV Cache GPU memory, capturing and replaying CUDA Graphs, plus the official benchmarks versus vLLM
