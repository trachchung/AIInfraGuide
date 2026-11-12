---
title: "Chapter 1: LLM inference basics"
description: "How LLMs generate text in two phases (prefill and decode), how the KV cache works, and which metrics matter for inference performance"
tags: ["LLM inference", "Prefill", "Decode", "KV Cache", "TTFT", "TPOT", "Roofline"]
sourceHash: "06e92a77b36ae05a"
---

## About this chapter

Inference is where a trained model turns into something people use. This chapter sets up the basic concepts and the way of analyzing performance that every later optimization in this module relies on. vLLM is the running example throughout.

The section on autoregressive generation covers the two phases of LLM inference. Prefill processes the whole prompt at once and is compute bound. Decode produces one token at a time and is memory bound, which is why decode uses the GPU far less efficiently than prefill.

The KV cache is the central idea in inference optimization. It stores the keys and values that were already computed so they are not computed again. We follow a KV cache through its life (it is allocated, filled, read and freed), work out how much GPU memory it needs, and look at how it fragments.

The metrics section covers TTFT (time to first token), TPOT/ITL (time per output token), throughput, P50/P95/P99 tail latency, and goodput, the throughput that still meets your SLOs.

The section on bottleneck analysis introduces arithmetic intensity and the Roofline model. From first principles they show why decode is memory bound and cannot use the GPU's full compute, and they give us one way to reason about every optimization that follows.

The last section walks through the full request path: tokenize, prefill, the decode loop, sampling and detokenize. For each step we look at what kind of computation it does and where it can become the bottleneck, and compare that with how vLLM actually runs it.

## Sections

- 1.1 Autoregressive generation: how the computation differs between prefill and decode
- 1.2 The KV cache: its lifecycle, a formula for its memory use, fragmentation
- 1.3 Key performance metrics: TTFT, TPOT/ITL, throughput, tail latency, goodput
- 1.4 Bottleneck analysis: arithmetic intensity, the Roofline model, what memory bound really means
- 1.5 The inference path step by step: from tokenize to detokenize, and where the bottlenecks are
