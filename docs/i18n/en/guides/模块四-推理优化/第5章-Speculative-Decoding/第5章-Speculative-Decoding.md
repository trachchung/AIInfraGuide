---
title: "Chapter 5: Speculative Decoding"
description: "Draft + Verify and distribution-preserving sampling, comparing independent drafts, N-gram/Suffix, Medusa/EAGLE/MTP, plus where the gains stop and how to measure them in vLLM"
tags: ["Speculative Decoding", "Speculative decoding", "Medusa", "EAGLE", "MTP", "N-gram"]
sourceHash: "cbf066a0f2531f27"
---

## About this chapter

Decode is slow because a large model reads its weights once and writes a single token. Speculative decoding changes the split of labor: a cheap Draft writes a short draft first, then the Target runs one forward pass to check the batch. Accepted guesses keep; rejected ones get rewritten by the Target. If accept and correction are done right, the final output has the same distribution as generating token by token with the Target alone.

This chapter goes in four steps: why it is correct, how to guess, when it pays off, and how to measure it:

- **Principles**: one round of Draft + Verify, acceptance probability and the residual distribution, the lossless proof, and how the KV cache rolls back rejected draft tokens.
- **Where drafts come from**: a separate small model, parallel drafts, N-gram and Suffix Decoding; Medusa, the EAGLE family, and other schemes that reuse Target features; plus MTP modules that ship with the model.
- **Where the gains stop**: build a performance ledger from "how many tokens you commit per round / how long each round takes," then look at speculative length, concurrency, long context, quantization, and streaming.
- **Hands-on**: on vLLM, hold the Target fixed and compare plain Decode, N-gram, an independent Draft, and EAGLE-3, collecting acceptance metrics, output text, and latency/throughput together.

> ⏰ **Currency:** The principles do not depend on any framework. vLLM parameters, method names, and metrics here are based on v0.32.0 (released 2026-10-18). Speculative decoding knobs have moved a lot in recent releases; when you change versions, check the official docs.

## Sections

- [**5.1 Core idea of speculative decoding: guess a stretch, then let the large model verify**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/): Draft + Verify flow, distribution-preserving proof, one-hot Draft, first rejection and KV rollback
- [**5.2 Draft models and model-free schemes: cheaper ways to guess better**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/): independent small models and PARD, tokenizer compatibility, N-gram/Suffix, and a shared vocabulary for acceptance rate, mean accepted length, and candidate coverage
- [**5.3 Medusa, EAGLE, and MTP: better drafts by reusing the large model's information**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/): Medusa multi-head, Tree Attention, EAGLE-1/2/3, DeepSeek-V3 MTP, DFlash, and Block Verification
- [**5.4 Where speculative decoding stops helping: high acceptance can still mean slower**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/): commit length and round time, diminishing returns on speculative length, Dynamic SD, concurrency, quantization, and streaming latency
- [**5.5 Speculative decoding in vLLM: configure it, then measure acceptance and performance together**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/): baseline, N-gram, Draft, and EAGLE-3 side by side, per-request acceptance metrics and benchmarks
