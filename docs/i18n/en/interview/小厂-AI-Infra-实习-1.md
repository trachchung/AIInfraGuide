---
title: "Small company AI Infra internship (1)"
description: "Real small-company AI Infra internship interview questions covering inference optimization"
tags: ["Inference optimization"]
sourceHash: "cc40b75c5cab7e05"
---

### Inference optimization

1. Describe the scheduling flow of the scheduler in vLLM.
2. What happens after a request is preempted in vLLM?
3. In speculative decoding, how do the draft model and target model interact? Compare how vLLM and SGLang implement this.
4. Explain the principles of GPTQ quantization and SmoothQuant.
5. Introduce the EPLB inference mechanism in DeepSeek V3.
6. How does MLA’s compute complexity differ between prefill and decode, and what is the idea behind matrix absorption optimization?
7. What innovations does DeepSeek V3.2 bring over prior versions?
8. In SGLang multimodal scenarios with TP enabled, how are ViT image embeddings efficiently reused across processes?
