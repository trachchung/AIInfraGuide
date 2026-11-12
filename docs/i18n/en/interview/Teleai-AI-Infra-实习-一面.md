---
title: "Teleai AI Infra internship first round"
description: "Real Teleai AI Infra internship first-round interview questions covering inference optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "ef94ec94e148e8bb"
---
## Theory

1. How does CLIP work, and what is its inference path?
2. Common operator-optimization directions in LLM inference (memory access, parallel strategies, and so on)?
3. How does DeepSeek-R1 differ from a typical base LLM?
4. Have you written CUDA operators? What performance did they reach?
5. If you customize or fork vLLM, which modules or layers do you start from?
6. How do vLLM and SGLang differ? Where is SGLang stronger than vLLM? Why does SGLang do better on reasoning models?
7. Basics of LLM quantization? Have you deployed quantized models? Have you studied AutoAWQ's code?
8. What core components do Huawei Ascend's CANN and MindIE include?

## Open scenario

9. How do you adapt vLLM to a custom model architecture?
10. How much memory does a V100 have? After INT8-quantizing a DeepSeek or Qwen 32B model, can it run on a V100? What is the deployment plan?
11. Under high concurrency, how do you load-test an inference service to find max concurrency? Which metrics matter (batch size, latency, throughput, and so on)?
