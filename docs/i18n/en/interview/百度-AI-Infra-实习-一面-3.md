---
title: "Baidu AI Infra internship first round (3)"
description: "Real Baidu AI Infra internship first-round interview questions covering inference and operator optimization"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "8fbfa52c8d13de78"
---

## Distributed training

1. Explain what TP, PP, and DP mean and how each runs
2. How do you choose a parallel strategy based on TP vs PP communication cost?

## Quantization

3. Explain differences among per-tensor, per-channel, and group-wise quantization granularities
4. Accuracy differences across quantization methods, and why group-wise can further cut quantization error
5. Compute-cost comparison across quantization methods and optimization techniques
6. How do you handle outliers during quantization?
7. Introduce the core ideas of GPTQ, AWQ, and SmoothQuant
8. Principle and implementation of KV Cache quantization

## Attention and inference optimization

9. What is FlashAttention’s acceleration principle?
10. Main differences between FlashAttention v1 and v2
11. Bc-block tiling strategy in FlashAttention, and how 1-loop FlashAttention is implemented
12. Design idea of PagedAttention
13. Differences between large-model prefill and decoding stages, and why they differ
14. What problems does FlashAttention have in decoding? How does FlashDecoding improve it?

## Operator implementation

15. Concrete implementation of RMSNorm
16. General approach and methodology for CUDA Kernel optimization
17. Given Conv2D input [C=64, W=64, H=128], 3x3 kernel, output [C=128, W=64, H=128], compute parameter count and FLOPs
18. What methods exist for operator optimization on CPU (e.g. AVX-512)?

## C++ basics

19. Categories of smart pointers and their use cases

## Coding

20. Implement a CUDA LayerNorm Kernel
