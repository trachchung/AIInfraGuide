---
title: "Meituan Beidou AI Infra campus hire"
description: "Real Meituan AI Infra campus-hire interview questions covering inference optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "526d9a571a47ffdf"
---

Interview topics for Meituan Beidou AI Infra campus hire.

### Theory

1. Explain the overall Transformer architecture, and what advantages it has over classic RNNs
2. Where do Transformer parameters mainly sit? Which part has the most parameters? Which part has the most compute, and why
3. Explain the difference between CUDA Core and Tensor Core on GPUs; list common GPU models with memory capacity and bandwidth
4. Introduce the basic idea of large-model quantization, and describe how quantization operators are implemented
5. Detail the design idea of Prefill-Decode separation (PD disaggregation)
6. Detail how Paged Attention works

### Live coding

7. LeetCode 25: reverse nodes in k-group
8. Write a CUDA operator: basic prefix sum, and explain feasible optimizations
9. Write a CUDA operator: basic GEMM, and explain feasible optimizations
