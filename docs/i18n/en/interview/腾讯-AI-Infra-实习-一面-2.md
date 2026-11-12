---
title: "Tencent AI Infra internship first round (2)"
description: "Real Tencent AI Infra internship first-round interview questions covering inference optimization, high-performance computing, and related topics"
tags: ["Inference optimization", "High-performance computing"]
sourceHash: "b30efd1fc731f1d3"
---

### Project experience

1. Introduce internship work and project experience, and why you left

### Inference optimization

2. Why can an MoE architecture keep training and inference cost under control even with a huge parameter count, and where is the real technical hard part
3. What methods are commonly used for load balancing in MoE, and why some models look fine on loss while experts have already collapsed
4. What is the core difference among GQA, MQA, and standard MHA, and why production inference cares more about GQA
5. How does RoPE positional encoding work, and why does long-context extrapolation often distort
6. What is the acceleration idea behind FlashAttention, and is the core bottleneck compute or memory access
