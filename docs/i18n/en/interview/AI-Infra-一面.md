---
title: "AI Infra first-round interview"
description: "Real AI Infra first-round interview questions covering operator optimization and related topics"
tags: ["Operator optimization"]
sourceHash: "1c52faf5b830f106"
---
1. Explain the basic idea of the Roofline model. How do you use it to tell whether an operator has hit a compute bottleneck?
2. When you double the number of GPUs used for training or inference, which parts of the system are most likely to become bottlenecks? Analyze why, and outline how you would optimize or mitigate them.
3. Is GEMM always a compute-bound operator? If you need to optimize it, what is the overall approach?
4. During performance tuning, how do you find and verify bottlenecks? Which methods or tools do you usually use?
5. Explain your understanding of FlashAttention.
6. In C++, if an out-of-bounds array write corrupts other data structures and you still have a coredump, how do you investigate?
7. Coding: hand-write an Attention module that includes GQA (Grouped Query Attention).
