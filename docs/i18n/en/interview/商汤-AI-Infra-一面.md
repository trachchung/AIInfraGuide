---
title: "SenseTime AI Infra interview"
description: "SenseTime AI Infra interview questions covering operator optimization"
tags: ["Operator optimization"]
sourceHash: "4ecc277482ce796b"
---
The following topics were covered in the SenseTime AI Infra interview, mainly deep dives into projects.

### Project experience (GEMM optimization)

1. Describe the GEMM optimization implementation in detail
2. Why does one thread compute an 8x8 block of C?
3. Can registers spill, and how do you handle that?
4. Did the project include your own innovations?
5. What does `#pragma unroll` do, and when do you use it?
6. How many compute iterations run inside a single tile?
7. How does your implementation compare in performance to cuTLASS?

### Fundamentals

8. Have you optimized convolution operators?
9. Do you know PPL (SenseTime's HPC library)?
10. How do Tensor Cores work, and how do you use them?
11. Have you built low-bit GEMM?
