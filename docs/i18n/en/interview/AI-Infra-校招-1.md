---
title: "AI Infra campus recruiting (1)"
description: "Real AI Infra campus recruiting interview questions covering operator optimization and high-performance computing"
tags: ["Operator optimization", "High-performance computing"]
sourceHash: "a94b0ab375c48f68"
---
Suitable roles: heterogeneous computing / AI framework R&D / high-performance computing / model deployment / algorithm optimization / operator development

## Basics

1. How is the CUDA software hierarchy (Grid, Block, Thread) organized?
2. What types make up the CUDA memory hierarchy (global memory, shared memory, registers, and so on)? What are the traits of each?
3. How do CPU and GPU architectures differ at a fundamental level? Why is the GPU better suited to parallel computing?
4. How do you understand the relationship between an SM (Streaming Multiprocessor) and an SP (Streaming Processor)?
5. What is a CUDA Stream? How do synchronous and asynchronous streams differ?
6. What roles do the GPU's L1 and L2 caches play?
7. What should you watch for when using shared memory (thread sync, bank conflicts, and so on)?
8. When you optimize a CUDA kernel, which dimensions can you work on?

## Coding

1. Implement a matrix-multiplication kernel
2. Implement Softmax reduction
3. Convert data layout from NCHW to NHWC
4. Given an array of length n (values in 0~256), count how often each value appears and store the counts in another array
5. Move odd-index elements to the left and even-index elements to the right, in place
