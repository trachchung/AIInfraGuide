---
title: "Bilibili AI Infra internship first round"
description: "Real Bilibili AI Infra internship first-round interview questions covering operator optimization and high-performance computing"
tags: ["Operator optimization", "High-performance computing"]
sourceHash: "59d780b6e7ab843f"
---
## Theory

1. How do you define process, thread, and coroutine? How do they differ in resource use and scheduling?
2. In CPU scheduling, how does fairness differ when you schedule by process versus by thread?
3. In distributed training, what data exchange do All-Reduce, All-Gather, and All-to-All each perform? Which parallelism strategies fit each?
4. How do you use profiling tools to tell whether a GPU operator is memory-bound or compute-bound? What are the usual optimization ideas?
5. Once you know the bottleneck type, what targeted tuning do you apply? For example coalescing memory access, simplifying compute, and choosing block size.

## Project deep dive

6. Detailed questions about the optimization work you took part in

## Live coding

7. CUDA: sum-reduce a large array using Shared Memory for intra-block reduction, then multi-level kernels for the global sum
