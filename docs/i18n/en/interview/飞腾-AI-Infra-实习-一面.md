---
title: "Phytium AI Infra internship first round"
description: "Real Phytium AI Infra internship first-round interview questions covering inference optimization and operator optimization"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "01e74c196558a84d"
---

### Basics

1. What is the core difference between processes and threads? What are the cache hierarchy levels and common replacement policies?
2. In the IEEE floating-point standard, how are the bit widths of FP16, FP32, and FP64 allocated?
3. What is the execution flow of quicksort? What are the basic properties of a heap? Which scenarios suit topological sort?

### Inference and operator optimization

4. Explain the traits of a Decoder-only Transformer, including how QKV are produced and when positional encodings are injected.
5. What is the RMSNorm formula? What are its compute and memory-access traits? From which angles can you optimize it (e.g. load balancing, double buffer, instruction substitution)?
6. How do you handle Softmax numerical stability? What is the idea behind Online Softmax?
7. What memory optimization strategies apply to a fused matmul + dequant operator?
8. In sparse SpMV, how do you achieve load balancing and bandwidth optimization?

### Project experience

9. Deep dive into the project.

### Tools and engineering

10. In Git, what are the ways to fetch a remote branch? How do fetch+checkout and pull differ?
