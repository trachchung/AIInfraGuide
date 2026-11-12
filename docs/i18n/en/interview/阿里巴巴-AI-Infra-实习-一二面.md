---
title: "Alibaba AI Infra internship first round"
description: "Real Alibaba AI Infra internship first-round interview questions covering inference optimization, training optimization, operator optimization, and related topics"
tags: ["Inference optimization", "Training optimization", "Operator optimization"]
sourceHash: "d923d75f757c6164"
---

## First round

### Fundamentals

1. What is a memory-bound operator? Give examples.
2. Explain underlying storage formats and differences among FP32, FP16, and INT8.
3. Basic idea of quantization and the concrete workflow?
4. Compare GPU and CPU architecture; which compute tasks fit each?
5. When do UDP and TCP fit, and what are the pros and cons of each?
6. Common Linux commands to check disk usage?

### Project experience

7. Deep discussion of the deployment quantization flow and GridSample operator optimization, including the operator's low-level definition.
8. Deep follow-up around model structure.

### Coding

9. Detect whether a linked list has a cycle.
10. Implement vector add in CUDA.

## Second round

### Fundamentals

1. Explain locks and common lock types.
2. Your understanding of cross-entropy loss.
3. In CUDA programming, what do parallelism and concurrency mean, and how do they differ?
4. Brief principles and differences of GAN versus CVAE.

### Project experience

5. Deep project follow-up (same directions as the first round).

### Coding

6. Hand-write the core of SVM.
7. Hand-write the core Transformer structure.
