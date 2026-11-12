---
title: "BAAI AI Infra round 2"
description: "BAAI AI Infra round-2 interview questions covering inference optimization and operator optimization"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "8d38c706736c9b89"
---

## Fundamentals

1. Describe the CUDA compute model, including block/grid configuration and impact on operator performance.
2. How do different warps in the same block communicate data?
3. Explain the approach for a CUDA reduce.
4. When implementing Softmax in CUDA, how do warp-level vs block-level handling differ?
5. What are static graphs vs dynamic graphs? How do you handle dynamic shapes?
6. In an inference framework, what do the compute graph, runtime graph, and memory management each do? How would you design the overall architecture?
7. Common graph-optimization techniques?

## Operator optimization

8. Compare the core improvements from FlashAttention v1 to v2.
