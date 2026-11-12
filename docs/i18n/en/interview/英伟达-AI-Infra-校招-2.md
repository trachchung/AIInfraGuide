---
title: "NVIDIA AI Infra campus hire (2)"
description: "NVIDIA AI Infra campus-hire interview questions covering operator optimization"
tags: ["Operator optimization"]
sourceHash: "7290ca82917fde4e"
---

### Fundamentals

1. Meaning and working principle of SIMT (Single Instruction, Multiple Threads).
2. What factors affect Occupancy? How do you tune it?
3. What is the granularity of a bank conflict?
4. What constrains GEMM tile-size choices?
5. Why does float4 load/store of global memory improve performance?
6. Can one Block be scheduled across different SMs?
7. Cache capacities of mainstream GPU models?
8. Concrete performance impact of Warp Divergence.
9. How instruction-level parallelism (ILP) is realized on NVIDIA GPUs.

### Coding

10. Implement CUDA matrix transpose.
11. Implement vector outer product.
