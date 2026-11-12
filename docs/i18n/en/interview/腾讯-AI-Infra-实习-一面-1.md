---
title: "Tencent AI Infra internship first round (1)"
description: "Real Tencent AI Infra internship first-round interview questions covering inference optimization, operator optimization, high-performance computing, and related topics"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "89dd3103f8ec16ef"
---

### Project experience

1. Introduce project experience and related technical details

### Inference optimization

2. Explain the design motivation of chunked prefill and the core problem it solves
3. In a large cluster (NVLink within a node, RDMA between some machines across nodes), how would you design a distributed inference plan

### High-performance computing

4. Compare reduce-scatter and all-to-all collective communication
5. What methods can reduce launch kernel overhead
6. Causes of bank conflict in CUDA programming and how to fix them

### Coding

7. Reverse nodes in k-Group
