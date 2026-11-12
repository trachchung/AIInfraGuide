---
title: "Huixi Intelligence AI Infra internship rounds 1–3"
description: "Real Huixi Intelligence AI Infra internship rounds 1–3 interview questions covering operator optimization and related topics"
tags: ["Operator optimization"]
sourceHash: "5295c2296f11847f"
---

### First round

1. Company intro and basic background discussion.

### Second round

#### Operator optimization

2. What is bank conflict? What causes it, and what fixes exist?
3. When writing operators, how do you maximize cache use? How do you tile data based on L1 and L2 capacity?
4. What is warp divergence, and how does it hurt performance?
5. What are the maxima of blockDim.x and gridDim.x?
6. Hand-write a naive CUDA matmul operator, then describe later optimization directions and how to pick the best tile size.

#### Fundamentals

7. Describe architecture traits and design differences of CPU versus GPU.
8. Your understanding of CUDA's three-level grid / block / thread hierarchy.
9. Common convolution algorithms and their pros and cons?

#### Project experience

10. Deep dive into project experience.

### Third round

#### Operator optimization

11. Difference between shared memory and hardware cache?
12. For matmul acceleration, which is faster between Tensor Core and CUDA Core? How does Tensor Core work?
13. Hand-write a CUDA Softmax operator.

#### Fundamentals

14. What is the overall Transformer architecture?
15. Typical Softmax uses in deep learning?
