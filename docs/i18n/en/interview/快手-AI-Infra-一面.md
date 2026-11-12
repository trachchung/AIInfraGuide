---
title: "Kuaishou AI Infra first round"
description: "Real Kuaishou AI Infra first-round interview questions covering inference optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "3cd5d07c227c628d"
---

### Theory

1. Do you know Prefill-Decode (PD) disaggregation? What is AF disaggregation for? If PD already exists, why still need AF disaggregation?
2. What did FlashAttention V2 improve over V1? What was the concrete optimization idea? Do you know the latest V4?
3. How many linear layers sit in one Transformer layer of a large model? How do you shard under tensor parallelism (TP), and why that way? What ideas can optimize the middle AllReduce?
4. How is Ray implemented underneath? What key features does it have? How have you used Ray in real research?
5. List and explain common CUDA GEMM optimization methods.

### Project deep dive

6. Detail project background, technical design, and implementation.

### Live coding

7. LeetCode Word Ladder.
