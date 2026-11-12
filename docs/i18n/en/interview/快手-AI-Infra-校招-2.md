---
title: "Kuaishou AI Infra campus hire (2)"
description: "Real Kuaishou AI Infra campus-hire interview questions covering training optimization, operator optimization, and high-performance computing"
tags: ["Training optimization", "Operator optimization", "High-performance computing"]
sourceHash: "1788f7a51035943a"
---

### Theory

1. What key architectural improvements does H100 bring versus A100?
2. Explain the Warp concept in CUDA and its role in the execution model.
3. In data parallelism (DP) and tensor/sequence parallelism (TP-SP), what is the idea of overlapping compute with communication? Which communication ops overlap with which compute ops?
4. Go deep on FlashAttention's core idea and key optimization points.
5. With vs without pipeline parallelism (PP), is peak GPU memory the same? Why?
6. What does the CUDA_DEVICE_MAX_CONNECTIONS environment variable mean?
7. What does launch bound mean in CUDA? Can H2D and D2H transfers overlap?

### Live coding

8. Implement an LRU cache.
9. Write pseudocode for Online Softmax and FlashAttention.
