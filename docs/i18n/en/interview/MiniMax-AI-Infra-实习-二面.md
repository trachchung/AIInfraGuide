---
title: "MiniMax AI Infra internship second round"
description: "Real MiniMax AI Infra internship second-round interview questions covering training optimization and high-performance computing"
tags: ["Training optimization", "High-performance computing"]
sourceHash: "f02b195fd83324bc"
---
## Project questions

1. Share an internship project, focusing on the hardest technical problem you hit and the optimization results you got

## Theory

2. What state does each DeepSpeed ZeRO stage (Stage-1 / Stage-2 / Stage-3) shard? How do they progress in communication volume and memory savings?
3. Which parallelism schemes have you used for large-model training? What are the design ideas and fit conditions for data, tensor, and pipeline parallelism?
4. To train a ~70B model, how do you roughly estimate memory needed on one GPU?
5. What is the core idea of LoRA? Why does low-rank factorization cut the number of updated parameters so much?
6. In LoRA, why are down-projection A and up-projection B initialized differently? What is that for?
7. Besides ZeRO, what other large-model training optimizations do you know?

## Coding

8. Implement sliding-window maximum
