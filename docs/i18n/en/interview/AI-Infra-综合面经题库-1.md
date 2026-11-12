---
title: "AI Infra mixed interview question bank (1)"
description: "Real AI Infra interview questions covering inference optimization, training optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Training optimization", "Operator optimization", "High-performance computing"]
sourceHash: "6d01440abbaa282f"
---
## Basics

1. What are the advantages of TMA on Hopper? How do you invoke it? Does the data transfer go through the L1 cache?
2. In FlashAttention v2, why does the outer loop iterate over Q? Roughly what share of time does Flash Decoding's combine kernel take?
3. How do you analyze the compute-to-memory ratio of MLA decode? Does that ratio depend on sequence length or batch size?
4. Design ideas behind Mooncake's KV-Cache-centric PD separation scheme
5. How does a DiT inference framework compare with an LLM inference framework?
6. Is knowledge distillation for large language models a good fit for the pretraining stage?
7. What are the training and inference steps of a Diffusion Model? When inference sets `num_inference_steps` to 40, why does training still need `timesteps` set to 1000?
8. Introduce dLLM (discrete large language models). How do they differ at a fundamental level from autoregressive (AR) models?
9. How do `torch.repeat` and `torch.expand` differ in behavior and memory use?
10. What launch arguments does `torchrun` take? On Linux, how do you kill all processes that involve `torchrun` in bulk?

## CUDA coding

1. Implement 4D tensor elementwise multiply that supports torch broadcast semantics
2. Given A: (1, 256), B: (256, 128), C: (128, 256), compute (A * B) * C
3. Embedding Sparse Feature Pooling: A holds 1 million discrete IDs (range 0~999), B holds 1 million floats. Compute length-1000 array C where C[i] = sum of B[j] for all j with A[j] = i

## PyTorch coding

1. Implement a LoRA Adapter

## Algorithms

1. Implement a memory pool (support something like `new Foo[]` / `delete[]`)
2. In C++, how do you compare two floats for equality?
3. Implement an LRU cache
4. Number of islands
5. Level-order traversal of a binary tree
6. Compute Hamming Weight
7. K-Coverage Intervals
