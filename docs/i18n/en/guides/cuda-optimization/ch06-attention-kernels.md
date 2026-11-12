---
title: "Chapter 6: The Attention operator"
description: "How FlashAttention V1/V2/V3 work and how they are implemented, plus Flash-Decoding and the PagedAttention CUDA path for the decode stage"
tags: ["FlashAttention", "Attention", "Flash-Decoding", "PagedAttention", "CUDA"]
sourceHash: "e94f6639ae541ab0"
---

## About this chapter

Attention is the core compute in a Transformer, and a top target for AI Infra optimization. This chapter starts from the performance problems of standard Attention, then goes into the FlashAttention family and decode-stage acceleration.

The performance problems of standard Attention cover the O(N²) memory of the naive implementation and the compute / memory bottlenecks.

FlashAttention V1 unpacks the core idea: tiling plus Online Softmax so you never materialize the N×N matrix, cutting extra memory from O(N²) to O(N) and HBM traffic from Θ(N²) to Θ(N²d²/M) (M is SRAM size), plus the recomputation strategy in the backward pass.

FlashAttention V2 improves the parallel strategy (parallel along the Q sequence dimension) and Causal Mask handling.

FlashAttention-3 uses Hopper features (WGMMA, TMA, FP8) for async pipeline optimization.

Decode-stage acceleration covers Flash-Decoding, FlashDecoding++, FlashInfer, and other Attention engines aimed at serving.

The PagedAttention CUDA section reads how vLLM maps virtual pages to physical pages on the GPU.

Hands-on: derive the FlashAttention tiling process on a whiteboard, and implement a simplified FlashAttention in Triton.
