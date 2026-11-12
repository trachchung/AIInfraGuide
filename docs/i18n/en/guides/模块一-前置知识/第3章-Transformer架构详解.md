---
title: "Chapter 3: Transformer architecture for AI Infra engineers"
description: "Every Transformer component in depth: Self-Attention, FFN, positional encoding, normalization, and the architecture path from MHA to GQA/MLA and from FFN to MoE"
tags: ["Transformer", "Attention", "RoPE", "MoE", "Model architecture"]
sourceHash: "5d7e87c5f7c1080d"
---

## About this chapter

Transformer is the core architecture of the large-model era, and something every AI Infra engineer has to understand in depth. Every operator you optimize and every parallelism strategy you design ends up acting on some Transformer component.

This chapter starts from Self-Attention: the Q/K/V projections, the full Attention computation, and its O(N²) complexity analysis. Then it covers FFN and activation functions (ReLU/GELU/SwiGLU), positional encodings (Sinusoidal/RoPE/ALiBi), and normalization layers (LayerNorm/RMSNorm, Pre-Norm vs Post-Norm).

The full forward pass section traces the data flow from Token Embedding to output Logits step by step, with input/output shapes marked at each step, and a hand calculation of the parameter count.

Finally, from a distributed and inference angle, it looks at architecture variants: why MHA evolved into MQA, GQA and MLA, and how the shift from FFN to GLU to MoE changes parallel sharding and inference optimization.
