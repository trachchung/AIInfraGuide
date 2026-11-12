---
title: "Ant Group AI Infra internship third round"
description: "Real Ant Group AI Infra internship third-round interview questions covering inference optimization, operator optimization, high-performance computing, and related topics"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "ec364156b1bac6a0"
---

This round focuses on how deeply you understand large-model inference, quantization, and Attention.

### Fundamentals

1. Besides W4A16, how well do you know common industry schemes like INT8 and FP8? What are the two common FP8 formats, and how do they differ for training versus inference?
2. How do you compute KV Cache size? (batch size, sequence length, head count, head dim, layer count, dtype, and related factors)
3. Why does large-model inference need KV Cache, and what is its essential role?
4. How do you understand Attention? Why is Attention-related quantization harder than weight quantization?

### Project experience

5. Introduce the quantized-inference Runtime project: which open-source Runtime did you modify? Concrete W4A16 details? How much accuracy loss? Which dataset for evaluation?
6. Core internship work: completing the fusion path and plugging in high-performance libraries; locating and fixing stability issues in long full-network runs; why a lightweight Timing tool was designed and how it helped find bottlenecks.
