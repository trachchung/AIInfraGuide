---
title: "Tencent AI Infra"
description: "Real Tencent AI Infra interview questions covering inference optimization and related topics"
tags: ["Inference optimization"]
sourceHash: "e8a386ad0217f2c7"
---

### Inference optimization

1. Introduce the basic idea of FP4 quantization
2. How do matrix-multiplication dimensions change under quantization
3. How does hardware execute quantization operations
4. What concrete hardware requirements does quantized inference have
5. What does FlashAttention do, and what are its core optimization points
6. How do you derive the attention-matrix dimensions in FlashAttention
7. In FlashAttention, does the K matrix include the dimension that corresponds to Q
8. How do you parallelize multi-head attention and multi-batch settings

### Fundamentals

9. GPU architecture questions
10. How do you analyze system performance bottlenecks
