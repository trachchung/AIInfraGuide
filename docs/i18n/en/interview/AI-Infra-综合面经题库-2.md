---
title: "AI Infra mixed interview question bank (2)"
description: "Real AI Infra interview questions covering inference optimization, training optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Training optimization", "Operator optimization", "High-performance computing"]
sourceHash: "b7acbe904c34a807"
---
## Basics

1. Explain the core idea and implementation approach of FlashAttention
2. Why is column-wise traversal of a row-major matrix on the CPU much slower than row-wise traversal? Which performance metric gets worse?
3. What is the benefit of using Shared Memory for a GPU matrix transpose?
4. What does a Flow Matching model predict? How do you understand the conditional velocity field conditioned on data sample x0?
5. How do you compute the time shift in QwenImage?
6. What Weight-Only quantization schemes exist? When writing a Weight-Only quantization CUDA kernel, how do you optimize memory access? Do you know the Marlin kernel?
7. How is sequence parallelism (SP) implemented in Megatron?
8. How do DeepSpeed ZeRO Stage 1 and Stage 2 differ in communication volume? Is there a gap between the paper and the code?
9. In multi-GPU communication, how do NVSHMEM and NVLink differ?

## PyTorch coding

1. Implement Multi-Head Attention (3 variants)
2. Implement FlashAttention v1
3. Write pseudocode for Flow Matching Model sampling

## Algorithms

1. Quicksort
2. Median of two sorted arrays
3. Next permutation
4. Binary tree maximum path sum
5. Path Sum III
6. Given point coordinates on a number line and a fixed number of equal-length segments, what is the shortest segment length that can cover all points?
7. Top-K frequent strings (ties broken by ascending lexicographic order)
8. Start from string s; each step rotate right by one and append (length doubles each time). What is the character at position N after infinite expansion?
9. Two fingers on a 26-letter keyboard: minimum travel distance to type string s
