---
title: "Baidu AI Infra internship first round (1)"
description: "Real Baidu AI Infra internship first-round interview questions covering inference, operator optimization, and HPC"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "979db1a2f725d1c9"
---

## Large-model inference

1. Core principle of FlashAttention and its mathematical derivation
2. What optimization strategies does FusedAttention use?
3. Why is RMSNorm faster than LayerNorm?
4. How many fully connected layers does a Llama model contain?
5. Describe Llama2’s inference flow; which operators does each layer include?
6. Design a more flexible and efficient GPU-memory allocator than cudaMalloc (cudaAllocator)

## C++ basics

7. What important new features did C++11 introduce?
8. Categories of smart pointers and their characteristics
9. How does unique_ptr guarantee unique ownership?
10. After shared_ptr’s reference count hits zero, when does destruction run?
11. Can a class member function be defined as a template function?
12. Definitions and differences of lvalues and rvalues

## CUDA programming

13. What programming models or styles does CUDA provide?
14. Functional differences between Tensor Core and CUDA Core

## Coding

15. Longest Consecutive Sequence (LeetCode 128)
16. Longest Substring with At Most K Distinct Characters (LeetCode 340)
