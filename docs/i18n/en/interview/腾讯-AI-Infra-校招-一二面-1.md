---
title: "Tencent AI Infra campus hiring first and second rounds (1)"
description: "Real Tencent AI Infra campus hiring first- and second-round interview questions covering operator optimization and related topics"
tags: ["Operator optimization"]
sourceHash: "ff8f819876634132"
---

### First round

#### Inference optimization

1. Compare source-level differences among TensorRT-LLM, AGI, and vLLM
2. What is the design motivation for continuous batching
3. Introduce the source structure and implementation approach of fastllm.cpp

#### Fundamentals

4. For CPU-bound work in Python, should you use multiprocessing or multithreading
5. How inheritance is implemented under the hood in C++
6. Common ways to speed up YOLO model inference

#### Coding

7. Maximum subarray sum

---

### Second round

#### Inference optimization

1. Introduce the core ideas of vLLM and PagedAttention
2. Approach to implementing PagedAttention with Triton

#### Fundamentals

3. Hierarchy and traits of the CUDA memory model
4. How to run work at compile time in C++ (constexpr and related)

#### Coding

5. Count the number of 1 bits in an integer's binary representation
