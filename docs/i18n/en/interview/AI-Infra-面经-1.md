---
title: "AI Infra interview notes (1)"
description: "Real AI Infra interview questions covering inference optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "6144516798b11ea1"
---
## C++ basics

1. How do pointers and references compare?
2. Uses of `static`; differences when it modifies member variables versus member functions; static global variables versus ordinary globals
3. Kinds and use of smart pointers; is `shared_ptr` thread-safe?
4. Rvalue references and when to use them
5. Roles of `std::move` and `std::forward`, and how they differ
6. Why must a copy constructor take its argument by reference rather than by value?
7. `new` versus `malloc`; how `new` works underneath
8. Overview of the C++ memory model
9. Why can a constructor not be virtual? Why should a destructor usually be virtual?
10. How polymorphism works and the underlying mechanism; where the vtable lives; template polymorphism and partial specialization
11. How does `vector` grow? Why is 2x common?
12. Lambda syntax and capture modes
13. Lock types and when to use them
14. When sharing memory across threads, when do you use condition variables versus mutexes? How do they differ?
15. Four necessary conditions for deadlock and how to prevent it; `lock_guard` versus `unique_lock`
16. How do you implement a thread-safe singleton?
17. How do you force an object to be created only on the heap?
18. C++ compilation pipeline; dynamic versus static libraries
19. How does C++ call a function written in C?
20. Process versus thread
21. How do you detect big-endian versus little-endian storage (at least two ways)?

## High-performance computing

1. Overview of GPU architecture
2. Global memory versus shared (local) memory on the GPU; how do you use shared memory well?
3. How cache works and how to raise hit rate
4. Temporal locality versus spatial locality
5. Compute-bound versus memory-bound
6. Common performance optimization ideas
7. OpenCL execution flow
8. Why reduce branches in OpenCL kernels? What are masks for?
9. Main parameters of an OpenCL kernel
10. Why is separable convolution often slow on GPU? Why is it memory-bound?
11. Operator fusion; derive the formula for fusing conv + BN and why it works
12. Common ways to implement convolution in inference frameworks
13. Causes of bank conflicts and fixes
14. Basics of the TVM compiler framework

## Deep learning basics

1. Batch Normalization: computation, role, train versus infer differences
2. Depthwise versus Pointwise convolution
3. Evolution of MobileNet v1/v2/v3; GhostNet design; SE module structure
4. YOLO v1/v2/v3 versus SSD
5. Where is global pooling usually placed in a network?
6. Knowledge distillation: idea and pipeline
7. Quantization taxonomy; INT8 details: symmetric versus asymmetric; QAT and inference details
8. Dropout at train versus infer time
9. Deconvolution versus dilated convolution
10. GIoU loss: definition and role
11. Softmax formula and why it is used for classification
12. Improvements on NMS

## Coding

1. Implement convolution
2. Implement a computation graph
3. Implement Pooling
4. Implement NMS
5. Matrix multiply and vector sum in OpenCL
