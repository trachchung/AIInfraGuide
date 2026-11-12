---
title: "AI Infra mixed interview question bank (6)"
description: "Real AI Infra interview questions covering inference optimization, training optimization, and operator optimization"
tags: ["Inference optimization", "Training optimization", "Operator optimization"]
sourceHash: "ec63b83691608603"
---
## CUDA basics

1. Role and idea of CUDA Graph; full kernel launch path
2. How do you choose blockSize and gridSize?
3. What is the default stream? What problems can it cause?
4. Causes of Shared Memory bank conflicts and how to fix them
5. What does threadfence do?
6. How do you debug a CUDA kernel?
7. Unified Memory versus Zero-Copy Memory
8. How are sorting algorithms implemented in CUDA?
9. Which hardware unit runs `sin` on the GPU? What else does that unit support?
10. Traits of the Volta architecture; Independent Thread Scheduling (ITS)
11. Max Shared Memory per block on a 3090?
12. PTX versus SASS
13. How is a GPU's advertised xx TFLOPS figure calculated?

## C++ basics

14. How virtual functions work; memory layout under single, multiple, and virtual inheritance
15. Four casts (`static_cast` / `dynamic_cast` / `const_cast` / `reinterpret_cast`): use and differences
16. Three smart pointers (`unique_ptr` / `shared_ptr` / `weak_ptr`): when to use each
17. Can a function template's declaration and definition live in different files?
18. How CRTP implements static polymorphism
19. How `vector` grows; `resize` versus `reserve`
20. Ways to implement the singleton pattern

## CUDA coding

1. Reduction sum
2. Softmax
3. Matrix transpose
4. Average pooling
5. IoU of two sets of BBoxes

## C++ coding

1. NMS (non-maximum suppression)
2. Conv2D
3. Bilinear interpolation
4. LayerNorm
5. Singleton
