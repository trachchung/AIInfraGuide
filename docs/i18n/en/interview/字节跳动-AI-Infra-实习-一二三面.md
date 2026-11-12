---
title: "ByteDance AI Infra internship rounds 1-3"
description: "ByteDance AI Infra internship rounds 1-3 interview questions covering inference optimization, training optimization, and operator optimization"
tags: ["Inference optimization", "Training optimization", "Operator optimization"]
sourceHash: "1aca517c305883ec"
---
### First round

#### Project experience

1. Introduce your work related to Efficient Transformer.

#### Fundamentals

2. What three parts make up each CPU cache line, and what does each do?
3. Why design multi-level caches?
4. After a cache miss, what does the hardware do?
5. What performance costs does the page table introduce, and how do you mitigate them?
6. What benefits does the page table bring?
7. How does C++ function overloading work, and how does the compiler implement it?
8. What are atomic operations in C++, and why were they introduced?

#### Coding

9. Board connectivity: decide whether the first row connects to the last row.

### Second round

#### Project experience

10. Dig deeper into Efficient Transformer research.

#### Fundamentals

11. How do you implement a Sparse Attention Python kernel?
12. What common parallel strategies appear in distributed large-model training?
13. When training or inference uses too much GPU memory, what options do you have? (for example sparsification, gradient checkpointing, quantized inference, mixed precision, distillation)
14. How do you write GPU parallel compute code?

#### Coding

15. Given a full Graph class definition, implement topological sort.

### Third round

#### Project experience

16. What is novel about the Sparse Attention mechanism you proposed?
17. How well do Efficient Attention, MoE, and distributed training work in industrial settings?

#### Fundamentals

18. In C++ polymorphism, when are vtable function addresses fixed? When is an object's vptr assigned?
19. How do you free memory already allocated by a vector in C++?
20. Design a data structure that supports all stack operations and can return the current minimum in O(1).

#### Coding

21. Next permutation (LeetCode 31).
