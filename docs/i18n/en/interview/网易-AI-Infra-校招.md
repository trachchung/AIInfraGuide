---
title: "NetEase AI Infra campus hiring"
description: "Real NetEase AI Infra campus hiring interview questions covering inference optimization, operator optimization, and related topics"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "7d537d1bdd6d0796"
---

### Project experience

1. Describe in detail the main modules and core technical contributions of the AI-related projects on your resume

### Inference optimization

2. When designing a high-throughput, low-latency model inference service, what architecture-level and engineering-level issues matter most

### C++ basics

3. Explain the traits and differences of stack, heap, and static/global storage in program memory
4. Main differences between `new/delete` and `malloc/free`, and why C++ prefers the former
5. Deep copy versus shallow copy, and when you must use deep copy
6. Design intent, use cases, and differences among `std::unique_ptr`, `std::shared_ptr`, and `std::weak_ptr`
7. How the vtable implements runtime polymorphism; semantic and usage differences between virtual and pure virtual functions
8. Memory alignment and compiler struct alignment rules; compute `sizeof` for a given struct
9. C++11 rvalue references and move semantics, and how they enable efficient data transfer

### CUDA programming

10. Basic idea of the SIMT programming model and the hierarchy of thread, block, and grid
11. Where thread-local variables live in a CUDA kernel, and how that relates to register allocation
12. How shared memory cuts repeated global-memory access (tiled matrix multiply as an example)
13. Warp shuffle: the idea and its advantages for in-warp exchange and reduction
14. How vectorized load/store instructions (e.g. float4, int4) enable coalesced access and what performance you gain
15. Causes of shared-memory bank conflict, with examples
16. Avoiding bank conflict via padding or memory-layout changes

### Fundamentals

17. How CPU caches work, including temporal locality, spatial locality, and common replacement policies

### Coding

18. In a stream of many integers, find the top-K most frequent elements in real time (hash table + min-heap), and analyze time complexity
