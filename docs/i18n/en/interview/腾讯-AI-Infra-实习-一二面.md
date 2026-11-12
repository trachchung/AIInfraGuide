---
title: "Tencent AI Infra internship first and second rounds"
description: "Real Tencent AI Infra internship first- and second-round interview questions covering operator optimization, high-performance computing, and related topics"
tags: ["Operator optimization", "High-performance computing"]
sourceHash: "4dc559481f6b96b3"
---

### First round

#### Project experience

1. Introduce project experience, including technical hard parts and highlights

#### Fundamentals

2. Explain basic MPI concepts and how to use it
3. What is the basic idea of multithreaded programming
4. What common locking mechanisms are there
5. Conditions that cause deadlock in an OS, and how to resolve it
6. Compare similarities and differences between UDP and TCP
7. Which mechanisms does TCP use to guarantee reliable delivery

#### Coding

8. Given an unsorted array and a sorted array, insert elements from the unsorted array into the sorted one while keeping order, and analyze time complexity
9. Explain an algorithm orally: a classic big-data processing problem

---

### Second round

#### Fundamentals

1. How deep is your understanding of CPU and GPU architecture
2. Do you have experience writing CUDA kernels
3. What is the definition and what are the traits of POD types in C++
4. Besides priority queues, which other STL containers in C++ can implement a heap

#### High-performance computing

5. Explain these concepts: shuffle, Hyper-Q, SM, slot
6. What NVIDIA Nsight is for and how to use it
7. What the SOL (Speed of Light) metric means
8. Basic idea of the Polyhedral Model
9. Core idea of the TVM compilation framework
10. Difference between Array of Struct and Struct of Array, and when each fits
11. Design idea of the Halide programming language
12. How to implement a warp-level reduce

#### Coding

13. Sliding-window median (implement with a max-heap and a min-heap)
