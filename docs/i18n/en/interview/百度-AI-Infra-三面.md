---
title: "Baidu AI Infra (2)"
description: "Real Baidu AI Infra interview questions covering operator optimization"
tags: ["Operator optimization"]
sourceHash: "a698b6407b984744"
---

## Round 1

### Theory

1. Describe the GPU memory hierarchy and which levels are shared how.
2. In OpenMP, how do you declare a parallel region? Give basic usage.
3. In a parallel setting, does `for(int i=0;i<100;i++){sum++;}` match serial results? How do you ensure correctness?
4. What cache-mapping categories exist, and what are their traits?
5. Explain C++ smart-pointer types and when to use them.
6. What is C++ polymorphism, and how is it implemented?
7. What are dynamic linking and dynamic binding, and how do they differ?

### Project deep dive

8. Introduce resume projects and papers, and discuss related technical details.
9. What did you mainly study in high-performance computing? Any experience with large-scale compute systems?

## Round 2

### Theory

1. What is the underlying data structure of C++ STL map, and how does it work?
2. How do AVL and red-black trees differ in balancing strategy, and what are their performance traits?
3. How is unordered_map implemented underneath?
4. What common ways resolve hash collisions?
5. Introduce the computer storage hierarchy.
6. Which programmer-visible register classes exist in a CPU? What does the program counter (PC) do?
7. List OS commands you use often day to day.

### Project deep dive

8. Introduce resume projects and research work.

## Round 3

### Theory

1. How do you run a custom function before main executes?
2. How do you set a conditional breakpoint while debugging?
3. How do you watch the call stack?
4. How is a program laid out in memory segments?
5. What is the internal mechanism of STL deque?
6. Which is faster to access, registers or Cache? What tech implements Cache vs main memory? How do they compare?
7. What CUDA memory categories exist?
8. What is a CUDA Kernel? What does `__global__` do?
9. How do you get the maximum thread count a GPU supports?
10. Compare greedy algorithms and dynamic programming: ideas and when each fits.
11. What common graph-search algorithms exist?
