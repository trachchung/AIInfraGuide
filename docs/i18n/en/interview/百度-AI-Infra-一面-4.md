---
title: "Baidu AI Infra (2)"
description: "Real Baidu AI Infra interview questions covering operator optimization"
tags: ["Operator optimization"]
sourceHash: "9a42400ec2f95e5d"
---

## Round 1

### Project deep dive

1. Introduce resume projects and papers, with follow-ups on the technical points involved

### High-performance computing

2. How do you open an OpenMP parallel region? (pragma omp parallel related syntax)
3. Inside a parallel region, does `for(int i=0;i<100;i++){sum++;}` match serial results? How do you keep consistency?
4. GPU memory hierarchy: roles/scopes of Register File, L1 Cache, Shared Memory, L2 Cache, Global Memory

### Computer architecture

5. Cache mapping types and traits (set-associative, fully associative, direct-mapped)

### C++ basics

6. Smart-pointer types and when to use each
7. C++ polymorphism concepts and implementation (override vs overload)
8. Dynamic linking vs dynamic binding

## Round 2

### Project deep dive

9. Deep discussion around resume and projects

### C++ basics

10. What is the underlying data structure of STL map?
11. Difference between balanced BST (AVL) and red-black trees
12. How unordered_map is implemented underneath
13. Common ways to resolve hash collisions

### Computer architecture

14. Computer storage hierarchy (registers, Cache, main memory, secondary storage)
15. Which programmer-visible register classes exist in a CPU? What does the program counter (PC) do?

### Other

16. What common OS commands do you use?

## Round 3

### C++ basics

17. How do you run a function before main executes?
18. How do you set a conditional breakpoint while debugging?
19. How do you watch the call stack?
20. How a program is segmented in memory (code, data, heap, stack, etc.)
21. How STL deque is implemented underneath

### Computer architecture

22. Which is faster, registers or Cache? What tech implements Cache vs main memory? How do they compare?

### CUDA programming

23. CUDA memory categories (local memory, global memory, etc.)
24. What a Kernel is; role of the `__global__` keyword
25. How do you get the maximum thread count a GPU allows?

### Algorithms

26. Greedy vs dynamic programming
27. Common graph-search algorithms (BFS, DFS)
