---
title: "Xiaomi AI Infra internship rounds 1–2"
description: "Xiaomi AI Infra internship interview questions covering inference optimization, training optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Training optimization", "Operator optimization", "High-performance computing"]
sourceHash: "6f0d4540e4f35c78"
---

### Round 1

#### Project experience

1. Introduce the main work done during the internship.
2. Technical approach and implementation details for a thousand-GPU-scale training project.
3. Introduce competition project experience.

#### Fundamentals

4. Difference between deep copy and shallow copy in Python.
5. Characteristics and use cases of the three C++ smart pointers (unique_ptr, shared_ptr, weak_ptr).
6. How Copy-on-Write works.
7. How Zero-Copy works.

#### Coding

8. Implement matrix transpose.

### Round 2

#### Training optimization

9. End-to-end flow of large-model distributed training and how to choose parallel strategies.
10. Introduce common distributed parallel strategies.
11. Why does tensor parallelism (TP) have both row-wise and column-wise splits? What does each mean?
12. Design principles and implementation of sequence parallelism (SP) in Megatron.

#### Fundamentals

13. Which layers and operators does the Transformer architecture include?
14. Structural traits and differences between Encoder and Decoder.
15. Core idea and implementation of FlashAttention.
16. How Online Softmax is computed.

#### Operator optimization

17. In CUDA, is a Block a software concept or a hardware concept?
18. What common CUDA optimization methods are there?
19. What concrete strategies exist for memory-access optimization?
20. How do you overlap compute and memory access?
21. Difference between L1 Cache and L2 Cache.
22. Relationship and differences between shared memory and L1 Cache.

#### High-performance computing

23. Differences among the three C++ smart pointers; what design goal does shared_ptr serve?
24. How does AllReduce work, and what common implementations exist?
25. Communication-volume analysis of Ring AllReduce.
26. What advantages does Tree AllReduce have over Ring AllReduce?

#### Coding

27. Merge two sorted linked lists.
