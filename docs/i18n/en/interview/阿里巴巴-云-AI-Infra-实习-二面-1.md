---
title: "Alibaba Cloud AI Infra internship second round (1)"
description: "Real Alibaba AI Infra internship second-round interview questions covering inference optimization, operator optimization, high-performance computing, and related topics"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "c94c40de90487d91"
---

## Project experience

1. Introduce your work on operator optimization, operator fusion and graph optimization, and stability fixes.
2. When GPU memory goes abnormal or long-run stability fails during inference or training, how do you locate and debug? Use concrete cases.
3. From an engineering angle, how do you optimize one operator? Walk through approach, gain measurement, and limit analysis with an example.
4. What key steps does quantized inference (e.g. W4A16 under AWQ) involve in practice? Linear-layer changes, data structures, memory layout, correctness checks, etc.

## Fundamentals

5. In multi-GPU setups, what inter-GPU communication options exist? Traits of NVLink versus RDMA?
6. Main differences across GPU ecosystems?
7. How do operator-level gains propagate to the full network? Beyond the network, how do they connect to multi-GPU interconnect, profiling, and software/hardware co-design?
8. How do you view using AI Agents to help write code, review, and organize engineering work?
