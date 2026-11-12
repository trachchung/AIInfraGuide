---
title: "Huawei AI Infra (2)"
description: "Real Huawei AI Infra interview questions covering training optimization and high-performance computing"
tags: ["Training optimization", "High-performance computing"]
sourceHash: "7c0c435b83fa6acd"
---

### Training optimization

1. Introduce the principles and traits of the LAMB optimizer.
2. How should you adjust the learning rate when increasing batch size? If the model is very large but batch size stays fixed, what then?
3. List the parallel sharding strategies you know.
4. Introduce ZeRO's core idea and how the stages differ.
5. Under pipeline parallelism (PP), is memory use and compute the same on every GPU? How are activations distributed?
6. What combinations and scenarios fit 3D parallelism?

### High-performance computing

7. How do DCU and Huawei AI accelerators differ in architecture, ecosystem, and communication libraries?
8. How are the async save mechanisms in DDP and DeepSpeed implemented?

### Project experience

9. Detail the technical approach and implementation of the thousand-GPU training project.
10. What work did you do on async save?

### Fundamentals

11. What are the main architectural differences between T5 and GPT-2?
12. What is the overall structure of Transformer?
13. What role do residual connections play in deep networks?
