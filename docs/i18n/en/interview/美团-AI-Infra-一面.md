---
title: "Meituan AI Infra first round"
description: "Real Meituan AI Infra first-round interview questions covering inference optimization and operator optimization"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "9042cebc29dd096f"
---

Notes from Meituan AI Infra first and second rounds.

### Project deep dive

1. Deep questions on every project on the resume, focusing on details and technical implementation

### Theory

2. Pros and cons of Decoder-only vs Encoder-only architectures
3. Outline the main design traits of LLaMA
4. Basic concepts and hierarchy of the CUDA programming model
5. Which levels does the CUDA memory model include, and what are their traits
6. How do you handle bank conflicts when using shared memory, and what avoidance strategies exist
7. What issues can show up when using registers (for example register spilling), and how do you handle them
8. How does GPU multithreading scheduling differ in essence from CPU multithreading
9. Explain the core idea of Flash Attention
10. Explain the design idea and working mechanism of Paged Attention
