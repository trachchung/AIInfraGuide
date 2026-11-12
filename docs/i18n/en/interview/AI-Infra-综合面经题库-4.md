---
title: "AI Infra mixed interview question bank (4)"
description: "Real AI Infra interview questions covering operator optimization and high-performance computing"
tags: ["Operator optimization", "High-performance computing"]
sourceHash: "95080c5875b2ec35"
---
## Project experience

1. Deep dive into your project

## Basics

2. Have you done kernel-level optimization, for example with CuTe DSL or hand-written CUDA for operator fusion? Describe what you did.
3. When fusing kernels, which approach do you usually prefer?
4. Have you seen performance drop after operator fusion? Why?
5. When writing CUDA, have you looked at low-level details such as Warp Specialization on Hopper and how it works underneath?
6. When deploying or training large models, have you used low-level debugging tools? How do you locate and fix an NCCL Timeout at thousand-GPU scale?
7. Have you optimized MoE models for RL settings?
8. Before shipping an inference service, what performance work did you do?
9. Have you tried using an Agent to auto-generate CUDA kernels? How did that work?
