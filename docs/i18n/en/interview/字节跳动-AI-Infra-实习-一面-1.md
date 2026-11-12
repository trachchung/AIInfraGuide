---
title: "ByteDance AI Infra internship first round (1)"
description: "ByteDance AI Infra internship first-round interview questions covering operator optimization"
tags: ["Operator optimization"]
sourceHash: "6b3e5d53a813efe7"
---
## Operator optimization

1. Do you have kernel-level optimization experience? For example using CUTE DSL or hand-written CUDA for operator fusion—please be concrete.
2. When doing kernel fusion, which implementation style do you usually prefer?
3. Have you seen fusion make performance worse? Why?
4. How does warp specialization work on Hopper, and how is it implemented underneath?
5. If you drop warp specialization and keep only tiling and shared memory optimizations, where does most of the performance loss show up?

## Inference optimization

6. Do you have hands-on experience optimizing MoE inference (for example in RL settings)?
7. How do you tell that an MoE model actually learned expert specialization, rather than just splitting a dense model?
8. In RL + MoE training, have you seen rewards cause routing collapse (all traffic going to a few experts)? How do you handle that?

## Project experience

9. Introduce your project experience.
10. Introduce your internship, including optimizations you made for online inference services.
11. Have you tried using an Agent to generate CUDA kernels? What was the approach?
