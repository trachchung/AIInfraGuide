---
title: "Alibaba AI Infra (1)"
description: "Real Alibaba AI Infra interview questions covering operator optimization, high-performance computing, and related topics"
tags: ["Operator optimization", "High-performance computing"]
sourceHash: "024785503f4a45da"
---

## Project follow-ups

1. Deep follow-up around the project

## Theory

2. Have you used an Agent to auto-generate CUDA kernels? What was the concrete approach?
3. During large-model training or deployment, have you used low-level debug tools? When NCCL Timeout shows up on a thousand-GPU cluster, how do you find the cause and fix it?
4. If you drop Warp Specialization and keep only tile blocking and shared-memory opts, which dimensions show most of the performance drop?
5. Do you have kernel-level optimization experience? For example fusion-style operators with CuTe DSL or hand-written CUDA—walk through it
6. For kernel fusion, which technical approach do you prefer, and why?
7. Before an inference service went live, what performance work did you do?
8. When writing CUDA, have you gone deep into hardware details—e.g. Warp Specialization mechanics and low-level impl on Hopper?
9. Any optimization experience for MoE models in RL settings?
10. Ever seen kernel fusion make performance worse? Where do you think the cause was?
11. How can you tell a MoE model truly has expert specialization, rather than just splitting a dense model's parameters?
12. In RL + MoE training, have you hit reward-driven routing bias (e.g. most requests routed to a few experts for higher reward)? What did you do about it?
