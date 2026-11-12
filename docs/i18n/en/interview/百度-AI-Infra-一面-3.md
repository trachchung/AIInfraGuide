---
title: "Baidu AI Infra first round (3)"
description: "Real Baidu AI Infra first-round interview questions covering training optimization and high-performance computing"
tags: ["Training optimization", "High-performance computing"]
sourceHash: "84dbdce0313934e0"
---

## Reinforcement learning

1. Relationship between Trust Region methods and PPO
2. Is PPO on-policy or off-policy? Why introduce importance sampling?
3. In PPO, how does the clip mechanism bound the upper and lower limits when advantage A is positive vs negative?
4. How is the PPO loss computed? How is Generalized Advantage Estimation (GAE) computed, and how does lambda affect variance and bias?
5. How GRPO computes loss; how sequence-level loss is assigned to each token; difference between sequence-level average and batch-level average
6. Traits of other GRPO variants (such as DAPO, GSPO, GFPO)
7. Basic idea and applications of Agentic RL

## Distributed training

8. During training, what share of GPU memory do optimizer states, gradients, and model parameters each take
9. Compare FSDP with DeepSpeed ZeRO Stage 1/2/3

## Project deep dive

10. Project introduction and technical design; how you combine LLMs with concrete application scenarios; evaluation methods and metric results

## Live coding

11. Level-order traversal of a binary tree, also recording each node's level (LeetCode 102 variant)
