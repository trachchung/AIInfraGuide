---
title: "MiniMax AI Infra internship first round (1)"
description: "Real MiniMax AI Infra internship first-round interview questions covering inference and training optimization"
tags: ["Inference optimization", "Training optimization"]
sourceHash: "b1a06f5f3d484055"
---
## Project deep dive

1. Introduce your internship project and research work

## Theory

2. How do the training objectives of SFT and RLHF differ at a fundamental level? Why do most models still do RLHF after SFT?
3. In an MoE model, how does the router decide which experts get a token? What causes severe expert load imbalance?
4. When some experts are clearly underused, how can you change the routing strategy to raise utilization?
5. Introduce LLM architectures you know well, and compare them on attention compute, training methods, and inference efficiency
6. What is the core objective PPO optimizes in RLHF? Write the objective and explain each term
7. As parameter count grows, why can MoE still keep relatively high training efficiency?

## Live coding

8. Given K sorted arrays, find the median of the merged result
