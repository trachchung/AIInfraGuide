---
title: "Huawei AI Infra internship (2)"
description: "Real Huawei AI Infra internship interview questions covering inference optimization, training optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Training optimization", "Operator optimization", "High-performance computing"]
sourceHash: "929ea59d66f77d9a"
---

### Inference optimization

1. How do you estimate parameter counts needed when deploying a model versus running inference?
2. In model inference, how do Prefill and Decoding differ? How do you compute KV Cache size?
3. What is the math behind the Temperature hyperparameter? In what order do Temperature, Top-k, and Top-p apply?
4. What is Softmax's definition and compute process?

### Training optimization

5. In DeepSpeed, what do ZeRO Stage 1, 2, and 3 each optimize? How do the stages differ?
6. Suppose you distribute matrix multiply A x B across 4 GPUs: how do you shard A and B, and use inter-GPU communication to save memory?

### High-performance computing

7. What communication operators do you know? What common collective communication primitives exist?
