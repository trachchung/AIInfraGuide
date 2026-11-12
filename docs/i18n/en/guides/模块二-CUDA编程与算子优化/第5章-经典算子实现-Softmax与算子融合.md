---
title: "Chapter 5: Classic operator implementations — Softmax and operator fusion"
description: "Numerically stable Softmax and Online Softmax, plus the ideas and practice of operator fusion"
tags: ["CUDA", "Softmax", "Online Softmax", "Operator fusion", "Kernel Fusion"]
sourceHash: "95a5976b10752aef"
---

## About this chapter

Softmax is one of the critical nonlinear ops in a Transformer, and background you need before FlashAttention. This chapter starts from numerical stability, works up to Online Softmax, and brings in operator fusion.

Numerically stable Softmax looks at overflow in the naive version, introduces Safe Softmax (subtract the max), and breaks down the cost of the three-pass scan (max → exp-sum → normalize).

Online Softmax covers online normalizer calculation, derives a one-pass Softmax algorithm, and implements it efficiently on GPU — core prior knowledge for FlashAttention.

Operator fusion explains why you fuse (fewer kernel launches and less global-memory traffic), covers common patterns (Bias + Activation, LayerNorm + Dropout), and compares hand fusion with compiler auto-fusion.

Hands-on: implement an Online Softmax CUDA kernel, fuse Softmax + Scale into one kernel, and compare performance.
