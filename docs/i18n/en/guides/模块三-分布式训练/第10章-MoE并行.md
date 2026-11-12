---
title: "Chapter 10: MoE parallelism"
description: "How sparse expert models parallelize: the Router, All-to-All in Expert Parallelism, combining EP×DP×TP×PP, and load balancing"
tags: ["MoE", "Expert Parallelism", "All-to-All", "Load balancing", "Sparse models"]
sourceHash: "5f4162626e229486"
---

## About this chapter

MoE (Mixture of Experts) is the mainstream architecture for today's hundred-billion and trillion-parameter models (DeepSeek-V3, Mixtral, and others). Sparse activation grows the parameter count without a matching jump in compute. MoE also adds a new parallelism dimension: Expert Parallelism (EP), and its signature All-to-All communication pattern. This chapter stands on its own because its communication pattern and load-balancing problems differ in kind from dense-model parallelism.

---

## Sections

### 1. MoE model structure recap

- Dense FFN → sparse MoE: replace a single FFN with $E$ Experts
- Router (Gating Network): pick the Top-K Experts for each token
- Sparse activation: parameter count $\times E$, but per-token compute only grows by a factor of $K$
- Capacity Factor and token dropping

### 2. Expert Parallelism (EP)

- Core idea: put different Experts on different GPUs; each GPU stores only $\frac{E}{N}$ Experts
- Difference from TP: TP shards a single weight matrix; EP decides which Experts live where

### 3. All-to-All communication (focus of this chapter)

- Two All-to-Alls: dispatch (send tokens to the GPUs that hold their Experts) + combine (send results back to the original GPUs)
- Communication volume analysis: depends on token count, Top-K, and Expert placement
- Why All-to-All is the main bottleneck in MoE training
- Communication optimizations: grouped All-to-All, compute/communication overlap, high-performance libraries such as DeepEP

### 4. Load balancing

- Router skew: a few Experts get chosen too often → some GPUs overload, others sit idle
- Auxiliary Loss: an aux loss that steers the Router toward even token distribution
- DeepSeek's aux-loss-free load balancing (bias adjustment)
- Expert capacity and drop/pad strategies

### 5. Combining EP with other parallelism

- EP × DP: orthogonal partitioning of EP groups and DP groups
- EP × TP: further tensor parallelism inside an Expert (for very large Experts)
- EP × PP: how MoE layers and dense layers are placed in the pipeline
- Case studies: parallelism configs in DeepSeek-V3 / Mixtral

### 6. Hands-on lab

- Implement a minimal MoE layer + Top-2 Router, and print the token routing distribution
- Simulate the All-to-All dispatch/combine process
- Observe load imbalance, then add aux loss and compare

---

## Learning goals

- Draw the MoE layer data flow: Router → dispatch → Expert → combine
- Explain what each of Expert Parallelism's two All-to-Alls carries
- Explain what causes MoE load imbalance and the two families of fixes: aux loss and aux-loss-free
- Analyze communication-domain partitioning when EP is combined with DP/TP/PP
