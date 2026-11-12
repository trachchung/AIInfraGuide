---
title: "AI Infra mixed interview question bank (3)"
description: "Real AI Infra interview questions covering inference optimization, training optimization, operator optimization, and high-performance computing"
tags: ["Inference optimization", "Training optimization", "Operator optimization", "High-performance computing"]
sourceHash: "0eb8b78f11362561"
---
## Basics

1. What should you watch for when accessing CUDA Global Memory versus Shared Memory?
2. Given the total number of tokens needed for training, how do you estimate wall-clock time to finish training?
3. What mainstream optimizations apply to the Prefill stage and to the Decode stage?
4. What does two-batch overlap mean? In which cases does it become a net negative?
5. How does Megatron-LM implement communication optimizations?
6. Multi-node PD separation adds KV Cache transfer cost. Why is PD separation still worth doing?
7. Why can Muon and AdamW not be mixed across Pretrain and Post-train stages?
8. How do you view cross-SM PD separation and AF separation?
9. What are the key optimization points in DeepSeek-V3?
10. How do DeepSeek-DSA, NSA, and MoBA differ?
11. Which communication primitives does NCCL include? How many communication steps does one All-Reduce parameter update need?
12. With small payloads, if you use NVSHMEM so each GPU reads other GPUs' data and reduces locally, what is the advantage over Ring All-Reduce?
13. How should you design the parallelism strategy when training very long sequences?
14. When porting an Ampere operator to Hopper, what needs to be upgraded?
