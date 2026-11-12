---
title: "Small company AI Infra internship (2)"
description: "Real small-company AI Infra internship interview questions covering inference, training, and operator optimization"
tags: ["Inference optimization", "Training optimization", "Operator optimization"]
sourceHash: "bdf9922858adb21f"
---

### Fundamentals

1. How do pretraining and SFT differ in loss design and dataset construction?
2. In the Transformer architecture, how does Megatron tensor parallelism (TP) shard parameter matrices? For the first and second linear layers in the MLP, are they row-sharded or column-sharded? Which communication primitives does each use?
3. Briefly note technical points from DeepSeek papers that stuck with you (e.g. the FP8 training scheme).

### Training optimization

4. Introduce pipeline parallelism for pretraining; explain the design of 1F1B scheduling and DualPipe.
5. Walk through the full large-model RL pipeline. Which model roles are involved? What is the core difference between PPO and GRPO?
6. In RL training, roughly what fraction of time is spent in Rollout? Roughly what is the Policy model’s MFU? Give the MFU formula and what the 6Nd formula means.
7. What common optimizations exist for the RL Rollout stage (e.g. Rollout quantization, async Rollout)?
8. In the RL training pipeline, how do you sync pretrained weights into the inference engine?

### Inference optimization

9. Introduce how Continuous Batching works in vLLM or SGLang.
