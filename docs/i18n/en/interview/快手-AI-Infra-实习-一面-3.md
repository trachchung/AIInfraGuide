---
title: "Kuaishou AI Infra internship first round (3)"
description: "Real Kuaishou AI Infra internship first-round interview questions covering inference optimization, training optimization, and operator optimization"
tags: ["Inference optimization", "Training optimization", "Operator optimization"]
sourceHash: "d13e24878c05638a"
---

### Theory

1. What common methods compress KV Cache?
2. Explain MHA, MQA, and GQA respectively, and how KV broadcast is implemented. What is the data correspondence between Multi-head Latent Attention (MLA) and GQA? Given hidden_size and RoPE dim, how many GQA heads does MLA map to?
3. What does each DeepSpeed ZeRO-1/2/3 stage optimize? If the model has N parameters and uses Adam, how does ZeRO-1 split GPU memory across P GPUs? How do data types (FP32/FP16) affect memory use?
4. How does SmoothQuant work? Why Smooth? How do you set hyperparameters? How do you tell whether a model fits SmoothQuant? If you use per-layer activation distributions, do you watch input channels or output channels?
5. How do AWQ and GPTQ work respectively, and how do they differ?
6. What GPU distributed communication primitives exist? When do All-Gather and All-to-All fit?

### Project deep dive

7. Detail sparse KV Cache compute implementation and how you write the vLLM Triton kernel. Why doesn't sparse KV Cache compute use a mask?
8. Why did the project pick different quantization methods for different scenarios? When do GPTQ and SmoothQuant each fit?
9. What does the distillation-model pipeline look like? Which key techniques did you use?
