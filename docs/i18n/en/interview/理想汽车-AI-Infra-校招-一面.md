---
title: "Li Auto AI Infra campus hire first round"
description: "Real Li Auto AI Infra campus-hire first-round interview questions covering inference, training, and operator optimization"
tags: ["Inference optimization", "Training optimization", "Operator optimization"]
sourceHash: "e9b95bb866a85491"
---

### Fundamentals

1. What is the principle and implementation of PyTorch DDP?
2. Trace the development of CV (from AlexNet onward, including architecture and activation-function evolution).
3. Trace the development of NLP (RNN -> LSTM -> Transformer).
4. Compare pros and cons of RNN vs Transformer, with emphasis on Transformer’s parallelism advantage.
5. What parallel methods exist in model training? (data parallelism, tensor parallelism, pipeline parallelism, etc.)
6. Where does Transformer meet CV? (e.g. ViT’s design)
7. What methods exist for lightweight model deployment? (quantization, pruning, compression, etc.)
8. What are TensorRT’s strengths and weaknesses? When an operator is missing, how do you solve it with a custom Plugin?
9. What does Batch Normalization do? Why can it speed up convergence?

### Coding

10. Shortest path from top-left to bottom-right (simple DP).
