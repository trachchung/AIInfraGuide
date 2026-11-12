---
title: "MiniMax AI Infra first round"
description: "Real MiniMax AI Infra first-round interview questions covering training optimization"
tags: ["Training optimization"]
sourceHash: "135eb890ab8d4bb8"
---
## Theory

1. What is RAG? How does it improve generation quality? What weaknesses do standard RAG setups usually have? How does it differ from a classic retrieve-then-generate pipeline?
2. What is the main SFT workflow? How are training datasets usually built and filtered?
3. From a math angle, what is the core meaning of Attention in Transformer?
4. Do you know Agent techniques? What do you gain by wrapping RAG as an Agent?
5. How do you systematically evaluate a RAG system? Which metrics or standard benchmarks are useful?
6. How does LoRA work? At inference time, does a LoRA-tuned model still need to load an extra Adapter?
7. What is the core difference between PPO and DPO for LLM alignment? What should you watch for when training with DPO?
8. Have you used GRPO? Briefly explain.
9. In multi-turn Agent apps, what weaknesses does Attention show?
10. After SFT, what Post-Training methods are common (for example RLHF)? What does each optimize for?
11. When fine-tuning Qwen, how do you choose the training-stage strategy? What drives the loss design?
12. How is Prompt auto-recommendation optimized? Have you tried Prompt compression or Embedding-based representations for efficiency?

## Project deep dive

13. How is tool-call scheduling designed? Does the system support Fallback on failures?
14. How does the project's Modular Agent do multi-step planning?
15. What does the Agent evaluation cover? How do you measure planning quality and hallucination rate?

## Open scenario

16. An Agent's reasoning path calls 3 external tools in sequence, and high concurrency makes end-to-end latency high. What engineering moves would you use to cut latency?

## Live coding

17. In PyTorch, write the SFT loss computation (watch the label shift-right alignment)
