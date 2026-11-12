---
title: "Baidu AI Infra campus hire second round (3)"
description: "Real Baidu AI Infra campus-hire second-round interview questions covering inference optimization"
tags: ["Inference optimization"]
sourceHash: "14b22afb60cdfbad"
---

## Large models and search

1. How does Beam Search work? How does it differ from direct sampling?
2. How prompts are generated, what they optimize for, and which task types they match
3. What common methods reduce large-model hallucination?
4. When search accuracy is low, how do you debug?
5. Between recall and ranking, which more often becomes the bottleneck?
6. Can generative search guarantee top-1 accuracy? How do you improve it?
7. Basic idea of generative ranking

## Model training and fine-tuning

8. Compare BERT vs GPT pretraining approaches and training-detail differences
9. How do you build ranking-model training data? How do you handle positive/negative imbalance?

## Engineering practice

10. What relevance optimizations did you do in the project? Any quantified evaluation results?
11. How do you keep Agent services highly available and robust?
