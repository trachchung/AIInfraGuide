---
title: "Zhuoyu AI Infra internship"
description: "Real Zhuoyu AI Infra internship interview questions covering inference optimization and operator optimization"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "5f8f68262f252e23"
---

## Fundamentals

1. What stages does large-model inference split into? What are the traits of each stage?
2. What is the relationship between operators and tensors?
3. How are compute graphs usually built?
4. What is the definition of throughput, and how do you measure it in inference?
5. How do you tell memory-bound from compute-bound?
6. With model and operators fixed, what factors still affect inference speed?
7. Does prompt length affect inference speed? What about the decode stage? How does context length matter?
8. How do you load weight files into a model you built yourself?

## Inference optimization

9. In prefill, are there ways to speed up KV Cache generation?
10. In prefill, do tokens depend on each other? Can they be processed in parallel?
11. Does sparse KV Cache require changing the training pipeline, or is it pure engineering optimization?
12. What further techniques can raise large-model inference performance?

## Tools and methods

13. How do you use Nsight for performance optimization? Which metrics matter most?
14. Have you looked into compute architecture or chip architecture topics?

## Project experience

15. Introduce the research direction of your thesis.
16. Introduce key components and module design in the project.
