---
title: "Ant Group AI Infra internship first round (1)"
description: "Real Ant Group AI Infra internship first-round interview questions covering inference optimization, operator optimization, high-performance computing, and related topics"
tags: ["Inference optimization", "Operator optimization", "High-performance computing"]
sourceHash: "4b843cc598354ffa"
---

### Fundamentals

1. How do you decide whether an operator needs optimization? What evidence and analysis path do you use? (hotspot analysis; bottleneck type: compute / memory / scheduling overhead)
2. Faced with a not-yet-quantized model, how do you systematically push quantization forward? (model-structure analysis, deployment target, quantization granularity, calibration choice, iterative accuracy/performance validation)
3. After AWQ-style methods add a dequantization step, why can overall inference still speed up?
4. Traits of static versus dynamic graphs? Why convert dynamic graphs to static before fusion, and how do you handle conversion cost itself?
5. What are vLLM's core design ideas for inference scheduling and KV Cache management?

### Project experience

6. Hard problems you hit in the internship? (e.g. OOM; abnormal GPU-memory growth and random crashes in long multi-turn inference; debug path and fix)
7. What is the underlying implementation of the Timing plugin you built? Why not use Nsight / NCU and similar profilers directly?
8. Do different models differ much in local deployment and quantization?
9. How do you choose graph optimizations, and what is the development flow?

### Coding

10. Copy a linked list with random pointers.
