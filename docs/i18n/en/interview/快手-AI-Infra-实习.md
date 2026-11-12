---
title: "Kuaishou AI Infra internship"
description: "Real Kuaishou AI Infra internship interview questions covering inference optimization and operator optimization"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "fda726ebd5ca5450"
---

### Theory

1. Besides the quantization plan used in the project, which other quantization methods do you know?
2. From a quantization-error angle, what data-distribution conditions are more ideal?
3. With outliers, more uniform distributions usually quantize better. That holds for INT8, FP8, and INT4 — why does it stop holding for FP4?
4. Describe CUDA GEMM optimization methods, and how you choose Shared Memory size during the compute.

### Project deep dive

5. Introduce the main internship work.
6. Detail the project's technical design and implementation.

### Live coding

7. Write a Norm operator in CUDA. Input: x[N]; output: norm = (x - u) / o, where u = mean(x[N]), o = sqrt(sum((x - u)^2) / N).
