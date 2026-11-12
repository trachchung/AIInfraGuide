---
title: "Zhipu AI Infra internship first round"
description: "Real Zhipu AI Infra internship first-round interview questions covering inference optimization"
tags: ["Inference optimization"]
sourceHash: "4f6614f4357e29f6"
---

## Basics

1. Explain the differences among per-tensor, per-channel, and per-group quantization granularities. Which is the finest?
2. What is the core difference between minmax calibration and percentile calibration?
3. Besides minmax and percentile, what other common calibration algorithms are there? Briefly describe the basic ideas of KL-divergence calibration and MSE calibration.
4. Explain SmoothQuant's quantization granularity and how it works, and compare its workflow with AWQ and GPTQ.
5. What is the quantization principle of NV FP4? On which dimension is the scale factor computed? How is the storage format designed?
6. In real deployments, what common quantization target formats are there? Compare the applicable scenarios for FP8 and NV FP4.

## Project experience

7. Introduce your internship project experience.

## Coding

8. Implement minmax calibration and percentile calibration respectively.
