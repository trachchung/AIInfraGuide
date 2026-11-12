---
title: "Chapter 9: On-device inference"
description: "Resource constraints, heterogeneous hardware, runtimes, and optimization methods for on-device inference, plus a reproducible performance evaluation workflow"
tags: ["On-device inference", "Edge AI", "ExecuTorch", "Quantization", "Heterogeneous computing"]
sourceHash: "4928b909f69d712a"
---

## About this chapter

Deploying a model to a phone, car, camera, PC, or embedded device is a different problem from deploying it in a data center. On-device hardware has tighter limits on memory, power, cooling, and software, and the hardware models are more fragmented. A fast one-shot run does not mean the model can keep running stably.

This chapter follows the full engineering path: how a model is exported from PyTorch into an on-device program, how the runtime assigns work to the CPU, GPU, NPU, or DSP, and how quantization, graph optimization, memory planning, and performance evaluation put together a reliable on-device inference system.

## Sections

- 9.1 On-device inference basics: from model export to heterogeneous hardware execution: on-device constraints, the software and hardware stack, runtime choices, optimization methods, benchmarks, and a minimal ExecuTorch / XNNPACK example
