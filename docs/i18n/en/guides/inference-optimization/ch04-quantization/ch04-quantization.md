---
title: "Chapter 4: Quantization"
description: "From scale and granularity through SmoothQuant, GPTQ/AWQ, KV cache quantization, and FP8/NVFP4/MXFP4, then pick a scheme and run controlled experiments on vLLM v0.32.0"
tags: ["Quantization", "SmoothQuant", "GPTQ", "AWQ", "FP8", "NVFP4", "KV Cache quantization"]
sourceHash: "abe8c0fba9d0ee8e"
---

## About this chapter

Quantization is writing numbers with a coarser ruler: each value takes less space and reads and writes go faster, at the cost of coarser readings. The hard part is not "how many bits," but how to keep the coarser values from hurting the model's output much, and how to make the hardware actually take the win from lower bit widths.

This chapter goes in four steps: concepts, methods, formats, then putting it into practice:

- **Basic concepts**: scale and zero-point, symmetric vs asymmetric, bit width and step size, Per-tensor / Per-channel / Per-group / Per-token granularity, calibration, and PTQ vs QAT.
- **Three things you quantize**: W8A8 uses SmoothQuant to move activation outliers onto the weights; W4A16 compresses weights with GPTQ's error compensation or AWQ's activation-aware scaling, then runs them efficiently with kernels like Marlin; the KV cache uses KIVI's grouping or an FP8 cache to cut how much each past token costs.
- **Low-bit floating point**: how E4M3/E5M2 split range vs precision, why FP4 needs micro-block scaling, and where NVFP4 and MXFP4 differ in scale design.
- **Choosing and shipping**: filter candidates by bottleneck and hardware, see how vLLM recognizes checkpoints and picks kernels, and compare FP16 against AWQ INT4 on throughput and generation quality.

> ⏰ **Currency:** The principles do not depend on any framework. vLLM parameters, scheme names, hardware support, and kernel selection here are based on v0.32.0 (released 2026-10-18). Quantization support matrices move fast; when you change versions, check the official docs and the startup logs.

## Sections

- [**4.1 Quantization basics: scale, granularity, and PTQ/QAT**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/): affine quantization and dequantization, symmetric vs asymmetric, bit width and step size, granularity, calibration, PTQ and QAT
- [**4.2 W8A8 quantization: how SmoothQuant handles activation outliers**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/): channel outliers, equivalent scaling, migration strength α, INT8 GEMM, and LLM Compressor recipes
- [**4.3 Weight-only INT4: GPTQ, AWQ, and Marlin**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/): W4A16 storage, GPTQ second-order compensation, AWQ activation-aware scaling, Marlin fused kernels, and how vLLM picks a kernel
- [**4.4 KV cache quantization: KIVI and FP8 caches**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/): estimating KV footprint, KIVI's Key/Value grouping and residual cache, FP8 KV scale calibration, skipping layers, and long-context checks
- [**4.5 FP8 and NVFP4/MXFP4: how low-bit floats work**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/): E4M3/E5M2, FP8 scales, E2M1 and micro-block scaling, NVFP4 vs MXFP4, and hardware conditions for running them
- [**4.6 Choosing a quantization scheme and hands-on with vLLM**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/): picking candidates by bottleneck, method / format / kernel layers, online quantization entry points, and FP16 vs AWQ INT4 experiments
