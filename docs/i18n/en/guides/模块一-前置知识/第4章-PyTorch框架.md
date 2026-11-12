---
title: "Chapter 4: The PyTorch framework"
description: "Master PyTorch Tensors, autograd, the Module training loop and debugging tools, then build a GPT-2-scale small model from scratch and train it end to end"
tags: ["PyTorch", "autograd", "Training loop", "Performance profiling"]
sourceHash: "a08faea24505e53c"
---

## About this chapter

PyTorch is the dominant training framework in AI Infra, and the foundation for the distributed training and inference work that comes later. The goal of this chapter is to get you fluent in PyTorch's core mechanisms so you can develop and debug models on your own.

The Tensor and autograd section covers Tensor operations, CPU/GPU device management, how the autograd computation graph works, and gradient accumulation and zeroing.

The Module and training loop section walks through a full training cycle: DataLoader → forward → loss → backward → optimizer.step, including learning rate scheduling and Checkpoint save/load.

The debugging and profiling section covers `torch.cuda.memory_summary()` for GPU memory, `torch.profiler` for performance analysis, and common errors (shape/device mismatch, OOM).

Hands-on lab: implement a GPT-2-scale small model from scratch in PyTorch, and run the full flow of data loading, training, evaluation and saving on a single GPU.
