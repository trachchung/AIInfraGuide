---
title: "Chapter 7: AI compilers"
description: "Triton's Block-level programming model, torch.compile's compilation modes, and how TVM and XLA differ in role"
tags: ["Triton", "torch.compile", "AI compilers", "TorchInductor", "TVM"]
sourceHash: "add9d72071ff1f99"
---

## About this chapter

Hand-written CUDA kernels can hit peak performance, but they cost a lot to build. AI compilers sit at a higher level of abstraction and trade some of that peak for much less engineering work. This chapter covers three of the main approaches.

**Triton** is the AI kernel framework most people are watching right now. It uses a Block-level programming model (CUDA is Thread-level), which lowers the bar for GPU programming a lot. This section covers the basic Triton APIs (`tl.load` / `tl.store` / `tl.dot` / `tl.where`) and walks through the official Fused Softmax tutorial.

**torch.compile** is the compilation mode in PyTorch 2.x. It has TorchDynamo (graph capture at the Python bytecode level) and TorchInductor (backend codegen). The focus is Graph Breaks—what interrupts compilation, how to avoid them—and how to judge whether the compile actually pays off.

**TVM / XLA (overview)** covers how the two fit in: TVM targets cross-hardware compile and optimize, XLA targets whole-graph optimization, and how both relate to Triton and torch.compile.
