---
title: "Chapter 1: CUDA programming intro"
description: "Set up a CUDA development environment, learn the Grid/Block/Thread programming model and the memory model, and write your first practical CUDA kernel"
tags: ["CUDA", "Programming model", "Memory model", "Kernel"]
sourceHash: "ec526a1f6db6b0c8"
---

## About this chapter

This chapter is the starting point for CUDA programming. You set up the development environment from scratch and write your first high-performance kernel.

The environment setup covers installing and managing CUDA Toolkit versions, the nvcc toolchain and CMake integration, and writing a first Hello World kernel.

The programming model section walks through the three-level thread hierarchy Grid/Block/Thread, how to compute thread indices (threadIdx/blockIdx/blockDim/gridDim), the kernel launch syntax `<<<gridDim, blockDim>>>`, and how to choose Block size.

The memory model section covers global memory, shared memory, registers, constant memory, and unified memory—what each is good for and when to use it.

The first practical kernel uses vector addition to walk through the full path: cudaMalloc → cudaMemcpy → kernel launch → error checking, and compares CPU vs GPU time.
