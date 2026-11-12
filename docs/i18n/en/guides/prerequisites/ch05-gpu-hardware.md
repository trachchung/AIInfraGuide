---
title: "Chapter 5: GPU hardware overview"
description: "GPU architecture design philosophy, the memory hierarchy, mainstream GPU specs and interconnect topology, so you have the hardware picture before writing CUDA"
tags: ["GPU", "Hardware architecture", "NVLink", "Roofline", "Memory hierarchy"]
sourceHash: "9e53e5b54c1923ef"
---

## About this chapter

Before you write CUDA code, you need to understand how the factory is laid out. GPU hardware architecture directly decides how fast your code can run.

The GPU architecture overview contrasts the design philosophies of CPU vs GPU (latency optimized vs throughput optimized), covers the SM / CUDA Core / Tensor Core hierarchy, and the idea of a Warp (the 32-thread minimum scheduling unit).

The GPU memory hierarchy walks through the full stack from registers to HBM to host memory, including capacity, bandwidth and latency orders of magnitude at each level, and the Memory Wall problem: why memory bandwidth often becomes the bottleneck before compute does.

The mainstream GPU specs section lists key parameters for A100/H100/H200/B200 and introduces Arithmetic Intensity and the Roofline Model.

The interconnect topology section covers single-node NVLink/NVSwitch and multi-node InfiniBand, how to read `nvidia-smi topo -m` output, and why interconnect bandwidth directly shapes which parallelism strategy you pick.
