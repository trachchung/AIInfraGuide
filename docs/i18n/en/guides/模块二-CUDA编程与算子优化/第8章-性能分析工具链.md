---
title: "Chapter 8: Performance profiling toolchain"
description: "Nsight Systems, Nsight Compute, and PyTorch Profiler—the three tools for systematic performance diagnosis"
tags: ["Nsight Systems", "Nsight Compute", "PyTorch Profiler", "Performance analysis"]
sourceHash: "ba11f43ec78f95f9"
---

## About this chapter

You cannot optimize what you cannot measure—tuning without a profiler is guessing. This chapter covers the three core profilers used in AI Infra work.

**Nsight Systems** looks at the full CPU–GPU timeline. You use it to find where GPU idle gaps come from (CPU bottleneck, waiting on communication, kernel launch overhead), capture a training iteration's trace, and read it at a high level.

**Nsight Compute** drills into a single kernel. The main skill is reading the SOL (Speed of Light) panel—deciding whether a kernel is memory bound or compute bound—plus the Shared Memory and Warp State panels, and comparing two kernel versions side by side.

**PyTorch Profiler** ties `torch.profiler` into TensorBoard so you can find the slowest operators and measure what torch.compile actually gains you.

**Hands-on**: capture one training iteration with Nsight Systems and point out where the GPU idle gaps come from; compare the three Reduce kernel versions with Nsight Compute reports.
