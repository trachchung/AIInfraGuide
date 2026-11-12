---
title: "Chapter 4: Classic operator implementations — GEMM"
description: "From naive matrix multiply through Block-Warp-Thread tiling, vectorized loads, bank-conflict removal, and double buffering — a full playbook for CUDA GEMM optimization"
tags: ["CUDA", "GEMM", "Matrix multiplication", "Tiling", "Tensor Core", "cuBLAS"]
sourceHash: "475c129fa983c763"
---

## About this chapter

GEMM (General Matrix Multiply) is the central operator in deep learning — linear layers, the QKV projections in Attention, and FFN compute are all GEMM under the hood. This chapter builds a high-performance GEMM from scratch so you see the bridge between hardware and the frameworks above it.

1. Background and problem setup covers where GEMM sits in deep learning, problem scale, compute and memory-access traits, and what we are optimizing for.

2. Performance analysis brings in arithmetic intensity, the Roofline model, and evaluation metrics, so every later step has a quantitative frame.

3. The naive baseline is the obvious triple-loop kernel. It exposes the memory-bound bottleneck from repeated global-memory access.

4. Thread-block tiling cuts the matrices into tiles loaded into shared memory, quantifies how much global traffic drops, and gets the first round of reuse.

5. Thread-level tiling keeps sub-block results in registers so each thread computes several outputs, raising compute density and reuse further.

6. Warp-level tiling organizes tile layout by warp, maps threads to hardware execution units, and cuts redundant work and sync cost inside a warp.

7. Vectorized access with `float4` merges loads and stores, cuts memory-instruction count, and raises effective bandwidth use.

8. Bank-conflict removal uses padding or address remapping so shared-memory bank conflicts go away and throughput returns toward peak.

9. Double buffering and pipelining keeps two buffers in shared memory so loads and compute overlap and hide global-memory latency.

10. Ampere async-copy pipelines use `cp.async` to make global-to-shared copies asynchronous, build a multi-stage software pipeline, and squeeze more latency out.

11. SASS-level tuning and register scheduling go into instruction-level work: register allocation, instruction scheduling, and bank conflicts, reading cuBLAS SASS to see where peak performance comes from.

12. Parameter selection and tuning wraps up how to choose tile size, thread layout, and block count, with tuning ideas for different matrix sizes and hardware.
