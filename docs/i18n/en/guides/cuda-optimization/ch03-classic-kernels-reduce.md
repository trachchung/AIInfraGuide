---
title: "Chapter 3: Classic operator implementations — Reduce"
description: "Three progressive versions of Reduce (naive, shared-memory tree reduce, Warp Shuffle) as a way to learn step-by-step CUDA operator optimization"
tags: ["CUDA", "Reduce", "Shared memory", "Warp Shuffle", "Operator optimization"]
sourceHash: "cf8dc4a8e5c43081"
---

## About this chapter

Reduce is one of the classic parallel algorithms, and a good first case for learning CUDA optimization. Through three progressive versions, you walk the full loop of finding a bottleneck, fixing it, and measuring the gain.

The naive version uses global memory and atomic add. It is the simplest and the slowest; performance analysis shows where the time goes.

Shared memory plus tree reduce does the reduction inside a block in shared memory, removes warp divergence, and deals with the second reduction across multiple blocks.

The Warp Shuffle version replaces shared memory with `__shfl_down_sync`, removes bank conflicts entirely, and gets an efficient warp-level Reduce.

Multi-level reduction and large inputs cover the Grid-stride loop trick for arrays that do not fit in one pass.

Hands-on: implement three versions of Reduce Sum, compare throughput in Nsight Compute, and explain where each step actually saves time.
