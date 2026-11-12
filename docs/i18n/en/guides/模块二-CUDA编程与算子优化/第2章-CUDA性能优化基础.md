---
title: "Chapter 2: CUDA performance optimization basics"
description: "Warp execution, memory-access tuning, Occupancy, and synchronization—the core playbook for making CUDA kernels fast"
tags: ["CUDA", "Warp", "Coalesced access", "Occupancy", "Performance optimization"]
sourceHash: "5c1e74ef36575373"
---

## About this chapter

Getting CUDA code to run is only the start. Getting it to run fast is what AI Infra engineers are paid for. This chapter builds the core playbook for CUDA performance work.

The Warp and execution model section covers SIMT execution, the cost of Warp Divergence, and Warp Shuffle for fast data exchange between threads.

Memory-access tuning is usually the biggest lever: Coalesced Access decides global-memory efficiency, Bank Conflict hurts shared-memory performance (padding fixes it), and vectorized loads (float4/int4) push bandwidth utilization further.

Occupancy and resource allocation define what Occupancy means, what drives it (registers, shared memory, Block size), and why higher is not always better—you balance Latency Hiding against Resource Utilization.

Synchronization and atomics cover `__syncthreads()` for block-level sync and how atomics behave, including their performance cost.
