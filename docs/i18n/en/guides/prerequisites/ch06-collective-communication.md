---
title: "Chapter 6: Collective communication basics"
description: "The communication primitives of distributed training (AllReduce, AllGather and others), the Ring and Tree algorithms, and how to use and tune NCCL"
tags: ["Collective communication", "AllReduce", "NCCL", "Ring algorithm"]
sourceHash: "691a50c42b4629b7"
---

## About this chapter

Collective communication is the circulatory system of distributed training. You need to understand the primitives and the algorithms behind them before you can design a parallelism strategy.

The primitives part covers point-to-point communication (Send/Recv) and the collectives: Broadcast, Reduce, AllReduce, AllGather and ReduceScatter. Each comes with a data flow diagram and a formula for how much data it moves, so you can build an intuition for them.

The algorithms part focuses on how Ring AllReduce works and how much data it sends (2(N-1)/N × the data size). It also covers Tree AllReduce and the basic idea of overlapping communication with computation.

The NCCL part explains where NVIDIA's collective communication library fits, how to use it, which environment variables to tune, and how to measure communication performance with nccl-tests.
