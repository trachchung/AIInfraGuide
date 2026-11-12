---
title: "第5章：GPU 硬件概论"
description: "理解 GPU 架构设计哲学、存储层次、主流 GPU 规格对比和互联拓扑，为 CUDA 编程建立硬件认知"
pubDate: 2026-04-16
category: "prerequisites"
order: 5
tags: ["GPU", "硬件架构", "NVLink", "Roofline", "存储层次"]
---

## 本章简介

在动手写 CUDA 代码之前，必须先搞清楚"工厂怎么布局"——GPU 的硬件架构直接决定了你写出的代码能跑多快。

**GPU 架构总览**对比 CPU vs GPU 的设计哲学差异（延迟优化 vs 吞吐优化），介绍 SM、CUDA Core、Tensor Core 的层次结构，以及 Warp（32 线程最小调度单位）的概念。

**GPU 存储层次**详解从寄存器到 HBM 再到主机内存的完整存储层次，包括各级存储的容量、带宽、延迟量级，以及 Memory Wall 问题——为什么显存带宽往往比算力先成为瓶颈。

**主流 GPU 规格对比**列出 A100/H100/H200/B200 的关键参数，引入 Arithmetic Intensity 和 Roofline Model 的概念。

**互联拓扑**介绍单机 NVLink/NVSwitch 和多机 InfiniBand 网络，解读 `nvidia-smi topo -m` 输出，理解为什么互联带宽直接决定并行策略的选择。
