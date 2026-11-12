---
title: "第6章：集合通信基础"
description: "掌握分布式训练的通信原语（AllReduce、AllGather 等）、Ring/Tree 通信算法和 NCCL 的使用与调优"
pubDate: 2026-04-16
category: "prerequisites"
order: 6
tags: ["集合通信", "AllReduce", "NCCL", "Ring算法"]
---

## 本章简介

集合通信是分布式训练的"血管系统"，理解通信原语和算法是设计并行策略的前提。

**通信原语**部分覆盖点对点通信（Send/Recv）和集合通信（Broadcast、Reduce、AllReduce、AllGather、ReduceScatter），配合数据流示意图和通信量公式，建立直觉。

**通信算法**部分重点讲解 Ring AllReduce 的原理与通信量分析（2(N-1)/N × 数据量）、Tree AllReduce，以及通信与计算 overlap 的核心思想。

**NCCL**部分介绍 NVIDIA 官方集合通信库的定位、基本用法、环境变量调优，以及使用 nccl-tests 进行通信性能测试。
