---
title: "第4章：PyTorch 框架"
description: "掌握 PyTorch 的 Tensor、自动微分、Module 训练流程和调试工具，并从零实现一个 GPT-2 级别小模型完成完整训练"
pubDate: 2026-04-16
category: "prerequisites"
order: 4
tags: ["PyTorch", "autograd", "训练流程", "性能分析"]
---

## 本章简介

PyTorch 是 AI Infra 领域最主流的训练框架，也是后续分布式训练和推理优化的基础载体。本章目标是让你熟练掌握 PyTorch 的核心机制，能够独立完成模型开发和调试。

**Tensor 与自动微分**部分涵盖 Tensor 操作、CPU/GPU 设备管理、autograd 计算图机制以及梯度累积与清零。

**Module 与训练流程**部分走通完整训练循环：DataLoader → forward → loss → backward → optimizer.step，包括学习率调度和 Checkpoint 保存/加载。

**调试与性能分析**部分掌握 `torch.cuda.memory_summary()` 查看显存、`torch.profiler` 性能分析，以及常见错误排查（shape/device mismatch、OOM）。

**动手实验**：用 PyTorch 从零实现 GPT-2 级别小模型，在单卡上完成数据加载、训练、评估、保存的完整流程。
