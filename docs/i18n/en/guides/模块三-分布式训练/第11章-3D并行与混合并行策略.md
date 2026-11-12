---
title: "Chapter 11: 3D parallelism and hybrid parallelism strategies"
description: "Putting every parallelism dimension together—topology design for TP×PP×DP×EP×CP, communication-domain mapping, rank layout, and how to pick the best mix for a given cluster"
tags: ["3D parallelism", "Hybrid parallelism", "Topology design", "Communication domain", "Megatron-LM"]
sourceHash: "3e959e797e257fe3"
---

## About this chapter

Earlier chapters treated each parallelism strategy on its own. Real large-model training almost never uses just one—thousand-GPU jobs are usually TP + PP + DP as 3D parallelism; add MoE and it becomes 4D; add long sequences and you have 5D. This chapter is the final assembly shop for parallelism: how orthogonal dimensions combine, how communication domains map onto physical hardware, how ranks are laid out, and how to design the best parallel config for a given cluster.

---

## Chapter structure

### 1. Why hybrid parallelism

- Ceiling of a single strategy: DP does not save parameters, TP does not leave the node, PP has Bubble
- Dimensional orthogonality: different strategies shard different things, so they can stack
- The constraint `world_size = TP × PP × DP` (× EP × CP)

### 2. Principles for splitting communication domains (focus of this chapter)

- **Map by communication frequency / bandwidth need onto hardware**:
  - TP: communicates every layer, most latency-sensitive → intra-node NVLink (within 8 GPUs)
  - CP/SP: communication inside Attention → usually the same domain as TP, or intra-node
  - PP: ships activations at each micro-batch boundary → can cross nodes over InfiniBand
  - DP: one gradient sync per step → cross-node, lowest bandwidth demand
- Communication volume largest to smallest: TP > CP > PP > DP; assign hardware bandwidth from high to low to match

### 3. Rank layout and building process groups

- Mapping a rank to multi-dimensional coordinates (tp_rank, pp_rank, dp_rank)
- Effect of dimension order (which dimension sits "next to" physically adjacent GPUs)
- Walkthrough of Megatron-LM `initialize_model_parallel()` grouping logic
- Creating an NCCL communicator per communication group

### 4. Topology design in practice

- Example: 64-GPU cluster (8 nodes × 8 GPUs)
  - Plan A: TP=8, PP=4, DP=2
  - Plan B: TP=4, PP=8, DP=2
  - How to choose: look at layer count, per-layer size, and network bandwidth
- Example: configuration thinking for training 10B/100B-class models on a thousand-GPU cluster
- Multi-dimensional layout after adding EP/CP

### 5. Joint estimate of memory and throughput

- Given a (TP, PP, DP) mix, estimate per-GPU memory
- Estimate end-to-end throughput: compute time + communication time on each dimension + PP Bubble
- Tuning method: first satisfy "it fits", then optimize "it runs fast"

### 6. Hands-on labs

- For a given model + cluster, hand-calculate memory and communication volume for 3 parallel plans and pick the best
- Read Megatron-LM's `parallel_state.py` and draw the rank grouping diagram

---

## Learning goals

- Explain why multiple parallelism dimensions can combine orthogonally, and write the world_size constraint
- Map TP/CP/PP/DP onto NVLink/IB/Ethernet correctly by communication frequency
- Read Megatron's rank-layout logic and draw the communication-group diagram
- Design and compare several 3D / multi-dimensional parallel plans for a given model scale and cluster topology
