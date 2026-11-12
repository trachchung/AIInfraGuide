---
title: "Ant Group AI Infra internship first round (2)"
description: "Real Ant Group AI Infra internship first-round interview questions covering inference optimization, training optimization, operator optimization, and related topics"
tags: ["Inference optimization", "Training optimization", "Operator optimization"]
sourceHash: "74f1cf22a8ef216b"
---

### Fundamentals

1. What do W8A8 and W4A16 mean? When weights are quantized to 4-bit, why do activations usually still need higher precision?
2. Traits of uniform versus non-uniform quantization? What costs does non-uniform quantization pay in compute complexity, scale/zero-point storage, and hardware acceleration fit?
3. How do you analyze bottlenecks on an inference path? What optimizations work best? If you still miss the target after optimizing, where does the gap usually come from?
4. How do you decide whether a set of operators is worth fusing? What criteria? (chain length, frequency, dependencies, whether intermediate outputs are referenced by other branches)
5. Which metrics matter most when profiling? (total time breakdown, warp divergence, pipeline efficiency under multiple streams)

### Project experience

6. Single-operator optimization practice: how you replaced a hand-written impl with a high-performance library; how you unit-tested and tested the full network
7. Locating and fixing stability issues across multi-turn inference (root-cause analysis for random crashes and OOM)
8. In a static graph, how do you pick operator groups that are good fusion candidates?

### Coding (CUDA)

9. Implement a Histogram operator: input is integers in 0–9; output is counts per bucket. Discuss performance issues of a naive atomic impl, and an optimization that reduces via local histograms in shared memory.
