---
title: "Enflame AI Infra internship first round"
description: "Real Enflame AI Infra internship first-round interview questions covering inference optimization, operator optimization, and related topics"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "8c6b902c1ebe6063"
---

Below is what the Enflame AI Infra internship first round covers.

### Project experience

1. Deep follow-up on internship experience
2. Deep follow-up on project experience

### Quantization

3. How you choose a quantization strategy: why INT8; how A100 and H100 support different quantization precisions
4. Do you quantize model weights or KV Cache, and how do you set the scale
5. After quantization, did you evaluate accuracy loss

### Operator development

6. Triton operator implementation logic, including tiling and related strategies
7. Which official baseline you compared against, and which dtypes
8. Where the speedup numbers come from; dynamic tiling strategy and operator config
9. Did you consider writing operators in CUDA instead of Triton, and why you chose Triton
10. Did you profile; how did memory throughput and related metrics look; what next optimizations

### Inference optimization

11. What share of end-to-end system latency does the Attention module take
12. Is decode compute-bound or memory-bound; which side does KV Cache quantization improve
13. Theoretical peak memory bandwidth of A100
