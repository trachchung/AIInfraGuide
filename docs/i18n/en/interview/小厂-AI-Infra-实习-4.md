---
title: "Small company AI Infra internship (4)"
description: "Real small-company AI Infra internship interview questions covering inference and operator optimization"
tags: ["Inference optimization", "Operator optimization"]
sourceHash: "23320300f7edf0a5"
---

### Project experience

1. Introduce prior deployment-related project experience.
2. The project used TensorRT for quantization — was it a PTQ scheme? Why choose TensorRT for quantization? What concrete inference-speed requirements did the project have?
3. Describe how you used CUDA to accelerate image preprocessing, and which optimization strategies you applied.
4. Give a real example of analyzing and optimizing a CUDA operator (e.g. why vectorized loads did not speed things up).
5. Why choose the RK3588 chip? Describe the full flow of environment setup and model deployment.

### Fundamentals

6. Explain the GPU memory hierarchy, including characteristics and roles of L1 Cache, shared memory, and L2 Cache.
7. What ways are there to optimize shared memory? What is a Bank Conflict?
8. Do you know performance optimization methods on ARM platforms?
9. Briefly describe ByteTrack’s matching flow.

### Coding

10. Use CUDA to sum one million floats; describe the approach.
