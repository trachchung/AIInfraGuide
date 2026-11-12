---
title: "OPPO AI Infra internship second round"
description: "Real OPPO AI Infra internship second-round interview questions covering operator optimization and high-performance computing"
tags: ["Operator optimization", "High-performance computing"]
sourceHash: "261c1a4ad3a2b0be"
---
## Project questions

1. Deep dive and follow-ups on your project

## Theory

2. When can you tell that operator fusion pays off? Which typical cases are a good fit for fusion?
3. How does asynchronous execution with CUDA Streams work? What preconditions does it need (for example avoiding memory-access conflicts)?
4. How do you find the best thread-count configuration for a CUDA kernel?
5. In CUDA, how do Thread, Warp, Block, SM, and Grid nest and map to each other?
6. If Shared Memory with some odd shapes produces wrong results, what should you check?
7. What are the traits of NHWC versus NCHW? How should you choose for training versus inference deployment?
8. When should you skip Shared Memory, for example when bank conflicts are severe or going through L2 Cache is faster?
