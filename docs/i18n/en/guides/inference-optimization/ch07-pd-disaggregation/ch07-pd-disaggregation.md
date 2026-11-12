---
title: "Chapter 7: Prefill/Decode disaggregation"
description: "Problems with mixed P/D batching, DistServe/Splitwise-style disaggregation, KV cache transfer, and goodput with SLO-aware scheduling"
tags: ["P/D disaggregation", "DistServe", "Splitwise", "KV Cache transfer", "Goodput", "SLO"]
sourceHash: "9ef6708bcc011816"
---

## About this chapter

Prefill and Decode have very different compute profiles (compute bound vs memory bound), and mixing them in one batch makes them interfere. This chapter goes into how disaggregated architectures are designed and where they hurt, then lands on vLLM's Disaggregated Prefill.

**Problems with mixed batching** quantifies how Prefill hurts Decode: Decode P95 TPOT can slow by 3–5×.

**Disaggregated architecture design** covers DistServe (OSDI'24, a systematic case for P/D disaggregation), Splitwise (ISCA'24, placing them on different GPU pools), TaiChi (a unified frame for aggregation and disaggregation), and MLC Microserving (cross-engine orchestration).

**KV cache transfer and Connector** digs into the hardest engineering piece of disaggregation: how to move the KV cache Prefill computed onto the Decode node efficiently. It covers vLLM's KV Connector abstraction, backends such as NIXL/NCCL, and how cross-node bandwidth shows up in end-to-end latency.

**Goodput and SLO-aware scheduling** stresses that raw QPS is not user experience; goodput (throughput that still meets SLOs) is the real optimization target.

**Challenges of disaggregation** include scheduler complexity, deriving the P/D GPU pool mix, and when you should or should not disaggregate.

**Disaggregated Prefill with vLLM**: build a mixed load with vLLM's disaggregated deploy, measure how much P and D interfere, and derive a P/D GPU pool mix for a given workload.

## Sections

- **7.1 Problems with mixed batching**: quantitative Prefill interference on Decode
- **7.2 Disaggregated architecture design**: DistServe, Splitwise, TaiChi, Microserving
- **7.3 KV cache transfer and Connector**: KV Connector, NIXL/NCCL, bandwidth pressure
- **7.4 Goodput and SLO-aware scheduling**: optimizing for effective throughput
- **7.5 Challenges and pool sizing**: scheduler complexity, P/D GPU pool ratios
- **7.6 Disaggregated Prefill with vLLM**: disaggregated deploy and measuring interference
