---
title: "Chapter 8: Production deployment and operations"
description: "Containerizing and deploying vLLM inference services on Kubernetes, plus observability, autoscaling, load balancing, and capacity planning"
tags: ["Production deployment", "Kubernetes", "Observability", "Autoscaling", "Capacity planning"]
sourceHash: "e7b8964e53eac356"
---

## About this chapter

Earlier chapters made inference services fast and cheap. This chapter is about serving real traffic reliably: how to package, orchestrate, observe, scale, and plan capacity. It is the last mile from demo to production, with vLLM as the running example.

The section on containerization and Kubernetes covers packaging a vLLM service into an image and running it on K8s: GPU resource requests, readiness and liveness probes (model loading takes a long time, so probe timeouts need to be looser), image and weight pull strategies, and production deployment options such as the vLLM Production Stack and KServe.

Observability is how you see what the service is doing. vLLM exposes a Prometheus metrics endpoint (TTFT, TPOT, running and waiting request counts, KV Cache utilization, preemption counts, and more). With Grafana dashboards and structured logs you can locate failures and track performance.

Autoscaling and load balancing cover request distribution across replicas: HPA based on QPS, queue length, or KV Cache utilization, and prefix-aware routing that sends requests that share the same system prompt to the same replica so the cache hit rate goes up.

⚠️ **Security note**: vLLM's OpenAI-compatible server does no authentication by default. Exposing it on the public internet is the same as opening your compute and data. In production you must put API Key checks, TLS, and network isolation at the gateway. This chapter calls that out so you do not ship with the service wide open.

Capacity planning shows how to work backwards from SLOs and peak QPS to the number of GPUs you need: combine the throughput ceiling from single-replica load tests, memory constraints, and a redundancy factor to estimate cluster size and cost.

## Sections

- 8.1 Containerization and Kubernetes deployment: images, GPU scheduling, probes, Production Stack / KServe
- 8.2 Observability: Prometheus metrics, Grafana dashboards, logs and alerts
- 8.3 Autoscaling and load balancing: HPA policies, prefix-aware routing, authentication and security
- 8.4 Capacity planning: sizing the cluster and its cost from SLOs and peak QPS
