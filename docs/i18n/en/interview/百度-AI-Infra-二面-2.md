---
title: "Baidu AI Infra second round"
description: "Real Baidu AI Infra second-round interview questions covering training optimization"
tags: ["Training optimization"]
sourceHash: "a4926cfbaf952800"
---

## Theory - Transformer and model structure

1. Fully describe the Transformer compute path from input tokens to output logits.
2. Why does the FFN expand then shrink dimension? How does that design affect model capacity?
3. How do MHA, MQA, and GQA differ in KV Cache size and compute efficiency at inference?
4. At inference, why does KV Cache store only K and V, not Q?
5. What is the core idea of RoPE? What limits show up in long-context settings?

## Theory - Fine-tuning and alignment

6. In Instruction Tuning with multi-turn dialogue data, how should the Loss Mask be designed?
7. After SFT, if a model gets stronger on a specific task but weaker generally, what do you do?
8. Why can LoRA's low-rank factorization approximate full fine-tuning?
9. If LoRA Rank is set poorly, what symptoms show up?
10. After DPO training, model outputs get clearly longer — how do you handle that in a real system?
11. After alignment, the model becomes too conservative and refuses often — how do you adjust training?
12. What usually causes the "repeater" / regurgitation failure mode in large models?

## Theory - RAG systems

13. What does knowledge-base cleaning and construction look like? How does inconsistent data quality hurt RAG?
14. How do you design document chunking? How do Chunk Size and Overlap affect recall quality and generation?
15. The answer exists in the knowledge base but the system fails to retrieve the right docs — how do you debug?
16. You retrieved the right docs but the generated answer is still wrong — how do you localize the issue?
17. Recall often looks semantically similar but factually unrelated — how do you improve retrieval?
18. When one question needs multiple documents, how should RAG handle it?
19. How do you tell whether a RAG failure is in retrieval or generation?

## Project deep dive

20. Deep questions around project content.
