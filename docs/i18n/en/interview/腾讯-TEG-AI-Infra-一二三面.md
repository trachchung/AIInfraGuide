---
title: "Tencent TEG AI Infra first, second, and third rounds"
description: "Real Tencent AI Infra first-, second-, and third-round interview questions covering operator optimization and related topics"
tags: ["Operator optimization"]
sourceHash: "094a0828b4f3d3e1"
---

### First round

#### Project experience

1. Introduce project-related experience

#### Databases

2. How aggregate functions and GROUP BY are implemented under the hood
3. How JOIN is implemented and how to optimize it (index optimization, hash-table optimization)
4. In a hash JOIN, if one side is much larger than the other, how do you optimize
5. Role of the database optimizer and how it works
6. How EXPLAIN executes
7. Design and implementation of the Buffer Pool
8. Dirty-page management and how to avoid data loss (redo log)
9. Weaknesses of LRU (prefetch failure, cache pollution) and improvements
10. How concurrent B+ trees are implemented
11. Volcano model vs other execution models (materialization model)
12. When row store vs column store fits
13. Basics of transactions and MVCC
14. Design ideas for growing a single-node database into a distributed one (sharding, metadata management, operator execution)
15. Difference between hash sharding and range sharding
16. High-availability options for metadata nodes (primary–replica replication, distributed consensus protocols)
17. How to guarantee strong consistency in MySQL primary–replica replication
18. Execution plan for `SELECT * FROM table` (Project projection operator + TableScan full-table-scan operator)
19. Which extra operators you need in a distributed setting for cross-node queries
20. Designing a distributed execution plan with WHERE and GROUP BY COUNT

#### Operating systems

21. Linux boot process
22. How the interrupt-handling module works
23. How the CPU handles multiple concurrent interrupts
24. Difference between processes and threads (including user-space threads)
25. How user-space threads are scheduled, and possible problems
26. Switching from user space to kernel space

#### Coding

27. Implement an in-memory file system (a trie variant)

---

### Second round

#### Project experience

1. Introduce project division of labor and what you owned
2. Walk through a full SQL query execution path based on MiniOb
3. Which steps in the query path hurt performance most
4. Did the project include an optimizer design
5. Relationship between the volcano model and the pipeline model
6. Optimization directions that break current execution-model bottlenecks (CRTP to cut virtual-call overhead, materialization model returning data in batches)
7. Pros and cons of the OceanBase project
8. Introduce the xv6 project and what you gained from it

#### Coding

9. Fast/slow pointers to find the middle of a linked list + reverse a linked list

---

### Third round

#### Project experience

1. Introduce which parts of the resume projects you owned and how work was split
2. How you handle disagreements in team collaboration

#### Coding

3. Convert uppercase letters in a string to lowercase and sort in lexicographic order (no library functions)
