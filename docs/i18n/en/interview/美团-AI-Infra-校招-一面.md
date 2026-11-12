---
title: "Meituan AI Infra campus hire first round"
description: "Real Meituan AI Infra campus-hire first-round interview questions covering operator optimization"
tags: ["Operator optimization"]
sourceHash: "218e9beebd2e9399"
---

### Round 1

#### Theory

1. Why did HashMap introduce red-black trees?
2. Since red-black trees query faster than linked lists, why not use red-black trees from the start?
3. Could you replace red-black trees with a BST or AVL tree? Pros and cons of each?
4. What are the core thread-pool parameters?
5. If code uses synchronized heavily and becomes slow, how do you optimize?
6. How does synchronized differ from AQS-based locks? Where do the performance differences show?
7. When a production system hits frequent Full GC, how do you debug and fix it?
8. When do CMS and G1 collectors fit? Do you know ZGC?
9. How do you understand the leftmost-prefix matching rule?
10. You build a joint index on columns a, b, c. For `a > 1 AND c = 1`, how is the index used?
11. When production hits a slow query, how do you locate and debug it?

#### Project deep dive

12. Introduce performance-optimization work during the internship, and how operator metadata logging stays non-invasive to business code.

#### Live coding

13. Remove duplicates from a sorted list II.

### Round 2

#### Theory

1. What is the result of this Java code? How do you fix the issues?
   ```java
   int a = 3;
   Integer b = null;
   int c = a + b;
   long d = 1L + 0x7fffffff;
   int e = a + (int)d;
   ```
2. What does `Integer a = 567; Integer b = 567; System.out.println(a == b);` print? How do you change it to match expectation?
3. Introduce the Java concurrency knowledge you know.
4. What common locks exist in Java, and how do they differ?
5. How does a thread pool work?
6. From pressing the power button to opening a browser into the interview link, what bottom-level steps does the computer go through? (POST, boot sector, OS boot, process/thread creation, display rendering, full network-request path)
7. Why do video calls use UDP underneath instead of TCP?
8. Introduce MySQL core knowledge from an application angle.
9. How do you tell whether a SQL statement used an index? Which EXPLAIN fields do you usually watch?
10. Describe the field design and indexes of a real table you have used.
11. What is the execution order of SELECT, FROM, WHERE, GROUP BY, HAVING, and similar keywords?

#### Project deep dive

12. Pick one internship experience and walk through it in detail, focusing on technical hard points and challenges.

#### Live coding

13. Search for a target value in a 2D matrix that is sorted by row and by column.
