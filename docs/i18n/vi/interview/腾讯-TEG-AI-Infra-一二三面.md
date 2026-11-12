---
title: "Tencent TEG AI Infra vòng một, hai và ba"
description: "Câu hỏi phỏng vấn Tencent AI Infra vòng một, hai và ba thật, gồm tối ưu toán tử và các hướng liên quan"
tags: ["Tối ưu toán tử"]
sourceHash: "094a0828b4f3d3e1"
---

### Vòng một

#### Kinh nghiệm dự án

1. Giới thiệu kinh nghiệm liên quan dự án

#### Cơ sở dữ liệu

2. Cách hiện thực tầng dưới của aggregate function và GROUP BY
3. Cách hiện thực JOIN và chiến lược tối ưu (tối ưu index, tối ưu hash table)
4. Trong hash JOIN, nếu một phía bảng lớn hơn nhiều phía còn lại thì tối ưu thế nào
5. Vai trò của database optimizer và nguyên lý hiện thực
6. Cơ chế thực thi của câu EXPLAIN
7. Thiết kế và hiện thực Buffer Pool
8. Chiến lược quản lý dirty page và cách xử lý mất dữ liệu (redo log)
9. Hạn chế của LRU (prefetch miss, cache pollution) và phương án cải tiến
10. Cơ chế hiện thực B+ tree đồng thời
11. So sánh volcano model với các execution model khác (materialization model)
12. Phân tích scenario phù hợp của row store và column store
13. Khái niệm cơ bản của transaction và MVCC
14. Ý tưởng thiết kế mở rộng database đơn máy thành distributed database (sharding dữ liệu, quản lý metadata, thực thi toán tử)
15. Khác biệt giữa hash sharding và range sharding
16. Phương án high availability cho metadata node (replication primary–replica, giao thức đồng thuận phân tán)
17. Trong MySQL primary–replica replication, đảm bảo strong consistency thế nào
18. Execution plan của `SELECT * FROM table` (toán tử Project chiếu + toán tử TableScan quét toàn bảng)
19. Trong scenario phân tán cần thêm những toán tử nào để truy vấn xuyên node
20. Thiết kế distributed execution plan có WHERE và GROUP BY COUNT

#### Hệ điều hành

21. Quá trình khởi động Linux
22. Cơ chế hiện thực module xử lý interrupt
23. CPU xử lý nhiều interrupt đồng thời thế nào
24. Khác biệt giữa process và thread (gồm cả user-space thread)
25. Cách scheduling user-space thread và các vấn đề có thể gặp
26. Quá trình chuyển từ user space sang kernel space

#### Bài lập trình

27. Hiện thực file system trong memory (biến thể trie)

---

### Vòng hai

#### Kinh nghiệm dự án

1. Giới thiệu phân công dự án và phần việc cá nhân đảm nhận
2. Dựa trên MiniOb trình bày toàn bộ flow thực thi một câu SQL
3. Trong quá trình query, khâu nào ảnh hưởng hiệu năng nhiều nhất
4. Dự án có thiết kế optimizer không
5. Quan hệ giữa volcano model và pipeline model
6. Hướng tối ưu phá bottleneck của execution model hiện tại (CRTP giảm overhead virtual function, materialization model trả dữ liệu theo batch)
7. Phân tích ưu nhược của dự án OceanBase
8. Giới thiệu dự án xv6 và những gì thu được từ đó

#### Bài lập trình

9. Fast/slow pointer tìm điểm giữa linked list + đảo linked list

---

### Vòng ba

#### Kinh nghiệm dự án

1. Giới thiệu phần bạn phụ trách trong dự án trên CV và cách phân công
2. Khi cộng tác nhóm gặp bất đồng thì xử lý thế nào

#### Bài lập trình

3. Đưa chữ hoa trong chuỗi về chữ thường rồi sắp xếp theo thứ tự từ điển (không dùng thư viện)
