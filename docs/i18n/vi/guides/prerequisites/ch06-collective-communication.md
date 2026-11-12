---
title: "Chương 6: Cơ bản về giao tiếp tập thể"
description: "Các primitive giao tiếp trong huấn luyện phân tán (AllReduce, AllGather và các primitive khác), thuật toán Ring và Tree, cách dùng và tinh chỉnh NCCL"
tags: ["Giao tiếp tập thể", "AllReduce", "NCCL", "Thuật toán Ring"]
sourceHash: "691a50c42b4629b7"
---

## Giới thiệu chương

Giao tiếp tập thể (collective communication) là hệ tuần hoàn của huấn luyện phân tán. Phải hiểu các primitive và thuật toán đằng sau chúng thì mới thiết kế được chiến lược song song.

Phần primitive giới thiệu giao tiếp điểm-điểm (Send/Recv) và các phép giao tiếp tập thể: Broadcast, Reduce, AllReduce, AllGather và ReduceScatter. Mỗi phép đều có sơ đồ luồng dữ liệu và công thức tính lượng dữ liệu truyền đi, giúp bạn hình dung cách chúng hoạt động.

Phần thuật toán tập trung vào cách Ring AllReduce hoạt động và lượng dữ liệu nó truyền (2(N-1)/N × kích thước dữ liệu). Phần này cũng nói về Tree AllReduce và ý tưởng cơ bản của việc chồng giao tiếp lên tính toán (overlap).

Phần NCCL giải thích vị trí của thư viện giao tiếp tập thể chính thức của NVIDIA, cách dùng cơ bản, các biến môi trường cần tinh chỉnh và cách đo hiệu năng giao tiếp bằng nccl-tests.
