---
title: "Baidu AI Infra (2)"
description: "Câu hỏi phỏng vấn AI Infra tại Baidu thật, gồm tối ưu toán tử"
tags: ["Tối ưu toán tử"]
sourceHash: "9a42400ec2f95e5d"
---

## Vòng một

### Đào sâu dự án

1. Giới thiệu dự án và bài báo trên CV, hỏi tiếp quanh các điểm kỹ thuật liên quan

### Tính toán hiệu năng cao

2. Mở vùng song song OpenMP thế nào? (cú pháp liên quan pragma omp parallel)
3. Trong vùng song song, chạy `for(int i=0;i<100;i++){sum++;}` kết quả có khớp tuần tự không? Làm sao bảo đảm nhất quán?
4. Phân cấp bộ nhớ GPU: phạm vi của Register File, L1 Cache, Shared Memory, L2 Cache, Global Memory

### Kiến trúc máy tính

5. Phân loại và đặc điểm ánh xạ Cache (set-associative, fully associative, direct-mapped)

### Nền tảng C++

6. Phân loại smart pointer và kịch bản dùng
7. Khái niệm và cách triển khai đa hình C++ (khác biệt override và overload)
8. Khác biệt dynamic linking và dynamic binding

## Vòng hai

### Đào sâu dự án

9. Thảo luận sâu quanh CV và dự án

### Nền tảng C++

10. Cấu trúc dữ liệu tầng dưới của STL map là gì?
11. Khác biệt BST cân bằng (AVL) và red-black tree
12. Nguyên lý triển khai tầng dưới của unordered_map
13. Cách xử lý xung đột hash phổ biến

### Kiến trúc máy tính

14. Phân cấp lưu trữ máy tính (register, Cache, main memory, secondary storage)
15. Trong CPU, register mà lập trình viên ứng dụng thấy được chia loại nào? Program counter (PC) làm gì?

### Khác

16. Các lệnh hệ điều hành thường dùng?

## Vòng ba

### Nền tảng C++

17. Chạy một hàm trước khi main chạy thế nào?
18. Khi debug, đặt conditional breakpoint thế nào?
19. Theo dõi call stack thế nào?
20. Chương trình phân đoạn trong bộ nhớ thế nào (code, data, heap, stack, v.v.)
21. Nguyên lý triển khai tầng dưới của STL deque

### Kiến trúc máy tính

22. Register hay Cache nhanh hơn? Cache và main memory mỗi cái dùng kỹ thuật gì? Giống và khác gì?

### Lập trình CUDA

23. Phân loại bộ nhớ trong CUDA (local memory, global memory, v.v.)
24. Khái niệm Kernel; vai trò của `__global__`
25. Lấy số thread tối đa GPU cho phép thế nào?

### Thuật toán

26. Khác biệt greedy và dynamic programming
27. Thuật toán tìm kiếm trên đồ thị phổ biến (BFS, DFS)
