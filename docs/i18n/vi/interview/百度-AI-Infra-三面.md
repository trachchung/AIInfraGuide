---
title: "Baidu AI Infra (2)"
description: "Câu hỏi phỏng vấn AI Infra tại Baidu thật, gồm tối ưu toán tử"
tags: ["Tối ưu toán tử"]
sourceHash: "a698b6407b984744"
---

## Vòng một

### Lý thuyết

1. Mô tả phân cấp bộ nhớ GPU và quan hệ chia sẻ giữa các cấp.
2. Trong OpenMP, khai báo vùng song song thế nào? Cho cách dùng cơ bản.
3. Trong môi trường song song, chạy `for(int i=0;i<100;i++){sum++;}` có khớp tuần tự không? Làm sao bảo đảm đúng.
4. Ánh xạ Cache chia mấy loại? Đặc điểm mỗi loại?
5. Trình bày các loại smart pointer C++ và kịch bản dùng.
6. Đa hình C++ là gì? Có những cách triển khai nào?
7. Dynamic linking và dynamic binding là gì? Khác nhau thế nào?

### Đào sâu dự án

8. Giới thiệu dự án và bài báo trên CV, thảo luận chi tiết kỹ thuật liên quan.
9. Hướng tính toán hiệu năng cao chủ yếu nghiên cứu gì? Có kinh nghiệm hệ tính toán quy mô lớn không?

## Vòng hai

### Lý thuyết

1. Cấu trúc dữ liệu tầng dưới của C++ STL map là gì? Nguyên lý cơ bản thế nào?
2. AVL và red-black tree khác nhau ở chiến lược cân bằng thế nào? Đặc điểm hiệu năng mỗi cái?
3. Nguyên lý triển khai tầng dưới của unordered_map?
4. Cách xử lý xung đột hash phổ biến?
5. Giới thiệu phân cấp lưu trữ máy tính.
6. Trong CPU, register lập trình viên ứng dụng thấy được chia loại nào? Program counter (PC) làm gì?
7. Liệt kê lệnh hệ điều hành dùng hàng ngày khá thường xuyên.

### Đào sâu dự án

8. Giới thiệu kinh nghiệm dự án trên CV và công việc nghiên cứu.

## Vòng ba

### Lý thuyết

1. Chạy một hàm tùy chỉnh trước khi main chạy thế nào?
2. Khi debug, đặt conditional breakpoint thế nào?
3. Theo dõi call stack thế nào?
4. Chương trình bố trí phân đoạn trong bộ nhớ thế nào?
5. Cơ chế triển khai nội bộ của STL deque?
6. Register hay Cache truy cập nhanh hơn? Cache và main memory mỗi cái dùng kỹ thuật gì? Giống và khác gì?
7. CUDA có những phân loại bộ nhớ nào?
8. CUDA Kernel là gì? `__global__` làm gì?
9. Lấy số thread tối đa GPU hỗ trợ thế nào?
10. So sánh tư tưởng và kịch bản phù hợp của greedy và dynamic programming.
11. Các thuật toán tìm kiếm trên đồ thị phổ biến?
