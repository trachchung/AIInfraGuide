---
title: "ByteDance AI Infra thực tập vòng 1-2-3"
description: "Câu hỏi phỏng vấn ByteDance AI Infra thực tập vòng 1-2-3, gồm tối ưu suy luận, tối ưu huấn luyện và tối ưu toán tử"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử"]
sourceHash: "1aca517c305883ec"
---
### Vòng một

#### Kinh nghiệm dự án

1. Hãy giới thiệu nghiên cứu liên quan Efficient Transformer bạn đã tham gia.

#### Kiến thức cơ bản

2. Mỗi cache line trên CPU cache gồm ba phần nào? Vai trò từng phần?
3. Vì sao cần thiết kế bộ nhớ đệm nhiều cấp?
4. Sau cache miss, phần cứng xử lý thế nào?
5. Cơ chế bảng trang mang lại chi phí hiệu năng nào? Giảm thiểu ra sao?
6. Cơ chế bảng trang mang lại lợi ích gì?
7. Cơ chế nạp chồng hàm trong C++ là gì? Trình biên dịch triển khai thế nào?
8. Khái niệm atomic operation trong C++ và lý do đưa vào là gì?

#### Bài lập trình

9. Kiểm tra liên thông bàn cờ: hàng đầu có nối được tới hàng cuối không.

### Vòng hai

#### Kinh nghiệm dự án

10. Hãy đào sâu nghiên cứu Efficient Transformer.

#### Kiến thức cơ bản

11. Kernel Python của Sparse Attention triển khai thế nào?
12. Huấn luyện phân tán mô hình lớn có những chiến lược song song phổ biến nào?
13. Khi VRAM huấn luyện/suy luận quá cao có những hướng xử lý nào? (ví dụ thưa hóa, gradient checkpointing, suy luận lượng tử hóa, mixed precision, distillation)
14. Viết mã tính toán song song trên GPU thế nào?

#### Bài lập trình

15. Cho định nghĩa lớp Graph đầy đủ, triển khai hàm topological sort.

### Vòng ba

#### Kinh nghiệm dự án

16. Điểm đổi mới trong Sparse Attention bạn đề xuất là gì?
17. Efficient Attention, MoE, huấn luyện phân tán hiệu quả thực tế trong công nghiệp ra sao?

#### Kiến thức cơ bản

18. Trong đa hình C++, địa chỉ hàm trong vtable xác định lúc nào? vptr của đối tượng được gán lúc nào?
19. Trong C++, giải phóng bộ nhớ đã cấp phát của vector thế nào?
20. Thiết kế cấu trúc dữ liệu hỗ trợ đủ thao tác stack và lấy min hiện tại trong O(1).

#### Bài lập trình

21. Next permutation (LeetCode 31).
