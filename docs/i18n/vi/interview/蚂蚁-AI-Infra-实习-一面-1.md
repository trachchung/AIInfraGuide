---
title: "Ant Group AI Infra thực tập vòng một (1)"
description: "Câu hỏi phỏng vấn Ant Group AI Infra thực tập vòng một thật, gồm tối ưu suy luận, tối ưu toán tử, tính toán hiệu năng cao và các hướng liên quan"
tags: ["Tối ưu suy luận", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "4b843cc598354ffa"
---

### Kiến thức nền

1. Làm sao biết một toán tử có cần tối ưu không? Căn cứ và đường phân tích cụ thể? (phân tích hotspot; phân loại nút thắt: compute / memory access / scheduling overhead)
2. Với mô hình chưa quantization, đẩy việc quantization có hệ thống thế nào? (phân tích cấu trúc mô hình, xác định mục tiêu triển khai, chọn granularity quantization, chọn cách calibrate, kiểm chứng lặp lại độ chính xác và hiệu năng)
3. Các cách quantization kiểu AWQ thêm bước dequantization rồi, vì sao suy luận tổng thể vẫn tăng tốc được?
4. Static graph và dynamic graph có đặc điểm gì? Vì sao thường cần chuyển dynamic graph sang static rồi mới fusion; chi phí chuyển đổi xử lý thế nào?
5. Ý tưởng thiết kế cốt lõi của vLLM về lập lịch suy luận và quản lý KV Cache?

### Kinh nghiệm dự án

6. Trong thực tập gặp những bài khó nào? (như OOM; bộ nhớ GPU tăng bất thường và crash ngẫu nhiên trong suy luận nhiều vòng dài; chuỗi kiểm tra và phương án sửa)
7. Timing plugin bạn phát triển trong thực tập triển khai tầng dưới thế nào? Vì sao không dùng trực tiếp Nsight / NCU và các công cụ profiling khác?
8. Các mô hình khác nhau khi deploy local và quantization có khác biệt rõ không?
9. Chiến lược chọn graph optimization và quy trình phát triển ra sao?

### Bài lập trình

10. Sao chép danh sách liên kết có random pointer.
