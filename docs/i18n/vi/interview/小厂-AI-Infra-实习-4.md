---
title: "Công ty nhỏ AI Infra thực tập (4)"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại công ty nhỏ thật, gồm tối ưu suy luận và toán tử"
tags: ["Tối ưu suy luận", "Tối ưu toán tử"]
sourceHash: "23320300f7edf0a5"
---

### Kinh nghiệm dự án

1. Giới thiệu kinh nghiệm dự án liên quan triển khai trước đây.
2. Dự án dùng TensorRT để lượng tử hóa — có phải phương án PTQ không? Vì sao chọn TensorRT? Dự án có yêu cầu tốc độ suy luận cụ thể nào?
3. Mô tả cách dùng CUDA tăng tốc tiền xử lý ảnh, và các chiến lược tối ưu đã dùng.
4. Cho ví dụ thực tế phân tích và tối ưu CUDA operator (ví dụ vì sao vectorized load không mang lại tăng tốc).
5. Vì sao chọn chip RK3588? Mô tả quy trình dựng môi trường và deploy mô hình.

### Kiến thức nền

6. Nêu cấu trúc phân cấp bộ nhớ GPU, gồm đặc điểm và vai trò của L1 Cache, shared memory và L2 Cache.
7. Có những cách tối ưu shared memory nào? Bank Conflict là gì?
8. Bạn có biết phương pháp tối ưu hiệu năng trên nền tảng ARM không?
9. Tóm tắt quy trình matching của ByteTrack.

### Bài lập trình

10. Dùng CUDA cộng một triệu số float; mô tả hướng triển khai.
