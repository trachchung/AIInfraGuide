---
title: "Alibaba Cloud AI Infra thực tập vòng hai (1)"
description: "Câu hỏi phỏng vấn Alibaba AI Infra thực tập vòng hai thật, gồm tối ưu suy luận, tối ưu toán tử, tính toán hiệu năng cao và các hướng liên quan"
tags: ["Tối ưu suy luận", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "c94c40de90487d91"
---

## Kinh nghiệm dự án

1. Giới thiệu công việc về tối ưu toán tử, fusion toán tử và tối ưu đồ thị, cũng như sửa vấn đề ổn định.
2. Khi suy luận hoặc huấn luyện gặp bộ nhớ GPU bất thường và vấn đề ổn định dài hạn, định vị và kiểm tra thế nào? Nói kèm case cụ thể.
3. Từ góc kỹ thuật, tối ưu hiệu năng một toán tử thế nào? Lấy ví dụ cụ thể về ý tưởng tối ưu, đánh giá lợi ích và phân tích giới hạn.
4. Suy luận có quantization (như W4A16 theo hướng AWQ) ở triển khai cụ thể gồm những khâu then chốt nào? Gồm sửa Linear layer, thiết kế cấu trúc dữ liệu, xử lý memory layout, kiểm chứng đúng đắn, v.v.

## Kiến thức nền

5. Trong tình huống nhiều card phối hợp, giao tiếp giữa các card có những cách nào? Đặc điểm NVLink và RDMA?
6. Các hệ sinh thái GPU khác nhau chủ yếu khác ở đâu?
7. Hiệu quả tối ưu toán tử lan sang tầng cả mạng thế nào? Ngoài cả mạng, nối với liên kết đa card, Profiling, phối hợp phần mềm–phần cứng ra sao?
8. Bạn nhìn nhận việc dùng AI Agent hỗ trợ viết code, Review và sắp xếp kỹ thuật thế nào?
