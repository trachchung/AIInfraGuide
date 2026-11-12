---
title: "AI Infra vòng phỏng vấn một"
description: "Câu hỏi phỏng vấn AI Infra vòng một thật, gồm tối ưu toán tử và các hướng liên quan"
tags: ["Tối ưu toán tử"]
sourceHash: "1c52faf5b830f106"
---
1. Trình bày nguyên lý cơ bản của mô hình Roofline. Làm sao dùng mô hình này để xác định xem một toán tử đã đạt nút thắt tính toán chưa?
2. Khi số GPU dùng cho training hoặc inference tăng gấp đôi, hệ thống dễ nghẽn ở những khâu nào nhất? Phân tích nguyên nhân và nêu hướng tối ưu hoặc giảm thiểu.
3. GEMM có luôn là toán tử compute-bound không? Nếu cần tối ưu, tư duy tổng thể là gì?
4. Trong quá trình tinh chỉnh hiệu năng, làm sao định vị và kiểm tra nút thắt? Bạn thường dùng phương pháp hoặc công cụ nào?
5. Trình bày hiểu biết của bạn về FlashAttention.
6. Trong C++, nếu ghi vượt biên mảng làm hỏng cấu trúc dữ liệu khác và vẫn còn file coredump, làm sao điều tra?
7. Bài lập trình: tự viết module Attention có GQA (Grouped Query Attention).
