---
title: "Ant Group AI Infra thực tập vòng ba"
description: "Câu hỏi phỏng vấn Ant Group AI Infra thực tập vòng ba thật, gồm tối ưu suy luận, tối ưu toán tử, tính toán hiệu năng cao và các hướng liên quan"
tags: ["Tối ưu suy luận", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "ec364156b1bac6a0"
---

Vòng này nhấn mạnh độ sâu hiểu biết nền về suy luận mô hình lớn, quantization và cơ chế Attention.

### Kiến thức nền

1. Ngoài W4A16, bạn biết các phương án quantization phổ biến trong ngành như INT8, FP8 đến mức nào? Hai định dạng FP8 phổ biến là gì? Khác nhau thế nào về độ phù hợp cho huấn luyện và suy luận?
2. Tính kích thước KV Cache thế nào? (gồm batch size, độ dài chuỗi, số head, head dim, số layer, kiểu dữ liệu và các yếu tố liên quan)
3. Vì sao suy luận mô hình lớn cần KV Cache? Vai trò bản chất của nó là gì?
4. Hiểu Attention thế nào? Vì sao quantization liên quan Attention khó hơn quantization weight?

### Kinh nghiệm dự án

5. Giới thiệu dự án Runtime suy luận có quantization: sửa trên Runtime mã nguồn mở nào? Chi tiết triển khai W4A16? Mất bao nhiêu độ chính xác? Dùng dataset nào để đánh giá?
6. Giới thiệu công việc cốt lõi trong thực tập: hoàn thiện chuỗi fusion và gắn thư viện hiệu năng cao; định vị và sửa vấn đề ổn định trong chạy dài cả mạng; mục đích thiết kế công cụ Timing nhẹ và vai trò của nó khi tìm nút thắt.
