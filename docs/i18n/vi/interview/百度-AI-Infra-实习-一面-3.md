---
title: "Baidu AI Infra thực tập vòng một (3)"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại Baidu vòng một thật, gồm tối ưu suy luận và toán tử"
tags: ["Tối ưu suy luận", "Tối ưu toán tử"]
sourceHash: "8fbfa52c8d13de78"
---

## Train phân tán

1. Giải thích ý nghĩa và quy trình thực thi của ba chiến lược song song TP, PP, DP
2. Làm sao chọn chiến lược song song dựa trên chi phí giao tiếp của TP và PP?

## Quantization

3. Giải thích khác biệt giữa ba độ mịn quantization: per-tensor, per-channel, group-wise
4. Khác biệt độ chính xác giữa các phương pháp quantization, và vì sao group-wise giảm thêm được sai số quantization
5. So sánh chi phí tính toán của các cách quantization và các kỹ thuật tối ưu
6. Trong quantization, xử lý outlier thế nào?
7. Giới thiệu ý tưởng cốt lõi của GPTQ, AWQ, SmoothQuant
8. Nguyên lý và cách triển khai KV Cache quantization

## Attention và tối ưu suy luận

9. Nguyên lý tăng tốc của FlashAttention là gì?
10. Khác biệt chính giữa FlashAttention v1 và v2
11. Chiến lược chia khối Bc trong FlashAttention, và cách triển khai 1-loop FlashAttention
12. Ý tưởng thiết kế của PagedAttention
13. Khác biệt giữa giai đoạn prefill và decoding của mô hình lớn, cùng nguyên nhân
14. FlashAttention gặp vấn đề gì ở decoding? FlashDecoding cải tiến thế nào?

## Triển khai toán tử

15. Cách triển khai cụ thể của RMSNorm
16. Hướng tiếp cận và phương pháp luận chung để tối ưu CUDA Kernel
17. Cho Conv2D input [C=64, W=64, H=128], kernel 3x3, output [C=128, W=64, H=128], tính số tham số và FLOPs
18. Trên CPU có những cách tối ưu toán tử nào (ví dụ AVX-512)?

## Nền C++

19. Phân loại smart pointer và kịch bản dùng

## Bài lập trình

20. Cài đặt CUDA LayerNorm Kernel
