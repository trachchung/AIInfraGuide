---
title: "Kuaishou AI Infra vòng một"
description: "Câu hỏi phỏng vấn AI Infra tại Kuaishou vòng một thật, gồm tối ưu suy luận, tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu suy luận", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "3cd5d07c227c628d"
---

### Lý thuyết

1. Bạn có biết Prefill-Decode (PD) disaggregation không? AF disaggregation nhằm mục đích gì? Đã có PD rồi vì sao vẫn cần AF?
2. FlashAttention V2 cải tiến gì so với V1? Ý tưởng tối ưu cụ thể thế nào? Có biết bản V4 mới nhất không?
3. Trong mô hình lớn, một tầng Transformer có mấy linear layer? Tensor parallelism (TP) cắt thế nào và vì sao? Có hướng nào tối ưu AllReduce ở giữa?
4. Ray triển khai tầng dưới thế nào? Có đặc tính then chốt nào? Trong nghiên cứu thực tế dùng Ray ra sao?
5. Liệt kê và trình bày các phương pháp tối ưu CUDA GEMM phổ biến.

### Đào sâu dự án

6. Giới thiệu chi tiết bối cảnh dự án, phương án kỹ thuật và triển khai.

### Lập trình tại chỗ

7. LeetCode Word Ladder.
