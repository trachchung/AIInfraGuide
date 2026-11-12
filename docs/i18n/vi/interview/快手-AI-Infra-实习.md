---
title: "Kuaishou AI Infra thực tập"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại Kuaishou thật, gồm tối ưu suy luận và tối ưu toán tử"
tags: ["Tối ưu suy luận", "Tối ưu toán tử"]
sourceHash: "fda726ebd5ca5450"
---

### Lý thuyết

1. Ngoài phương án lượng tử hóa dùng trong dự án, bạn còn biết phương pháp lượng tử hóa nào?
2. Từ góc sai số lượng tử hóa, phân bố dữ liệu nên thỏa điều kiện gì thì lý tưởng hơn?
3. Khi có outlier, phân bố càng đều thường lượng tử hóa càng tốt. Điều này đúng với INT8, FP8, INT4 — vì sao với FP4 thì không còn đúng?
4. Mô tả phương pháp tối ưu CUDA GEMM, và chiến lược chọn kích thước Shared Memory trong quá trình tính.

### Đào sâu dự án

5. Giới thiệu phần việc chính trong kỳ thực tập.
6. Giới thiệu chi tiết phương án kỹ thuật và triển khai của dự án.

### Lập trình tại chỗ

7. Dùng CUDA viết toán tử Norm. Input: x[N]; output: norm = (x - u) / o, với u = mean(x[N]), o = sqrt(sum((x - u)^2) / N).
