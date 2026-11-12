---
title: "SenseTime AI Infra phỏng vấn"
description: "Câu hỏi phỏng vấn SenseTime AI Infra, gồm tối ưu toán tử"
tags: ["Tối ưu toán tử"]
sourceHash: "4ecc277482ce796b"
---
Dưới đây là nội dung khảo sát phỏng vấn SenseTime AI Infra, chủ yếu đào sâu dự án.

### Kinh nghiệm dự án (tối ưu GEMM)

1. Mô tả chi tiết quá trình triển khai tối ưu GEMM
2. Vì sao một thread tính 8x8 phần tử của ma trận C
3. Thanh ghi có thể bị spill không, xử lý thế nào
4. Dự án có điểm đổi mới của riêng bạn không
5. Vai trò và tình huống dùng của `#pragma unroll`
6. Trong một tile thực hiện bao nhiêu lần tính toán
7. So với cuTLASS, bản tự viết hiệu năng ra sao

### Kiến thức cơ bản

8. Đã từng tối ưu toán tử convolution chưa
9. Có biết PPL (thư viện HPC của SenseTime) không
10. Nguyên lý và cách dùng Tensor Core
11. Có kinh nghiệm phát triển GEMM low-bit không
