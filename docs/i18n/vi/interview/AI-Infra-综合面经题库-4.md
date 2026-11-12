---
title: "Ngân hàng câu hỏi phỏng vấn AI Infra tổng hợp (4)"
description: "Câu hỏi phỏng vấn AI Infra thật, gồm tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "95080c5875b2ec35"
---
## Kinh nghiệm dự án

1. Đào sâu dự án

## Kiến thức cơ bản

2. Bạn đã tối ưu ở mức kernel chưa, ví dụ dùng CuTe DSL hoặc CUDA viết tay để fusion toán tử? Mô tả cách làm.
3. Khi làm kernel fusion, bạn thường nghiêng về cách hiện thực nào?
4. Đã gặp trường hợp fusion xong lại chậm hơn chưa? Phân tích vì sao.
5. Khi viết CUDA, bạn có để ý chi tiết tầng thấp như Warp Specialization trên Hopper và cách nó hoạt động không?
6. Khi deploy hoặc train mô hình lớn, bạn có dùng công cụ debug tầng thấp không? Timeout NCCL ở quy mô nghìn card thường định vị và xử lý thế nào?
7. Đã tối ưu mô hình MoE trong ngữ cảnh RL chưa?
8. Trước khi đưa dịch vụ suy luận lên production, bạn đã làm những tối ưu hiệu năng nào?
9. Đã thử dùng Agent tự sinh CUDA kernel chưa? Làm thế nào?
