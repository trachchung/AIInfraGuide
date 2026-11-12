---
title: "Alibaba AI Infra (1)"
description: "Câu hỏi phỏng vấn Alibaba AI Infra thật, gồm tối ưu toán tử, tính toán hiệu năng cao và các hướng liên quan"
tags: ["Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "024785503f4a45da"
---

## Hỏi sâu dự án

1. Hỏi sâu quanh dự án

## Nền tảng lý thuyết

2. Bạn đã dùng Agent để tự sinh CUDA kernel chưa? Phương án triển khai cụ thể là gì?
3. Trong huấn luyện hoặc triển khai mô hình quy mô lớn, đã dùng công cụ debug tầng thấp để kiểm tra chưa? Khi cluster nghìn card gặp NCCL Timeout, định vị nguyên nhân và xử lý thế nào?
4. Nếu bỏ tối ưu Warp Specialization, chỉ giữ tile blocking và tối ưu Shared Memory, hiệu năng giảm chủ yếu thể hiện trên những chiều nào?
5. Bạn có kinh nghiệm tối ưu tầng kernel không? Ví dụ dùng CuTe DSL hoặc viết tay CUDA để làm toán tử kiểu fusion—nói rõ
6. Khi làm kernel fusion, bạn nghiêng về phương án kỹ thuật nào? Vì sao?
7. Trước khi dịch vụ suy luận lên production, bạn đã làm những tối ưu hiệu năng nào?
8. Khi viết CUDA có đi sâu chi tiết phần cứng không? Ví dụ cơ chế và triển khai tầng dưới của Warp Specialization trên kiến trúc Hopper
9. Đã có kinh nghiệm tối ưu mô hình MoE trong tình huống RL chưa?
10. Đã gặp trường hợp kernel fusion xong hiệu năng lại kém hơn không? Bạn nghĩ nguyên nhân ở đâu?
11. Làm sao biết một mô hình MoE thật sự có phân công giữa các expert, chứ không chỉ cắt tham số của mô hình Dense?
12. Trong huấn luyện RL + MoE, đã gặp reward làm lệch routing chưa (ví dụ mô hình đẩy hầu hết request sang vài expert để kiếm thưởng cao hơn)? Khi đó xử lý thế nào?
