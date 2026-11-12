---
title: "Kuaishou AI Infra thực tập vòng một (3)"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại Kuaishou vòng một thật, gồm tối ưu suy luận, tối ưu huấn luyện và tối ưu toán tử"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử"]
sourceHash: "d13e24878c05638a"
---

### Lý thuyết

1. Nén KV Cache có những phương pháp phổ biến nào?
2. Lần lượt nêu khái niệm MHA, MQA, GQA, và cách triển khai KV broadcast. Quan hệ dữ liệu giữa Multi-head Latent Attention (MLA) và GQA thế nào? Cho hidden_size và chiều RoPE, MLA tương ứng bao nhiêu GQA head?
3. DeepSpeed ZeRO-1/2/3 mỗi giai đoạn tối ưu gì? Giả sử mô hình có N tham số, dùng Adam, ZeRO-1 phân VRAM trên P GPU thế nào? Kiểu dữ liệu (FP32/FP16) ảnh hưởng chiếm VRAM ra sao?
4. Nguyên lý SmoothQuant là gì? Vì sao cần Smooth? Siêu tham số xác định thế nào? Làm sao biết mô hình có hợp SmoothQuant không? Nếu dùng phân bố activation theo tầng thì nhìn input channel hay output channel?
5. Nguyên lý AWQ và GPTQ lần lượt là gì? Khác nhau thế nào?
6. GPU có những primitive giao tiếp phân tán nào? All-Gather và All-to-All phù hợp kịch bản nào?

### Đào sâu dự án

7. Giới thiệu chi tiết triển khai tính thưa KV Cache và cách viết vLLM Triton kernel. Vì sao tính thưa KV Cache không dùng mask?
8. Vì sao dự án chọn phương pháp lượng tử hóa khác nhau theo kịch bản? GPTQ và SmoothQuant mỗi cái phù hợp khi nào?
9. Quy trình triển khai mô hình distillation cụ thể thế nào? Dùng kỹ thuật then chốt nào?
