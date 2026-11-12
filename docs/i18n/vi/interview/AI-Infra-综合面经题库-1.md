---
title: "Ngân hàng câu hỏi phỏng vấn AI Infra tổng hợp (1)"
description: "Câu hỏi phỏng vấn AI Infra thật, gồm tối ưu suy luận, tối ưu huấn luyện, tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "6d01440abbaa282f"
---
## Kiến thức cơ bản

1. Ưu điểm của TMA trên kiến trúc Hopper là gì? Cách gọi như thế nào? Truyền dữ liệu có đi qua cache L1 không?
2. Trong FlashAttention v2, vì sao vòng lặp ngoài duyệt theo Q? Kernel combine của Flash Decoding chiếm bao nhiêu thời gian?
3. Làm sao phân tích tỉ lệ tính toán/truy cập bộ nhớ của MLA decode? Tỉ lệ đó có liên quan sequence length và batch size không?
4. Tư duy thiết kế phương án tách PD lấy KV-Cache làm trung tâm trong Mooncake
5. Framework suy luận DiT so với framework suy luận LLM giống và khác ở đâu?
6. Knowledge distillation cho LLM có phù hợp làm ở giai đoạn pretrain không?
7. Các bước train và infer của Diffusion Model là gì? Khi infer đặt `num_inference_steps` = 40, vì sao train vẫn cần `timesteps` = 1000?
8. Giới thiệu dLLM (discrete large language model). Khác AR model ở điểm căn bản nào?
9. `torch.repeat` và `torch.expand` khác nhau thế nào về chức năng và hành vi bộ nhớ?
10. Tham số khởi động của `torchrun` gồm những gì? Trên Linux, làm sao hủy hàng loạt process có `torchrun`?

## Bài CUDA

1. Hiện thực phép nhân elementwise tensor 4D hỗ trợ broadcast của torch
2. Cho A: (1, 256), B: (256, 128), C: (128, 256), tính (A * B) * C
3. Embedding Sparse Feature Pooling: A gồm 1 triệu ID rời rạc (0~999), B gồm 1 triệu float. Tính mảng C dài 1000, C[i] = tổng B[j] với mọi j thỏa A[j] = i

## Bài PyTorch

1. Hiện thực LoRA Adapter

## Thuật toán

1. Hiện thực memory pool (hỗ trợ kiểu `new Foo[]` / `delete[]`)
2. Trong C++, so sánh hai float bằng nhau thế nào?
3. Hiện thực LRU cache
4. Bài số đảo
5. Duyệt cây nhị phân theo tầng
6. Tính Hamming Weight
7. Bài K-Coverage Intervals
