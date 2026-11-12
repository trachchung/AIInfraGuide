---
title: "Ngân hàng câu hỏi phỏng vấn AI Infra tổng hợp (2)"
description: "Câu hỏi phỏng vấn AI Infra thật, gồm tối ưu suy luận, tối ưu huấn luyện, tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "b7acbe904c34a807"
---
## Kiến thức cơ bản

1. Trình bày nguyên lý cốt lõi và cách hiện thực FlashAttention
2. Trên CPU, duyệt theo cột một ma trận lưu row-major chậm hơn duyệt theo hàng rõ rệt vì sao? Chỉ số hiệu năng nào xấu đi?
3. Dùng Shared Memory khi transpose ma trận trên GPU có lợi gì?
4. Flow Matching model dự đoán mục tiêu gì? Hiểu conditional velocity field có điều kiện theo mẫu x0 như thế nào?
5. Tính time shift trong QwenImage thế nào?
6. Có những phương án Weight-Only quantization nào? Khi viết CUDA kernel quantize Weight-Only, tối ưu truy cập bộ nhớ ra sao? Bạn có biết Marlin kernel không?
7. Sequence parallelism (SP) trong Megatron được hiện thực thế nào?
8. DeepSpeed ZeRO Stage 1 và Stage 2 khác nhau thế nào về lượng giao tiếp? Paper và code có lệch nhau không?
9. Trong giao tiếp nhiều GPU, NVSHMEM và NVLink khác gì?

## Bài PyTorch

1. Hiện thực Multi-Head Attention (3 biến thể)
2. Hiện thực FlashAttention v1
3. Viết pseudocode quá trình sampling của Flow Matching Model

## Thuật toán

1. Quicksort
2. Trung vị của hai mảng đã sắp
3. Next permutation
4. Đường đi có tổng lớn nhất trên cây nhị phân
5. Path Sum III
6. Cho tọa độ các điểm trên trục số và số đoạn thẳng cùng độ dài cố định, đoạn ngắn nhất bao nhiêu để phủ hết điểm?
7. Top-K chuỗi xuất hiện nhiều nhất (cùng tần suất thì xếp lexicographic tăng dần)
8. Bắt đầu từ chuỗi s; mỗi bước xoay phải một ký tự rồi nối vào cuối (độ dài nhân đôi). Sau khi mở rộng vô hạn, ký tự ở vị trí N là gì?
9. Hai ngón tay trên bàn phím 26 chữ cái: khoảng cách di chuyển tối thiểu để gõ chuỗi s
