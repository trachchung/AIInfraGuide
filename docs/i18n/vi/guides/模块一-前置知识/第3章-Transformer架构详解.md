---
title: "Chương 3: Kiến trúc Transformer cho kỹ sư AI Infra"
description: "Hiểu sâu từng thành phần của Transformer: Self-Attention, FFN, mã hóa vị trí, lớp chuẩn hóa, và sự tiến hóa kiến trúc từ MHA đến GQA/MLA, từ FFN đến MoE"
tags: ["Transformer", "Attention", "RoPE", "MoE", "Kiến trúc mô hình"]
sourceHash: "5d7e87c5f7c1080d"
---

## Giới thiệu chương

Transformer là kiến trúc trung tâm của kỷ nguyên mô hình lớn, cũng là thứ kỹ sư AI Infra phải hiểu sâu. Mọi operator bạn tối ưu và mọi chiến lược song song bạn thiết kế cuối cùng đều tác động lên một thành phần nào đó của Transformer.

Chương bắt đầu từ cơ chế Self-Attention: các phép chiếu Q/K/V, toàn bộ quá trình tính Attention và phân tích độ phức tạp O(N²). Tiếp theo là FFN cùng các hàm kích hoạt (ReLU/GELU/SwiGLU), mã hóa vị trí (Sinusoidal/RoPE/ALiBi) và các lớp chuẩn hóa (LayerNorm/RMSNorm, Pre-Norm vs Post-Norm).

Phần quá trình forward đầy đủ lần lượt theo dõi luồng dữ liệu từ Token Embedding đến Logits đầu ra, ghi rõ chiều input/output từng bước và tự tính số lượng tham số.

Cuối cùng, từ góc nhìn phân tán và suy luận, chương phân tích các biến thể kiến trúc: động lực tiến hóa MHA → MQA → GQA → MLA, và việc chuyển từ FFN → GLU → MoE ảnh hưởng thế nào đến cắt song song và tối ưu suy luận.
