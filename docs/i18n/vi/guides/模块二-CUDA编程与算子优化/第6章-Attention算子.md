---
title: "Chương 6: Toán tử Attention"
description: "Nguyên lý và hiện thực FlashAttention V1/V2/V3, cùng Flash-Decoding và hiện thực CUDA PagedAttention cho giai đoạn Decode"
tags: ["FlashAttention", "Attention", "Flash-Decoding", "PagedAttention", "CUDA"]
sourceHash: "e94f6639ae541ab0"
---

## Giới thiệu chương

Attention là phép tính cốt lõi của Transformer, cũng là trọng tâm tối ưu trong AI Infra. Chương này xuất phát từ vấn đề hiệu năng của Attention chuẩn, rồi đi sâu vào chuỗi FlashAttention và các kỹ thuật tăng tốc giai đoạn Decode.

Vấn đề hiệu năng của Attention chuẩn phân tích chiếm dụng bộ nhớ O(N²) của hiện thực naive và các điểm nghẽn tính toán/bộ nhớ.

FlashAttention V1 giải thích ý tưởng cốt lõi: Tiling + Online Softmax để tránh vật hóa ma trận N×N, đưa bộ nhớ phụ từ O(N²) xuống O(N), lưu lượng HBM từ Θ(N²) xuống Θ(N²d²/M) (M là kích thước SRAM), cùng chiến lược tính lại (recomputation) trong lan truyền ngược.

FlashAttention V2 cải thiện chiến lược song song (song song theo chiều chuỗi Q) và tối ưu Causal Mask.

FlashAttention-3 dùng đặc tính kiến trúc Hopper (WGMMA, TMA, FP8) để tối ưu pipeline bất đồng bộ.

Tăng tốc giai đoạn Decode gồm Flash-Decoding, FlashDecoding++, FlashInfer và các engine Attention hướng Serving khác.

Hiện thực CUDA PagedAttention đọc cách vLLM ánh xạ trang ảo sang trang vật lý trên GPU.

Thực hành: suy ra quá trình Tiling của FlashAttention trên bảng trắng, hiện thực bản FlashAttention đơn giản bằng Triton.
