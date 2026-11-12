---
title: "Chương 4: Lượng tử hóa"
description: "Từ Scale và độ chi tiết, nắm SmoothQuant, GPTQ/AWQ, lượng tử hóa KV Cache và FP8/NVFP4/MXFP4, rồi chọn phương án và chạy thí nghiệm đối chiếu trên vLLM v0.32.0"
tags: ["Lượng tử hóa", "SmoothQuant", "GPTQ", "AWQ", "FP8", "NVFP4", "Lượng tử hóa KV Cache"]
sourceHash: "abe8c0fba9d0ee8e"
---

## Giới thiệu chương

Lượng tử hóa (quantization) là dùng một thước có ít vạch hơn để ghi số: mỗi số chiếm ít chỗ hơn, đọc ghi nhanh hơn, đổi lại là độ đọc thô hơn. Khó không phải ở "nén xuống bao nhiêu bit", mà ở cách để số đã thô vẫn ít làm hỏng đầu ra của mô hình, và cách để phần cứng thật sự lấy được lợi ích từ bit thấp.

Chương này đi theo bốn bước "khái niệm, phương pháp, định dạng, triển khai":

- **Khái niệm cơ bản**: Scale và zero-point, đối xứng và bất đối xứng, bit width và khoảng lưới (step size), độ chi tiết Per-tensor / Per-channel / Per-group / Per-token, calibration, cùng PTQ và QAT.
- **Ba đối tượng lượng tử hóa**: W8A8 dùng SmoothQuant chuyển outlier của activation sang weight; W4A16 nén weight bằng bù sai số của GPTQ hoặc scale nhạy activation của AWQ, rồi chạy hiệu quả bằng các kernel như Marlin; KV Cache dùng chiến lược nhóm của KIVI hoặc cache FP8 để giảm chỗ chiếm của mỗi token lịch sử.
- **Số thực bit thấp**: E4M3/E5M2 chia phạm vi và độ chính xác thế nào, vì sao FP4 không thể thiếu micro-block scaling, và Scale của NVFP4 với MXFP4 khác nhau ở đâu.
- **Chọn phương án và thực hành**: lọc ứng viên theo bottleneck và điều kiện phần cứng, hiểu vLLM nhận checkpoint và chọn kernel ra sao, rồi so sánh throughput và chất lượng sinh giữa FP16 và AWQ INT4.

> ⏰ **Độ cập nhật:** Phần nguyên lý không phụ thuộc framework; tham số, tên phương án, hỗ trợ phần cứng và lựa chọn kernel liên quan vLLM dựa trên v0.32.0 (phát hành 2026-10-18). Ma trận hỗ trợ lượng tử hóa đổi khá nhanh; khi đổi phiên bản hãy đối chiếu tài liệu chính thức và log khởi động.

## Các bài trong chương

- [**4.1 Cơ bản lượng tử hóa: Scale, độ chi tiết và PTQ/QAT**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/): lượng tử hóa affine và dequantization, đối xứng và bất đối xứng, bit width và khoảng lưới, độ chi tiết lượng tử hóa, calibration, PTQ và QAT
- [**4.2 Lượng tử hóa W8A8: SmoothQuant xử lý outlier của activation thế nào**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/): outlier theo kênh, scale tương đương, cường độ chuyển α, INT8 GEMM và công thức LLM Compressor
- [**4.3 Weight-only INT4: GPTQ, AWQ và Marlin**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/): lưu trữ W4A16, bù bậc hai của GPTQ, scale nhạy activation của AWQ, kernel fusion Marlin và cách vLLM chọn kernel
- [**4.4 Lượng tử hóa KV Cache: KIVI và cache FP8**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/): ước lượng chỗ chiếm KV, nhóm Key/Value và residual cache của KIVI, calibration Scale cho FP8 KV, bỏ lớp và kiểm chứng ngữ cảnh dài
- [**4.5 FP8 và NVFP4/MXFP4: số thực bit thấp hoạt động thế nào**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/): E4M3/E5M2, Scale của FP8, E2M1 và micro-block scaling, so sánh NVFP4 với MXFP4 và điều kiện phần cứng để chạy
- [**4.6 Chọn lượng tử hóa và thực hành vLLM**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/): chọn ứng viên theo bottleneck, phân tầng method / format / kernel, cổng lượng tử hóa online, thí nghiệm đối chiếu FP16 và AWQ INT4
