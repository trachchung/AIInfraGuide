---
title: "Teleai AI Infra thực tập vòng một"
description: "Câu hỏi phỏng vấn thực tập Teleai AI Infra vòng một thật, gồm tối ưu suy luận, tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu suy luận", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "ef94ec94e148e8bb"
---
## Lý thuyết

1. CLIP hoạt động và luồng suy luận thế nào?
2. Các hướng tối ưu toán tử phổ biến trong suy luận LLM (truy cập bộ nhớ, chiến lược song song, ...)?
3. DeepSeek-R1 khác LLM nền tảng thông thường ở đâu?
4. Bạn đã viết CUDA operator chưa? Hiệu năng đạt mức nào?
5. Muốn customize hoặc phát triển thêm trên vLLM, bắt đầu từ module/layer nào?
6. vLLM và SGLang khác gì? SGLang mạnh hơn vLLM ở đâu? Vì sao SGLang tốt hơn trên mô hình reasoning?
7. Nguyên lý cơ bản của quantization LLM? Đã deploy mô hình quantize chưa? Đã đọc code AutoAWQ chưa?
8. CANN và MindIE trên Huawei Ascend gồm những thành phần cốt lõi nào?

## Tình huống mở

9. Adapt vLLM sang kiến trúc mô hình tự phát triển thế nào?
10. V100 có bao nhiêu VRAM? Quantize INT8 mô hình DeepSeek hoặc Qwen 32B có chạy trên V100 không? Phương án deploy cụ thể?
11. High concurrency thì pressure-test dịch vụ suy luận để tìm max concurrency thế nào? Cần theo dõi chỉ số nào (batch size, latency, throughput, ...)?
