---
title: "Chương 3: Đọc mã nguồn nano-vllm"
description: "Inference engine tối giản khoảng 1200 dòng làm mạch chính: continuous batching, PagedAttention, FlashAttention, tensor parallelism và CUDA Graph trong code thật"
tags: ["nano-vllm", "Đọc mã nguồn", "Inference engine", "PagedAttention", "Continuous batching", "CUDA Graph"]
sourceHash: "96bb8e19046ee354"
---

## 📖 Giới thiệu chương

Chương 2 đã tách từng kỹ thuật cốt lõi của inference engine: PagedAttention quản lý GPU memory thế nào, Continuous Batching lấp đầy GPU ra sao, Prefix Cache tái sử dụng tiền tố thế nào, Chunked Prefill thống nhất lịch điều phối ra sao. Các mảnh đó ghép thành một pipeline trong engine thật như thế nào—đó là câu hỏi chương này trả lời.

Cách làm là đọc code. Mẫu là [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), inference engine LLM tối giản do Xingkai Yu (GitHub ID `GeeeekExplorer`; trang cá nhân ghi DeepSeek, Đại học Nam Kinh) mã nguồn mở. Dự án khoảng 1200 dòng Python, phủ cả chuỗi cấu hình, lịch, quản lý KV, forward mô hình và sampling, kèm prefix cache, tensor parallelism, `torch.compile` và CUDA Graph. Quan hệ của nó với vLLM giống nanoGPT với framework huấn luyện quy mô lớn: code đủ nhỏ để đọc hết một lần, nhưng vẫn giữ xương sống của hệ thống sản xuất.

Sau chương này, bạn nên nhìn thư mục mã nguồn và nói rõ từng lớp làm gì từ lúc request dạng chuỗi vào, đến token đầu tiên ra, rồi hết cả đoạn sinh—và map các phần đó sang thiết kế của vLLM.

> ⏰ **Thời điểm thông tin**: phân tích mã dựa trên commit nhánh `main` của nano-vllm ngày 2026-04-26 (phiên bản `pyproject.toml` 0.2.0; tính đến 2026-11-01 repo khoảng 16.2k stars, giấy phép MIT). Mô hình ví dụ là `Qwen/Qwen3-0.6B`. Mã nguồn vẫn đang đổi; nếu định danh quan trọng lệch, lấy phiên bản mới nhất trên repo làm chuẩn.

## 📑 Các bài trong chương

- **3.1 Toàn cảnh dự án và kiến trúc bốn lớp**: định vị dự án, cấu trúc thư mục, và thiết kế phân lớp "lớp giao diện → lớp engine → lớp scheduler → lớp thực thi" cùng luồng dữ liệu end-to-end
- **3.2 Cấu hình, điểm vào và một vòng generate đầy đủ**: `Config` / `SamplingParams`, quan hệ `LLM` với `LLMEngine`, ví dụ dùng trong README, và vòng lặp chính `generate → add_request → step → postprocess`
- **3.3 Khung mô hình và toán tử cốt lõi**: Embedding/LMHead song song theo vocab, RMSNorm, RoPE, SwiGLU, họ Linear song song và cắt trọng số tensor parallel, đường ghi KV của Attention cùng hai đường FlashAttention, sampling nhiệt độ trong Sampler, và cách lắp Qwen3
- **3.4 KV Cache và BlockManager**: cấu trúc `Block`, ý tưởng chia khối của PagedAttention, prefix cache dựa trên hash, cấp phát và thu hồi khối, cùng vòng đời `Sequence`
- **3.5 Scheduler và continuous batching**: hàng Waiting/Running, quyết định lịch Prefill/Decode, Token Budget, preemption và tính lại
- **3.6 ModelRunner, CUDA Graph và benchmark hiệu năng**: chuẩn bị tensor đầu vào, inject Context toàn cục, cấp phát GPU memory cho KV Cache, capture và replay CUDA Graph, cùng benchmark chính thức so với vLLM
