---
title: "Chương 2: Công nghệ cốt lõi của inference engine"
description: "PagedAttention, Continuous Batching, Prefix Cache, Chunked Prefill, backend Attention và tối ưu đồ thị—năm kỹ thuật cốt lõi của inference engine hiện đại"
tags: ["PagedAttention", "Continuous Batching", "Prefix Cache", "Chunked Prefill", "FlashAttention", "CUDA Graph"]
sourceHash: "99bbf47949796d3f"
---

## 📖 Giới thiệu chương

Hiệu năng cao của inference engine LLM hiện đại đến từ nhiều kỹ thuật cốt lõi phối hợp với nhau. Chương này đi từng kỹ thuật và đối chiếu với cách vLLM triển khai.

**PagedAttention** mượn ý tưởng phân trang bộ nhớ ảo của hệ điều hành: dùng block table ánh xạ khối logic sang khối vật lý, xử lý cả phân mảnh nội và ngoại của KV Cache, và nâng mức tận dụng GPU memory rõ rệt.

**Continuous Batching** giải quyết vấn đề Static Batching khi request ngắn bị request dài kéo chậm: lịch điều phối xuống từng bước iteration (iteration-level scheduling), request đến là ghép vào, xong là thoát, nên batch không còn slot trống chạy.

**Prefix Cache / RadixAttention** nhắm vào trường hợp nhiều request dùng chung System Prompt. Prefix Cache của vLLM tái sử dụng khối KV sẵn có qua chuỗi hash khối; RadixAttention của SGLang khớp tiền tố dài nhất trên Radix Tree và mạnh hơn khi chia sẻ dạng cây. vLLM V1 ép chi phí ghi sổ của prefix cache xuống rất thấp và bật mặc định.

**Chunked Prefill và lịch Token Budget thống nhất** cắt Prompt dài thành nhiều Chunk để Prefill ít làm nhiễu Decode hơn. vLLM V1 đi xa hơn: bỏ phân đôi Prefill/Decode và dùng một scheduler ngân sách token duy nhất chứa cả hai loại request.

**Backend Attention và tối ưu đồ thị** giới thiệu các backend Attention có thể cắm của vLLM (FlashAttention 2/3/4, FlashInfer, FlashMLA, Triton) và cách chúng phục vụ batch Prefill/Decode hỗn hợp, cùng CUDA Graph và torch.compile để bỏ chi phí khởi chạy từng kernel và giảm nút thắt CPU ở giai đoạn Decode.

> ⏰ **Thời điểm thông tin**: nguyên lý từng kỹ thuật theo các paper tương ứng; giá trị mặc định tham số của vLLM, phần đọc mã scheduling và quản lý KV Cache, cùng chế độ CUDA Graph dựa trên v0.32.0 (phát hành 2026-10-18).

## 📑 Các bài trong chương

- [**2.1 PagedAttention: quản lý KV Cache như OS quản lý bộ nhớ**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/): hai loại phân mảnh khi cấp phát liên tục, ánh xạ Block Table, định địa chỉ trong kernel, CoW và chia sẻ chỉ đọc của V1, preemption và tính lại, mã nguồn `KVCacheManager` với `BlockPool`
- [**2.2 Continuous Batching: batch động để GPU không còn chờ không**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/): slot trống trong Static Batching, lịch theo iteration, phối hợp với PagedAttention, hàng đợi và preemption của V1 Scheduler
- [**2.3 Prefix Cache và RadixAttention: tái sử dụng KV của tiền tố chung**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/): chuỗi hash khối, loại bỏ theo LRU và đếm tham chiếu, Radix Tree của SGLang, mã nguồn APC của vLLM
- [**2.4 Chunked Prefill và lịch Token Budget thống nhất**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/): đỉnh TPOT do Prefill dài, cắt chunk, đóng gói theo ngân sách token, tinh chỉnh `max_num_batched_tokens`
- [**2.5 Backend Attention và tối ưu đồ thị**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/): backend có thể cắm và lựa chọn mặc định, batch hỗn hợp, CUDA Graph và Padding, Piecewise với `FULL_AND_PIECEWISE`, cắt đồ thị bằng torch.compile
