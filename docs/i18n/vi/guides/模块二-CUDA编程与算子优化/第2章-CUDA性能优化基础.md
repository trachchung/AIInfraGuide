---
title: "Chương 2: Cơ bản tối ưu hiệu năng CUDA"
description: "Mô hình thực thi Warp, tối ưu truy cập bộ nhớ, tinh chỉnh Occupancy và cơ chế đồng bộ—xây phương pháp tối ưu hiệu năng CUDA cốt lõi"
tags: ["CUDA", "Warp", "Coalesced access", "Occupancy", "Tối ưu hiệu năng"]
sourceHash: "5c1e74ef36575373"
---

## Giới thiệu chương

Viết được CUDA chạy được chỉ là điểm bắt đầu; viết được code chạy nhanh mới là năng lực cốt lõi của kỹ sư AI Infra. Chương này xây phương pháp tối ưu hiệu năng CUDA cốt lõi.

Phần Warp và mô hình thực thi đi chi tiết chế độ SIMT, tổn thất hiệu năng do Warp Divergence, và lệnh Warp Shuffle để trao đổi dữ liệu hiệu quả giữa các thread.

Tối ưu truy cập bộ nhớ thường là đòn bẩy lớn nhất: Coalesced Access quyết định hiệu quả global memory, Bank Conflict ảnh hưởng shared memory (kỹ thuật Padding khắc phục), và tải vector hóa (float4/int4) đẩy thêm mức dùng băng thông.

Occupancy và phân bổ tài nguyên giải thích định nghĩa cùng ý nghĩa của Occupancy, các yếu tố ảnh hưởng (số register, shared memory, kích thước Block), và nhấn mạnh Occupancy không phải càng cao càng tốt—cần cân giữa Latency Hiding và Resource Utilization.

Đồng bộ và atomic gồm `__syncthreads()` để đồng bộ trong block, cùng cách dùng atomic và ảnh hưởng hiệu năng của chúng.
