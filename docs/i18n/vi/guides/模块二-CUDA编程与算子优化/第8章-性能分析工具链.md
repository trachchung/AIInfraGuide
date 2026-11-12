---
title: "Chương 8: Chuỗi công cụ phân tích hiệu năng"
description: "Nsight Systems, Nsight Compute và PyTorch Profiler—ba công cụ cốt lõi để chẩn đoán hiệu năng có hệ thống"
tags: ["Nsight Systems", "Nsight Compute", "PyTorch Profiler", "Phân tích hiệu năng"]
sourceHash: "ba11f43ec78f95f9"
---

## Giới thiệu chương

Muốn tối ưu thì phải chẩn đoán đúng—tối ưu mà không dùng Profiler chỉ là đoán mò. Chương này nắm ba công cụ phân tích hiệu năng cốt lõi trong AI Infra.

**Nsight Systems** phân tích toàn chuỗi CPU–GPU, dùng để tìm nguồn gốc GPU idle gap (nút thắt CPU, chờ giao tiếp, Kernel Launch Overhead), bắt trace một iteration huấn luyện và đọc ở mức tổng quan.

**Nsight Compute** đào sâu ở mức kernel. Kỹ năng chính là đọc panel SOL (Speed of Light)—xác định kernel là Memory Bound hay Compute Bound—cùng các panel Shared Memory, Warp State, và so sánh hai phiên bản kernel.

**PyTorch Profiler** gắn `torch.profiler` với TensorBoard để tìm toán tử tốn thời gian nhất và đánh giá lợi ích biên dịch của torch.compile.

**Thực hành**: dùng Nsight Systems bắt một training iteration và chỉ ra nguồn GPU idle gap; dùng Nsight Compute so sánh báo cáo hiệu năng của ba phiên bản Reduce.
