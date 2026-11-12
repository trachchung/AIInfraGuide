---
title: "Chương 3: Hiện thực toán tử cổ điển — Reduce"
description: "Ba phiên bản tối ưu dần của Reduce (naive, tree reduce trên shared memory, Warp Shuffle) để nắm phương pháp tối ưu toán tử CUDA từng bước"
tags: ["CUDA", "Reduce", "Shared memory", "Warp Shuffle", "Tối ưu toán tử"]
sourceHash: "cf8dc4a8e5c43081"
---

## Giới thiệu chương

Reduce (phép quy ước) là một trong những thuật toán song song kinh điển nhất, cũng là ví dụ mở đầu tốt để học tối ưu CUDA. Qua ba phiên bản tiến dần, bạn đi hết vòng "phân tích điểm nghẽn → tối ưu có chủ đích → đo lợi ích định lượng".

Phiên bản naive dùng global memory và atomic add. Đơn giản nhất nhưng chậm nhất; phân tích hiệu năng sẽ chỉ ra điểm nghẽn.

Shared memory kết hợp tree reduce thực hiện quy ước trong Block trên shared memory, loại bỏ Warp Divergence, và xử lý bước quy ước lần hai khi có nhiều Block.

Phiên bản Warp Shuffle thay shared memory bằng `__shfl_down_sync`, loại hết Bank Conflict và đạt Reduce hiệu quả ở cấp Warp.

Phần quy ước nhiều cấp và dữ liệu lớn giới thiệu kỹ thuật Grid-stride loop để xử lý mảng siêu lớn.

Thực hành: hiện thực ba phiên bản Reduce Sum, so sánh throughput bằng Nsight Compute, và nói rõ mỗi bước tối ưu tiết kiệm ở đâu.
