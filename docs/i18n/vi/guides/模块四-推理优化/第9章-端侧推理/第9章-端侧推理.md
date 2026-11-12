---
title: "Chương 9: Suy luận trên thiết bị đầu cuối"
description: "Hiểu ràng buộc tài nguyên, phần cứng dị chủng, runtime và phương pháp tối ưu của suy luận trên thiết bị đầu cuối, đồng thời xây quy trình đánh giá hiệu năng có thể tái lập"
tags: ["Suy luận trên thiết bị đầu cuối", "Edge AI", "ExecuTorch", "Quantization", "Heterogeneous computing"]
sourceHash: "4928b909f69d712a"
---

## Giới thiệu chương

Đưa mô hình lên điện thoại, ô tô, camera, PC hoặc thiết bị nhúng không phải cùng một bài toán với triển khai trong data center. Thiết bị đầu cuối bị hạn chế hơn về bộ nhớ, công suất, tản nhiệt và môi trường phần mềm, trong khi chủng loại phần cứng lại phân tán hơn; chạy nhanh một lần chưa đồng nghĩa với việc chạy ổn định liên tục.

Chương này đi theo chuỗi kỹ thuật đầy đủ: cách xuất mô hình từ PyTorch thành chương trình chạy trên thiết bị đầu cuối, cách runtime phân công tính toán cho CPU, GPU, NPU hoặc DSP, cùng cách dùng quantization, tối ưu đồ thị, lập kế hoạch bộ nhớ và đánh giá hiệu năng để xây hệ thống suy luận trên thiết bị đầu cuối đáng tin cậy.

## Các bài trong chương

- 9.1 Cơ bản suy luận trên thiết bị đầu cuối: từ xuất mô hình đến thực thi trên phần cứng dị chủng: ràng buộc đầu cuối, ngăn xếp phần mềm/phần cứng, chọn runtime, phương pháp tối ưu, Benchmark và ví dụ tối thiểu với ExecuTorch/XNNPACK
