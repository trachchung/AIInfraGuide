---
title: "Chương 5: Tổng quan phần cứng GPU"
description: "Triết lý thiết kế kiến trúc GPU, hệ thống phân cấp bộ nhớ, so sánh thông số GPU phổ biến và topology kết nối, để có nền tảng phần cứng trước khi lập trình CUDA"
tags: ["GPU", "Kiến trúc phần cứng", "NVLink", "Roofline", "Phân cấp bộ nhớ"]
sourceHash: "9e53e5b54c1923ef"
---

## Giới thiệu chương

Trước khi viết CUDA, bạn cần hiểu "nhà máy được bố trí thế nào". Kiến trúc phần cứng GPU quyết định trực tiếp code của bạn chạy nhanh đến mức nào.

Phần tổng quan kiến trúc GPU so sánh triết lý thiết kế CPU vs GPU (tối ưu độ trễ vs tối ưu thông lượng), giới thiệu cấu trúc phân cấp SM, CUDA Core, Tensor Core, và khái niệm Warp (đơn vị lập lịch tối thiểu 32 thread).

Phần phân cấp bộ nhớ GPU đi từ thanh ghi đến HBM rồi bộ nhớ host, gồm dung lượng, băng thông và bậc độ trễ từng tầng, cùng vấn đề Memory Wall: vì sao băng thông VRAM thường thành nút thắt trước cả sức tính toán.

Phần so sánh thông số GPU phổ biến liệt kê các thông số chính của A100/H100/H200/B200, và giới thiệu Arithmetic Intensity cùng Roofline Model.

Phần topology kết nối nói về NVLink/NVSwitch trong một máy và mạng InfiniBand đa máy, cách đọc output của `nvidia-smi topo -m`, và vì sao băng thông kết nối quyết định trực tiếp việc chọn chiến lược song song.
