---
title: "Chương 1: Nhập môn lập trình CUDA"
description: "Dựng môi trường phát triển CUDA, hiểu mô hình lập trình Grid/Block/Thread và mô hình bộ nhớ, viết CUDA kernel thực dụng đầu tiên"
tags: ["CUDA", "Mô hình lập trình", "Mô hình bộ nhớ", "Kernel"]
sourceHash: "ec526a1f6db6b0c8"
---

## Giới thiệu chương

Chương này là điểm bắt đầu của lập trình CUDA: dựng môi trường phát triển từ đầu và viết kernel hiệu năng cao đầu tiên.

Phần dựng môi trường gồm cài đặt CUDA Toolkit và quản lý phiên bản, chuỗi công cụ biên dịch nvcc cùng tích hợp CMake, rồi viết kernel Hello World đầu tiên.

Phần mô hình lập trình đi chi tiết hệ phân cấp thread ba tầng Grid/Block/Thread, cách tính chỉ số thread (threadIdx/blockIdx/blockDim/gridDim), cú pháp launch kernel `<<<gridDim, blockDim>>>` và chiến lược chọn kích thước Block.

Phần mô hình bộ nhớ giới thiệu đặc tính và tình huống dùng của global memory, shared memory, register, constant memory và unified memory.

Kernel thực dụng đầu tiên lấy phép cộng vector làm ví dụ, đi hết luồng cudaMalloc → cudaMemcpy → kernel launch → kiểm tra lỗi, rồi so sánh thời gian CPU và GPU.
