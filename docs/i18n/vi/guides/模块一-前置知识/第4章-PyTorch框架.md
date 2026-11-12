---
title: "Chương 4: Framework PyTorch"
description: "Nắm Tensor, autograd, vòng huấn luyện Module và công cụ debug của PyTorch, rồi tự triển khai một mô hình nhỏ cấp GPT-2 và hoàn thành toàn bộ quy trình huấn luyện"
tags: ["PyTorch", "autograd", "Vòng huấn luyện", "Phân tích hiệu năng"]
sourceHash: "a08faea24505e53c"
---

## Giới thiệu chương

PyTorch là framework huấn luyện phổ biến nhất trong AI Infra, cũng là nền tảng cho các phần huấn luyện phân tán và tối ưu suy luận phía sau. Mục tiêu chương này là giúp bạn nắm vững cơ chế cốt lõi của PyTorch để tự phát triển và debug mô hình.

Phần Tensor và autograd gồm thao tác Tensor, quản lý thiết bị CPU/GPU, cơ chế đồ thị tính toán autograd, cộng dồn gradient và xóa gradient.

Phần Module và vòng huấn luyện đi hết một vòng huấn luyện đầy đủ: DataLoader → forward → loss → backward → optimizer.step, gồm cả lập lịch learning rate và lưu/tải Checkpoint.

Phần debug và phân tích hiệu năng dùng `torch.cuda.memory_summary()` để xem VRAM, `torch.profiler` để phân tích hiệu năng, và xử lý các lỗi thường gặp (shape/device mismatch, OOM).

Thực hành: dùng PyTorch tự viết một mô hình nhỏ cấp GPT-2, hoàn thành trên một GPU quy trình tải dữ liệu, huấn luyện, đánh giá và lưu mô hình.
