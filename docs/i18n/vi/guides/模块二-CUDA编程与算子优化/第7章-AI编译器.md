---
title: "Chương 7: Trình biên dịch AI"
description: "Mô hình lập trình Block-level của Triton, chế độ biên dịch torch.compile, cùng vai trò và khác biệt của TVM/XLA"
tags: ["Triton", "torch.compile", "Trình biên dịch AI", "TorchInductor", "TVM"]
sourceHash: "add9d72071ff1f99"
---

## Giới thiệu chương

Kernel CUDA viết tay có thể đạt hiệu năng cực đại nhưng chi phí phát triển cao. Trình biên dịch AI (AI compiler) đứng ở lớp trừu tượng cao hơn và cân bằng giữa hiệu năng với tốc độ phát triển. Chương này đi qua ba hướng phổ biến.

**Triton** là framework lập trình kernel AI đang được chú ý nhất hiện nay. Nó dùng mô hình lập trình Block-level (CUDA là Thread-level), nên hạ thấp ngưỡng lập trình GPU khá nhiều. Phần này giới thiệu cú pháp cơ bản của Triton (`tl.load` / `tl.store` / `tl.dot` / `tl.where`) và tái hiện tutorial Fused Softmax chính thức.

**torch.compile** là chế độ biên dịch của PyTorch 2.x, gồm TorchDynamo (bắt đồ thị ở mức bytecode Python) và TorchInductor (sinh mã phía backend). Trọng tâm là Graph Break—khi nào việc biên dịch bị cắt, cách tránh—và cách đánh giá lợi ích hiệu năng thực tế.

**TVM / XLA (tổng quan)** nói về khác biệt định vị: TVM hướng tới biên dịch và tối ưu đa phần cứng, XLA hướng tới tối ưu cả đồ thị tính toán, cùng mối quan hệ của chúng với Triton và torch.compile.
