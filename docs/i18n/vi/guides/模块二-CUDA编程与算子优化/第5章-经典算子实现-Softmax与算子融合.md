---
title: "Chương 5: Hiện thực toán tử cổ điển — Softmax và fusion toán tử"
description: "Hiện thực Softmax ổn định số và Online Softmax, nắm nguyên lý và thực hành fusion toán tử"
tags: ["CUDA", "Softmax", "Online Softmax", "Fusion toán tử", "Kernel Fusion"]
sourceHash: "95a5976b10752aef"
---

## Giới thiệu chương

Softmax là một trong những phép phi tuyến quan trọng nhất trong Transformer, cũng là kiến thức nền để hiểu FlashAttention. Chương này bắt đầu từ ổn định số, tối ưu dần tới Online Softmax, rồi đưa vào ý tưởng fusion toán tử (operator fusion).

Hiện thực Softmax ổn định số phân tích tràn số ở phiên bản naive, giới thiệu Safe Softmax (trừ giá trị lớn nhất), và phân tích chi phí ba lượt quét (max → exp-sum → normalize).

Online Softmax giảng nguyên lý online normalizer calculation, suy ra thuật toán Softmax một lượt quét, và hiện thực hiệu quả trên GPU — đây là kiến thức nền cốt lõi của FlashAttention.

Fusion toán tử giải thích vì sao cần fusion (giảm Kernel Launch và đọc/ghi global memory), giới thiệu các mẫu fusion phổ biến (Bias + Activation, LayerNorm + Dropout), và so sánh fusion thủ công với fusion tự động của trình biên dịch.

Thực hành: hiện thực Online Softmax CUDA Kernel, fusion Softmax + Scale thành một Kernel và so sánh hiệu năng.
