---
title: "Chương 9: Huấn luyện chuỗi dài và Context Parallel"
description: "Phá bức tường độ dài chuỗi—bài toán O(s²) của Attention, so sánh ba hướng song song chuỗi dài: Ring Attention, DeepSpeed-Ulysses và Megatron Context Parallel"
tags: ["Huấn luyện chuỗi dài", "Context Parallel", "Ring Attention", "Ulysses", "FlashAttention"]
sourceHash: "156ec276609e75bd"
---

## Giới thiệu chương

Khi nhu cầu tài liệu dài, ngữ cảnh dài và chuỗi suy nghĩ dài tăng lên, độ dài chuỗi đã từ 2K lên 128K thậm chí 1M. Lượng tính toán và bộ nhớ activation của Attention tỉ lệ $O(s^2)$ với độ dài chuỗi, một GPU đơn không chứa nổi—từ đó sinh ra **Context Parallelism (CP)**, chia theo chiều chuỗi. Chương này đứng riêng để làm rõ khác biệt chiều cắt của ba hướng phổ biến. Cần phân biệt: Sequence Parallel (SP) ở chương trước cắt activation ngoài vùng Attention; chương này cắt **chính phép tính Attention**.

---

## Cấu trúc chương

### 1. Bức tường độ dài chuỗi

- Tính toán và bộ nhớ $O(s^2)$ của Attention: $s$ gấp đôi thì chi phí gấp 4
- KV Cache và bộ nhớ activation tăng theo $s$
- Vì sao TP/PP/DP đều không giải được bài "một chuỗi quá dài"

### 2. Ôn lại FlashAttention (tiên quyết)

- Tính theo khối + online softmax, đưa bộ nhớ từ $O(s^2)$ xuống $O(s)$
- Vì sao hầu hết các hướng CP đều xây trên FlashAttention

### 3. Ring Attention

- Cắt Q/K/V theo chiều chuỗi ra nhiều GPU
- Truyền vòng các mảnh KV; mỗi GPU tính Attention cục bộ theo khối rồi gộp bằng online softmax
- Chồng tính toán với truyền KV
- Mất cân bằng tải dưới causal mask và tối ưu Zigzag/Striped

### 4. DeepSpeed-Ulysses

- Cắt theo chiều Head → All-to-All sắp xếp lại thành cắt theo chiều chuỗi
- Phân tích lượng giao tiếp: không phụ thuộc độ dài chuỗi, phụ thuộc chiều Head
- Đánh đổi với Ring Attention: kiểu giao tiếp vs giới hạn số Head

### 5. Megatron Context Parallel

- Cắt theo chiều chuỗi, kết hợp tính theo khối của FlashAttention
- Phối hợp với TP/SP: trong Megatron mở đồng thời TP + CP như thế nào
- Quan hệ giữa chi phí giao tiếp và độ dài chuỗi

### 6. So sánh ba hướng

| Hướng | Chiều cắt | Primitive giao tiếp | Giới hạn | Tình huống phù hợp |
|------|---------|---------|------|---------|
| Ring Attention | Chuỗi | P2P vòng truyền KV | Mất cân bằng tải dưới causal mask | Chuỗi cực dài, KV lớn |
| Ulysses | Head→chuỗi | All-to-All | Số Head cần ≥ bậc CP | Mô hình nhiều Head |
| Megatron CP | Chuỗi | P2P/AllGather | Gắn với Megatron | Hệ sinh thái Megatron |

### 7. Thực hành

- Dùng FlashAttention chạy chuỗi dài trên một GPU, quan sát bộ nhớ theo $s$
- Hiện thực bản rút gọn Ring Attention: truyền vòng KV + gộp online softmax

---

## Mục tiêu học tập

- Giải thích được bài toán $O(s^2)$ của Attention và vì sao cần cắt theo chiều chuỗi
- Phân biệt SP (activation ngoài Attention) với CP (tính toán Attention) theo chiều cắt
- So sánh Ring Attention, Ulysses, Megatron CP về chiều cắt và kiểu giao tiếp
- Nói được vấn đề cân bằng tải mà causal mask gây ra cho Ring Attention và hướng tối ưu
