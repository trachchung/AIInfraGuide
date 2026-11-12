---
title: "Tencent AI Infra tuyển mới tốt nghiệp vòng một và hai (1)"
description: "Câu hỏi phỏng vấn Tencent AI Infra tuyển mới tốt nghiệp vòng một và hai thật, gồm tối ưu toán tử và các hướng liên quan"
tags: ["Tối ưu toán tử"]
sourceHash: "ff8f819876634132"
---

### Vòng một

#### Tối ưu suy luận

1. So sánh khác biệt hiện thực source của TensorRT-LLM, AGI và vLLM
2. Động lực thiết kế của continuous batching là gì
3. Giới thiệu cấu trúc source và hướng hiện thực của fastllm.cpp

#### Kiến thức nền

4. Trong Python, tác vụ nặng compute nên dùng multiprocess hay multithread
5. Cơ chế hiện thực inheritance ở tầng thấp trong C++
6. Các cách tăng tốc inference mô hình YOLO phổ biến

#### Bài lập trình

7. Tổng mảng con lớn nhất

---

### Vòng hai

#### Tối ưu suy luận

1. Giới thiệu nguyên lý cốt lõi của vLLM và PagedAttention
2. Hướng hiện thực PagedAttention bằng Triton

#### Kiến thức nền

3. Phân tầng và đặc điểm của CUDA memory model
4. Cách chạy ở compile time trong C++ (constexpr và tương tự)

#### Bài lập trình

5. Đếm số bit 1 trong biểu diễn nhị phân của một số nguyên
