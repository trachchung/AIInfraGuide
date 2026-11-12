---
title: "Xiaomi AI Infra thực tập vòng 1–2"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại Xiaomi thật, gồm tối ưu suy luận, tối ưu huấn luyện, tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "6f0d4540e4f35c78"
---

### Vòng 1

#### Kinh nghiệm dự án

1. Giới thiệu công việc chính trong kỳ thực tập.
2. Phương án kỹ thuật và chi tiết triển khai dự án huấn luyện quy mô nghìn GPU.
3. Giới thiệu kinh nghiệm dự án thi đấu.

#### Kiến thức cơ bản

4. Khác biệt giữa deep copy và shallow copy trong Python.
5. Đặc điểm và use case của ba smart pointer C++ (unique_ptr, shared_ptr, weak_ptr).
6. Nguyên lý Copy-on-Write.
7. Nguyên lý Zero-Copy.

#### Bài lập trình

8. Hiện thực chuyển vị ma trận.

### Vòng 2

#### Tối ưu huấn luyện

9. Quy trình đầy đủ của distributed training cho large model và căn cứ chọn chiến lược parallel.
10. Giới thiệu các chiến lược parallel phân tán phổ biến.
11. Vì sao tensor parallelism (TP) có cả cắt theo hàng và theo cột? Mỗi cách nghĩa là gì?
12. Nguyên lý thiết kế và cách hiện thực sequence parallelism (SP) trong Megatron.

#### Kiến thức cơ bản

13. Kiến trúc Transformer gồm những layer và toán tử nào?
14. Đặc điểm cấu trúc và khác biệt giữa Encoder và Decoder.
15. Ý tưởng cốt lõi và cơ chế hiện thực của FlashAttention.
16. Nguyên lý tính Online Softmax.

#### Tối ưu toán tử

17. Trong CUDA, Block là khái niệm phần mềm hay phần cứng?
18. Các phương pháp tối ưu CUDA phổ biến?
19. Chiến lược cụ thể cho tối ưu truy cập bộ nhớ?
20. Làm sao overlap tính toán và truy cập bộ nhớ?
21. Khác biệt giữa L1 Cache và L2 Cache.
22. Quan hệ và khác biệt giữa shared memory và L1 Cache.

#### Tính toán hiệu năng cao

23. Khác biệt giữa ba smart pointer C++; động cơ thiết kế của shared_ptr là gì?
24. Nguyên lý AllReduce và các cách hiện thực phổ biến?
25. Phân tích lượng giao tiếp của Ring AllReduce.
26. Tree AllReduce có lợi thế gì so với Ring AllReduce?

#### Bài lập trình

27. Gộp hai linked list đã sắp xếp.
