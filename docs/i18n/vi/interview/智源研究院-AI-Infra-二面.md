---
title: "BAAI AI Infra vòng 2"
description: "Câu hỏi phỏng vấn AI Infra tại BAAI vòng hai thật, gồm tối ưu suy luận và tối ưu toán tử"
tags: ["Tối ưu suy luận", "Tối ưu toán tử"]
sourceHash: "8d38c706736c9b89"
---

## Kiến thức cơ bản

1. Mô tả mô hình tính CUDA, gồm cách cấu hình block/grid và ảnh hưởng tới hiệu năng toán tử.
2. Các warp khác nhau trong cùng block giao tiếp dữ liệu thế nào?
3. Ý tưởng hiện thực CUDA reduce.
4. Khi hiện thực Softmax bằng CUDA, xử lý cấp warp khác cấp block thế nào?
5. Static graph và dynamic graph là gì? Xử lý dynamic shape thế nào?
6. Trong framework suy luận, compute graph, runtime graph và quản lý bộ nhớ mỗi phần làm gì? Thiết kế kiến trúc tổng thể ra sao?
7. Các kỹ thuật tối ưu đồ thị phổ biến?

## Tối ưu toán tử

8. So sánh điểm cải tiến cốt lõi từ FlashAttention v1 sang v2.
