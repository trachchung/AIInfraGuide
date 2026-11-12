---
title: "Enflame AI Infra thực tập vòng một"
description: "Câu hỏi phỏng vấn Enflame AI Infra thực tập vòng một thật, gồm tối ưu suy luận, tối ưu toán tử và các hướng liên quan"
tags: ["Tối ưu suy luận", "Tối ưu toán tử"]
sourceHash: "8c6b902c1ebe6063"
---

Dưới đây là nội dung khảo sát vòng một thực tập AI Infra tại Enflame.

### Kinh nghiệm dự án

1. Hỏi sâu trải nghiệm thực tập
2. Hỏi sâu kinh nghiệm dự án

### Quantization

3. Căn cứ chọn chiến lược quantization: vì sao chọn INT8; A100 và H100 hỗ trợ các độ chính xác quantization khác nhau thế nào
4. Đối tượng quantization là weight mô hình hay KV Cache; tham số scale xác định thế nào
5. Sau quantization đã đánh giá mất mát độ chính xác chưa

### Phát triển toán tử

6. Logic triển khai toán tử Triton, gồm chiến lược chia khối và tương tự
7. Baseline chính thức dùng để so sánh và kiểu dữ liệu
8. Nguồn số liệu tăng hiệu năng; chiến lược chia khối động và cấu hình toán tử
9. Đã cân nhắc dùng CUDA thay Triton để viết toán tử chưa; vì sao chọn Triton
10. Đã profiling chưa; throughput bộ nhớ và các chỉ số hiệu năng ra sao; hướng tối ưu tiếp theo

### Tối ưu suy luận

11. Module Attention chiếm tỷ lệ bao nhiêu trong latency end-to-end của cả hệ thống
12. Giai đoạn decode thuộc compute bound hay memory bound; quantization KV Cache cải thiện mặt nào
13. Trần băng thông bộ nhớ lý thuyết của A100
