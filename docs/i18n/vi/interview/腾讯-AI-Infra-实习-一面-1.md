---
title: "Tencent AI Infra thực tập vòng một (1)"
description: "Câu hỏi phỏng vấn thực tập Tencent AI Infra vòng một thật, gồm tối ưu suy luận, tối ưu toán tử, high-performance computing và các hướng liên quan"
tags: ["Tối ưu suy luận", "Tối ưu toán tử", "High-performance computing"]
sourceHash: "89dd3103f8ec16ef"
---

### Kinh nghiệm dự án

1. Giới thiệu kinh nghiệm dự án và chi tiết kỹ thuật liên quan

### Tối ưu suy luận

2. Trình bày động lực thiết kế của chunked prefill và bài toán cốt lõi nó giải quyết
3. Trong cluster quy mô lớn (trong node có NVLink, giữa một số máy liên node có RDMA), thiết kế phương án distributed inference thế nào

### High-performance computing

4. So sánh hai collective communication reduce-scatter và all-to-all
5. Có những cách nào giảm overhead launch kernel
6. Nguyên nhân bank conflict trong lập trình CUDA và cách xử lý

### Bài lập trình

7. Reverse nodes in k-Group
