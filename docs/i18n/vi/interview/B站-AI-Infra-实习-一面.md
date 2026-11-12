---
title: "Bilibili AI Infra thực tập vòng một"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại Bilibili vòng một thật, gồm tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "59d780b6e7ab843f"
---
## Lý thuyết

1. Định nghĩa process, thread và coroutine? Chúng khác nhau thế nào về chiếm tài nguyên và cách lập lịch?
2. Trong lập lịch CPU, lập lịch theo process khác theo thread ở khía cạnh công bằng thế nào?
3. Trong train phân tán, All-Reduce, All-Gather, All-to-All mỗi cái trao đổi dữ liệu gì? Phù hợp chiến lược song song nào?
4. Dùng profiling thế nào để biết toán tử GPU nghẽn ở truy cập bộ nhớ hay tính toán? Các hướng tối ưu chung là gì?
5. Sau khi biết loại nút thắt, có những biện pháp tinh chỉnh nào? Ví dụ tăng tính liên tục của truy cập bộ nhớ, giản lược logic tính toán, chỉnh kích thước block.

## Đào sâu dự án

6. Hỏi chi tiết về công việc tối ưu bạn đã tham gia

## Lập trình tại chỗ

7. CUDA: cộng reduction mảng lớn, dùng Shared Memory cho reduction trong block, rồi nhiều cấp kernel để gom toàn cục
