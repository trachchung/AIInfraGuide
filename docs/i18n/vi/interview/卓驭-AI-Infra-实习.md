---
title: "Zhuoyu AI Infra thực tập"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại Zhuoyu thật, gồm tối ưu suy luận và tối ưu toán tử"
tags: ["Tối ưu suy luận", "Tối ưu toán tử"]
sourceHash: "5f8f68262f252e23"
---

## Kiến thức nền

1. Suy luận mô hình lớn chia thành những giai đoạn nào? Đặc điểm mỗi giai đoạn?
2. Quan hệ giữa operator và tensor là gì?
3. Compute graph thường được xây dựng bằng cách nào?
4. Định nghĩa throughput là gì? Đo trong kịch bản suy luận thế nào?
5. Phân biệt memory-bound và compute-bound thế nào?
6. Khi mô hình và toán tử đã cố định, những yếu tố nào còn ảnh hưởng tốc độ suy luận?
7. Độ dài prompt có ảnh hưởng tốc độ suy luận không? Giai đoạn decode thì sao? Ảnh hưởng của độ dài context?
8. Import file trọng số vào mô hình tự dựng thế nào?

## Tối ưu suy luận

9. Giai đoạn prefill có cách nào tăng tốc tạo KV Cache không?
10. Trong prefill, các token có phụ thuộc lẫn nhau không? Có xử lý song song được không?
11. Sparse KV Cache có cần sửa quy trình train mô hình, hay chỉ là tối ưu kỹ thuật thuần?
12. Để nâng thêm hiệu năng suy luận mô hình lớn, còn những kỹ thuật nào?

## Công cụ và phương pháp

13. Dùng Nsight hỗ trợ tối ưu hiệu năng thế nào? Chỉ số nào cần chú ý?
14. Đã quan tâm kiến trúc tính toán hay kiến trúc chip chưa?

## Kinh nghiệm dự án

15. Giới thiệu hướng nghiên cứu luận văn tốt nghiệp.
16. Giới thiệu các thành phần then chốt và thiết kế module trong dự án.
