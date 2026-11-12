---
title: "ByteDance AI Infra thực tập vòng một (1)"
description: "Câu hỏi phỏng vấn ByteDance AI Infra thực tập vòng một, gồm tối ưu toán tử"
tags: ["Tối ưu toán tử"]
sourceHash: "6b3e5d53a813efe7"
---
## Tối ưu toán tử

1. Có kinh nghiệm tối ưu cấp kernel không? Ví dụ dùng CUTE DSL hoặc tự viết CUDA để fusion toán tử — hãy nói cụ thể.
2. Khi làm kernel fusion, thường thiên về cách triển khai nào?
3. Đã gặp fusion xong hiệu năng lại giảm chưa? Nguyên nhân là gì?
4. Cơ chế warp specialization trên kiến trúc Hopper là gì? Triển khai tầng thấp ra sao?
5. Nếu bỏ warp specialization, chỉ giữ tiling và tối ưu shared memory, tổn thất hiệu năng chủ yếu nằm ở đâu?

## Tối ưu suy luận

6. Có kinh nghiệm thực tế tối ưu suy luận mô hình MoE (ví dụ trong kịch bản RL) không?
7. Làm sao biết MoE thực sự học được phân công chuyên gia, chứ không chỉ tách mô hình dense?
8. Trong huấn luyện RL + MoE, đã gặp reward khiến routing suy biến (mọi request dồn vào ít expert) chưa? Xử lý thế nào?

## Kinh nghiệm dự án

9. Hãy giới thiệu kinh nghiệm dự án của bạn.
10. Hãy giới thiệu kỳ thực tập, gồm các tối ưu cho dịch vụ suy luận đang chạy online.
11. Đã thử dùng Agent sinh CUDA kernel chưa? Phương án cụ thể là gì?
