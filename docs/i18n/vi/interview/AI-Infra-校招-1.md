---
title: "AI Infra tuyển dụng sinh viên (1)"
description: "Câu hỏi phỏng vấn AI Infra tuyển dụng sinh viên thật, gồm tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "a94b0ab375c48f68"
---
Vị trí phù hợp: heterogeneous computing / nghiên cứu framework AI / high-performance computing / triển khai mô hình / tối ưu thuật toán / phát triển toán tử

## Kiến thức cơ bản

1. Mô hình phân cấp phần mềm trong CUDA (Grid, Block, Thread) được tổ chức như thế nào?
2. Hệ thống phân cấp bộ nhớ CUDA gồm những loại nào (global memory, shared memory, register, ...)? Đặc điểm từng loại là gì?
3. CPU và GPU khác nhau thế nào về thiết kế kiến trúc? Vì sao GPU phù hợp tính toán song song hơn?
4. Hiểu quan hệ giữa SM (Streaming Multiprocessor) và SP (Streaming Processor) như thế nào?
5. CUDA Stream là gì? Stream đồng bộ và bất đồng bộ khác nhau ra sao?
6. Cache L1/L2 của GPU mỗi cái đảm nhận vai trò gì?
7. Khi dùng shared memory cần lưu ý gì (đồng bộ thread, bank conflict, ...)?
8. Tối ưu một CUDA kernel có thể bắt đầu từ những chiều nào?

## Bài lập trình

1. Hiện thực kernel nhân ma trận
2. Hiện thực Softmax reduction
3. Chuyển định dạng dữ liệu từ NCHW sang NHWC
4. Cho mảng độ dài n (phần tử trong 0~256), đếm tần suất mỗi giá trị và ghi vào mảng khác
5. Đưa phần tử chỉ số lẻ sang trái, chỉ số chẵn sang phải, thao tác tại chỗ
