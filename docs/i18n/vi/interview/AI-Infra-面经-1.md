---
title: "Ghi chép phỏng vấn AI Infra (1)"
description: "Câu hỏi phỏng vấn AI Infra thật, gồm tối ưu suy luận, tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu suy luận", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "6144516798b11ea1"
---
## Kiến thức C++

1. Con trỏ và tham chiếu giống và khác gì?
2. Công dụng của `static`; khác nhau khi gắn biến thành viên và hàm thành viên; biến toàn cục static khác biến toàn cục thường thế nào?
3. Các loại smart pointer và cách dùng; `shared_ptr` có thread-safe không?
4. Rvalue reference là gì và dùng khi nào?
5. Vai trò của `std::move` và `std::forward`, khác nhau ra sao?
6. Vì sao copy constructor phải nhận tham số theo tham chiếu chứ không theo giá trị?
7. `new` khác `malloc` thế nào; cơ chế tầng dưới của `new`
8. Tổng quan mô hình bộ nhớ C++
9. Vì sao constructor không thể là virtual? Vì sao destructor nên là virtual?
10. Cách hiện thực đa hình và nguyên lý tầng dưới; vtable nằm đâu; đa hình template và partial specialization
11. `vector` mở rộng thế nào? Vì sao thường nhân đôi?
12. Cú pháp lambda và cách capture
13. Các loại lock và tình huống dùng
14. Khi thread chia sẻ bộ nhớ, condition variable và mutex dùng khi nào? Khác gì?
15. Bốn điều kiện cần của deadlock và cách phòng; `lock_guard` khác `unique_lock`
16. Hiện thực singleton thread-safe thế nào?
17. Ép object chỉ được tạo trên heap thế nào?
18. Quy trình biên dịch chương trình C++; thư viện động khác thư viện tĩnh
19. C++ gọi hàm viết bằng C thế nào?
20. Process khác thread thế nào?
21. Phân biệt big-endian và little-endian bằng ít nhất hai cách?

## Tính toán hiệu năng cao

1. Tổng quan kiến trúc GPU
2. Global memory khác shared (local) memory; dùng shared memory hiệu quả thế nào?
3. Cache hoạt động thế nào và nâng hit rate ra sao?
4. Temporal locality khác spatial locality
5. Compute-bound khác memory-bound
6. Các hướng tối ưu hiệu năng thường gặp
7. Luồng thực thi OpenCL
8. Vì sao nên giảm nhánh trong OpenCL kernel? Mask dùng để làm gì?
9. Tham số chính của OpenCL kernel gồm gì?
10. Vì sao separable convolution trên GPU thường chậm? Vì sao thuộc memory-bound?
11. Operator fusion; suy công thức fusion conv + BN và vì sao fusion được
12. Cách hiện thực convolution phổ biến trong framework suy luận
13. Nguyên nhân bank conflict và cách xử lý
14. Khái niệm cơ bản của TVM

## Cơ bản deep learning

1. Batch Normalization: cách tính, vai trò, khác biệt train/infer
2. Depthwise convolution khác Pointwise convolution
3. Lộ trình MobileNet v1/v2/v3; tư duy GhostNet; cấu trúc SE
4. YOLO v1/v2/v3 so với SSD
5. Global pooling thường đặt ở đâu trong mạng?
6. Knowledge distillation: nguyên lý và quy trình
7. Phân loại quantization; INT8: symmetric/asymmetric; QAT và chi tiết infer
8. Dropout khác nhau lúc train và infer
9. Deconvolution khác dilated convolution
10. GIoU loss: định nghĩa và vai trò
11. Công thức Softmax và vì sao dùng cho classification
12. Các cải tiến của NMS

## Bài lập trình

1. Hiện thực convolution
2. Hiện thực computation graph
3. Hiện thực Pooling
4. Hiện thực NMS
5. Nhân ma trận và cộng vector bằng OpenCL
