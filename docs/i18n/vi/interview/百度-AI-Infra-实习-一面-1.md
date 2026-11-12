---
title: "Baidu AI Infra thực tập vòng một (1)"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại Baidu vòng một thật, gồm tối ưu suy luận, toán tử và HPC"
tags: ["Tối ưu suy luận", "Tối ưu toán tử", "Máy tính hiệu năng cao"]
sourceHash: "979db1a2f725d1c9"
---

## Suy luận mô hình lớn

1. Nguyên lý cốt lõi của FlashAttention và quá trình suy diễn toán
2. FusedAttention dùng những chiến lược tối ưu nào?
3. Vì sao RMSNorm nhanh hơn LayerNorm?
4. Mô hình Llama gồm bao nhiêu fully connected layer?
5. Mô tả luồng suy luận Llama2; mỗi layer gồm những toán tử nào?
6. Thiết kế một phương án cấp phát GPU memory linh hoạt và hiệu quả hơn cudaMalloc (cudaAllocator)

## Nền C++

7. C++11 đưa vào những tính năng mới quan trọng nào?
8. Phân loại smart pointer và đặc điểm từng loại
9. unique_ptr đảm bảo ownership duy nhất thế nào?
10. Sau khi reference count của shared_ptr về 0, destructor chạy khi nào?
11. Member function của class có thể định nghĩa thành template function không?
12. Định nghĩa và khác biệt giữa lvalue và rvalue

## CUDA programming

13. CUDA cung cấp những mô hình hoặc cách lập trình nào?
14. Khác biệt chức năng giữa Tensor Core và CUDA Core

## Bài lập trình

15. Longest Consecutive Sequence (LeetCode 128)
16. Longest Substring with At Most K Distinct Characters (LeetCode 340)
