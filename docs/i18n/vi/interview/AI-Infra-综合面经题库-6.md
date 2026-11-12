---
title: "Ngân hàng câu hỏi phỏng vấn AI Infra tổng hợp (6)"
description: "Câu hỏi phỏng vấn AI Infra thật, gồm tối ưu suy luận, tối ưu huấn luyện và tối ưu toán tử"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử"]
sourceHash: "ec63b83691608603"
---
## Kiến thức CUDA

1. Vai trò và nguyên lý CUDA Graph; toàn bộ luồng kernel launch
2. Chọn blockSize và gridSize thế nào?
3. Default stream là gì? Có vấn đề tiềm ẩn nào?
4. Nguyên nhân bank conflict của Shared Memory và cách xử lý
5. threadfence làm gì?
6. Debug CUDA kernel thế nào?
7. Unified Memory khác Zero-Copy Memory ra sao?
8. Thuật toán sort trên CUDA hiện thực thế nào?
9. Hàm `sin` chạy trên đơn vị phần cứng nào của GPU? Đơn vị đó còn hỗ trợ phép nào?
10. Đặc điểm kiến trúc Volta; khái niệm ITS (Independent Thread Scheduling)
11. Shared Memory tối đa một block trên 3090 là bao nhiêu?
12. PTX khác SASS thế nào?
13. Chỉ số xx TFLOPS ghi trên GPU được tính ra sao?

## Kiến thức C++

14. Cơ chế hàm ảo; bố cục bộ nhớ với đơn kế thừa, đa kế thừa, kế thừa ảo
15. Bốn kiểu cast (`static_cast` / `dynamic_cast` / `const_cast` / `reinterpret_cast`): cách dùng và khác biệt
16. Ba smart pointer (`unique_ptr` / `shared_ptr` / `weak_ptr`): khi nào dùng
17. Khai báo và định nghĩa function template có tách file được không?
18. CRTP hiện thực đa hình tĩnh thế nào?
19. Cơ chế mở rộng của `vector`; `resize` khác `reserve`
20. Cách hiện thực singleton

## Bài CUDA

1. Reduction tổng
2. Softmax
3. Transpose ma trận
4. Average pooling
5. IoU của hai nhóm BBox

## Bài C++

1. NMS (non-maximum suppression)
2. Conv2D
3. Nội suy bilinear
4. LayerNorm
5. Singleton
