---
title: "NetEase AI Infra tuyển mới tốt nghiệp"
description: "Câu hỏi phỏng vấn NetEase AI Infra tuyển mới tốt nghiệp thật, gồm tối ưu suy luận, tối ưu toán tử và các hướng liên quan"
tags: ["Tối ưu suy luận", "Tối ưu toán tử"]
sourceHash: "7d537d1bdd6d0796"
---

### Kinh nghiệm dự án

1. Mô tả chi tiết các module chính và đóng góp kỹ thuật cốt lõi của dự án liên quan AI trên CV

### Tối ưu suy luận

2. Khi thiết kế dịch vụ suy luận mô hình throughput cao, latency thấp, ở tầng kiến trúc và tầng kỹ thuật cần chú ý những vấn đề nào

### C++ cơ bản

3. Đặc điểm và khác biệt của stack, heap và vùng static/global trong bộ nhớ chương trình
4. `new/delete` khác `malloc/free` chủ yếu ở đâu; vì sao C++ khuyến khích dùng cái trước
5. Deep copy và shallow copy là gì; tình huống nào bắt buộc dùng deep copy
6. Mục đích thiết kế, tình huống dùng và khác biệt của `std::unique_ptr`, `std::shared_ptr`, `std::weak_ptr`
7. vtable triển khai đa hình runtime thế nào; virtual function và pure virtual khác gì về ngữ nghĩa và cách dùng
8. Memory alignment và quy tắc căn chỉnh struct của compiler; tính `sizeof` của một struct cho sẵn
9. Rvalue reference và move semantics trong C++11; dùng chúng để chuyển dữ liệu hiệu quả thế nào

### Lập trình CUDA

10. Khái niệm cơ bản của mô hình SIMT; quan hệ phân cấp thread, block, grid
11. Biến cục bộ của thread trong CUDA kernel nằm ở đâu và liên quan register allocation thế nào
12. Dùng shared memory để giảm truy cập lặp lại vào global memory (ví dụ GEMM chia khối)
13. Warp shuffle: khái niệm và lợi thế khi trao đổi dữ liệu / reduce trong warp
14. Nguyên lý và lợi ích hiệu năng của lệnh load/store vector hóa (như float4, int4) để coalesced access
15. Nguyên nhân bank conflict của shared memory và ví dụ
16. Tránh bank conflict bằng padding hoặc chỉnh memory layout

### Kiến thức nền

17. Nguyên lý cache CPU, gồm temporal locality, spatial locality và các chiến lược thay thế phổ biến

### Bài lập trình

18. Trong luồng nhiều số nguyên, tìm realtime top-K phần tử xuất hiện nhiều nhất (hash table + min-heap), phân tích độ phức tạp thời gian
