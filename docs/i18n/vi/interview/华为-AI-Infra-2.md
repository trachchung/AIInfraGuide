---
title: "Huawei AI Infra (2)"
description: "Câu hỏi phỏng vấn AI Infra tại Huawei thật, gồm tối ưu huấn luyện và tính toán hiệu năng cao"
tags: ["Tối ưu huấn luyện", "Tính toán hiệu năng cao"]
sourceHash: "7c0c435b83fa6acd"
---

### Tối ưu huấn luyện

1. Giới thiệu nguyên lý và đặc điểm của optimizer LAMB.
2. Khi tăng batch size, learning rate nên chỉnh thế nào? Nếu mô hình rất lớn nhưng batch size không đổi thì xử lý ra sao?
3. Liệt kê các chiến lược chia song song bạn biết.
4. Giới thiệu tư tưởng lõi của ZeRO và khác biệt giữa các stage.
5. Trong chế độ pipeline parallelism (PP), mỗi card chiếm VRAM và khối lượng tính toán có giống nhau không? Phân bố activation thế nào?
6. Cách kết hợp và kịch bản phù hợp của 3D parallelism là gì?

### Tính toán hiệu năng cao

7. DCU và card tăng tốc AI của Huawei khác nhau thế nào về kiến trúc, hệ sinh thái và thư viện truyền thông?
8. Cơ chế lưu bất đồng bộ trong DDP và DeepSpeed được triển khai ra sao?

### Kinh nghiệm dự án

9. Giới thiệu chi tiết hướng kỹ thuật và chi tiết triển khai dự án train nghìn card.
10. Đã làm những gì quanh async save?

### Kiến thức nền

11. T5 và GPT-2 khác nhau chính ở kiến trúc chỗ nào?
12. Cấu trúc tổng thể của Transformer ra sao?
13. Residual connection đóng vai trò gì trong mạng sâu?
