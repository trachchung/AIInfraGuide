---
title: "Huixi Intelligence AI Infra thực tập vòng 1–3"
description: "Câu hỏi phỏng vấn Huixi Intelligence AI Infra thực tập vòng 1–3 thật, gồm tối ưu toán tử và các hướng liên quan"
tags: ["Tối ưu toán tử"]
sourceHash: "5295c2296f11847f"
---

### Vòng một

1. Giới thiệu công ty và trao đổi tình hình cơ bản.

### Vòng hai

#### Tối ưu toán tử

2. Bank conflict là gì? Nguyên nhân? Có những cách xử lý nào?
3. Khi viết toán tử, tận dụng cache tối đa thế nào? Chia khối dữ liệu theo dung lượng L1, L2 ra sao?
4. Warp divergence là gì? Ảnh hưởng tới hiệu năng thế nào?
5. Giá trị lớn nhất của blockDim.x và gridDim.x lần lượt là bao nhiêu?
6. Viết tay toán tử nhân ma trận CUDA (bản naive), rồi nêu hướng tối ưu tiếp theo và cách xác định kích thước khối tốt nhất.

#### Kiến thức nền

7. Mô tả đặc điểm kiến trúc và khác biệt thiết kế của CPU với GPU.
8. Hiểu về cấu trúc ba cấp grid, block, thread trong CUDA.
9. Các thuật toán convolution phổ biến? Ưu nhược điểm từng loại?

#### Kinh nghiệm dự án

10. Khảo sát sâu kinh nghiệm dự án.

### Vòng ba

#### Tối ưu toán tử

11. Shared memory khác hardware cache ở điểm nào?
12. Tăng tốc nhân ma trận thì Tensor Core hay CUDA Core nhanh hơn? Nguyên lý Tensor Core?
13. Viết tay toán tử CUDA Softmax.

#### Kiến thức nền

14. Cấu trúc tổng thể của kiến trúc Transformer?
15. Softmax có những ứng dụng điển hình nào trong deep learning?
