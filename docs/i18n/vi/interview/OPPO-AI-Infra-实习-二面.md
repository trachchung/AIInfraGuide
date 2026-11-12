---
title: "OPPO AI Infra thực tập vòng hai"
description: "Câu hỏi phỏng vấn thực tập OPPO AI Infra vòng hai thật, gồm tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "261c1a4ad3a2b0be"
---
## Hỏi về dự án

1. Đào sâu và truy vấn thêm về dự án

## Lý thuyết

2. Khi nào có thể kết luận operator fusion có lợi? Những tình huống điển hình nào nên fusion?
3. Nguyên lý thực thi bất đồng bộ của CUDA Stream là gì? Điều kiện tiên quyết là gì (ví dụ tránh xung đột truy cập bộ nhớ)?
4. Làm sao xác định cấu hình số thread tối ưu cho một CUDA kernel?
5. Trong CUDA, quan hệ phân cấp giữa Thread, Warp, Block, SM, Grid là gì?
6. Nếu Shared Memory với một số Shape đặc biệt làm sai kết quả, nên kiểm tra những gì?
7. NHWC và NCHW mỗi cái có đặc điểm gì? Train và deploy suy luận nên chọn thế nào?
8. Khi nào nên bỏ Shared Memory, ví dụ bank conflict nặng hoặc đi thẳng L2 Cache lại nhanh hơn?
