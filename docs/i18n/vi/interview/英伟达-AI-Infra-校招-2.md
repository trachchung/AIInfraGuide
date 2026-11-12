---
title: "NVIDIA AI Infra tuyển mới tốt nghiệp (2)"
description: "Câu hỏi phỏng vấn tuyển mới tốt nghiệp AI Infra tại NVIDIA thật, gồm tối ưu toán tử"
tags: ["Tối ưu toán tử"]
sourceHash: "7290ca82917fde4e"
---

### Kiến thức cơ bản

1. Ý nghĩa và nguyên lý SIMT (Single Instruction, Multiple Threads).
2. Occupancy chịu ảnh hưởng yếu tố nào? Điều chỉnh thế nào?
3. Độ hạt của bank conflict là bao nhiêu?
4. Chọn kích thước tile GEMM bị ràng buộc bởi yếu tố nào?
5. Vì sao đọc/ghi global memory bằng float4 cải thiện hiệu năng?
6. Một Block có thể bị schedule sang các SM khác nhau không?
7. Dung lượng Cache của các dòng GPU mainstream?
8. Warp Divergence ảnh hưởng hiệu năng cụ thể thế nào.
9. Cách hiện thực instruction-level parallelism (ILP) trên NVIDIA GPU.

### Bài lập trình

10. Hiện thực chuyển vị ma trận CUDA.
11. Hiện thực tích ngoài vector.
