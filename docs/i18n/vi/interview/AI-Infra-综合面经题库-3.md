---
title: "Ngân hàng câu hỏi phỏng vấn AI Infra tổng hợp (3)"
description: "Câu hỏi phỏng vấn AI Infra thật, gồm tối ưu suy luận, tối ưu huấn luyện, tối ưu toán tử và tính toán hiệu năng cao"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử", "Tính toán hiệu năng cao"]
sourceHash: "0eb8b78f11362561"
---
## Kiến thức cơ bản

1. Khi truy cập CUDA Global Memory và Shared Memory lần lượt cần chú ý gì?
2. Biết tổng token cần để train, ước lượng thời gian hoàn thành training thế nào?
3. Prefill và Decode mỗi giai đoạn có những kỹ thuật tối ưu phổ biến nào?
4. Two-batch overlap nghĩa là gì? Trong trường hợp nào nó lại thành tối ưu âm?
5. Megatron-LM tối ưu giao tiếp như thế nào?
6. Tách PD nhiều máy mang thêm chi phí truyền KV Cache, vì sao vẫn cần tách PD?
7. Vì sao Muon và AdamW không thể dùng lẫn giữa Pretrain và Post-train?
8. Bạn nhìn nhận thế nào về tách PD xuyên SM và tách AF?
9. DeepSeek-V3 có những điểm tối ưu quan trọng nào?
10. DeepSeek-DSA, NSA và MoBA khác nhau ra sao?
11. NCCL gồm những primitive giao tiếp nào? Một lần cập nhật tham số All-Reduce cần bao nhiêu lần giao tiếp?
12. Với payload nhỏ, dùng NVSHMEM để mỗi GPU đọc dữ liệu GPU khác rồi Reduce local, so với Ring All-Reduce có lợi gì?
13. Train chuỗi rất dài nên thiết kế chiến lược song song thế nào?
14. Port toán tử từ Ampere sang Hopper cần nâng cấp những gì?
