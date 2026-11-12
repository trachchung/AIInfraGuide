---
title: "Tencent AI Infra thực tập vòng một (2)"
description: "Câu hỏi phỏng vấn thực tập Tencent AI Infra vòng một thật, gồm tối ưu suy luận, high-performance computing và các hướng liên quan"
tags: ["Tối ưu suy luận", "High-performance computing"]
sourceHash: "b30efd1fc731f1d3"
---

### Kinh nghiệm dự án

1. Giới thiệu nội dung công việc và kinh nghiệm dự án thời thực tập, cùng lý do nghỉ việc

### Tối ưu suy luận

2. Vì sao kiến trúc MoE kiểm soát được chi phí training và inference dù số tham số rất lớn, điểm khó kỹ thuật thật sự nằm ở đâu
3. Load balancing trong MoE thường dùng những phương pháp nào, vì sao một số mô hình loss vẫn bình thường nhưng expert đã thoái hóa
4. Khác biệt cốt lõi giữa GQA, MQA và MHA chuẩn là gì, vì sao inference trên production quan tâm GQA hơn
5. Nguyên lý RoPE positional encoding là gì, vì sao khi ngoại suy ngữ cảnh dài dễ bị méo
6. Nguyên lý tăng tốc của FlashAttention là gì, bottleneck tối ưu cốt lõi là compute hay truy cập bộ nhớ
