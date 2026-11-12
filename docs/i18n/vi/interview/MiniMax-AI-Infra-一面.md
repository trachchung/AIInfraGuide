---
title: "MiniMax AI Infra vòng một"
description: "Câu hỏi phỏng vấn MiniMax AI Infra vòng một thật, gồm tối ưu huấn luyện"
tags: ["Tối ưu huấn luyện"]
sourceHash: "135eb890ab8d4bb8"
---
## Lý thuyết

1. RAG là gì? Nó cải thiện chất lượng sinh bằng cách nào? Phương án RAG chuẩn thường thiếu gì? Khác pipeline "retrieve rồi generate" cổ điển ở điểm căn bản nào?
2. Quy trình chính của SFT là gì? Dataset train thường được xây và lọc thế nào?
3. Từ góc toán, ý nghĩa cốt lõi của Attention trong Transformer là gì?
4. Bạn có biết kỹ thuật Agent không? Đóng RAG thành Agent mang lại lợi ích gì?
5. Đánh giá hệ thống RAG một cách hệ thống thế nào? Có chỉ số hoặc benchmark chuẩn nào?
6. LoRA hoạt động thế nào? Lúc infer, mô hình fine-tune LoRA còn cần load Adapter riêng không?
7. PPO và DPO khác nhau cốt lõi thế nào khi align LLM? Train bằng DPO cần lưu ý gì?
8. Đã dùng GRPO chưa? Nêu ngắn gọn.
9. Trong Agent hội thoại nhiều vòng, Attention bộc lộ hạn chế gì?
10. Sau SFT, Post-Training thường gồm những phương pháp nào (như RLHF)? Mỗi cái tối ưu mục tiêu gì?
11. Fine-tune Qwen thì chọn chiến lược giai đoạn train thế nào? Loss được thiết kế dựa trên cân nhắc nào?
12. Tính năng gợi ý Prompt tự động được tối ưu ra sao? Đã thử nén Prompt hoặc biểu diễn Embedding để tăng hiệu quả chưa?

## Đào sâu dự án

13. Cơ chế lập lịch gọi tool thiết kế thế nào? Hệ thống có Fallback khi lỗi không?
14. Modular Agent trong dự án làm multi-step planning bằng cách nào?
15. Hệ đánh giá Agent gồm những mặt nào? Chất lượng planning và tần suất hallucination đo thế nào?

## Tình huống mở

16. Luồng suy luận Agent lần lượt gọi 3 tool ngoài, concurrency cao làm latency end-to-end cao. Bạn dùng biện pháp kỹ thuật nào để giảm latency?

## Lập trình tại chỗ

17. Dùng PyTorch viết logic tính loss trong SFT (chú ý căn chỉnh label shift right)
