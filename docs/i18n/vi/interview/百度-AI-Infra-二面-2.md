---
title: "Baidu AI Infra vòng hai"
description: "Câu hỏi phỏng vấn AI Infra tại Baidu vòng hai thật, gồm tối ưu huấn luyện"
tags: ["Tối ưu huấn luyện"]
sourceHash: "a4926cfbaf952800"
---

## Lý thuyết - Transformer và cấu trúc mô hình

1. Mô tả đầy đủ luồng tính Transformer từ token đầu vào tới logits đầu ra.
2. Vì sao lớp FFN tăng chiều rồi giảm chiều? Thiết kế này ảnh hưởng khả năng biểu đạt thế nào?
3. MHA, MQA, GQA khác nhau thế nào về chiếm KV Cache và hiệu quả tính lúc suy luận?
4. Lúc suy luận vì sao KV Cache chỉ cache K và V, không cache Q?
5. Nguyên lý cốt lõi của RoPE là gì? Trong ngữ cảnh dài có hạn chế nào?

## Lý thuyết - Fine-tune và alignment

6. Trong Instruction Tuning với dữ liệu hội thoại nhiều vòng, Loss Mask nên thiết kế thế nào?
7. Sau SFT, mô hình mạnh hơn trên task cụ thể nhưng yếu hơn khả năng tổng quát — xử lý thế nào?
8. Vì sao phân rã hạng thấp của LoRA xấp xỉ được fine-tune full tham số?
9. Rank LoRA đặt không hợp thì mô hình thể hiện hiện tượng gì?
10. Sau train DPO, output mô hình dài rõ — trong hệ thống thực tế ứng phó thế nào?
11. Sau alignment, mô hình quá thận trọng, thường từ chối trả lời — điều chỉnh chiến lược train thế nào?
12. Hiện tượng "máy đọc lại" / lặp của mô hình lớn thường do yếu tố nào?

## Lý thuyết - Hệ RAG

13. Quy trình làm sạch và dựng dữ liệu knowledge base thế nào? Chất lượng dữ liệu không đều ảnh hưởng RAG ra sao?
14. Chiến lược cắt document thiết kế thế nào? Chunk Size và Overlap ảnh hưởng chất lượng recall và sinh thế nào?
15. Câu hỏi thật sự có trong knowledge base nhưng hệ không recall đúng document — lần lỗi thế nào?
16. Đã retrieve đúng document nhưng câu trả lời sinh ra vẫn sai — định vị vấn đề thế nào?
17. Kết quả recall thường gần nghĩa nhưng không liên quan sự thật — tối ưu module retrieval thế nào?
18. Khi một câu hỏi cần nhiều document mới trả lời được, hệ RAG xử lý thế nào?
19. Làm sao phân biệt lỗi RAG nằm ở retrieval hay generation?

## Đào sâu dự án

20. Hỏi sâu quanh nội dung dự án.
