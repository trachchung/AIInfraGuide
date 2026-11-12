---
title: "Chương 5: Speculative Decoding"
description: "Hiểu Draft + Verify và sampling bảo toàn phân phối, so sánh Draft độc lập, N-gram/Suffix, Medusa/EAGLE/MTP, rồi nắm biên lợi ích và thí nghiệm đối chiếu trên vLLM"
tags: ["Speculative Decoding", "Giải mã đầu cơ", "Medusa", "EAGLE", "MTP", "N-gram"]
sourceHash: "cbf066a0f2531f27"
---

## Giới thiệu chương

Decode chậm vì mô hình lớn mỗi lần đọc hết weight chỉ viết ra một token. Speculative decoding (giải mã đầu cơ) đổi cách chia việc: để Draft rẻ viết trước một đoạn nháp, rồi Target chạy một forward để nghiệm thu hàng loạt. Đoán đúng thì nhận; đoán sai thì Target viết lại. Chỉ cần quy tắc nghiệm thu và sửa đúng, đầu ra cuối cùng có cùng phân phối với việc Target tự sinh từng token một.

Chương này đi theo bốn bước "vì sao đúng, đoán thế nào, khi nào đáng, đo ra sao":

- **Nguyên lý**: một vòng Draft + Verify, xác suất chấp nhận và phân phối phần dư, chứng minh không mất mát, cùng cách KV Cache hoàn tác các token nháp bị từ chối.
- **Nguồn nháp**: mô hình nhỏ độc lập, Draft song song, N-gram và Suffix Decoding; Medusa, dòng EAGLE và các phương án tái sử dụng đặc trưng của Target; cùng module MTP đi kèm mô hình.
- **Biên lợi ích**: lập sổ hiệu năng từ "mỗi vòng commit bao nhiêu token / mỗi vòng mất bao lâu", rồi phân tích ảnh hưởng của speculative length, concurrency, ngữ cảnh dài, lượng tử hóa và streaming.
- **Thực hành**: trên vLLM, giữ cùng một Target và lần lượt so sánh Decode thường, N-gram, Draft độc lập và EAGLE-3, thu thập chỉ số chấp nhận, nội dung đầu ra cùng latency/throughput.

> ⏰ **Độ cập nhật:** Phần nguyên lý không phụ thuộc framework; tham số, tên phương pháp và chỉ số liên quan vLLM dựa trên v0.32.0 (phát hành 2026-10-18). Các mục cấu hình speculative decoding đổi khá nhiều ở vài phiên bản gần đây; khi đổi phiên bản hãy đối chiếu tài liệu chính thức.

## Các bài trong chương

- [**5.1 Nguyên lý cốt lõi của speculative decoding: đoán một đoạn, rồi để mô hình lớn nghiệm thu**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/): luồng Draft + Verify, chứng minh bảo toàn phân phối, Draft one-hot, từ chối đầu tiên và hoàn tác KV
- [**5.2 Draft model và phương án không dùng model: đoán rẻ mà trúng hơn**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/): mô hình nhỏ độc lập và PARD, tương thích tokenizer, N-gram/Suffix, cùng cách đo thống nhất cho acceptance rate, độ dài chấp nhận trung bình và độ phủ ứng viên
- [**5.3 Medusa, EAGLE và MTP: tái sử dụng thông tin mô hình lớn để sinh nháp tốt hơn**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/): Medusa multi-head, Tree Attention, EAGLE-1/2/3, DeepSeek-V3 MTP, DFlash và Block Verification
- [**5.4 Biên lợi ích của speculative decoding: acceptance cao vẫn có thể chậm hơn**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/): độ dài commit và thời gian mỗi vòng, lợi ích giảm dần theo speculative length, Dynamic SD, concurrency, lượng tử hóa và độ trễ streaming
- [**5.5 Thực hành speculative decoding trên vLLM: cấu hình, đo acceptance và hiệu năng cùng lúc**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/): đối chiếu baseline, N-gram, Draft, EAGLE-3, chỉ số chấp nhận theo request và benchmark
