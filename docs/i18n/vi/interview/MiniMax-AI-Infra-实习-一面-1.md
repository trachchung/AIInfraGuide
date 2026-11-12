---
title: "MiniMax AI Infra thực tập vòng một (1)"
description: "Câu hỏi phỏng vấn thực tập MiniMax AI Infra vòng một thật, gồm tối ưu suy luận và tối ưu huấn luyện"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện"]
sourceHash: "b1a06f5f3d484055"
---
## Đào sâu dự án

1. Giới thiệu dự án thực tập và công việc nghiên cứu của bạn

## Lý thuyết

2. Mục tiêu train của SFT và RLHF khác nhau căn bản thế nào? Vì sao hầu hết mô hình sau SFT vẫn cần RLHF?
3. Trong MoE, module routing quyết định gán token cho expert nào thế nào? Vì sao load giữa các expert lệch nặng?
4. Khi một số expert dùng ít rõ rệt, cải thiện routing thế nào để tăng utilization?
5. Giới thiệu kiến trúc LLM bạn quen, so sánh về tính Attention, phương pháp train và hiệu quả infer
6. PPO trong RLHF tối ưu mục tiêu cốt lõi nào? Viết hàm mục tiêu và giải thích từng hạng mục
7. Khi số tham số tăng, vì sao MoE vẫn giữ hiệu quả train tương đối cao?

## Lập trình tại chỗ

8. Cho K mảng đã sắp, tìm trung vị sau khi gộp
