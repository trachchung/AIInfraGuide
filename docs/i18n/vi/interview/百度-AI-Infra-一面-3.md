---
title: "Baidu AI Infra vòng một (3)"
description: "Câu hỏi phỏng vấn AI Infra tại Baidu vòng một thật, gồm tối ưu huấn luyện và tính toán hiệu năng cao"
tags: ["Tối ưu huấn luyện", "Tính toán hiệu năng cao"]
sourceHash: "84dbdce0313934e0"
---

## Reinforcement learning

1. Quan hệ giữa phương pháp Trust Region và PPO
2. PPO thuộc on-policy hay off-policy? Vì sao cần importance sampling?
3. Trong PPO, cơ chế clip khi advantage A dương và âm lần lượt giới hạn cận trên/dưới thế nào?
4. Loss của PPO tính thế nào? Generalized Advantage Estimation (GAE) tính ra sao, và tham số lambda ảnh hưởng variance/bias thế nào?
5. Cách tính loss của GRPO; loss cấp sequence phân về từng token thế nào; khác biệt trung bình cấp sequence và cấp batch
6. Đặc điểm các biến thể GRPO khác (như DAPO, GSPO, GFPO)
7. Khái niệm cơ bản và ứng dụng của Agentic RL

## Huấn luyện phân tán

8. Trong train, trạng thái optimizer, gradient, tham số mô hình mỗi phần chiếm bao nhiêu VRAM
9. So sánh FSDP với DeepSpeed ZeRO Stage 1/2/3

## Đào sâu dự án

10. Giới thiệu dự án và phương án kỹ thuật; kết hợp LLM với kịch bản ứng dụng cụ thể thế nào; cách đánh giá và kết quả chỉ số

## Lập trình tại chỗ

11. Duyệt cây nhị phân theo tầng, đồng thời ghi lại tầng của mỗi nút (biến thể LeetCode 102)
