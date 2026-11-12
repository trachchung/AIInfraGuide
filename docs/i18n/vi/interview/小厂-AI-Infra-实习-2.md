---
title: "Công ty nhỏ AI Infra thực tập (2)"
description: "Câu hỏi phỏng vấn thực tập AI Infra tại công ty nhỏ thật, gồm tối ưu suy luận, huấn luyện và toán tử"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử"]
sourceHash: "bdf9922858adb21f"
---

### Kiến thức nền

1. Pretrain và SFT khác nhau thế nào về thiết kế loss và xây dựng dataset?
2. Trong kiến trúc Transformer, tensor parallelism (TP) của Megatron chia ma trận tham số ra sao? Ở MLP, linear layer thứ nhất và thứ hai lần lượt dùng row shard hay column shard? Mỗi cách tương ứng primitive giao tiếp nào?
3. Nêu ngắn các điểm kỹ thuật trong bài DeepSeek khiến bạn ấn tượng (ví dụ phương án train FP8).

### Tối ưu huấn luyện

4. Giới thiệu pipeline parallelism trong pretrain; giải thích thiết kế 1F1B và DualPipe.
5. Trình bày đầy đủ quy trình RL cho mô hình lớn. Có những vai trò mô hình nào? Khác biệt cốt lõi giữa PPO và GRPO là gì?
6. Trong train RL, giai đoạn Rollout chiếm khoảng bao nhiêu thời gian? MFU của Policy model khoảng bao nhiêu? Cho công thức MFU và ý nghĩa của công thức 6Nd.
7. Giai đoạn Rollout trong RL có những tối ưu phổ biến nào (ví dụ Rollout quantization, async Rollout)?
8. Trong pipeline train RL, làm sao đồng bộ trọng số pretrain sang inference engine?

### Tối ưu suy luận

9. Giới thiệu cơ chế Continuous Batching trong vLLM hoặc SGLang.
