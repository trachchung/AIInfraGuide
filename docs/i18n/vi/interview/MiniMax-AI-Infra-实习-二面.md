---
title: "MiniMax AI Infra thực tập vòng hai"
description: "Câu hỏi phỏng vấn thực tập MiniMax AI Infra vòng hai thật, gồm tối ưu huấn luyện và tính toán hiệu năng cao"
tags: ["Tối ưu huấn luyện", "Tính toán hiệu năng cao"]
sourceHash: "f02b195fd83324bc"
---
## Hỏi về dự án

1. Chia sẻ dự án thực tập, nhấn mạnh khó khăn kỹ thuật lớn nhất và kết quả tối ưu đạt được

## Lý thuyết

2. Ba stage DeepSpeed ZeRO (Stage-1 / Stage-2 / Stage-3) mỗi stage cắt những state nào? Lượng giao tiếp và tiết kiệm VRAM tiến triển ra sao?
3. Train mô hình lớn bạn đã dùng song song nào? Tư duy và điều kiện phù hợp của data / tensor / pipeline parallelism?
4. Train mô hình ~70B, ước lượng thô VRAM một GPU thế nào?
5. Ý tưởng cốt lõi của LoRA? Vì sao phân rã hạng thấp giảm mạnh số tham số cần cập nhật?
6. Trong LoRA, vì sao khởi tạo ma trận hạ chiều A và nâng chiều B khác nhau? Mục đích là gì?
7. Ngoài ZeRO, bạn còn biết kỹ thuật tối ưu train mô hình lớn nào?

## Bài code

8. Hiện thực thuật toán max trong sliding window
