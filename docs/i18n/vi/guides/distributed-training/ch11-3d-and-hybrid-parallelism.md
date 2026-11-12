---
title: "Chương 11: Song song 3D và chiến lược song song hỗn hợp"
description: "Ghép mọi chiều song song lại—thiết kế topology TP×PP×DP×EP×CP, ánh xạ miền giao tiếp, sắp xếp rank, và cách chọn tổ hợp tối ưu cho một cụm cho trước"
tags: ["Song song 3D", "Song song hỗn hợp", "Thiết kế topology", "Miền giao tiếp", "Megatron-LM"]
sourceHash: "3e959e797e257fe3"
---

## Giới thiệu chương

Các chương trước đã giảng kỹ từng chiến lược song song riêng. Nhưng huấn luyện mô hình lớn thực tế hầu như không "chỉ dùng một loại"—huấn luyện quy mô nghìn GPU thường là song song 3D dạng TP + PP + DP; thêm MoE thành 4D; thêm chuỗi dài thành 5D. Chương này là "xưởng lắp ráp" của chiến lược song song: làm rõ nhiều chiều kết hợp trực giao thế nào, miền giao tiếp ánh xạ lên phần cứng ra sao, rank được sắp xếp thế nào, và cuối cùng thiết kế được cấu hình song song tối ưu cho một cụm cho trước.

---

## Cấu trúc chương

### 1. Vì sao cần song song hỗn hợp

- Trần của một chiến lược đơn: DP không tiết kiệm tham số, TP không ra khỏi máy, PP có Bubble
- Tính trực giao của các chiều: chiến lược khác nhau chia thứ khác nhau, nên có thể chồng lên nhau
- Ràng buộc `world_size = TP × PP × DP` (× EP × CP)

### 2. Nguyên tắc chia miền giao tiếp (trọng tâm chương)

- **Ánh xạ theo tần suất giao tiếp / nhu cầu băng thông lên phần cứng**:
  - TP: giao tiếp mỗi lớp, nhạy độ trễ nhất → NVLink trong máy (trong 8 GPU)
  - CP/SP: giao tiếp trong Attention → thường cùng miền với TP hoặc trong máy
  - PP: truyền activation ở biên mỗi micro-batch → có thể xuyên máy qua InfiniBand
  - DP: đồng bộ gradient một lần mỗi bước → xuyên nút, đòi hỏi băng thông thấp nhất
- Lượng giao tiếp từ lớn đến nhỏ: TP > CP > PP > DP; gán băng thông phần cứng từ cao xuống thấp cho khớp

### 3. Sắp xếp rank và dựng nhóm giao tiếp

- Ánh xạ rank sang tọa độ nhiều chiều (tp_rank, pp_rank, dp_rank)
- Ảnh hưởng của thứ tự chiều (chiều nào "sát" các GPU kề nhau về mặt vật lý)
- Phân tích logic nhóm của Megatron-LM `initialize_model_parallel()`
- Tạo NCCL communicator cho từng nhóm giao tiếp

### 4. Thực chiến thiết kế topology

- Ví dụ: cụm 64 GPU (8 nút × 8 GPU)
  - Phương án A: TP=8, PP=4, DP=2
  - Phương án B: TP=4, PP=8, DP=2
  - Cách chọn: nhìn số lớp mô hình, kích thước một lớp, băng thông mạng
- Ví dụ: tư duy cấu hình khi huấn luyện mô hình hàng chục / hàng trăm tỷ tham số trên cụm nghìn GPU
- Sắp xếp nhiều chiều sau khi thêm EP/CP

### 5. Ước lượng kết hợp bộ nhớ và thông lượng

- Cho một tổ hợp (TP, PP, DP), ước lượng bộ nhớ mỗi GPU
- Ước lượng thông lượng đầu-cuối: thời gian tính + thời gian giao tiếp từng chiều + PP Bubble
- Phương pháp chỉnh tham số: trước hết thỏa "nhét được", rồi tối ưu "chạy nhanh"

### 6. Thực hành

- Với mô hình + cụm cho trước, tính tay bộ nhớ và lượng giao tiếp của 3 phương án song song rồi chọn cái tốt nhất
- Đọc `parallel_state.py` của Megatron-LM và vẽ sơ đồ nhóm rank

---

## Mục tiêu học tập

- Giải thích được vì sao nhiều chiều song song có thể kết hợp trực giao, và viết được ràng buộc world_size
- Ánh xạ đúng TP/CP/PP/DP lên NVLink/IB/Ethernet theo tần suất giao tiếp
- Đọc được logic sắp xếp rank của Megatron và vẽ sơ đồ nhóm giao tiếp
- Thiết kế và so sánh được nhiều phương án song song 3D / nhiều chiều cho quy mô mô hình và topology cụm cho trước
