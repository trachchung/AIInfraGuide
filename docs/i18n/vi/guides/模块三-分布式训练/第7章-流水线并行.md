---
title: "Chương 7: Pipeline parallelism (PP)"
description: "Hiểu bản chất vấn đề Bubble của pipeline parallelism, nắm thiết kế và phân tích định lượng của ba chiến lược lập lịch GPipe, 1F1B, Interleaved"
tags: ["Pipeline parallelism", "PP", "GPipe", "1F1B", "Interleaved", "Pipeline Bubble"]
sourceHash: "1f32a6994bb64113"
---

## Giới thiệu chương

Pipeline parallelism (PP, song song pipeline) gán các layer khác nhau của model cho các thiết bị khác nhau, và là cách chính để scale xuyên máy. Chương này tập trung vào thách thức cốt lõi của PP—Pipeline Bubble (bọt khí pipeline)—và cách ba thế hệ chiến lược lập lịch lần lượt giảm Bubble.

---

## Cấu trúc chương

### 1. Ý tưởng cơ bản của pipeline parallelism

- **Cắt theo layer**: chia đều $L$ layer của model cho $P$ Stage; mỗi Stage phụ trách $\frac{L}{P}$ layer
- **Ẩn dụ**: giống dây chuyền nhà máy—mỗi trạm làm một công đoạn, sản phẩm lần lượt qua từng trạm
- **Vấn đề cốt lõi (Bubble)**: trong cách làm ngây thơ, một micro-batch phải đi tuần tự qua mọi Stage → nhiều GPU chờ rồi chạy không
- **Tỷ lệ Bubble** (PP ngây thơ): $\frac{P-1}{P}$ (với $P$ Stage thì có $P-1$ bước thời gian bị lãng phí)

### 2. GPipe: pipeline theo micro-batch

- **Ý tưởng cốt lõi**: cắt một mini-batch thành $M$ micro-batch rồi bơm lần lượt vào pipeline
- **Mẫu thực thi**: xong forward của mọi micro-batch rồi mới làm backward thống nhất (F-F-F-...-B-B-B-...)
- **Tỷ lệ Bubble**: $\frac{P-1}{M+P-1}$; $M$ càng lớn Bubble càng nhỏ
- **Nhược điểm**:
  - phải xong forward mới bắt đầu backward → cần giữ activation của mọi micro-batch → chiếm nhiều VRAM
  - $M$ quá lớn thì Bubble nhỏ nhưng VRAM nổ

### 3. 1F1B (One Forward One Backward)

- **Ý tưởng cốt lõi**: xen kẽ forward và backward để giải phóng activation sớm
- **Mẫu thực thi**: giai đoạn Warmup (lấp đầy pipeline) → Steady (mỗi lần F liền theo một lần B) → Cooldown (xả hết pipeline)
- **Tỷ lệ Bubble**: giống GPipe, $\frac{P-1}{M+P-1}$
- **Lợi thế then chốt**: peak VRAM activation từ $M$ bản xuống $P$ bản (cùng lúc tối đa chỉ còn activation của $P$ micro-batch)
- **Lịch mặc định của Megatron-LM**

### 4. Interleaved 1F1B (Stage ảo)

- **Ý tưởng cốt lõi**: mỗi thiết bị không chỉ phụ trách $\frac{L}{P}$ layer liên tục, mà phụ trách nhiều đoạn layer không liên tục (Stage ảo)
- **Thiết kế**: mỗi thiết bị đảm nhận $v$ Stage ảo (mỗi Stage ảo có $\frac{L}{P \cdot v}$ layer)
- **Tỷ lệ Bubble**: $\frac{P-1}{M \cdot v + P - 1}$, tương đương nhân $M$ lên $v$ lần
- **Cái giá**: tần suất giao tiếp tăng $v$ lần (mọi ranh giới giữa các Stage ảo đều cần truyền activation xuyên thiết bị)
- **Tình huống phù hợp**: khi băng thông giữa máy đủ lớn, đổi thêm giao tiếp lấy Bubble thấp hơn

### 5. Thách thức kỹ thuật của PP

- **Cân bằng tải**: lượng tính của các layer có thể khác nhau (layer Embedding vs layer Transformer); cần phân Stage hợp lý
- **Xử lý Embedding và Loss**: Embedding ở đầu model, Loss ở cuối; thường đặt ở Stage đầu và Stage cuối
- **Truyền activation xuyên máy**: lượng dữ liệu giữa các Stage = $b \times s \times h$ (batch × seq_len × hidden); cần băng thông InfiniBand nâng đỡ
- **Phối hợp với Activation Checkpointing**: PP vốn hợp với recomputation—backward có thể tính lại thay vì lưu activation

### 6. So sánh tỷ lệ Bubble và hình dung

| Chiến lược lập lịch | Tỷ lệ Bubble | Peak VRAM activation | Tần suất giao tiếp |
|---------|------------|------------|---------|
| PP ngây thơ | $(P-1)/P$ | 1 bản | thấp |
| GPipe | $(P-1)/(M+P-1)$ | $M$ bản | thấp |
| 1F1B | $(P-1)/(M+P-1)$ | $P$ bản | thấp |
| Interleaved 1F1B | $(P-1)/(Mv+P-1)$ | $P$ bản | cao ($v$ lần) |

### 7. Bài thực hành

- Vẽ timeline của GPipe và 1F1B với 4 Stage, 8 Micro-batch
- Tính tỷ lệ Bubble dưới các tổ hợp $M$, $P$, $v$ khác nhau
- Phân tích phương án phân layer cho model 64 layer, PP=4

---

## Mục tiêu học

- Giải thích được nguyên nhân Pipeline Bubble và công thức định lượng
- Vẽ được timeline của ba lịch GPipe, 1F1B, Interleaved 1F1B
- Tính được tỷ lệ Bubble và peak VRAM activation dưới cấu hình cho trước
- Phân tích được trade-off "đổi giao tiếp lấy Bubble" của chiến lược Interleaved
