---
title: "Chương 8: Mixed precision và tối ưu bộ nhớ GPU"
description: "Quy trình huấn luyện mixed precision FP16/BF16/FP8, gradient accumulation, Activation Checkpointing và phân tích chi phí của từng kỹ thuật"
tags: ["Mixed precision", "BF16", "FP8", "Activation Checkpointing", "Gradient accumulation", "Tối ưu bộ nhớ GPU"]
sourceHash: "1f11d3e160eedb8c"
---

## Giới thiệu chương

Song song hóa giải quyết việc "chia ra nhiều GPU", nhưng trong từng card vẫn còn nhiều bộ nhớ GPU để tiết kiệm. Chương này tập trung vào ba kỹ thuật tối ưu bộ nhớ nằm ngoài các chiều song song, và gần như mọi bài huấn luyện đều dùng: **mixed precision** (độ rộng bit thấp hơn để tính nhanh hơn và tốn ít bộ nhớ hơn), **gradient accumulation** (bộ nhớ nhỏ mô phỏng batch lớn), và **Activation Checkpointing** (đổi tính toán lấy bộ nhớ). Đây là nền tảng để tận dụng phần cứng đến mức tối đa.


---

## Cấu trúc chương

### 1. Huấn luyện mixed precision (trọng tâm chương)

- **So sánh kiểu dữ liệu**:

  | Kiểu | Số bit | Số bit mũ | Số bit phần định trị | Phạm vi số | Tình huống dùng |
  |------|------|--------|--------|---------|---------|
  | FP32 | 32 | 8 | 23 | Lớn | Master weight, optimizer |
  | FP16 | 16 | 5 | 10 | Nhỏ, dễ tràn | Cách làm cũ |
  | BF16 | 16 | 8 | 7 | Giống FP32 | Lựa chọn hàng đầu khi huấn luyện mô hình lớn |
  | FP8 (E4M3) | 8 | 4 | 3 | Rất nhỏ | Tính toán forward trên H100+ |

- **Vì sao mixed precision vừa tiết kiệm bộ nhớ vừa tăng tốc**: độ rộng bit thấp → bộ nhớ giảm khoảng một nửa + thông lượng Tensor Core tăng gấp đôi
- **Quy trình mixed precision BF16**: forward/backward tính bằng BF16, cập nhật tham số bằng FP32 Master Weight (nối với trạng thái optimizer ở Chương 3)
- **FP16 vs BF16**: FP16 phạm vi nhỏ dễ tràn; BF16 cùng phạm vi với FP32 nhưng độ chính xác thấp hơn—vì sao mô hình lớn thường chọn BF16
- **Loss Scaling**: kỹ thuật scale gradient bắt buộc với FP16 (tĩnh / động); BF16 vì phạm vi lớn thường không cần
- **Huấn luyện FP8**: H100 Transformer Engine, forward dùng FP8 + per-tensor dynamic scaling, gradient backward dùng BF16; tình huống phù hợp và rủi ro về độ chính xác

### 2. Gradient Accumulation

- **Mục đích**: mô phỏng Effective Batch Size lớn hơn trong giới hạn bộ nhớ
- **Cơ chế**: chạy liên tiếp $K$ bước forward+backward để cộng dồn gradient, mỗi $K$ bước mới cập nhật tham số một lần
- **Effective Batch Size** = batch mỗi GPU × world_size × accumulation_steps
- **Phối hợp với PP**: micro-batch của PP vốn đã cho hiệu ứng tương tự; gradient accumulation phóng đại thêm
- **Phối hợp với DDP**: trong lúc tích lũy dùng `no_sync()` để bỏ AllReduce ở các bước giữa, tiết kiệm giao tiếp
- **Lưu ý**: Loss cần chia cho $K$ (hoặc tương đương chia gradient cho $K$)

### 3. Activation Checkpointing (tính lại activation)

- **Vấn đề**: activation lưu ở forward tăng tuyến tính theo số tầng và batch size; trong huấn luyện mô hình lớn, bộ nhớ activation có thể vượt bộ nhớ tham số
- **Ý tưởng cốt lõi**: chỉ lưu activation của một phần tầng (checkpoint); các tầng còn lại tính lại forward khi backward
- **Full Checkpointing**: chỉ lưu input của mỗi Transformer Block, backward thì tính lại cả Block
  - Tiết kiệm bộ nhớ: bộ nhớ activation từ $O(L)$ xuống $O(\sqrt{L})$ (mỗi $\sqrt{L}$ tầng lưu một checkpoint)
  - Chi phí tính toán: tăng khoảng 33% lượng tính forward
- **Selective Checkpointing**: chỉ tính lại các phép tính nhỏ nhưng activation lớn (ví dụ output softmax của Attention); giữ activation của các phép tính nặng
- **Phối hợp chiến lược song song**: PP vốn chỉ cần lưu activation của Stage mình; dưới TP, SP đã giảm redundancy của activation (nối với Chương 6/7)

### 4. Toàn cảnh tối ưu bộ nhớ và mối liên hệ

Phần này xếp tất cả cách "tiết kiệm bộ nhớ" trong series vào một khung nhìn thống nhất:

- **Nhóm chia cắt** (nhiều GPU): ZeRO/FSDP chia state, TP chia weight, PP chia tầng, SP/CP chia activation → xem Chương 5–9
- **Nhóm hạ độ chính xác**: mixed precision (mục 1 chương này)
- **Nhóm đổi thời gian lấy không gian**: gradient accumulation, Activation Checkpointing (mục 2–3 chương này)
- **Nhóm offload**: ZeRO-Offload/Infinity (Chương 5)
- Một bảng đối chiếu "bốn thành phần bộ nhớ × đối tượng từng kỹ thuật tối ưu", nối lại sổ bộ nhớ ở Chương 1

---

## Mục tiêu học tập

- Giải thích được toàn bộ quy trình huấn luyện mixed precision BF16 (gồm Master Weight) và vì sao mô hình lớn thích BF16
- Nói được Loss Scaling giải quyết vấn đề gì và vì sao BF16 thường không cần
- Dùng gradient accumulation tính Effective Batch Size và giải thích vai trò của `no_sync()`
- Phân tích được mức tiết kiệm bộ nhớ của Activation Checkpointing ($O(\sqrt{L})$) và chi phí tính toán (~33%)
- Vẽ được bảng đối chiếu "bốn thành phần bộ nhớ × các kỹ thuật tối ưu"
