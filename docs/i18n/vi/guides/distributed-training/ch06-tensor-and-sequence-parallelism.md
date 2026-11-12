---
title: "Chương 6: Tensor parallelism (TP) và sequence parallelism (SP)"
description: "Nắm phương án tensor parallelism của Megatron-LM (Column/Row Parallel Linear), suy ra vị trí chèn giao tiếp, và cách sequence parallelism tối ưu VRAM activation"
tags: ["Tensor parallelism", "Sequence parallelism", "Megatron-LM", "TP", "SP", "NVLink"]
sourceHash: "0182a70c58739e00"
---

## Giới thiệu chương

Khi parameters hoặc activations của chỉ một layer Transformer đã vượt VRAM một GPU, cần cắt phép nhân ma trận **trong layer** ra nhiều GPU—đó là tensor parallelism (TP, song song tensor). Sequence parallelism (SP, song song chuỗi) tiếp tục xử lý phần activation dư thừa ở vùng mà TP không phủ.

---

## Cấu trúc chương

### 1. Vì sao cần tensor parallelism

- **Giới hạn của data parallelism**: DP/FSDP chỉ cắt bản sao dư thừa của trạng thái train; nếu bản thân forward của một layer đã vượt compute/VRAM một GPU (ví dụ activation trung gian của FFN), DP không giúp được
- **Vị trí của TP**: tách một phép nhân $Y = XA$ ra nhiều GPU tính song song; mỗi GPU chỉ cần giữ $\frac{1}{N}$ weights và phần activation tương ứng
- **Ràng buộc băng thông**: TP cần giao tiếp ở mọi layer, rất nhạy latency → thường giới hạn trong một máy nối NVLink (≤ 8 GPU)

### 2. Column Parallel Linear (cắt theo cột)

- **Cách cắt**: weight $A$ cắt theo cột thành $[A_1, A_2, ..., A_N]$; mỗi GPU giữ $A_i$
- **Quá trình tính**: $Y_i = X \cdot A_i$ (mỗi GPU nhân input đầy đủ $X$ với shard cột của mình)
- **Output**: mỗi GPU nhận một shard cột khác nhau $Y_i$ của $Y$
- **Giao tiếp**: input $X$ cần Broadcast/sao chép tới từng GPU (hoặc mỗi GPU vốn đã có input đầy đủ)
- **Vị trí dùng**: linear layer đầu của FFN, projection QKV của Attention

### 3. Row Parallel Linear (cắt theo hàng)

- **Cách cắt**: weight $A$ cắt theo hàng; mỗi GPU giữ $A_i$ (shard theo chiều hàng)
- **Yêu cầu input**: input $X$ cũng phải cắt theo cột thành $X_i$ (khớp với output của Column Parallel)
- **Quá trình tính**: $Y_i = X_i \cdot A_i$ (mỗi GPU tính kết quả một phần)
- **Giao tiếp**: các $Y_i$ trên từng GPU được **AllReduce** (cộng) để ra output đầy đủ $Y$
- **Vị trí dùng**: linear layer thứ hai của FFN, output projection của Attention

### 4. Phương án cắt TP cho Transformer Block

- **Layer Attention**: QKV theo Column Parallel (multi-head vốn cắt được tự nhiên; mỗi GPU phụ trách một phần Head); output projection theo Row Parallel
- **Layer FFN**: linear đầu Column Parallel (cắt chiều trung gian); linear thứ hai Row Parallel (ghép output)
- **Điểm chèn giao tiếp**: mỗi Transformer Block cần 2 AllReduce trong forward (output Attention + output FFN)
- **Backward**: AllReduce ở forward tương ứng AllReduce ở backward; lượng giao tiếp đối xứng
- **Lượng giao tiếp mỗi layer**: forward $2 \times 2bsh$ (2 AllReduce, mỗi lần truyền $2bsh$ dữ liệu)

### 5. Sequence parallelism (SP)

- **Vấn đề**: LayerNorm và Dropout không tham gia cắt TP; input/output của chúng trên mỗi GPU là đầy đủ → activation dư thừa
- **Cách giải**: ở vùng non-TP (LayerNorm, Dropout), cắt activation theo chiều sequence
- **Thay đổi giao tiếp**: AllReduce ở vùng TP trở thành ReduceScatter (khi vào vùng non-TP) + AllGather (khi quay lại vùng TP)
- **Lợi ích**: VRAM activation vùng non-TP từ đầy đủ $bsh$ xuống $\frac{bsh}{N}$; tổng VRAM activation gần scale tuyến tính
- **Thay đổi lượng giao tiếp**: tổng lượng giao tiếp không đổi (AllReduce = ReduceScatter + AllGather), nhưng VRAM activation giảm mạnh

### 6. Cắt TP dưới GQA/MQA

- **Vấn đề**: số KV Head của GQA có thể < bậc TP (ví dụ KV Head = 4, TP = 8)
- **Chiến lược xử lý**:
  - nhân bản KV Head (mỗi TP rank giữ một bản đầy đủ)
  - một số rank chia sẻ KV Head (chiến lược nhóm)
- **Phân tích ảnh hưởng**: nhân bản KV Head làm việc cắt parameters của TP không còn đều, nhưng tính toán vẫn song song được

### 7. Bài thực hành

- Vẽ sơ đồ cắt một Transformer Block dưới TP=4 (ghi rõ mỗi bước giao tiếp)
- Đọc source `ColumnParallelLinear` và `RowParallelLinear` của Megatron-LM

---

## Mục tiêu học

- Suy ra được shape input/output và vị trí giao tiếp của Column Parallel và Row Parallel
- Vẽ được sơ đồ cắt đầy đủ của Transformer Block (Attention + FFN) dưới TP
- Giải thích được cách sequence parallelism tách AllReduce thành ReduceScatter + AllGather để tiết kiệm VRAM activation
- Phân tích được chiến lược xử lý khi KV Head không đủ lúc cắt TP trên model GQA
