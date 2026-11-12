---
title: "Chương 5: Series tối ưu VRAM ZeRO"
description: "Tháo từng lớp chiến lược cắt và chi phí giao tiếp của ZeRO-1/2/3, nắm cơ chế offload bộ nhớ dị thể của ZeRO-Offload/Infinity"
tags: ["ZeRO", "DeepSpeed", "Tối ưu VRAM", "Offload", "Phân tích giao tiếp"]
sourceHash: "a041cbca1820d56b"
---

## Giới thiệu chương

ZeRO (Zero Redundancy Optimizer) là kỹ thuật cốt lõi của DeepSpeed. Chương này xuất phát từ "sự dư thừa trong trạng thái train nằm ở đâu", lần lượt giải thích cách cắt (shard) để xóa dư thừa, rồi phân tích chi phí giao tiếp và tình huống phù hợp của từng giai đoạn.

---

## Cấu trúc chương

### 1. Insight cốt lõi của ZeRO

- **Phân tích dư thừa**: trong DDP mỗi GPU đều dư thừa lưu toàn bộ optimizer state ($12\Psi$), gradients ($2\Psi$), parameters ($2\Psi$); trên cụm $N$ GPU thì $(N-1)/N$ dung lượng đó là lãng phí
- **Ý tưởng thiết kế**: mô hình "cắt–ghép"—bình thường mỗi GPU chỉ giữ $\frac{1}{N}$, khi cần thì lấy đủ dữ liệu qua giao tiếp rồi bỏ ngay sau khi dùng
- **Ba giai đoạn tiến dần**: từ phần dễ cắt nhất (optimizer state) đến khó cắt nhất (parameters), lần lượt xóa dư thừa

### 2. ZeRO-1: cắt optimizer state

- **Nội dung cắt**: bản sao parameters FP32 của Adam + first-order momentum + second-order momentum (tổng $12\Psi$ Bytes)
- **Mẫu giao tiếp**: gradient vẫn cần AllReduce (giống DDP); sau khi cập nhật parameters cần AllGather để đồng bộ parameters
- **Tiết kiệm VRAM**: mỗi GPU từ $16\Psi$ xuống $4\Psi + \frac{12\Psi}{N}$
- **Lượng giao tiếp**: giống DDP ($2\Psi$) + thêm AllGather parameters
- **Tình huống phù hợp**: khi optimizer state là phần lớn VRAM (Adam chiếm ~75%); thêm một ít giao tiếp để đổi lấy tiết kiệm VRAM lớn

### 3. ZeRO-2: cắt tiếp gradients

- **Nội dung cắt**: trên nền ZeRO-1, gradients cũng được lưu theo shard
- **Mẫu giao tiếp**: backward dùng ReduceScatter (thay AllReduce); mỗi GPU chỉ giữ gradient đã aggregate của shard mình phụ trách
- **Tiết kiệm VRAM**: mỗi GPU xuống $2\Psi + \frac{14\Psi}{N}$
- **Lượng giao tiếp**: ReduceScatter = $\Psi$ (bằng một nửa $2\Psi$ của AllReduce, nhưng mất AllGather ngầm trong AllReduce)
- **Tình huống phù hợp**: model cỡ trung đến lớn, sau khi cắt optimizer + gradients thì một GPU vẫn chứa được parameters

### 4. ZeRO-3: cắt luôn parameters

- **Nội dung cắt**: parameters, gradients và optimizer state đều được cắt
- **Mẫu giao tiếp**:
  - Forward: AllGather lắp lại parameters của layer hiện tại → tính → giải phóng
  - Backward: AllGather parameters → tính gradient → ReduceScatter gradient → giải phóng parameters
- **Tiết kiệm VRAM**: mỗi GPU xuống $\frac{16\Psi}{N}$ (scale tuyến tính lý tưởng)
- **Lượng giao tiếp**: $3\Psi$ (forward AllGather $\Psi$ + backward AllGather $\Psi$ + ReduceScatter $\Psi$), nhiều hơn DDP 50%
- **Tình huống phù hợp**: bản thân parameters một GPU không chứa nổi, bắt buộc phải shard

### 5. ZeRO-Offload và ZeRO-Infinity

- **ZeRO-Offload**: đẩy optimizer state và tính gradient sang CPU; GPU chỉ làm forward/backward
  - Tình huống phù hợp: train model lớn trên ít GPU (1–4)
  - Cái giá: băng thông truyền CPU–GPU trở thành nút thắt (PCIe 4.0: ~32 GB/s)
- **ZeRO-Infinity**: trên nền Offload, tận dụng thêm lưu trữ NVMe SSD
  - Tình huống phù hợp: model cực lớn (nghìn tỷ parameters) train trên ít GPU
  - Kỹ thuật then chốt: prefetch theo khối (chunked prefetch), chồng tính toán với I/O

### 6. Hướng dẫn chọn ZeRO

| Giai đoạn | Tiết kiệm VRAM | Chi phí giao tiếp | Tình huống phù hợp |
|------|---------|---------|---------|
| ZeRO-1 | optimizer state $\frac{12\Psi}{N}$ | gần với DDP | parameters + gradients một GPU chứa được |
| ZeRO-2 | + gradients | hơi cao hơn DDP | parameters một GPU chứa được |
| ZeRO-3 | toàn bộ | nhiều hơn DDP 50% | parameters cũng không chứa nổi |
| Offload | dùng bộ nhớ CPU | + truyền PCIe | ít GPU, model lớn |

### 7. Quan hệ giữa ZeRO và FSDP

- FSDP `FULL_SHARD` ≈ ZeRO-3, `SHARD_GRAD_OP` ≈ ZeRO-2
- ZeRO là khái niệm ở tầng thuật toán/paper; FSDP là triển khai native của PyTorch; DeepSpeed là triển khai độc lập của Microsoft
- Chọn FSDP hay DeepSpeed: trong hệ sinh thái PyTorch ưu tiên FSDP; cần Offload/Infinity hoặc đã có cấu hình DeepSpeed thì dùng DeepSpeed

---

## Mục tiêu học

- Vẽ được sơ đồ "cắt gì, giao tiếp thế nào" cho từng giai đoạn ZeRO-1/2/3
- Tính được công thức VRAM mỗi GPU và lượng giao tiếp mỗi bước ở từng giai đoạn
- Giải thích được cơ chế offload CPU của ZeRO-Offload và nút thắt hiệu năng
- Chọn được giai đoạn ZeRO phù hợp theo quy mô model và điều kiện phần cứng
