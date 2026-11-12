---
title: "Chương 4: Data parallelism (DP, DDP, FSDP)"
description: "Hiểu logic tiến hóa, cơ chế cốt lõi và thực hành của ba thế hệ data parallelism, cùng khác biệt bản chất giữa đồng bộ gradient bằng AllReduce và parameter sharding"
tags: ["Data parallelism", "DDP", "FSDP", "AllReduce", "ReduceScatter"]
sourceHash: "fdc6b4e5232dfb8e"
---

## Giới thiệu chương

Data parallelism (song song dữ liệu) là chiến lược song song cơ bản và dùng nhiều nhất. Chương này đi theo mạch "vì sao phải tiến hóa" để làm rõ mỗi thế hệ DP → DDP → FSDP giải quyết gì và phải trả giá gì. Có hai bài chính kèm theo:

- **4.1 Data parallelism chi tiết**: phần nguyên lý—nền tảng toán, logic tiến hóa ba thế hệ, cơ chế Bucket, suy ra lượng giao tiếp Ring AllReduce, sổ kế toán VRAM và cách chọn.
- **4.2 Data parallelism trong PyTorch từ nguyên lý đến thực chiến**: phần thực hành—nền tảng giao tiếp PyTorch, code DDP/FSDP chạy được đầy đủ, khởi động đa máy bằng torchrun và xử lý các lỗi thường gặp.

Nên đọc 4.1 trước để nắm nguyên lý, rồi dùng 4.2 để đưa vào thực tế kỹ thuật.

---

## Cấu trúc chương

### 1. DataParallel (DP) — một process, nhiều thread

- **Cách hoạt động**: một process giữ model; mỗi lần forward thì Scatter model ra nhiều GPU, rồi Gather output về GPU chính
- **Lỗi chí mạng**: giới hạn Python GIL, VRAM/tính toán lệch về GPU chính, giao tiếp đi qua PCIe thay vì NVLink
- **Vị trí**: chỉ dùng để prototype nhanh; không dùng trong production

### 2. DistributedDataParallel (DDP) — lựa chọn chuẩn production

- **Cơ chế cốt lõi**:
  - mỗi GPU một process độc lập, mỗi process giữ bản sao model đầy đủ
  - `init_process_group` tạo process group
  - `DistributedSampler` đảm bảo dữ liệu các process không chồng nhau
  - đồng bộ gradient bằng AllReduce trong backward
- **Cơ chế Bucket và chồng tính toán–giao tiếp**:
  - gradient tham số được nhóm thành Bucket theo thứ tự tính reverse
  - mỗi Bucket sẵn sàng là AllReduce ngay, không chờ hết backward
  - giao tiếp chồng lên tính reverse của các Bucket sau (overlap)
- **Phân tích lượng giao tiếp**: lượng AllReduce mỗi bước = $2\Psi$ (tổng gửi+nhận mỗi GPU dưới Ring AllReduce)
- **Điểm dùng chính**: `DistributedSampler` + `set_epoch()`, chỉ rank 0 log/save, truy cập tham số gốc qua `model.module`
- **Hạn chế**: mỗi GPU giữ model đầy đủ → parameters + gradients + optimizer state phải ≤ VRAM một GPU

### 3. FullyShardedDataParallel (FSDP) — data parallelism tối ưu VRAM

- **Ý tưởng cốt lõi**: không để mỗi GPU dư thừa lưu toàn bộ trạng thái; shard rồi AllGather khi cần tính
- **Luồng làm việc**:
  1. Trạng thái ban đầu: mỗi GPU chỉ giữ $\frac{1}{N}$ parameters / gradients / optimizer state
  2. Forward: AllGather lắp lại parameters đầy đủ → tính → giải phóng shard không thuộc local
  3. Backward: AllGather parameters → tính gradient → ReduceScatter gradient đã shard → giải phóng parameters đầy đủ
  4. Update: mỗi GPU chỉ cập nhật shard mình phụ trách
- **Chọn chiến lược shard**:
  - `FULL_SHARD` (ZeRO-3): shard parameters + gradients + optimizer state; tiết kiệm VRAM nhất
  - `SHARD_GRAD_OP` (ZeRO-2): shard gradients + optimizer state; không shard parameters
  - `HYBRID_SHARD`: full shard trong máy + data parallelism giữa các máy; khuyến nghị cho train đa máy
  - `NO_SHARD`: tương đương DDP; dùng để debug
- **Phân tích lượng giao tiếp**: mỗi bước = AllGather (forward) + AllGather + ReduceScatter (backward) = $3\Psi$, nhiều hơn DDP 50%
- **Cấu hình kỹ thuật**: `auto_wrap_policy` (tự shard theo kích thước hoặc loại module), `MixedPrecision`, lưu/tải bằng DCP
- **FSDP2 ngắn gọn**: shard theo từng parameter, dựa trên DTensor, kết hợp tốt hơn với TP

### 4. Hướng dẫn chọn DDP vs FSDP

- Bảng so sánh: chiếm VRAM, lượng giao tiếp, độ phức tạp code, quy mô phù hợp
- Quy tắc quyết định: một GPU chứa được → DDP (đơn giản, hiệu quả); chứa không nổi → FSDP; model lớn đa máy → FSDP + `HYBRID_SHARD`

### 5. Bài thực hành

Code kèm theo và triển khai đầy đủ nằm trong **4.2 Data parallelism trong PyTorch từ nguyên lý đến thực chiến**:

- Bài A: cải tạo script một GPU thành bản DDP (30 phút)
- Bài B: trên 2–4 GPU, so sánh peak VRAM và throughput train của DDP vs FSDP
- Bài C: đo ảnh hưởng của các `sharding_strategy` khác nhau lên VRAM và tốc độ

---

## Mục tiêu học

- Giải thích được mỗi bước DP → DDP → FSDP giải quyết gì và đưa vào cái giá gì
- Dùng PyTorch DDP để đổi script một GPU thành train nhiều GPU (gồm Sampler, set_epoch, lưu model)
- Cấu hình được sharding strategy và auto-wrap policy của FSDP
- Tính được lượng giao tiếp mỗi bước của DDP và FSDP rồi giải thích sự khác biệt
