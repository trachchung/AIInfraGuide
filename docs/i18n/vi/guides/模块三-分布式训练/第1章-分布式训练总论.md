---
title: "Chương 1: Tổng quan huấn luyện phân tán"
description: "Từ nút thắt quy mô mô hình, tự tính sổ bộ nhớ huấn luyện, nhìn toàn cảnh năm chiến lược song song, rồi hạ xuống cấu hình rank/torchrun/NCCL và script DDP đầu tiên"
tags: ["Huấn luyện phân tán", "Phân tích bộ nhớ", "Chiến lược song song", "Song song 3D", "Topology giao tiếp", "torchrun", "NCCL", "DDP", "rank"]
sourceHash: "8a55f9a66a117430"
---

## Giới thiệu chương

Chương này là khung tổng của module huấn luyện phân tán. Nó dựng một khung nhận thức ba tầng: **vì sao** cần huấn luyện phân tán → huấn luyện thực sự cần **bao nhiêu bộ nhớ GPU** → có những chiến lược song song nào và **cách chúng chia việc**. Sau đó hạ xuống nền tảng kỹ thuật: GPU "lập nhóm" thế nào, dùng một lệnh torchrun để khởi chạy ra sao, và chạy thông script DDP đầu tiên thế nào. Đọc xong chương này, bạn có hệ tọa độ để hiểu mỗi chương sau đang giải bài toán gì, đồng thời đủ khả năng tự chạy huấn luyện đa GPU.

> Chương này gồm hai phần: nửa đầu là tổng luận nhận thức (sổ bộ nhớ + toàn cảnh song song), nửa sau là nền tảng kỹ thuật (mô hình tiến trình + torchrun + NCCL + script DDP tối thiểu), tương ứng hai bài chính bên dưới.

---

## Cấu trúc chương

### Phần 1: Tổng luận nhận thức (tương ứng bài 1.1)

#### 1. Vì sao cần huấn luyện phân tán

- **Nút thắt bộ nhớ**: Lấy dòng LLaMA làm ví dụ (7B → 70B → 405B), cho thấy quy mô mô hình và nhu cầu bộ nhớ tăng thế nào. Một H100 80GB đơn lẻ từ mức 7B trở lên đã không chứa nổi toàn bộ trạng thái huấn luyện
- **Nút thắt thời gian**: Huấn luyện mô hình 70B trên 1T token bằng một GPU mất nhiều năm; song song đa GPU có thể rút ngắn thời gian gần như tuyến tính
- **Kết luận lõi**: Huấn luyện phân tán giải hai bài toán—"không nhét nổi" (bộ nhớ) và "không chạy xong" (thời gian)

#### 2. Phân tích bộ nhớ trạng thái huấn luyện (trọng tâm chương)

- **Bốn thành phần bộ nhớ**:
  - Bộ nhớ tham số (FP16: $2\Psi$ Bytes)
  - Bộ nhớ gradient (FP16: $2\Psi$ Bytes)
  - Bộ nhớ trạng thái optimizer (Adam: bản sao tham số FP32 $4\Psi$ + moment bậc một $4\Psi$ + moment bậc hai $4\Psi$ = $12\Psi$ Bytes)
  - Bộ nhớ activation (phụ thuộc batch size, độ dài chuỗi và độ rộng mô hình)
- **Ví dụ tính tay**: Sổ bộ nhớ đầy đủ khi huấn luyện mô hình 7B bằng mixed precision BF16
  - Bộ nhớ tĩnh = $(2 + 2 + 12) \times 7 \times 10^9$ Bytes ≈ 112 GB
  - Ước lượng bộ nhớ activation (ví dụ seq_len=2048, batch=1)
- **Tóm tắt công thức bộ nhớ**: 16–18 Bytes mỗi tham số (mixed precision + Adam) + activation

#### 3. Toàn cảnh chiến lược song song

- **Song song dữ liệu (DP/DDP)**: Mỗi GPU giữ nguyên mô hình, chia dữ liệu, AllReduce đồng bộ gradient—giải "không chạy xong"
- **ZeRO / FSDP**: Chia trạng thái optimizer / gradient / tham số, gom khi cần—giải "không nhét nổi"
- **Song song tensor (TP)**: Chia phép nhân ma trận trong một lớp sang nhiều GPU—giải "một lớp quá lớn"
- **Song song pipeline (PP)**: Gán các lớp khác nhau cho các GPU khác nhau—giải "quá nhiều lớp"
- **Song song chuỗi (SP) / song song ngữ cảnh (CP)**: Chia theo chiều chuỗi—giải "chuỗi quá dài"

#### 4. Nguyên tắc chọn chiến lược song song

- **Băng thông giao tiếp quyết định phạm vi**: TP giới hạn trong máy nối bằng NVLink (cần băng thông cao, độ trễ thấp); PP có thể xuyên máy (chỉ truyền activation); DP đòi hỏi băng thông thấp nhất
- **Cây quyết định**: Một GPU nhét được → DDP; một GPU không nhét được → FSDP/ZeRO; một lớp không nhét được → TP; quá nhiều lớp → PP; kết hợp → song song 3D
- **Ánh xạ topology giao tiếp với phần cứng**: NVLink (900 GB/s) vs InfiniBand (400 Gbps) vs Ethernet ảnh hưởng thế nào tới việc chọn chiến lược

### Phần 2: Dựng môi trường và khởi chạy phân tán (tương ứng bài 1.2)

#### 5. Nhóm tiến trình và khái niệm lõi

- Ba quy mô—một GPU / nhiều GPU một máy / nhiều máy nhiều GPU—và phân tầng băng thông (NVLink vs InfiniBand)
- Mô hình tiến trình: vì sao là "một GPU một tiến trình" (tránh GIL, độc chiếm CUDA context)
- Khái niệm lõi: `rank` / `local_rank` / `world_size`, và vì sao `set_device` phải dùng `local_rank`
- Nhóm tiến trình `init_process_group` và chọn backend (nccl / gloo)
- Nhóm con `new_group`: mở đường cho song song 3D

#### 6. Cơ chế khởi chạy của torchrun

- Vì sao cần launcher riêng; tiến hóa từ `torch.distributed.launch` → `torchrun`
- Các biến môi trường torchrun inject (RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT)
- Khởi chạy nhiều GPU một máy, nhiều máy nhiều GPU, và Rendezvous (c10d elastic rendezvous)
- Khả năng đàn hồi và chịu lỗi của torchrun (xem Chương 14)

#### 7. Biến môi trường NCCL và gỡ lỗi giao tiếp

- `NCCL_DEBUG=INFO` là bước gỡ lỗi đầu tiên
- `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` để chỉ định card mạng
- `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` làm công tắc tìm kiếm nhị phân khi gỡ lỗi
- `CUDA_VISIBLE_DEVICES` và gắn thiết bị

#### 8. Tự viết script huấn luyện đa GPU đầu tiên

- Khung năm đoạn của script DDP tối thiểu: init → set_device → bọc DDP → vòng huấn luyện (có Sampler) → destroy
- Template chạy được đầy đủ và phân tích từng đoạn
- Quy ước kỹ thuật chỉ rank 0 làm việc phụ; `DistributedSampler` + `set_epoch`
- Các bẫy thường gặp lần đầu chạy đa GPU

---

## Mục tiêu học tập

- Tính được nhu cầu bộ nhớ huấn luyện mixed precision BF16 cho mô hình bất kỳ quy mô (biết số tham số)
- Vẽ được bảng so sánh năm chiến lược song song theo "chia cái gì × giao tiếp thế nào"
- Chọn được tổ hợp chiến lược song song phù hợp theo quy mô mô hình và topology cụm
- Phân biệt chính xác rank / local_rank / world_size, và viết được lệnh torchrun khởi chạy một máy cũng như nhiều máy
- Tự viết và dùng torchrun chạy thông một script DDP tối thiểu
- Nói được việc đầu tiên khi gặp lỗi giao tiếp đa GPU (bật `NCCL_DEBUG=INFO`) cùng các cách kiểm tra thường dùng
