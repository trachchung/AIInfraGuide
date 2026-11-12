---
title: "Chương 2: Primitive giao tiếp tập thể"
description: "Nền tảng vật lý của huấn luyện phân tán—ngữ nghĩa các phép AllReduce/ReduceScatter/AllGather/All-to-All và các collective khác, nguyên lý thuật toán Ring, và phân tích định lượng lượng giao tiếp"
tags: ["Giao tiếp tập thể", "AllReduce", "Ring AllReduce", "NCCL", "Phân tích lượng giao tiếp"]
sourceHash: "52e8a069756a7f48"
---

## Giới thiệu chương

Mọi chiến lược song song cuối cùng đều đổ về một việc: **các GPU trao đổi dữ liệu hiệu quả thế nào**. Các tên như AllReduce, ReduceScatter, AllGather sẽ xuất hiện lại ở mọi chương sau—DDP dùng AllReduce đồng bộ gradient, ZeRO-2 dùng ReduceScatter, TP dùng AllReduce, MoE dùng All-to-All. Chương này giảng hết bộ "bảng chữ cái giao tiếp" một lần, và đưa ra cách định lượng lượng giao tiếp, để sau này phân tích chi phí giao tiếp của bất kỳ chiến lược nào bạn cũng tự tính được.

> Chương này là kiến thức nền để hiểu chi phí giao tiếp của mọi chiến lược song song phía sau. Nên đọc đến mức nhớ và vẽ được sơ đồ ngữ nghĩa "đầu vào → đầu ra" của từng primitive.

---

## Cấu trúc chương

### 1. Vì sao phải nói riêng về primitive giao tiếp

- Bản chất chiến lược song song = chia + giao tiếp; chia quyết định bộ nhớ, giao tiếp quyết định tốc độ
- Một bảng đối chiếu "phép giao tiếp vs chiến lược song song" (toàn cảnh trước, chi tiết sau)
- Giao tiếp xảy ra ở đâu: NVLink (trong máy) vs InfiniBand/RoCE (giữa máy) vs PCIe, băng thông lệch cả một bậc độ lớn

### 2. Giao tiếp điểm-điểm (Point-to-Point)

- `send` / `recv`: giao tiếp hai tiến trình cơ bản nhất; PP truyền activation giữa các Stage nhờ nó
- Đồng bộ vs bất đồng bộ (`isend` / `irecv`) và rủi ro deadlock

### 3. Chi tiết các primitive giao tiếp tập thể

Với từng phép: sơ đồ ngữ nghĩa "đầu vào → đầu ra" + công dụng điển hình:

- **Broadcast**: Sao chép dữ liệu của một rank sang mọi rank (ví dụ đồng bộ tham số lúc khởi tạo)
- **Scatter / Gather**: Phân tán / thu gom (cách DP cũ, chia dữ liệu)
- **Reduce**: Quy ước dữ liệu mọi rank (cộng / lấy max) về một rank
- **AllReduce**: Phát kết quả đã quy ước tới mọi rank (đồng bộ gradient DDP, gộp đầu ra TP)—trọng tâm chương
- **ReduceScatter**: Sau khi quy ước, mỗi rank chỉ giữ shard mình phụ trách (ZeRO-2/3, chiều ngược FSDP)
- **AllGather**: Ghép shard của mọi rank thành dữ liệu đầy đủ rồi phát cho mọi rank (chiều xuôi ZeRO-3, FSDP lắp lại tham số)
- **All-to-All**: Mỗi rank gửi dữ liệu khác nhau cho mỗi rank khác (MoE Expert Parallel, song song chuỗi Ulysses)
- **Đẳng thức then chốt**: `AllReduce = ReduceScatter + AllGather` (xuyên suốt ZeRO/SP)

### 4. Nguyên lý thuật toán Ring AllReduce (trọng tâm chương)

- Vấn đề của AllReduce ngây thơ: nút thắt băng thông ở nút trung tâm
- Hai giai đoạn của Ring: giai đoạn Reduce-Scatter + giai đoạn AllGather
- **Suy ra lượng giao tiếp**: mỗi GPU gửi/nhận $2\frac{N-1}{N}\Psi \approx 2\Psi$ (không phụ thuộc số GPU—đó là tinh túy của Ring)
- Tối ưu về băng thông: vì sao Ring với dữ liệu lớn tiến sát trần băng thông
- Các topology khác (tóm tắt): Tree AllReduce (độ trễ thấp với dữ liệu nhỏ), Double Binary Tree, và cách NCCL tự chọn

### 5. Cách định lượng lượng giao tiếp

- Ký hiệu thống nhất: số tham số $\Psi$, batch $b$, chuỗi $s$, hidden dim $h$
- Ba công thức lượng giao tiếp: AllReduce $2\Psi$, ReduceScatter / AllGather mỗi cái $\Psi$, và All-to-All
- Ước lượng thời gian giao tiếp: $T \approx \alpha + \frac{\text{lượng dữ liệu}}{\text{băng thông}}$ (hạng độ trễ $\alpha$ + hạng băng thông)
- Nguyên lý chồng tính toán/giao tiếp (overlap): vì sao có thể "tính đồng thời với truyền"

### 6. Thực hành

- Dùng `torch.distributed` chạy thông 7 primitive giao tiếp tập thể, in đầu vào/đầu ra từng rank để kiểm chứng ngữ nghĩa
- Dùng `NCCL_DEBUG=INFO` quan sát NCCL chọn thuật toán nào
- Đo băng thông AllReduce thực tế với các kích thước dữ liệu khác nhau, so sánh NVLink với PCIe

---

## Mục tiêu học tập

- Vẽ được sơ đồ ngữ nghĩa "đầu vào → đầu ra" của 7 primitive giao tiếp tập thể
- Suy ra được Ring AllReduce mỗi GPU truyền $2\Psi$ và không phụ thuộc số GPU
- Thuộc đẳng thức $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$ và giải thích ý nghĩa của nó trong ZeRO
- Ước lượng được thời gian một lần giao tiếp bằng $T \approx \alpha + \text{lượng dữ liệu}/\text{băng thông}$
