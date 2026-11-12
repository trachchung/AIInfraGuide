---
title: "Nhập môn framework PyTorch"
description: "PyTorch là framework chuẩn thực tế cho huấn luyện và suy luận mô hình lớn hiện nay"
tags: ["PyTorch", "Tensor", "autograd", "Đồ thị tính toán", "GPU"]
sourceHash: "a223c60d2643cf27"
---

PyTorch là framework chuẩn thực tế cho huấn luyện và suy luận mô hình lớn hiện nay. Bài này tập trung vào những gì kỹ sư AI Infra cần: từ thao tác Tensor đến vòng huấn luyện đầy đủ, rồi dùng profiler để tìm nút thắt hiệu năng, làm nền cho lập trình CUDA, huấn luyện phân tán và tối ưu suy luận sau này.

<!-- more -->

## Mục lục

- [1. Tensor: đơn vị dữ liệu cơ bản của PyTorch](#1-tensor-đơn-vị-dữ-liệu-cơ-bản-của-pytorch)
- [2. Vi phân tự động (autograd)](#2-vi-phân-tự-động-autograd)
- [3. nn.Module: cách tổ chức mô hình](#3-nnmodule-cách-tổ-chức-mô-hình)
- [4. Vòng huấn luyện đầy đủ](#4-vòng-huấn-luyện-đầy-đủ)
- [5. Cơ bản huấn luyện trên GPU](#5-cơ-bản-huấn-luyện-trên-gpu)
- [6. Giới thiệu phân tích hiệu năng](#6-giới-thiệu-phân-tích-hiệu-năng)
- [Tóm tắt](#tóm-tắt)
- [Danh sách tự kiểm](#danh-sách-tự-kiểm)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## 1. Tensor: đơn vị dữ liệu cơ bản của PyTorch

Nếu deep learning giống xây bằng khối, Tensor là khối cơ bản. Tham số mô hình, dữ liệu đầu vào và kết quả trung gian trong PyTorch đều sống dưới dạng Tensor.

**Định nghĩa hình thức**: Tensor là mảng nhiều chiều. Bạn có thể coi nó như ndarray của NumPy được tăng tốc GPU, đồng thời hỗ trợ vi phân tự động.

### 1.1 Các cách tạo Tensor thường dùng

```python
import torch
import numpy as np

a = torch.tensor([1.0, 2.0, 3.0])          # tạo từ list Python
zeros = torch.zeros(3, 4)                    # ma trận 3x4 toàn 0
ones = torch.ones(2, 3, 4)                   # tensor 2x3x4 toàn 1
rand_normal = torch.randn(3, 4)              # phân phối chuẩn chuẩn
seq = torch.arange(0, 10, 2)                 # tensor([0, 2, 4, 6, 8])

x = torch.randn(3, 4)
y = torch.zeros_like(x)                      # Tensor toàn 0 cùng shape với x

t = torch.from_numpy(np.array([1.0, 2.0]))   # từ NumPy (chia sẻ bộ nhớ)
```

### 1.2 Thao tác shape: view, reshape, permute, squeeze, unsqueeze

Hãy nghĩ Tensor như đất sét bạn có thể nhào lại hình: dữ liệu giữ nguyên, chỉ layout đổi.

```python
import torch

x = torch.arange(12)             # 1D, 12 phần tử
a = x.view(3, 4)                 # thành 3x4 (cần bộ nhớ contiguous)
b = x.reshape(3, 4)              # giống view, nhưng cũng chạy khi bộ nhớ không contiguous
c = x.view(-1, 4)                # -1 được suy ra là 3

# permute: đổi thứ tự dim, hay gặp trong Transformer
t = torch.randn(2, 8, 12, 64)              # (batch, seq_len, heads, dim)
t_permuted = t.permute(0, 2, 1, 3)         # (batch, heads, seq_len, dim)

# squeeze/unsqueeze: bỏ hoặc chèn dim kích thước 1
e = torch.randn(1, 3, 1, 4)
print(e.squeeze().shape)                    # torch.Size([3, 4])
f = torch.randn(3, 4)
print(f.unsqueeze(0).shape)                 # torch.Size([1, 3, 4])
```

### 1.3 Quản lý device: chuyển dữ liệu giữa CPU và GPU

```python
import torch

print(torch.cuda.is_available())            # GPU có sẵn không

x = torch.randn(3, 4)
x_gpu = x.to('cuda')                        # CPU → GPU (nên dùng)
x_gpu = x.cuda()                             # dạng ngắn tương đương
x_cpu = x_gpu.cpu()                          # GPU → CPU

# tạo thẳng trên GPU (tránh một lần copy thừa)
x_gpu = torch.randn(3, 4, device='cuda')
```

> **Góc nhìn AI Infra**: Chuyển CPU–GPU đi qua PCIe, băng thông thấp hơn nhiều so với bộ nhớ GPU. Gọi `.cpu()` / `.cuda()` thường xuyên thường là nút thắt ẩn. Mục 6 chỉ cách bắt chúng bằng profiler.

### 1.4 dtype và bộ nhớ: fp32, fp16, bf16

Độ chính xác dữ liệu quyết định trực tiếp mức dùng bộ nhớ và tốc độ tính:

| dtype | Byte/phần tử | Dải số | Trường hợp dùng |
|-------|:-------:|---------|---------|
| float32 (fp32) | 4 | Lớn | Độ chính xác mặc định, trạng thái optimizer |
| float16 (fp16) | 2 | Nhỏ, dễ overflow | Huấn luyện mixed-precision (cần Loss Scaling) |
| bfloat16 (bf16) | 2 | Như fp32 | Huấn luyện mixed-precision (khuyến nghị, GPU Ampere+) |

> **Nói thường**: fp32 như ảnh độ phân giải cao: lớn và sắc. fp16 như ảnh nén: nửa kích thước, đôi khi "méo" (overflow). bf16 là nén khôn hơn: cùng kích thước với fp16 nhưng ít "méo" hơn nhiều.

```python
import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)   # 4 byte/phần tử
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # 2 byte/phần tử
print(f"fp32: {x_fp32.nelement() * x_fp32.element_size() / 1024:.0f} KB")  # 3906 KB
print(f"bf16: {x_bf16.nelement() * x_bf16.element_size() / 1024:.0f} KB")  # 1953 KB

x = torch.randn(3, 4)                     # mặc định fp32
x_half = x.half()                          # sang fp16
x_bf16 = x.to(torch.bfloat16)             # sang bf16
```

---

## 2. Vi phân tự động (autograd)

Huấn luyện mạng nơ-ron là "điều chỉnh tham số từ loss", và tín hiệu điều chỉnh là gradient. Engine autograd của PyTorch làm việc đó giúp bạn.

### 2.1 Đồ thị tính toán là gì

> **Ẩn dụ**: Hãy tưởng tượng nấu một món. Bạn trộn nguyên liệu A và B thành C, rồi đun C thành D (món xong). Nếu D vị sai (loss quá cao), bạn lần ngược: lỗi ở C hay tỷ lệ A/B? Đồ thị tính toán là "công thức" PyTorch giữ: nó ghi lại từng bước để đi ngược từ kết quả và lấy đóng góp của từng nguyên liệu (gradient).

**Định nghĩa hình thức**: Đồ thị tính toán là đồ thị có hướng không chu trình (DAG). Nút là Tensor; cạnh là op. PyTorch dựng đồ thị động trong forward pass và tính gradient dọc theo nó trong backward pass. PyTorch dùng **đồ thị tính toán động** (Define-by-Run), nên control flow Python như if/else và vòng for cứ chạy bình thường.

### 2.2 requires_grad và backward()

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)  # theo dõi mọi op
y = x * 3
z = y.sum()        # z = 3*2 + 3*3 = 15
z.backward()       # backward pass
print(x.grad)      # tensor([3., 3.]), dz/dx = 3
```

Điểm then chốt: `backward()` chỉ chạy trên scalar; chỉ leaf node giữ gradient; đồ thị được giải phóng sau khi dùng.

### 2.3 Tích lũy gradient và xóa gradient

> **Nói thường**: Mặc định PyTorch **tích lũy** gradient và không tự xóa. Nghĩ như một bộ đếm tăng mỗi lần `backward()`. Trong vòng huấn luyện chuẩn bạn phải zero ở đầu mỗi step, nếu không gradient step trước sẽ trộn vào.

Thiết kế đó có chủ đích: gradient accumulation là mẹo phổ biến để giả lập batch lớn khi bộ nhớ eo hẹp.

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
(x * 2).sum().backward()
print(x.grad)         # tensor([2.])
(x * 3).sum().backward()
print(x.grad)         # tensor([5.]) ← 2+3, gradient đã tích lũy!
x.grad.zero_()        # xóa thủ công
(x * 4).sum().backward()
print(x.grad)         # tensor([4.]) ← đúng sau khi xóa
# Trong huấn luyện thật, dùng optimizer.zero_grad() để xóa hết grad tham số một lần
```

### Ví dụ code: gradient descent thủ công cho y = 2x + 1

```python
import torch

w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
x_train = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_train = torch.tensor([3.0, 5.0, 7.0, 9.0])

for epoch in range(100):
    loss = ((w * x_train + b - y_train) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(f"Learned model: y = {w.item():.2f}x + {b.item():.2f}")
# output is close to y = 2.00x + 1.00
```

---

## 3. nn.Module: cách tổ chức mô hình

> **Ẩn dụ**: Nếu Tensor là khối xây, `nn.Module` là tờ hướng dẫn: nó định nghĩa khối nối thế nào (forward pass) và giúp bạn kiểm kê từng bộ phận (quản lý tham số).

### 3.1 Định nghĩa mô hình: subclass nn.Module, implement __init__ và forward

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

model = SimpleModel(784, 256, 10)
output = model(torch.randn(32, 784))    # gọi model(x); đừng gọi forward trực tiếp
print(output.shape)                       # torch.Size([32, 10])
```

### 3.2 Quản lý tham số: parameters(), named_parameters(), state_dict()

```python
import torch.nn as nn

model = nn.Linear(10, 5)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")   # weight: [5, 10], bias: [5]

# đếm tham số (rất hay gặp trong AI Infra)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameter count: {total_params}")       # 55

# state_dict dùng để lưu và nạp mô hình
print(model.state_dict().keys())       # odict_keys(['weight', 'bias'])
```

### 3.3 Các layer thường gặp

```python
import torch
import torch.nn as nn

# nn.Linear: layer dùng nhiều nhất trong mô hình lớn (Attention Q/K/V projection, FFN)
linear = nn.Linear(512, 256)

# nn.Embedding: map token id sang vector dày (layer đầu của mô hình lớn)
embedding = nn.Embedding(num_embeddings=50000, embedding_dim=768)
print(embedding(torch.tensor([0, 42])).shape)   # torch.Size([2, 768])

# nn.LayerNorm: sau mọi Transformer sublayer, ổn định huấn luyện
layer_norm = nn.LayerNorm(768)

# nn.Dropout: ngẫu nhiên tắt neuron khi train; tự tắt sau model.eval()
dropout = nn.Dropout(p=0.1)
```

### Ví dụ code: MLP hai lớp

```python
import torch
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

model = TwoLayerMLP(784, 256, 10)
print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")  # 203,530
print(model(torch.randn(64, 784)).shape)  # torch.Size([64, 10])
```

---

## 4. Vòng huấn luyện đầy đủ

Vòng huấn luyện là code kỹ sư AI Infra chạm mỗi ngày. Ngay cả trong setup phân tán, vòng lõi vẫn cùng một mẫu.

### 4.1 Nạp dữ liệu: Dataset và DataLoader

- **Dataset**: định nghĩa "dataset có gì" và "lấy một sample thế nào"
- **DataLoader**: định nghĩa "đưa batch vào mô hình thế nào" (batching, shuffling, nạp đa tiến trình)

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = SimpleDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
loader = DataLoader(dataset, batch_size=64, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
```

> **Góc nhìn AI Infra**: `num_workers` quá ít khiến GPU chờ dữ liệu (nút thắt data loading). `pin_memory=True` cho phép transfer dùng DMA bất đồng bộ và tránh một lần copy CPU.

### 4.2 Luồng huấn luyện chuẩn

Năm bước lõi mỗi step: forward → loss → backward → optimizer.step → zero_grad.

### 4.3 Lập lịch learning rate

```python
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# các scheduler thường gặp
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# gọi scheduler.step() sau mỗi epoch
```

### 4.4 Lưu và khôi phục checkpoint

```python
import torch

# lưu (mô hình + optimizer + tiến độ huấn luyện; resume cần cả ba)
torch.save({
    'epoch': epoch, 'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss,
}, 'checkpoint.pt')

# nạp
ckpt = torch.load('checkpoint.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
```

### Ví dụ code: script huấn luyện MNIST đầy đủ

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size, lr, num_epochs = 128, 1e-3, 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2, pin_memory=True)

# model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                                 nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# train
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    scheduler.step()

    # validate
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} "
          f"| Acc: {correct/len(test_set)*100:.2f}%")

    # save checkpoint
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, f'ckpt_ep{epoch+1}.pt')
```

---

## 5. Cơ bản huấn luyện trên GPU

### 5.1 Đưa mô hình và dữ liệu lên GPU

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Linear(100, 10).to(device)
x = torch.randn(32, 100).to(device)     # mô hình và dữ liệu phải cùng device
y = model(x)
```

### 5.2 Giới thiệu huấn luyện mixed-precision

Ý tưởng lõi: dùng độ chính xác thấp (fp16/bf16) cho forward và backward để tăng tốc tính; giữ độ chính xác cao (fp32) khi cập nhật tham số để không mất độ chính xác.

> **Nói thường**: Nháp bằng bút chì (độ chính xác thấp, nhanh); chốt bằng bút mực (độ chính xác cao, chính xác). Giữ bước cập nhật tham số then chốt ở độ chính xác cao để không mất thông tin; chạy phần lớn phép toán ở độ chính xác thấp để nhanh hơn.

```python
import torch
import torch.nn as nn

device = torch.device('cuda')
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Mixed precision BF16 (khuyến nghị, GPU Ampere+, không cần GradScaler)
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Mixed precision FP16 (cần GradScaler để tránh gradient underflow)
scaler = torch.cuda.amp.GradScaler()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = nn.functional.cross_entropy(model(images), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 5.3 Kiểm tra bộ nhớ GPU

```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Peak:      {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(torch.cuda.memory_summary(abbreviated=True))

# phân tích bộ nhớ cho một đoạn code
torch.cuda.reset_peak_memory_stats()
# ... chạy code mục tiêu ...
print(f"Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

### Ví dụ code: so sánh bộ nhớ mixed precision với fp32

```python
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "GPU required"
device = torch.device('cuda')
criterion = nn.CrossEntropyLoss()

# Batch phải đủ lớn để activation chiếm phần lớn bộ nhớ. Mixed precision không
# thu nhỏ weight FP32 hay trạng thái optimizer; nếu batch quá nhỏ overhead cố định đó
# che hết phần tiết kiệm activation, thậm chí bạn thấy "tiết kiệm âm"
BATCH = 65536
data = torch.randn(BATCH, 1024, device=device)
labels = torch.randint(0, 10, (BATCH,), device=device)

def make_model():
    return nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(),
                         nn.Linear(2048, 1024), nn.ReLU(),
                         nn.Linear(1024, 10)).to(device)

def step_peak(amp_dtype=None):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if amp_dtype is None:
        loss = criterion(model(data), labels)
    else:
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            loss = criterion(model(data), labels)
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    # Quan trọng: thật sự giải phóng model và trạng thái optimizer vòng này, nếu không sẽ cộng vào peak vòng sau
    del model, opt, loss
    torch.cuda.empty_cache()
    return peak

fp32_peak = step_peak(None)
bf16_peak = step_peak(torch.bfloat16)

print(f"FP32: {fp32_peak:.1f} MB | BF16: {bf16_peak:.1f} MB | Saved: {(1-bf16_peak/fp32_peak)*100:.1f}%")
# reference output: FP32: 1881.6 MB | BF16: 1246.6 MB | Saved: 33.7%
```

⚠️ **Hai bẫy dễ gặp**:

1. `torch.cuda.reset_peak_memory_stats()` chỉ reset bộ đếm peak; nó **không** giải phóng bộ nhớ đã cấp. Nếu model FP32 và trạng thái optimizer vòng trước còn sống khi bạn đo BF16, chúng bị tính vào peak BF16 và mức tiết kiệm có thể trông âm. Bạn phải `del` object rồi mới `empty_cache()`.
2. Mixed precision chủ yếu tiết kiệm **activation** (và một số trung gian). Weight và moment bậc một/hai của AdamW vẫn FP32. Với mô hình nhỏ và batch nhỏ, overhead cố định đó chiếm ưu thế, và bạn có thể đo được "0% tiết kiệm." Tăng batch/sequence length đến khi activation chiếm phần lớn; lúc đó bạn sẽ thấy khoảng 30%–50% tiết kiệm.

---

## 6. Giới thiệu phân tích hiệu năng

Cho code huấn luyện chạy được chỉ là bước một. Làm cho nó nhanh mới là mục tiêu lõi của AI Infra. `torch.profiler` cho biết mỗi op mất bao lâu, GPU tận dụng thế nào, và chỗ nào nó ngồi chờ dữ liệu.

### 6.1 Cách dùng torch.profiler cơ bản

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.nn.Linear(1024, 1024).cuda()
x = torch.randn(64, 1024).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    with record_function("forward_pass"):
        output = model(x)
    with record_function("backward_pass"):
        output.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 6.2 Cách đọc output của profiler

| Cột | Ý nghĩa | Cần chú ý gì |
|------|------|--------|
| Name | Tên op | Tìm các op đắt nhất |
| CPU total | Tổng thời gian phía CPU | Lớn hơn CUDA total nhiều → nút thắt CPU |
| CUDA total | Tổng thời gian phía GPU | Chỉ số lõi |
| # of Calls | Số lần gọi | Nhiều op nhỏ tần suất cao có thể cộng dồn thành nút thắt |

Các mẫu hiệu năng thường gặp:

1. **CPU time lớn hơn CUDA time nhiều**: GPU đang chờ CPU (tiền xử lý dữ liệu, overhead Python)
2. **Rất nhiều CUDA kernel nhỏ**: launch overhead cộng dồn; cân nhắc `torch.compile` hoặc operator fusion
3. **Đồng bộ CPU–GPU thường xuyên**: `.item()` và `print(tensor)` buộc sync và làm đứng pipeline GPU

### 6.3 Các vấn đề hiệu năng thường gặp

**Nút thắt data loading**: tăng `num_workers`, dùng `prefetch_factor` để prefetch.

**Di chuyển dữ liệu CPU–GPU**: tránh gọi `.cuda()` lặp trong vòng lặp; đưa lên GPU một lần.

**Sync ẩn**: `loss.item()` kích hoạt sync CPU–GPU; chỉ ghi lại mỗi N step.

```python
# bad: sync every step
for batch in dataloader:
    loss = train_step(batch)
    print(f"loss: {loss.item()}")       # sync every step!

# good: record every 100 steps
for i, batch in enumerate(dataloader):
    loss = train_step(batch)
    if i % 100 == 0:
        print(f"step {i}, loss: {loss.item()}")
```

### Ví dụ code: profile một training step

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(),
                      nn.Linear(4096, 4096), nn.ReLU(),
                      nn.Linear(4096, 10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
data = torch.randn(128, 1024, device=device)
labels = torch.randint(0, 10, (128,), device=device)

# warmup (tránh overhead init lần gọi đầu)
for _ in range(3):
    criterion(model(data), labels).backward()
    optimizer.step(); optimizer.zero_grad()

# phân tích profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    with record_function("data_to_gpu"):
        d = torch.randn(128, 1024).to(device)
        l = torch.randint(0, 10, (128,)).to(device)
    with record_function("forward"):
        loss = criterion(model(d), l)
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step(); optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")   # visualize in chrome://tracing
```

Từ output bạn nhanh chóng biết backward có chậm hơn forward không (khoảng 2× là bình thường), di chuyển dữ liệu có chiếm quá nhiều thời gian không, và bước optimizer có trông đắt bất thường không.

---

## Tóm tắt

| Lớp | Nội dung | Kỹ năng lõi |
|------|------|---------|
| Tensor | Biểu diễn dữ liệu và thao tác | Thao tác shape, quản lý device, chọn dtype |
| autograd | Vi phân tự động | Đồ thị tính toán, tính gradient, xóa gradient |
| nn.Module | Tổ chức mô hình | Định nghĩa mô hình, quản lý tham số, các layer thường gặp |
| Vòng huấn luyện | Huấn luyện end-to-end | DataLoader, luồng huấn luyện, checkpoint |
| Huấn luyện GPU | Tăng tốc phần cứng | Mixed precision, quản lý bộ nhớ |
| Profiling | Tìm nút thắt | Dùng profiler, nhận diện nút thắt thường gặp |

Đây là nền bạn cần trước khi vào huấn luyện phân tán, lập trình CUDA và tối ưu suy luận. Có chúng rồi thì đọc code DeepSpeed hay Megatron-LM ít bị kẹt vì nền PyTorch lung lay.

---

## Danh sách tự kiểm

Sau bài này bạn nên làm được:

- Tạo Tensor mọi shape với `torch.randn`, `torch.zeros`, và reshape bằng `view`, `permute`, `squeeze` cùng các op tương tự
- Giải thích `requires_grad=True` làm gì, dựng đồ thị tính toán bằng tay, và gọi `backward()` lấy gradient
- Giải thích vì sao mỗi training step cần `optimizer.zero_grad()`, và gradient accumulation hoạt động thế nào
- Subclass `nn.Module` cho mô hình tùy chỉnh, và quản lý tham số bằng `parameters()` cùng `state_dict()`
- Viết vòng huấn luyện đầy đủ: DataLoader → forward → loss → backward → optimizer.step → checkpoint
- Dùng `torch.autocast` cho huấn luyện mixed-precision bf16/fp16, và giải thích `GradScaler` để làm gì
- Dùng `torch.profiler` trên một training step, đọc output, và phát hiện nút thắt data-loading cùng transfer CPU–GPU
- Giải thích khác biệt giữa fp32, fp16 và bf16 cùng cách chọn

## Tài liệu tham khảo

- [PyTorch official docs](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [torch.profiler docs](https://pytorch.org/docs/stable/profiler.html)
- [Automatic Mixed Precision (AMP) docs](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
