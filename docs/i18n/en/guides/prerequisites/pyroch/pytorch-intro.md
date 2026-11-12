---
title: "PyTorch framework quick start"
description: "PyTorch is the de facto standard framework for large-model training and inference today"
tags: ["PyTorch", "Tensor", "autograd", "Computational graph", "GPU"]
sourceHash: "a223c60d2643cf27"
---

PyTorch is the de facto standard framework for large-model training and inference today. This article focuses on what an AI Infra engineer needs: from Tensor ops to a full training loop, then using the profiler to find performance bottlenecks, so you have a base for CUDA programming, distributed training, and inference optimization later.

<!-- more -->

## Table of contents

- [1. Tensor: PyTorch's basic data unit](#1-tensor-pytorchs-basic-data-unit)
- [2. Automatic differentiation (autograd)](#2-automatic-differentiation-autograd)
- [3. nn.Module: how models are organized](#3-nnmodule-how-models-are-organized)
- [4. A full training loop](#4-a-full-training-loop)
- [5. GPU training basics](#5-gpu-training-basics)
- [6. Performance profiling intro](#6-performance-profiling-intro)
- [Summary](#summary)
- [Self-check list](#self-check-list)
- [References](#references)

---

## 1. Tensor: PyTorch's basic data unit

If deep learning is building with blocks, Tensor is the basic block. Model parameters, input data, and intermediate results all live as Tensors in PyTorch.

**Formal definition**: A Tensor is a multi-dimensional array. You can treat it as a GPU-accelerated NumPy ndarray that also supports automatic differentiation.

### 1.1 Common ways to create a Tensor

```python
import torch
import numpy as np

a = torch.tensor([1.0, 2.0, 3.0])          # create from a Python list
zeros = torch.zeros(3, 4)                    # 3x4 all-zero matrix
ones = torch.ones(2, 3, 4)                   # 2x3x4 all-ones tensor
rand_normal = torch.randn(3, 4)              # standard normal
seq = torch.arange(0, 10, 2)                 # tensor([0, 2, 4, 6, 8])

x = torch.randn(3, 4)
y = torch.zeros_like(x)                      # all-zero Tensor with the same shape as x

t = torch.from_numpy(np.array([1.0, 2.0]))   # from NumPy (shares memory)
```

### 1.2 Shape ops: view, reshape, permute, squeeze, unsqueeze

Think of a Tensor as clay you can reshape freely: the data stays the same; only the layout changes.

```python
import torch

x = torch.arange(12)             # 1D, 12 elements
a = x.view(3, 4)                 # become 3x4 (requires contiguous memory)
b = x.reshape(3, 4)              # like view, but also works when memory is non-contiguous
c = x.view(-1, 4)                # -1 is inferred as 3

# permute: swap dims, common in Transformers
t = torch.randn(2, 8, 12, 64)              # (batch, seq_len, heads, dim)
t_permuted = t.permute(0, 2, 1, 3)         # (batch, heads, seq_len, dim)

# squeeze/unsqueeze: remove or insert size-1 dims
e = torch.randn(1, 3, 1, 4)
print(e.squeeze().shape)                    # torch.Size([3, 4])
f = torch.randn(3, 4)
print(f.unsqueeze(0).shape)                 # torch.Size([1, 3, 4])
```

### 1.3 Device management: moving data between CPU and GPU

```python
import torch

print(torch.cuda.is_available())            # whether a GPU is available

x = torch.randn(3, 4)
x_gpu = x.to('cuda')                        # CPU → GPU (preferred)
x_gpu = x.cuda()                             # equivalent short form
x_cpu = x_gpu.cpu()                          # GPU → CPU

# create directly on GPU (avoid an extra copy)
x_gpu = torch.randn(3, 4, device='cuda')
```

> **AI Infra view**: CPU–GPU transfers go over PCIe, whose bandwidth is far below GPU memory bandwidth. Frequent `.cpu()` / `.cuda()` calls are often a hidden bottleneck. Section 6 shows how to catch them with the profiler.

### 1.4 dtype and memory: fp32, fp16, bf16

Data precision directly decides memory use and compute speed:

| dtype | Bytes/element | Numeric range | Use case |
|-------|:-------:|---------|---------|
| float32 (fp32) | 4 | Large | Default precision, optimizer state |
| float16 (fp16) | 2 | Small, easy to overflow | Mixed-precision training (needs Loss Scaling) |
| bfloat16 (bf16) | 2 | Same as fp32 | Mixed-precision training (recommended, Ampere+ GPUs) |

> **Plain-language take**: fp32 is like a high-res photo: large and sharp. fp16 is a compressed photo: half the size, sometimes "distorted" (overflow). bf16 is a smarter compression: same size as fp16, but much less likely to "distort."

```python
import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)   # 4 bytes/element
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # 2 bytes/element
print(f"fp32: {x_fp32.nelement() * x_fp32.element_size() / 1024:.0f} KB")  # 3906 KB
print(f"bf16: {x_bf16.nelement() * x_bf16.element_size() / 1024:.0f} KB")  # 1953 KB

x = torch.randn(3, 4)                     # default fp32
x_half = x.half()                          # to fp16
x_bf16 = x.to(torch.bfloat16)             # to bf16
```

---

## 2. Automatic differentiation (autograd)

Training a neural net is about "adjust parameters from the loss," and the adjustment signal is the gradient. PyTorch's autograd engine does that for you.

### 2.1 What is a computational graph

> **Analogy**: Imagine cooking a dish. You mix ingredients A and B into C, then heat C into D (the finished dish). If D tastes wrong (loss too high), you reverse-engineer: is C the problem, or the A/B ratio? The computational graph is the "recipe" PyTorch keeps: it records every step so you can go backward from the result and get each ingredient's contribution (the gradient).

**Formal definition**: A computational graph is a directed acyclic graph (DAG). Nodes are Tensors; edges are ops. PyTorch builds the graph dynamically during the forward pass and computes gradients along it during the backward pass. PyTorch uses a **dynamic computational graph** (Define-by-Run), so Python control flow like if/else and for loops just works.

### 2.2 requires_grad and backward()

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)  # track all ops
y = x * 3
z = y.sum()        # z = 3*2 + 3*3 = 15
z.backward()       # backward pass
print(x.grad)      # tensor([3., 3.]), dz/dx = 3
```

Key points: `backward()` only works on scalars; only leaf nodes keep gradients; the graph is freed after use.

### 2.3 Gradient accumulation and zeroing

> **Plain-language take**: By default PyTorch **accumulates** gradients and does not clear them. Think of a counter that goes up on every `backward()`. In a standard training loop you must zero at the start of each step, or the previous step's gradients mix in.

That design is intentional: gradient accumulation is a common trick to simulate a large batch when memory is tight.

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
(x * 2).sum().backward()
print(x.grad)         # tensor([2.])
(x * 3).sum().backward()
print(x.grad)         # tensor([5.]) ← 2+3, gradients accumulated!
x.grad.zero_()        # clear manually
(x * 4).sum().backward()
print(x.grad)         # tensor([4.]) ← correct after clearing
# In real training, use optimizer.zero_grad() to clear all parameter grads at once
```

### Code example: manual gradient descent for y = 2x + 1

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

## 3. nn.Module: how models are organized

> **Analogy**: If Tensor is a building block, `nn.Module` is the instruction sheet: it defines how blocks connect (the forward pass) and helps you inventory every part (parameter management).

### 3.1 Define a model: subclass nn.Module, implement __init__ and forward

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
output = model(torch.randn(32, 784))    # call model(x); do not call forward directly
print(output.shape)                       # torch.Size([32, 10])
```

### 3.2 Parameter management: parameters(), named_parameters(), state_dict()

```python
import torch.nn as nn

model = nn.Linear(10, 5)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")   # weight: [5, 10], bias: [5]

# count parameters (very common in AI Infra)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameter count: {total_params}")       # 55

# state_dict is used to save and load models
print(model.state_dict().keys())       # odict_keys(['weight', 'bias'])
```

### 3.3 Common layers

```python
import torch
import torch.nn as nn

# nn.Linear: the most-used layer in large models (Attention Q/K/V projections, FFN)
linear = nn.Linear(512, 256)

# nn.Embedding: map token ids to dense vectors (first layer of large models)
embedding = nn.Embedding(num_embeddings=50000, embedding_dim=768)
print(embedding(torch.tensor([0, 42])).shape)   # torch.Size([2, 768])

# nn.LayerNorm: after every Transformer sublayer, stabilizes training
layer_norm = nn.LayerNorm(768)

# nn.Dropout: randomly drops neurons in training; turns off automatically after model.eval()
dropout = nn.Dropout(p=0.1)
```

### Code example: two-layer MLP

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

## 4. A full training loop

The training loop is code AI Infra engineers touch every day. Even in distributed setups, the core loop is the same pattern.

### 4.1 Data loading: Dataset and DataLoader

- **Dataset**: defines "what is in the dataset" and "how to fetch one sample"
- **DataLoader**: defines "how to feed batches to the model" (batching, shuffling, multi-process loading)

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

> **AI Infra view**: Too few `num_workers` leaves the GPU waiting on data (data loading bottleneck). `pin_memory=True` lets transfers use async DMA and avoids a CPU copy.

### 4.2 Standard training flow

The five core steps each step: forward → loss → backward → optimizer.step → zero_grad.

### 4.3 Learning-rate scheduling

```python
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# common schedulers
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# call scheduler.step() after each epoch
```

### 4.4 Checkpoint save and restore

```python
import torch

# save (model + optimizer + training progress; resume needs all of them)
torch.save({
    'epoch': epoch, 'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss,
}, 'checkpoint.pt')

# load
ckpt = torch.load('checkpoint.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
```

### Code example: full MNIST training script

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

## 5. GPU training basics

### 5.1 Move model and data to the GPU

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Linear(100, 10).to(device)
x = torch.randn(32, 100).to(device)     # model and data must be on the same device
y = model(x)
```

### 5.2 Mixed-precision training intro

Core idea: use low precision (fp16/bf16) for forward and backward to speed up compute; keep high precision (fp32) for parameter updates so you do not lose accuracy.

> **Plain-language take**: Draft with a pencil (low precision, fast); finalize with a pen (high precision, accurate). Keep the critical parameter update in high precision so information is not lost; run the bulk of the math in low precision to go faster.

```python
import torch
import torch.nn as nn

device = torch.device('cuda')
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# BF16 mixed precision (recommended, Ampere+ GPUs, no GradScaler needed)
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# FP16 mixed precision (needs GradScaler to prevent gradient underflow)
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

### 5.3 Inspecting GPU memory

```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Peak:      {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(torch.cuda.memory_summary(abbreviated=True))

# analyze memory for a code region
torch.cuda.reset_peak_memory_stats()
# ... run the target code ...
print(f"Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

### Code example: mixed precision vs fp32 memory comparison

```python
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "GPU required"
device = torch.device('cuda')
criterion = nn.CrossEntropyLoss()

# Batch must be large enough for activations to dominate memory. Mixed precision does not
# shrink FP32 weights or optimizer state; if the batch is too small that fixed overhead
# fully hides the activation savings, and you can even see "negative savings"
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
    # Critical: truly free this round's model and optimizer state, or it stacks into the next peak
    del model, opt, loss
    torch.cuda.empty_cache()
    return peak

fp32_peak = step_peak(None)
bf16_peak = step_peak(torch.bfloat16)

print(f"FP32: {fp32_peak:.1f} MB | BF16: {bf16_peak:.1f} MB | Saved: {(1-bf16_peak/fp32_peak)*100:.1f}%")
# reference output: FP32: 1881.6 MB | BF16: 1246.6 MB | Saved: 33.7%
```

⚠️ **Two easy pitfalls**:

1. `torch.cuda.reset_peak_memory_stats()` only resets the peak counter; it does **not** free allocated memory. If the previous FP32 model and optimizer state are still alive when you measure BF16, they get counted into the BF16 peak and savings can look negative. You must `del` the objects and then `empty_cache()`.
2. Mixed precision mainly saves **activations** (and some intermediates). Weights and AdamW's first/second moments stay FP32. With a small model and small batch that fixed overhead dominates, and you may measure "0% savings." Scale batch/sequence length until activations dominate; then you should see roughly 30%–50% savings.

---

## 6. Performance profiling intro

Getting training code to run is only step one. Making it fast is the core AI Infra goal. `torch.profiler` tells you how long each op takes, how utilized the GPU is, and where it sits idle waiting on data.

### 6.1 Basic torch.profiler usage

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

### 6.2 How to read profiler output

| Column | Meaning | What to watch |
|------|------|--------|
| Name | Op name | Find the most expensive ops |
| CPU total | Total CPU-side time | Much larger than CUDA total → CPU bottleneck |
| CUDA total | Total GPU-side time | The core metric |
| # of Calls | Call count | Many tiny high-frequency ops can accumulate into a bottleneck |

Common performance patterns:

1. **CPU time much larger than CUDA time**: the GPU is waiting on the CPU (data preprocessing, Python overhead)
2. **Lots of tiny CUDA kernels**: launch overhead adds up; consider `torch.compile` or operator fusion
3. **Frequent CPU–GPU sync**: `.item()` and `print(tensor)` force sync and stall the GPU pipeline

### 6.3 Common performance issues

**Data loading bottleneck**: increase `num_workers`, use `prefetch_factor` to prefetch.

**CPU–GPU data movement**: avoid repeated `.cuda()` inside the loop; move to GPU once.

**Implicit sync**: `loss.item()` triggers CPU–GPU sync; record only every N steps.

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

### Code example: profile one training step

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

# warmup (avoid first-call init overhead)
for _ in range(3):
    criterion(model(data), labels).backward()
    optimizer.step(); optimizer.zero_grad()

# profiler analysis
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

From the output you can quickly tell whether backward is slower than forward (about 2× is normal), whether data movement takes too much time, and whether the optimizer step looks unusually expensive.

---

## Summary

| Layer | Content | Core skill |
|------|------|---------|
| Tensor | Data representation and ops | Shape ops, device management, dtype choice |
| autograd | Automatic differentiation | Computational graph, gradient compute, gradient zeroing |
| nn.Module | Model organization | Define models, manage parameters, common layers |
| Training loop | End-to-end training | DataLoader, training flow, checkpoint |
| GPU training | Hardware acceleration | Mixed precision, memory management |
| Profiling | Bottleneck finding | Using the profiler, recognizing common bottlenecks |

These are the base you need before distributed training, CUDA programming, and inference optimization. Once you have them, reading DeepSpeed or Megatron-LM code is less likely to stall on shaky PyTorch foundations.

---

## Self-check list

After this article you should be able to:

- Create Tensors of any shape with `torch.randn`, `torch.zeros`, and reshape with `view`, `permute`, `squeeze`, and similar ops
- Explain what `requires_grad=True` does, build a computational graph by hand, and call `backward()` for gradients
- Explain why every training step needs `optimizer.zero_grad()`, and how gradient accumulation works
- Subclass `nn.Module` for a custom model, and manage parameters with `parameters()` and `state_dict()`
- Write a full training loop: DataLoader → forward → loss → backward → optimizer.step → checkpoint
- Use `torch.autocast` for bf16/fp16 mixed-precision training, and explain what `GradScaler` is for
- Use `torch.profiler` on a training step, read the output, and spot data-loading and CPU–GPU transfer bottlenecks
- Explain the differences among fp32, fp16, and bf16 and how to choose

## References

- [PyTorch official docs](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [torch.profiler docs](https://pytorch.org/docs/stable/profiler.html)
- [Automatic Mixed Precision (AMP) docs](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
