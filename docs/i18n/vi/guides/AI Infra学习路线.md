---
title: "Lộ trình học AI Infra"
description: "Từ kiến thức nền, tối ưu kernel CUDA, huấn luyện phân tán đến triển khai suy luận: kiến thức từng lớp, tài liệu gợi ý và tiêu chí kiểm tra đo được cho AI Infra"
tags: ["AI Infra", "Lộ trình học", "CUDA", "Huấn luyện phân tán", "Tối ưu suy luận"]
sourceHash: "eec736e8e747b86c"
---

AI Infra (hạ tầng trí tuệ nhân tạo) giúp mô hình lớn huấn luyện được trên cụm GPU, chạy nhanh và đủ rẻ để dùng. Bài này đi từ nền tảng đến triển khai suy luận, lần lượt từng lớp: kiến thức cần nắm, tài liệu nên đọc, và tiêu chí kiểm tra đo được.

Các bài trên site này viết theo đúng lộ trình đó. Trong mục 「Tài liệu gợi ý」 của mỗi lớp đều có **bài trên site**, bấm vào là đọc luôn.

<!-- more -->

## 📑 Mục lục

- [Toàn cảnh: kiến trúc ba lớp](#️-toàn-cảnh-kiến-trúc-ba-lớp)
- [Lớp 0: Kiến thức nền](#-lớp-0-kiến-thức-nền)
- [Lớp 1: Lập trình CUDA và tối ưu toán tử](#-lớp-1-lập-trình-cuda-và-tối-ưu-toán-tử)
- [Lớp 2: Huấn luyện phân tán](#️-lớp-2-huấn-luyện-phân-tán)
- [Lớp 3: Suy luận và triển khai](#-lớp-3-suy-luận-và-triển-khai)
- [Hướng dẫn phá thế cho người mới](#-hướng-dẫn-phá-thế-cho-người-mới)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🗺️ Toàn cảnh: kiến trúc ba lớp

Bản chất của AI Infra là **"dùng kỹ thuật hệ thống để giải phóng sức tính của phần cứng"**. Từ dưới lên có thể chia thành ba lớp lõi, cộng thêm một lớp kiến thức nền:

| Lớp | Tên | Trọng tâm | Module tương ứng trên site |
|------|------|-----------|-------------|
| Lớp 0 | Kiến thức nền | Ngôn ngữ lập trình, toán, kiến trúc Transformer, PyTorch, phần cứng GPU, topology giao tiếp | [Module 1: Kiến thức nền](https://caomaolufei.github.io/AIInfraGuide/prerequisites/) |
| Lớp 1 | Lập trình CUDA và tối ưu toán tử | Kiến trúc GPU, hierarchy bộ nhớ, viết Kernel, FlashAttention, AI compiler | [Module 2: Lập trình CUDA và tối ưu toán tử](https://caomaolufei.github.io/AIInfraGuide/cuda/) |
| Lớp 2 | Huấn luyện phân tán | Data parallelism, ZeRO, 3D parallelism, context parallelism, MoE parallelism, mixed precision | [Module 3: Huấn luyện phân tán](https://caomaolufei.github.io/AIInfraGuide/distributed/) |
| Lớp 3 | Suy luận và triển khai | KV Cache, PagedAttention, quantization, Speculative Decoding, PD disaggregation, distributed inference | [Module 4: Tối ưu suy luận](https://caomaolufei.github.io/AIInfraGuide/inference/) |

Mọi tối ưu đều là đánh đổi trong tam giác bất khả thi **"tính toán, giao tiếp, VRAM"**: ZeRO đổi giao tiếp lấy VRAM; rematerialization (Activation Checkpointing) đổi tính toán lấy VRAM; quantization đổi độ chính xác lấy VRAM và bandwidth. Khi học hãy luôn tự hỏi: **kỹ thuật này hy sinh gì, đổi lấy gì?**

---

## 📖 Lớp 0: Kiến thức nền

AI Infra không phải lĩnh vực học từ số không—nó dựng trên năng lực lập trình, trực giác toán và hiểu mô hình. Nếu ba lớp sau là xây nhà, lớp này là móng: móng yếu thì càng lên cao càng lung. Tin vui là bạn không cần thành chuyên gia mọi hướng; đạt mức "đủ dùng" là leo tiếp được.

### 0.1 Kiến thức cần nắm

**Ngôn ngữ lập trình**

- **Python**: ngôn ngữ chung của hệ sinh thái AI. Cần dùng thành thạo, không chỉ "biết viết". Trọng tâm: OOP, decorator, generator, multiprocessing/multithreading, performance profiling
- **C/C++**: ngôn ngữ host của CUDA. Không cần thành thạo template metaprogramming, nhưng phải hiểu pointer, quản lý bộ nhớ, quá trình compile/link. Đọc được code dự án C++, viết và biên dịch chạy được hàm C++ đơn giản là đủ
- **Linux cơ bản**: thao tác dòng lệnh, Shell script, quản lý process, cấu hình biến môi trường. Gần như toàn bộ phát triển và triển khai AI Infra diễn ra trên Linux

**Nền tảng toán**

- **Đại số tuyến tính**: nhân ma trận, transpose, phép toán ma trận theo khối, khái niệm cơ bản về phân tích trị riêng. Hiểu "vì sao tính toán lõi của Transformer toàn là nhân ma trận" cần nền này. Không cần chứng minh định lý, nhưng phải có trực giác về đổi chiều—thấy (B, S, H) x (H, V) là biết ngay kết quả là (B, S, V)
- **Xác suất thống kê cơ bản**: phân phối xác suất, kỳ vọng, phương sai, cách Softmax được giải thích theo xác suất, ý nghĩa của cross-entropy loss. Chứng minh tính đúng của quantization và Speculative Decoding đều dựa vào xác suất
- **Giải tích** (biết qua là đủ): quy tắc chuỗi, ý nghĩa của gradient. Hiểu vì sao backpropagation hoạt động, vì sao gradient overflow trong mixed precision, cần nền này

**Kiến trúc Transformer**

Đây là "đế chung" của thời đại mô hình lớn. Mọi lớp sau đều tối ưu quanh nó—lớp CUDA tối ưu toán tử của nó, lớp phân tán cắt tham số của nó, lớp suy luận tăng tốc sinh token. Phải hiểu:

- **Cơ chế Self-Attention**: ý nghĩa và cách tính Q, K, V (QK^T → scale → softmax → PV); hiểu vì sao độ phức tạp Attention là O(N^2)—đó là tiền đề để sau này hiểu tối ưu FlashAttention
- **Feed-forward network (FFN)**: Transformer gốc là hai lớp tuyến tính + hàm kích hoạt; các mô hình mainstream như LLaMA chuyển sang SwiGLU (ba ma trận trọng số). FFN chiếm phần lớn tham số mô hình
- **Positional encoding**: Sinusoidal, RoPE, v.v.; hiểu vì sao Transformer cần thêm thông tin vị trí
- **Normalization**: LayerNorm và RMSNorm, Pre-Norm vs Post-Norm, vì sao mô hình lớn phổ biến dùng Pre-Norm
- **Toàn bộ forward**: bắt đầu từ token embedding, lần theo dữ liệu qua một Transformer Block, nói rõ chiều input/output từng bước. Paper gốc là Post-Norm (Attention → Add & Norm → FFN → Add & Norm); LLM mainstream là Pre-Norm (Norm → Attention → Add → Norm → FFN → Add)

**Framework PyTorch**

- **Khái niệm lõi**: thao tác Tensor, autograd, cách tổ chức Module / Parameter
- **Vòng huấn luyện**: luồng chuẩn DataLoader → forward → loss → backward → optimizer.step
- **Lưu và nạp mô hình**: state_dict, checkpoint
- **Debug cơ bản**: dùng `torch.cuda.memory_summary()` xem VRAM, dùng `torch.profiler` phân tích hiệu năng đơn giản

**Topology giao tiếp**

Huấn luyện phân tán và suy luận đa GPU đều cần giao tiếp hiệu quả giữa các card và giữa các máy. Phần này là tiền đề để hiểu chiến lược song song sau—nếu không biết bandwidth NVLink và PCIe chênh một bậc độ lớn, sẽ không hiểu "vì sao tensor parallelism không thể xuyên máy".

- **Giao tiếp trong một máy**: bandwidth và topology của NVLink / NVSwitch. NVLink là kênh tốc độ cao dành riêng giữa các GPU trong cùng chassis—hãy nghĩ như băng tải riêng trong khu nhà máy
- **Giao tiếp đa máy**: mạng InfiniBand (IB), giao thức RoCE. IB giống đường sắt cao tốc liên thành phố: bandwidth thấp hơn NVLink nhưng nối được các node xa
- **Primitive giao tiếp tập thể**: ý nghĩa và chi phí của AllReduce, AllGather, ReduceScatter—các thao tác thường xuyên nhất trong huấn luyện phân tán; công thức lượng giao tiếp của chúng là nền để phân tích chi phí chiến lược song song
- **NCCL**: cách dùng cơ bản và tinh chỉnh thư viện giao tiếp tập thể của NVIDIA

### 0.2 Tài liệu gợi ý

**Bài trên site:** [Nền tảng ngôn ngữ lập trình](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/第1章-编程语言基础/) · [Nền tảng toán](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/第2章-数学基础/) · [Nhập môn nhanh kiến trúc Transformer](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/transformer架构快速入门/) (và [chuỗi bài Chương 3](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/第3章-transformer架构详解/)) · [Nhập môn nhanh framework PyTorch](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/pyroch/pytorch框架入门/) · [Kiến thức cơ bản về GPU](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics/) · [Mạng giao tiếp cụm và NCCL](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/collective-communication-primer/)

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Paper | Attention Is All You Need | Paper gốc Transformer, bắt buộc đọc |
| Tutorial | The Illustrated Transformer (Jay Alammar) | Transformer bằng hình, tốt để dựng trực giác |
| Walkthrough | Linlang Amu: minh họa LLM inference | Kiến trúc LLM chi tiết; nên đọc kèm paper |
| Tutorial | Tutorial chính thức PyTorch (60 Minute Blitz) | Nhập môn nhanh PyTorch |
| Video | 3Blue1Brown: Essence of linear algebra | Trực giác hình học cho đại số tuyến tính; hiệu quả hơn sách giáo khoa |
| Course | MIT The Missing Semester | Shell, Vim, Git, debug và profiling—công lực dòng lệnh |
| Video | Andrej Karpathy: Let's build GPT from scratch | Viết GPT từ đầu, đi qua từng module Transformer |
| Official docs | Tài liệu NVIDIA NCCL | Primitive giao tiếp tập thể và lập trình đa GPU |
| Tutorial | NVIDIA Deep Learning Performance Guide | Phương pháp tìm bottleneck phần cứng |

### 0.3 Tiêu chí kiểm tra

Lớp này không cần đào sâu, nhưng phải vượt các ngưỡng "đủ dùng" sau:

- **Viết Transformer lên bảng**: không nhìn tài liệu, vẽ đủ cấu trúc một Pre-Norm Transformer Decoder Block (RMSNorm → Masked Self-Attention → cộng residual → RMSNorm → FFN → cộng residual), ghi chiều input/output từng bước (ví dụ (B, S, D) → (B, S, D)), và giải thích Q, K, V được chiếu tuyến tính từ input như thế nào
- **Suy ra số chiều**: cho cấu hình mô hình 7B (hidden_dim=4096, num_heads=32, num_layers=32, chiều trung gian FFN 11008 và dùng SwiGLU, vocab_size=32000), tự tính gần đúng tổng tham số (trọng số Attention + FFN + Embedding), sai số không quá 20% (đáp án tham chiếu khoảng 6.7B)
- **Script huấn luyện PyTorch**: tự viết một vòng huấn luyện PyTorch đầy đủ (không nhờ Trainer), gồm DataLoader, forward, tính loss, backward, optimizer step, lịch learning rate, lưu checkpoint, và chạy được trên GPU
- **Đọc/viết C++ cơ bản**: đọc được phía host của một CUDA kernel đơn giản (malloc, memcpy, kernel launch, free), hiểu luồng chuyển dữ liệu giữa CPU và GPU
- **Linux hàng ngày**: trên server tự làm được: SSH đăng nhập, quản lý session bằng tmux, quản lý môi trường conda/pip, xem trạng thái GPU (`nvidia-smi`), quản lý code bằng git, viết bash script đơn giản để submit job hàng loạt
- **Nhận topology**: trên máy 8 GPU chạy `nvidia-smi topo -m`, đọc được các dấu NV12, SYS, NODE trong ma trận, phán đoán cặp nào đi NVLink, cặp nào đi PCIe
- **Trực giác giao tiếp tập thể**: vẽ sơ đồ luồng dữ liệu của AllReduce, AllGather, ReduceScatter, nói rõ công thức lượng giao tiếp từng phép (ví dụ Ring AllReduce ≈ 2(N-1)/N × kích thước dữ liệu), hiểu vì sao chênh bandwidth giữa NVLink và IB quyết định chiến lược song song nào có thể xuyên máy

---

## 💻 Lớp 1: Lập trình CUDA và tối ưu toán tử

Lớp này là cầu nối phần cứng và phần mềm: biến phép toán mức cao thành lệnh máy mà GPU chạy hiệu quả nhất. Trước khi viết kernel, phải hiểu kiến trúc phần cứng GPU—như viết sổ quy trình cho công nhân thì phải biết nhà máy có bao nhiêu người, xưởng bố trí thế nào, kệ hàng ở đâu.

### 1.1 Kiến thức cần nắm

**Kiến trúc phần cứng GPU**

Hiểu phần cứng GPU là tiền đề viết CUDA kernel tốt. Hãy hình dung một GPU như **nhà máy siêu lớn với hàng nghìn công nhân đơn giản**—mỗi công nhân (CUDA Core) chỉ làm cộng trừ nhân chia cơ bản, nhưng đông người cùng làm thì throughput vượt xa xưởng nhỏ chỉ có vài kỹ sư cao cấp (CPU core).

- **Kiến trúc lõi GPU**: SM (Streaming Multiprocessor), Tensor Core, CUDA Core—khác nhau và phối hợp thế nào
- **So sánh thông số GPU mainstream**: A100 / H100 / H200 / B200—compute, memory bandwidth, dung lượng HBM, và các định dạng độ chính xác thấp mới theo từng thế hệ (FP8 của Hopper, FP4 của Blackwell)
- **Memory Wall**: vì sao bottleneck bandwidth thường chết người hơn bottleneck compute—như đầu bếp siêu nhanh, dao và lửa đều ổn, nhưng băng tải nguyên liệu quá chậm nên hầu hết thời gian ngồi đợi đồ lên bàn
- **Hierarchy bộ nhớ**: register > shared memory > L1/L2 Cache > HBM > host memory. Cốt lõi khi viết kernel là tối đa hóa tái sử dụng dữ liệu trong hierarchy này, giảm truy cập bộ nhớ chậm

**Cơ bản lập trình CUDA**

- Mô hình lập trình: hierarchy Grid / Block / Thread, tính chỉ số thread
- Mô hình bộ nhớ: global memory, shared memory, register, constant memory—đặc tính và cách dùng
- Khái niệm then chốt: Warp (32 thread là đơn vị schedule nhỏ nhất; GPU phát lệnh theo Warp, như quân đội hành động theo "tiểu đội"), Bank Conflict, Coalesced Access, Occupancy
- Trực giác lõi: "kiểu truy cập bộ nhớ quyết định tốc độ chạy"

**Cài đặt và tối ưu toán tử thường gặp**

- Reduce: nhiều cách parallel reduction và tối ưu (Warp Shuffle, reduce nhiều cấp)
- GEMM: blocking, vectorization, Shared Memory Tiling, tận dụng Tensor Core
- Softmax: Online normalizer calculation, Softmax kernel hiệu quả
- Operator fusion: gộp nhiều toán tử nhỏ thành một kernel, giảm đọc/ghi global memory

**Toán tử Attention**

- FlashAttention V1/V2: Attention chính xác memory-aware, dùng tiling giảm truy cập HBM—như xếp puzzle trên bàn lớn bằng cách mỗi lần chỉ mang một mảnh nhỏ về tay ghép xong rồi mới lấy mảnh tiếp, thay vì đổ hết mảnh ra chiếm cả bàn
- FlashAttention-3: đẩy utilization cao hơn nữa trên Hopper
- Flash-Decoding / FlashDecoding++: tăng tốc Attention cho giai đoạn Decode
- FlashInfer: Attention engine tùy biến được, format kết hợp được và lưu trữ KV dị thể, hướng Serving
- Attention Kernel cho KV phân trang: CUDA Kernel `paged_attention` sớm của vLLM, và cách vLLM V1 đọc trực tiếp paged KV Cache qua backend FlashAttention / FlashInfer

**AI compiler**

- Triton: ngôn ngữ lập trình GPU mã nguồn mở của OpenAI, hạ thấp ngưỡng viết toán tử hiệu quả
- TVM / XLA: tối ưu đồ thị tính toán và sinh code
- `torch.compile`: chế độ compile của PyTorch 2.x; hiểu Graph Break và lợi ích hiệu năng

### 1.2 Tài liệu gợi ý

**Bài trên site:** [Hướng dẫn nhập môn nhanh CUDA](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/cuda编程入门指南/) · [Nhập môn lập trình CUDA (Chương 1)](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第1章-cuda编程入门/) · [Cơ bản tối ưu hiệu năng CUDA (Chương 2)](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第2章-cuda性能优化基础/) · [Tối ưu toán tử Reduce](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/31-cuda-reduce算子优化/) · [Tối ưu hiệu năng toán tử GEMM](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化/) · [Tối ưu Softmax naive](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/51-cuda-softmax朴素实现优化/) · [Online Softmax](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现/) · [FlashAttention V1](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解/) · [FlashAttention V2](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解/) · [Mô hình lập trình Triton](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/71-triton编程框架/)

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Official docs | Whitepaper kiến trúc GPU NVIDIA (Ampere / Hopper / Blackwell) | Hiểu thiết kế SM, Tensor Core, HBM |
| Intro tutorial | Xiaoxiao Jiang: tutorial CUDA cực ngắn | Nhập môn CUDA từ zero |
| Official docs | NVIDIA CUDA Programming Guide | Tài liệu tham chiếu chính thức CUDA |
| Reduce | PeakCrosser: tối ưu toán tử CUDA Reduce | Tóm tắt chi tiết cài đặt và tối ưu Reduce |
| Practice platform | [CUDA hand-written operators · interview practice](https://kkcocoa.github.io/cuda-operator-interview/) | Bài tập viết tay theo cấp độ phủ memory access, Reduce, Softmax, Attention, GEMM; có gợi ý, checklist tự kiểm và editor trong trình duyệt |
| GEMM | Meng Yuan: từ zero đến tối ưu CUDA GEMM | Tutorial GEMM từ blocking cơ bản đến tối ưu cực hạn |
| GEMM | MegEngine Bot: hướng dẫn tối ưu nhân ma trận CUDA tối thượng | Tham chiếu tối ưu GEMM có hệ thống |
| Softmax | Online normalizer calculation for softmax | Paper gốc Online Softmax (NVIDIA); nền thuật toán của FlashAttention |
| Softmax | OneFlow: cách cài Softmax CUDA kernel hiệu quả | Thực hành kỹ thuật Softmax kernel |
| Operator fusion | Cheng Cheng: OneFlow trở thành framework DL nhanh nhất thế giới như thế nào | Tư duy và phương pháp fusion toán tử |
| Attention | FlashAttention V1 Paper | Paper cột mốc Memory-aware Attention |
| Attention | FlashAttention V2 Paper | Parallelism và tiling tốt hơn |
| Attention | FlashAttention-3 Paper | Tối ưu thêm trên Hopper |
| Attention | Báo cáo kỹ thuật Flash-Decoding (Stanford CRFM) | Tăng tốc Attention giai đoạn Decode |
| Attention | FlashInfer Paper + Repo | Attention engine kết hợp được |
| Walkthrough | Meng Yuan: minh họa series FlashAttention V1/V2 | Walkthrough bằng hình cho người mới |
| Walkthrough | Fang Jiarui: hiểu cài đặt CUDA PagedAttention từ nông đến sâu | Walkthrough kernel PagedAttention của vLLM |
| Compiler | Tutorial chính thức Triton | Paradigm mới cho lập trình GPU |
| Compiler | PyTorch profiling torch.compile | Bắt Graph Break, lợi và hại của compile |
| Tool | Nsight Systems User Guide | Phân tích tương tác CPU–GPU; xem host có kéo chân không |
| Tool | Nsight Compute Profiling Guide | Khoan xuống mức kernel, tìm bottleneck SM / Memory / Tensor Core |

### 1.3 Tiêu chí kiểm tra

Làm tay là tiêu chuẩn duy nhất của lớp này; nói suông không tính:

- **Trực giác thông số phần cứng**: cầm một H100, không tra tài liệu nói được bậc độ lớn của dung lượng HBM (80GB), bandwidth HBM (~3.35TB/s), kích thước L2 (50MB), trần shared memory (228KB/SM), và giải thích vì sao "memory bandwidth" thường thành bottleneck trước "compute"
- **Ước lượng bandwidth**: cho một AllReduce với lượng dữ liệu (ví dụ 2GB gradient), ước được chênh thời gian lý thuyết trên NVLink 4.0 (450GB/s một chiều mỗi GPU, 900GB/s hai chiều) vs PCIe Gen5 x16 (64GB/s một chiều)
- **Bộ ba Reduce**: bắt đầu từ atomic add trên global memory, viết Reduce Sum kernel; rồi shared memory + tree reduction để bỏ atomic; cuối cùng Warp Shuffle để bỏ shared memory; chạy cả ba trên Nsight Compute so sánh throughput, nói rõ từng bước tối ưu tiết kiệm ở đâu
- **Trực giác Bank Conflict**: tự viết kernel transpose ma trận 32x32, trước tiên bản có 32-way bank conflict, rồi thêm một cột padding để hết conflict, dùng panel Shared Memory của Nsight Compute xác nhận conflict từ vài chục xuống 0
- **GEMM tiling**: cài GEMM dựa trên Shared Memory Tiling, trên ma trận 1024x1024 so với cuBLAS đạt ≥50% là đạt—trong quá trình này bạn mới thật sự hiểu "vì sao kiểu truy cập quyết định tất cả"
- **Suy ra FlashAttention trên bảng**: không nhìn paper, vẽ được quá trình tiling của FlashAttention—lấy V1 làm ví dụ, vòng ngoài duyệt block KV, vòng trong duyệt block Q (V2 đảo thành ngoài Q, trong KV); mỗi tile hoàn thành QK^T → scale → mask → softmax → PV trong SRAM, dùng online softmax để tránh duyệt hai lần; then chốt là nói rõ vì sao VRAM phụ thêm từ O(N^2) xuống O(N) (không còn materialize ma trận N×N), và lượng truy cập HBM vì sao từ Θ(N^2) xuống Θ(N^2·d^2/M) (M là kích thước SRAM; d^2 ≪ M, vẫn tăng theo N^2 nhưng hằng số nhỏ hơn nhiều)
- **Triton thực hành**: dùng Triton cài fused Softmax kernel (theo tutorial chính thức), so đúng đắn và hiệu năng với bản native PyTorch, cảm nhận mô hình block-level của Triton khác mô hình thread-level của CUDA thế nào
- **Profiling thực chiến**: dùng Nsight Systems bắt trace một training iteration, chỉ ra GPU idle gap đến từ tiền xử lý CPU, chờ giao tiếp, hay kernel launch overhead; mở báo cáo kernel trên Nsight Compute, đọc panel SOL (Speed of Light) để phán memory bound hay compute bound

---

## 🏋️ Lớp 2: Huấn luyện phân tán

Khi tham số vượt giới hạn VRAM một card, huấn luyện phân tán là đường buộc phải đi, cũng là công việc lõi của đội huấn luyện mô hình lớn. Ví dụ: huấn luyện mô hình trăm tỷ tham số giống **chép tay một bộ bách khoa hàng vạn trang**—một người chép đến già cũng không xong. Data parallelism là photocopy nhiều cuốn, mỗi người chép chương khác rồi gộp; tensor parallelism là cắt mỗi trang thành vài cột, mỗi người chỉ chép cột của mình; pipeline parallelism là người thứ nhất chép xong chương một thì chuyển cho người hai, rồi tự chép mẻ tiếp. Cắt thế nào, truyền thế nào, gộp thế nào—đó là bài toán lõi.

### 2.1 Kiến thức cần nắm

**Tiến hóa kiến trúc mô hình (góc nhìn phân tán)**

Lớp 0 đã nói cấu trúc Transformer chuẩn. Ở đây tập trung các biến thể ảnh hưởng trực tiếp đến chiến lược cắt phân tán và chi phí giao tiếp:

- Biến thể Attention: MHA → MQA → GQA → MLA. MQA/GQA giảm kích thước KV Cache bằng cách chia sẻ KV Head; MLA (DeepSeek V2) nén low-rank thêm nữa để hạ VRAM của KV—các biến thể này ảnh hưởng trực tiếp cách cắt TP và kế hoạch VRAM suy luận
- Biến thể FFN: Mixture of Experts MoE (DeepSeekMoE). Đặc tính kích hoạt thưa của MoE biến chiến lược song song từ "cắt ma trận" thành "chia expert", thêm chiều Expert Parallelism

**Optimizer**

Optimizer là "bánh lái" của huấn luyện—sau khi có gradient, cập nhật tham số ra sao đều nhờ nó. Hiểu trạng thái nội bộ của optimizer là tiền đề trực tiếp để hiểu tối ưu VRAM ZeRO và mixed precision: ZeRO-1 cắt đúng optimizer state; trong mixed precision, trọng số chính FP32 cũng nằm trong optimizer.

- **SGD**: optimizer tối giản, mỗi tham số chỉ cần một gradient, không có state phụ. Như bước theo hướng dốc nhất—đơn giản nhưng dễ dao động trong thung lũng
- **Momentum SGD**: thêm momentum (moment bậc một) lên SGD, mỗi tham số thêm một buffer momentum. Như cho quả bóng xuống dốc thêm quán tính, khó bị hố nhỏ cản
- **Adam / AdamW**: chuẩn thực tế của huấn luyện mô hình lớn. Mỗi tham số giữ **hai state**—moment bậc một (EMA của gradient) và moment bậc hai (EMA của bình phương gradient), vừa nhớ "đi đâu" vừa nhớ "đường gập ghềnh ra sao". AdamW sửa cách Adam làm weight decay, hiện là lựa chọn mainstream nhất
- **Chi phí VRAM của optimizer state**: then chốt để hiểu ZeRO. Với AdamW + mixed precision, mỗi tham số cần thêm: bản sao tham số FP32 (4B) + moment bậc một FP32 (4B) + moment bậc hai FP32 (4B) = 12 byte/tham số. Optimizer state của mô hình 7B chiếm ~84GB, vượt xa ~14GB của tham số FP16—đó là lý do ZeRO bắt đầu từ việc cắt optimizer state
- **Optimizer large-batch (LAMB / LARS)**: trong huấn luyện phân tán, data parallelism phóng đại tuyến tính effective Batch Size. Khi Batch Size từ vài trăm lên vài chục nghìn, learning rate của Adam thường khó chỉnh, dễ phân kỳ. LAMB (Layer-wise Adaptive Moments) tính trust-region scale độc lập theo từng lớp, giúp large-batch vẫn hội tụ ổn định

**Data parallelism**

- DP (DataParallel): data parallelism cơ bản, một process nhiều GPU
- DDP (DistributedDataParallel): data parallelism đa process; hiểu AllReduce đồng bộ gradient
- FSDP (Fully Sharded Data Parallel): bản ZeRO-3 native của PyTorch

**Model parallelism (3D parallelism)**

- Tensor parallelism (TP): cắt nhân ma trận theo chiều nhất định ra nhiều GPU; giao tiếp dày, thường giới hạn trong một máy
- Pipeline parallelism (PP): đặt các lớp khác nhau lên GPU khác nhau, truyền dữ liệu như dây chuyền. Vấn đề lõi là pipeline bubble; GPipe, 1F1B, Interleaved 1F1B đều nhằm nén bubble
- Sequence parallelism (SP): cắt theo chiều sequence, phối hợp với TP để giảm VRAM activation

**Song song chuỗi dài và MoE**

- Context parallelism (CP): tính toán Attention và VRAM activation tăng theo bình phương độ dài sequence; Ring Attention, DeepSpeed-Ulysses, Megatron Context Parallel cắt một chuỗi dài ra nhiều GPU để tính
- Expert parallelism (EP): mô hình MoE đặt expert khác nhau lên GPU khác nhau; Token theo kết quả Router đi All-to-All tới expert tương ứng; cân bằng tải là khó chính. Thực tế thường kết hợp EP với DP, TP, PP

**Tối ưu VRAM**

- Series ZeRO (DeepSpeed): tách trạng thái huấn luyện chia cho từng card—như nhà thuê chung mỗi người chỉ giữ đồ của mình, cần thì mượn nhau, nhờ đó chừa thêm chỗ:
  - ZeRO-1: cắt optimizer state
  - ZeRO-2: cắt optimizer state + gradient
  - ZeRO-3: cắt optimizer state + gradient + tham số (đổi giao tiếp lấy VRAM)
- Mixed precision: huấn luyện FP16 / BF16 / FP8, giảm VRAM và chi phí tính. Như hàng ngày dùng giấy nháp (độ chính xác thấp) để nhanh, chỉ ở bước then chốt mới dùng giấy thi chính thức (độ chính xác cao) để đảm bảo đúng
- Gradient accumulation: dưới VRAM hạn chế vẫn mô phỏng effective Batch Size lớn hơn
- Activation Checkpointing (rematerialization): đổi tính toán lấy VRAM—chỉ giữ một phần activation, cần thì tính lại. Như thi không giữ hết giấy nháp trên bàn, chỉ nhớ kết quả trung gian quan trọng, cần thì suy lại

**Framework huấn luyện**

- Megatron-LM: hiện thực mẫu mực của TP và PP
- DeepSpeed: hiện thực lõi của series ZeRO, bộ công cụ tối ưu huấn luyện phong phú
- PyTorch FSDP: đường huấn luyện phân tán native

### 2.2 Tài liệu gợi ý

**Bài trên site:** [Tổng luận huấn luyện phân tán](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/11-分布式训练总论/) · [Dựng môi trường và khởi động phân tán](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/12-环境搭建与分布式启动/) · [Primitive giao tiếp tập thể chi tiết](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/21-集合通信原语详解/) · [Optimizer và phân tích chi phí VRAM](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/31-优化器原理与显存开销分析/) · [Data parallelism chi tiết](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/41-数据并行详解/) · [Data parallelism PyTorch từ nguyên lý đến thực chiến](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/42-pytorch-数据并行从原理到实战/)

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Paper | Báo cáo kỹ thuật DeepSeek V2 | Cơ chế Attention MLA |
| Paper | DeepSeekMoE Paper | Thiết kế kiến trúc MoE |
| Tutorial | Mixture of Experts (MoE) giải thích | Nhập môn MoE |
| Walkthrough | Su Jianlin: từ MHA, MQA, GQA đến MLA | Tiến hóa biến thể Attention |
| Paper | Megatron-LM Paper | Paper cột mốc về nguyên lý TP và PP |
| Paper | ZeRO Paper (DeepSpeed) | Phương pháp lõi tối ưu VRAM |
| Paper | Ring Attention Paper | Context parallelism bằng cách truyền vòng khối KV |
| Paper | DeepSpeed-Ulysses Paper | Context parallelism dùng All-to-All chuyển giữa chiều sequence và chiều head |
| Docs | Tài liệu chính thức DeepSpeed | Cấu hình và dùng ZeRO |
| Docs | Tutorial PyTorch DDP / FSDP | Nhập môn huấn luyện phân tán native |

### 2.3 Tiêu chí kiểm tra

Lõi kiểm tra lớp này là **tính sổ được và chạy code được**:

- **Sổ VRAM**: cầm mô hình 7B (ví dụ LLaMA-2-7B), không tra tài liệu ước miệng được tham số FP16 ~14GB, Adam optimizer state ~84GB (bản sao tham số FP32 + moment bậc một/hai mỗi cái 28GB), rồi phán một card 80GB có nhét hết trạng thái huấn luyện không, có bắt buộc lên ZeRO không
- **Tháo ZeRO**: ai hỏi "ZeRO-2 và ZeRO-3 khác nhau chỗ nào", bạn nói một hơi: ZeRO-2 ReduceScatter gradient lúc backward (mỗi card chỉ giữ phần mình phụ trách), cập nhật xong rồi AllGather; tham số mỗi card vẫn giữ đủ một bản, lượng giao tiếp 2Ψ ngang DDP; ZeRO-3 cắt luôn tham số, forward/backward đều phải AllGather lấy tham số rồi dùng xong bỏ, lượng giao tiếp 3Ψ (khoảng 1.5× DDP), nhưng VRAM trạng thái mô hình mỗi card xuống còn 1/N
- **Chuyển DDP**: cầm script huấn luyện PyTorch một card, trong 30 phút đổi thành DDP đa GPU và chạy được—gồm `init_process_group`, `DistributedSampler`, wrap model, đồng bộ gradient; không cần lê tài liệu quá nhiều
- **Topology 3D parallelism**: cho cụm 64 GPU (8 node × 8 GPU), thiết kế được phương án TP=8 (trong máy), PP=4 (xuyên máy), DP=2, vẽ topology ghi rõ giao tiếp nào đi NVLink, nào đi IB, và giải thích vì sao TP không thể xuyên máy (bandwidth không đủ)
- **Nguyên lý mixed precision**: trả lời được "BF16 và FP16 đều 16 bit, vì sao mô hình lớn thích BF16 hơn"—vì BF16 có số bit mũ rộng hơn (8 vs 5), dải động gần FP32, khó overflow/underflow hơn, hầu hết trường hợp không cần Loss Scaling

---

## 🚀 Lớp 3: Suy luận và triển khai

Huấn luyện mới chỉ là bước đầu. Làm sao để mô hình phục vụ người dùng nhanh và rẻ mới là điều ngành quan tâm nhất. Nếu huấn luyện là "dạy mô hình kiến thức", suy luận là "cho mô hình vào phòng thi"—trên phòng thi quan trọng nhất là **tốc độ trả lời** và **phục vụ được bao nhiêu thí sinh cùng lúc**, và ghi kết quả trung gian đã tính lên giấy nháp (KV Cache) tránh tính lại, tăng tốc rõ.

### 3.1 Cơ bản suy luận LLM

**Kiến thức cần nắm**

- Hai giai đoạn suy luận LLM: Prefill (xử lý input) và Decode (sinh từng token)
- KV Cache: "sát thủ VRAM" của sinh tự hồi quy; hiểu vòng đời và vấn đề phân mảnh. KV Cache như giấy nháp lúc thi—ghi kết quả trung gian đã tính để mỗi token mới không phải tính lại từ đầu, nhưng giấy nháp nhiều quá cũng chiếm hết bàn
- Chỉ số hiệu năng then chốt: TTFT (độ trễ token đầu), TPOT (độ trễ mỗi token), throughput (token/s), đuôi trễ P50/P95

**Tài liệu gợi ý**

**Bài trên site:** [Cơ bản suy luận LLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础/) · [Nhập môn nhanh vLLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/12-vllm快速入门/) · [Từ Transformer đến sinh tự hồi quy LLM](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解/)

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Walkthrough | Linlang Amu: minh họa LLM inference—KV Cache | Giải thích sâu nguyên lý KV Cache |
| Survey | Towards Efficient Generative LLM Serving: A Survey | Survey suy luận LLM của CMU (thuật toán + hệ thống) |

**Tiêu chí kiểm tra**

Đặt mình vào vị trí kỹ sư vừa nhận yêu cầu suy luận, trả lời:

- **Trực giác hai giai đoạn**: đồng nghiệp hỏi "vì sao Prefill nhanh mà Decode chậm", bạn trả lời ngay: Prefill xử lý cả prompt một lần, ma trận lớn, utilization compute cao (compute bound); Decode mỗi bước chỉ sinh một token, ma trận thoái hóa thành vector, phần lớn thời gian chờ VRAM chuyển KV Cache (memory bound)
- **Tính sổ KV Cache**: cho LLaMA-2-7B (32 lớp, 32 head, head_dim=128), độ dài ngữ cảnh 4096, batch_size=16, lưu FP16, tự tính tổng KV Cache = 2 × 32 × 32 × 128 × 4096 × 16 × 2B ≈ 32GB, rồi phán card 80GB còn bao nhiêu chỗ cho tham số mô hình và buffer tạm
- **Tháo pipeline**: tách một request suy luận từ đầu đến cuối thành tokenize → prefill (GEMM dày) → decode loop (sinh từng token, memory bound) → sampling (Top-p/Top-k) → detokenize, chỉ ra dưới concurrent cao đoạn nào dễ thành bottleneck nhất (thường là bandwidth KV Cache giai Decode)

### 3.2 Inference engine

**Kiến thức cần nắm**

- PagedAttention: ý tưởng phân trang bộ nhớ ảo của vLLM để quản lý KV Cache, giải quyết phân mảnh—như OS cắt bộ nhớ thành trang kích thước cố định, không còn đòi một khối liên tục lớn
- Continuous Batching: ghép batch động, request đến là xử lý; khác static batching truyền thống. Cách cũ như xe bus du lịch—đủ người mới chạy, người đến sớm ngồi chờ; Continuous Batching giống ghép xe công nghệ, đến là ghép, có người xuống thì bổ sung khách mới ngay
- Prefix Cache / RadixAttention: tái sử dụng KV Cache đã tính, tối ưu cảnh prefix lặp
- Chunked Prefill: xử lý prompt dài theo khối, giảm nhiễu của Prefill lên Decode

**Framework suy luận mainstream**

| Framework | Đặc tính lõi | Phù hợp |
|------|---------|---------|
| vLLM | PagedAttention, Continuous Batching, Prefix Cache | Serving suy luận chung; cộng đồng sôi |
| SGLang | RadixAttention, tăng tốc structured output bằng cFSM | Agent phức tạp, sinh đa vòng, structured output |
| TensorRT-LLM | Paged KV, Inflight Batching, tối ưu phần cứng sâu | Đuổi hiệu năng cực hạn; hệ sinh thái NVIDIA |

**Tài liệu gợi ý**

**Bài trên site:** [PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) · [Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) · [Prefix Cache và RadixAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) · [Chunked Prefill và lịch trình thống nhất](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) · [Backend Attention và tối ưu đồ thị](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) · [Đọc source nano-vllm (Chương 3)](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/第3章-nano-vllm源码解读/)

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Paper | vLLM Paper (PagedAttention) | Paper cột mốc inference engine |
| Source | vLLM GitHub Repo | Bộ tính năng thực chiến đầy đủ |
| Docs | Tài liệu chính thức vLLM | Triển khai và cấu hình |
| Walkthrough | Meng Yuan: series phân tích source vLLM (kiến trúc / scheduler / BlockManager) | Hiểu sâu cơ chế nội bộ vLLM |
| Paper | SGLang Paper | RadixAttention + ngôn ngữ frontend |
| Source | SGLang GitHub Repo | Hiện thực kỹ thuật |
| Docs | Tài liệu chính thức TensorRT-LLM | Hướng triển khai sản xuất |
| Paper | Orca Paper | Paper gốc Continuous Batching |
| Walkthrough | Chi Guodong Bu Tu Guodong Pi: Continuous Batching | Walkthrough Continuous Batching tiếng Trung |
| Walkthrough | DefTruth: minh họa nguyên lý Prefix Cache của vLLM | Prefix Cache giải thích dài |

**Tiêu chí kiểm tra**

Inference engine là nơi kỹ thuật落地; tiêu chí nghiêng về "dùng được, chọn được, debug được":

- **Triển khai end-to-end**: cầm mô hình 7B, lần lượt dùng vLLM và SGLang triển khai API tương thích OpenAI, cùng một script press (cố định concurrent, độ dài input/output) chạy ra bảng so sánh TTFT, TPOT, throughput
- **Nguyên lý Continuous Batching**: giải thích cho đồng nghiệp không kỹ thuật—Static Batching phải đợi cả batch sinh xong mới nhận request mới, request ngắn bị request dài kéo; Continuous Batching cho request đã xong thoát bất cứ lúc nào, request mới chen vào bất cứ lúc nào, GPU utilization có thể từ 30% lên 80%+
- **Toàn chuỗi quản lý KV Cache**: từ cấp phát, dùng, phân mảnh, đến ánh xạ trang ảo/trang vật lý của PagedAttention, rồi Prefix Cache tái sử dụng khối KV sẵn có bằng hash match—vẽ được luồng dữ liệu đầy đủ
- **Quyết định chọn framework**: sếp hỏi "ta nên dùng framework suy luận nào", bạn trả lời có cấu trúc—đuổi throughput và hệ sinh thái cộng đồng chọn vLLM; đa vòng / Agent / structured output chọn SGLang (RadixAttention + cFSM); đuổi latency cực hạn và sẵn sàng đầu tư adapt chọn TensorRT-LLM

### 3.3 Quantization

**Kiến thức cần nắm**

- W8A8 (SmoothQuant): chuyển bài toán outlier của activation sang weights, thân thiện kỹ thuật. Bản chất quantization như nén ảnh độ phân giải cao thành thumbnail—biểu diễn trọng số bằng ít bit hơn, tiết kiệm VRAM và bandwidth, đổi lại chi tiết (độ chính xác) mất một phần
- Weight-only INT4 (GPTQ / AWQ): chỉ quantize trọng số xuống 3/4-bit, giảm VRAM và bandwidth; lúc suy luận nhờ kernel fusion kiểu Marlin dequantize trên chip
- Quantization KV Cache (KIVI / FP8 KV Cache): KIVI quantize Key theo channel, Value theo Token ở 2-bit; FP8 KV Cache là phương án kỹ thuật phổ biến hơn trong engine như vLLM, lợi rõ ở ngữ cảnh dài
- Quantization FP8: Tensor Core của Hopper và Ada hỗ trợ native; điểm cân bằng giữa độ chính xác và hiệu năng
- NVFP4 / MXFP4: định dạng float 4-bit native trên Blackwell, dùng micro-block scale (mỗi 16 hoặc 32 phần tử chia sẻ một Scale) để kiểm soát sai số
- Cây quyết định chọn quantization:

```
Mục tiêu: tiết VRAM? tiết bandwidth? tăng throughput?
├─ Chung, thân thiện kỹ thuật → W8A8 (SmoothQuant); Hopper trở lên ưu tiên FP8
├─ Tiết VRAM/bandwidth hơn → INT4 weight-only (AWQ/GPTQ); trên Blackwell có thể chọn NVFP4
└─ Ngữ cảnh dài / concurrent cao → Quantization KV Cache (FP8 KV / KIVI)
```

**Tài liệu gợi ý**

**Bài trên site:** [Cơ bản quantization](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) · [W8A8 và SmoothQuant](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) · [Weight-only INT4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) · [Quantization KV Cache](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) · [FP8 và NVFP4/MXFP4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) · [Chọn quantization và thực chiến vLLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/)

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Paper | SmoothQuant Paper | Phương án W8A8 kinh điển |
| Paper | GPTQ Paper | Weight-only PTQ 3/4-bit |
| Paper | AWQ Paper | Quantization 4-bit dựa phân phối activation |
| Paper | KIVI Paper | Quantization KV Cache 2-bit |
| Paper | Marlin Paper (FP16xINT4) | Matmul kernel INT4 hiệu năng cao |
| Code | Chuỗi công cụ quantization TensorRT-LLM | Tích hợp FP8/INT4/AWQ/SmoothQuant |
| Code | Hỗ trợ quantization của vLLM | Nhiều lối vào GPTQ/AWQ/FP8/INT8 |

**Tiêu chí kiểm tra**

Quantization là lĩnh vực "nhìn đơn giản, chỉnh như huyền học"; kiểm tra trọng tâm là hiểu trade-off chứ không thuộc công thức:

- **Chọn phương án**: đội phải lên 70B nhưng chỉ có 2 card A100 80GB. Phán nhanh: trọng số FP16 ~140GB, hai card 160GB, để trọng số xong chỉ còn ~20GB cho KV Cache và buffer tạm, concurrent không lên được; W8A8 nén trọng số xuống ~70GB, triển khai TP hai card chừa đủ chỗ KV Cache; INT4 weight-only ~35GB, một card cũng nhét được, nhưng phải đánh giá mất độ chính xác và hiệu quả kernel. Ngoài ra Tensor Core của A100 không hỗ trợ FP8, phương án FP8 trên máy này không dùng được
- **Xác minh thực hành**: dùng vLLM hoặc TensorRT-LLM lần lượt nạp bản FP16 và AWQ-INT4 cùng một mô hình, chạy cùng bộ benchmark prompt, xuất so sánh throughput + vài cặp kết quả sinh để người đọc đối chiếu, viết thành báo cáo đánh giá tái sử dụng được
- **Hiểu phản trực giác**: ai nói "bit càng thấp càng nhanh", bạn giải thích vì sao có cảnh INT4 lại chậm hơn INT8—Decode batch nhỏ là memory bound, W4A16 chuyển ít hơn một nửa trọng số nên thường nhanh hơn; batch lớn bước vào compute bound thì W4A16 phải dequantize trọng số về FP16 rồi mới tính trên FP16 Tensor Core, còn W8A8 đi thẳng INT8 Tensor Core (compute khoảng 2× FP16), lúc đó INT8 lại nhanh hơn
- **Điều tra sự cố**: sau quantization chất lượng output tụt rõ, liệt kê checklist—một số lớp activation outlier quá lớn (cần per-channel hoặc SmoothQuant), dữ liệu calibration không đại diện, một số cấu trúc đặc biệt (như gate của MoE) không hợp low-bit

### 3.4 Speculative Decoding

**Kiến thức cần nắm**

- Speculative Sampling: khung kinh điển—mô hình nhỏ (Draft) "đoán" hàng loạt token, mô hình lớn (Target) xác minh một lần, phân phối không lệch. Như để thực tập sinh soạn nhanh một đoạn rồi biên tập viên cao cấp duyệt một lượt: đoán đúng thì dùng luôn, đoán sai sửa ngay, nhanh hơn biên tập viết từng chữ từ đầu
- Medusa: không dùng Draft model ngoài; nhiều Decoding Head dự đoán multi-token rồi xác minh song song
- EAGLE-2 / EAGLE-3: huấn luyện Draft head nhẹ trên hidden state của Target. EAGLE-2 dựng cây ứng viên động theo độ tin cậy Draft; EAGLE-3 hợp nhất đặc trưng nhiều lớp và mô phỏng suy luận lúc huấn luyện, acceptance rate cao hơn nữa
- MTP (Multi-Token Prediction): module dự đoán đa token kèm theo lúc huấn luyện DeepSeek-V3; lúc suy luận dùng trực tiếp làm Draft
- N-gram / Suffix: không cần Draft model nào; khớp ứng viên từ prompt hoặc văn bản đã sinh—phù hợp sửa code, RAG và cảnh nội dung lặp nhiều
- Block Verification: nâng xác minh mức token lên xác minh liên kết mức block, tăng tốc thêm

**Tài liệu gợi ý**

**Bài trên site:** [Nguyên lý lõi Speculative Decoding](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) · [Draft model và phương án không model](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) · [Medusa, EAGLE và MTP](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) · [Biên lợi ích và hạn chế](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) · [Thực chiến Speculative Decoding với vLLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/)

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Paper | Speculative Decoding (Leviathan et al.) / Speculative Sampling (Chen et al.) | Hai paper độc lập đề xuất khung Draft + Verify |
| Paper | Medusa Paper + Repo | Multi-head decoding, không cần Draft model |
| Paper | EAGLE-2 Paper | Dynamic Draft Tree |
| Paper | EAGLE-3 Paper | Hợp nhất đặc trưng nhiều lớp và train-time testing |
| Paper | Block Verification Paper | Xác minh liên kết mức block |
| Code | Hỗ trợ Speculative của vLLM / TensorRT-LLM | Tham chiếu triển khai sản xuất |

**Tiêu chí kiểm tra**

Tinh túy Speculative Decoding là "đổi không gian lấy thời gian, đổi song song lấy tuần tự"; kiểm tra trọng tâm hiểu tính đúng và biên lợi ích:

- **Đảm bảo tính đúng**: interviewer hỏi "Speculative Decoding có đổi phân phối output của mô hình không", bạn trả lời: không, vì rejection sampling đảm bảo token được chấp nhận tuân thủ nghiêm phân phối của target model—Draft đoán đúng thì dùng luôn, đoán sai thì sample sửa theo chênh xác suất target và draft, toán học tương đương sample trực tiếp từ target
- **Đo lợi ích thực**: dùng speculative decoding của vLLM, lần lượt benchmark trên sinh code (acceptance cao) và hội thoại mở (acceptance thấp), giải thích vì sao cái trước tăng tốc rõ (token code dễ đoán, Draft trúng nhiều) còn cái sau lợi ít hoặc âm
- **Biên không lời**: liệt kê ít nhất 3 cảnh speculative decoding kém hiệu quả—sample nhiệt độ cao làm hit rate Draft lao dốc, batch size đã lớn khiến forward Draft thêm chiếm compute, Draft và Target cách quá xa khiến acceptance rate < 50%
- **Rủi ro ghép kỹ thuật**: chỉ ra xung đột với tối ưu khác—ghép với quantization làm độ chính xác Draft tụt có thể kéo acceptance xuống thêm; ghép với Continuous Batching thì độ dài speculative khác nhau giữa các request, độ phức tạp lịch trình tăng

### 3.5 Kiến trúc hệ thống: Prefill/Decode disaggregation

**Kiến thức cần nắm**

- Vấn đề lõi: Prefill và Decode chung batching gây ghép tài nguyên và nhiễu lẫn, làm đuôi trễ nổ. Như một đầu bếp vừa phải xào nhanh (Decode, latency thấp) vừa phải hầm chậm (Prefill, tính nặng)—kéo nhau. Disaggregation là tách khu đồ nhanh và khu hầm chậm, mỗi khu có đầu bếp riêng
- DistServe (OSDI'24): lập luận hệ thống và hiện thực Prefill/Decode disaggregation, lịch trình quanh goodput
- Splitwise (ISCA'24): gán Prefill và Decode vào các pool GPU khác nhau, tối ưu throughput và chi phí
- TaiChi (2025): thống nhất aggregation và disaggregation, tối ưu goodput theo các tổ hợp SLO khác nhau
- Goodput: throughput hữu hiệu thỏa SLO, khác raw QPS

**Tài liệu gợi ý**

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Paper | DistServe (OSDI'24) | Lập luận hệ thống cho Prefill/Decode disaggregation |
| Paper | Splitwise (ISCA'24) | Thiết kế Phase Splitting |
| Paper | TaiChi (2025) | Khung thống nhất aggregation / disaggregation |
| Tutorial | Disaggregated Inference: 18 Months Later | Vì sao disaggregation trở thành cấu hình mặc định |
| Tutorial | MLC microserving | API lập trình được cho orchestration xuyên engine |

**Tiêu chí kiểm tra**

Prefill/Decode disaggregation là tối ưu ở tầng kiến trúc hệ thống; kiểm tra trọng tâm là hiểu "vì sao phải tách" và "tách xong vấn đề mới là gì":

- **Định lượng nhiễu**: trên dịch vụ suy luận mixed batching, dựng cảnh—vài request Prefill prompt siêu dài đến cùng lúc với nhiều Decode ngắn—dùng chỉ số chứng minh P95 TPOT của Decode bị Prefill kéo chậm 3–5 lần; đó là động cơ của disaggregation
- **Khái niệm Goodput**: sếp hỏi "QPS cao vậy sao user vẫn chê chậm", bạn giải thích goodput—tỷ lệ request thỏa SLO (ví dụ TTFT < 500ms và TPOT < 50ms) mới là chỉ số chất lượng thật; raw QPS không bằng trải nghiệm user
- **Suy tỷ lệ tài nguyên**: cho đặc trưng workload (prompt trung bình 2000 token, output trung bình 500 token), ước tỷ lệ compute Prefill và Decode, rồi suy tỷ lệ hợp lý giữa pool Prefill GPU và Decode GPU (ví dụ 1:3 hoặc 1:4)
- **Danh sách rủi ro**: liệt kê vấn đề mới do kiến trúc disaggregation—áp lực bandwidth mạng khi chuyển KV Cache từ node Prefill sang Decode (ví dụ LLaMA-2-7B, một sequence dài 2048 token KV khoảng 1GB; IB 200Gb/s lý thuyết cũng cần ~40ms), độ phức tạp quản lý hàng đợi của scheduler, khi tải Prefill/Decode lệch thì một pool ngồi không lãng phí

### 3.6 Phân tích hiệu năng và Benchmark

**Kiến thức cần nắm**

- Bộ chỉ số lõi: QPS, TTFT, TPOT, token/s, đuôi trễ P50/P95
- Chuỗi công cụ phân tích:
  - `torch.profiler`: profiler chính thức của PyTorch, định vị toán tử và shape
  - Nsight Systems: phân tích toàn chuỗi tương tác CPU–GPU
  - Nsight Compute: khoan hiệu năng mức kernel
- Công cụ press: `vllm bench serve` (press online serving kèm theo vLLM), GenAI-Perf / AIPerf (công cụ press LLM của NVIDIA; AIPerf có hướng dẫn migrate từ GenAI-Perf), Triton Perf Analyzer
- Benchmark uy tín: MLPerf Inference (Datacenter), thống nhất định nghĩa và luật
- Cổng hồi quy: mỗi thay đổi xuất cùng một bảng so sánh chỉ số; thoái hóa thì chặn merge

**Tài liệu gợi ý**

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Tool | Tài liệu torch.profiler | Điểm bắt đầu phân tích hiệu năng PyTorch |
| Tool | GenAI-Perf / AIPerf | Press một chỗ cho TTFT/TPOT/token throughput |
| Tool | Triton Inference Server Quickstart | Baseline suy luận online container hóa |
| Benchmark | MLPerf Inference (Datacenter) | Cửa vào benchmark uy tín |
| Walkthrough | Đọc nhiệm vụ LLM của MLPerf Inference v5.0 | Hiểu xu hướng benchmark LLM latency thấp |

**Tiêu chí kiểm tra**

Phân tích hiệu năng không chỉ là "biết dùng tool"; lõi là dựng hệ kỹ thuật **lặp lại được, so sánh được, truy vết được**:

- **Bộ chỉ số đủ**: mỗi báo cáo đánh giá hiệu năng ít nhất 6 chỉ số—QPS, TTFT (P50/P95), TPOT (P50/P95), throughput end-to-end (token/s), đỉnh VRAM GPU, GPU utilization; thiếu cái nào cũng có thể sót bottleneck
- **Benchmark tái hiện được**: cấu hình benchmark (mô hình, batch size, độ dài input/output, concurrent, model phần cứng, phiên bản driver) toàn bộ nằm trong một file config; ai cầm file đó cũng tái hiện được kết quả của bạn; giữa hai lần test chỉ đổi một biến
- **Định vị hồi quy**: sau một commit TPOT P95 thoái hóa 15%, bạn dùng `git bisect` thu hẹp tới commit cụ thể, rồi so trace Nsight Systems trước/sau—một kernel chậm hơn, thêm một sync không cần thiết, hay chiến lược lịch batch đổi?
- **Cổng lên production**: đặt cho đội bộ luật cổng hiệu năng đơn giản thực thi được—ví dụ "TPOT P95 thoái hóa >5% thì block merge", "VRAM tăng >10% phải kèm báo cáo phân tích"—và gắn vào CI chạy tự động

### 3.7 Suy luận phân tán và triển khai sản xuất

**Kiến thức cần nắm**

- Suy luận phân tán: một card không nhét nổi hoặc throughput không đủ thì dùng TP, PP, DP và EP trải mô hình ra đa GPU đa máy. MoE lớn thường dùng tổ hợp "Attention đi DP, expert đi EP"; triển khai đa node có thể nhờ công cụ orchestration như Ray
- Triển khai sản xuất: container và Kubernetes, chỉ số Prometheus và observability, autoscaling theo độ dài hàng đợi hoặc mức chiếm KV Cache, load balancing nhận biết Prefix Cache, lập kế hoạch dung lượng
- Suy luận trên thiết bị đầu cuối: xuất mô hình, runtime đầu cuối (như ExecuTorch, llama.cpp), thực thi dị thể CPU/GPU/NPU, cùng quantization và tối ưu bộ nhớ

**Tài liệu gợi ý**

**Bài trên site:** [Cơ bản suy luận đầu cuối: từ xuất mô hình đến thực thi phần cứng dị thể](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第9章-端侧推理/91-端侧推理基础/)

| Loại | Tài liệu | Ghi chú |
|------|------|------|
| Docs | Tài liệu chính thức vLLM (Parallelism and Scaling) | Cấu hình TP/PP/DP/EP và triển khai đa node |
| Docs | Tài liệu chính thức ExecuTorch | Runtime suy luận đầu cuối chính thức của PyTorch |

**Tiêu chí kiểm tra**

- **Chọn độ song song**: cho mô hình Dense 70B và một máy 8×H100, nói rõ TP=8 vs TP=4 + DP=2 khác nhau thế nào về latency một request, tổng throughput và dung lượng KV Cache
- **Tín hiệu autoscaling**: giải thích vì sao dịch vụ LLM không nên chỉ scale theo GPU utilization, mà cần kết hợp số request đang xếp hàng, mức dùng KV Cache, TTFT và các chỉ số liên quan

### 3.8 Cây quyết định chọn tối ưu

Tối ưu suy luận không phải "chồng hết kỹ thuật lên là xong", mà **trước tiên định vị bottleneck, rồi trị đúng bệnh**. Như đi khám: bác sĩ không kê thuốc ngay, mà hỏi "đâu khó chịu", làm xét nghiệm xác nhận nguyên nhân, rồi mới ra toa. Cây quyết định dưới đây là "quy trình hỏi bệnh" của tối ưu suy luận—bắt đầu từ triệu chứng, thu hẹp dần tới phương án cụ thể.

**Bước 1: Xác nhận triệu chứng chính**

Khi có vấn đề hiệu năng suy luận, trước hết dùng công cụ press (`vllm bench serve`, GenAI-Perf / AIPerf hoặc script tự viết) chạy baseline, xem chỉ số nào lệch nhất, rồi đối số:

```
Triệu chứng là gì?
│
├─ [A] TTFT quá cao (user chờ token đầu quá lâu)
├─ [B] TPOT quá cao (token nhích từng cái, cảm giác giật)
├─ [C] VRAM không đủ (OOM hoặc không tăng được concurrent)
└─ [D] Đuôi trễ P95/P99 mất kiểm soát (trung bình ổn nhưng gai nặng)
```

**[A] TTFT quá cao—token đầu mãi không tới**

TTFT chủ yếu bị ảnh hưởng bởi Prefill. Prefill phải xử lý hết cả prompt một lần, là thao tác compute-bound điển hình. Hướng định vị:

| Tiểu triệu chứng | Cách chẩn đoán | Phương án gợi ý |
|--------|---------|---------|
| Prompt rất dài (vài nghìn đến vài chục nghìn token) | Xem thời gian Prefill tăng tuyến tính hay siêu tuyến tính theo độ dài prompt | Chunked Prefill xử lý theo khối, tránh Prefill dài chiếm độc GPU; nâng GEMM kernel hiệu quả hơn |
| CPU hoặc lịch trình thành bottleneck | Bắt trace bằng Nsight Systems, xem GPU idle gap có do phía host không | Săn chi phí CPU của tokenizer, tiền xử lý dữ liệu; cân nhắc đổi sang inference engine lịch trình chín hơn |
| Nhiều request chia sẻ cùng prefix (ví dụ system prompt) | Thống kê tỷ lệ trùng prefix của prompt | Bật Prefix Cache (vLLM) hoặc RadixAttention (SGLang), tái sử dụng KV Cache đã tính |

**[B] TPOT quá cao—sinh token như nặn kem đánh răng**

Decode mỗi bước chỉ sinh một token, ma trận thoái hóa thành phép vector, là thao tác memory-bound điển hình. Có ba hướng tối ưu:

| Tiểu triệu chứng | Cách chẩn đoán | Phương án gợi ý |
|--------|---------|---------|
| Chuyển KV Cache thành bottleneck bandwidth | Trên Nsight Compute xem memory throughput của Attention kernel có gần trần bandwidth HBM không | Dùng FlashAttention / FlashInfer giảm truy cập VRAM vô ích; tối ưu layout KV Cache |
| Giới hạn bản chất của sinh token tuần tự | TPOT một request đã sát sàn lý thuyết, user vẫn chê chậm | Speculative Decoding (EAGLE-3 / MTP / N-gram), dùng "đoán + xác minh" phá bottleneck tuần tự |
| Concurrent thấp, compute GPU chưa no | GPU utilization dưới 60%, mỗi batch chỉ vài request | Bật Continuous Batching + tăng số request concurrent, làm batch lớn hơn để tăng mật độ tính |

**[C] VRAM không đủ—OOM hoặc concurrent không lên**

VRAM là tài nguyên khan nhất trong suy luận, bị hai khối chiếm: trọng số mô hình và KV Cache. Chọn phương án theo khối nào chiếm lớn:

| "Ông lớn" VRAM | Cách chẩn đoán | Phương án gợi ý |
|---------|---------|---------|
| KV Cache (ngữ cảnh dài hoặc concurrent cao) | So tỷ lệ VRAM tham số mô hình và VRAM KV Cache (theo cách tính ở 3.1) | PagedAttention giải quyết phân mảnh; quantization KV Cache (FP8 KV, KIVI 2-bit) nén lưu trữ |
| Bản thân trọng số mô hình | Trọng số FP16 đã vượt dung lượng một card | INT4 weight-only (AWQ/GPTQ) có thể nén trọng số xuống ~1/4; W8A8 (SmoothQuant hoặc FP8) xuống ~1/2, batch lớn thì hiệu quả tính cao hơn |

**[D] Đuôi trễ P95/P99 mất kiểm soát—trung bình ổn nhưng gai nặng**

Đây là bài khó nhất: thường không phải một đoạn chậm, mà là tranh tài nguyên và lịch trình kém ở tầng hệ thống:

| Tiểu triệu chứng | Cách chẩn đoán | Phương án gợi ý |
|--------|---------|---------|
| Prefill và Decode nhiễu lẫn | Dưới tải hỗn hợp so TPOT P95 thuần Decode với cảnh hỗn hợp; chênh >2 lần là xác nhận nhiễu | Triển khai Prefill/Decode disaggregation (DistServe / Splitwise), hai loại request dùng pool GPU riêng |
| SLO rất chặt, cần lịch trình tối ưu toàn cục | Sau khi chỉ disaggregation vẫn có request timeout, hoặc request có SLO khác nhau | Khung lịch trình thống nhất kiểu TaiChi, phân bổ tài nguyên Prefill/Decode động theo ưu tiên SLO |

**Gợi ý dùng thực tế**

Thực tế vấn đề hiệu năng thường không chỉ một triệu chứng—TTFT và VRAM có thể cùng lệch. Thứ tự nên làm:

1. **Xử lý OOM trước** (vấn đề VRAM), không thì dịch vụ còn chưa lên được
2. **Rồi tối ưu TTFT**, vì độ trễ token đầu ảnh hưởng trực tiếp cảm nhận user
3. **Sau đó nâng TPOT và throughput**, thường phải trade-off giữa latency và throughput
4. **Cuối cùng xử lý đuôi trễ**, cần đổi kiến trúc hệ thống, đầu tư lớn nhất

---

## 🧭 Hướng dẫn phá thế cho người mới

### Lộ trình học gợi ý

Đối mặt stack lớn thế này, nên **dùng thời gian ngắn nhất thông nền từng lớp, rồi theo vị trí mục tiêu chọn một hướng đào sâu**—không cần đợi mỗi lớp học thấu mới leo tiếp:

**Giai đoạn nền (0–3 tháng)**

1. Hoàn thành toàn bộ tiêu chí kiểm tra Lớp 0: ngôn ngữ lập trình, toán, kiến trúc Transformer, vòng huấn luyện PyTorch, cơ bản topology giao tiếp
2. Học cơ bản lập trình CUDA, viết được Reduce / GEMM kernel đơn giản
3. Thử dùng PyTorch DDP trải huấn luyện ra hai GPU, quan sát thay đổi VRAM và giao tiếp

**Đào sâu chuyên (3–6 tháng)**

1. Đọc kỹ bốn paper cột mốc đối chiếu với code:
   - Megatron-LM (nguyên lý TP và PP)
   - ZeRO (lõi DeepSpeed)
   - FlashAttention (thuật toán memory-aware)
   - vLLM (PagedAttention, KV Cache)
2. Đọc xuyên một inference engine nhỏ (ví dụ nano-vllm, xem [đọc source](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/第3章-nano-vllm源码解读/) trên site), nối lịch trình, quản lý KV Cache và thực thi mô hình thành một chuỗi
3. Tham gia dự án mã nguồn mở (vLLM, DeepSpeed, SGLang), đóng góp tối ưu toán tử hoặc module tính năng
4. Nắm quantization, Speculative Decoding và các kỹ thuật tối ưu suy luận khác

**Thực chiến kỹ thuật (từ 6 tháng trở lên)**

1. Triển khai mô hình 10B / 100B tham số trên cụm GPU, tối ưu hiệu năng end-to-end
2. Dựng hệ phân tích hiệu năng và hồi quy đầy đủ
3. Nghiên cứu kiến trúc hệ thống tiên phong như Prefill/Decode disaggregation
4. Theo dõi vòng lặp kỹ thuật mới (độ chính xác thấp FP8/FP4, tối ưu mạng RDMA, thích ứng kiến trúc mới, v.v.)

**Gợi ý chọn hướng**: vị trí pretrain quy mô lớn tập trung ở số ít đội có cụm vạn card, ghế hữu hạn; tối ưu suy luận, phát triển toán tử và framework huấn luyện/suy luận có nhu cầu rộng hơn. Nếu mục tiêu là xin việc, nên ưu tiên làm dự án trưng được trên các hướng đó.

### Mô hình tư duy lõi

Trong quá trình học AI Infra, luôn nhớ một khung đánh đổi:

| Kỹ thuật tối ưu | Hy sinh gì | Đổi lấy gì |
|---------|-----------|-----------|
| ZeRO | Bandwidth giao tiếp | Không gian VRAM |
| Activation Checkpointing | Thời gian tính | Không gian VRAM |
| Quantization | Độ chính xác | VRAM + bandwidth + throughput |
| Speculative Decoding | Tính toán thêm (forward Draft + xác minh token bị từ chối) | Latency Decode |
| Prefill/Decode disaggregation | Độ phức tạp hệ thống + chi phí chuyển KV | Đuôi trễ + goodput |
| FlashAttention | Độ phức tạp hiện thực | VRAM + tốc độ |

---

## 📚 Tài liệu tham khảo

### Paper

- **Attention Is All You Need** (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **Megatron-LM** (Shoeybi et al., 2019): [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2019): [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
- **FlashAttention V1** (Dao et al., 2022): [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- **FlashAttention V2** (Dao, 2023): [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- **FlashAttention-3** (Shah et al., 2024): [https://arxiv.org/abs/2407.08691](https://arxiv.org/abs/2407.08691)
- **Flash-Decoding** (Stanford CRFM, 2023): [https://crfm.stanford.edu/2023/10/12/flashdecoding.html](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- **FlashInfer** (Ye et al., 2025): [https://arxiv.org/abs/2501.01005](https://arxiv.org/abs/2501.01005)
- **vLLM / PagedAttention** (Kwon et al., 2023): [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **SGLang** (Zheng et al., 2023): [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- **Orca** (Yu et al., 2022): [https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)
- **DistServe** (Zhong et al., OSDI'24): [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)
- **Splitwise** (Patel et al., ISCA 2024): [https://arxiv.org/abs/2311.18677](https://arxiv.org/abs/2311.18677)
- **SmoothQuant** (Xiao et al., 2022): [https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)
- **GPTQ** (Frantar et al., 2022): [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ** (Lin et al., 2023): [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- **KIVI** (Liu et al., 2024): [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)
- **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2022): [https://arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
- **Accelerating LLM Decoding with Speculative Sampling** (Chen et al., 2023): [https://arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318)
- **Medusa** (Cai et al., 2024): [https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- **EAGLE-2** (Li et al., 2024): [https://arxiv.org/abs/2406.16858](https://arxiv.org/abs/2406.16858)
- **EAGLE-3** (Li et al., 2025): [https://arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840)
- **Block Verification Accelerates Speculative Decoding** (Sun et al., 2024): [https://arxiv.org/abs/2403.10444](https://arxiv.org/abs/2403.10444)
- **Ring Attention** (Liu et al., 2023): [https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)
- **DeepSpeed Ulysses** (Jacobs et al., 2023): [https://arxiv.org/abs/2309.14509](https://arxiv.org/abs/2309.14509)
- **Online normalizer calculation for softmax** (Milakov & Gimelshein, 2018): [https://arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)
- **Báo cáo kỹ thuật DeepSeek V2**: [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- **DeepSeekMoE**: [https://arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)
- **Báo cáo kỹ thuật DeepSeek V3** (gồm MTP, huấn luyện FP8): [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- **Towards Efficient Generative LLM Serving: A Survey** (CMU): [https://arxiv.org/abs/2312.15234](https://arxiv.org/abs/2312.15234)

### Tutorial và video

- **The Illustrated Transformer** (Jay Alammar): [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- **3Blue1Brown: Essence of linear algebra**: [https://www.3blue1brown.com/topics/linear-algebra](https://www.3blue1brown.com/topics/linear-algebra)
- **Andrej Karpathy: Let's build GPT from scratch**: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **MIT The Missing Semester**: [https://missing.csail.mit.edu/](https://missing.csail.mit.edu/)

### Dự án mã nguồn mở và công cụ

- **vLLM** GitHub: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **SGLang** GitHub: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
- **DeepSpeed** GitHub: [https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- **Megatron-LM** GitHub: [https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- **FlashAttention** GitHub: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **FlashInfer** GitHub: [https://github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **TensorRT-LLM** GitHub: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **Triton** GitHub: [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)
- **AIPerf** GitHub: [https://github.com/ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf)

### Tài liệu chính thức

- **NVIDIA CUDA Programming Guide**: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **Tài liệu NVIDIA NCCL**: [https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- **Nsight Systems User Guide**: [https://docs.nvidia.com/nsight-systems/UserGuide/](https://docs.nvidia.com/nsight-systems/UserGuide/)
- **Nsight Compute**: [https://docs.nvidia.com/nsight-compute/ProfilingGuide/](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- **NVIDIA Deep Learning Performance Guide**: [https://docs.nvidia.com/deeplearning/performance/](https://docs.nvidia.com/deeplearning/performance/)
- **Tài liệu chính thức DeepSpeed**: [https://www.deepspeed.ai/](https://www.deepspeed.ai/)
- **Tutorial chính thức PyTorch (60 Minute Blitz)**: [https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- **Tutorial PyTorch DDP**: [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- **Tutorial PyTorch FSDP**: [https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- **Tài liệu chính thức vLLM**: [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **Tài liệu chính thức TensorRT-LLM**: [https://nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
- **Tutorial chính thức Triton**: [https://triton-lang.org/main/getting-started/tutorials/](https://triton-lang.org/main/getting-started/tutorials/)
- **MLPerf Inference**: [https://mlcommons.org/benchmarks/inference-datacenter/](https://mlcommons.org/benchmarks/inference-datacenter/)
- **GenAI-Perf**: [https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/)
