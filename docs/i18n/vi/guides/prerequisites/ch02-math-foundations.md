---
title: "Chương 2: Nền tảng toán học"
description: "Xây dựng trực giác đại số tuyến tính, xác suất và giải tích mà AI Infra cần: toán đứng sau GEMM tiling, Softmax, lan truyền ngược và huấn luyện mixed precision"
tags: ["Đại số tuyến tính", "Xác suất", "Giải tích", "Nền tảng toán"]
sourceHash: "eeef6ebebd63bceb"
---

AI Infra không đòi bạn mỗi ngày tự chứng minh định lý phức tạp. Nó đòi khi nhìn thấy công thức, bạn trả lời ngay được ba câu hỏi kỹ thuật: **tensor có shape nào, cần bao nhiêu tính toán và bộ nhớ, và triển khai thế nào cho ổn định**.

Chương này xây trực giác toán quanh ba câu hỏi đó. Ta bắt đầu từ vector và ma trận, nối sang GEMM và tiling; từ phân phối xác suất sang Softmax, cross-entropy và sampling; từ quy tắc chuỗi sang lan truyền ngược; cuối cùng là số dấu phẩy động, sai số và mixed precision. Mục tiêu không phải chứng minh mọi kết luận, mà đạt mức “đọc được công thức, suy được chiều, ước được bậc độ lớn, phán được rủi ro số”.

<!-- more -->

## 📑 Mục lục

- [1. Mục tiêu học, ký hiệu và ba câu hỏi](#1-mục-tiêu-học-ký-hiệu-và-ba-câu-hỏi)
- [2. Đại số tuyến tính: tensor, phép biến đổi và phân tích](#2-đại-số-tuyến-tính-tensor-phép-biến-đổi-và-phân-tích)
- [3. Ma trận khối và trực giác kỹ thuật GEMM](#3-ma-trận-khối-và-trực-giác-kỹ-thuật-gemm)
- [4. Xác suất: từ biến ngẫu nhiên đến phân phối mô hình](#4-xác-suất-từ-biến-ngẫu-nhiên-đến-phân-phối-mô-hình)
- [5. Softmax, cross-entropy và KL divergence](#5-softmax-cross-entropy-và-kl-divergence)
- [6. Giải tích và lan truyền ngược](#6-giải-tích-và-lan-truyền-ngược)
- [7. Tối ưu, ổn định gradient và chuẩn hóa](#7-tối-ưu-ổn-định-gradient-và-chuẩn-hóa)
- [8. Tính toán số và mixed precision](#8-tính-toán-số-và-mixed-precision)
- [9. Ví dụ tổng hợp AI Infra](#9-ví-dụ-tổng-hợp-ai-infra)
- [Tóm tắt](#tóm-tắt)
- [Checklist tự kiểm](#checklist-tự-kiểm)
- [Bài tập](#bài-tập)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## 1. Mục tiêu học, ký hiệu và ba câu hỏi

Sau chương này, bạn nên:

1. Phân biệt được scalar, vector, ma trận và tensor bậc cao, và suy shape sau phép toán một cách thành thạo;
2. Giải thích được trực giác của tích vô hướng, chuẩn (norm), nhân ma trận, hạng (rank), phân tích riêng và SVD;
3. Tách nhân ma trận thành block và hiểu vì sao GEMM tiling tương đương về mặt toán;
4. Tính kỳ vọng, phương sai, hiệp phương sai; hiểu các phân phối thường gặp và xác suất có điều kiện;
5. Từ logits ra Softmax ổn định; hiểu temperature, cross-entropy, entropy và KL divergence;
6. Đọc computation graph bằng quy tắc chuỗi; tự suy gradient của linear layer và Softmax cross-entropy;
7. Giải thích gradient biến mất/bùng nổ, residual connection và chuẩn hóa ảnh hưởng thế nào tới huấn luyện;
8. Phân biệt độ chính xác và dải động của FP32, TF32, FP16, BF16 và FP8;
9. Nhận ra overflow, underflow, catastrophic cancellation, tính không kết hợp và các rủi ro số khác;
10. Ước số tham số, kích thước activation, FLOPs, lượng truy cập bộ nhớ và arithmetic intensity của một toán tử.

### 1.1 Ký hiệu trong chương

| Ký hiệu | Nghĩa |
|------|------|
| $x$ | Scalar, hoặc biến khi ngữ cảnh rõ |
| $\mathbf{x}$ | Vector |
| $\mathbf{X}$ | Ma trận hoặc tensor bậc cao |
| $X_{ij}$ | Phần tử hàng $i$, cột $j$ của ma trận |
| $\mathbb{R}^{m \times n}$ | Tập ma trận thực $m$ hàng, $n$ cột |
| $\mathbf{X}^\top$ | Chuyển vị |
| $\mathbf{X}^{-1}$ | Nghịch đảo (nếu tồn tại) |
| $\lVert \mathbf{x} \rVert_2$ | Chuẩn L2 của vector |
| $\mathbb{E}[X]$ | Kỳ vọng |
| $\operatorname{Var}(X)$ | Phương sai |
| $\nabla_x f$ | Gradient của $f$ theo $x$ |
| $\partial f / \partial x$ | Đạo hàm riêng |
| $\odot$ | Nhân từng phần tử |

Code deep learning thường dùng ký hiệu batch-first:

- $B$: batch size;
- $S$: sequence length;
- $H$: hidden size;
- $V$: vocabulary size;
- $N_h$: số attention head;
- $D_h = H / N_h$: chiều mỗi head.

### 1.2 Ba câu hỏi khi đọc công thức

Lấy phép chiếu đầu ra của language model làm ví dụ:

$$
\mathbf{Y} = \mathbf{X}\mathbf{W}, \qquad
\mathbf{X} \in \mathbb{R}^{(BS) \times H},\quad
\mathbf{W} \in \mathbb{R}^{H \times V}
$$

Câu một: **shape có khớp không?** Chiều trong $H$ giống nhau, đầu ra là $(BS) \times V$.

Câu hai: **chi phí bao nhiêu?** Mỗi phần tử đầu ra làm $H$ lần nhân-cộng, tổng khoảng:

$$
2BSHV \quad \text{FLOPs}
$$

Ở đây một phép nhân và một phép cộng mỗi cái tính một thao tác dấu phẩy động.

Câu ba: **rủi ro triển khai là gì?** Weight lớn, logits đầu ra lớn; cần nghĩ tới dtype, memory layout, tiling, chiến lược song song và ổn định số của Softmax.

Ba câu hỏi này xuyên suốt cả stack AI Infra.

---

## 2. Đại số tuyến tính: tensor, phép biến đổi và phân tích

### 2.1 Scalar, vector, ma trận và tensor

- **Scalar** là một số, ví dụ learning rate $\eta = 10^{-4}$;
- **Vector** là mảng có thứ tự một chiều, ví dụ hidden state của một token $\mathbf{x} \in \mathbb{R}^{H}$;
- **Ma trận** là mảng hai chiều, ví dụ weight linear layer $\mathbf{W} \in \mathbb{R}^{H \times 4H}$;
- **Tensor** là mảng nhiều chiều, ví dụ hidden state của một batch chuỗi $\mathbf{X} \in \mathbb{R}^{B \times S \times H}$.

Trong kỹ thuật, “tensor” còn mang thông tin ngoài shape toán học:

```text
data address + shape + stride + dtype + device + layout
```

Hai tensor cùng shape chưa chắc cùng memory layout. Giá trị giống nhau cũng chưa chắc cùng dtype và hành vi sai số.

### 2.2 Shape, chỉ số và stride

Xét ma trận row-major shape $(2, 3)$:

$$
\mathbf{X} =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

Bộ nhớ liên tục bên dưới là `[1, 2, 3, 4, 5, 6]`, stride theo phần tử là $(3, 1)$. Offset tuyến tính của $X_{ij}$ là:

$$
\text{offset}(i,j) = 3i + j
$$

View chuyển vị $\mathbf{X}^\top$ có shape $(3,2)$, stride thành $(1,3)$, thường không cần copy dữ liệu. Nhưng khi quét theo chiều cuối sau chuyển vị, truy cập không còn liên tục.

`reshape` có copy hay không phụ thuộc layout gốc có biểu diễn được bằng shape/stride mới hay không. Khi thấy `view`, `reshape`, `transpose`, `contiguous`, hãy nghĩ đồng thời tới chiều logic và layout vật lý.

### 2.3 Phép từng phần tử và broadcasting

Cộng từng phần tử đòi các phần tử tương ứng ghép được với nhau:

$$
\mathbf{C} = \mathbf{A} + \mathbf{B}, \qquad C_{ij} = A_{ij} + B_{ij}
$$

Broadcasting cho phép một số chiều độ dài 1 hoặc thiếu được mở rộng về mặt logic. Ví dụ:

$$
\mathbf{X} \in \mathbb{R}^{B \times S \times H}, \qquad
\mathbf{b} \in \mathbb{R}^{H}
$$

$\mathbf{X}+\mathbf{b}$ tái sử dụng cùng một bias trên chiều batch và sequence; shape kết quả vẫn $(B,S,H)$.

Khi căn từ chiều đuôi về trước, mỗi cặp chiều phải thỏa:

1. Hai kích thước bằng nhau; hoặc
2. Ít nhất một kích thước bằng 1; hoặc
3. Một phía không có chiều đó.

Broadcasting thường là view logic, không materialize sẵn cả tensor. Nhưng pattern truy cập và reduce của kernel sau đó vẫn bị ảnh hưởng. Nguy hiểm hơn: shape sai vẫn “vừa broadcast”, code không báo lỗi nhưng ngữ nghĩa tính sai.

### 2.4 Tích vô hướng: độ tương đồng và tổng có trọng số

Tích vô hướng của hai vector $n$ chiều:

$$
\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{n}x_i y_i
$$

Hai trực giác thường dùng:

**Tổng có trọng số**: $\mathbf{x}$ là trọng số, $\mathbf{y}$ là giá trị.

**Độ tương đồng hướng**:

$$
\mathbf{x}^\top \mathbf{y}
= \lVert \mathbf{x}\rVert_2
  \lVert \mathbf{y}\rVert_2 \cos\theta
$$

Vậy cosine similarity là:

$$
\cos\theta =
\frac{\mathbf{x}^\top\mathbf{y}}
{\lVert\mathbf{x}\rVert_2\lVert\mathbf{y}\rVert_2}
$$

Attention dùng tích vô hướng Query–Key để đo mức khớp. Embedding retrieval thường dùng tích vô hướng đã chuẩn hóa để xấp xỉ cosine similarity.

Nếu chuẩn vector gần 0, mẫu số không ổn định; triển khai thường cộng thêm $\epsilon$ nhỏ.

### 2.5 Chuẩn: độ dài, sai số và ràng buộc

Các chuẩn vector thường gặp:

$$
\lVert \mathbf{x} \rVert_1 = \sum_i |x_i|
$$

$$
\lVert \mathbf{x} \rVert_2 = \sqrt{\sum_i x_i^2}
$$

$$
\lVert \mathbf{x} \rVert_\infty = \max_i |x_i|
$$

Chuẩn Frobenius của ma trận xem mọi phần tử như một vector dài:

$$
\lVert \mathbf{A} \rVert_F
= \sqrt{\sum_i\sum_j A_{ij}^2}
$$

Trong kỹ thuật, chuẩn dùng để:

- Đo quy mô tham số, activation hoặc gradient;
- Gradient clipping;
- So sai số giữa bản tham chiếu và bản tối ưu;
- Regularization;
- Chuẩn hóa vector.

Khi so kết quả dấu phẩy động, sai số tuyệt đối và tương đối đều có vai trò:

$$
e_{abs} = |\hat{x}-x|,\qquad
e_{rel} = \frac{|\hat{x}-x|}{|x|}
$$

Khi giá trị tham chiếu $x$ gần 0, sai số tương đối bị phóng đại, nên thư viện kiểm thử thường kết hợp absolute tolerance `atol` và relative tolerance `rtol`:

$$
|\hat{x}-x| \le \text{atol} + \text{rtol}\cdot |x|
$$

### 2.6 Nhân ma trận là một tập tích vô hướng

Nếu:

$$
\mathbf{A} \in \mathbb{R}^{M \times K}, \qquad
\mathbf{B} \in \mathbb{R}^{K \times N}
$$

thì:

$$
\mathbf{C} = \mathbf{A}\mathbf{B}
\in \mathbb{R}^{M \times N}
$$

Mỗi phần tử là:

$$
C_{ij} = \sum_{k=1}^{K} A_{ik}B_{kj}
$$

Coi shape như kiểm tra giao diện:

```text
(M, K) @ (K, N) -> (M, N)
```

$K$ ở giữa bị reduce; $M,N$ phía ngoài được giữ. GEMM chuẩn thường viết:

$$
\mathbf{C} \leftarrow
\alpha\mathbf{A}\mathbf{B} + \beta\mathbf{C}
$$

Nhân ma trận nói chung **không giao hoán**: $\mathbf{A}\mathbf{B}$ và $\mathbf{B}\mathbf{A}$ có thể khác shape hoặc khác giá trị. Nhưng có tính kết hợp:

$$
(\mathbf{A}\mathbf{B})\mathbf{C}
= \mathbf{A}(\mathbf{B}\mathbf{C})
$$

Bằng nhau về toán không có nghĩa kết quả dấu phẩy động giống từng bit, cũng không có nghĩa chi phí tính toán giống nhau. Đổi thứ tự kết hợp có thể đổi mạnh kích thước tensor trung gian và FLOPs.

### 2.7 Batched matrix multiplication

Trong multi-head Attention thường gặp:

$$
\mathbf{Q} \in \mathbb{R}^{B \times N_h \times S_q \times D_h}
$$

$$
\mathbf{K} \in \mathbb{R}^{B \times N_h \times S_k \times D_h}
$$

Sau khi chuyển vị hai chiều cuối:

$$
\mathbf{K}^\top
\in \mathbb{R}^{B \times N_h \times D_h \times S_k}
$$

Nhân ma trận theo batch:

$$
\mathbf{Q}\mathbf{K}^\top
\in \mathbb{R}^{B \times N_h \times S_q \times S_k}
$$

Hai chiều đầu $(B,N_h)$ là chiều batch; mỗi `(batch, head)` chạy một phép nhân ma trận độc lập. Đừng thần thánh hóa “nhân tensor bậc cao”: thường chỉ là nhân ma trận trên hai chiều cuối, các chiều trước lo batch và broadcast.

### 2.8 Phép biến đổi tuyến tính và affine

Phép biến đổi tuyến tính nghiêm ngặt thỏa:

$$
f(a\mathbf{x}+b\mathbf{y})
= af(\mathbf{x})+bf(\mathbf{y})
$$

Nhân ma trận $f(\mathbf{x})=\mathbf{W}\mathbf{x}$ là tuyến tính. “Linear layer” trong mạng nơ-ron thường còn có bias:

$$
\mathbf{y}=\mathbf{W}\mathbf{x}+\mathbf{b}
$$

Về toán đó là phép affine. Nhiều lớp linear/affine không activation có thể gộp thành một, nên activation phi tuyến mới là thứ giúp mạng sâu biểu diễn hàm phức tạp.

### 2.9 Chuyển vị, ma trận đơn vị và nghịch đảo

Chuyển vị đổi hàng–cột:

$$
(\mathbf{A}^\top)_{ij} = A_{ji}
$$

Quy tắc hay dùng:

$$
(\mathbf{A}\mathbf{B})^\top
= \mathbf{B}^\top\mathbf{A}^\top
$$

Ma trận đơn vị $\mathbf{I}$ thỏa:

$$
\mathbf{I}\mathbf{A}=\mathbf{A}\mathbf{I}=\mathbf{A}
$$

Nếu ma trận vuông $\mathbf{A}$ có nghịch đảo:

$$
\mathbf{A}^{-1}\mathbf{A}=
\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}
$$

Khi giải $\mathbf{A}\mathbf{x}=\mathbf{b}$, toán học có thể viết $\mathbf{x}=\mathbf{A}^{-1}\mathbf{b}$, nhưng triển khai số thường không lập nghịch đảo tường minh; dùng LU, QR, Cholesky… để giải trực tiếp. Lập nghịch đảo tường minh thường tốn hơn và sai số lớn hơn.

### 2.10 Độc lập tuyến tính, không gian sinh và hạng

Nếu trong một tập vector có vector viết được bằng tổ hợp tuyến tính của các vector còn lại, chúng phụ thuộc tuyến tính. Hạng (rank) của ma trận có thể hiểu là số hướng độc lập:

$$
\operatorname{rank}(\mathbf{A})
\le \min(M,N)
$$

Hạng thấp nghĩa là thông tin trong ma trận thừa, có thể xấp xỉ bằng hai ma trận nhỏ hơn:

$$
\mathbf{A} \approx
\mathbf{U}\mathbf{V},\qquad
\mathbf{U}\in\mathbb{R}^{M\times r},
\mathbf{V}\in\mathbb{R}^{r\times N},
\quad r \ll \min(M,N)
$$

Số tham số giảm từ $MN$ xuống $r(M+N)$, và đó là nền tảng toán của low-rank adapter cùng nén mô hình.

### 2.11 Giá trị riêng và vector riêng

Với ma trận vuông $\mathbf{A}$, nếu tồn tại vector khác không $\mathbf{v}$ và scalar $\lambda$ sao cho:

$$
\mathbf{A}\mathbf{v}=\lambda\mathbf{v}
$$

thì $\mathbf{v}$ là vector riêng, $\lambda$ là giá trị riêng tương ứng. Trực giác: trên hướng đó phép biến đổi chỉ scale, không đổi hướng.

Giá trị riêng giúp hiểu:

- Hệ động học tuyến tính khuếch đại hay suy giảm;
- Độ cong Hessian theo các hướng khác nhau;
- Vì sao gradient bùng nổ hoặc biến mất khi nhiều Jacobian nhân liên tiếp;
- Hướng biến thiên chính của ma trận hiệp phương sai (PCA).

Không phải ma trận nào cũng phân tích riêng tốt trên trường thực. Trong kỹ thuật hãy chọn thuật toán khớp cấu trúc ma trận, đừng coi `eig` là công cụ vạn năng.

### 2.12 Phân tích giá trị singular (SVD)

Mọi ma trận thực $\mathbf{A}\in\mathbb{R}^{M\times N}$ đều viết được:

$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top
$$

trong đó:

- Các cột của $\mathbf{U}$ là left singular vector;
- Các cột của $\mathbf{V}$ là right singular vector;
- Đường chéo $\mathbf{\Sigma}$ chứa singular value $\sigma_1\ge\sigma_2\ge\cdots\ge0$.

Giữ $r$ singular value lớn nhất:

$$
\mathbf{A}_r
= \mathbf{U}_{:,1:r}
  \mathbf{\Sigma}_{1:r,1:r}
  \mathbf{V}_{:,1:r}^\top
$$

$\mathbf{A}_r$ là một xấp xỉ rank-$r$ tối ưu (theo chuẩn 2 và Frobenius thông thường). Trực giác: singular value nhỏ tương ứng hướng ảnh hưởng yếu, có thể bỏ để đổi lấy nén.

Bản thân SVD cũng có chi phí tính và bộ nhớ. Nén weight mô hình lớn thường dùng truncation, thuật toán ngẫu nhiên, hoặc huấn luyện trực tiếp nhân tử hạng thấp, chứ không SVD đầy đủ trên mọi ma trận khổng lồ.

### 2.13 Cập nhật hạng thấp của LoRA

Với weight tiền huấn luyện $\mathbf{W}_0\in\mathbb{R}^{d_{out}\times d_{in}}$, LoRA đóng băng $\mathbf{W}_0$ và chỉ huấn luyện cập nhật hạng thấp:

$$
\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W}
$$

$$
\Delta\mathbf{W}
= \frac{\alpha}{r}\mathbf{B}\mathbf{A}
$$

với:

$$
\mathbf{A}\in\mathbb{R}^{r\times d_{in}},\qquad
\mathbf{B}\in\mathbb{R}^{d_{out}\times r}
$$

Số tham số mới giảm từ $d_{out}d_{in}$ của cập nhật đầy đủ xuống:

$$
r(d_{in}+d_{out})
$$

Ví dụ $d_{in}=d_{out}=4096$, $r=16$:

$$
\frac{16(4096+4096)}{4096^2}
= 0.0078125 \approx 0.78\%
$$

Hạng thấp chỉ giảm tham số huấn luyện được và optimizer state tương ứng; không tự loại bỏ chi phí weight nền, activation và tính forward. Khi deploy có thể gộp weight, hoặc giữ adapter để chuyển động; hai cách đánh đổi VRAM, latency và độ linh hoạt khác nhau.

---

## 3. Ma trận khối và trực giác kỹ thuật GEMM

### 3.1 Vì sao có thể tính theo khối

Cắt ma trận theo $M$, $N$ và chiều reduce $K$. Ví dụ block $2\times2$:

$$
\mathbf{A}=
\begin{bmatrix}
\mathbf{A}_{11} & \mathbf{A}_{12}\\
\mathbf{A}_{21} & \mathbf{A}_{22}
\end{bmatrix},\qquad
\mathbf{B}=
\begin{bmatrix}
\mathbf{B}_{11} & \mathbf{B}_{12}\\
\mathbf{B}_{21} & \mathbf{B}_{22}
\end{bmatrix}
$$

thì:

$$
\mathbf{C}_{11}
= \mathbf{A}_{11}\mathbf{B}_{11}
+ \mathbf{A}_{12}\mathbf{B}_{21}
$$

$$
\mathbf{C}_{12}
= \mathbf{A}_{11}\mathbf{B}_{12}
+ \mathbf{A}_{12}\mathbf{B}_{22}
$$

Các block khác tương tự. Mỗi block đầu ra cộng dồn nhiều tích cục bộ theo hướng $K$. Vì vậy GPU kernel có thể:

1. Lấy một khối nhỏ $\mathbf{A}$ và một khối nhỏ $\mathbf{B}$;
2. Đưa vào shared memory / register trên chip;
3. Tái sử dụng dữ liệu đó để tính nhiều phần tử đầu ra;
4. Lặp theo $K$ và cộng dồn;
5. Cuối cùng ghi block đầu ra về HBM.

Đó là cơ sở toán của tiling. Nó đổi thứ tự tính và cách chuyển dữ liệu, không đổi công thức đích.

### 3.2 Tính toán và truy cập bộ nhớ của GEMM ngây thơ

Với $(M,K)@(K,N)$:

- Số phần tử đầu ra: $MN$;
- Mỗi phần tử làm $K$ lần nhân-cộng;
- FLOPs khoảng $2MKN$.

Triển khai ngây thơ nếu mỗi lần nhân-cộng đều load $A_{ik}$ và $B_{kj}$ từ global memory thì cùng một phần tử bị đọc lại rất nhiều lần. Hiệu năng GEMM đến từ tái sử dụng dữ liệu:

- Một phần tử của $A$ được tái sử dụng trên nhiều cột cùng hàng đầu ra;
- Một phần tử của $B$ được tái sử dụng trên nhiều hàng cùng cột đầu ra;
- Accumulator trong register tái sử dụng $K$ lần.

### 3.3 Ước arithmetic intensity đơn giản

Arithmetic intensity định nghĩa:

$$
\text{Arithmetic Intensity}
= \frac{\text{FLOPs}}{\text{bytes transferred from target memory}}
$$

Lý tưởng, FP16 GEMM mỗi ma trận đầu vào chỉ đọc một lần từ HBM, đầu ra ghi một lần; số byte thô khoảng:

$$
2(MK+KN+MN)
$$

Vậy:

$$
AI \approx
\frac{2MKN}{2(MK+KN+MN)}
$$

Thực tế còn cache, căn chỉnh, read-modify-write, kết quả tạm và phép fused. Ước này không phải mô hình hiệu năng chính xác, nhưng giúp phán toán tử nghiêng về bị giới hạn bởi compute hay bởi băng thông.

### 3.4 Tile không phải càng lớn càng tốt

Tile lớn hơn tăng tái sử dụng dữ liệu, nhưng cũng tiêu nhiều hơn:

- Shared memory;
- Register;
- Thread và đồng bộ mỗi block;
- Chi phí xử lý biên.

Tài nguyên quá cao làm giảm số block/warp một SM giữ được đồng thời, hại latency hiding. Chọn tile là đánh đổi giữa tái sử dụng dữ liệu, occupancy, hiệu quả lệnh và khớp shape.

### 3.5 Tail block và padding

Khi $M,N,K$ không chia hết cho tile size, khối cuối vượt biên hợp lệ. Cách thường gặp:

- Thêm kiểm tra biên / mask;
- Padding đầu vào tới kích thước căn chỉnh;
- Fast path cho shape đẹp thường gặp, shape khác đi đường generic.

Padding tăng tính toán và lưu trữ; nhánh tăng chi phí điều khiển. Cách nào nhanh hơn phụ thuộc shape và phần cứng.

### 3.6 Vì sao kết quả tiling dấu phẩy động có thể khác

Cộng số thực có tính kết hợp, nhưng cộng dấu phẩy động không nghiêm ngặt:

$$
(a+b)+c \ne a+(b+c)
$$

Tile khác, cây reduce của thread khác, đường Tensor Core khác sẽ đổi thứ tự cộng dồn, nên kết quả tối ưu và tham chiếu có thể không khớp từng bit. Kiểm đúng đắn nên dùng dung sai lỗi phù hợp dtype và quy mô bài toán, và để ý lỗi có phóng đại có hệ thống theo độ dài reduce hay không.

---

## 4. Xác suất: từ biến ngẫu nhiên đến phân phối mô hình

### 4.1 Biến ngẫu nhiên và phân phối xác suất

Biến ngẫu nhiên ánh xạ kết quả ngẫu nhiên thành số. Biến rời rạc dùng hàm khối xác suất:

$$
p_X(x)=P(X=x),\qquad \sum_x p_X(x)=1
$$

Biến liên tục dùng hàm mật độ xác suất:

$$
p_X(x)\ge0,\qquad
\int_{-\infty}^{\infty}p_X(x)\,dx=1
$$

Xác suất tại một điểm của biến liên tục thường bằng 0; xác suất trên khoảng lấy bằng tích phân mật độ. Phân phối hay gặp trong mô hình:

| Phân phối | Giá trị | Tình huống điển hình |
|------|------|----------|
| Bernoulli | $0/1$ | Phân loại nhị phân, Dropout mask |
| Categorical | $1,\ldots,V$ | Phân phối token tiếp theo |
| Uniform | Khoảng / tập hữu hạn | Khởi tạo, lấy mẫu ngẫu nhiên |
| Gaussian | Số thực | Khởi tạo, nhiễu, phân tích xấp xỉ |

### 4.2 Xác suất đồng thời, biên và có điều kiện

Xác suất đồng thời mô tả nhiều sự kiện xảy ra cùng lúc:

$$
P(X=x,Y=y)
$$

Cộng hoặc tích phân hết biến kia cho xác suất biên:

$$
P(X=x)=\sum_y P(X=x,Y=y)
$$

Xác suất có điều kiện:

$$
P(X=x\mid Y=y)
= \frac{P(X=x,Y=y)}{P(Y=y)}
$$

Language model tự hồi quy phân tích xác suất chuỗi bằng quy tắc chuỗi:

$$
P(x_1,x_2,\ldots,x_T)
= \prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1})
$$

Điều đó giải thích vì sao next-token prediction lúc huấn luyện và sinh từng token lúc suy luận là hai quá trình của cùng một mô hình xác suất.

### 4.3 Độc lập và độc lập có điều kiện

Nếu:

$$
P(X,Y)=P(X)P(Y)
$$

thì $X,Y$ độc lập. Độc lập mạnh hơn “không tương quan”; hiệp phương sai bằng 0 chưa chắc độc lập.

Độc lập có điều kiện cho $Z$ viết:

$$
P(X,Y\mid Z)=P(X\mid Z)P(Y\mid Z)
$$

Lấy mẫu phân tán, data parallel và ước lượng thống kê thường ngầm giả định i.i.d. Dữ liệu thật có thể phá giả định đó bằng mẫu trùng, lệch shard hoặc tương quan chuỗi.

### 4.4 Công thức Bayes

$$
P(X\mid Y)
= \frac{P(Y\mid X)P(X)}{P(Y)}
$$

Trong đó $P(X)$ là prior, $P(Y\mid X)$ là likelihood, $P(X\mid Y)$ là posterior. Huấn luyện LLM thông thường không tự tính tay Bayes, nhưng đó vẫn là nền của suy luận xác suất, ước lượng tham số và mô hình hóa bất định.

### 4.5 Kỳ vọng: trung bình có trọng số xác suất

Biến rời rạc:

$$
\mathbb{E}[X]=\sum_x xP(X=x)
$$

Biến liên tục:

$$
\mathbb{E}[X]=\int x p(x)\,dx
$$

Kỳ vọng có tính tuyến tính, không đòi biến độc lập:

$$
\mathbb{E}[aX+bY]
= a\mathbb{E}[X]+b\mathbb{E}[Y]
$$

Mục tiêu huấn luyện thường là kỳ vọng loss trên phân phối dữ liệu:

$$
\min_\theta
\mathbb{E}_{(x,y)\sim p_{data}}
[\mathcal{L}(f_\theta(x),y)]
$$

Không thể duyệt phân phối thật, nên xấp xỉ bằng trung bình mẫu mini-batch.

### 4.6 Phương sai và độ lệch chuẩn

Phương sai đo dao động quanh trung bình:

$$
\operatorname{Var}(X)
= \mathbb{E}[(X-\mu)^2]
= \mathbb{E}[X^2]-\mu^2
$$

Độ lệch chuẩn cùng đơn vị với biến gốc:

$$
\sigma = \sqrt{\operatorname{Var}(X)}
$$

Nếu $X,Y$ độc lập:

$$
\operatorname{Var}(X+Y)
= \operatorname{Var}(X)+\operatorname{Var}(Y)
$$

Nhiễu gradient lúc huấn luyện, thang khởi tạo, chuẩn hóa và sai số lượng tử hóa đều liên quan tới phương sai.

### 4.7 Hiệp phương sai và hệ số tương quan

Hiệp phương sai đo hai biến cùng biến thiên:

$$
\operatorname{Cov}(X,Y)
= \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
$$

Chuẩn hóa được hệ số tương quan:

$$
\rho_{X,Y}
= \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
$$

Với biến ngẫu nhiên dạng vector, ma trận hiệp phương sai là:

$$
\mathbf{\Sigma}
= \mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})
(\mathbf{x}-\boldsymbol{\mu})^\top]
$$

Nó đối xứng bán xác định dương. PCA phân tích riêng ma trận hiệp phương sai để tìm hướng trực giao có phương sai lớn nhất.

### 4.8 Trung bình mẫu, phương sai và mini-batch

Cho mẫu $x_1,\ldots,x_n$, trung bình mẫu:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

Phương sai mẫu không chệch thường dùng:

$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

“Phương sai” trong toán tử deep learning chia cho $n$ hay $n-1$ phụ thuộc định nghĩa cụ thể. LayerNorm thường dùng dạng tổng thể (chia cho số phần tử); thư viện thống kê có thể mặc định khác. Khi tái hiện toán tử hãy xem ngữ nghĩa API, đừng chỉ nhìn tên.

### 4.9 Maximum likelihood và negative log-likelihood

Cho dữ liệu $D=\{(x_i,y_i)\}$, ước lượng maximum likelihood chọn tham số làm dữ liệu quan sát khả dĩ nhất:

$$
\theta^*
= \arg\max_\theta
\prod_i p_\theta(y_i\mid x_i)
$$

Tích dễ underflow số và bất tiện khi lấy đạo hàm. Lấy log biến tích thành tổng:

$$
\theta^*
= \arg\max_\theta
\sum_i \log p_\theta(y_i\mid x_i)
$$

Tương đương tối thiểu hóa negative log-likelihood (NLL):

$$
\mathcal{L}_{NLL}
= -\sum_i \log p_\theta(y_i\mid x_i)
$$

Logarithm là cầu nối quan trọng giữa tính toán số và mô hình xác suất.

---

## 5. Softmax, cross-entropy và KL divergence

### 5.1 Logits không phải xác suất

Lớp cuối mô hình xuất $V$ số thực tùy ý:

$$
\mathbf{z}=(z_1,z_2,\ldots,z_V)
$$

Các giá trị này gọi là logits. Chúng có thể âm và không cần cộng bằng 1. Softmax biến chúng thành phân phối categorical:

$$
p_i = \operatorname{softmax}(\mathbf{z})_i
= \frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

Rõ ràng $p_i>0$ và $\sum_i p_i=1$.

Softmax bất biến với dịch chuyển đồng nhất:

$$
\operatorname{softmax}(\mathbf{z}+c)
= \operatorname{softmax}(\mathbf{z})
$$

vì tử và mẫu đều nhân thêm $e^c$. Tính chất này chính là chỗ dựa của triển khai ổn định.

### 5.2 Softmax ổn định số

Tính thẳng $e^{1000}$ sẽ overflow với số dấu phẩy động hữu hạn. Đặt:

$$
m=\max_j z_j
$$

Dạng ổn định:

$$
p_i=
\frac{e^{z_i-m}}
{\sum_j e^{z_j-m}}
$$

Đầu vào mũ lớn nhất là 0 nên mũ lớn nhất bằng 1; các giá trị khác không vượt quá 1. Pseudocode:

```python
def stable_softmax(x):
    maximum = max(x)
    exps = [exp(value - maximum) for value in x]
    denominator = sum(exps)
    return [value / denominator for value in exps]
```

Trừ max chống overflow chiều xuôi, nhưng hạng rất nhỏ vẫn có thể underflow về 0. Thường nghĩa là so với hạng lớn nhất thì có thể bỏ qua; nếu sau đó còn lấy log, hãy dùng `log_softmax` ổn định trực tiếp, tránh thành 0 rồi `log(0)`.

### 5.3 LogSumExp

Định nghĩa:

$$
\operatorname{LSE}(\mathbf{z})
= \log\sum_j e^{z_j}
$$

Dạng ổn định:

$$
\operatorname{LSE}(\mathbf{z})
= m + \log\sum_j e^{z_j-m}
$$

Vậy:

$$
\log\operatorname{softmax}(\mathbf{z})_i
= z_i - \operatorname{LSE}(\mathbf{z})
$$

Triển khai cross-entropy thường fuse `log_softmax + NLLLoss`, vừa giảm tensor trung gian và truy cập bộ nhớ, vừa tránh đường không ổn định “Softmax rồi mới log”.

### 5.4 Tham số temperature

Softmax với temperature $T>0$:

$$
p_i(T)=
\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

- $T<1$: phóng đại chênh lệch logit, phân phối nhọn hơn;
- $T>1$: thu hẹp chênh lệch, phân phối phẳng hơn;
- $T\to0^+$: dần chỉ chọn logit lớn nhất;
- $T\to\infty$: dần về phân phối đều.

Temperature đổi phân phối sampling, không bằng sửa weight mô hình. Triển khai nên scale logits trước, rồi Softmax ổn định, và cẩn thận với temperature rất nhỏ.

### 5.5 Cross-entropy

Cross-entropy giữa phân phối thật $\mathbf{q}$ và phân phối mô hình $\mathbf{p}$:

$$
H(\mathbf{q},\mathbf{p})
= -\sum_i q_i\log p_i
$$

Nếu nhãn thật one-hot với lớp đúng $y$:

$$
q_y=1,\quad q_{i\ne y}=0
$$

thì:

$$
\mathcal{L}
= -\log p_y
$$

Mô hình gán lớp đúng xác suất càng cao thì loss càng nhỏ. Nếu $p_y=1$, loss bằng 0; nếu $p_y$ gần 0, loss rất lớn.

### 5.6 Entropy thông tin

Entropy của bản thân phân phối:

$$
H(\mathbf{p})
= -\sum_i p_i\log p_i
$$

Entropy đo độ bất định:

- Toàn bộ xác suất tập trung một lớp thì entropy nhỏ nhất;
- $V$ lớp phân bố đều thì entropy lớn nhất, bằng $\log V$.

Cơ số log quyết định đơn vị: log tự nhiên là nat, cơ số 2 là bit. Loss machine learning thường dùng log tự nhiên.

### 5.7 KL divergence

KL divergence từ phân phối $\mathbf{q}$ tới $\mathbf{p}$:

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})
= \sum_i q_i\log\frac{q_i}{p_i}
$$

Nó thỏa:

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})\ge0
$$

và bằng 0 khi hai phân phối trùng. Nhưng thường không đối xứng:

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})
\ne D_{KL}(\mathbf{p}\lVert\mathbf{q})
$$

nên KL divergence không phải khoảng cách theo nghĩa chặt.

Cross-entropy phân tích thành:

$$
H(\mathbf{q},\mathbf{p})
= H(\mathbf{q})
+ D_{KL}(\mathbf{q}\lVert\mathbf{p})
$$

Khi huấn luyện phân phối thật $\mathbf{q}$ cố định, $H(\mathbf{q})$ không phụ thuộc tham số mô hình, nên tối thiểu hóa cross-entropy tương đương tối thiểu hóa KL tương ứng.

KL xuất hiện trong knowledge distillation, khớp phân phối, ràng buộc RLHF/PPO và phân tích speculative decoding. Khi triển khai hãy rõ API nhận xác suất, log-probability hay logits, và chiều của KL.

### 5.8 Perplexity

Nếu NLL trung bình mỗi token là $\bar{L}$, perplexity:

$$
\operatorname{PPL}=e^{\bar{L}}
$$

Trực giác là “số ứng viên hiệu dụng” mô hình đối mặt mỗi bước. Nhưng tokenizer khác, tiền xử lý khác, độ dài ngữ cảnh khác, và có bỏ special token hay không đều làm đổi PPL; không so ngang giữa các thiết lập đánh giá.

### 5.9 Sampling Top-k và Top-p

Sau khi mô hình xuất phân phối, chiến lược giải mã quyết định chọn token thế nào:

- Greedy: chọn token xác suất lớn nhất;
- Top-k: chỉ lấy mẫu trong $k$ token xác suất cao nhất;
- Top-p (nucleus): chọn tập ứng viên nhỏ nhất có xác suất tích lũy ít nhất $p$, chuẩn hóa lại rồi lấy mẫu.

Kích thước tập Top-p đổi theo độ nhọn của phân phối. Triển khai thường lọc logits trước, đặt hạng bị loại thành âm vô cực, rồi Softmax ổn định và lấy mẫu.

Suy luận phân tán nếu vocabulary bị cắt theo tensor parallelism thì global top-k/top-p cần gom ứng viên cục bộ giữa các thiết bị, hoặc dùng thuật toán phân tán tương đương.

---

## 6. Giải tích và lan truyền ngược

### 6.1 Đạo hàm: tốc độ biến thiên cục bộ

Đạo hàm hàm một biến:

$$
f'(x)=
\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

Gần $x$, thay đổi nhỏ của đầu vào ảnh hưởng đầu ra:

$$
f(x+\Delta x)
\approx f(x)+f'(x)\Delta x
$$

Đạo hàm thường gặp:

$$
\frac{d}{dx}x^n=nx^{n-1}
$$

$$
\frac{d}{dx}e^x=e^x
$$

$$
\frac{d}{dx}\log x=\frac{1}{x}
$$

### 6.2 Đạo hàm riêng và gradient

Với hàm scalar nhiều biến $f(x_1,\ldots,x_n)$, tốc độ đổi theo một biến là đạo hàm riêng. Xếp mọi đạo hàm riêng thành vector được gradient:

$$
\nabla_{\mathbf{x}}f=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

Gradient chỉ hướng tăng nhanh nhất, nên gradient descent cập nhật theo hướng ngược:

$$
\mathbf{x}_{t+1}
= \mathbf{x}_t-\eta\nabla f(\mathbf{x}_t)
$$

$\eta$ là learning rate.

### 6.3 Jacobian và Hessian

Với hàm vector $\mathbf{y}=f(\mathbf{x})$, Jacobian:

$$
\mathbf{J}_{ij}
= \frac{\partial y_i}{\partial x_j}
$$

Nó mô tả ảnh hưởng cục bộ của từng thành phần đầu vào lên từng thành phần đầu ra.

Hessian của hàm scalar là ma trận đạo hàm riêng bậc hai:

$$
\mathbf{H}_{ij}
= \frac{\partial^2 f}
{\partial x_i\partial x_j}
$$

Hessian mô tả độ cong cục bộ. Số tham số deep learning quá lớn nên thường không dựng Hessian đầy đủ tường minh, nhưng Hessian-vector product, xấp xỉ độ cong và phân tích giá trị riêng vẫn dùng trong nghiên cứu tối ưu và ổn định.

### 6.4 Quy tắc chuỗi

Nếu:

$$
y=f(u),\qquad u=g(x)
$$

thì:

$$
\frac{dy}{dx}
= \frac{dy}{du}\frac{du}{dx}
$$

Mạng nhiều lớp là hàm hợp. Với:

$$
\mathbf{h}_1=f_1(\mathbf{x}),\quad
\mathbf{h}_2=f_2(\mathbf{h}_1),\quad
L=f_3(\mathbf{h}_2)
$$

gradient lan từ loss ngược lại:

$$
\frac{\partial L}{\partial \mathbf{x}}
= \frac{\partial L}{\partial \mathbf{h}_2}
  \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}
  \frac{\partial \mathbf{h}_1}{\partial \mathbf{x}}
$$

Về khái niệm là tích Jacobian; autodiff thực tế không dựng Jacobian khổng lồ mà tính hiệu quả vector-Jacobian product (reverse mode).

### 6.5 Computation graph và autodiff reverse mode

Đặt:

$$
a=xy,\qquad b=a+x,\qquad L=b^2
$$

Giai đoạn xuôi lưu các lượng trung gian cần thiết. Ngược bắt đầu từ:

$$
\frac{\partial L}{\partial L}=1
$$

rồi:

$$
\frac{\partial L}{\partial b}=2b
$$

Vì $b=a+x$:

$$
\frac{\partial L}{\partial a}=2b
$$

$x$ ảnh hưởng $L$ theo hai đường, gradient phải cộng:

$$
\frac{\partial L}{\partial x}
= \frac{\partial L}{\partial b}\frac{\partial b}{\partial x}
+ \frac{\partial L}{\partial a}\frac{\partial a}{\partial x}
= 2b + 2by
$$

$$
\frac{\partial L}{\partial y}=2bx
$$

Điều đó giải thích vì sao một tensor dùng ở nhiều chỗ trên graph phải tích lũy gradient.

### 6.6 Lan truyền ngược qua linear layer

Dùng quy ước hàng là batch:

$$
\mathbf{Y}=\mathbf{X}\mathbf{W}+\mathbf{b}
$$

với:

$$
\mathbf{X}\in\mathbb{R}^{B\times d_{in}},\quad
\mathbf{W}\in\mathbb{R}^{d_{in}\times d_{out}},\quad
\mathbf{Y}\in\mathbb{R}^{B\times d_{out}}
$$

Cho gradient thượng nguồn:

$$
\mathbf{G}=\frac{\partial L}{\partial \mathbf{Y}}
\in\mathbb{R}^{B\times d_{out}}
$$

thì:

$$
\frac{\partial L}{\partial \mathbf{X}}
= \mathbf{G}\mathbf{W}^\top
\in\mathbb{R}^{B\times d_{in}}
$$

$$
\frac{\partial L}{\partial \mathbf{W}}
= \mathbf{X}^\top\mathbf{G}
\in\mathbb{R}^{d_{in}\times d_{out}}
$$

$$
\frac{\partial L}{\partial \mathbf{b}}
= \sum_{i=1}^{B}\mathbf{G}_{i,:}
\in\mathbb{R}^{d_{out}}
$$

Chỉ nhìn shape cũng kiểm nhanh được công thức:

```text
dX: (B, dout) @ (dout, din) -> (B, din)
dW: (din, B) @ (B, dout) -> (din, dout)
db: reduce theo chiều batch -> (dout,)
```

Chiều ngược của một linear layer thường gồm hai GEMM và một lần reduce; đó cũng là một lý do compute huấn luyện cao hơn hẳn một lần forward.

### 6.7 Jacobian của Softmax

Softmax:

$$
p_i=\frac{e^{z_i}}{\sum_k e^{z_k}}
$$

Đạo hàm riêng:

$$
\frac{\partial p_i}{\partial z_j}
= p_i(\delta_{ij}-p_j)
$$

Viết dạng ma trận:

$$
\mathbf{J}_{softmax}
= \operatorname{diag}(\mathbf{p})
- \mathbf{p}\mathbf{p}^\top
$$

Mỗi đầu ra Softmax phụ thuộc mọi đầu vào, nên Jacobian không chéo. Nhưng backward thực tế vẫn tính trong $O(V)$ mà không cần materialize ma trận $V\times V$.

### 6.8 Gradient gọn của Softmax + cross-entropy

Nhãn one-hot $\mathbf{y}$, loss:

$$
L=-\sum_i y_i\log p_i
$$

Lấy đạo hàm kết hợp với Softmax được:

$$
\frac{\partial L}{\partial z_i}=p_i-y_i
$$

Đây là một trong những rút gọn quan trọng nhất của deep learning. Với lớp đúng, gradient là $p_y-1$; với lớp khác là $p_i$. Dự đoán càng sai, lực đẩy sửa logits càng mạnh.

Kernel cross-entropy fused có thể chia sẻ max và kết quả reduce, giảm tensor xác suất trung gian, và sinh thẳng gradient cần thiết.

### 6.9 Kiểm gradient

Triển khai autodiff có thể kiểm ở quy mô nhỏ bằng sai phân hữu hạn:

$$
\frac{\partial f}{\partial x_i}
\approx
\frac{f(\mathbf{x}+\epsilon\mathbf{e}_i)
-f(\mathbf{x}-\epsilon\mathbf{e}_i)}{2\epsilon}
$$

Sai phân trung tâm thường chính xác hơn một phía, nhưng $\epsilon$ không được quá lớn hay quá nhỏ: quá lớn có truncation error, quá nhỏ bị sai số làm tròn dấu phẩy động át. Kiểm gradient hợp kích thước nhỏ, double precision và hàm xác định; không hợp kiểm trực tiếp job huấn luyện có ngẫu nhiên hoặc tensor khổng lồ.

---

## 7. Tối ưu, ổn định gradient và chuẩn hóa

### 7.1 Gradient toàn batch, SGD và mini-batch

Rủi ro thực nghiệm:

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}L_i(\theta)
$$

Gradient toàn batch mỗi bước dùng mọi mẫu, chi phí cao. Gradient mini-batch:

$$
\hat{g}
= \frac{1}{B}\sum_{i\in\mathcal{B}}
\nabla_\theta L_i(\theta)
$$

là ước lượng ngẫu nhiên của gradient đầy đủ. Cập nhật SGD cơ bản:

$$
\theta_{t+1}=\theta_t-\eta\hat{g}_t
$$

Batch lớn hơn thường giảm phương sai ước lượng gradient và tăng tận dụng phần cứng, nhưng cần thêm bộ nhớ activation, và siêu tham số tối ưu có thể phải chỉnh theo.

### 7.2 Momentum

Một dạng momentum thường gặp:

$$
\mathbf{v}_t
= \beta\mathbf{v}_{t-1}+(1-\beta)\mathbf{g}_t
$$

$$
\theta_{t+1}
= \theta_t-\eta\mathbf{v}_t
$$

Momentum lấy trung bình mũ của gradient lịch sử, giảm nhiễu ngắn hạn, và tăng tốc trên hướng giữ nhất quán.

### 7.3 Trực giác thích ứng thang của Adam

Adam đồng thời giữ EMA của moment bậc một và bậc hai:

$$
\mathbf{m}_t
= \beta_1\mathbf{m}_{t-1}
+(1-\beta_1)\mathbf{g}_t
$$

$$
\mathbf{v}_t
= \beta_2\mathbf{v}_{t-1}
+(1-\beta_2)\mathbf{g}_t^2
$$

Sau hiệu chỉnh bias:

$$
\theta_{t+1}
= \theta_t
-\eta\frac{\hat{\mathbf{m}}_t}
{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}
$$

Bình phương và căn đều từng phần tử. Adam lưu thêm state cho mỗi tham số; nếu state dùng FP32, chi phí bộ nhớ thường là phần lớn của huấn luyện mô hình lớn, và đó cũng là lý do có kỹ thuật sharding như ZeRO.

### 7.4 Gradient biến mất và bùng nổ

Gradient của hàm hợp sâu chứa nhiều tích Jacobian:

$$
\frac{\partial L}{\partial \mathbf{h}_0}
= \frac{\partial L}{\partial \mathbf{h}_L}
\prod_{l=1}^{L}
\frac{\partial \mathbf{h}_l}
{\partial \mathbf{h}_{l-1}}
$$

Nếu các phép biến đổi trên hướng liên quan lâu dài có thang nhỏ hơn 1, gradient suy mũ; lâu dài lớn hơn 1, gradient tăng mũ.

Triệu chứng kỹ thuật gồm:

- Chuẩn gradient gần 0, lớp sớm hầu như không cập nhật;
- Chuẩn gradient đột ngột rất lớn, loss thành `inf`/`nan`;
- Mixed precision kích hoạt underflow hoặc overflow sớm hơn.

Cách giảm gồm khởi tạo hợp lý, residual connection, chuẩn hóa, gradient clipping, learning rate phù hợp và dtype ổn định.

### 7.5 Vì sao residual connection giúp lan truyền gradient

Khối residual:

$$
\mathbf{y}=\mathbf{x}+F(\mathbf{x})
$$

có Jacobian:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
= \mathbf{I}+\frac{\partial F}{\partial \mathbf{x}}
$$

Đường identity cung cấp lối gradient không phụ thuộc hoàn toàn vào $F$. Nó không đảm bảo mãi mãi không bùng nổ/biến mất, nhưng cải thiện rõ điều kiện tối ưu của mạng sâu.

### 7.6 Khởi tạo và lan truyền phương sai

Xét:

$$
y=\sum_{i=1}^{n}w_i x_i
$$

Nếu $w_i,x_i$ độc lập, kỳ vọng 0:

$$
\operatorname{Var}(y)
= n\operatorname{Var}(w)\operatorname{Var}(x)
$$

Để phương sai activation các lớp tương đối ổn định, đặt $\operatorname{Var}(w)$ cùng bậc $1/n$. Xavier, Kaiming chọn thang theo fan-in, fan-out và hàm activation.

Không phải càng nhỏ càng ổn định: quá nhỏ làm tín hiệu và gradient suy; quá lớn gây bão hòa hoặc bùng nổ.

### 7.7 LayerNorm

Với hidden vector của một token $\mathbf{x}\in\mathbb{R}^{H}$:

$$
\mu=\frac{1}{H}\sum_{i=1}^{H}x_i
$$

$$
\sigma^2=\frac{1}{H}
\sum_{i=1}^{H}(x_i-\mu)^2
$$

$$
\operatorname{LayerNorm}(x_i)
= \gamma_i
\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
+\beta_i
$$

LayerNorm reduce theo chiều ẩn, không phụ thuộc mẫu khác trong batch, nên hợp chuỗi độ dài biến thiên và mô hình tự hồi quy.

Góc nhìn kernel: ít nhất gồm reduce trung bình, reduce phương sai và biến đổi từng phần tử. Triển khai ngây thơ đọc/ghi HBM nhiều lần; bản tối ưu fuse reduce với phép affine và xử lý reduce song song trên vector dài.

### 7.8 RMSNorm

RMSNorm không trừ trung bình, chỉ scale theo căn quân phương:

$$
\operatorname{RMS}(\mathbf{x})
= \sqrt{\frac{1}{H}\sum_i x_i^2+\epsilon}
$$

$$
\operatorname{RMSNorm}(x_i)
= \gamma_i\frac{x_i}{\operatorname{RMS}(\mathbf{x})}
$$

Nó bỏ phần tính liên quan trung bình, nhưng “ít công thức hơn” không tự động đồng nghĩa tăng tốc end-to-end theo cùng tỉ lệ. Lợi ích thực phụ thuộc fusion, truy cập bộ nhớ và bottleneck tổng thể của mô hình.

### 7.9 Gradient clipping

Clipping theo chuẩn L2 toàn cục:

$$
\mathbf{g}\leftarrow
\mathbf{g}\cdot
\min\left(1,\frac{c}{\lVert\mathbf{g}\rVert_2+\epsilon}\right)
$$

Khi chuẩn gradient vượt ngưỡng $c$, toàn bộ bị thu theo tỉ lệ, hướng giữ nguyên. Trong huấn luyện phân tán, nếu gradient bị shard, tính chuẩn toàn cục cần gom tổng bình phương cục bộ giữa các rank:

$$
\lVert\mathbf{g}\rVert_2
= \sqrt{\sum_r\sum_{i\in r}g_i^2}
$$

Vậy một thao tác toán trông đơn giản cũng có thể kéo theo collective communication.

---

## 8. Tính toán số và mixed precision

### 8.1 Số dấu phẩy động là tập hữu hạn

Biểu diễn điển hình của số nhị phân dấu phẩy động:

$$
(-1)^s \times \text{significand}
\times 2^{\text{exponent}}
$$

Bit được chia cho dấu, số mũ và phần định trị (mantissa). Bit mũ quyết định dải động; bit định trị quyết định độ mịn giữa các số biểu diễn được kề nhau.

| Định dạng | Tổng bit | Bit mũ | Bit trường định trị | Đặc điểm chính |
|------|--------|--------|------------|----------|
| FP32 | 32 | 8 | 23 | Dải và độ chính xác cân bằng hơn |
| TF32 | 19 (ngữ nghĩa đầu vào nhân) | 8 | 10 | Giữ dải FP32, giảm độ chính xác nhân, thường cộng dồn FP32 |
| FP16 | 16 | 5 | 10 | Độ chính xác còn chấp nhận được nhưng dải động nhỏ hơn |
| BF16 | 16 | 8 | 7 | Dải động gần FP32, độ chính xác thấp hơn FP16 |
| FP8 E4M3 | 8 | 4 | 3 | Ưu tiên độ chính xác, dải nhỏ hơn; mã hóa cụ thể theo đặc tả |
| FP8 E5M2 | 8 | 5 | 2 | Ưu tiên dải, độ chính xác thấp hơn nữa; mã hóa cụ thể theo đặc tả |

Bảng không tính bit dẫn ẩn trong trường định trị. TF32 thường là chế độ tính của NVIDIA Tensor Core, không phải dtype lưu trong bộ nhớ thông thường.

### 8.2 Độ chính xác và dải động là hai chuyện khác nhau

FP16 có trường định trị rộng hơn BF16, nên ở cùng bậc độ lớn thường biểu diễn được chênh lệch mịn hơn; nhưng FP16 ít bit mũ hơn, giá trị hữu hạn lớn nhất chỉ 65504, và dải số khác không cũng hẹp hơn nhiều.

BF16 đổi một phần độ chính xác lấy cùng số bit mũ với FP32, nên lúc huấn luyện thường ít overflow/underflow hơn. Nó vẫn có sai số làm tròn rõ, nhất là khi cộng increment rất nhỏ vào số lớn.

### 8.3 Làm tròn và độ chính xác máy

Nhiều số thập phân không biểu diễn chính xác bằng dấu phẩy động nhị phân:

```python
>>> 0.1 + 0.2 == 0.3
False
```

Khoảng cách giữa các số float tăng theo giá trị tuyệt đối. Nếu $|\delta|$ nhỏ hơn nhiều so với khoảng biểu diễn được quanh $x$ hiện tại:

$$
\operatorname{fl}(x+\delta)=x
$$

Đó là vì sao optimizer thường giữ FP32 master weights: cập nhật quá nhỏ trên weight độ chính xác thấp có thể bị làm tròn mất hẳn.

### 8.4 Không kết hợp và thứ tự reduce

Vì mỗi bước đều làm tròn:

$$
\operatorname{fl}(\operatorname{fl}(a+b)+c)
\ne
\operatorname{fl}(a+\operatorname{fl}(b+c))
$$

Reduce song song dùng cộng dạng cây; số thread khác, chia block khác và thứ tự atomic khác có thể cho kết quả hơi khác. Không xác định không tự động nghĩa là lỗi triển khai, nhưng phải nằm trong sai số chấp nhận được và khớp yêu cầu tái hiện của bài toán.

Cách nâng độ chính xác reduce gồm:

- Cộng dồn ở độ chính xác cao hơn;
- Cộng từng cặp / dạng cây, tránh cộng lâu dài các thang cực đoan;
- Tổng bù như Kahan (thêm chi phí lệnh);
- Reduce cục bộ trước, rồi gộp tổng riêng phần.

### 8.5 Overflow, underflow và giá trị không hữu hạn

- **overflow**: trị tuyệt đối vượt giá trị hữu hạn lớn nhất, thường thành $\pm\infty$;
- **underflow**: kết quả quá gần 0, vào subnormal hoặc làm tròn về 0;
- `NaN`: ví dụ $0/0$, $\infty-\infty$ hoặc phép bất hợp lệ;
- `inf` tham gia phép sau có thể lan nhanh thành `NaN`.

Khi loss thành `NaN`, hãy tìm tensor không hữu hạn **đầu tiên**, đừng chỉ nhìn loss cuối. Nguồn thường gặp: logits quá lớn, chia cho 0, căn số âm, mask sai, gradient overflow và biên kernel tùy chỉnh sai.

### 8.6 Catastrophic cancellation

Hai số lớn gần bằng nhau trừ nhau sẽ triệt chữ số có nghĩa cao; sai số tương đối của phần còn lại có thể rất lớn. Ví dụ tính phương sai bằng:

$$
\operatorname{Var}(X)
= \mathbb{E}[X^2]-\mathbb{E}[X]^2
$$

hai hạng có thể đều lớn và gần nhau, trừ xong mất nhiều độ chính xác.

Cách ổn định hơn gồm two-pass variance hoặc thuật toán online Welford. Cách sau cập nhật dần trung bình và độ lệch bình phương:

$$
n \leftarrow n+1
$$

$$
\delta = x_n-\mu_{n-1}
$$

$$
\mu_n = \mu_{n-1}+\frac{\delta}{n}
$$

$$
M_{2,n}=M_{2,n-1}+\delta(x_n-\mu_n)
$$

Phương sai tổng thể là $M_{2,n}/n$. Welford còn gộp được thống kê từ các khối khác nhau, hợp reduce song song.

### 8.7 Condition number: sai số đầu vào bị phóng đại bao nhiêu

Condition number của bài toán mô tả ảnh hưởng của nhiễu nhỏ ở đầu vào lên đầu ra. Với ma trận khả nghịch, trong một chuẩn nhất quán:

$$
\kappa(\mathbf{A})
= \lVert\mathbf{A}\rVert
  \lVert\mathbf{A}^{-1}\rVert
$$

Condition number lớn nghĩa là bài toán ill-conditioned: dù thuật toán đúng và đầu vào chỉ có sai số làm tròn nhỏ, đầu ra vẫn có thể đổi mạnh.

Cần tách:

- **bản thân bài toán có ill-conditioned không**;
- **thuật toán chọn có ổn định số không**.

Thêm bit giảm được sai số làm tròn, nhưng không triệt tiêu độ nhạy đến từ ill-conditioning.

### 8.8 Mẫu cơ bản của huấn luyện mixed precision

Mixed precision không phải “đổi hết sang FP16”. Mẫu thường gặp:

1. Nhân ma trận lớn dùng đầu vào FP16/BF16, tăng tốc bằng Tensor Core;
2. Nhân-cộng hoàn tất ở FP32 hoặc độ chính xác cộng dồn cao hơn được chỉ định;
3. Phép nhạy như reduce, chuẩn hóa, Softmax giữ hoặc nâng nội bộ lên FP32;
4. Optimizer state và/hoặc master weights giữ FP32;
5. Chọn output dtype theo phần cứng và toán tử.

Chiến lược cụ thể khác nhau theo framework và phần cứng; hãy xem tài liệu toán tử thật và profiler, đừng đoán đường tính bên trong từ dtype đầu vào.

### 8.9 Vì sao Loss Scaling hữu ích

Gradient FP16 nhỏ trong backward có thể underflow về 0. Với hệ số scale $S$:

$$
L' = S L
$$

thì:

$$
\nabla L'=S\nabla L
$$

Gradient được phóng trước vào dải FP16 biểu diễn được; sau backward chia lại cho $S$:

$$
\nabla L=\frac{\nabla L'}{S}
$$

Dynamic Loss Scaling sẽ:

1. Chạy backward với scale hiện tại;
2. Kiểm tra gradient có `inf`/`NaN` không;
3. Nếu overflow, bỏ cập nhật và giảm scale;
4. Sau vài bước ổn định liên tiếp, thử tăng scale.

Loss Scaling chủ yếu chống underflow gradient nhỏ; scale quá lớn lại gây overflow. BF16 dải động lớn hơn, thường không phụ thuộc Loss Scaling, nhưng mô hình vẫn có thể sinh giá trị không hữu hạn vì bản thân không ổn định.

### 8.10 Thuật toán ổn định thường cũng hợp fusion hơn

Ổn định số và tối ưu hiệu năng không phải lúc nào cũng xung đột:

- Stable Softmax reduce max trước, rồi exp/sum; Online Softmax có thể gộp state theo block;
- Fuse `log_softmax + NLL` giảm ghi xác suất trung gian;
- Welford giữ thống kê hợp reduce theo khối của LayerNorm;
- GEMM đầu vào độ chính xác thấp, cộng dồn độ chính xác cao cân bằng throughput và sai số.

Điểm then chốt là chọn dạng toán có state kết hợp được, để kernel xử lý theo tile trong bộ nhớ trên chip hữu hạn.

---

## 9. Ví dụ tổng hợp AI Infra

### 9.1 Ví dụ 1: Shape và chi phí linear layer Transformer

Đầu vào:

$$
\mathbf{X}\in\mathbb{R}^{B\times S\times H}
$$

Weight:

$$
\mathbf{W}\in\mathbb{R}^{H\times 4H}
$$

Làm phẳng hai chiều đầu:

$$
\mathbf{X}'\in\mathbb{R}^{(BS)\times H}
$$

Đầu ra:

$$
\mathbf{Y}'=\mathbf{X}'\mathbf{W}
\in\mathbb{R}^{(BS)\times 4H}
$$

rồi reshape thành $(B,S,4H)$. FLOPs:

$$
2\cdot(BS)\cdot H\cdot4H
=8BSH^2
$$

Nếu $B=8,S=2048,H=4096$:

$$
8\times8\times2048\times4096^2
\approx 2.20\times10^{12}\ \text{FLOPs}
$$

Đó mới chỉ một phép chiếu linear forward, cho thấy vì sao mô hình lớn phụ thuộc mạnh vào Tensor Core GEMM.

### 9.2 Ví dụ 2: Đầy đủ chiều multi-head Attention

Từ:

$$
\mathbf{X}\in\mathbb{R}^{B\times S\times H}
$$

sinh Q/K/V rồi reshape và chuyển vị:

$$
\mathbf{Q},\mathbf{K},\mathbf{V}
\in\mathbb{R}^{B\times N_h\times S\times D_h}
$$

với $H=N_hD_h$.

Ma trận điểm:

$$
\mathbf{S}
=\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D_h}}
\in\mathbb{R}^{B\times N_h\times S\times S}
$$

Áp causal mask và Softmax:

$$
\mathbf{P}=\operatorname{softmax}(\mathbf{S}+\mathbf{M})
$$

trong đó vị trí tương lai được mask logic thành $-\infty$ để xác suất Softmax bằng 0.

Đầu ra:

$$
\mathbf{O}=\mathbf{P}\mathbf{V}
\in\mathbb{R}^{B\times N_h\times S\times D_h}
$$

Gộp head rồi về $(B,S,H)$.

Nếu materialize tường minh $\mathbf{S}$ kiểu FP16, chỉ nó đã cần:

$$
2BN_hS^2\ \text{bytes}
$$

Khi $B=1,N_h=32,S=32768$:

$$
2\times1\times32\times32768^2
=64\ \text{GiB}
$$

và đó chưa tính xác suất, lượng trung gian backward và các lớp khác. Ý chính của FlashAttention là không ghi ma trận trung gian $S\times S$ đầy đủ về HBM, mà hoàn tất Softmax ổn định và cộng dồn $PV$ theo tile trên chip.

### 9.3 Ví dụ 3: Vì sao chia $\sqrt{D_h}$

Giả sử $q_i,k_i$ độc lập, kỳ vọng 0, phương sai 1:

$$
\mathbf{q}^\top\mathbf{k}
=\sum_{i=1}^{D_h}q_i k_i
$$

Mỗi hạng $q_i k_i$ kỳ vọng khoảng 0, phương sai khoảng 1, nên tổng có phương sai khoảng $D_h$, độ lệch chuẩn khoảng $\sqrt{D_h}$.

Chia $\sqrt{D_h}$ đưa phương sai điểm về khoảng 1:

$$
\operatorname{Var}\left(
\frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{D_h}}
\right)\approx1
$$

Tránh logits nở quá khi chiều tăng và Softmax rơi vào vùng bão hòa cực đoan.

### 9.4 Ví dụ 4: Gộp block Online Softmax

Một hàng logits bị cắt thành nhiều block. Phần đã xử lý có max và tổng mũ $(m,l)$:

$$
m=\max_i x_i,\qquad
l=\sum_i e^{x_i-m}
$$

State block mới là $(m_b,l_b)$. Gộp max:

$$
m_{new}=\max(m,m_b)
$$

Đưa về cùng mốc rồi gộp tổng mũ:

$$
l_{new}
= e^{m-m_{new}}l
+e^{m_b-m_{new}}l_b
$$

Nhờ đó Softmax tính được theo tile, streaming và ổn định số. Nếu đồng thời giữ accumulator Value có trọng số, còn tránh lưu ma trận attention đầy đủ—một lõi toán của FlashAttention.

### 9.5 Ví dụ 5: Tham số và tính toán của LoRA

Linear layer đầy đủ:

$$
\mathbf{y}=\mathbf{x}\mathbf{W}_0^\top
$$

LoRA:

$$
\mathbf{y}
=\mathbf{x}\mathbf{W}_0^\top
+\frac{\alpha}{r}
  (\mathbf{x}\mathbf{A}^\top)\mathbf{B}^\top
$$

Shape:

```text
x:     (..., din)
A^T:   (din, r)       -> (..., r)
B^T:   (r, dout)      -> (..., dout)
```

Adapter online thêm hai GEMM nhỏ; nếu gộp sẵn $\mathbf{B}\mathbf{A}$ vào weight nền, suy luận vẫn là một GEMM lớn nhưng mất tiện ích chuyển adapter động với chi phí thấp.

### 9.6 Ví dụ 6: Vì sao trung bình phân tán phải có trọng số

Hai rank có số mẫu cục bộ $n_1,n_2$, trung bình cục bộ $\mu_1,\mu_2$. Trung bình toàn cục không đơn giản là $(\mu_1+\mu_2)/2$, trừ khi $n_1=n_2$. Giá trị đúng:

$$
\mu=
\frac{n_1\mu_1+n_2\mu_2}{n_1+n_2}
$$

Tổng quát hơn, mỗi rank gom `sum` và `count` cục bộ trước, AllReduce rồi mới chia. Loss, metric và thống kê throughput phân tán đều cần mẫu số rõ, nhất là batch cuối không đầy hoặc sequence mask khác nhau.

### 9.7 Ví dụ 7: Ước VRAM

Nếu tensor shape $(B,S,H)$, mỗi phần tử $b$ byte, kích thước lưu liên tục:

$$
M=B\times S\times H\times b
$$

Ví dụ BF16 $(8,4096,8192)$:

$$
8\times4096\times8192\times2
=536{,}870{,}912\ \text{bytes}
=512\ \text{MiB}
$$

Đỉnh VRAM lúc huấn luyện còn gồm tham số, gradient, optimizer state, activation đã lưu, workspace tạm, buffer giao tiếp, phân mảnh bộ nhớ và ngữ cảnh framework. Ước một tensor chỉ là điểm khởi đầu.

### 9.8 Checklist cố định từ công thức tới kernel

Khi gặp toán tử mới, tháo theo thứ tự:

1. Shape của đầu vào, đầu ra và lượng trung gian là gì?
2. Chiều nào giữ, chiều nào reduce?
3. Có tiling được không? State các tile gộp thế nào?
4. FLOPs và lượng truy cập bộ nhớ tối thiểu lý thuyết là bao nhiêu?
5. Có broadcast, chuyển vị hoặc stride không liên tục không?
6. Phép nào nhạy độ chính xác, cần cộng dồn độ chính xác cao?
7. Có thể gặp overflow exp, chia 0, cancellation hoặc sai số reduce dài không?
8. Tensor trung gian có fuse bỏ được không?
9. Shape biên và tail block xử lý thế nào?
10. Dùng bản tham chiếu nào, dung sai dtype nào và đầu vào cực đoan nào để kiểm?

Checklist này biến toán trừu tượng thành workflow Infra có thể thực thi.

---

## Tóm tắt

Chương này dựng bốn mạch chính nối với nhau:

- **Đại số tuyến tính** mô tả tensor biến đổi thế nào: shape quyết định tính hợp lệ, hạng và phân tích quyết định khả năng nén, chia khối quyết định cách tái sử dụng dữ liệu;
- **Xác suất** mô tả ý nghĩa đầu ra mô hình: logits qua Softmax thành phân phối, cross-entropy tương ứng negative log-likelihood, KL đo chênh lệch phân phối;
- **Giải tích** mô tả tham số học thế nào: quy tắc chuỗi lan ngược theo computation graph, backward linear layer vẫn gồm GEMM và reduce;
- **Tính toán số** quyết định công thức trên phần cứng có đáng tin không: độ chính xác hữu hạn sẽ làm tròn, overflow và underflow, nên công thức ổn định và chiến lược mixed precision cũng quan trọng không kém.

Với kỹ sư AI Infra, “hiểu toán” cuối cùng thể hiện ở chỗ: dịch được công thức thành shape, reduce, tiling, chuyển dữ liệu, chọn độ chính xác và tiêu chuẩn kiểm chứng.

## Checklist tự kiểm

- [ ] Thấy `(B, S, H) @ (H, V)` viết ngay được output shape
- [ ] Phân biệt được nhân từng phần tử, tích vô hướng, nhân ma trận và batched matmul
- [ ] Từ shape/stride phán được view chuyển vị có liên tục không
- [ ] Giải thích được trực giác của chuẩn, hạng, giá trị riêng và singular value
- [ ] Tính được tỉ lệ tiết kiệm tham số của phân tích hạng thấp và LoRA
- [ ] Từ nhân ma trận block giải thích được tính đúng của GEMM tiling
- [ ] Ước được FLOPs, lượng truy cập tối thiểu và arithmetic intensity của GEMM
- [ ] Tính được kỳ vọng, phương sai, hiệp phương sai và xác suất có điều kiện
- [ ] Viết được Softmax ổn định và LogSumExp
- [ ] Nói được quan hệ giữa cross-entropy, entropy và KL divergence
- [ ] Dùng quy tắc chuỗi suy gradient trên computation graph đơn giản
- [ ] Viết được shape gradient của linear layer theo đầu vào, weight và bias
- [ ] Biết gradient Softmax + cross-entropy theo logits là `p - y`
- [ ] Giải thích được vì sao residual connection cải thiện đường gradient
- [ ] Phân biệt được độ chính xác, dải động, overflow và underflow
- [ ] Nói được vì sao FP16 thường cần Loss Scaling còn BF16 thường không
- [ ] Giải thích được vì sao reduce song song có thể không khớp từng bit
- [ ] Từ shape Attention suy được lượng trung gian $O(S^2)$ và chi phí VRAM

## Bài tập

1. Suy output shape của `(B, Nh, Sq, Dh) @ (B, Nh, Dh, Sk)`, và đánh dấu chiều batch với chiều reduce.
2. Cho FP16 GEMM $M=4096,K=4096,N=16384$, ước FLOPs và số byte đầu vào/đầu ra tối thiểu lý tưởng.
3. Viết chương trình kiểm $\operatorname{softmax}(\mathbf{z}+c)=\operatorname{softmax}(\mathbf{z})$, và so bản ngây thơ với bản ổn định trên logits lớn.
4. Từ Softmax Jacobian và cross-entropy, tự suy $\partial L/\partial z_i=p_i-y_i$.
5. Suy shape backward của `Y = GELU(XW + b)`; không cần bung đạo hàm cụ thể của GELU.
6. So sai số FP16, BF16 và FP32 khi cộng dãy `[1, 1e-3, 1e-3, ...]`, và thử đổi thứ tự cộng.
7. Cài Welford variance, so với `E[x²] - E[x]²` trên dữ liệu “trung bình lớn, phương sai nhỏ”.
8. Với weight $4096\times4096$, tính tỉ lệ tham số và FLOPs forward thêm của LoRA rank 8, 16, 64.
9. Tính kích thước ma trận Attention score FP16 tường minh khi $B=2,N_h=32,S=8192,D_h=128$.
10. Hai rank xử lý lần lượt 7 và 5 token hợp lệ, trung bình loss cục bộ 2.0 và 3.0; tính trung bình toàn cục đúng, và giải thích vì sao trung bình đơn giản sai.
11. Dùng sai phân hữu hạn kiểm backward linear layer nhỏ viết tay; thử các $\epsilon$ khác nhau và quan sát đường cong sai số.
12. Chọn một toán tử PyTorch, viết một trang phân tích theo “shape, reduce, FLOPs, truy cập bộ nhớ, rủi ro số, cách kiểm”.

## Tài liệu tham khảo

- [Deep Learning Book：Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Deep Learning Book：Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html)
- [Deep Learning Book：Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html)
- [Matrix Calculus for Deep Learning](https://arxiv.org/abs/1802.01528)
- [Automatic Differentiation in Machine Learning: a Survey](https://jmlr.org/papers/v18/17-468.html)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [NVIDIA TensorFloat-32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)
- [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch Broadcasting Semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
