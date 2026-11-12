---
title: "Chapter 2: Math foundations"
description: "Build the linear algebra, probability, and calculus intuition AI Infra needs: the math behind GEMM tiling, Softmax, backpropagation, and mixed-precision training"
tags: ["Linear algebra", "Probability", "Calculus", "Math foundations"]
sourceHash: "eeef6ebebd63bceb"
---

AI Infra does not ask you to derive hard theorems every day. It does ask that, when you see a formula, you can answer three engineering questions right away: **what shape the tensors have, how much compute and memory they need, and how to implement them stably**.

This chapter builds that intuition around those three questions. We start from vectors and matrices and connect them to GEMM and tiling; from probability distributions to Softmax, cross-entropy, and sampling; from the chain rule to backpropagation; and finally floating-point numbers, error, and mixed precision. The goal is not to prove every result. It is to get to the point where you can read a formula, push dimensions through it, estimate the order of magnitude, and judge numerical risk.

<!-- more -->

## 📑 Contents

- [1. Learning goals, notation, and three questions](#1-learning-goals-notation-and-three-questions)
- [2. Linear algebra: tensors, transforms, and decompositions](#2-linear-algebra-tensors-transforms-and-decompositions)
- [3. Block matrices and GEMM engineering intuition](#3-block-matrices-and-gemm-engineering-intuition)
- [4. Probability: from random variables to model distributions](#4-probability-from-random-variables-to-model-distributions)
- [5. Softmax, cross-entropy, and KL divergence](#5-softmax-cross-entropy-and-kl-divergence)
- [6. Calculus and backpropagation](#6-calculus-and-backpropagation)
- [7. Optimization, gradient stability, and normalization](#7-optimization-gradient-stability-and-normalization)
- [8. Numerical computing and mixed precision](#8-numerical-computing-and-mixed-precision)
- [9. Worked AI Infra examples](#9-worked-ai-infra-examples)
- [Summary](#summary)
- [Self-check list](#self-check-list)
- [Exercises](#exercises)
- [References](#references)

---

## 1. Learning goals, notation, and three questions

After this chapter you should be able to:

1. Tell scalars, vectors, matrices, and higher-order tensors apart, and confidently derive shapes after each operation;
2. Explain the intuition behind dot products, norms, matrix multiplication, rank, eigendecomposition, and SVD;
3. Split a matrix multiply into blocks and see why GEMM tiling is mathematically equivalent;
4. Compute expectation, variance, and covariance, and understand common distributions and conditional probability;
5. Get a stable Softmax from logits, and understand temperature, cross-entropy, entropy, and KL divergence;
6. Read a computation graph with the chain rule, and hand-derive gradients for a linear layer and Softmax cross-entropy;
7. Explain how vanishing/exploding gradients, residual connections, and normalization affect training;
8. Tell apart the precision and dynamic range of FP32, TF32, FP16, BF16, and FP8;
9. Spot overflow, underflow, catastrophic cancellation, non-associativity, and other numerical risks;
10. Estimate an operator's parameter count, activation size, FLOPs, memory traffic, and arithmetic intensity.

### 1.1 Notation in this chapter

| Symbol | Meaning |
|------|------|
| $x$ | Scalar, or a variable when the context is clear |
| $\mathbf{x}$ | Vector |
| $\mathbf{X}$ | Matrix or higher-order tensor |
| $X_{ij}$ | Element at row $i$, column $j$ of a matrix |
| $\mathbb{R}^{m \times n}$ | Set of real matrices with $m$ rows and $n$ columns |
| $\mathbf{X}^\top$ | Transpose |
| $\mathbf{X}^{-1}$ | Inverse (when it exists) |
| $\lVert \mathbf{x} \rVert_2$ | L2 norm of a vector |
| $\mathbb{E}[X]$ | Expectation |
| $\operatorname{Var}(X)$ | Variance |
| $\nabla_x f$ | Gradient of $f$ w.r.t. $x$ |
| $\partial f / \partial x$ | Partial derivative |
| $\odot$ | Element-wise multiplication |

Deep learning code usually uses batch-first notation:

- $B$: batch size;
- $S$: sequence length;
- $H$: hidden size;
- $V$: vocabulary size;
- $N_h$: number of attention heads;
- $D_h = H / N_h$: dimension per head.

### 1.2 Three questions to ask when you read a formula

Take a language-model output projection:

$$
\mathbf{Y} = \mathbf{X}\mathbf{W}, \qquad
\mathbf{X} \in \mathbb{R}^{(BS) \times H},\quad
\mathbf{W} \in \mathbb{R}^{H \times V}
$$

First: **do the shapes match?** The inner dimension $H$ is the same, so the output is $(BS) \times V$.

Second: **what does it cost?** Each output element does $H$ multiply-adds, so the total is about:

$$
2BSHV \quad \text{FLOPs}
$$

Here one multiply and one add each count as one floating-point operation.

Third: **what can go wrong in the implementation?** The weights are large and the output logits are large. You have to think about dtype, memory layout, tiling, parallel strategy, and Softmax numerical stability.

Those three questions show up across the whole AI Infra stack.

---

## 2. Linear algebra: tensors, transforms, and decompositions

### 2.1 Scalars, vectors, matrices, and tensors

- A **scalar** is a single number, like a learning rate $\eta = 10^{-4}$;
- A **vector** is a 1-D ordered array, like a token's hidden state $\mathbf{x} \in \mathbb{R}^{H}$;
- A **matrix** is a 2-D array, like a linear-layer weight $\mathbf{W} \in \mathbb{R}^{H \times 4H}$;
- A **tensor** is a multi-dimensional array, like a batch of sequence hidden states $\mathbf{X} \in \mathbb{R}^{B \times S \times H}$.

In engineering, a "tensor" also carries information beyond its mathematical shape:

```text
data address + shape + stride + dtype + device + layout
```

Two tensors with the same shape need not share the same memory layout. Equal values need not share the same dtype or error behavior.

### 2.2 Shape, indexing, and stride

Take a row-major matrix with shape $(2, 3)$:

$$
\mathbf{X} =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

Contiguous storage underneath is `[1, 2, 3, 4, 5, 6]`, and the element-wise stride is $(3, 1)$. The linear offset of element $X_{ij}$ is:

$$
\text{offset}(i,j) = 3i + j
$$

A transpose view $\mathbf{X}^\top$ has shape $(3,2)$ and stride $(1,3)$, usually without copying data. Scanning along the last dimension after the transpose is no longer contiguous.

Whether `reshape` copies depends on whether the original layout can be expressed with the new shape/stride. When you see `view`, `reshape`, `transpose`, or `contiguous`, think about both logical dimensions and physical layout.

### 2.3 Element-wise ops and broadcasting

Element-wise addition needs corresponding elements to pair up:

$$
\mathbf{C} = \mathbf{A} + \mathbf{B}, \qquad C_{ij} = A_{ij} + B_{ij}
$$

Broadcasting lets some dimensions of length 1, or missing dimensions, expand logically. For example:

$$
\mathbf{X} \in \mathbb{R}^{B \times S \times H}, \qquad
\mathbf{b} \in \mathbb{R}^{H}
$$

$\mathbf{X}+\mathbf{b}$ reuses the same bias across the batch and sequence dimensions, and the result shape is still $(B,S,H)$.

Aligning from the trailing dimensions forward, each pair of dimensions must satisfy:

1. Both sizes are equal; or
2. At least one size is 1; or
3. One side does not have that dimension.

Broadcasting is usually a logical view; it does not first materialize a full copy. Later kernel memory access and reduction patterns still feel the effect. The more dangerous case is a wrong shape that "happens to broadcast": the code does not error, but the semantics are wrong.

### 2.4 Dot product: similarity and weighted sums

The dot product of two $n$-dimensional vectors:

$$
\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{n}x_i y_i
$$

Two common intuitions:

**Weighted sum**: $\mathbf{x}$ is the weights, $\mathbf{y}$ is the values.

**Directional similarity**:

$$
\mathbf{x}^\top \mathbf{y}
= \lVert \mathbf{x}\rVert_2
  \lVert \mathbf{y}\rVert_2 \cos\theta
$$

So cosine similarity is:

$$
\cos\theta =
\frac{\mathbf{x}^\top\mathbf{y}}
{\lVert\mathbf{x}\rVert_2\lVert\mathbf{y}\rVert_2}
$$

Attention uses the Query–Key dot product to measure how well they match. Embedding retrieval often uses a normalized dot product as an approximation to cosine similarity.

If a vector's norm is near 0, the denominator is unstable. Implementations usually add a small $\epsilon$.

### 2.5 Norms: length, error, and constraints

Common vector norms:

$$
\lVert \mathbf{x} \rVert_1 = \sum_i |x_i|
$$

$$
\lVert \mathbf{x} \rVert_2 = \sqrt{\sum_i x_i^2}
$$

$$
\lVert \mathbf{x} \rVert_\infty = \max_i |x_i|
$$

The Frobenius norm of a matrix treats every element as one long vector:

$$
\lVert \mathbf{A} \rVert_F
= \sqrt{\sum_i\sum_j A_{ij}^2}
$$

In practice norms are used to:

- Measure the scale of parameters, activations, or gradients;
- Clip gradients;
- Compare error between a reference and an optimized implementation;
- Regularize;
- Normalize vectors.

When comparing floating-point results, absolute and relative error each matter:

$$
e_{abs} = |\hat{x}-x|,\qquad
e_{rel} = \frac{|\hat{x}-x|}{|x|}
$$

When the reference $x$ is near 0, relative error blows up, so test libraries usually combine absolute tolerance `atol` and relative tolerance `rtol`:

$$
|\hat{x}-x| \le \text{atol} + \text{rtol}\cdot |x|
$$

### 2.6 Matrix multiplication is a set of dot products

If:

$$
\mathbf{A} \in \mathbb{R}^{M \times K}, \qquad
\mathbf{B} \in \mathbb{R}^{K \times N}
$$

then:

$$
\mathbf{C} = \mathbf{A}\mathbf{B}
\in \mathbb{R}^{M \times N}
$$

Each element is:

$$
C_{ij} = \sum_{k=1}^{K} A_{ik}B_{kj}
$$

Treat shapes as an interface check:

```text
(M, K) @ (K, N) -> (M, N)
```

The middle $K$ is reduced; the outer $M,N$ stay. Standard GEMM is often written:

$$
\mathbf{C} \leftarrow
\alpha\mathbf{A}\mathbf{B} + \beta\mathbf{C}
$$

Matrix multiplication generally **does not commute**: $\mathbf{A}\mathbf{B}$ and $\mathbf{B}\mathbf{A}$ may differ in shape or in values. It does associate:

$$
(\mathbf{A}\mathbf{B})\mathbf{C}
= \mathbf{A}(\mathbf{B}\mathbf{C})
$$

Mathematical equality does not mean bit-identical floating-point results, or equal compute cost. Different association orders can change intermediate tensor sizes and FLOPs a lot.

### 2.7 Batched matrix multiplication

In multi-head Attention you often see:

$$
\mathbf{Q} \in \mathbb{R}^{B \times N_h \times S_q \times D_h}
$$

$$
\mathbf{K} \in \mathbb{R}^{B \times N_h \times S_k \times D_h}
$$

After transposing the last two dimensions:

$$
\mathbf{K}^\top
\in \mathbb{R}^{B \times N_h \times D_h \times S_k}
$$

Batched matrix multiply:

$$
\mathbf{Q}\mathbf{K}^\top
\in \mathbb{R}^{B \times N_h \times S_q \times S_k}
$$

The leading dimensions $(B,N_h)$ are batch dims; each `(batch, head)` runs its own matrix multiply. Higher-order tensor multiplication is usually just matrix multiplication on the last two dims, with the leading dims handling batching and broadcast.

### 2.8 Linear and affine transforms

A strict linear transform satisfies:

$$
f(a\mathbf{x}+b\mathbf{y})
= af(\mathbf{x})+bf(\mathbf{y})
$$

Matrix multiplication $f(\mathbf{x})=\mathbf{W}\mathbf{x}$ is linear. The "Linear" layer in a neural net usually also has a bias:

$$
\mathbf{y}=\mathbf{W}\mathbf{x}+\mathbf{b}
$$

Mathematically that is an affine transform. Several linear/affine layers with no activation can be fused into one, so nonlinear activations are what let deep nets express complex functions.

### 2.9 Transpose, identity, and inverse

Transpose swaps rows and columns:

$$
(\mathbf{A}^\top)_{ij} = A_{ji}
$$

A useful identity:

$$
(\mathbf{A}\mathbf{B})^\top
= \mathbf{B}^\top\mathbf{A}^\top
$$

The identity $\mathbf{I}$ satisfies:

$$
\mathbf{I}\mathbf{A}=\mathbf{A}\mathbf{I}=\mathbf{A}
$$

If a square matrix $\mathbf{A}$ has an inverse:

$$
\mathbf{A}^{-1}\mathbf{A}=
\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}
$$

To solve $\mathbf{A}\mathbf{x}=\mathbf{b}$ you can write $\mathbf{x}=\mathbf{A}^{-1}\mathbf{b}$, but numerical code usually does not form the inverse explicitly. It uses LU, QR, Cholesky, or similar factorizations and solves directly. Explicit inversion usually costs more and accumulates more error.

### 2.10 Linear dependence, span, and rank

If one vector in a set can be written as a linear combination of the others, they are linearly dependent. The rank of a matrix is the number of independent directions:

$$
\operatorname{rank}(\mathbf{A})
\le \min(M,N)
$$

Low rank means the matrix has redundant information and can be approximated by two smaller matrices:

$$
\mathbf{A} \approx
\mathbf{U}\mathbf{V},\qquad
\mathbf{U}\in\mathbb{R}^{M\times r},
\mathbf{V}\in\mathbb{R}^{r\times N},
\quad r \ll \min(M,N)
$$

That drops parameter count from $MN$ to $r(M+N)$, and it is the math behind low-rank adapters and model compression.

### 2.11 Eigenvalues and eigenvectors

For a square matrix $\mathbf{A}$, if a nonzero vector $\mathbf{v}$ and scalar $\lambda$ satisfy:

$$
\mathbf{A}\mathbf{v}=\lambda\mathbf{v}
$$

then $\mathbf{v}$ is an eigenvector and $\lambda$ is the matching eigenvalue. Intuitively, along that direction the transform only scales; it does not change direction.

Eigenvalues help you reason about:

- Whether a linear dynamical system amplifies or damps;
- Hessian curvature along different directions;
- Why gradients explode or vanish when many Jacobians multiply;
- Principal directions of change in a covariance matrix (PCA).

Not every matrix has a nice eigendecomposition over the reals. In practice pick an algorithm that fits the matrix structure; do not treat `eig` as a universal tool.

### 2.12 Singular value decomposition (SVD)

Any real matrix $\mathbf{A}\in\mathbb{R}^{M\times N}$ can be written:

$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top
$$

where:

- Columns of $\mathbf{U}$ are left singular vectors;
- Columns of $\mathbf{V}$ are right singular vectors;
- The diagonal of $\mathbf{\Sigma}$ holds singular values $\sigma_1\ge\sigma_2\ge\cdots\ge0$.

Keeping the top $r$ singular values:

$$
\mathbf{A}_r
= \mathbf{U}_{:,1:r}
  \mathbf{\Sigma}_{1:r,1:r}
  \mathbf{V}_{:,1:r}^\top
$$

$\mathbf{A}_r$ is an optimal rank-$r$ approximation (in the usual 2-norm and Frobenius sense). Small singular values correspond to weak directions you can drop for compression.

SVD itself has compute and memory cost. Compressing large model weights usually uses truncation, randomized algorithms, or training low-rank factors directly, not a full SVD on every huge matrix.

### 2.13 LoRA's low-rank update

For a pretrained weight $\mathbf{W}_0\in\mathbb{R}^{d_{out}\times d_{in}}$, LoRA freezes $\mathbf{W}_0$ and only trains a low-rank update:

$$
\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W}
$$

$$
\Delta\mathbf{W}
= \frac{\alpha}{r}\mathbf{B}\mathbf{A}
$$

where:

$$
\mathbf{A}\in\mathbb{R}^{r\times d_{in}},\qquad
\mathbf{B}\in\mathbb{R}^{d_{out}\times r}
$$

New parameters drop from a full update's $d_{out}d_{in}$ to:

$$
r(d_{in}+d_{out})
$$

For example $d_{in}=d_{out}=4096$, $r=16$:

$$
\frac{16(4096+4096)}{4096^2}
= 0.0078125 \approx 0.78\%
$$

Low rank only cuts trainable parameters and the matching optimizer state. It does not automatically remove base-weight, activation, or forward-compute cost. At deploy time you can merge the weights, or keep adapters and switch them dynamically; the two choices trade off memory, latency, and flexibility differently.

---

## 3. Block matrices and GEMM engineering intuition

### 3.1 Why block computation works

Cut the matrices along $M$, $N$, and the reduction dimension $K$. For a $2\times2$ block example:

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

then:

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

Other blocks work the same way. Each output block accumulates several local products along $K$. So a GPU kernel can:

1. Take a small block of $\mathbf{A}$ and a small block of $\mathbf{B}$;
2. Move them into on-chip shared memory / registers;
3. Reuse that data to compute many output elements;
4. Iterate along $K$ and accumulate;
5. Write the output block back to HBM.

That is the mathematical basis of tiling. It changes compute order and data movement, not the target formula.

### 3.2 Compute and memory traffic for naive GEMM

For $(M,K)@(K,N)$:

- Number of output elements: $MN$;
- Each element does $K$ multiply-adds;
- FLOPs are about $2MKN$.

A naive implementation that loads $A_{ik}$ and $B_{kj}$ from global memory on every multiply-add re-reads the same elements many times. High-performance GEMM comes from reuse:

- One element of $A$ is reused across many columns of the same output row;
- One element of $B$ is reused across many rows of the same output column;
- Accumulators in registers reuse data $K$ times.

### 3.3 A simplified arithmetic-intensity estimate

Arithmetic intensity is:

$$
\text{Arithmetic Intensity}
= \frac{\text{FLOPs}}{\text{bytes transferred from target memory}}
$$

Ideally an FP16 GEMM reads each input matrix from HBM once and writes the output once, so a rough byte count is:

$$
2(MK+KN+MN)
$$

Then:

$$
AI \approx
\frac{2MKN}{2(MK+KN+MN)}
$$

In reality you also have cache, alignment, read-modify-write, temporaries, and fused ops. This estimate is not a precise performance model, but it helps you decide whether an operator is more likely compute-bound or bandwidth-bound.

### 3.4 Bigger tiles are not always better

Larger tiles improve data reuse, but they also consume more:

- Shared memory;
- Registers;
- Threads and synchronization per block;
- Boundary-handling cost.

Too much resource use lowers how many blocks/warps an SM can keep resident, which hurts latency hiding. Choosing a tile size trades off data reuse, occupancy, instruction efficiency, and shape fit.

### 3.5 Tail blocks and padding

When $M,N,K$ are not integer multiples of the tile size, the last block crosses the valid boundary. Common approaches:

- Add boundary checks / masks;
- Pad inputs to an aligned size;
- Keep a fast path for common aligned shapes and a generic path for the rest.

Padding adds compute and storage; branches add control overhead. Which is faster depends on shape and hardware.

### 3.6 Why floating-point tiled results can differ

Real addition associates, but floating-point addition does not strictly:

$$
(a+b)+c \ne a+(b+c)
$$

Different tiles, thread reduction trees, and Tensor Core paths change accumulation order, so optimized and reference results may not match bit for bit. Correctness checks should use error tolerances that fit the dtype and problem size, and watch whether error grows systematically with reduction length.

---

## 4. Probability: from random variables to model distributions

### 4.1 Random variables and probability distributions

A random variable maps random outcomes to numbers. Discrete random variables use a probability mass function:

$$
p_X(x)=P(X=x),\qquad \sum_x p_X(x)=1
$$

Continuous random variables use a probability density function:

$$
p_X(x)\ge0,\qquad
\int_{-\infty}^{\infty}p_X(x)\,dx=1
$$

The probability of a continuous variable at a single point is usually 0; interval probabilities come from integrating the density. Distributions you see in models:

| Distribution | Values | Typical use |
|------|------|----------|
| Bernoulli | $0/1$ | Binary classification, Dropout masks |
| Categorical | $1,\ldots,V$ | Next-token distribution |
| Uniform | Interval / finite set | Initialization, random sampling |
| Gaussian | Reals | Initialization, noise, approximate analysis |

### 4.2 Joint, marginal, and conditional probability

Joint probability describes several events happening together:

$$
P(X=x,Y=y)
$$

Summing or integrating out the other variable gives the marginal:

$$
P(X=x)=\sum_y P(X=x,Y=y)
$$

Conditional probability:

$$
P(X=x\mid Y=y)
= \frac{P(X=x,Y=y)}{P(Y=y)}
$$

Autoregressive language models factor sequence probability with the chain rule:

$$
P(x_1,x_2,\ldots,x_T)
= \prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1})
$$

That is why next-token prediction at train time and token-by-token generation at inference are two processes of the same probability model.

### 4.3 Independence and conditional independence

If:

$$
P(X,Y)=P(X)P(Y)
$$

then $X,Y$ are independent. Independence is stronger than "uncorrelated"; zero covariance does not imply independence.

Conditional independence given $Z$ is written:

$$
P(X,Y\mid Z)=P(X\mid Z)P(Y\mid Z)
$$

Distributed sampling, data parallelism, and statistical estimation often assume i.i.d. Real data can break that with duplicate samples, shard bias, or sequence correlation.

### 4.4 Bayes' rule

$$
P(X\mid Y)
= \frac{P(Y\mid X)P(X)}{P(Y)}
$$

Here $P(X)$ is the prior, $P(Y\mid X)$ the likelihood, and $P(X\mid Y)$ the posterior. Ordinary LLM training does not hand-compute Bayes' rule, but it is still the foundation of probabilistic inference, parameter estimation, and uncertainty modeling.

### 4.5 Expectation: a probability-weighted average

Discrete:

$$
\mathbb{E}[X]=\sum_x xP(X=x)
$$

Continuous:

$$
\mathbb{E}[X]=\int x p(x)\,dx
$$

Expectation is linear; the variables need not be independent:

$$
\mathbb{E}[aX+bY]
= a\mathbb{E}[X]+b\mathbb{E}[Y]
$$

Training objectives are usually expected loss under the data distribution:

$$
\min_\theta
\mathbb{E}_{(x,y)\sim p_{data}}
[\mathcal{L}(f_\theta(x),y)]
$$

You cannot walk the true distribution, so you approximate with a mini-batch sample mean.

### 4.6 Variance and standard deviation

Variance measures spread around the mean:

$$
\operatorname{Var}(X)
= \mathbb{E}[(X-\mu)^2]
= \mathbb{E}[X^2]-\mu^2
$$

Standard deviation has the same units as the original variable:

$$
\sigma = \sqrt{\operatorname{Var}(X)}
$$

If $X,Y$ are independent:

$$
\operatorname{Var}(X+Y)
= \operatorname{Var}(X)+\operatorname{Var}(Y)
$$

Gradient noise in training, initialization scale, normalization, and quantization error all connect to variance.

### 4.7 Covariance and correlation

Covariance measures how two variables move together:

$$
\operatorname{Cov}(X,Y)
= \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
$$

Normalize to get the correlation coefficient:

$$
\rho_{X,Y}
= \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
$$

For a vector random variable, the covariance matrix is:

$$
\mathbf{\Sigma}
= \mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})
(\mathbf{x}-\boldsymbol{\mu})^\top]
$$

It is symmetric positive semidefinite. PCA eigendecomposes the covariance matrix to find orthogonal directions of largest variance.

### 4.8 Sample mean, variance, and mini-batches

Given samples $x_1,\ldots,x_n$, the sample mean is:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

A common unbiased sample variance is:

$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

Whether a deep-learning operator's "variance" divides by $n$ or $n-1$ depends on the definition. LayerNorm usually uses the population form (divide by the element count); stats libraries may default differently. When you reimplement an operator, check the API semantics; do not trust the name alone.

### 4.9 Maximum likelihood and negative log-likelihood

Given data $D=\{(x_i,y_i)\}$, maximum likelihood picks parameters that make the observed data most probable:

$$
\theta^*
= \arg\max_\theta
\prod_i p_\theta(y_i\mid x_i)
$$

Products underflow easily and are awkward to differentiate. Taking the log turns the product into a sum:

$$
\theta^*
= \arg\max_\theta
\sum_i \log p_\theta(y_i\mid x_i)
$$

Equivalently, minimize the negative log-likelihood (NLL):

$$
\mathcal{L}_{NLL}
= -\sum_i \log p_\theta(y_i\mid x_i)
$$

The logarithm is an important bridge between numerical computing and probabilistic modeling.

---

## 5. Softmax, cross-entropy, and KL divergence

### 5.1 Logits are not probabilities

The last layer of a model outputs $V$ arbitrary reals:

$$
\mathbf{z}=(z_1,z_2,\ldots,z_V)
$$

These values are called logits. They can be negative and need not sum to 1. Softmax turns them into a categorical distribution:

$$
p_i = \operatorname{softmax}(\mathbf{z})_i
= \frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
$$

Clearly $p_i>0$ and $\sum_i p_i=1$.

Softmax is invariant to a uniform shift:

$$
\operatorname{softmax}(\mathbf{z}+c)
= \operatorname{softmax}(\mathbf{z})
$$

because numerator and denominator both multiply by $e^c$. That property is exactly what the stable implementation relies on.

### 5.2 Numerically stable Softmax

Directly computing $e^{1000}$ overflows finite-precision floats. Let:

$$
m=\max_j z_j
$$

The stable form is:

$$
p_i=
\frac{e^{z_i-m}}
{\sum_j e^{z_j-m}}
$$

The largest exponent input is 0, so the largest exponential is 1; everything else is at most 1. Pseudocode:

```python
def stable_softmax(x):
    maximum = max(x)
    exps = [exp(value - maximum) for value in x]
    denominator = sum(exps)
    return [value / denominator for value in exps]
```

Subtracting the max prevents overflow in the forward direction, but tiny terms can still underflow to 0. That usually means they are negligible relative to the max; if you still need a logarithm afterward, use a stable `log_softmax` directly instead of going through 0 and then `log(0)`.

### 5.3 LogSumExp

Define:

$$
\operatorname{LSE}(\mathbf{z})
= \log\sum_j e^{z_j}
$$

Stable form:

$$
\operatorname{LSE}(\mathbf{z})
= m + \log\sum_j e^{z_j-m}
$$

Then:

$$
\log\operatorname{softmax}(\mathbf{z})_i
= z_i - \operatorname{LSE}(\mathbf{z})
$$

Cross-entropy implementations usually fuse `log_softmax + NLLLoss`, which cuts intermediate tensors and memory traffic and avoids the unstable path of Softmax then log.

### 5.4 Temperature

Softmax with temperature $T>0$:

$$
p_i(T)=
\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

- $T<1$: amplifies logit differences, sharper distribution;
- $T>1$: shrinks differences, flatter distribution;
- $T\to0^+$: approaches picking only the max logit;
- $T\to\infty$: approaches the uniform distribution.

Temperature changes the sampling distribution; it does not change model weights. Implementations should scale logits first, then apply stable Softmax, and be careful with very small temperatures.

### 5.5 Cross-entropy

Cross-entropy between a true distribution $\mathbf{q}$ and a model distribution $\mathbf{p}$:

$$
H(\mathbf{q},\mathbf{p})
= -\sum_i q_i\log p_i
$$

If the true label is one-hot with correct class $y$:

$$
q_y=1,\quad q_{i\ne y}=0
$$

then:

$$
\mathcal{L}
= -\log p_y
$$

Higher probability on the correct class means lower loss. If $p_y=1$, loss is 0; if $p_y$ is near 0, loss is large.

### 5.6 Information entropy

Entropy of a distribution itself:

$$
H(\mathbf{p})
= -\sum_i p_i\log p_i
$$

Entropy measures uncertainty:

- When all mass sits on one class, entropy is minimal;
- With $V$ classes uniform, entropy is maximal: $\log V$.

The log base sets the unit: natural log gives nats, base 2 gives bits. Machine-learning losses usually use the natural log.

### 5.7 KL divergence

KL divergence from distribution $\mathbf{q}$ to $\mathbf{p}$:

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})
= \sum_i q_i\log\frac{q_i}{p_i}
$$

It satisfies:

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})\ge0
$$

and is 0 when the two distributions match. It is usually not symmetric:

$$
D_{KL}(\mathbf{q}\lVert\mathbf{p})
\ne D_{KL}(\mathbf{p}\lVert\mathbf{q})
$$

so KL divergence is not a distance in the strict sense.

Cross-entropy decomposes as:

$$
H(\mathbf{q},\mathbf{p})
= H(\mathbf{q})
+ D_{KL}(\mathbf{q}\lVert\mathbf{p})
$$

During training the true distribution $\mathbf{q}$ is fixed, so $H(\mathbf{q})$ does not depend on model parameters, and minimizing cross-entropy is equivalent to minimizing the matching KL.

KL shows up in knowledge distillation, distribution matching, RLHF/PPO constraints, and speculative-decoding analysis. When you implement it, be clear whether the API takes probabilities, log probabilities, or logits, and which direction of KL it uses.

### 5.8 Perplexity

If the average token negative log-likelihood is $\bar{L}$, perplexity is:

$$
\operatorname{PPL}=e^{\bar{L}}
$$

Intuitively it is the model's "effective number of candidates" at each step. Different tokenizers, preprocessing, context lengths, and whether special tokens are ignored all change PPL, so you cannot compare numbers across eval setups blindly.

### 5.9 Top-k and Top-p sampling

After the model outputs a distribution, the decoding strategy decides how to pick a token:

- Greedy: pick the highest-probability token;
- Top-k: sample only among the $k$ highest-probability tokens;
- Top-p (nucleus): take the smallest candidate set whose cumulative probability is at least $p$, renormalize, then sample.

The Top-p set size changes with how sharp the distribution is. Implementations usually filter logits first, set excluded entries to negative infinity, then run stable Softmax and sample.

In distributed inference, if the vocabulary is sharded by tensor parallelism, global top-k/top-p needs cross-device aggregation of local candidates, or an equivalent distributed algorithm.

---

## 6. Calculus and backpropagation

### 6.1 Derivatives: local rate of change

Derivative of a univariate function:

$$
f'(x)=
\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

Near $x$, a small input change affects the output as:

$$
f(x+\Delta x)
\approx f(x)+f'(x)\Delta x
$$

Common derivatives:

$$
\frac{d}{dx}x^n=nx^{n-1}
$$

$$
\frac{d}{dx}e^x=e^x
$$

$$
\frac{d}{dx}\log x=\frac{1}{x}
$$

### 6.2 Partial derivatives and gradients

For a multivariate scalar $f(x_1,\ldots,x_n)$, the rate of change w.r.t. one variable is a partial. Stacking all partials gives the gradient:

$$
\nabla_{\mathbf{x}}f=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

The gradient points in the direction of steepest ascent, so gradient descent updates along the negative gradient:

$$
\mathbf{x}_{t+1}
= \mathbf{x}_t-\eta\nabla f(\mathbf{x}_t)
$$

$\eta$ is the learning rate.

### 6.3 Jacobian and Hessian

For a vector-valued function $\mathbf{y}=f(\mathbf{x})$, the Jacobian is:

$$
\mathbf{J}_{ij}
= \frac{\partial y_i}{\partial x_j}
$$

It describes each input component's local effect on each output component.

The Hessian of a scalar function is the matrix of second partials:

$$
\mathbf{H}_{ij}
= \frac{\partial^2 f}
{\partial x_i\partial x_j}
$$

The Hessian describes local curvature. Deep-learning parameter counts are too large to build a full Hessian explicitly, but Hessian-vector products, curvature approximations, and eigenvalue analysis still show up in optimization and stability work.

### 6.4 Chain rule

If:

$$
y=f(u),\qquad u=g(x)
$$

then:

$$
\frac{dy}{dx}
= \frac{dy}{du}\frac{du}{dx}
$$

A multi-layer network is a composition. For:

$$
\mathbf{h}_1=f_1(\mathbf{x}),\quad
\mathbf{h}_2=f_2(\mathbf{h}_1),\quad
L=f_3(\mathbf{h}_2)
$$

gradients flow from the loss backward:

$$
\frac{\partial L}{\partial \mathbf{x}}
= \frac{\partial L}{\partial \mathbf{h}_2}
  \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}
  \frac{\partial \mathbf{h}_1}{\partial \mathbf{x}}
$$

Conceptually that is a product of Jacobians. Real automatic differentiation does not materialize huge Jacobians; it efficiently computes vector-Jacobian products (reverse mode).

### 6.5 Computation graphs and reverse-mode autodiff

Let:

$$
a=xy,\qquad b=a+x,\qquad L=b^2
$$

The forward pass saves needed intermediates. Reverse starts from:

$$
\frac{\partial L}{\partial L}=1
$$

then:

$$
\frac{\partial L}{\partial b}=2b
$$

Because $b=a+x$:

$$
\frac{\partial L}{\partial a}=2b
$$

$x$ affects $L$ along two paths, so the gradients add:

$$
\frac{\partial L}{\partial x}
= \frac{\partial L}{\partial b}\frac{\partial b}{\partial x}
+ \frac{\partial L}{\partial a}\frac{\partial a}{\partial x}
= 2b + 2by
$$

$$
\frac{\partial L}{\partial y}=2bx
$$

That is why a tensor used in more than one place on the graph accumulates gradient.

### 6.6 Backprop through a linear layer

Using the batch-as-rows convention:

$$
\mathbf{Y}=\mathbf{X}\mathbf{W}+\mathbf{b}
$$

where:

$$
\mathbf{X}\in\mathbb{R}^{B\times d_{in}},\quad
\mathbf{W}\in\mathbb{R}^{d_{in}\times d_{out}},\quad
\mathbf{Y}\in\mathbb{R}^{B\times d_{out}}
$$

Given upstream gradient:

$$
\mathbf{G}=\frac{\partial L}{\partial \mathbf{Y}}
\in\mathbb{R}^{B\times d_{out}}
$$

then:

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

Shapes alone are a fast check on the formulas:

```text
dX: (B, dout) @ (dout, din) -> (B, din)
dW: (din, B) @ (B, dout) -> (din, dout)
db: reduce over batch dim -> (dout,)
```

A linear layer's backward pass usually has two GEMMs and one reduction, which is one reason training compute is much higher than a single forward pass.

### 6.7 Softmax Jacobian

Softmax:

$$
p_i=\frac{e^{z_i}}{\sum_k e^{z_k}}
$$

Its partials:

$$
\frac{\partial p_i}{\partial z_j}
= p_i(\delta_{ij}-p_j)
$$

In matrix form:

$$
\mathbf{J}_{softmax}
= \operatorname{diag}(\mathbf{p})
- \mathbf{p}\mathbf{p}^\top
$$

Every Softmax output depends on every input, so the Jacobian is not diagonal. The backward pass can still run in $O(V)$ time without materializing a $V\times V$ matrix.

### 6.8 The clean gradient of Softmax + cross-entropy

One-hot label $\mathbf{y}$, loss:

$$
L=-\sum_i y_i\log p_i
$$

Differentiating jointly with Softmax yields:

$$
\frac{\partial L}{\partial z_i}=p_i-y_i
$$

This is one of the most important simplifications in deep learning. For the correct class the gradient is $p_y-1$; for other classes it is $p_i$. The farther the prediction is off, the harder the push to correct the logits.

A fused cross-entropy kernel can share the max and reduction results, skip intermediate probability tensors, and emit the needed gradients directly.

### 6.9 Gradient checking

Automatic-differentiation implementations can spot-check with finite differences at small scale:

$$
\frac{\partial f}{\partial x_i}
\approx
\frac{f(\mathbf{x}+\epsilon\mathbf{e}_i)
-f(\mathbf{x}-\epsilon\mathbf{e}_i)}{2\epsilon}
$$

Central differences are usually more accurate than one-sided ones, but $\epsilon$ must not be too large or too small: too large brings truncation error; too small is drowned by floating-point rounding. Gradient checks fit small sizes, double precision, and deterministic functions. They are a poor fit for checking full training jobs with randomness or huge tensors.

---

## 7. Optimization, gradient stability, and normalization

### 7.1 Full-batch gradients, SGD, and mini-batches

Empirical risk:

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}L_i(\theta)
$$

A full-batch gradient uses every sample each step and is expensive. A mini-batch gradient:

$$
\hat{g}
= \frac{1}{B}\sum_{i\in\mathcal{B}}
\nabla_\theta L_i(\theta)
$$

is a stochastic estimate of the full gradient. Basic SGD update:

$$
\theta_{t+1}=\theta_t-\eta\hat{g}_t
$$

Larger batches often cut gradient-estimate variance and raise hardware utilization, but they need more activation memory, and optimizer hyperparameters may need matching adjustments.

### 7.2 Momentum

A common momentum form:

$$
\mathbf{v}_t
= \beta\mathbf{v}_{t-1}+(1-\beta)\mathbf{g}_t
$$

$$
\theta_{t+1}
= \theta_t-\eta\mathbf{v}_t
$$

Momentum exponentially averages past gradients, damps short-term noise, and accelerates along directions that stay consistent.

### 7.3 Adam's scale-adaptive intuition

Adam keeps exponential moving averages of first and second moments:

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

After bias correction:

$$
\theta_{t+1}
= \theta_t
-\eta\frac{\hat{\mathbf{m}}_t}
{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}
$$

Squaring and square roots are element-wise. Adam stores extra state per parameter; if that state is FP32, the memory cost is often a large share of large-model training, which is why techniques like ZeRO sharding exist.

### 7.4 Vanishing and exploding gradients

Gradients of a deep composition include many Jacobian products:

$$
\frac{\partial L}{\partial \mathbf{h}_0}
= \frac{\partial L}{\partial \mathbf{h}_L}
\prod_{l=1}^{L}
\frac{\partial \mathbf{h}_l}
{\partial \mathbf{h}_{l-1}}
$$

If those transforms stay below scale 1 along the relevant directions for long, gradients decay exponentially; if they stay above 1, gradients grow exponentially.

Engineering symptoms include:

- Gradient norms near 0, early layers barely updating;
- Sudden huge gradient norms, loss becoming `inf`/`nan`;
- Mixed precision triggering underflow or overflow earlier.

Mitigations include careful initialization, residual connections, normalization, gradient clipping, a sensible learning rate, and stable dtypes.

### 7.5 Why residual connections help gradient flow

A residual block:

$$
\mathbf{y}=\mathbf{x}+F(\mathbf{x})
$$

has Jacobian:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
= \mathbf{I}+\frac{\partial F}{\partial \mathbf{x}}
$$

The identity path gives a gradient route that does not fully depend on $F$. It does not guarantee you never explode or vanish, but it improves the optimization landscape for deep nets a lot.

### 7.6 Initialization and variance propagation

Consider:

$$
y=\sum_{i=1}^{n}w_i x_i
$$

If $w_i,x_i$ are independent with mean 0:

$$
\operatorname{Var}(y)
= n\operatorname{Var}(w)\operatorname{Var}(x)
$$

To keep activation variance roughly stable across layers, set $\operatorname{Var}(w)$ on the order of $1/n$. Xavier and Kaiming initialization pick scales from fan-in, fan-out, and the activation.

Smaller is not always more stable: too small damps signals and gradients; too large saturates or explodes.

### 7.7 LayerNorm

For one token's hidden vector $\mathbf{x}\in\mathbb{R}^{H}$:

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

LayerNorm reduces over the hidden dimension and does not depend on other samples in the batch, so it fits variable-length sequences and autoregressive models.

From a kernel view it at least involves a mean reduction, a variance reduction, and an element-wise transform. A naive implementation reads and writes HBM multiple times; an optimized one fuses the reductions with the affine transform and handles parallel reduction over long vectors.

### 7.8 RMSNorm

RMSNorm skips subtracting the mean and only scales by the root mean square:

$$
\operatorname{RMS}(\mathbf{x})
= \sqrt{\frac{1}{H}\sum_i x_i^2+\epsilon}
$$

$$
\operatorname{RMSNorm}(x_i)
= \gamma_i\frac{x_i}{\operatorname{RMS}(\mathbf{x})}
$$

It drops the mean-related work, but "fewer formulas" does not automatically translate into matching end-to-end speedups. Real gains depend on fusion, memory traffic, and the model's overall bottleneck.

### 7.9 Gradient clipping

Global L2-norm clipping:

$$
\mathbf{g}\leftarrow
\mathbf{g}\cdot
\min\left(1,\frac{c}{\lVert\mathbf{g}\rVert_2+\epsilon}\right)
$$

When the gradient norm exceeds threshold $c$, the whole vector is scaled down while keeping direction. In distributed training, if gradients are sharded, computing the global norm needs an AllReduce of local squared sums across ranks:

$$
\lVert\mathbf{g}\rVert_2
= \sqrt{\sum_r\sum_{i\in r}g_i^2}
$$

So even a simple-looking math op can introduce collective communication.

---

## 8. Numerical computing and mixed precision

### 8.1 Floating-point numbers are a finite set

A typical binary float looks like:

$$
(-1)^s \times \text{significand}
\times 2^{\text{exponent}}
$$

Bits are split among sign, exponent, and significand (mantissa). Exponent bits set dynamic range; significand bits set how finely nearby representable numbers are spaced.

| Format | Total bits | Exponent bits | Significand field bits | Main traits |
|------|--------|--------|------------|----------|
| FP32 | 32 | 8 | 23 | Balanced range and precision |
| TF32 | 19 (multiply-input semantics) | 8 | 10 | Keeps FP32 range, lowers multiply precision, often accumulates in FP32 |
| FP16 | 16 | 5 | 10 | Decent precision, smaller dynamic range |
| BF16 | 16 | 8 | 7 | Near-FP32 dynamic range, less precision than FP16 |
| FP8 E4M3 | 8 | 4 | 3 | Precision-first, smaller range; exact encoding depends on the spec |
| FP8 E5M2 | 8 | 5 | 2 | Range-first, even less precision; exact encoding depends on the spec |

The table does not count the implicit leading bit in the significand field. TF32 is usually an NVIDIA Tensor Core compute mode, not a normal in-memory storage dtype.

### 8.2 Precision and dynamic range are different things

FP16 has a wider significand field than BF16, so at the same magnitude it can usually represent finer differences. But FP16 has fewer exponent bits: its largest finite value is only 65504, and the range of nonzero values is much narrower.

BF16 trades some precision for the same exponent-bit count as FP32, so training is usually less prone to overflow/underflow. It still has clear rounding error, especially when adding a tiny increment onto a large number.

### 8.3 Rounding and machine precision

Many decimal fractions cannot be represented exactly in binary floating point:

```python
>>> 0.1 + 0.2 == 0.3
False
```

Spacing between floats grows with absolute magnitude. If $|\delta|$ is far smaller than the representable spacing near the current $x$:

$$
\operatorname{fl}(x+\delta)=x
$$

That is why optimizers often keep FP32 master weights: updates that are too small on low-precision weights can be rounded away entirely.

### 8.4 Non-associativity and reduction order

Because every step rounds:

$$
\operatorname{fl}(\operatorname{fl}(a+b)+c)
\ne
\operatorname{fl}(a+\operatorname{fl}(b+c))
$$

Parallel reductions use tree-shaped addition; different thread counts, block partitions, and atomics can yield slightly different results. Nondeterminism is not automatically a bug, but the error must stay acceptable and match the task's reproducibility needs.

Ways to improve reduction accuracy include:

- Accumulate in higher precision;
- Pairwise / tree summation to avoid adding extreme scales for long stretches;
- Compensated summation such as Kahan (extra instruction cost);
- Local reduce first, then merge partial sums.

### 8.5 Overflow, underflow, and non-finite values

- **overflow**: absolute value exceeds the largest finite value, often becoming $\pm\infty$;
- **underflow**: result too close to 0, entering subnormals or rounding to 0;
- `NaN`: e.g. $0/0$, $\infty-\infty$, or other illegal ops;
- `inf` in later ops can quickly spread into `NaN`.

When loss becomes `NaN`, find the **first** non-finite tensor; do not only stare at the final loss. Common sources include oversized logits, division by zero, square root of a negative, bad masks, gradient overflow, and wrong custom-kernel boundaries.

### 8.6 Catastrophic cancellation

Subtracting two nearly equal large numbers cancels leading significant digits, and the relative error of what remains can be huge. For example, computing variance with:

$$
\operatorname{Var}(X)
= \mathbb{E}[X^2]-\mathbb{E}[X]^2
$$

both terms can be large and close, so subtraction loses a lot of precision.

More stable options include two-pass variance or Welford's online algorithm. The latter updates mean and squared deviation step by step:

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

Population variance is $M_{2,n}/n$. Welford can also merge statistics from different chunks, which fits parallel reduction.

### 8.7 Condition number: how much input error is amplified

A problem's condition number describes how a tiny input perturbation affects the output. For an invertible matrix, in a consistent norm:

$$
\kappa(\mathbf{A})
= \lVert\mathbf{A}\rVert
  \lVert\mathbf{A}^{-1}\rVert
$$

A large condition number means an ill-conditioned problem: even with a correct algorithm and tiny rounding in the input, the output can change a lot.

Separate:

- **whether the problem itself is ill-conditioned**;
- **whether the chosen algorithm is numerically stable**.

More bits ease rounding error; they do not remove sensitivity that comes from ill-conditioning.

### 8.8 Basic patterns for mixed-precision training

Mixed precision is not "cast everything to FP16". Common patterns:

1. Large matrix multiplies use FP16/BF16 inputs and Tensor Core acceleration;
2. Multiply-adds finish in FP32 or another specified higher accumulation precision;
3. Sensitive ops such as reductions, normalization, and Softmax stay in or promote internally to FP32;
4. Optimizer state and/or master weights stay FP32;
5. Output dtype is chosen per hardware and operator.

Exact strategies differ by framework and hardware. Read the real operator docs and the profiler; do not guess the internal path from the input dtype alone.

### 8.9 Why loss scaling helps

Small FP16 gradients in the backward pass can underflow to 0. With scale factor $S$:

$$
L' = S L
$$

then:

$$
\nabla L'=S\nabla L
$$

Gradients are first amplified into the FP16 representable range; after the backward pass they are divided by $S$:

$$
\nabla L=\frac{\nabla L'}{S}
$$

Dynamic loss scaling:

1. Runs backward with the current scale;
2. Checks whether gradients contain `inf`/`NaN`;
3. On overflow, skips the update and shrinks the scale;
4. After several stable steps, tries growing the scale.

Loss scaling mainly fights underflow of small gradients; too large a scale causes overflow instead. BF16's larger dynamic range usually does not need loss scaling, but the model can still produce non-finite values from its own instability.

### 8.10 Stable algorithms often fuse better too

Numerical stability and performance optimization are not always in conflict:

- Stable Softmax does a max reduction first, then exp/sum; Online Softmax can merge block states;
- Fusing `log_softmax + NLL` avoids writing intermediate probabilities;
- Welford keeps statistics that fit LayerNorm's block reductions;
- GEMM with low-precision inputs and high-precision accumulation balances throughput and error.

The key is choosing a mathematical form with composable state so kernels can process tiles in limited on-chip storage.

---

## 9. Worked AI Infra examples

### 9.1 Example 1: Transformer linear-layer shape and cost

Input:

$$
\mathbf{X}\in\mathbb{R}^{B\times S\times H}
$$

Weight:

$$
\mathbf{W}\in\mathbb{R}^{H\times 4H}
$$

Flatten the first two dims:

$$
\mathbf{X}'\in\mathbb{R}^{(BS)\times H}
$$

Output:

$$
\mathbf{Y}'=\mathbf{X}'\mathbf{W}
\in\mathbb{R}^{(BS)\times 4H}
$$

then reshape to $(B,S,4H)$. FLOPs:

$$
2\cdot(BS)\cdot H\cdot4H
=8BSH^2
$$

If $B=8,S=2048,H=4096$:

$$
8\times8\times2048\times4096^2
\approx 2.20\times10^{12}\ \text{FLOPs}
$$

That is only one forward linear projection, which shows why large models lean so hard on Tensor Core GEMM.

### 9.2 Example 2: Full multi-head Attention dimensions

From:

$$
\mathbf{X}\in\mathbb{R}^{B\times S\times H}
$$

after producing Q/K/V, reshape and transpose:

$$
\mathbf{Q},\mathbf{K},\mathbf{V}
\in\mathbb{R}^{B\times N_h\times S\times D_h}
$$

with $H=N_hD_h$.

Score matrix:

$$
\mathbf{S}
=\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D_h}}
\in\mathbb{R}^{B\times N_h\times S\times S}
$$

Apply causal mask and Softmax:

$$
\mathbf{P}=\operatorname{softmax}(\mathbf{S}+\mathbf{M})
$$

where future positions are masked as $-\infty$ so their Softmax probability is 0.

Output:

$$
\mathbf{O}=\mathbf{P}\mathbf{V}
\in\mathbb{R}^{B\times N_h\times S\times D_h}
$$

Merge heads and return to $(B,S,H)$.

If you explicitly materialize FP16 $\mathbf{S}$, that alone needs:

$$
2BN_hS^2\ \text{bytes}
$$

When $B=1,N_h=32,S=32768$:

$$
2\times1\times32\times32768^2
=64\ \text{GiB}
$$

and that still ignores probabilities, backward intermediates, and other layers. FlashAttention's key idea is not writing the full $S\times S$ intermediate back to HBM, but finishing stable Softmax and $PV$ accumulation in on-chip tiles.

### 9.3 Example 3: Why divide by $\sqrt{D_h}$

Assume $q_i,k_i$ independent, mean 0, variance 1:

$$
\mathbf{q}^\top\mathbf{k}
=\sum_{i=1}^{D_h}q_i k_i
$$

Each $q_i k_i$ has expectation about 0 and variance about 1, so the sum has variance about $D_h$ and standard deviation about $\sqrt{D_h}$.

After dividing by $\sqrt{D_h}$, score variance returns to about 1:

$$
\operatorname{Var}\left(
\frac{\mathbf{q}^\top\mathbf{k}}{\sqrt{D_h}}
\right)\approx1
$$

That keeps logits from over-expanding as dimension grows and Softmax from entering extreme saturation.

### 9.4 Example 4: Merging Online Softmax blocks

A row of logits is split into blocks. The processed part has max and exp-sum $(m,l)$:

$$
m=\max_i x_i,\qquad
l=\sum_i e^{x_i-m}
$$

The new block's state is $(m_b,l_b)$. Merge the max:

$$
m_{new}=\max(m,m_b)
$$

Rescale to a common baseline and merge the exp-sums:

$$
l_{new}
= e^{m-m_{new}}l
+e^{m_b-m_{new}}l_b
$$

That lets Softmax run tiled, streaming, and numerically stable. If you also keep a weighted Value accumulator, you can avoid storing the full attention matrix—one of FlashAttention's mathematical cores.

### 9.5 Example 5: LoRA parameters and compute

Full linear layer:

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

Shapes:

```text
x:     (..., din)
A^T:   (din, r)       -> (..., r)
B^T:   (r, dout)      -> (..., dout)
```

An online adapter adds two small GEMMs. If you merge $\mathbf{B}\mathbf{A}$ into the base weight ahead of time, inference is still one large GEMM, but you lose cheap dynamic adapter switching.

### 9.6 Example 6: Why distributed means must be weighted

Two ranks have local sample counts $n_1,n_2$ and local means $\mu_1,\mu_2$. The global mean is not simply $(\mu_1+\mu_2)/2$ unless $n_1=n_2$. The correct value:

$$
\mu=
\frac{n_1\mu_1+n_2\mu_2}{n_1+n_2}
$$

More generally each rank first aggregates local `sum` and `count`, AllReduces, then divides. Distributed loss, metrics, and throughput stats all need a clear denominator, especially with a trailing incomplete batch or differing sequence masks.

### 9.7 Example 7: Memory estimate

For a tensor of shape $(B,S,H)$ with $b$ bytes per element, contiguous storage size is:

$$
M=B\times S\times H\times b
$$

For BF16 $(8,4096,8192)$:

$$
8\times4096\times8192\times2
=536{,}870{,}912\ \text{bytes}
=512\ \text{MiB}
$$

Peak training memory also includes parameters, gradients, optimizer state, saved activations, temporary workspace, communication buffers, fragmentation, and framework context. A single-tensor estimate is only the starting point.

### 9.8 A fixed checklist from formula to kernel

When you see a new operator, break it down in this order:

1. What are the shapes of inputs, outputs, and intermediates?
2. Which dimensions are kept, and which are reduced?
3. Can it be tiled? How do tile states merge?
4. What are the FLOPs and the theoretical minimum memory traffic?
5. Are there broadcasts, transposes, or non-contiguous strides?
6. Which ops are precision-sensitive and need high-precision accumulation?
7. Can you hit exp overflow, division by zero, cancellation, or long-reduction error?
8. Can intermediate tensors be fused away?
9. How are boundary shapes and tail blocks handled?
10. What reference implementation, dtype tolerances, and extreme inputs do you use to validate?

That checklist turns abstract math into an executable Infra workflow.

---

## Summary

This chapter built four linked threads:

- **Linear algebra** describes how tensors transform: shape decides legality, rank and decompositions decide compression options, and blocking decides how data is reused;
- **Probability** describes what model outputs mean: logits become a distribution through Softmax, cross-entropy matches negative log-likelihood, and KL measures distribution difference;
- **Calculus** describes how parameters learn: the chain rule backpropagates along the computation graph, and a linear layer's backward pass is still GEMMs plus a reduction;
- **Numerical computing** decides whether a formula is trustworthy on hardware: finite precision rounds, overflows, and underflows, so stable formulas and mixed-precision strategy matter as much as the algebra.

For an AI Infra engineer, "knowing the math" ends up meaning: you can turn a formula into shapes, reductions, tiles, data movement, precision choices, and validation criteria.

## Self-check list

- [ ] Seeing `(B, S, H) @ (H, V)` you can write the output shape immediately
- [ ] You can tell element-wise multiply, dot product, matrix multiply, and batched matmul apart
- [ ] From shape/stride you can tell whether a transpose view is contiguous
- [ ] You can explain the intuition of norms, rank, eigenvalues, and singular values
- [ ] You can compute parameter savings for low-rank factorization and LoRA
- [ ] You can explain GEMM tiling correctness from block matrix multiplication
- [ ] You can estimate GEMM FLOPs, minimum memory traffic, and arithmetic intensity
- [ ] You can compute expectation, variance, covariance, and conditional probability
- [ ] You can write stable Softmax and LogSumExp
- [ ] You can state how cross-entropy, entropy, and KL relate
- [ ] You can derive gradients on a small computation graph with the chain rule
- [ ] You can write the gradient shapes of a linear layer w.r.t. input, weight, and bias
- [ ] You know Softmax + cross-entropy's gradient w.r.t. logits is `p - y`
- [ ] You can explain why residual connections improve gradient pathways
- [ ] You can distinguish precision, dynamic range, overflow, and underflow
- [ ] You can explain why FP16 often needs Loss Scaling while BF16 usually does not
- [ ] You can explain why parallel reductions may not be bit-identical
- [ ] From Attention shapes you can derive $O(S^2)$ intermediates and their memory cost

## Exercises

1. Derive the output shape of `(B, Nh, Sq, Dh) @ (B, Nh, Dh, Sk)`, and mark the batch dims and reduction dims.
2. For an FP16 GEMM with $M=4096,K=4096,N=16384$, estimate FLOPs and the ideal minimum input/output byte count.
3. Write a program that verifies $\operatorname{softmax}(\mathbf{z}+c)=\operatorname{softmax}(\mathbf{z})$, and compare naive vs stable Softmax on large logits.
4. Starting from the Softmax Jacobian and cross-entropy, hand-derive $\partial L/\partial z_i=p_i-y_i$.
5. Derive the backward shapes for `Y = GELU(XW + b)`; you do not need to expand GELU's exact derivative.
6. Compare FP16, BF16, and FP32 error when summing the sequence `[1, 1e-3, 1e-3, ...]`, and try changing the summation order.
7. Implement Welford variance and compare it with `E[x²] - E[x]²` on data with a large mean and small variance.
8. For a $4096\times4096$ weight, compute parameter ratios and extra forward FLOPs for rank-8, 16, and 64 LoRA.
9. Compute the size of an explicit FP16 Attention score matrix when $B=2,N_h=32,S=8192,D_h=128$.
10. Two ranks process 7 and 5 valid tokens with local mean losses 2.0 and 3.0; compute the correct global mean and explain why a simple average is wrong.
11. Finite-difference-check a small handwritten linear-layer backward; try different $\epsilon$ and look at the error curve.
12. Pick a PyTorch operator and write a one-page analysis covering shape, reduction, FLOPs, memory traffic, numerical risk, and validation method.

## References

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
