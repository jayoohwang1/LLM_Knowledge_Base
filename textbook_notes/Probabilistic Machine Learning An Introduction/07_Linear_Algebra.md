# Chapter 7: Linear Algebra

Quick orientation from someone who actually uses this stuff: linear algebra in 2026 is the substrate of frontier ML. Most of this chapter is foundational notation. The parts that *really* matter for modern LLM research are: **matrix multiplication efficiency / einsum**, **norms (esp. spectral & Frobenius) for stability and scaling**, **SVD / low-rank structure** (LoRA, KV-cache compression, model compression, mech interp), **eigenstructure** (optimization landscape, NTK, signal propagation), **condition number** (training stability, preconditioning), **matrix calculus** (autograd, JVP/VJP). Stuff like Schur complements, MVN conditionals, LU/QR — useful but you basically never touch them by hand in LLM work; they live inside libraries.

---

## 7.1 Introduction and Notation

### 7.1.1 Notation
- **Vector** $\mathbf{x} \in \mathbb{R}^n$ as column by default. $\mathbf{1}, \mathbf{0}$ for all-ones / all-zeros. **Unit vector** $\mathbf{e}_i$ (one-hot).
- **Matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$. $A_{ij}$ entry, $\mathbf{A}_{i,:}$ row, $\mathbf{A}_{:,j}$ column.
- **Transpose** $(\mathbf{A}^\top)_{ij} = A_{ji}$. Properties: $(\mathbf{A}^\top)^\top = \mathbf{A}$, $(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top$, $(\mathbf{A}+\mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top$.
- **Symmetric**: $\mathbf{A} = \mathbf{A}^\top$, denoted $\mathbb{S}^n$.
- **Tensor**: generalization to $>2$ dims. The word "rank" of a tensor (order = number of dims) is *different* from matrix rank.
  - **Reshape / vec**: $\text{vec}(\mathbf{A})$ stacks columns. **Row-major** (Python, C++) vs **column-major** (Julia, MATLAB, Fortran). Mixing these up causes silent bugs when interfacing C/Python/CUDA code.

### 7.1.2 Vector spaces
- **Span**, **linear independence**, **basis**. $\mathbb{R}^n$ has $n$-element bases; standard basis $\{\mathbf{e}_i\}$.
- **Linear map** $f: \mathcal{V} \to \mathcal{W}$ fully determined by images of basis vectors $\Rightarrow$ representable as matrix multiplication $\mathbf{y} = \mathbf{Ax}$.
- **Range / column space**: $\text{range}(\mathbf{A}) = \{\mathbf{Ax} : \mathbf{x} \in \mathbb{R}^n\}$.
- **Nullspace**: $\text{null}(\mathbf{A}) = \{\mathbf{x} : \mathbf{Ax} = \mathbf{0}\}$.
- **Linear projection** onto range: $\text{Proj}(\mathbf{y}; \mathbf{A}) = \mathbf{A}(\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top \mathbf{y}$ (normal equations form).

**Why this matters for LLMs**: every attention head, every MLP layer is a linear map composed with a nonlinearity. The geometry of the *residual stream* and how features get embedded as directions ("superposition", linear probes, sparse autoencoders) is fundamentally about subspaces, bases, and projections in high-D vector spaces. **Mech interp** lives here.

### 7.1.3 Norms — **IMPORTANT**

- **Norm axioms**: non-negativity, definiteness, absolute homogeneity, triangle inequality.

**Vector norms**:
- **$p$-norm**: $\|\mathbf{x}\|_p = (\sum_i |x_i|^p)^{1/p}$.
- **2-norm** (Euclidean): $\|\mathbf{x}\|_2^2 = \mathbf{x}^\top \mathbf{x}$.
- **1-norm**: $\sum_i |x_i|$ — sparsity-inducing (Lasso, L1 reg).
- **Max-norm**: $\max_i |x_i|$.
- **0-"norm"**: count of non-zeros (pseudo-norm).

**Matrix norms** — these are the ones you should actually know cold:
- **Induced (operator) $p$-norm**: $\|\mathbf{A}\|_p = \max_{\|\mathbf{x}\|=1} \|\mathbf{Ax}\|_p$. For $p=2$: $\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A}) = \sqrt{\lambda_{\max}(\mathbf{A}^\top \mathbf{A})}$ — **spectral norm**.
- **Nuclear (trace) norm**: $\|\mathbf{A}\|_* = \sum_i \sigma_i$. Convex surrogate for rank — used in low-rank recovery / matrix completion.
- **Frobenius norm**: $\|\mathbf{A}\|_F = \sqrt{\sum_{ij} A_{ij}^2} = \sqrt{\text{tr}(\mathbf{A}^\top \mathbf{A})} = \|\text{vec}(\mathbf{A})\|_2$.
- **Schatten $p$-norm**: $(\sum_i \sigma_i^p)^{1/p}$ — generalizes nuclear (p=1), Frobenius (p=2), spectral (p=∞).
- **Hutchinson trace estimator**: $\|\mathbf{A}\|_F^2 = \mathbb{E}[\|\mathbf{Av}\|_2^2]$ for $\mathbf{v} \sim \mathcal{N}(0, \mathbf{I})$. Cheap stochastic norm estimate via matvecs only — huge in scalable ML (Hessian trace est., influence functions, randomized linalg).

**Why LLM-relevant**:
- **Spectral norm** controls Lipschitz constants of layers $\Rightarrow$ controls signal propagation, gradient explosion/vanishing. Spectral normalization (Miyato), **muP / Tensor Programs** (Greg Yang) scaling rules are basically arguments about operator norms across width. **Adam** and its variants effectively normalize by something proportional to per-coordinate norms.
- **Frobenius** is what weight decay actually penalizes (when applied to parameters as vectors).
- **Nuclear norm** is the convex relaxation behind matrix completion and motivates **low-rank fine-tuning (LoRA)**.
- **Hutchinson** powers Hessian-trace estimators used in second-order optimizers (Sophia, AdaHessian) and influence-function approximations for data attribution.

### 7.1.4 Properties of a matrix — **IMPORTANT**

**Trace** $\text{tr}(\mathbf{A}) = \sum_i A_{ii}$:
- Linear; $\text{tr}(\mathbf{A}^\top) = \text{tr}(\mathbf{A})$; $\text{tr}(c\mathbf{A}) = c\,\text{tr}(\mathbf{A})$.
- **Cyclic property**: $\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{BCA}) = \text{tr}(\mathbf{CAB})$ — the workhorse identity in ML derivations.
- $\text{tr}(\mathbf{A}) = \sum_i \lambda_i$.
- **Trace trick**: $\mathbf{x}^\top \mathbf{A x} = \text{tr}(\mathbf{x}\mathbf{x}^\top \mathbf{A})$ — used constantly in Gaussian log-likelihoods.
- **Hutchinson**: $\text{tr}(\mathbf{A}) = \mathbb{E}[\mathbf{v}^\top \mathbf{Av}]$, $\mathbf{v}$ s.t. $\mathbb{E}[\mathbf{vv}^\top] = \mathbf{I}$.

**Determinant**:
- Geometric: volume scaling under the linear map.
- $|\mathbf{AB}| = |\mathbf{A}||\mathbf{B}|$, $|\mathbf{A}^{-1}| = 1/|\mathbf{A}|$, $|\mathbf{A}| = \prod_i \lambda_i$, $|\mathbf{A}| = 0$ iff singular.
- **Log-det via Cholesky**: $\mathbf{A} = \mathbf{L}\mathbf{L}^\top \Rightarrow \log|\mathbf{A}| = 2 \sum_i \log L_{ii}$ — numerically stable. Used in Gaussian likelihoods, **normalizing flows** (change-of-variables Jacobian determinants).

**Rank**:
- column-rank = row-rank = rank; $\text{rank}(\mathbf{A}) \le \min(m,n)$.
- $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{A}^\top) = \text{rank}(\mathbf{A}^\top \mathbf{A}) = \text{rank}(\mathbf{AA}^\top)$.
- $\text{rank}(\mathbf{AB}) \le \min(\text{rank}\,\mathbf{A}, \text{rank}\,\mathbf{B})$.
- Square invertible iff full rank.

**Condition number** — **VERY IMPORTANT** for ML:
- $\kappa(\mathbf{A}) = \|\mathbf{A}\| \|\mathbf{A}^{-1}\|$; for 2-norm: $\kappa_2(\mathbf{A}) = \sigma_{\max}/\sigma_{\min} = \sqrt{\lambda_{\max}/\lambda_{\min}}$ (for symmetric PSD).
- Geometrically: loss surface ellipsoid axis ratio. Ill-conditioned $\Rightarrow$ narrow valley $\Rightarrow$ SGD bounces.
- **Why this dominates LLM training**: training instability, learning-rate sensitivity, the rationale behind **Adam** (per-parameter rescaling $\approx$ diagonal preconditioning), **Shampoo / SOAP** (block-Kronecker preconditioners), and **normalization** layers (LayerNorm/RMSNorm reshape the local conditioning of activations).

### 7.1.5 Special matrices
- **Diagonal**, **identity**, **block-diagonal**, **band-diagonal / tridiagonal**.
- **Triangular**: eigenvalues = diagonal; $\det = \prod A_{ii}$.
- **Positive (semi)definite**:
  - $\mathbf{A} \succ 0$: $\mathbf{x}^\top \mathbf{A x} > 0\ \forall \mathbf{x} \ne 0$. **PSD** when $\ge 0$.
  - Symmetric matrix is PD iff all eigenvalues $> 0$.
  - **Diagonally dominant** $\Rightarrow$ PD (sufficient).
  - **Gram matrix** $\mathbf{G} = \mathbf{A}^\top \mathbf{A}$ always PSD; PD if $\mathbf{A}$ is full-rank with $m \ge n$.
- **Orthogonal** (real) / **unitary** (complex): $\mathbf{U}^\top \mathbf{U} = \mathbf{I} = \mathbf{UU}^\top$.
  - Preserve $\ell_2$ norm and angles: $\|\mathbf{Ux}\|_2 = \|\mathbf{x}\|_2$, $\cos(\angle(\mathbf{Ux}, \mathbf{Uy})) = \cos(\angle(\mathbf{x}, \mathbf{y}))$.
  - Rotations ($\det = +1$) / reflections ($\det = -1$).
  - **LLM relevance**: **RoPE** (Rotary Position Embeddings) literally applies rotation matrices in 2D subspaces of $\mathbf{Q}$/$\mathbf{K}$. **Householder reflections** show up in attention parametrizations. Orthogonal init (Saxe et al.) for stable signal propagation in deep nets. **Givens rotations** in quantization-aware fine-tuning. Orthogonality is the secret weapon for keeping activations from blowing up at depth.

---

## 7.2 Matrix Multiplication — **CORE**

$\mathbf{C} = \mathbf{AB}$, $C_{ij} = \sum_k A_{ik} B_{kj}$. $O(mnp)$ time, but GPUs/TPUs do this massively in parallel — **this is literally why deep learning works at scale**.

- Associative, distributive, **not** commutative.

### 7.2.1–7.2.3 Different views of matmul
Four equivalent perspectives — having all four in your head is a senior-researcher mental superpower:
1. **Inner products**: $C_{ij} = \mathbf{a}_i^\top \mathbf{b}_j$ (rows of A × cols of B).
2. **Sum of outer products**: $\mathbf{AB} = \sum_k \mathbf{a}_{:,k} \mathbf{b}_{k,:}^\top$ — each rank-1 piece contributes a "feature direction".
3. **Cols of C as matvecs**: $\mathbf{c}_j = \mathbf{A}\mathbf{b}_j$.
4. **Rows of C as matvecs**: $\mathbf{c}_i^\top = \mathbf{a}_i^\top \mathbf{B}$.

The sum-of-outer-products view is the right intuition for **low-rank approximation**, **LoRA** ($\Delta W = BA$ with $r \ll d$), **attention** ($\mathbf{QK}^\top = \sum_h \mathbf{q}_h \mathbf{k}_h^\top$ at the head level).

### 7.2.4 Manipulating data matrices
Useful matrix-form operations on the $N \times D$ design matrix $\mathbf{X}$:
- **Row sums**: $\mathbf{1}_N^\top \mathbf{X}$; **column sums**: $\mathbf{X} \mathbf{1}_D$.
- **Mean**: $\bar{\mathbf{x}}^\top = \frac{1}{N} \mathbf{1}_N^\top \mathbf{X}$.
- **Standardize**: $(\mathbf{X} - \mathbf{1}_N \boldsymbol{\mu}^\top) \text{diag}(\boldsymbol{\sigma})^{-1}$.
- **Sum-of-squares matrix**: $\mathbf{S}_0 = \mathbf{X}^\top \mathbf{X}$.
- **Centering matrix**: $\mathbf{C}_N = \mathbf{I}_N - \frac{1}{N} \mathbf{J}_N$, idempotent. Centered data $\tilde{\mathbf{X}} = \mathbf{C}_N \mathbf{X}$.
- **Scatter matrix**: $\mathbf{S}_{\bar{x}} = \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} = \mathbf{X}^\top \mathbf{C}_N \mathbf{X}$.
- **Gram matrix**: $\mathbf{K} = \mathbf{XX}^\top$ ($N \times N$ inner products). **Double-centering trick** to center a Gram matrix without the raw features: $\tilde{\mathbf{K}} = \mathbf{C}_N \mathbf{K} \mathbf{C}_N$ — foundation of **kernel PCA**, classical MDS.
- **Pairwise distance matrix**: $D_{ij} = \|\mathbf{x}_i\|^2 - 2 \mathbf{x}_i^\top \mathbf{y}_j + \|\mathbf{y}_j\|^2$ vectorized as $\mathbf{D} = \hat{\mathbf{x}} \mathbf{1}^\top - 2\mathbf{XY}^\top + \mathbf{1}\hat{\mathbf{y}}^\top$. This pattern is what `torch.cdist` does — used everywhere (k-NN, contrastive losses, retrieval, RAG).

### 7.2.5 Kronecker product
- $\mathbf{A} \otimes \mathbf{B}$ is the $mp \times nq$ block matrix with blocks $A_{ij}\mathbf{B}$.
- Identities: $(\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}$; $(\mathbf{A} \otimes \mathbf{B}) \text{vec}(\mathbf{C}) = \text{vec}(\mathbf{BCA}^\top)$.
- **Modern use**: **Shampoo/SOAP** preconditioners approximate the full Hessian-ish stat as Kronecker factors $\mathbf{L} \otimes \mathbf{R}$, one per layer side, making 2nd-order updates tractable for huge weight matrices. **K-FAC** (natural gradient approximation) is built on this. These are *the* current candidates to beat AdamW at scale.

### 7.2.6 Einstein summation (einsum) — **PRACTICAL ESSENTIAL**

The notation that runs modern ML codebases. Drop the $\sum$ and any index appearing twice on RHS gets contracted.

```python
# Matrix multiply
C = einsum('ik,kj->ij', A, B)

# Batched attention (B batch, H heads, T tokens, D head_dim)
attn = einsum('bhtd,bhsd->bhts', Q, K)  # scores
out  = einsum('bhts,bhsd->bhtd', attn, V)

# Batch one-hot to embeddings: S [n,t,k], W [k,d] -> E [n,t,d]
E = einsum('ntk,kd->ntd', S, W)
```

- `opt_einsum` / NumPy/JAX/PyTorch `optimize=True` find a near-optimal contraction order — exponential search, often greedy. **For repeated calls with same shapes, compute the path once and reuse.**
- This is the language of tensor-network contraction in physics too; cross-pollinates into ML methods like tensor-train decompositions.

---

## 7.3 Matrix Inversion

### 7.3.1 Inverse of a square matrix
- $\mathbf{A}^{-1}$ exists iff $\det(\mathbf{A}) \ne 0$.
- $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$; $(\mathbf{A}^\top)^{-1} = (\mathbf{A}^{-1})^\top \triangleq \mathbf{A}^{-\top}$.
- 2×2 closed form; block-diagonal: invert each block independently.

**Honest advice**: in numerical practice you basically *never* explicitly form $\mathbf{A}^{-1}$. You solve $\mathbf{Ax}=\mathbf{b}$ via factorization (LU, Cholesky, QR) and backsub. Forming the inverse is slower and less numerically stable.

### 7.3.2–7.3.4 Schur complements, matrix inversion lemma, det lemma (\*starred sections)
- **Schur complement**: for block matrix $\mathbf{M} = \begin{pmatrix} \mathbf{E} & \mathbf{F} \\ \mathbf{G} & \mathbf{H} \end{pmatrix}$, $\mathbf{M}/\mathbf{H} \triangleq \mathbf{E} - \mathbf{FH}^{-1}\mathbf{G}$. Gives block inverse formulae.
- **Matrix Inversion Lemma (Sherman–Morrison–Woodbury)**:
  $$(\mathbf{\Sigma} + \mathbf{XX}^\top)^{-1} = \mathbf{\Sigma}^{-1} - \mathbf{\Sigma}^{-1}\mathbf{X}(\mathbf{I} + \mathbf{X}^\top \mathbf{\Sigma}^{-1} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{\Sigma}^{-1}$$
  Trades $O(N^3)$ for $O(D^3)$ when one side is much smaller — **the backbone of Gaussian Process inference and Bayesian linear regression**.
- **Sherman–Morrison rank-1 update**: $(\mathbf{A} + \mathbf{uv}^\top)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{uv}^\top \mathbf{A}^{-1}}{1 + \mathbf{v}^\top \mathbf{A}^{-1} \mathbf{u}}$.
- **Matrix determinant lemma**: $|\mathbf{A} + \mathbf{uv}^\top| = (1 + \mathbf{v}^\top \mathbf{A}^{-1} \mathbf{u})|\mathbf{A}|$.

### 7.3.5 Conditionals of an MVN
Schur complement derivation: for $p(\mathbf{x}_1, \mathbf{x}_2) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ block-partitioned, $p(\mathbf{x}_1 | \mathbf{x}_2) = \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$ with $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$, $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$.

**Practical relevance for LLMs**: low. But the same algebra reappears in **Kalman filtering / state-space models (Mamba, S4, S5)**, and in any Gaussian-posterior reasoning.

---

## 7.4 Eigenvalue Decomposition (EVD) — **IMPORTANT**

### 7.4.1 Basics
- $\mathbf{Au} = \lambda \mathbf{u}$ with $\mathbf{u} \ne 0$. Characteristic equation $\det(\lambda \mathbf{I} - \mathbf{A}) = 0$.
- Properties: $\text{tr}(\mathbf{A}) = \sum_i \lambda_i$, $\det(\mathbf{A}) = \prod_i \lambda_i$, rank = # nonzero $\lambda_i$, $\mathbf{A}^{-1}$ has eigenvalues $1/\lambda_i$.
- Triangular / diagonal: eigenvalues = diagonal entries.

### 7.4.2 Diagonalization
- $\mathbf{A} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^{-1}$ when eigenvectors are LI.

### 7.4.3 Symmetric matrices — **central case for ML**
- Real eigenvalues, **orthonormal** eigenvectors: $\mathbf{A} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top = \sum_i \lambda_i \mathbf{u}_i \mathbf{u}_i^\top$.
- Multiplying by symmetric A = rotate ($\mathbf{U}^\top$), scale ($\boldsymbol{\Lambda}$), rotate back ($\mathbf{U}$).
- $\mathbf{A}^{-1} = \mathbf{U} \boldsymbol{\Lambda}^{-1} \mathbf{U}^\top$.
- PD iff all $\lambda_i > 0$; PSD iff all $\ge 0$; indefinite if both signs.

### 7.4.4 Quadratic forms
- $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax}$. Level sets are hyper-ellipsoids; axes = eigenvectors; semi-axis lengths $\propto 1/\sqrt{\lambda_i}$.
- Big LLM relevance: **loss landscape geometry**. Hessian eigenstructure determines the directions in weight space that are sharp vs flat. **Sharpness-aware minimization (SAM)**, **flatness/generalization** debates, **Hessian eigenvalue spectra** in NNs (Sagun, Papyan), **edge of stability** (Cohen et al.) all live here. NTK theory: when training is in the lazy regime, dynamics are governed by the eigenstructure of the NTK Gram matrix.

### 7.4.5 Standardizing and whitening
- Empirical cov $\boldsymbol{\Sigma} = \mathbf{E} \mathbf{D} \mathbf{E}^\top$.
- **PCA whitening**: $\mathbf{W}_{pca} = \mathbf{D}^{-1/2} \mathbf{E}^\top$ — makes $\text{Cov}[\mathbf{Wx}] = \mathbf{I}$.
- **ZCA whitening**: $\mathbf{W}_{zca} = \mathbf{E} \mathbf{D}^{-1/2} \mathbf{E}^\top = \boldsymbol{\Sigma}^{-1/2}$ — minimum distortion (closest to original in $\ell_2$). Preserves visual structure for images.
- Modern LLM analogs: **BatchNorm / LayerNorm / RMSNorm** are cheap diagonal-only whitening. **Decorrelated normalization** (e.g., DBN, IterNorm) tries to whiten more fully. **Activation/gradient whitening** is part of why preconditioners like Shampoo work.

### 7.4.6 Power method
- $\mathbf{v}_t \propto \mathbf{Av}_{t-1}$ converges to top eigenvector at rate $|\lambda_2/\lambda_1|$.
- **Rayleigh quotient** $R(\mathbf{A}, \mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} / \mathbf{x}^\top \mathbf{x}$ recovers eigenvalue once you have the vector.
- Original PageRank used this on a 3B×3B stochastic matrix. In modern ML: **Lanczos / Arnoldi** (extension) for Hessian spectrum estimation, **spectral norm estimation** (a couple power iterations) used in **spectral normalization** for GANs and stability tricks.

### 7.4.7 Deflation
- $\mathbf{A}^{(2)} = \mathbf{A} - \lambda_1 \mathbf{u}_1 \mathbf{u}_1^\top$ projects out the top eigencomponent; iterate to get subsequent ones. Foundation of iterative PCA and sparse-PCA variants.

### 7.4.8 Eigenvectors as quadratic-form optimizers
- $\max \mathbf{x}^\top \mathbf{Ax}$ s.t. $\|\mathbf{x}\|=1$ has Lagrangian gradient $2\mathbf{Ax} = 2\lambda \mathbf{x}$ — i.e., eigenequation. Solutions are eigenvectors; max value = $\lambda_{\max}$. This is *why* PCA picks eigenvectors.

---

## 7.5 Singular Value Decomposition (SVD) — **THE MOST USEFUL DECOMPOSITION IN ML**

### 7.5.1 Basics
- Any real $\mathbf{A} \in \mathbb{R}^{m \times n}$: $\mathbf{A} = \mathbf{U S V}^\top = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$.
- $\mathbf{U} \in \mathbb{R}^{m\times m}$, $\mathbf{V} \in \mathbb{R}^{n \times n}$ orthogonal; $\mathbf{S}$ diagonal with $\sigma_i \ge 0$.
- **Economy / thin SVD**: only keep first $\min(m,n)$ singular vectors. Cost $O(\min(mn^2, m^2 n))$.

### 7.5.2 Connection to EVD
- $\mathbf{A}^\top \mathbf{A} = \mathbf{V S^\top S V}^\top$ — right singular vectors = eigvecs of $\mathbf{A}^\top \mathbf{A}$, eigenvalues are $\sigma_i^2$.
- $\mathbf{AA}^\top = \mathbf{U SS^\top U}^\top$ — analogously for left.
- For symmetric PSD: singular vectors = eigenvectors, singular values = eigenvalues.
- SVD always exists; EVD may not.

### 7.5.3 Pseudo-inverse
- **Moore–Penrose** $\mathbf{A}^\dagger$, unique. For tall full-column-rank: $\mathbf{A}^\dagger = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top$ (left inverse → OLS solution). For fat full-row-rank: $\mathbf{A}^\dagger = \mathbf{A}^\top (\mathbf{AA}^\top)^{-1}$ (right inverse → min-norm).
- Via SVD: $\mathbf{A}^\dagger = \mathbf{V} \mathbf{S}^{-1} \mathbf{U}^\top$ (with $1/\sigma_i$ for nonzero, 0 for zero singular values).

### 7.5.4 Range / nullspace from SVD
- $\text{range}(\mathbf{A}) = \text{span}(\{\mathbf{u}_i : \sigma_i > 0\})$, dim $= r$.
- $\text{null}(\mathbf{A}) = \text{span}(\{\mathbf{v}_j : \sigma_j = 0\})$, dim $= n - r$.
- **Rank–nullity**: $r + (n-r) = n$. Rank = number of nonzero singular values.

### 7.5.5 Truncated SVD — **HUGELY IMPORTANT FOR MODERN ML**
- $\hat{\mathbf{A}}_K = \mathbf{U}_K \mathbf{S}_K \mathbf{V}_K^\top$ is **the optimal rank-$K$ approximation** in Frobenius norm (Eckart–Young).
- Error: $\|\mathbf{A} - \hat{\mathbf{A}}_K\|_F^2 = \sum_{k > K} \sigma_k^2$.
- Param count $K(N + D + 1)$ vs $ND$ — huge compression when singular values decay fast.

**Massive list of modern uses**:
- **LoRA / DoRA / VeRA / PiSSA**: parametrize fine-tuning update as $\Delta W = BA$ with rank $r \ll d$. PiSSA explicitly initializes $A, B$ from top singular components of $W$.
- **Model compression / weight pruning**: low-rank factorization of large weight matrices, especially for on-device inference.
- **KV-cache compression** in long-context LLMs: SVD of K/V caches across tokens.
- **Mechanistic interp**: SVD of attention output-value matrices, transcoders, and weight matrices reveals interpretable subspaces.
- **Randomized SVD** (Halko–Martinsson–Tropp): standard tool for truncated SVD of huge matrices — used in fast PCA, embedding distillation, sketching.
- **Latent Semantic Analysis** (the OG truncated SVD of term-doc matrices) and modern variants in embedding-space analysis.
- **Spectral analysis of trained networks**: singular spectra reveal implicit regularization, model collapse, capacity.

---

## 7.6 Other Decompositions (\*starred)

### 7.6.1 LU
- $\mathbf{PA} = \mathbf{LU}$ (lower×upper, P=permutation for **partial pivoting**). Solve linear systems via two triangular solves.

### 7.6.2 QR
- $\mathbf{A} = \mathbf{QR}$, $\mathbf{Q}$ orthonormal columns, $\mathbf{R}$ upper triangular. Gram–Schmidt formalism.
- Used for least squares (numerically more stable than normal equations), eigenvalue algorithms (QR algorithm), and in numerical implementations of LayerNorm-free attention parametrizations.

### 7.6.3 Cholesky
- Symmetric PD: $\mathbf{A} = \mathbf{LL}^\top$. About 2× faster than LU, $O(V^3)$. `np.linalg.cholesky`.
- **Application: sampling MVN**: $\mathbf{y} = \mathbf{Lx} + \boldsymbol{\mu}$ for $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ yields $\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{LL}^\top)$. Standard in GPs, diffusion priors, latent-variable models.
- **Stable log-det**: $\log\det \boldsymbol{\Sigma} = 2 \sum_i \log L_{ii}$.

---

## 7.7 Solving Linear Systems (\*starred)

- **Square** ($m=n$, full rank): unique solution via LU/Cholesky + backsub. Never explicitly invert.
- **Underdetermined** ($m < n$): infinitely many solutions; pick **min-norm**: $\hat{\mathbf{x}} = \mathbf{A}^\top (\mathbf{AA}^\top)^{-1} \mathbf{b}$ (right pseudo-inverse).
- **Overdetermined** ($m > n$): no exact solution; **least squares**: $\hat{\mathbf{x}} = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top \mathbf{b}$ via **normal equations** $\mathbf{A}^\top \mathbf{A} \mathbf{x} = \mathbf{A}^\top \mathbf{b}$.
  - Hessian $\mathbf{A}^\top \mathbf{A}$ PD when full column rank → unique global min.
  - In practice: solve via QR, not normal equations directly (better conditioning).

**LLM-context honesty**: in deep learning we rarely solve explicit linear systems — we run SGD. But these solutions appear *inside* methods: **iterative refinement in conjugate-gradient solvers** for influence functions, **HVP (Hessian-vector products)** in second-order methods, **normal-equations-based update rules** in linear probes, **closed-form last-layer fitting** in transfer learning and Bayesian linear-head heads.

---

## 7.8 Matrix Calculus — **YOU NEED THIS DAILY**

### 7.8.1 Derivatives
- Scalar derivative $f'(x) = \lim_{h\to 0} (f(x+h)-f(x))/h$.
- **Finite differences**: forward / central / backward. Central is more accurate. Watch out for too-small $h$ (catastrophic cancellation). Used to check autograd correctness.

### 7.8.2 Gradient
- For $f: \mathbb{R}^n \to \mathbb{R}$: $\nabla f = (\partial f / \partial x_1, \dots)^\top$.
- A **vector field** (vector-valued function of $\mathbf{x}$).

### 7.8.3 Directional derivative
- $D_{\mathbf{v}} f(\mathbf{x}) = \nabla f \cdot \mathbf{v}$. Numerically: 2 function calls regardless of dim $n$. Compare to full-gradient FD: $n+1$ (or $2n$) calls. This asymmetry is the heart of **JVP** efficiency.

### 7.8.4 Total derivative
- $\frac{df}{dt} = \frac{\partial f}{\partial t} + \frac{\partial f}{\partial x}\frac{dx}{dt} + \cdots$. Lets you handle implicit dependencies — appears in **implicit differentiation** (DEQ models, meta-learning, hyperparameter optimization).

### 7.8.5 Jacobian — **CRITICAL**
- For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$: $\mathbf{J}_{\mathbf{f}}(\mathbf{x}) \in \mathbb{R}^{m \times n}$, rows = $\nabla f_i^\top$. **Numerator layout**.

**JVP and VJP**:
- **JVP** (Jacobian-vector product, "forward-mode"): $\mathbf{Jv}$, right-multiply by direction. Forward-mode autodiff. Efficient when $m \ge n$.
- **VJP** (vector-Jacobian product, "reverse-mode"): $\mathbf{u}^\top \mathbf{J}$, left-multiply by cotangent. Reverse-mode = **backprop**. Efficient when $m \le n$.

**This is the single most important algorithmic fact for LLM research**: $f: \mathbb{R}^{|\theta|} \to \mathbb{R}$ (loss), $|\theta| \gg 1$, so we want $m=1 \ll n$ → reverse mode. **One backward pass per loss**, no matter how big the model. JAX `vjp/jvp/grad/vmap`, PyTorch `torch.autograd.grad / torch.func.{vjp,jvp,jacrev,jacfwd}`.

- **Jacobian of composition**: $\mathbf{J}_{g \circ f}(\mathbf{x}) = \mathbf{J}_g(f(\mathbf{x})) \mathbf{J}_f(\mathbf{x})$. Chain rule = matrix multiplication. Every backward pass is a sequence of VJPs strung together this way.

### 7.8.6 Hessian
- $\mathbf{H}_f = \nabla^2 f$, symmetric $n \times n$. Hessian = Jacobian of gradient.
- For huge models, never materialize. Use **HVP via Pearlmutter trick** (two backward passes give $\mathbf{Hv}$ without forming $\mathbf{H}$) — backbone of:
  - **Conjugate gradient** for natural gradient (KFAC, Shampoo) and **influence functions** (data attribution at LLM scale).
  - **Newton/quasi-Newton in deep learning** (Sophia approximates diag(H) via HVPs with random vectors + Hutchinson).
  - **Hessian spectrum estimation** for sharpness analysis.

### 7.8.7 Common gradients — memorize these

Scalar:
- $\frac{d}{dx} cx^n = cnx^{n-1}$, $\frac{d}{dx} \log x = 1/x$, $\frac{d}{dx} e^x = e^x$.
- Sum/product rules; chain rule $\frac{d}{dx} f(u(x)) = f'(u) u'(x)$.

Vector-to-scalar:
$$\frac{\partial (\mathbf{a}^\top \mathbf{x})}{\partial \mathbf{x}} = \mathbf{a}, \quad \frac{\partial (\mathbf{b}^\top \mathbf{Ax})}{\partial \mathbf{x}} = \mathbf{A}^\top \mathbf{b}, \quad \frac{\partial (\mathbf{x}^\top \mathbf{Ax})}{\partial \mathbf{x}} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$$

Matrix-to-scalar (quadratic forms):
$$\frac{\partial}{\partial \mathbf{X}} (\mathbf{a}^\top \mathbf{Xb}) = \mathbf{ab}^\top, \quad \frac{\partial}{\partial \mathbf{X}} (\mathbf{a}^\top \mathbf{X}^\top \mathbf{b}) = \mathbf{ba}^\top$$

Trace identities:
$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{AXB}) = \mathbf{A}^\top \mathbf{B}^\top, \quad \frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{X}^\top \mathbf{A}) = \mathbf{A}$$
$$\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{X}^{-1} \mathbf{A}) = -\mathbf{X}^{-\top} \mathbf{A}^\top \mathbf{X}^{-\top}, \quad \frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{X}^\top \mathbf{AX}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{X}$$

Determinant:
$$\frac{\partial}{\partial \mathbf{X}} \det(\mathbf{AXB}) = \det(\mathbf{AXB}) \mathbf{X}^{-\top}, \quad \frac{\partial}{\partial \mathbf{X}} \log\det(\mathbf{X}) = \mathbf{X}^{-\top}$$

The last one (log-det gradient) is everywhere in flows, GP marginals, and Bayesian deep learning.

---

## 7.9 Exercises
- Verify rotation matrix is orthogonal; find eigenvector with eigenvalue 1 (the rotation axis).
- Compute eigenvalues/eigenvectors of a 2×2 diagonal matrix by hand.

---

## Senior-Researcher Take: What Actually Matters for LLM Research

**Tier 1 — internalize these so deeply they're reflexes**:
- **Matrix multiplication views** (especially sum-of-outer-products). Every modern arch is a matmul.
- **Einsum**. You'll write it constantly. Know the contraction-order pitfalls.
- **SVD and low-rank approximation**. LoRA-family methods, KV compression, weight surgery, mech interp — all SVD-shaped.
- **Norms**: spectral & Frobenius. They govern initialization (orthogonal/Kaiming), normalization (LayerNorm/RMSNorm), regularization (weight decay), stability arguments (muP), and Lipschitz bounds for safety/robustness.
- **Matrix calculus (JVP/VJP), chain rule**. This is autograd. Understanding when to use forward vs reverse mode, and how HVPs work, separates senior ICs.
- **Condition number / eigenstructure of the Hessian**. The story of why optimizers behave as they do — Adam, Shampoo, SOAP, Sophia, muP — is told in this language.

**Tier 2 — know they exist, reach for when needed**:
- **Orthogonal matrices, rotations** (RoPE, Householder, orthogonal init).
- **Kronecker products** (Shampoo, K-FAC, structured layers).
- **Cholesky / log-det** (probabilistic models, flows).
- **Pseudo-inverse and normal equations** (closed-form heads, ridge regression in last-layer Bayesian methods).
- **Whitening / decorrelation** (BN/LN heritage, preconditioning intuition).
- **Power iteration / Lanczos** (spectral norm estimation, Hessian spectrum probes).

**Tier 3 — useful background, but rarely hand-computed**:
- Schur complements, MVN conditionals, matrix inversion lemma — these power GP libraries and Kalman filters; you typically interact through APIs.
- LU/QR — inside `np.linalg`/`torch.linalg`; you almost never roll your own.
- Determinant and trace identities beyond a few you've seen above — look them up.

**One honest meta-point**: most published LLM-scale work doesn't *derive* anything fancy from this chapter. But knowing it cold lets you (a) read theory papers without flinching, (b) reason about training instabilities without resorting to vibes, (c) invent or modify methods (e.g., propose a new low-rank parametrization, build a custom preconditioner, design a normalization layer) rather than just running existing ones. The decompositions + calculus are the vocabulary of the field — fluency, not memorization, is the goal.
