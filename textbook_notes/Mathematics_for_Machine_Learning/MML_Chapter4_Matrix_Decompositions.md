# MML Chapter 4 — Matrix Decompositions

> **Big picture**: Decompositions = factoring a matrix into building blocks that expose its structure (rank, energy, geometry). For modern LLM/DL research, the heavy hitters are **SVD**, **eigendecomposition (eigvals/eigvecs of symmetric PSD matrices)**, and **low-rank approximation (Eckart–Young)**. Everything else (det, trace, Cholesky) is supporting cast — useful but secondary.
>
> **Opinionated TL;DR for frontier-lab work**:
> - **Internalize cold**: SVD (full + truncated), Eckart–Young low-rank approximation, spectral theorem, eigendecomposition of $A^\top A$ / $AA^\top$.
> - **Know fluently**: trace cyclic property (used everywhere — Jacobians, attention rank, kernel methods, nuclear norm), $\det = \prod \lambda_i$, $\text{tr} = \sum \lambda_i$.
> - **Skim/lookup**: Laplace expansion, Sarrus' rule, defective matrices, Jordan form (the book even punts on this).
> - **Niche but real**: Cholesky — Gaussian sampling, reparam trick in VAEs, Laplace approximation, GP inference, second-order optimizers (Shampoo, K-FAC use Cholesky-like factors).

---

## 4.1 Determinant and Trace

### 4.1.1 Determinant

- **Def**: $\det(A)$ for $A \in \mathbb{R}^{n\times n}$ — scalar summary of a square matrix. Only defined for square.
- **Why care**: $A$ invertible $\iff \det(A) \neq 0 \iff \text{rk}(A) = n$ (Theorem 4.3).
- **Geometric meaning**: signed volume of the parallelepiped spanned by columns of $A$. Sign = orientation.
  - $n=2$: signed area of parallelogram.
  - $n=3$: signed volume of parallelepiped.
- **Closed forms for small $n$**:
  - $n=1$: $\det(A) = a_{11}$.
  - $n=2$: $\det(A) = a_{11}a_{22} - a_{12}a_{21}$.
  - $n=3$: **Sarrus' rule** — six product terms (3 + diagonals minus 3 - diagonals).
- **Triangular matrices**: $\det(T) = \prod_i T_{ii}$ — just multiply the diagonal.
- **Laplace expansion** (Theorem 4.2): recursive cofactor expansion along any row/col.
  - $\det(A) = \sum_k (-1)^{k+j} a_{kj} \det(A_{k,j})$.
  - $A_{k,j}$ = submatrix with row $k$, col $j$ removed (the **minor**); $(-1)^{k+j}\det(A_{k,j})$ is the **cofactor**.
  - **Don't actually use this in code**. Use Gaussian elimination ($O(n^3)$) or LU/Cholesky.

**Key properties** (memorize the multiplicative + invariance ones):
- $\det(AB) = \det(A)\det(B)$
- $\det(A^\top) = \det(A)$
- $\det(A^{-1}) = 1/\det(A)$
- **Similar matrices share determinants** → $\det$ is basis-invariant (intrinsic to the linear map).
- Adding multiple of row/col to another: unchanged.
- Scaling a row/col by $\lambda$: scales $\det$ by $\lambda$. Hence $\det(\lambda A) = \lambda^n \det(A)$.
- Row/col swap: flips sign.
- $\det(A) = \prod_i \lambda_i$ (Theorem 4.16) — **link to eigenvalues**.

> **LLM-relevance check on $\det$**: Mostly theoretical glue. Direct uses are rare in deep learning at scale — Gaussian elimination supersedes it. BUT:
> - **Normalizing flows** literally compute $\log|\det(J)|$ of the Jacobian (RealNVP, Glow, neural ODEs use Hutchinson trace estimators / FFJORD). The whole field lives on tractable Jacobian determinants.
> - **Log-likelihood under change of variables**: $\log p(x) = \log p(z) + \log|\det(\partial z/\partial x)|$.
> - **Determinantal Point Processes (DPPs)** for diverse sampling/retrieval.
> - **MLE for Gaussians**: $\log\det(\Sigma)$ shows up everywhere — but you compute it via Cholesky as $2\sum\log L_{ii}$, never via the cofactor formula.

### 4.1.2 Trace

- **Def**: $\text{tr}(A) = \sum_i a_{ii}$ — sum of diagonal entries.
- **Properties**:
  - Linear: $\text{tr}(A+B) = \text{tr}(A) + \text{tr}(B)$, $\text{tr}(\alpha A) = \alpha\text{tr}(A)$.
  - $\text{tr}(I_n) = n$.
  - **Cyclic invariance** (THE big one): $\text{tr}(AKL) = \text{tr}(KLA) = \text{tr}(LAK)$.
  - Special case: $\text{tr}(xy^\top) = y^\top x$ (Frobenius inner product as trace).
  - Basis-invariant: $\text{tr}(S^{-1}AS) = \text{tr}(A)$.
  - $\text{tr}(A) = \sum_i \lambda_i$ (Theorem 4.17).

> **LLM-relevance — trace is everywhere**:
> - **Jacobian/Hessian estimators**: Hutchinson's trick $\text{tr}(M) = \mathbb{E}_v[v^\top M v]$ for $v$ with $\mathbb{E}[vv^\top]=I$. Used in influence functions, score matching, FFJORD, stochastic Lanczos, NTK estimation.
> - **Nuclear norm $\|A\|_* = \text{tr}(\sqrt{A^\top A}) = \sum_i \sigma_i$**: convex surrogate for rank; appears in LoRA theory, low-rank regularization, soft rank.
> - **Frobenius inner product** $\langle A, B \rangle_F = \text{tr}(A^\top B)$ — basis for matrix gradients, attention rank analysis.
> - **Attention analysis**: $\text{tr}(QK^\top / \sqrt{d})$-like quantities show up in measuring attention head similarity, effective rank of attention matrices (Dong et al. "Attention is not all you need" rank-collapse analysis uses spectral tools).
> - **VI / ELBO**: KL divergences between Gaussians involve $\text{tr}(\Sigma_1^{-1}\Sigma_2)$ — used in VAE training.
> - **Gradient computation**: derivative of $\text{tr}(f(W))$ — most matrix calculus identities derived via trace.

### 4.1.3 Characteristic Polynomial

- $p_A(\lambda) := \det(A - \lambda I) = c_0 + c_1\lambda + \dots + (-1)^n \lambda^n$.
- Roots = eigenvalues.
- $c_0 = \det(A)$, $c_{n-1} = (-1)^{n-1}\text{tr}(A)$.
- **In practice**: nobody computes eigenvalues by rooting the characteristic polynomial (numerically catastrophic). Use QR algorithm, Lanczos, or randomized SVD.

---

## 4.2 Eigenvalues and Eigenvectors

> **The single most important section for understanding how matrices "act" geometrically.**

- **Eigenvalue equation**: $Ax = \lambda x$, $x \neq 0$.
- **Equivalent statements** (all the same condition):
  - $\lambda$ is an eigenvalue of $A$.
  - $(A - \lambda I)x = 0$ has nontrivial solution.
  - $\text{rk}(A - \lambda I) < n$.
  - $\det(A - \lambda I) = 0$.
- **Eigenspace** $E_\lambda$: null space of $A - \lambda I$. All eigenvectors for $\lambda$ + zero.
- **Eigenspectrum**: set of all eigenvalues.
- **Multiplicities**:
  - **Algebraic multiplicity**: root multiplicity in $p_A(\lambda)$.
  - **Geometric multiplicity**: $\dim E_\lambda$ = # linearly independent eigenvectors for $\lambda$.
  - $1 \le \text{geo mult} \le \text{alg mult}$.
- **Defective**: $A$ has < $n$ linearly independent eigenvectors. Cannot be diagonalized over $\mathbb{R}$.
- **Useful facts**:
  - $A$ and $A^\top$ have same eigenvalues (different eigenvectors generally).
  - Similar matrices share eigenvalues → eigenvalues are intrinsic to the linear map.
  - Symmetric PSD matrices have non-negative real eigenvalues.
  - $n$ distinct eigenvalues $\Rightarrow$ eigenvectors linearly independent (Theorem 4.12).
  - $\det(A) = \prod_i \lambda_i$, $\text{tr}(A) = \sum_i \lambda_i$.
- **Geometric intuition** (2D grid figure): eigenvectors are the directions stretched by $A$; eigenvalues are the stretch factors.
  - $\lambda > 1$ stretch, $0 < \lambda < 1$ compress, $\lambda < 0$ flip, $\lambda = 0$ collapse (loss of rank), complex $\lambda$ → rotation (no real eigenvector).

### Symmetrization trick (Theorem 4.14)
Given any $A \in \mathbb{R}^{m\times n}$, the matrix $S = A^\top A$ is symmetric PSD:
- Symmetric: $(A^\top A)^\top = A^\top A$. ✓
- PSD: $x^\top A^\top A x = \|Ax\|^2 \ge 0$. ✓
- If $\text{rk}(A) = n$, then $S$ is positive definite.

> This is the gateway to SVD. **Internalize this.** Also underpins:
> - **Gram matrix** in kernel methods.
> - **Fisher Information matrix** $\mathbb{E}[\nabla\log p \,\nabla\log p^\top]$ — always PSD.
> - **Empirical covariance** $\frac{1}{N}X^\top X$.
> - **Neural Tangent Kernel** $\Theta = J J^\top$.
> - **GGN / Gauss-Newton approximation** of Hessian.

### Spectral Theorem (Theorem 4.15) — **THE most important theorem in this chapter**
If $A \in \mathbb{R}^{n\times n}$ is **symmetric**, then:
- All eigenvalues are real.
- There exists an **orthonormal basis** of eigenvectors.
- $\Rightarrow A = PDP^\top$ where $P$ orthogonal, $D$ diagonal of eigenvalues.

> **Why this matters for ML**:
> - Covariance, Gram, Hessian (at critical points), Fisher, NTK — all symmetric. The spectral theorem says they have orthonormal eigenbases and can be diagonalized cheanly.
> - **PCA** is literally the spectral decomposition of the empirical covariance.
> - **Hessian eigenspectrum analysis** in deep learning (Sagun et al., Ghorbani et al.) — bulk + outliers, sharp vs flat minima, edge of stability (Cohen et al.) — all relies on spectral theorem.
> - **Loss landscape**: most positive curvature in few directions (low effective rank of Hessian).

### Use cases / research connections
- **PageRank** (Example 4.9): top eigenvector of stochastic transition matrix. Same math underlies graph neural net analysis, knowledge graph embeddings, spectral graph theory.
- **Eigenspectra of biological neural nets** (Example 4.7): $S$-shaped spectrum is typical for connectivity matrices. Modern analogue: spectrum of weight matrices in trained transformers shows heavy tails (Martin & Mahoney "Heavy-Tailed Self-Regularization").
- **Mechanistic interpretability**: eigendecomposition of OV/QK circuits in transformers (Elhage et al. "Mathematical Framework for Transformer Circuits") — eigenvectors of attention head matrices reveal what the head does.
- **Spectral norm regularization**, Lipschitz-bounded networks (Miyato et al. spectral norm GANs).

---

## 4.3 Cholesky Decomposition

- **Theorem 4.18**: any symmetric PD $A$ factors uniquely as $A = LL^\top$, $L$ lower-triangular with positive diagonal.
- **Algorithm** (constructive): match entries of $LL^\top$ to $A$ row by row.
  - $l_{11} = \sqrt{a_{11}}$, $l_{22} = \sqrt{a_{22} - l_{21}^2}$, $l_{21} = a_{21}/l_{11}$, etc.
- $\det(A) = \det(L)^2 = \prod_i l_{ii}^2$ — cheap log-det.

> **LLM/ML uses (niche but real)**:
> - **Sampling multivariate Gaussians**: $x = \mu + L\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$. Used in VAEs (reparam trick), Bayesian deep learning, diffusion priors.
> - **Solving $Ax = b$ for SPD $A$**: forward + back substitution on $LL^\top x = b$. Used in Gaussian process regression (kernel matrix inversion), Laplace approximation, natural gradient (when Fisher is dense).
> - **Log-likelihood of Gaussian**: $-\frac{1}{2}\log\det(\Sigma) = -\sum\log L_{ii}$.
> - **K-FAC / Shampoo**: preconditioners use Cholesky-like factors of Kronecker block approximations.
> - **Less central than SVD/eigen for LLM research** — but if you do anything Bayesian/probabilistic, you'll use it.

```python
# Reparam trick (VAE)
L = torch.linalg.cholesky(Sigma)  # Sigma = L @ L.T
eps = torch.randn_like(mu)
z = mu + L @ eps  # z ~ N(mu, Sigma), backprop-friendly
```

---

## 4.4 Eigendecomposition and Diagonalization

- **Diagonalizable**: $A = PDP^{-1}$, $D$ diagonal of eigenvalues, $P$ columns are eigenvectors.
- Exists $\iff$ eigenvectors of $A$ form a basis of $\mathbb{R}^n$ ($A$ non-defective).
- Derivation: $AP = PD$ (column-wise: $Ap_i = \lambda_i p_i$).
- **For symmetric $A$**: $P$ can be chosen **orthogonal**, so $A = PDP^\top$. (Spectral theorem.)
- **Defective matrices** can't be diagonalized over $\mathbb{R}$ — need Jordan form (book skips).

### Geometric picture (3-step)
$A = PDP^{-1}$ acts as:
1. $P^{-1}$: rotate from standard basis into eigenbasis.
2. $D$: scale each axis by $\lambda_i$.
3. $P$: rotate back.

### Why diagonalize
- **Matrix powers**: $A^k = PD^kP^{-1}$ — $D^k$ is just $\lambda_i^k$ on diag. $O(n)$ per power.
- **Determinant**: $\det(A) = \det(D) = \prod \lambda_i$.
- **Matrix functions**: $f(A) = Pf(D)P^{-1}$ (e.g., $\exp(A)$, $A^{1/2}$ for PSD).

> **LLM/DL relevance — huge**:
> - **Dynamical systems view of training**: linearized dynamics $\dot\theta = -H\theta$ has solution $\theta(t) = e^{-Ht}\theta_0 = Pe^{-Dt}P^\top\theta_0$. Slow directions = small eigvals of $H$. Used to analyze training, momentum, learning-rate schedules (Jacot NTK, Cohen edge-of-stability).
> - **Spectral analysis of weight matrices**: eigenvalues of $WW^\top$ predict generalization (Martin & Mahoney). Stable rank $= \|W\|_F^2/\|W\|_2^2$ — built from eigenvalues.
> - **Matrix square root / inverse square root** via eigendecomp: needed for Shampoo, AdamW variants, whitening, batch norm at scale.
> - **Power iteration**: cheap top-eigenvalue compute — used for spectral norm regularization in GANs, Lipschitz networks.
> - **Mechanistic interp of attention heads**: Anthropic's "Mathematical Framework" decomposes OV circuit matrix's eigenstructure to identify positive/negative copying eigenvalues — eigenvectors literally are the directions the head writes into the residual stream.

---

## 4.5 Singular Value Decomposition (SVD)

> **Strang calls SVD the "fundamental theorem of linear algebra".** For ML, it's the single most important matrix decomposition — works for ANY matrix (rectangular, rank-deficient, anything). **Internalize cold.**

### Theorem 4.22 (SVD)
For any $A \in \mathbb{R}^{m\times n}$ of rank $r$:
$$A = U\Sigma V^\top$$
- $U \in \mathbb{R}^{m\times m}$ orthogonal, columns = **left singular vectors** $u_i$.
- $V \in \mathbb{R}^{n\times n}$ orthogonal, columns = **right singular vectors** $v_j$.
- $\Sigma \in \mathbb{R}^{m\times n}$ "diagonal" with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ on diag, zeros elsewhere (rectangular).
- $r = $ number of nonzero singular values $=$ rank.

### Geometric intuition (3-step, in possibly different spaces)
$A: \mathbb{R}^n \to \mathbb{R}^m$ as:
1. $V^\top$: basis change in the domain (rotation).
2. $\Sigma$: scale by $\sigma_i$ and embed/project between dimensions ($n \to m$).
3. $U$: basis change in the codomain (rotation).

Contrast with eigendecomposition: SVD uses **two different orthonormal bases** ($V$ for domain, $U$ for codomain), connected through $\Sigma$. Eigendecomposition uses same basis to rotate-scale-unrotate within one space.

### Construction (relation to eigendecomp)
- $A^\top A = V \Sigma^\top \Sigma V^\top$ → $V$ = eigenvectors of $A^\top A$, $\sigma_i^2 = \lambda_i(A^\top A)$.
- $AA^\top = U \Sigma\Sigma^\top U^\top$ → $U$ = eigenvectors of $AA^\top$.
- $u_i = \frac{1}{\sigma_i} A v_i$ (singular value equation $Av_i = \sigma_i u_i$).
- $A^\top A$ and $AA^\top$ share nonzero eigenvalues = $\sigma_i^2$.
- **Caveat**: forming $A^\top A$ explicitly is numerically bad (squares the condition number). Real SVD algorithms (Golub-Reinsch, randomized SVD) avoid it.

### SVD vs Eigendecomposition (table)

| Property | Eigendecomp $A = PDP^{-1}$ | SVD $A = U\Sigma V^\top$ |
|---|---|---|
| Existence | Square + non-defective | **Always exists** |
| Matrix shape | Square $n\times n$ | Any $m\times n$ |
| Basis vectors | $P$ generally not orthogonal | $U, V$ orthonormal (rotations) |
| Same basis? | Yes (one space) | No (domain ≠ codomain) |
| Diagonal entries | Eigenvalues (possibly complex/negative) | Singular values (real, $\ge 0$) |
| For symmetric PSD | $D = \Sigma$, $P = U = V$ | Same as eigendecomp |

### Variants
- **Full SVD**: $U \in \mathbb{R}^{m\times m}$, $V \in \mathbb{R}^{n\times n}$, $\Sigma$ rectangular.
- **Reduced/Thin SVD**: $U \in \mathbb{R}^{m\times r}$, $\Sigma \in \mathbb{R}^{r\times r}$, $V \in \mathbb{R}^{n\times r}$. Drops zero rows of $\Sigma$.
- **Truncated SVD** (rank-$k$): keep top $k$ singular triplets. This is the workhorse.

```python
# PyTorch
U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # thin SVD
# Truncated rank-k
Uk, Sk, Vhk = U[:, :k], S[:k], Vh[:k, :]
A_approx = Uk @ torch.diag(Sk) @ Vhk
```

### Example (Movie ratings, Example 4.14)
- People × movies rating matrix → SVD reveals latent "themes" (sci-fi, art house) as singular vectors.
- $u_1$ = sci-fi movies, $v_1$ = sci-fi lovers, $\sigma_1$ = strength of that theme.
- Classic **latent factor model**, ancestor of all matrix-factorization recommenders.

> **LLM/DL applications of SVD — extensive list**:
>
> - **LoRA (Low-Rank Adaptation)**: parameterize weight update as $\Delta W = BA$, $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times d}$. Direct application of low-rank structure. SVD of fine-tuning deltas is approximately low-rank → motivates LoRA. **DoRA, LoRA+, AdaLoRA** all build on SVD intuition.
> - **Model compression / weight surgery**: replace $W$ with $U_k\Sigma_k V_k^\top$ — fewer parameters, often fine for inference. Used in distillation, post-training compression.
> - **KV cache compression**: low-rank factorizations of K, V projections (MLA in DeepSeek, Multi-Query/Grouped-Query as structured low rank). DeepSeek's "Multi-head Latent Attention" literally projects K/V to low-rank latent and back.
> - **Activation analysis / probing**: SVD of activation matrices reveals "concept directions" — see Anthropic's dictionary learning is related (sparse vs dense factorization).
> - **Mechanistic interpretability**:
>   - SVD of OV / QK matrices to find writing/reading directions.
>   - "Singular vectors as features" (Millidge et al.).
>   - Logit lens, tuned lens — projections onto top singular directions of unembedding.
> - **Rank collapse in attention**: Dong et al. — repeated attention application drives token representations toward rank-1 subspace; analyzed via singular values.
> - **Stable rank** $\|W\|_F^2/\|W\|_2^2 = \sum\sigma_i^2/\sigma_1^2$ — effective rank metric used in generalization theory.
> - **Spectral norm regularization** (Miyato et al.): bound $\sigma_1(W)$ via power iteration; used in WGAN-SP, robust training.
> - **PCA / whitening**: SVD of centered data matrix. Used in BatchNorm theory, contrastive learning analysis.
> - **Linear least squares / pseudoinverse**: $A^+ = V\Sigma^+ U^\top$. Underlies linear probes, ridgeless regression.
> - **Randomized SVD** (Halko-Martinsson-Tropp): scale SVD to huge weight matrices via sketching. Essential for billion-param model analysis.
> - **Word embeddings**: classic LSA (latent semantic analysis) = truncated SVD of term-doc matrix. GloVe ≈ implicit matrix factorization. PMI matrix SVD reproduces word2vec (Levy & Goldberg).
> - **Recommender systems**: collaborative filtering = matrix factorization = approximate SVD with missing entries.

---

## 4.6 Matrix Approximation — Low-Rank and Eckart–Young

> **If you only learn one result from this chapter, learn Eckart–Young.** It's the math behind every compression, dimensionality reduction, and low-rank fine-tuning method in deep learning.

### Rank-1 outer-product representation
Any rank-$r$ matrix is a sum of $r$ rank-1 outer products:
$$A = \sum_{i=1}^r \sigma_i u_i v_i^\top = \sum_{i=1}^r \sigma_i A_i, \quad A_i := u_i v_i^\top$$

### Rank-$k$ approximation (truncated SVD)
$$\widehat{A}(k) = \sum_{i=1}^k \sigma_i u_i v_i^\top$$

- Storage: $k(m+n+1)$ numbers vs $mn$ original. Stonehenge example in book: rank-5 image uses 0.6% of original storage.
- Each successive $\sigma_i$ adds an "energy layer" to the reconstruction.

### Spectral norm (Definition 4.23)
$$\|A\|_2 := \max_{x \neq 0} \frac{\|Ax\|_2}{\|x\|_2} = \sigma_1$$
(largest singular value).

### Theorem 4.25 (Eckart–Young)
For any $k \le r$:
$$\widehat{A}(k) = \arg\min_{B: \text{rk}(B) = k} \|A - B\|_2, \qquad \|A - \widehat{A}(k)\|_2 = \sigma_{k+1}$$

**Truncated SVD is the optimal rank-$k$ approximation** (in spectral norm — and also Frobenius norm, though book proves only spectral). Error = next singular value you dropped.

> **Why this is THE most important practical result in the chapter**:
>
> - **LoRA** assumes $\Delta W$ has low intrinsic rank (~Aghajanyan 2020 "Intrinsic Dimensionality of Fine-Tuning"). Eckart–Young tells you the best rank-$r$ approximation is via SVD of $\Delta W$. AdaLoRA picks $r$ per layer based on singular value decay.
> - **Model pruning / structured pruning**: replace $W \approx U_k\Sigma_k V_k^\top$ — direct application.
> - **KV cache compression** (LESS, DeepSeek MLA, low-rank attention): use singular structure of K, V matrices over context.
> - **Activation steering / representation engineering**: project residual stream onto top-$k$ singular directions of a difference-in-means matrix to extract concept vectors.
> - **PCA = Eckart–Young applied to centered data covariance**. Every dimensionality reduction (UMAP, t-SNE init, autoencoders) is reaching for this.
> - **Spectral clustering, Laplacian eigenmaps, Isomap**: low-rank approximation of a similarity matrix.
> - **Matrix completion** (Netflix prize, MC for inpainting): nuclear-norm minimization is convex relaxation of "find low-rank $\hat A$ matching observed entries". Theory due to Candes-Recht.
> - **Empirical observation**: singular spectra of trained NN weight matrices show heavy tails / power laws ("Heavy-Tailed Self-Regularization", Martin & Mahoney) → predicts generalization without test data. Hot topic.
> - **Effective rank / participation ratio** $(\sum\sigma_i)^2 / \sum\sigma_i^2$ used in neural collapse analysis, in-context learning theory.

```python
# Low-rank approximation as model compression
W = layer.weight  # (out_dim, in_dim)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
k = 64
W_lowrank = (U[:, :k] * S[:k]) @ Vh[:k]
# Or store as two matrices for compute savings:
# A = (U[:, :k] * S[:k].sqrt())  # (out, k)
# B = (S[:k].sqrt().unsqueeze(1) * Vh[:k])  # (k, in)
# forward: x @ B.T @ A.T  -> O((out+in)*k) per matvec instead of out*in
```

```python
# LoRA pattern (training-time low-rank delta)
class LoRA(nn.Module):
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)  # init non-zero
        self.B = nn.Parameter(torch.zeros(d_out, r))         # init zero -> delta=0 at start
        self.scale = alpha / r
    def forward(self, x, base_W):
        return x @ base_W.T + self.scale * (x @ self.A.T) @ self.B.T
```

---

## 4.7 Matrix Phylogeny (Taxonomy)

A "family tree" of matrix types (Figure 4.13). Useful as a mental map.

```
Real matrices (∃ pseudoinverse, ∃ SVD)
├── Nonsquare  (only SVD, no det/trace/eigvals)
└── Square (∃ det, trace)
    ├── Singular (det = 0)        — not invertible
    └── Regular / Invertible (det ≠ 0)
        Orthogonal (A^T A = AA^T = I): rotations/reflections, σ_i = 1
        
    Cross-cut by defectiveness:
    ├── Defective (< n indep eigvecs) — no eigendecomposition
    └── Non-defective (diagonalizable, A = PDP^-1)
        ├── Non-normal (A^T A ≠ AA^T)
        └── Normal (A^T A = AA^T)
            └── Symmetric (S = S^T, real eigvals, ∃ ONB of eigvecs)
                ├── Positive definite (x^T S x > 0): Cholesky exists, σ = λ > 0
                │   └── Diagonal
                │       └── Identity
```

Key membership rules:
- **Symmetric ⊂ Normal ⊂ Non-defective ⊂ Square**.
- **Orthogonal ⊂ Regular**.
- **Positive definite ⊂ Symmetric**.
- **Non-singular** ≠ **non-defective**: rotation matrices are invertible but not diagonalizable over $\mathbb{R}$.

> **Why care**: when you see a matrix in a paper, classifying it tells you which tools apply. "Hessian is symmetric" → spectral theorem applies, real eigvals, ONB of eigvecs. "Attention matrix $QK^\top$" → generally not symmetric, no nice eigendecomp; but $QK^\top + (QK^\top)^\top$ symmetrization gives you something analyzable.

---

## 4.8 Further Reading + Loose ends

### Mentioned in book — modern ML connections
- **Spectral methods**: PCA, Fisher discriminant analysis, MDS, Isomap, Laplacian eigenmaps, Hessian eigenmaps, spectral clustering. All boil down to top-$k$ eigenvectors of some PSD kernel/Laplacian.
- **Tensor decompositions** (Tucker, CP): generalizations of SVD to higher-order arrays. Used in tensorized neural nets, TT-decomposition for compression, modeling multi-way interactions.
- **Reparametrization trick**: Cholesky enables differentiable sampling (VAE).
- **Numerical efficiency**: low-rank approx for compression and missing-data filling.

### Things NOT in MML but you should know
- **Jordan normal form**: for defective matrices. Book explicitly skips. Mostly theoretical for DL.
- **Schur decomposition**: $A = QTQ^\top$ with $T$ upper triangular, $Q$ orthogonal. Numerically stable; basis of QR algorithm.
- **QR decomposition**: $A = QR$, $Q$ orthogonal, $R$ upper-triangular. Used in solving least-squares, Gram-Schmidt. Cheaper than SVD when you just need orthogonal basis of column space.
- **LU decomposition**: $A = LU$. Solving general linear systems.
- **Randomized SVD** (Halko-Martinsson-Tropp 2011): the way you actually compute SVD of huge matrices. $O(mn\log k + (m+n)k^2)$. Standard in `scikit-learn`, used for analyzing LLM weight matrices.
- **Polar decomposition**: $A = UP$, $U$ orthogonal, $P$ PSD. Used in orthogonalization (e.g., orthogonal initialization in RNNs/transformers, Shampoo optimizer).
- **Pseudoinverse $A^+ = V\Sigma^+ U^\top$**: minimum-norm least-squares solver. Used in linear probes, post-hoc analyses.

---

## What to actually internalize (cheat-sheet)

### Tier S — must know cold
- **SVD existence + form** $A = U\Sigma V^\top$; what $U, V, \Sigma$ are; how to compute.
- **Eckart–Young**: truncated SVD is optimal low-rank approximation; error $= \sigma_{k+1}$.
- **Spectral theorem**: symmetric ⇒ real eigvals + orthonormal eigvecs.
- $A^\top A$ is symmetric PSD; eigendecomp gives right-singular vectors.
- Trace cyclic property, $\text{tr} = \sum\lambda$, $\det = \prod\lambda$.

### Tier A — fluently
- Eigenvalue equation, defectiveness, algebraic vs geometric multiplicity.
- Diagonalization $A = PDP^{-1}$ and what it means for $A^k$, $f(A)$.
- Cholesky for SPD, log-det via Cholesky.
- Spectral norm = $\sigma_1$, Frobenius norm $= \sqrt{\sum\sigma_i^2}$.

### Tier B — skim, lookup
- Laplace/Sarrus determinant formulas.
- Characteristic polynomial computation.
- Cholesky derivation step-by-step.

### Tier C — historical context, rarely useful directly
- Computing eigvals by rooting char polynomial (numerically bad).
- Computing SVD via $A^\top A$ (numerically bad).
- Sarrus rule.
- Jordan form (book skips it).

---

## Closing opinions (frontier-lab perspective)

- **The SVD is your scalpel.** Any time you suspect structure in a weight matrix, activation matrix, attention head, or dataset matrix — run SVD. The singular value spectrum tells you the rank, the energy distribution, and (via Eckart–Young) the cost of compression. Most "geometry of deep learning" papers boil down to "we ran SVD on something and observed structure."

- **Eigendecomposition is for symmetric things.** Hessian, Fisher, NTK, covariance, Gram, $W^\top W$ — all symmetric, spectral theorem applies. Whenever you see a symmetric PSD matrix in a paper, ask: what does its top eigenvector mean? What's the eigenvalue decay?

- **Low-rank is the unreasonable effectiveness story of DL.** Fine-tuning deltas are low-rank (LoRA). Activations are low-rank (linear probing works). Attention matrices rank-collapse. KV caches compress. Weight matrices have heavy-tailed spectra. The Eckart–Young theorem is the mathematical underpinning of *why these tricks work*.

- **Trace is a connective tissue.** You won't write a paper *about* trace, but it shows up in derivations constantly. Get fluent with the cyclic property; it makes matrix calculus painless.

- **Cholesky is niche but classy.** If you ever touch probabilistic ML (VAEs, GPs, diffusion priors with Gaussian latents, Laplace approximation, second-order optimization) you'll meet it. Otherwise, life goes on.

- **Determinant is mostly theoretical.** Useful for normalizing flows (Jacobian dets) and Gaussian log-likelihoods (via Cholesky). Don't bother memorizing Laplace expansion.

- **Skip the Jordan form rabbit hole.** Defective matrices barely matter in DL — real weight matrices are generically non-defective, and even when not, we treat them via SVD which always exists.
