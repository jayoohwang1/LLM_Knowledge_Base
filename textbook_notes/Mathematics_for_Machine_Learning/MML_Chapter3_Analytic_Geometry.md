# MML Chapter 3 — Analytic Geometry

> Chapter 2 was algebra (vector spaces, linear maps) at an abstract level. Chapter 3 puts **geometry** on top: lengths, angles, distances, orthogonality, projections, rotations. Almost every modern LLM/DL trick you care about lives downstream of this chapter (attention = inner products, LayerNorm = norm rescaling, LoRA/PCA = projections, RoPE = rotations, weight decay = norm penalty, RAG retrieval = cosine similarity). Skim the abstract definitions, internalize the geometric intuition.

---

## TL;DR — What Actually Matters for LLM Research

> **Opinionated priority ranking** (in terms of "if you only learned X from this chapter, you'd be fine"):

| Concept | LLM relevance | Why |
|---|---|---|
| **Inner products / dot products** | Critical | Attention scores $QK^\top$, retrieval, similarity, kernels. The single most important geometric op in DL. |
| **Norms ($\ell_2$, $\ell_1$, $\ell_\infty$)** | Critical | LayerNorm/RMSNorm, weight decay, gradient clipping, spectral norm, embedding normalization for cosine. |
| **Orthogonal projections** | Critical | LoRA, SVD compression, PCA, ablating directions in activations (interp), Gram-Schmidt-style initializations. |
| **Orthogonality / orthonormal bases** | High | Orthogonal init, parameter-efficient methods (OFT/BOFT), Householder reflections, stable training. |
| **Angles (cosine)** | High | Cosine similarity for RAG/embeddings; representational similarity; CKA. |
| **Rotations** | Medium | RoPE positional encodings; mostly $\mathrm{SO}(n)$ structure matters for equivariance and PE. |
| **Inner products of functions** | Low-ish | Conceptually nice (kernels, Fourier, attention-as-operator views) but rarely invoked directly. |
| **Gram-Schmidt / pseudoinverse** | Medium | Conceptual foundation for QR, least squares, lots of "Procrustes" tricks. |

---

## 3.1 Norms

> **Definition.** A norm on $V$ is a function $\|\cdot\|: V \to \mathbb{R}$, $x \mapsto \|x\|$ satisfying:
> - **Absolutely homogeneous**: $\|\lambda x\| = |\lambda|\,\|x\|$
> - **Triangle inequality**: $\|x+y\| \le \|x\| + \|y\|$
> - **Positive definite**: $\|x\| \ge 0$, with $=0 \iff x = 0$

- **Geometric reading**: norms = "lengths." Triangle inequality = the geometric fact that going through a detour can only be longer.
- **Manhattan / $\ell_1$**: $\|x\|_1 = \sum_i |x_i|$. Unit ball is a diamond. Encourages **sparsity** (Lasso, L1-regularized weights, top-K attention pruning ideas).
- **Euclidean / $\ell_2$**: $\|x\|_2 = \sqrt{\sum_i x_i^2} = \sqrt{x^\top x}$. Unit ball is a sphere. Induced by the dot product. **The default everywhere.**
- **$\ell_\infty$** (not in chapter but worth noting): $\|x\|_\infty = \max_i |x_i|$. Used in adversarial robustness (PGD attacks), gradient clipping by max.
- **$\ell_p$ general**: $\|x\|_p = (\sum_i |x_i|^p)^{1/p}$. Between $\ell_1$ (corner-seeking, sparse) and $\ell_\infty$ (max-bounded).

### Why norms are everywhere in LLM-land

- **LayerNorm / RMSNorm**: literally divide activations by their $\ell_2$ norm (RMSNorm uses $\sqrt{\frac{1}{n}\sum x_i^2}$, basically $\|x\|_2/\sqrt{n}$). Without this, training transformers is unstable.
  - **RMSNorm** vs **LayerNorm**: RMSNorm drops the mean centering. Modern LLMs (LLaMA, GPT-NeoX) mostly use RMSNorm — cheaper and works.
- **Weight decay = $\ell_2$ penalty** on weights. AdamW decouples it from gradient. Equivalent to Gaussian prior in MAP framing.
- **Gradient clipping**: clip $g \leftarrow g \cdot \min(1, \tau/\|g\|_2)$. Essential for stable training of large nets.
- **Spectral norm** (operator $\ell_2 \to \ell_2$ norm of a matrix = top singular value): used in spectral normalization (GANs), Muon optimizer (steepest descent under spectral norm), analyses of Lipschitz constants, mu-Transfer scaling laws.
- **Embedding normalization**: many retrieval setups L2-normalize embeddings so dot product ≡ cosine similarity. E.g., sentence-transformers, CLIP image/text embeddings.
- **Norm growth tracking** during training: people watch $\|W\|_F$, activation norms, gradient norms as diagnostics of training health (Soft-MoE, Chinchilla-style debugging).

### Matrix norms (not in chapter, but the natural extension)

- **Frobenius**: $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$ — treat matrix as flat vector. Used in LoRA loss regularizers.
- **Spectral**: $\|A\|_2 = \sigma_{\max}(A)$ — controls input/output gain.
- **Nuclear**: $\|A\|_* = \sum_i \sigma_i$ — convex surrogate for rank, used in low-rank recovery.

---

## 3.2 Inner Products

> **Definition (general inner product).** A symmetric, positive-definite bilinear map $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$. We write $\langle x, y \rangle$ for the inner product.

- **Bilinear**: linear in each argument (with the other fixed).
- **Symmetric**: $\langle x, y \rangle = \langle y, x \rangle$.
- **Positive definite**: $\langle x, x \rangle > 0$ for $x \ne 0$.

### 3.2.1 Dot Product

- $x^\top y = \sum_i x_i y_i$. The canonical inner product on $\mathbb{R}^n$.
- **(V, dot product)** = Euclidean vector space.

### 3.2.2 General Inner Products

- Example of a non-dot product on $\mathbb{R}^2$: $\langle x, y \rangle = x_1 y_1 - (x_1 y_2 + x_2 y_1) + 2 x_2 y_2$.
- **Key insight**: different inner products → different geometries (different angles, lengths, notions of "similar"). The choice of inner product is the choice of geometry.

### 3.2.3 Symmetric, Positive Definite (SPD) Matrices

> **Theorem 3.5.** Every inner product on a finite-dim space can be written $\langle x, y \rangle = \hat{x}^\top A \hat{y}$ for some SPD matrix $A$.

- **SPD matrix** $A \in \mathbb{R}^{n\times n}$ iff symmetric ($A = A^\top$) and $x^\top A x > 0\ \forall x \ne 0$.
- **Properties**:
  - Null space = $\{0\}$
  - All diagonal entries $a_{ii} > 0$
  - All eigenvalues $> 0$ (chapter 4)
  - Has a Cholesky factorization $A = LL^\top$
- **Why SPD matters in LLM/DL**:
  - **Covariance matrices** are SPSD (positive semi-definite) — show up in PCA, whitening, Mahalanobis distance, natural gradient.
  - **Fisher Information Matrix** is SPSD — basis of K-FAC, Shampoo, natural gradient optimizers.
  - **Hessian** of a convex loss is SPSD — second-order optimizers (Newton, GGN approximations).
  - **Kernel matrices** $K_{ij} = k(x_i, x_j)$ are SPSD — Gaussian processes, kernel-PCA.
  - **Mahalanobis distance** $d(x,y) = \sqrt{(x-y)^\top A (x-y)}$ — used in OOD detection on embeddings.
  - **Riemannian / Fisher–Rao geometry**: $A(\theta)$ varies with location, giving curved geometry on parameter manifold.

---

## 3.3 Lengths and Distances

> Every inner product **induces a norm**: $\|x\| := \sqrt{\langle x, x \rangle}$. Not every norm comes from an inner product (e.g., $\ell_1$ doesn't).

- **Cauchy-Schwarz**: $|\langle x, y \rangle| \le \|x\|\,\|y\|$. Single most-used inequality in analysis/ML proofs.
- **Distance / metric**: $d(x,y) = \|x - y\| = \sqrt{\langle x-y, x-y\rangle}$.
  - Properties: positive definite, symmetric, triangle inequality.
- **Euclidean distance** = induced by dot product.
- **Note**: $\langle x, y \rangle$ and $d(x, y)$ behave *oppositely* — similar $x,y$ → large $\langle x, y\rangle$ but small $d(x,y)$.

### LLM-relevant uses

- **L2 distance** between embeddings — used in FAISS-style retrieval (HNSW indexes can use either L2 or inner product).
- **Cosine distance** $= 1 - \cos\omega$ — used for sentence/document similarity, RAG, semantic search.
- **KL divergence** is not a metric (not symmetric, no triangle inequality) but is the natural divergence on probability distributions. Comes up in distillation, RLHF (KL penalty against ref policy).
- **Wasserstein distance** — actual metric on distributions, used in optimal transport / GAN training.

---

## 3.4 Angles and Orthogonality

> Cauchy-Schwarz $\implies -1 \le \frac{\langle x,y\rangle}{\|x\|\|y\|} \le 1$, so:
> $$\cos\omega = \frac{\langle x, y\rangle}{\|x\|\,\|y\|}$$
> defines a unique $\omega \in [0, \pi]$.

- **Cosine similarity** = $\cos\omega$ for the dot product. The single most-used similarity in NLP/IR.
- **Orthogonal**: $\langle x, y\rangle = 0$, written $x \perp y$. **Orthonormal**: orthogonal + unit length.
- $0$-vector is orthogonal to everything.
- **Orthogonality is inner-product-dependent**: vectors orthogonal under one inner product need not be under another. E.g., $x=[1,1], y=[-1,1]$ are orthogonal under dot product but not under $\langle x, y\rangle = x^\top \mathrm{diag}(2,1) y$.

### Orthogonal Matrices

> **Definition 3.8.** $A \in \mathbb{R}^{n\times n}$ is **orthogonal** iff columns are orthonormal:
> $$A A^\top = A^\top A = I \quad\implies\quad A^{-1} = A^\top.$$

- **Pedantic note**: textbook calls them "orthogonal" but "orthonormal" is more accurate (columns are orthonormal, not just orthogonal).
- **Geometric properties**:
  - **Preserve lengths**: $\|Ax\|^2 = x^\top A^\top A x = x^\top x = \|x\|^2$.
  - **Preserve angles**: $\cos\omega$ between $Ax, Ay$ equals $\cos\omega$ between $x, y$.
  - These are isometries of Euclidean space = **rotations + reflections** = $\mathrm{O}(n)$ (with $\det = \pm 1$); rotations alone = $\mathrm{SO}(n)$ ($\det=+1$).

### Why orthogonality is huge in LLM research

- **Orthogonal initialization** (Saxe et al.): initialize weights as random orthogonal matrices. Preserves activation norms through layers, helps train deep nets, motivates dynamical isometry.
- **Parameter-efficient finetuning via rotations**:
  - **OFT (Orthogonal Finetuning)** and **BOFT**: instead of additive low-rank LoRA updates, multiply weights by a learned orthogonal $R$: $W' = RW$. Preserves spectrum, more stable.
  - Parameterize via Cayley transform or Householder reflections to stay on the orthogonal manifold.
- **Householder reflections**: $H = I - 2vv^\top / \|v\|^2$ — orthogonal, rank-1 update. Used in normalizing flows (e.g., Sylvester flows), efficient orthogonal parameterizations.
- **Mechanistic interpretability**: orthogonal components of representations (e.g., "concept directions") matter. Ablating along a direction = projecting out, requires orthogonal complement reasoning.
- **Superposition / feature directions**: Anthropic's superposition hypothesis is fundamentally about how non-orthogonal directions in a smaller space can encode many features. Polysemanticity = lack of orthogonality.
- **Compositional rotations in RoPE** (see §3.9): each frequency band rotates a 2D subspace; cumulative composition is still orthogonal.

---

## 3.5 Orthonormal Basis (ONB)

> **Definition.** Basis $\{b_1, \ldots, b_n\}$ of $V$ is **orthonormal** iff $\langle b_i, b_j\rangle = \delta_{ij}$ (1 if $i=j$, else 0).

- If only $\langle b_i, b_j\rangle = 0$ for $i\ne j$, it's an **orthogonal basis** (lengths arbitrary).
- **Canonical basis** $\{e_1,\ldots,e_n\}$ of $\mathbb{R}^n$ is the standard ONB (with dot product).
- **Gram-Schmidt** constructs ONB from any basis (covered in §3.8.3).
- **Why this rules in practice**:
  - In an ONB, coordinates of $x$ are just inner products: $x = \sum_i \langle x, b_i\rangle b_i$.
  - The Gram matrix is $I$, so least-squares / projection reduces to dot products (no matrix inverses).
  - **PCA basis** is orthonormal (eigenvectors of symmetric covariance). Used everywhere: dimensionality reduction of activations, weight matrices, "feature dictionaries."
  - **SVD** gives orthonormal bases for row/column spaces — backbone of low-rank compression (LoRA, model pruning, quantization-aware decompositions).

---

## 3.6 Orthogonal Complement

> Given $D$-dim $V$ and $M$-dim subspace $U \subseteq V$, the **orthogonal complement** $U^\perp$ is the $(D-M)$-dim subspace of vectors orthogonal to every vector in $U$.
> - $U \cap U^\perp = \{0\}$.
> - Every $x \in V$ uniquely decomposes as $x = \sum_{m=1}^M \lambda_m b_m + \sum_{j=1}^{D-M} \psi_j b_j^\perp$.

- **Normal vector** $w$ to a 2D plane in 3D: $\mathrm{span}\{w\} = U^\perp$, $\|w\| = 1$.
- **Hyperplane** = $(n-1)$-dim subspace, described by single normal vector.
- **LLM/DL uses**:
  - **Activation steering / direction ablation**: removing a "refusal direction" or "concept direction" from activations is $x \leftarrow x - (w^\top x) w$ where $w$ spans the direction — i.e., project onto $U^\perp$.
  - **SVD-based compression**: split signal/noise via top-$k$ basis and its orthogonal complement.
  - **Decision boundaries of SVMs** and classifier heads: $w^\top x + b = 0$, $w$ is normal to the decision hyperplane.
  - **Null-space tuning / orthogonal-subspace continual learning**: update weights only in $U^\perp$ of previous task's gradient/activation subspace to avoid catastrophic forgetting (e.g., O-LoRA).

---

## 3.7 Inner Product of Functions

> $\langle u, v\rangle := \int_a^b u(x) v(x)\, dx$ generalizes the dot product to (infinite-dim) function spaces. With proper measure theory, gives a **Hilbert space**.

- **Example**: $\sin$ and $\cos$ are orthogonal on $[-\pi, \pi]$ (integrand $\sin x \cos x$ is odd).
- $\{1, \cos x, \cos 2x, \ldots\}$ is an orthogonal family; projecting onto its span = **Fourier series**.

### Modest LLM relevance

- **Kernel methods**: $k(x, y) = \langle \phi(x), \phi(y)\rangle$ for feature map $\phi$ — often infinite-dim. RBF kernel = inner product in an infinite-dim function space.
- **Attention as kernel / operator**: viewing attention as a learned kernel/integral operator pulls on Hilbert-space intuition. Performer (random features), linear attention, RKHS framings, "attention is an operator" papers (Tsai et al.).
- **Fourier features / RoPE / Sinusoidal PE**: positional encoding via $\sin/\cos$ basis exploits orthogonality of the Fourier family.
- **Neural Tangent Kernel (NTK)**: training infinite-width nets = kernel regression in an RKHS.
- **Honest take**: you mostly don't need to compute these integrals directly. The conceptual scaffold (everything in DL can be cast as inner products in some feature space) is the takeaway.

---

## 3.8 Orthogonal Projections

> **THE most useful concept in this chapter for modern DL.** PCA, LoRA, least squares, ablations, mech interp, low-rank compression — all projections.

> **Definition (Projection).** Linear $\pi: V \to U$ with $\pi^2 = \pi$ (**idempotent**). As matrix: $P_\pi^2 = P_\pi$. **Orthogonal projection**: additionally, $(x - \pi(x)) \perp U$.

### 3.8.1 Projection onto a Line (1D subspace)

For line $U = \mathrm{span}\{b\}$, projection $\pi_U(x) = \lambda b$. Derivation in 3 steps:

1. **Find coordinate $\lambda$**. Orthogonality $\langle x - \lambda b, b\rangle = 0$ gives:
   $$\lambda = \frac{\langle x, b\rangle}{\langle b, b\rangle} = \frac{b^\top x}{\|b\|^2}.$$
   If $\|b\| = 1$, $\lambda = b^\top x$ — just a dot product.
2. **Find projection point**: $\pi_U(x) = \lambda b = \dfrac{b^\top x}{\|b\|^2} b = \dfrac{b b^\top}{\|b\|^2} x$.
3. **Projection matrix**:
   $$\boxed{P_\pi = \frac{b b^\top}{\|b\|^2}}$$
   - Symmetric, rank 1, idempotent.
   - $\|\pi_U(x)\| = |\cos\omega|\,\|x\|$ when $\|b\| = 1$ — pure trigonometry.

### 3.8.2 Projection onto General Subspace

For subspace $U$ with basis $B = [b_1, \ldots, b_m] \in \mathbb{R}^{n\times m}$. Projection $\pi_U(x) = B\lambda$.

- Orthogonality $\Rightarrow B^\top(x - B\lambda) = 0$ → **normal equation**:
  $$\boxed{B^\top B \lambda = B^\top x \quad\Rightarrow\quad \lambda = (B^\top B)^{-1} B^\top x}$$
- **Projection itself**:
  $$\pi_U(x) = B (B^\top B)^{-1} B^\top x$$
- **Projection matrix**:
  $$\boxed{P_\pi = B(B^\top B)^{-1} B^\top}$$
- **Pseudo-inverse**: $B^+ = (B^\top B)^{-1} B^\top$ — the Moore-Penrose pseudo-inverse (for full-column-rank $B$).
- **Projection error / reconstruction error**: $\|x - \pi_U(x)\|$.
- **Big simplification when $B$ has orthonormal columns** ($B^\top B = I$):
  $$P_\pi = B B^\top, \qquad \lambda = B^\top x.$$
  - No inverse needed! This is why PCA / SVD bases are convenient.

#### Code intuition

```python
# Project x onto column space of B (Python / NumPy)
import numpy as np

# General case (B has linearly independent cols)
lam = np.linalg.solve(B.T @ B, B.T @ x)        # coordinates
proj = B @ lam                                  # projection
P = B @ np.linalg.solve(B.T @ B, B.T)           # projection matrix

# Orthonormal columns (cheap)
lam = B.T @ x
proj = B @ lam
```

### Massive list of LLM/DL uses

- **Least squares / linear regression**: solve $Ax = b$ when no exact solution. Best $\hat{x}$ minimizes $\|b - A\hat{x}\|$, i.e., projects $b$ onto column space of $A$. Same normal equation. Adding **jitter $\epsilon I$** to $B^\top B$ → **ridge regression**.
- **PCA = projection onto top-$k$ eigenvectors of covariance**. Used for:
  - Activation visualization (probing).
  - Weight compression / structured pruning.
  - Whitening before training (rare in transformers but ubiquitous in vision/RL feature pipelines).
- **LoRA**: $W + \Delta W$ with $\Delta W = BA$, $A \in \mathbb{R}^{r\times d}$, $B \in \mathbb{R}^{d \times r}$. The update lives in a **rank-$r$ subspace** — implicitly projects gradients/updates into a low-dim subspace of weight space. Geometric framing: gradient is projected onto $\mathrm{range}(B) \otimes \mathrm{range}(A^\top)$.
- **SVD-based model compression / quantization**:
  - Replace $W \approx U_k \Sigma_k V_k^\top$ — projection onto best rank-$k$ approximation (Eckart-Young).
  - **GPT-style activation-aware**: AWQ, GPTQ use Hessian-aware projection of weight matrices.
- **Mechanistic interpretability**:
  - **Direction ablation**: $x \to x - (w^\top x) w$ removes refusal/sycophancy/etc. directions (Arditi et al. on refusal direction).
  - **Causal tracing / activation patching**: implicitly works with subspaces of residual stream.
  - **Probing classifiers**: linear probes find directions encoding a concept; projection onto/away from these directions tests causality.
- **Low-rank gradient methods**: GaLore projects gradients onto a low-rank subspace updated periodically — memory-efficient training.
- **Speculative decoding / model distillation**: align student/teacher representations via projection.
- **Continual learning / multi-task**: Orthogonal Gradient Descent (OGD) projects new task gradients onto orthogonal complement of past gradient subspaces.

### 3.8.3 Gram-Schmidt Orthogonalization

> Construct ONB iteratively:
> $$u_1 := b_1, \qquad u_k := b_k - \pi_{\mathrm{span}[u_1,\ldots,u_{k-1}]}(b_k).$$
> Then normalize each $u_k$.

- Each step: subtract the part of $b_k$ already lying in the span of earlier vectors → what remains is orthogonal.
- **Practical note**: classical Gram-Schmidt is numerically unstable; in practice use **Modified Gram-Schmidt** or **Householder QR**. NumPy's `np.linalg.qr` uses Householder.
- **Uses**:
  - QR factorization (least squares, eigensolvers).
  - Orthogonal initialization (Saxe).
  - Power iteration with deflation for top-$k$ eigenvectors.
  - **Lanczos / Krylov subspace methods** (conjugate gradient, GMRES) for huge linear systems — used inside second-order optimizers (CG-Newton, Hessian-free) and influence functions for LLMs.

### 3.8.4 Projection onto Affine Subspaces

> Affine $L = x_0 + U$. To project: translate by $-x_0$, project onto subspace $U$, translate back:
> $$\pi_L(x) = x_0 + \pi_U(x - x_0).$$
> Distance to $L$ = distance from $x - x_0$ to $U$.

- **Uses**:
  - **SVM separating hyperplanes** (Ch. 12) — hyperplanes are affine, not linear.
  - **Solution sets** $\{x : Ax = b\}$ for $b \ne 0$ are affine subspaces.
  - **Manifold tangent spaces** — locally affine.
  - Conceptually underpins **steering vectors with bias** in interp / activation engineering.

---

## 3.9 Rotations

> **Rotation** = orthogonal matrix with $\det = +1$ (no flip). Lives in $\mathrm{SO}(n)$. Preserves lengths and angles.

### 3.9.1 Rotations in $\mathbb{R}^2$

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}.$$

- Counterclockwise by $\theta$.
- $R(\theta_1) R(\theta_2) = R(\theta_1 + \theta_2)$ — abelian in 2D.

### 3.9.2 Rotations in $\mathbb{R}^3$

- 3 elementary rotations about $e_1, e_2, e_3$ axes.
- **Not commutative** ($\mathrm{SO}(3)$ is non-abelian).

### 3.9.3 Rotations in $n$ Dimensions — Givens Rotations

> **Givens rotation** $R_{ij}(\theta)$ = identity $I_n$ with $\cos\theta$ at positions $(i,i), (j,j)$ and $\mp\sin\theta$ at $(i,j), (j,i)$. Rotates in the $e_i, e_j$ plane, leaves other coordinates alone.

- Any $\mathrm{SO}(n)$ matrix is product of Givens rotations.

### 3.9.4 Properties

- Preserve distances and angles (since orthogonal).
- **Non-commutative in 3D+**; abelian in 2D.

### LLM uses (selective)

- **RoPE (Rotary Position Embeddings)**: applies block-diagonal 2D Givens rotations to query/key vectors, with rotation angle $\theta_i = \mathrm{pos} \cdot \omega_i$ varying by frequency band. Inner product of rotated queries and keys depends only on **relative** position — a beautiful exploitation of $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$.
- **Orthogonal finetuning (OFT/BOFT)**: $W' = R W$ where $R$ is parameterized as product of Givens / Householder / Cayley to live in $\mathrm{SO}(n)$.
- **Equivariant networks**: $\mathrm{SO}(3)$-equivariant nets for molecules/proteins (e.g., AlphaFold's structure module rotates frames; Equiformer). Less relevant for text LLMs but huge in science ML.
- **Robotics, computer graphics**: rotation matrices, quaternions, rotation prediction. Not LLM but adjacent.

---

## 3.10 Further Reading (Author's Pointers + My Commentary)

- **Kernel methods**: many linear algorithms can be expressed purely via inner products → "kernel trick" lets you compute them in (possibly infinite-dim) feature spaces. Gaussian processes, kernel-PCA, SVMs. Modern relevance: NTK theory, attention-as-kernel papers, RBF/random features.
- **Projections everywhere**: graphics (shadows), optimization (residual minimization), regression, PCA. In LLMs: LoRA, SVD compression, activation ablation, GaLore, low-rank Hessian approximations.
- **Recommended deeper reading**: Axler "Linear Algebra Done Right" (axiomatic), Boyd-Vandenberghe "Introduction to Applied Linear Algebra" (applied).

---

## Senior-Researcher-Take Cheat Sheet

> If a junior LLM researcher asked "what should I really internalize from this chapter," I'd say:

1. **Inner product = similarity, norm = magnitude, projection = best low-dim approximation.** That trio describes 80% of what's happening geometrically in any deep net.
2. **Attention is inner products at scale**: $\mathrm{softmax}(QK^\top/\sqrt{d})V$. Understanding how cosine and dot product behave in high dimensions (concentration, scale of $\sqrt{d}$ normalization) is part of grokking transformers.
3. **Normalize aggressively**: LayerNorm/RMSNorm, weight decay, gradient clipping, embedding normalization. Almost any DL stability problem traces back to some norm exploding or collapsing.
4. **Low-rank ≈ projection**. LoRA, SVD compression, PCA, mechanistic-interp direction ablations are all the same geometric operation in different costumes. Master $P = B(B^\top B)^{-1}B^\top$ and its simplification $P = BB^\top$ for orthonormal $B$.
5. **Orthogonal $\ne$ independent (but for ONB, they coincide).** When constructing bases for analysis (probing directions, interpretability subspaces, weight init), aim for orthonormal — math is cleaner, computations are stable.
6. **RoPE is the only place rotations matter much for transformer-LLMs.** But the orthogonal-matrix structure showing up in OFT/BOFT and equivariant architectures is worth knowing.
7. **Don't fear non-Euclidean inner products** $\langle x, y\rangle = x^\top A y$. They show up as Mahalanobis distance, Fisher-Rao metric, natural gradient — the "right" geometry on parameter manifolds.
8. **Pseudo-inverse $(B^\top B)^{-1} B^\top$ + jitter $\epsilon I$ = ridge regression.** That one trick (Tikhonov regularization) saves you in least-squares fits all over ML.

---

## Quick Symbol/Formula Reference

| Object | Formula |
|---|---|
| $\ell_1$ norm | $\|x\|_1 = \sum_i \|x_i\|$ |
| $\ell_2$ norm | $\|x\|_2 = \sqrt{x^\top x}$ |
| Dot product | $x^\top y = \sum_i x_i y_i$ |
| General inner product | $\langle x, y\rangle = x^\top A y$, $A$ SPD |
| Induced norm | $\|x\| = \sqrt{\langle x, x\rangle}$ |
| Cauchy-Schwarz | $\|\langle x,y\rangle\| \le \|x\|\|y\|$ |
| Cosine angle | $\cos\omega = \frac{\langle x, y\rangle}{\|x\|\|y\|}$ |
| Orthogonality | $\langle x, y\rangle = 0$ |
| Orthogonal matrix | $A^\top A = A A^\top = I$ |
| 1D projection matrix | $P = bb^\top / \|b\|^2$ |
| General projection | $P = B(B^\top B)^{-1} B^\top$ |
| ONB projection | $P = BB^\top$, $\lambda = B^\top x$ |
| Pseudo-inverse | $B^+ = (B^\top B)^{-1} B^\top$ |
| Normal equation | $B^\top B \lambda = B^\top x$ |
| Gram-Schmidt step | $u_k = b_k - \pi_{\mathrm{span}[u_1,\ldots,u_{k-1}]}(b_k)$ |
| 2D rotation | $R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\\sin\theta & \cos\theta\end{bmatrix}$ |
| Affine projection | $\pi_L(x) = x_0 + \pi_U(x - x_0)$ |
