# MML Chapter 2 — Linear Algebra

> Source: Deisenroth, Faisal, Ong — *Mathematics for Machine Learning*, Ch. 2 (pp. 17-63 printed).
> Voice: honest, applications-first take from someone who lives in PyTorch all day. This chapter is the bedrock for everything downstream in ML/LLM-land — but only a few subsections actually show up in your daily life as a researcher. I'll flag what matters.

---

## TL;DR — What Actually Matters for LLM Research

| Section | Why you care | Priority |
|---|---|---|
| **2.2 Matrices** (mult, transpose, inverse) | Every attention head, FFN, MLP is just $\boldsymbol{A}\boldsymbol{x}$. Cost model lives here. | **Must-know** |
| **2.4 Vector Spaces / Subspaces** | Mental model for embedding spaces, representations, "directions in latent space". | **Must-know** |
| **2.5 Linear Independence** + **2.6 Basis & Rank** | Rank governs LoRA, low-rank approximation, intrinsic dimensionality of features, mech interp. | **Must-know** |
| **2.7 Linear Mappings** (esp. image/kernel + rank-nullity) | Column space = what the layer can output. Null space = what gets thrown away. | **Must-know** |
| **2.3 Solving $\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$ / Gaussian elimination** | Conceptually important. In practice nobody runs Gaussian elimination on $10^{11}$ params; iterative + autodiff handle this. | Skim |
| **2.7.2 Basis change** | Matters when you read papers about "rotating" the residual stream (e.g. SVD, weight tying, mech interp rotations). | Useful |
| **2.1 Systems of linear equations** | Motivation only. | Skim |
| **2.4.1 Groups** | Formal background. You'll only re-encounter "general linear group" in equivariant nets/geometric DL. | Skim |
| **2.8 Affine spaces** | Bias terms are affine offsets. That's basically the whole connection. | Skim |

---

## 2.0 Big Picture: What Is Linear Algebra For?

- **Algebra** = set of objects + rules to manipulate them.
- **Linear algebra** = vectors + scalars, with addition and scalar multiplication closed.
- **Vectors are anything you can add and scale**: geometric arrows, polynomials, audio signals, $\mathbb{R}^n$ tuples, activations of a transformer layer.
	- The textbook focuses on $\mathbb{R}^n$ since finite-dim vector spaces are all isomorphic to it anyway.
	- **LLM lens**: every token embedding, residual stream state, KV cache entry, gradient is just an element of some $\mathbb{R}^d$.
- **Closure** is the central organizing idea: starting from a small set of vectors, what's the set of everything you can reach by adding and scaling? Answer: a vector (sub)space.

---

## 2.1 Systems of Linear Equations

- General form: $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ where $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, $\boldsymbol{x} \in \mathbb{R}^n$, $\boldsymbol{b} \in \mathbb{R}^m$.
- **Three possible outcomes**: no solution, unique solution, infinitely many solutions. Never two, never seventeen.
	- **Geometric**: each equation defines a hyperplane; solution set is intersection.
	- 2 eqs in $\mathbb{R}^2$: intersection of two lines (point / line / empty).
- **Column picture** (most useful intuition): $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ means "write $\boldsymbol{b}$ as a linear combination of $\boldsymbol{A}$'s columns, with weights $x_i$."
	- This is the picture you should burn in. Forget the row picture for most ML purposes.

> **Researcher take**: You will basically never *solve* a system by hand. But the column-picture intuition (every matrix-vector product is a linear combination of columns) is the single most useful mental model for understanding why transformers work — every attention output is a convex combo of value vectors.

---

## 2.2 Matrices — THE CORE

### 2.2.1 Matrix Multiplication

- $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, $\boldsymbol{B} \in \mathbb{R}^{n \times k}$ $\Rightarrow$ $\boldsymbol{C} = \boldsymbol{A}\boldsymbol{B} \in \mathbb{R}^{m \times k}$ with
$$c_{ij} = \sum_{l=1}^n a_{il}\, b_{lj}$$
- **Neighboring dims must match**. Memorize this until it's a reflex.
- **NOT commutative**: $\boldsymbol{A}\boldsymbol{B} \ne \boldsymbol{B}\boldsymbol{A}$ in general. Different shapes might even make one undefined.
- **NOT element-wise**. Element-wise is the *Hadamard product* ($\odot$ in papers, `*` in numpy/torch).
	- This bites everyone at some point. `A * B` in numpy is Hadamard, `A @ B` is matmul.
- **Properties**:
	- Associative: $(\boldsymbol{AB})\boldsymbol{C} = \boldsymbol{A}(\boldsymbol{BC})$ — basis of order-of-operations optimizations (e.g. compute $(\boldsymbol{Q}\boldsymbol{K}^\top)\boldsymbol{V}$ as $\boldsymbol{Q}(\boldsymbol{K}^\top \boldsymbol{V})$ in *linear attention* to skip the $O(n^2)$ matrix!)
	- Distributive over addition.
- **Identity**: $\boldsymbol{I}_n$; $\boldsymbol{I}\boldsymbol{A} = \boldsymbol{A}\boldsymbol{I} = \boldsymbol{A}$.

```python
# Three equivalent views — all useful
C = A @ B                          # matmul
C = torch.einsum('il,lj->ij', A, B)  # explicit indices
# (i) row × column dot products: C[i,j] = A[i,:] · B[:,j]
# (ii) linear combo of columns: C[:,j] = A @ B[:,j]
# (iii) sum of outer products: C = sum_l A[:,l] outer B[l,:]
```

> **Researcher take**: Matmul *is* the workhorse. A transformer forward pass is 95% matmuls; FLOPs ≈ $2 \cdot \text{params} \cdot \text{tokens}$ because each weight participates in one multiply-add per token. The associative trick (view (iii) — sum of outer products) is the foundation of **low-rank adaptation (LoRA)**: $\Delta \boldsymbol{W} = \boldsymbol{B}\boldsymbol{A}$ where $\boldsymbol{B} \in \mathbb{R}^{d\times r}$, $\boldsymbol{A} \in \mathbb{R}^{r\times d}$, $r \ll d$. View (ii) (matrix-vector = linear combo of columns) is the cleanest way to understand attention.

### 2.2.2 Inverse and Transpose

- **Inverse**: $\boldsymbol{A}^{-1}$ s.t. $\boldsymbol{A}\boldsymbol{A}^{-1} = \boldsymbol{A}^{-1}\boldsymbol{A} = \boldsymbol{I}$. Only defined for **square** matrices, and only when $\boldsymbol{A}$ is *regular/invertible/non-singular* (these terms all mean the same).
	- $2\times 2$ case: $\boldsymbol{A}^{-1} = \frac{1}{a_{11}a_{22} - a_{12}a_{21}} \begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix}$, denominator = determinant.
	- General case: computed via Gaussian elimination on $[\boldsymbol{A} | \boldsymbol{I}] \rightsquigarrow [\boldsymbol{I} | \boldsymbol{A}^{-1}]$.
	- **Cost**: $O(n^3)$ via Gauss; nobody computes $10^{10}\times 10^{10}$ inverses.
- **Transpose**: $\boldsymbol{B} = \boldsymbol{A}^\top$ with $b_{ij} = a_{ji}$. Rows become columns.
- **Identities to memorize** (you will use these constantly):
	- $(\boldsymbol{AB})^{-1} = \boldsymbol{B}^{-1}\boldsymbol{A}^{-1}$ ← order flip
	- $(\boldsymbol{AB})^\top = \boldsymbol{B}^\top \boldsymbol{A}^\top$ ← order flip
	- $(\boldsymbol{A}+\boldsymbol{B})^{-1} \ne \boldsymbol{A}^{-1} + \boldsymbol{B}^{-1}$ ← classic trap
	- $(\boldsymbol{A}^\top)^\top = \boldsymbol{A}$
	- $(\boldsymbol{A}^{-1})^\top = (\boldsymbol{A}^\top)^{-1} =: \boldsymbol{A}^{-\top}$
- **Symmetric**: $\boldsymbol{A} = \boldsymbol{A}^\top$. Only square matrices can be symmetric.
	- Sum of symmetric is symmetric; product is not.
	- **Gram matrices** $\boldsymbol{A}^\top \boldsymbol{A}$ are always symmetric (and PSD). Used everywhere: covariance, kernel methods, attention scores.

> **Researcher take**: You'll see $(\boldsymbol{A}^\top \boldsymbol{A})^{-1}\boldsymbol{A}^\top \boldsymbol{b}$ (normal equations) in classical ML. In modern LLM-land you almost never *invert* a real matrix; you solve linear systems implicitly via SGD/Adam, or use iterative solvers (CG, GMRES) when needed. Inverses do show up in: K-FAC / Shampoo (preconditioners), Gauss-Newton, natural gradient, Laplace approximation, GP regression. Knowing inverse identities matters because they show up in derivations even if you never invert anything.

### 2.2.3 Scalar Multiplication
- $\lambda \boldsymbol{A}$ scales every entry. Associative, distributive, plays nicely with transpose ($(\lambda\boldsymbol{C})^\top = \lambda \boldsymbol{C}^\top$).
- Total non-event. Don't lose sleep.

### 2.2.4 Compact Representation
- $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ ← whole equation system in one expression.
- Mental model: $\boldsymbol{A}\boldsymbol{x}$ = linear combination of $\boldsymbol{A}$'s columns weighted by $x_i$.

---

## 2.3 Solving Linear Systems

### 2.3.1 Particular + General Solution

**Strategy** (the standard recipe):
1. Find a **particular solution** $\boldsymbol{x}_p$ with $\boldsymbol{A}\boldsymbol{x}_p = \boldsymbol{b}$.
2. Find **all solutions to homogeneous** $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}$ (this is the null space / kernel).
3. **General solution** = $\boldsymbol{x}_p$ + (anything in null space). All solutions form an *affine* set.

### 2.3.2 Elementary Row Operations + Gaussian Elimination

- **Allowed ops** (preserve solution set):
	- Swap two rows.
	- Multiply a row by $\lambda \ne 0$.
	- Add a multiple of one row to another.
- Apply these to the **augmented matrix** $[\boldsymbol{A} | \boldsymbol{b}]$ until in **row-echelon form (REF)**.
- **REF**:
	- All-zero rows at bottom.
	- Leading nonzero of each row (the **pivot**) strictly right of pivot above. "Staircase".
- **Reduced REF (RREF)**: REF + every pivot is 1 + pivot is the only nonzero in its column.
- **Basic variables** = pivot columns; **free variables** = the rest. Free variables become parameters of the general solution.

### 2.3.3 Minus-1 Trick
- Algorithmic trick to read off a basis of the kernel directly from RREF by inserting $-1$ rows at missing pivot positions. Cute but not worth memorizing — just know it exists.

### 2.3.4 Computing Inverses
- Run Gaussian elimination on $[\boldsymbol{A} | \boldsymbol{I}_n] \rightsquigarrow [\boldsymbol{I}_n | \boldsymbol{A}^{-1}]$.

### Algorithms for $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ in practice
- If $\boldsymbol{A}$ square + invertible: $\boldsymbol{x} = \boldsymbol{A}^{-1}\boldsymbol{b}$ (conceptually; never actually invert numerically).
- If overdetermined / non-square: use **Moore-Penrose pseudoinverse** $(\boldsymbol{A}^\top \boldsymbol{A})^{-1}\boldsymbol{A}^\top$ — gives the min-norm least-squares solution.
	- Numerically bad (squares the condition number). Use SVD-based pinv instead.
- **Gaussian elimination is $O(n^3)$** → impractical for millions of variables.
- For huge sparse systems: **iterative methods** — Jacobi, Gauss-Seidel, SOR (stationary), or **Krylov subspace** methods: **conjugate gradients (CG)**, GMRES, BiCG.
	- Iteration: $\boldsymbol{x}^{(k+1)} = \boldsymbol{C}\boldsymbol{x}^{(k)} + \boldsymbol{d}$ converging to fixed point.

> **Researcher take**: For LLMs, you basically never solve a linear system the textbook way. SGD/Adam are doing implicit iterative solving every step. But CG is still alive: it's the backbone of **Hessian-vector-product** based second-order methods (Newton-CG, K-FAC variants), of **influence functions** for data attribution, and of GP marginal likelihood. Skim Gaussian elimination, get the intuition for REF/pivots, then move on.

---

## 2.4 Vector Spaces

### 2.4.1 Groups (skim)
- $(\mathcal{G}, \otimes)$ with closure, associativity, neutral element, inverses.
- **Abelian** = also commutative.
- Examples: $(\mathbb{Z}, +)$, $(\mathbb{R}\setminus\{0\}, \cdot)$.
- **General linear group $GL(n, \mathbb{R})$**: invertible $n\times n$ matrices under multiplication. Non-abelian (matmul isn't commutative).

> **Researcher take**: Groups themselves rarely matter for LLM work day-to-day. But they're foundational for **equivariant neural networks** (SE(3) for proteins/molecules, $SO(3)$ for 3D vision, permutation groups for set-structured data, Lie groups in geometric DL). If you go into that area, come back and learn this properly.

### 2.4.2 Vector Space — Real Workhorse

A **vector space** $V = (\mathcal{V}, +, \cdot)$:
- $(\mathcal{V}, +)$ is an abelian group (closed under addition, has zero, inverses).
- Outer op $\cdot: \mathbb{R} \times \mathcal{V} \to \mathcal{V}$ (scalar multiplication).
- Distributivity (both ways), associativity of scalar mult, $1\cdot \boldsymbol{x} = \boldsymbol{x}$.

**Standard examples**: $\mathbb{R}^n$, $\mathbb{R}^{m\times n}$ (matrices form a vector space!), $\mathbb{C}$, polynomial spaces, function spaces.

**Useful conventions**:
- Treat $\mathbb{R}^n$, $\mathbb{R}^{n\times 1}$ as the same. Vectors are columns by default; rows are $\boldsymbol{x}^\top$.
- For vectors: only outer product $\boldsymbol{a}\boldsymbol{b}^\top \in \mathbb{R}^{n\times n}$ and inner product $\boldsymbol{a}^\top \boldsymbol{b} \in \mathbb{R}$ are well-defined matrix operations. "Element-wise" $\boldsymbol{a} \cdot \boldsymbol{b}$ is just a programming convenience (Hadamard), not standard math.

### 2.4.3 Vector Subspaces

$U \subseteq V$ is a **subspace** if:
1. $\boldsymbol{0} \in U$ (in particular $U \ne \emptyset$).
2. **Closed under +**: $\boldsymbol{x}, \boldsymbol{y} \in U \Rightarrow \boldsymbol{x} + \boldsymbol{y} \in U$.
3. **Closed under scalar mult**: $\lambda \in \mathbb{R}, \boldsymbol{x} \in U \Rightarrow \lambda \boldsymbol{x} \in U$.

- **All subspaces contain the origin** — a subspace that "misses zero" isn't a subspace (that's an affine space).
- **Solution set of $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}$** is always a subspace of $\mathbb{R}^n$ (the null space / kernel).
- **Solution set of $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$, $\boldsymbol{b}\ne\boldsymbol{0}$** is *not* a subspace (it's an affine subspace).
- Intersection of subspaces is a subspace. (Union usually isn't!)

> **Researcher take**: The mental picture of subspaces is huge for representation learning. "The model has learned a 12-dim subspace of the 4096-dim residual stream that encodes politeness" — that's a subspace claim. **Mech interp** is largely about identifying meaningful subspaces. **Steering vectors / activation engineering** add a vector to push activations into/along a specific subspace direction. **PCA, ICA, sparse autoencoders** are all "find an interesting subspace / basis of subspace" methods. If you internalize one section of this chapter, make it this one + 2.6.

---

## 2.5 Linear Independence

- **Linear combination**: $\boldsymbol{v} = \sum_i \lambda_i \boldsymbol{x}_i$.
- **Linearly dependent**: $\exists$ non-trivial $\lambda$'s (not all zero) with $\sum \lambda_i \boldsymbol{x}_i = \boldsymbol{0}$.
- **Linearly independent**: only the trivial $\lambda_i = 0$ works.

**Intuition**: independence = no redundancy. Remove any one vector and you lose information about the spanned space.

**Quick sanity checks**:
- Any set containing $\boldsymbol{0}$ is dependent.
- Two vectors with one a scalar multiple of the other: dependent.
- $m > k$ linear combos of $k$ independent vectors → those $m$ vectors are dependent (pigeonhole).
- **Algorithmic test**: stack as columns of matrix, row-reduce to REF. Independent iff every column is a pivot column.

> **Researcher take**: Linear independence underlies the entire notion of *effective rank* / *intrinsic dimensionality* of features. When you compute a 4096-dim activation but the "true" information is in a 50-dim subspace, the columns of the activation matrix are highly dependent. This is what motivates LoRA (gradients of fine-tuning lie in a low-rank subspace), **bottleneck layers**, distillation, **JL-style** random projections, sketching. The Kigali/Kampala/Nairobi example in the book is a nice toy: 3 vectors in $\mathbb{R}^2$ are necessarily dependent.

---

## 2.6 Basis and Rank

### 2.6.1 Generating Set, Span, Basis

- **Generating set** $\mathcal{A}$ of $V$: every $\boldsymbol{v} \in V$ can be written as linear combo of elements of $\mathcal{A}$.
- **Span**: $\mathrm{span}[\mathcal{A}] = \{\text{all linear combos of elements of } \mathcal{A}\}$.
- **Basis**: minimal generating set ≡ maximal linearly independent set. Equivalent characterizations:
	- $\mathcal{B}$ is a minimal generating set.
	- $\mathcal{B}$ is a maximal linearly independent set.
	- Every $\boldsymbol{x} \in V$ has a *unique* representation $\boldsymbol{x} = \sum \lambda_i \boldsymbol{b}_i$.

**Examples**:
- **Canonical / standard basis** of $\mathbb{R}^n$: $\boldsymbol{e}_1, \ldots, \boldsymbol{e}_n$. The "axes".
- Bases are not unique — there are infinitely many bases of $\mathbb{R}^n$.
- All bases of $V$ have the same cardinality = $\dim(V)$.

**Dimension** = number of basis vectors = "number of independent directions". Subspace $U \subseteq V$: $\dim(U) \le \dim(V)$, equality iff $U = V$.

**Recipe to find basis of $U = \mathrm{span}[\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m]$**:
1. Stack $\boldsymbol{x}_i$ as columns of $\boldsymbol{A}$.
2. Compute REF of $\boldsymbol{A}$.
3. The original columns corresponding to pivot columns form a basis of $U$.

### 2.6.2 Rank — THE KEY INVARIANT

$\mathrm{rk}(\boldsymbol{A})$ = number of linearly independent columns = number of linearly independent rows.

**Properties you must know**:
- $\mathrm{rk}(\boldsymbol{A}) = \mathrm{rk}(\boldsymbol{A}^\top)$ (column rank = row rank — a small miracle).
- Columns of $\boldsymbol{A}$ span a subspace of $\mathbb{R}^m$ (the **image / column space / range**) of dim $\mathrm{rk}(\boldsymbol{A})$.
- Rows span subspace of $\mathbb{R}^n$ of same dim.
- Square $\boldsymbol{A} \in \mathbb{R}^{n\times n}$: invertible $\iff \mathrm{rk}(\boldsymbol{A}) = n$ (full rank).
- $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ solvable $\iff \mathrm{rk}(\boldsymbol{A}) = \mathrm{rk}([\boldsymbol{A}|\boldsymbol{b}])$.
- $\dim(\ker \boldsymbol{A}) = n - \mathrm{rk}(\boldsymbol{A})$ (rank-nullity, see 2.7.3).
- **Full rank**: $\mathrm{rk}(\boldsymbol{A}) = \min(m, n)$. Otherwise **rank-deficient**.

> **Researcher take**: This is one of the highest-leverage concepts in modern ML. Where rank actually shows up:
> - **LoRA / QLoRA / DoRA**: parameter-efficient fine-tuning assumes weight updates are low-rank. $\boldsymbol{W} \leftarrow \boldsymbol{W} + \boldsymbol{B}\boldsymbol{A}$ with $\mathrm{rk} \le r$.
> - **Low-rank approximation / SVD**: best rank-$k$ approximation of a matrix (Eckart-Young). Foundation of PCA, latent semantic analysis, model compression, **Shampoo**, K-FAC.
> - **Attention rank collapse**: in deep transformers attention matrices can collapse to rank-1. Real failure mode discussed in mech interp papers.
> - **Neural tangent kernel** regimes: low effective rank governs implicit bias of GD.
> - **Linear probing**: how much information about concept X is linearly decodable = effective rank of the probe's solution subspace.
> - **Sparse autoencoder dictionaries**: overcomplete (rank > dim) but with structured sparsity.
> - **Intrinsic dimensionality** of fine-tuning (Aghajanyan et al.) — empirically rank ~100s suffices.

---

## 2.7 Linear Mappings — THE Bridge from Math to ML

A **linear mapping** (homomorphism / linear transformation) $\Phi: V \to W$ satisfies
$$\Phi(\lambda \boldsymbol{x} + \psi \boldsymbol{y}) = \lambda \Phi(\boldsymbol{x}) + \psi \Phi(\boldsymbol{y}).$$
That's it. Every matrix multiplication is a linear map and vice versa (for finite-dim).

**Type taxonomy**:
- **Injective**: $\Phi(\boldsymbol{x}) = \Phi(\boldsymbol{y}) \Rightarrow \boldsymbol{x} = \boldsymbol{y}$ (one-to-one).
- **Surjective**: $\Phi(V) = W$ (onto).
- **Bijective**: both. Equivalent to invertible.
- **Isomorphism**: linear + bijective.
- **Endomorphism**: $V \to V$.
- **Automorphism**: $V \to V$ + bijective.
- **Identity mapping**: $\mathrm{id}_V$.

**Big theorem (2.17)**: Finite-dim $V, W$ are isomorphic iff $\dim V = \dim W$.
- So $\mathbb{R}^{m\times n} \cong \mathbb{R}^{mn}$ — that's why "flattening" a matrix into a vector is mathematically harmless.

### 2.7.1 Matrix Representation of Linear Mappings

**Coordinates**: With ordered basis $B = (\boldsymbol{b}_1, \ldots, \boldsymbol{b}_n)$, every $\boldsymbol{x}$ has unique coords $\boldsymbol{\alpha} \in \mathbb{R}^n$ with $\boldsymbol{x} = \sum \alpha_i \boldsymbol{b}_i$. The "vector" is basis-free; coordinates are not.

**Transformation matrix**: For linear $\Phi: V \to W$ with bases $B$ of $V$, $C$ of $W$, define $\boldsymbol{A}_\Phi \in \mathbb{R}^{m\times n}$ by
$$\Phi(\boldsymbol{b}_j) = \sum_{i=1}^m \alpha_{ij} \boldsymbol{c}_i, \quad A_\Phi(i,j) = \alpha_{ij}.$$
Then if $\hat{\boldsymbol{x}}$ is the coord rep of $\boldsymbol{x}$ in $B$ and $\hat{\boldsymbol{y}}$ is the coord rep of $\Phi(\boldsymbol{x})$ in $C$:
$$\hat{\boldsymbol{y}} = \boldsymbol{A}_\Phi \hat{\boldsymbol{x}}.$$

**Examples** (all matrices in $\mathbb{R}^2$):
- Rotation by $\pi/4$: $\begin{bmatrix} \cos\frac{\pi}{4} & -\sin\frac{\pi}{4} \\ \sin\frac{\pi}{4} & \cos\frac{\pi}{4} \end{bmatrix}$.
- Stretch along $x_1$ by 2: $\mathrm{diag}(2, 1)$.
- General linear map: arbitrary $2\times 2$ matrix.

### 2.7.2 Basis Change

**Question**: If you change the basis in $V$ and/or $W$, how does $\boldsymbol{A}_\Phi$ change?

**Theorem 2.20**: With new bases $\tilde{B}$ of $V$, $\tilde{C}$ of $W$:
$$\tilde{\boldsymbol{A}}_\Phi = \boldsymbol{T}^{-1} \boldsymbol{A}_\Phi \boldsymbol{S}$$
where
- $\boldsymbol{S}$ is the **change-of-basis matrix** taking $\tilde{B}$-coords to $B$-coords (columns of $\boldsymbol{S}$ = $\tilde{\boldsymbol{b}}_j$ in $B$-coords).
- $\boldsymbol{T}$ does the same in $W$.

**Special cases**:
- **Equivalence**: $\tilde{\boldsymbol{A}} = \boldsymbol{T}^{-1}\boldsymbol{A}\boldsymbol{S}$ for some invertible $\boldsymbol{S}, \boldsymbol{T}$.
- **Similarity** (endomorphisms, same basis on both sides): $\tilde{\boldsymbol{A}} = \boldsymbol{S}^{-1}\boldsymbol{A}\boldsymbol{S}$. Used heavily in eigendecomposition.

> **Researcher take**: Basis change is the math behind a lot of mech interp moves. "Rotate the residual stream so the attention output is diagonal" = pick a basis where $\boldsymbol{A}_\Phi$ is simpler. SVD literally diagonalizes $\boldsymbol{A}$ in well-chosen bases ($\boldsymbol{U}, \boldsymbol{V}$). **Weight tying** assumes you can re-use a basis. **Concept directions** found via probing live in a basis you choose. Practical version: similarity transforms preserve eigenvalues (so eigenvalues of $\boldsymbol{W}^l$, e.g. spectral norms / Lipschitz analyses, are basis-independent).

### 2.7.3 Image and Kernel — Critical for Understanding Layers

For $\Phi: V \to W$:
- **Kernel / null space**: $\ker(\Phi) = \{\boldsymbol{v} \in V : \Phi(\boldsymbol{v}) = \boldsymbol{0}_W\}$.
- **Image / range**: $\mathrm{Im}(\Phi) = \{\Phi(\boldsymbol{v}) : \boldsymbol{v} \in V\} \subseteq W$.
- **Domain**: $V$. **Codomain**: $W$.

**Always true**:
- $\boldsymbol{0}_V \in \ker(\Phi)$. Kernel is never empty.
- $\ker(\Phi) \le V$ subspace; $\mathrm{Im}(\Phi) \le W$ subspace.
- $\Phi$ injective $\iff \ker(\Phi) = \{\boldsymbol{0}\}$.

**For a matrix $\boldsymbol{A} \in \mathbb{R}^{m\times n}$ as a map $\mathbb{R}^n \to \mathbb{R}^m$**:
- $\mathrm{Im}(\Phi) = \mathrm{span}[\boldsymbol{a}_1, \ldots, \boldsymbol{a}_n]$ = **column space** $\subseteq \mathbb{R}^m$.
- $\ker(\Phi)$ = solution set of $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}$ $\subseteq \mathbb{R}^n$.
- $\mathrm{rk}(\boldsymbol{A}) = \dim(\mathrm{Im}(\Phi))$.

### Rank-Nullity (Theorem 2.24) — "Fundamental Theorem of Linear Mappings"
$$\dim(\ker \Phi) + \dim(\mathrm{Im}\, \Phi) = \dim(V)$$

Consequences:
- If $\dim(\mathrm{Im}) < \dim(V)$, kernel is non-trivial → infinitely many solutions to $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}$.
- For $\dim V = \dim W$: injective $\iff$ surjective $\iff$ bijective.

> **Researcher take**: This is the math underlying *what a layer can and can't represent*. A linear projection from $d_\text{in}$ to $d_\text{out} < d_\text{in}$ necessarily has a kernel of dim $\ge d_\text{in} - d_\text{out}$ — information is destroyed. Bottleneck layers, MLP down-projections in transformers ($d_\text{ff} \to d_\text{model}$), embedding layers (vocab → $d$): all losing information into kernels. Concept erasure literature (INLP, RLACE) explicitly tries to *enlarge* the kernel to remove a concept. Conversely, you want $\mathrm{Im}$ to be expressive — that's why MLP up-projection $d \to 4d$ comes first. Rank-nullity is the inescapable accounting identity behind all of this.

---

## 2.8 Affine Spaces (skim)

**Affine subspace** = subspace + offset:
$$L = \boldsymbol{x}_0 + U = \{\boldsymbol{x}_0 + \boldsymbol{u} : \boldsymbol{u} \in U\}$$
where $U \le V$ is a vector subspace (the *direction*) and $\boldsymbol{x}_0$ is the *support point*.

- If $\boldsymbol{x}_0 \notin U$, $L$ is **not** a vector subspace (doesn't contain $\boldsymbol{0}$).
- Examples: lines/planes/hyperplanes not through origin.
- **Parametric form**: $\boldsymbol{x} = \boldsymbol{x}_0 + \sum_{i=1}^k \lambda_i \boldsymbol{b}_i$.
- **Lines** = 1-dim affine subspaces. **Planes** = 2-dim. **Hyperplane** = $(n-1)$-dim affine subspace of $\mathbb{R}^n$.
- Solutions of inhomogeneous $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ form an affine subspace of dim $n - \mathrm{rk}(\boldsymbol{A})$.

**Affine mapping**: $\phi(\boldsymbol{x}) = \boldsymbol{a} + \Phi(\boldsymbol{x})$ where $\Phi$ is linear and $\boldsymbol{a}$ is the **translation vector**.
- Every affine map = linear map + translation. Composition of affine maps is affine. Bijective affine maps preserve parallelism and dimension.

> **Researcher take**: The whole reason "linear layer" in PyTorch is `nn.Linear` with a bias is because it implements an **affine** map $\boldsymbol{x} \mapsto \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}$. Drop the bias and you get a *true* linear map (passes through origin). For LLMs, bias terms are usually small or zero; LayerNorm-style normalizations often eat the bias. Hyperplanes show up as **decision boundaries** in linear classifiers and as the "concept directions" in linear probing. Beyond that, affine geometry is mostly formalism. Skim and move on.

---

## Bonus: Reusable Mental Models

| Concept | One-line mental model | Where you see it in LLMs |
|---|---|---|
| Matrix multiplication | "Linear combos of columns, with weights from the right-hand vector" | Everywhere — attention, FFN, embeddings |
| Inverse | "Undo the map" | Implicit in optimization; rarely computed directly |
| Vector space | "Closed-under-add-and-scale playground" | Embedding/activation spaces |
| Subspace | "A flat slice through origin" | Concept directions, LoRA subspaces, probes |
| Span | "Everything you can reach by mixing these vectors" | Sparse autoencoder dictionaries |
| Linear independence | "No redundancy in this set" | Effective dimensionality of features |
| Basis | "Minimal complete coordinate system" | Choice of which directions to interpret |
| Dimension | "Number of true degrees of freedom" | Intrinsic dim of fine-tuning, NTK rank |
| Rank | "How much info this map preserves" | LoRA $r$, attention collapse, compression |
| Image/column space | "Outputs the map can produce" | What a layer can express |
| Kernel/null space | "Inputs the map sends to zero" | What a layer destroys (e.g. concept erasure) |
| Rank-nullity | "Info in = info out + info destroyed" | Bottleneck design, lossy projection |
| Basis change | "Re-coordinate without changing the underlying map" | SVD, mech interp rotations, weight symmetry |
| Affine map | "Linear + bias" | `nn.Linear(d_in, d_out, bias=True)` |

---

## Pragmatic Code Reference

```python
import torch

# matmul vs Hadamard - the eternal confusion
A = torch.randn(3, 4); B = torch.randn(4, 5)
C1 = A @ B                           # matmul, shape (3,5)
C2 = torch.einsum('ij,jk->ik', A, B) # explicit

# transpose
At = A.T                             # or A.transpose(-1,-2) for batch

# inverse (only square invertible; usually avoid)
M = torch.randn(4,4); Minv = torch.linalg.inv(M)

# better than inv for solving Mx=b
x = torch.linalg.solve(M, b)

# pseudoinverse (Moore-Penrose) - use this for least squares
A_pinv = torch.linalg.pinv(A)
x_lstsq = torch.linalg.lstsq(A, b).solution

# rank (numerical, via SVD)
r = torch.linalg.matrix_rank(A)

# null space / kernel via SVD
U, S, Vh = torch.linalg.svd(A)
# columns of Vh.T corresponding to ~zero singular values span ker(A)

# basis change of a transformation B (in standard basis) to new basis P
# where columns of P are the new basis vectors expressed in std basis
# (so P maps new-coords -> std-coords, P_inv: std -> new)
B_new = torch.linalg.solve(P, B @ P)  # = P^-1 B P  (similarity transform)
```

---

## Quick Self-Check Before Moving On

- Can you state when $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ has no/one/infinite solutions in terms of ranks? (Yes if $\mathrm{rk}(\boldsymbol{A}) = \mathrm{rk}([\boldsymbol{A}|\boldsymbol{b}])$; unique if also $= n$.)
- Why is $(\boldsymbol{AB})^{-1} = \boldsymbol{B}^{-1}\boldsymbol{A}^{-1}$ and not $\boldsymbol{A}^{-1}\boldsymbol{B}^{-1}$? (Order reversal — verify by $\boldsymbol{AB} \cdot \boldsymbol{B}^{-1}\boldsymbol{A}^{-1} = \boldsymbol{I}$.)
- What's the difference between a subspace and an affine subspace? (Origin / no origin.)
- For $\boldsymbol{A}\in\mathbb{R}^{4\times 7}$ with $\mathrm{rk}(\boldsymbol{A}) = 3$: what's $\dim(\ker)$? ($7 - 3 = 4$.) What's $\dim(\mathrm{Im})$? (3.) Largest possible rank? ($\min(4,7) = 4$.)
- Why does LoRA work? (Empirically the update $\Delta\boldsymbol{W}$ has low rank, so factoring as $\boldsymbol{B}\boldsymbol{A}$ with small $r$ captures it.)

---

**Bottom line**: This chapter sets up the vocabulary for everything. Memorize matmul rules cold, internalize subspaces and rank, treat affine spaces and group theory as background. Next chapters add geometry (inner products, norms, projections) — that's where things get interesting for ML.
