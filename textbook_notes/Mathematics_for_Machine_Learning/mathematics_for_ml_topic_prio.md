# Mathematics for Machine Learning — Topic Priority Guide

> Perspective: senior researcher at a frontier lab, honest take on what actually matters for LLM work.

---

## The Short Version

If you're here to do LLM research, this book is foundational background — not something you'll cite in a paper, but stuff you need internalized so you're not confused when it comes up. The calculus and optimization chapters are the ones you'll thank yourself for knowing cold. Linear algebra is essential but in a narrower way than you'd expect. The second half (Ch 8–12) is mostly useful as context; SVMs in particular you can basically skip.

---

## Part I: Mathematical Foundations

### Ch 2 — Linear Algebra ⭐⭐⭐⭐⭐ (Essential)

This is non-negotiable. Matrices, vector spaces, linear maps — you're manipulating these every time you think about weight matrices, attention heads, or embedding spaces. The sections you'll use constantly:

- **Matrices and linear mappings** — every layer in a transformer is a linear map composed with a nonlinearity. You need this fluently.
- **Eigenvalues/eigenvectors (Ch 4.2)** — understanding the geometry of covariance matrices, understanding why attention patterns look the way they do. Also critical for understanding optimization dynamics (loss landscape curvature).
- **SVD (Ch 4.5) and matrix approximation (Ch 4.6)** — probably the most important decomposition in modern ML. LoRA (the dominant PEFT method) is literally low-rank approximation of weight matrices. Model compression, understanding over-parameterization, intrinsic dimensionality of fine-tuning — all SVD. Read this carefully.
- **Orthogonal projections (Ch 3.8)** — attention is computing projections. Understanding this geometrically helps a lot.

The affine spaces section (2.8) is fine to skim — mostly useful for understanding how bias terms work geometrically.

### Ch 3 — Analytic Geometry ⭐⭐⭐

Norms, inner products, distances — you'll use these constantly. Cosine similarity between embeddings? That's inner products over L2 norms. The rotations section (3.9) is less directly useful unless you're working on RoPE positional encodings (where rotations in 2D subspaces encode position), in which case it's actually relevant.

### Ch 4 — Matrix Decompositions ⭐⭐⭐⭐⭐ (Essential)

The single most practically useful chapter in this book for LLM researchers.

- **SVD (4.5)** — see above. Read it twice.
- **Eigendecomposition (4.4)** — needed to understand loss curvature, Hessians, and methods like KFAC or Shampoo that you'll encounter in optimization papers.
- **Cholesky (4.3)** — mainly useful if you work with Gaussian processes or Bayesian methods; less LLM-critical.
- **Determinant and Trace (4.1)** — understanding these is needed for probabilistic reasoning and for some normalization arguments; medium importance.

### Ch 5 — Vector Calculus ⭐⭐⭐⭐⭐ (Essential)

You can't do deep learning research without this. Key sections:

- **Backpropagation and Automatic Differentiation (5.6)** — this is the engine of everything. You need to understand how gradients flow through computation graphs. If you're debugging gradient issues or designing custom architectures, this is what you lean on.
- **Partial differentiation and Jacobians (5.2–5.3)** — critical for understanding how gradients flow through attention layers, softmax, normalization layers.
- **Gradients of matrices (5.4)** — when you're deriving custom loss functions or doing theory work, this matters.
- **Taylor series / linearization (5.8)** — useful for analyzing local behavior near optima; appears in NTK theory and understanding training dynamics.
- **Higher-order derivatives (5.7)** — mostly useful for curvature-based methods (second-order optimization), less commonly needed day-to-day.

### Ch 6 — Probability and Distributions ⭐⭐⭐⭐⭐ (Essential)

The probabilistic framing is fundamental to LLMs. Understand this well.

- **Sum/product rules, Bayes' theorem (6.3)** — the machinery you're using every time you think about posterior distributions, conditional generation, or calibration.
- **Gaussian distribution (6.5)** — ubiquitous. Weight initialization schemes, noise schedules in diffusion, variational methods — all Gaussian.
- **KL divergence / change of variables (6.7)** — critical. KL appears in RLHF (the KL penalty against the reference policy), in VAE training objectives (ELBO), and in understanding information-theoretic aspects of model training.
- **Exponential family and conjugacy (6.6)** — useful background for understanding why certain distributions behave nicely with certain priors; less directly used in practice but helps when reading theory papers.
- **Construction of probability spaces (6.1)** — mostly foundational formalism; skim unless you're doing theory work.

### Ch 7 — Continuous Optimization ⭐⭐⭐⭐⭐ (Essential)

This is core. You're doing gradient descent all day.

- **Gradient descent and variants (7.1)** — the mechanics. You need these cold: SGD, learning rate schedules, convergence intuitions.
- **Constrained optimization / Lagrange multipliers (7.2)** — less commonly used in practice, but shows up in RLHF policy derivations, in some regularization formulations, and in understanding why maximum entropy principles work.
- **Convex optimization (7.3)** — mostly useful as background. Deep learning is emphatically not convex, but understanding what convexity guarantees and why we don't have it helps calibrate your intuitions. Some alignment/safety work uses convexity arguments. Less day-to-day than the rest.

---

## Part II: Central ML Problems

### Ch 8 — When Models Meet Data ⭐⭐⭐

Empirical risk minimization, MLE, graphical models — this is the conceptual glue that ties together everything you're doing. The graphical models section (8.5) is useful for understanding dependencies and conditional independence, which is background for understanding why autoregressive generation works the way it does. Model selection (8.6) is practical but largely superseded by validation loss curves in practice.

### Ch 9 — Linear Regression ⭐⭐

Foundational but not directly used. The Bayesian linear regression section (9.3) is worth reading if you're interested in uncertainty quantification or working on calibration — it's the simplest illustration of posterior inference over parameters. Otherwise, skim for context.

### Ch 10 — Dimensionality Reduction / PCA ⭐⭐

PCA is less critical than it used to be — you're rarely running PCA on raw features. But the conceptual ideas (low-dimensional structure, variance maximization, projection perspective) show up in:
- Analyzing the intrinsic dimensionality of LLM representations
- Understanding why LoRA works (fine-tuning lives on a low-dim manifold)
- Mechanistic interpretability work (probing directions in activation space)

The latent variable perspective (10.7) is good background for VAEs. The high-dimensional PCA section (10.5) is useful if you work with representation analysis.

### Ch 11 — Gaussian Mixture Models / EM ⭐

Mostly background. GMMs and EM are historically important but not something you'll use directly in LLM research. The ELBO derivation here is useful because the same math shows up in VAEs and diffusion models — if you want to understand those, understanding EM first helps. Otherwise, this chapter is not that important.

### Ch 12 — Support Vector Machines ⭐

Largely irrelevant to modern LLM research. SVMs are still used in some classical ML applications and as evaluation probes in mechanistic interpretability (linear probes over representations), but you won't use the SVM machinery itself. The kernel section (12.4) is interesting theoretically and connects to NTK theory, but you don't need to understand SVMs to use that connection. Skip unless you're specifically interested.

---

## Summary Priority Table

| Chapter | Topic | Priority | Why |
|---------|-------|----------|-----|
| 4 | Matrix Decompositions (esp. SVD) | ⭐⭐⭐⭐⭐ | LoRA, compression, representation analysis |
| 5 | Vector Calculus / Backprop | ⭐⭐⭐⭐⭐ | The engine of training |
| 6 | Probability & Distributions | ⭐⭐⭐⭐⭐ | KL, MLE, everything generative |
| 7 | Continuous Optimization | ⭐⭐⭐⭐⭐ | Gradient descent is your job |
| 2 | Linear Algebra | ⭐⭐⭐⭐⭐ | Weight matrices, attention geometry |
| 3 | Analytic Geometry | ⭐⭐⭐ | Embeddings, similarity, RoPE |
| 8 | Models & Data | ⭐⭐⭐ | Conceptual grounding |
| 10 | PCA | ⭐⭐ | Representation analysis, LoRA intuition |
| 9 | Linear Regression | ⭐⭐ | Bayesian inference background |
| 11 | GMMs / EM | ⭐ | ELBO background only |
| 12 | SVMs | ⭐ | Not used; skip |
