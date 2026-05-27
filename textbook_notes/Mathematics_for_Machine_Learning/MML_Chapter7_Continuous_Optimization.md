# MML Chapter 7 — Continuous Optimization

> Training ML models = continuous optimization over $\mathbb{R}^D$. The chapter splits into **unconstrained** (Section 7.1, gradient descent + variants), **constrained** (Section 7.2, Lagrange multipliers/duality), and **convex** (Section 7.3, LP/QP, Legendre–Fenchel/convex conjugate). Convention: **minimize** $f$; gradients point uphill so we go $-\nabla f$.

---

## TL;DR — What matters for LLMs / frontier ML

| Topic | Textbook depth | Actually used at frontier labs? | Why |
|---|---|---|---|
| Vanilla GD | Heavy | Almost never | Replaced by Adam/AdamW/Lion/Shampoo. But the conceptual core (follow $-\nabla$) is the foundation of everything. |
| Step size / learning rate | Light | Critical | LR schedules (warmup + cosine/linear decay), $\mu$P/scaling laws, layerwise LR all matter enormously. |
| Momentum | Light | Critical | Adam's first-moment term IS momentum. Nesterov, heavy-ball, all live inside modern optimizers. |
| SGD | Light | Foundational | All modern optimizers are stochastic. Mini-batch noise is a feature (regularization/escape). |
| Lagrange multipliers / KKT | Medium | Very (conceptually) | Regularization = constrained opt; **RLHF KL constraint**, PPO trust region, DPO derivation, constrained decoding all rely on this. |
| Linear / Quadratic programs | Medium | Niche | Old-school SVMs, calibration, scheduling. Rarely show up directly in LLM training. |
| Convex optimization theory | Medium | Indirectly | NN loss is non-convex but convex intuitions (Jensen, strong duality, conjugates) guide algorithm design. |
| **Legendre–Fenchel / convex conjugate** | Medium | **Critical for theory** | **DPO**, entropy regularization in RL, exponential family / softmax duality, **mirror descent**, max-entropy IRL, variational inference all use it. |

**Opinionated take**: Day-to-day you write `optimizer = AdamW(...)`. The math from this chapter that actually pays off when you're designing new training objectives is **(a)** Lagrangian view of constraints (KL-constrained RL → unconstrained Lagrangian-penalty form), and **(b)** **convex conjugate** (this is the secret machinery behind DPO/IPO closed forms, max-ent RL softmax policies, F-divergences, mirror descent). The book's LP/QP material is mostly orthogonal to modern LLM work.

---

## 7.0 Setup

- **Problem**: $\min_{\boldsymbol{x}} f(\boldsymbol{x})$, $f:\mathbb{R}^D \to \mathbb{R}$, assumed differentiable.
- **Stationary points**: roots of $\nabla f = 0$. Could be min/max/saddle. Distinguish via Hessian $\nabla^2 f$ sign.
- **Local vs global minimum**: in 1D toy $\ell(x)=x^4+7x^3+5x^2-17x+3$, global at $x \approx -4.5$, local at $x \approx 0.7$. Starting point matters for non-convex problems.
- **Why optimization, not closed form?** Abel–Ruffini: no algebraic solution for polynomial degree $\geq 5$. In high-D ML, almost never have closed forms — iterative.
- **Convex functions**: special class where all local minima are global. NN loss landscapes are wildly non-convex but many ML losses (logistic, hinge, MSE on linear models, regularizers) are convex in their primitive form.

---

## 7.1 Optimization Using Gradient Descent

**Core update**:
$$\boldsymbol{x}_{i+1} = \boldsymbol{x}_i - \gamma_i (\nabla f(\boldsymbol{x}_i))^\top$$

- Book uses **row-vector gradients** (hence transpose). Most PyTorch code treats grads as same shape as params — convention drift, no semantic change.
- **Geometric intuition**: $-\nabla f$ is steepest descent direction; gradients are **orthogonal to contour lines** ($f=c$ level sets).
- **First-order**: only uses gradient, no curvature. → slow near optima, especially in ill-conditioned valleys.

### 7.1.1 Step size (learning rate)

- Too small → slow. Too large → overshoot, diverge.
- **Naive adaptive heuristic** (Toussaint 2012):
  - If $f$ goes up after a step, **undo** and **shrink** $\gamma$.
  - If $f$ goes down, **try increasing** $\gamma$.
  - Monotone but wasteful.
- **Condition number** $\kappa = \sigma_{\max}(A)/\sigma_{\min}(A)$ controls convergence speed for $Ax=b$ style problems. Long thin valleys = poor conditioning = zigzag.
- **Preconditioning**: solve $P^{-1}(Ax-b)=0$ where $P^{-1}A$ is better-conditioned. $P$ must be cheap to apply.

> **LLM lens**: This is where the textbook lags reality. In practice:
> - **LR schedules**: linear warmup → cosine decay (Chinchilla-style) or WSD (warmup-stable-decay). Warmup matters because early-training gradient stats are unstable.
> - **Adaptive methods** (Adam, AdamW, Lion, Adafactor, Shampoo, SOAP, Muon) all *implicitly* precondition. Adam uses diagonal $1/\sqrt{v}$ preconditioning. Shampoo/Muon do approximate full-matrix (Kronecker) preconditioning — this is the next frontier (Muon for hidden weights matters at scale).
> - **$\mu$P / $\mu$Transfer** (Yang et al.): principled LR scaling across model widths so HP transfers from small → big models. Saves .
> - **Layerwise LR**: LARS/LAMB for large-batch, AdamW with separate LR for embeddings vs body is common.
> - **Gradient clipping**: not in the book at all, but ubiquitous in LLM training (prevents loss spikes from rare bad batches).

### 7.1.2 Gradient Descent with Momentum

**Update**:
$$\boldsymbol{x}_{i+1} = \boldsymbol{x}_i - \gamma_i (\nabla f(\boldsymbol{x}_i))^\top + \alpha \Delta \boldsymbol{x}_i$$
$$\Delta \boldsymbol{x}_i = \boldsymbol{x}_i - \boldsymbol{x}_{i-1} = \alpha \Delta \boldsymbol{x}_{i-1} - \gamma_{i-1}(\nabla f(\boldsymbol{x}_{i-1}))^\top$$

- **Heavy-ball analogy**: reluctant to change direction. Dampens zigzags.
- **Smoothing noisy gradient estimates** (helpful for SGD).
- $\alpha \in [0,1]$. Typical $\alpha \in [0.9, 0.99]$.

> **LLM lens**:
> - **Adam's $m_t$**: exponential moving average of gradients = momentum.
> - **Nesterov momentum**: lookahead variant, "evaluate gradient at where you'd be after the momentum step" — slightly better convergence in convex case, mostly cosmetic in deep learning.
> - **Lion (Chen et al. 2023)**: uses only sign of momentum-EMA. Memory savings, competitive perf — used in some PaLM-line work.
> - **Polyak averaging / EMA of weights**: separate concept (averaging *parameters* not gradients) — heavily used for diffusion model checkpointing and stabilizing RL/SSL.

### 7.1.3 Stochastic Gradient Descent (SGD)

**Sum-structured objective**:
$$L(\boldsymbol{\theta}) = \sum_{n=1}^N L_n(\boldsymbol{\theta})$$
e.g. negative log-likelihood $L(\boldsymbol{\theta}) = -\sum_n \log p(y_n|\boldsymbol{x}_n,\boldsymbol{\theta})$.

- **Batch GD**: use all $N$ — exact but expensive.
- **Mini-batch SGD**: subsample $B$ examples → unbiased gradient estimate.
- **Single-sample SGD**: $B=1$, max noise.
- **Convergence**: under decreasing step size (Robbins–Monro: $\sum \gamma_i = \infty$, $\sum \gamma_i^2 < \infty$), SGD converges a.s. to local min (Bottou 1998).

**Mini-batch size tradeoffs**:
| Large batch | Small batch |
|---|---|
| Lower variance, smoother | Higher variance |
| Better hardware utilization (matmul throughput) | Faster wallclock per step |
| May get stuck in sharp minima | Noise helps escape bad local optima → better generalization (folklore + Keskar et al.) |
| Higher per-step memory | Lower memory |

> **LLM lens**:
> - **Critical batch size** (McCandlish et al. 2018): there's a regime where doubling batch ≈ halving steps. Beyond it, diminishing returns.
> - **Gradient accumulation**: simulate large batch on small memory.
> - **Data parallel / ZeRO / FSDP**: how you actually run SGD at scale — sharding gradients, params, opt-state.
> - **Implicit regularization** of SGD noise is partly *why* deep nets generalize despite overparametrization. Active theory area (Smith et al., NTK, etc.).
> - Modern recipes: **AdamW + cosine schedule + warmup + grad clipping + LR/$\beta_2$ tuning per scale**. Pure SGD is mostly used for vision (ResNet) and is rare for LLMs.

---

## 7.2 Constrained Optimization & Lagrange Multipliers

**Constrained problem**:
$$\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{s.t.}\quad g_i(\boldsymbol{x}) \leq 0, \quad i=1,\ldots,m$$

### From indicator → Lagrangian

- **Indicator approach** (infeasible to optimize): $J(\boldsymbol{x}) = f(\boldsymbol{x}) + \sum_i \mathbf{1}(g_i(\boldsymbol{x}))$, where $\mathbf{1}(z)=0$ if $z\leq 0$ else $\infty$. Hard penalty, non-differentiable.
- **Lagrangian**: replace step with linear function.
$$\mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda}) = f(\boldsymbol{x}) + \sum_i \lambda_i g_i(\boldsymbol{x}) = f(\boldsymbol{x}) + \boldsymbol{\lambda}^\top \boldsymbol{g}(\boldsymbol{x}), \quad \boldsymbol{\lambda} \geq \boldsymbol{0}.$$
- $\lambda_i \geq 0$ for inequalities; **unconstrained** $\lambda_j$ for equalities $h_j(x)=0$ (or model an equality as two inequalities).

### Primal vs Dual

- **Primal**: $\min_x f(x)$ s.t. $g_i(x) \leq 0$.
- **Lagrangian dual**: $\max_{\lambda \geq 0} \mathfrak{D}(\lambda)$ where $\mathfrak{D}(\boldsymbol{\lambda}) = \min_{\boldsymbol{x}} \mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda})$.
- Get dual by setting $\nabla_x \mathfrak{L} = 0$, solving for $x^*(\lambda)$, plugging back.

### Minimax inequality & weak duality

- **Minimax inequality**: $\max_y \min_x \varphi(x,y) \leq \min_x \max_y \varphi(x,y)$.
- Since $J(\boldsymbol{x}) = \max_{\lambda \geq 0} \mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ (the max over $\lambda$ forces $g_i \leq 0$, else $+\infty$):
$$\min_x \max_{\lambda \geq 0} \mathfrak{L}(x,\lambda) \geq \max_{\lambda \geq 0} \min_x \mathfrak{L}(x,\lambda) \quad \Leftrightarrow \quad p^* \geq d^*.$$
- **Weak duality** always holds. **Strong duality** ($p^* = d^*$) holds for convex problems with Slater's condition.

### Why the dual is nice
- $\min_x \mathfrak{L}(x,\lambda)$ is **unconstrained** in $x$.
- $\mathfrak{L}$ is **affine in $\lambda$** ⇒ $\mathfrak{D}(\lambda)$ is **concave** (pointwise min of affine functions) **even if $f, g_i$ are nonconvex**.
- Outer max is **concave maximization** → tractable.

### KKT conditions (book introduces implicitly via setting gradient to zero)

Although MML doesn't enumerate them by name, the standard **KKT optimality conditions** for $\min f$ s.t. $g_i \leq 0$, $h_j = 0$ at $x^*, \lambda^*, \nu^*$:

1. **Stationarity**: $\nabla f(x^*) + \sum_i \lambda_i^* \nabla g_i(x^*) + \sum_j \nu_j^* \nabla h_j(x^*) = 0$.
2. **Primal feasibility**: $g_i(x^*) \leq 0$, $h_j(x^*)=0$.
3. **Dual feasibility**: $\lambda_i^* \geq 0$.
4. **Complementary slackness**: $\lambda_i^* g_i(x^*) = 0$ — either constraint active ($g_i=0$) or multiplier zero.

> **LLM / modern ML lens — this is the section that punches above its weight**:
>
> - **Regularization = constrained opt**: $\min L + \beta \|w\|_2^2$ is the Lagrangian form of $\min L$ s.t. $\|w\|_2^2 \leq C$. The multiplier $\beta$ ↔ constraint radius $C$ via duality. Same trick: $\ell_1$ (LASSO), trust-region.
> - **RLHF / PPO**: KL-constrained policy improvement.
>   - **Primal**: $\max_\pi \mathbb{E}_\pi[r(x,y)]$ s.t. $\mathrm{KL}(\pi \| \pi_{\mathrm{ref}}) \leq \delta$.
>   - **Lagrangian**: $\max_\pi \mathbb{E}_\pi[r] - \beta \,\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})$. The KL coefficient $\beta$ IS a Lagrange multiplier.
>   - PPO's "adaptive KL" updates $\beta$ as if doing dual ascent.
> - **DPO derivation**: starts from KL-constrained reward max, closed-form optimal policy is $\pi^*(y|x) \propto \pi_{\mathrm{ref}}(y|x)\exp(r(x,y)/\beta)$, then re-parametrize reward in terms of policy. The KL constraint is the entire reason DPO works.
> - **Constrained decoding**: lookahead/MCMC/penalty methods for "decode satisfying X" all view soft-constraints as Lagrangians.
> - **Conditional computation / MoE balance losses**: load-balancing constraints get folded in as Lagrangian penalties (auxiliary losses with tuned weights).
> - **Safety / RL with budgets**: CPO, Lagrangian-PPO maintain explicit dual variables for cost constraints.
> - **Trust region methods (TRPO)**: literal KKT formulation with KL ball constraint.
> - **Differential privacy / DP-SGD**: clip-and-noise is implicitly a per-sample norm constraint.

---

## 7.3 Convex Optimization

**Convex optimization problem**:
$$\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{s.t.}\quad g_i(\boldsymbol{x}) \leq 0,\ h_j(\boldsymbol{x}) = 0$$
where $f$ and $g_i$ are convex functions and $\{x : h_j(x)=0\}$ are convex sets.

- **Strong duality** holds → dual = primal optimal.
- **All local minima are global**.

### Definitions

- **Convex set** $\mathcal{C}$: $\forall x,y \in \mathcal{C}, \theta \in [0,1]: \theta x + (1-\theta) y \in \mathcal{C}$. Line segment stays in.
- **Convex function**: $f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y)$ — chord above the function. (Jensen's inequality, generalized.)
- **Concave** = negative of convex.
- **Epigraph** = filled-in region above graph; convex iff $f$ convex.

### Equivalent characterizations (for differentiable $f$)

- **First-order**: $f(y) \geq f(x) + \nabla f(x)^\top (y - x)$ — tangent line below function.
- **Second-order**: $\nabla^2 f(x) \succeq 0$ (PSD Hessian everywhere in domain).

### Convexity-preserving operations (closure properties)

- Nonneg scaling: $\alpha f$, $\alpha \geq 0$.
- Sum: $f_1 + f_2$.
- Nonneg weighted sum: $\sum \alpha_i f_i$, $\alpha_i \geq 0$.
- Max: $\max(f_1, f_2)$.
- Affine composition: $f(Ax+b)$ convex if $f$ convex.
- Intersection of convex sets — convex.
- Union — **not** convex in general.

### Jensen's inequality

In general form for convex $f$ and random $X$:
$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$
- Direct generalization of the chord property to expectations.
- Reverse direction for concave $f$.

> **ML uses everywhere**: ELBO derivation in VAEs/variational inference (Jensen on $\log$), EM, variance bounds, F-divergence inequalities, information-theoretic bounds, generalization bounds.

### Examples of convex functions in ML

- Negative entropy $f(x) = x\log x$ (book Example 7.3).
- Squared error (in parameters of linear model).
- Log-sum-exp / softmax denominator: $\mathrm{LSE}(z) = \log\sum_i e^{z_i}$ convex in $z$.
- Logistic/cross-entropy loss (in linear params).
- $\ell_p$ norms for $p \geq 1$.
- Indicator of convex set.

> **LLM lens**:
> - NN loss landscape is **non-convex**. But the **per-example loss given the logits** (softmax + cross-entropy) is convex in logits — that's why backprop has well-conditioned local objectives even when the global landscape is messy.
> - Recent theory (NTK, mean-field, lazy/feature learning) tries to translate non-convex deep nets into effectively convex / kernel-like regimes.
> - **Loss landscape geometry**: flat vs sharp minima discussion, mode connectivity, linear-mode-connectivity along training trajectories — all "convex intuitions transferred to non-convex world."
> - **Convex surrogates**: cross-entropy is convex surrogate for 0-1 loss; hinge is convex surrogate for SVM 0-1.

---

### 7.3.1 Linear Programming (LP)

$$\min_x c^\top x \quad \text{s.t. } Ax \leq b$$

- $d$ variables, $m$ constraints.
- **Lagrangian**: $\mathfrak{L}(x,\lambda) = c^\top x + \lambda^\top(Ax-b) = (c+A^\top \lambda)^\top x - \lambda^\top b$.
- $\nabla_x \mathfrak{L} = c + A^\top \lambda = 0$ (else $\inf_x = -\infty$).
- **Dual**:
$$\max_\lambda -b^\top \lambda \quad \text{s.t. } c + A^\top \lambda = 0,\ \lambda \geq 0$$
- Dual of LP is LP. If $m \ll d$, solve dual; else solve primal.

> **LLM relevance**: marginal. LPs show up in:
> - Optimal transport (Sinkhorn is approx LP w/ entropic regularization).
> - Scheduling/routing in serving / KV-cache placement (industrial, not training).
> - Some constrained decoding formulations.
> - Linear assignment for matching (Hungarian / DETR).
> - Verifier-guided generation with linear constraints.

### 7.3.2 Quadratic Programming (QP)

$$\min_x \tfrac{1}{2} x^\top Q x + c^\top x \quad \text{s.t. } Ax \leq b, \quad Q \succ 0$$

- Convex iff $Q \succeq 0$.
- **Lagrangian**: $\mathfrak{L}(x,\lambda) = \tfrac{1}{2}x^\top Q x + (c + A^\top \lambda)^\top x - \lambda^\top b$.
- $\nabla_x = 0 \Rightarrow Qx + (c+A^\top\lambda) = 0 \Rightarrow x^* = -Q^{-1}(c+A^\top \lambda)$.
- **Dual**:
$$\max_\lambda -\tfrac{1}{2}(c+A^\top\lambda)^\top Q^{-1}(c+A^\top\lambda) - \lambda^\top b \quad \text{s.t. } \lambda \geq 0$$

> **LLM relevance**: low-direct. Historically central for **SVMs** (Chapter 12 application). Today:
> - **Trust region / second-order methods** (K-FAC, Shampoo's inner problem) reduce to QP-like subproblems.
> - Quadratic approximations of loss (Hessian-based, Newton, Gauss–Newton) underpin many advanced optimizers.
> - **Sequential Quadratic Programming (SQP)** for constrained NLPs — niche in ML.

---

### 7.3.3 Legendre–Fenchel Transform / Convex Conjugate

> **This is the most theoretically important piece of the chapter for modern LLM research.**

**Geometric intuition**:
- A convex function ≡ its epigraph (convex set) ≡ its supporting hyperplanes (tangent lines).
- A function of $x$ becomes a function of its **gradient/slope** $s$.

**Definition** (general, no differentiability/convexity required):
$$f^*(\boldsymbol{s}) = \sup_{\boldsymbol{x}\in\mathbb{R}^D} \left( \langle \boldsymbol{s}, \boldsymbol{x}\rangle - f(\boldsymbol{x})\right)$$

For convex differentiable $f$, the sup is attained uniquely at $x$ with $s = \nabla f(x)$ — bijection between $x$ and slope $s$.

**1D derivation sketch**: For $y = sx + c$ tangent to $f$ at $x_0$, $-c = sx_0 - f(x_0) = f^*(s)$. Tangent slope $s$ indexes the conjugate.

**Key properties**:
- $f^*$ is **always convex** (sup of affine functions of $s$).
- Fenchel–Moreau: for convex lsc $f$, $f^{**} = f$.
- **Young–Fenchel inequality**: $f(x) + f^*(s) \geq \langle s, x \rangle$, equality iff $s \in \partial f(x)$.

**Example 7.7 — quadratic**: $f(y) = \tfrac{\lambda}{2} y^\top K^{-1} y$ (PD $K$).
- Set $\nabla_y[\langle y,\alpha\rangle - f(y)] = 0 \Rightarrow y = \tfrac{1}{\lambda} K\alpha$.
- $f^*(\alpha) = \tfrac{1}{2\lambda} \alpha^\top K \alpha$.

**Example 7.8 — separable sum** $\mathcal{L}(t) = \sum_i \ell_i(t_i)$:
- $\mathcal{L}^*(z) = \sum_i \ell_i^*(z_i)$ — conjugate separates over examples. Huge in ML where loss = sum over training points.

**Example 7.9 — Fenchel duality with linear coupling** $\min_x f(Ax) + g(x)$:
$$\min_x f(Ax) + g(x) = \max_u\ -f^*(u) - g^*(-A^\top u)$$
- Mirror of Lagrangian duality but uses conjugates instead.
- For general inner products, $A^\top$ becomes adjoint $A^*$.

**Common conjugate pairs (memorize)**:

| $f(x)$ | $f^*(s)$ |
|---|---|
| $\tfrac{1}{2}\|x\|_2^2$ | $\tfrac{1}{2}\|s\|_2^2$ (self-dual) |
| $\tfrac{1}{2} x^\top Q x$ ($Q \succ 0$) | $\tfrac{1}{2} s^\top Q^{-1} s$ |
| $e^x$ | $s\log s - s$ (entropy-like), $s \geq 0$ |
| $x\log x - x$ (neg entropy) | $e^s$ |
| $\sum_i x_i \log x_i$ on simplex (neg entropy) | $\log \sum_i e^{s_i}$ (LSE) |
| $\|x\|$ (any norm) | $\mathbb{1}\{\|s\|_* \leq 1\}$ (indicator of dual norm ball) |
| Hinge $\max(0, 1-x)$ | $-s$ if $s\in[-1,0]$, else $+\infty$ |
| Indicator $\mathbb{1}_C(x)$ | Support function $\sigma_C(s) = \sup_{x\in C} \langle s,x\rangle$ |

> **The negative-entropy ↔ log-sum-exp pair is the most important duality in modern ML.** It is the reason **softmax shows up everywhere you have entropy-regularized optimization over the simplex**.

> **LLM / ML applications of convex conjugates**:
>
> - **Entropy-regularized RL / max-ent RL**: $\max_\pi \mathbb{E}_\pi[r] + \tau H(\pi)$ on the simplex → optimal policy $\pi \propto \exp(r/\tau)$. This is precisely the LSE-↔-negentropy conjugacy. Soft Actor-Critic, soft Q-learning, max-ent IRL all live here.
> - **RLHF / DPO**: KL-constrained reward maximization solution $\pi^*(y|x) \propto \pi_{\mathrm{ref}}(y|x)\exp(r(x,y)/\beta)$ is the Fenchel dual answer with $-\log \pi_{\mathrm{ref}}$ playing the role of the base. The closed form is *the* conjugate identity.
> - **IPO / KTO / SimPO / generalized DPO objectives**: each can be derived by choosing a different convex regularizer and computing its conjugate / Fenchel dual.
> - **Mirror descent**: choose a Bregman divergence $D_\psi$; the update is $x_{t+1} = \nabla \psi^*(\nabla \psi(x_t) - \eta g_t)$. Conjugate is *the* primitive. Exponentiated gradient (multiplicative weights) is mirror descent with neg-entropy mirror map → conjugate is LSE.
> - **Exponential family duality**: log-partition function $A(\theta) = \log \int e^{\langle \theta, T(x)\rangle} d\nu(x)$ is convex; its conjugate is **negative entropy** of the distribution, mapping natural params ↔ mean params. The entire EM / variational inference machinery uses this.
> - **F-divergence variational bounds**: $D_f(P\|Q) = \sup_T \mathbb{E}_P[T] - \mathbb{E}_Q[f^*(T)]$ (Donsker–Varadhan / variational repr). Backbone of f-GAN, MINE, noise contrastive estimation, contrastive learning bounds.
> - **GANs**: Wasserstein-GAN duality (Kantorovich–Rubinstein) is a Fenchel duality over the 1-Lipschitz unit ball.
> - **Sinkhorn / entropic OT**: regularize OT with entropy → dual is smooth and solvable by Sinkhorn iterations. Used in deep representation learning, Sinkhorn attention, OT-based matching.
> - **Smoothing nondifferentiable losses**: hinge loss is non-smooth; conjugate-smoothed via Moreau envelope = smoothed hinge — easier to optimize with first-order methods (Exercise 7.11 in this chapter).
> - **Proximal operators**: $\mathrm{prox}_{\eta f}(v) = \arg\min_x \tfrac{1}{2\eta}\|x-v\|^2 + f(x)$. Tight conjugate connection via Moreau decomposition $v = \mathrm{prox}_f(v) + \mathrm{prox}_{f^*}(v)$. Underlies FISTA/ISTA, ADMM, primal-dual splitting — used in compressed sensing, deconvolution, model pruning.
> - **Variational free energy / Helmholtz duality**: in stat mech, energy ↔ entropy via Legendre. Same math reappears in variational inference (ELBO = energy − entropy).

---

## 7.4 Further reading (book pointers + my annotations)

- **Acceleration**: Nesterov (2018). Beyond momentum: Polyak heavy-ball, restart schemes.
- **Conjugate gradient**: Shewchuk's "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain." Niche in deep learning but standard for Gaussian process posteriors.
- **Second-order methods**: Newton uses full Hessian — too expensive for LLMs. **L-BFGS** approximates inverse Hessian via past gradients; good for small/medium problems, **rarely used at LLM scale** (too memory-hungry, doesn't play nice with mini-batches).
- **Mirror descent / natural gradient**: see above. Natural gradient ≈ Newton with Fisher info, motivates K-FAC, Shampoo, SOAP. **Muon (Jordan et al. 2024)** is closest current frontier — orthogonalize momentum updates for hidden weights, big gains.
- **Subgradient methods** (Shor 1985): for nondifferentiable convex (e.g. $\ell_1$, hinge). Modern alternative: proximal/conjugate smoothing.
- **Bubeck (2015)** "Convex Optimization: Algorithms and Complexity" — best modern intro.
- **Boyd & Vandenberghe** — canonical, free.
- **Bottou et al. (2018)** survey of SGD-era methods for ML.

---

## Exercises (book) — quick translation

- **7.1** Univariate stationary points: take $f'$, find roots, check $f''$ sign.
- **7.2** Mini-batch SGD with $B=1$: $\theta_{i+1} = \theta_i - \gamma_i \nabla L_n(\theta_i)^\top$ with $n$ randomly sampled.
- **7.3 a)** intersection convex ✓ **b)** union ✗ **c)** set difference ✗.
- **7.4 a)** sum convex ✓ **b)** diff ✗ **c)** product ✗ (in general) **d)** max ✓.
- **7.8** $\min \tfrac{1}{2}\|w\|^2$ s.t. $w^\top x \geq 1$: $\mathfrak{L} = \tfrac{1}{2}\|w\|^2 - \lambda(w^\top x - 1)$. KKT $\Rightarrow w^* = \lambda x$, dual $= \lambda - \tfrac{1}{2}\lambda^2 \|x\|^2$, $\max_\lambda \Rightarrow \lambda^* = 1/\|x\|^2$. Classic SVM-style derivation.
- **7.9** Negative entropy $f(x) = \sum_d x_d \log x_d$ conjugate: $\nabla = 0 \Rightarrow x_d = e^{s_d-1}$, plug back: $f^*(s) = \sum_d e^{s_d - 1}$. (For simplex-constrained negentropy you get LSE — different.)
- **7.10** $f(x) = \tfrac{1}{2}x^\top A x + b^\top x + c$, $A \succ 0$: $f^*(s) = \tfrac{1}{2}(s-b)^\top A^{-1}(s-b) - c$.
- **7.11** Hinge $L(\alpha) = \max(0, 1-\alpha)$: $L^*(\beta) = \beta$ if $\beta \in [-1, 0]$, else $\infty$. Adding $\tfrac{\gamma}{2}\beta^2$ smooths in dual → infimal convolution back gives Moreau-smoothed hinge in primal.

---

## Concept map for revisiting

```
                 ┌────────────────────────────────┐
                 │  CONTINUOUS OPTIMIZATION       │
                 └────────────────────────────────┘
                              │
              ┌───────────────┼────────────────────┐
              │               │                    │
       Unconstrained    Constrained          Convex (special)
              │               │                    │
        Gradient Descent  Lagrange Multipliers   Strong duality
              │               │                    │
       ┌──────┼──────┐    KKT + Duality      ┌─────┼─────┐
     Step  Momentum  SGD  (primal/dual)      LP   QP   Legendre-Fenchel
     size                  weak ≤ strong              (convex conjugate)
                                                          │
                                                Bridges to: DPO, max-ent RL,
                                                exp family, mirror descent,
                                                F-divergence, OT, prox/ADMM
```

---

## Practical recipe for an LLM researcher reading this chapter

1. **Skim** GD/SGD math — you know it. Pay attention to the **condition number / preconditioning** remark because it motivates Adam/Shampoo.
2. **Internalize** the Lagrangian view of constraints. Every time you see "add $\beta \cdot$ penalty," ask "what's the primal constraint $\beta$ is dualizing?" — clarifies what $\beta$ does and how to tune it.
3. **Master** convex conjugate definition + 3-4 standard pairs (quadratic, LSE/negentropy, norm/indicator, hinge). When deriving new RLHF-style losses or variational bounds, you'll reach for this constantly.
4. **Skip-but-know-it-exists**: LP/QP explicit duality. Useful for SVMs and old-school constrained ML; rarely needed for LLM work directly.
5. **Augment** your mental model with what's NOT in the book: AdamW, schedules, gradient clipping, distributed opt, $\mu$P scaling, second-order/Shampoo/Muon, mixed precision dynamics.
