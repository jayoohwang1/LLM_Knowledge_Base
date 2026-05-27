# Chapter 8: Optimization

Honest take: this is *the* chapter for ML practitioners. Optimization is what we actually do all day at frontier labs — every training run, every fine-tune, every RLHF step is an optimization problem. Murphy's treatment is solid as background but a bit dated for the LLM era: he barely mentions Adam variants used in modern training (AdamW, Lion, Shampoo/Distributed Shampoo, Muon) and skips the distributed/large-scale aspects entirely. The **most important sections for modern LLM work**: 8.2.4 (momentum), 8.4 (SGD + minibatch + schedules), 8.4.6 (adaptive/Adam), and conceptually 8.7 (EM/MM as the basis of variational methods, RLHF surrogate objectives). The rest (Newton, BFGS, LP/QP, KKT, proximal) is classical optimization — useful background but you'll rarely write a Newton solver for an LLM.

---

## 8.1 Introduction

- **Setup**: minimize loss $\mathcal{L}: \Theta \to \mathbb{R}$, $\theta^* \in \arg\min_\theta \mathcal{L}(\theta)$. Continuous, $\Theta \subseteq \mathbb{R}^D$.
- **Score/reward functions**: equivalent to minimizing $-R(\theta)$. **Important for LLMs**: RLHF maximizes a reward model score, but is implemented as gradient descent on the negative.
- **Solver**: any algorithm that finds an optimum.

### 8.1.1 Local vs global

- **Global optimization**: NP-hard in general (Neumaier). For non-convex problems (which is literally every deep net), we settle for **local minima**.
  - **LLM reality**: nobody cares about global optima for neural nets; SGD finds "good enough" local minima that generalize. The empirical observation that "most local minima of deep nets are about equally good" is a major reason deep learning works.
- **Strict** vs **flat** local minimum. Modern intuition: **flat minima generalize better** (Hochreiter & Schmidhuber, Keskar et al.) — this motivates SWA and is part of why large-batch training sometimes hurts.
- **Globally convergent** = converges to some stationary point from any init (not the same as finding a global optimum — annoying terminology).

### 8.1.1.1 Optimality conditions

- **Necessary**: $g^* = \mathbf{0}$ and $H^* \succeq 0$ (stationary + PSD Hessian).
- **Sufficient**: $g^* = 0$ and $H^* \succ 0$.
- **Saddle points**: $g=0$ but Hessian has mixed eigenvalues. **Critical for deep learning**: in high-D non-convex landscapes, saddles vastly outnumber local minima (Dauphin et al. 2014). Most of the difficulty of training big nets is escaping saddles, not avoiding minima.

### 8.1.2 Constrained vs unconstrained

- **Feasible set** $\mathcal{C} = \{\theta: g_j(\theta)\le 0, h_k(\theta)=0\}$.
- Constrained problems handled by **Lagrangian** / penalty methods (Sec 8.5).
- **LLM relevance**: mostly unconstrained, but constraints show up in:
  - **Trust region methods in RLHF** (TRPO/PPO use KL constraints).
  - **Constrained decoding** (e.g., logit constraints during inference).
  - **Safety constraints** in alignment.

### 8.1.3 Convex vs nonconvex

- **Convex set**: $\lambda x + (1-\lambda)x' \in \mathcal{S}$.
- **Convex function**: $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$. Equivalently epigraph is convex.
- **Strongly convex** with parameter $m$: $(\nabla f(x)-\nabla f(y))^\top(x-y) \ge m\|x-y\|^2$, i.e., $\nabla^2 f \succeq m I$.
- **Why we care little in deep learning**: NN losses are emphatically non-convex. The classical convex theory is great for logistic regression / linear models / SVM / lasso but doesn't tell you why SGD on a transformer works.
- **Why we still care a bit**: many sub-problems are convex (LP/QP/SOCP solvers used inside RL algos, OT for distribution matching). Modern theory (NTK, lazy training) studies when wide nets behave "almost convexly".
- **Theorem**: $f$ convex iff $H \succeq 0$ everywhere.
- 1D convex examples: $x^2$, $e^{ax}$, $-\log x$, $|x|^a$ for $a\ge 1$, $x\log x$ for $x>0$.

### 8.1.4 Smooth vs nonsmooth

- **Lipschitz constant** $L$: $|f(x_1)-f(x_2)| \le L|x_1 - x_2|$.
- **Composite objective** $\mathcal{L} = \mathcal{L}_s + \mathcal{L}_r$ — smooth + rough. Standard ML pattern: differentiable loss + non-smooth regularizer (e.g., $\ell_1$).
- **Subgradient** $g \in \partial f(x)$ if $f(z) \ge f(x) + g^\top(z-x)$. For $f(x) = |x|$, $\partial f(0) = [-1,1]$.
- **Modern relevance**: ReLU nets are technically nonsmooth at 0; autograd just picks a subgradient. Same with $|x|$ for L1 reg. Gradient clipping is another form of non-smoothness handling.

---

## 8.2 First-order methods

**The bread and butter of modern ML.** Every LLM is trained with a first-order method. Update form:

$$\theta_{t+1} = \theta_t + \eta_t d_t$$

with **step size / learning rate** $\eta_t$ and **descent direction** $d_t$.

### 8.2.1 Descent direction

- $d$ is descent if $\exists \eta_{\max}>0: \mathcal{L}(\theta + \eta d) < \mathcal{L}(\theta)$ for all $0 < \eta < \eta_{\max}$.
- Any direction with $d^\top g_t < 0$ (angle $<90°$ with $-g$) works.
- **Steepest descent**: $d = -g$.

### 8.2.2 Step size / learning rate

- Too large: oscillation / divergence; too small: glacial convergence.

#### Constant step size
- For quadratic $\mathcal{L} = \tfrac{1}{2}\theta^\top A\theta + b^\top \theta + c$: converges iff $\eta < 2/\lambda_{\max}(A)$.
- General smooth: $\eta < 2/L$ where $L$ is gradient Lipschitz constant.

#### Line search (8.2.2.2)
- **Exact line search**: $\eta_t = \arg\min_\eta \mathcal{L}(\theta_t + \eta d_t)$. For quadratics: $\eta = -d^\top(A\theta + b)/(d^\top A d)$.
- **Armijo backtracking**: shrink $\eta$ by factor $c<1$ until $\mathcal{L}(\theta + \eta d) \le \mathcal{L}(\theta) + c\eta d^\top \nabla \mathcal{L}$. Constant $c \approx 10^{-4}$.
- **Rarely used in deep learning**: noisy gradients break line search. People mostly use fixed schedules + warmup + cosine. Some recent work (Schedule-Free, AdaLOMO) tries to bring back adaptivity.

### 8.2.3 Convergence rates

- **Linear rate**: $|\mathcal{L}(\theta_{t+1}) - \mathcal{L}_*| \le \mu |\mathcal{L}(\theta_t) - \mathcal{L}_*|$.
- Quadratic + exact line search: $\mu = \left(\tfrac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}}\right)^2 = \left(\tfrac{\kappa-1}{\kappa+1}\right)^2$, $\kappa = $ condition number.
- **High $\kappa$ = slow zigzag**. Why preconditioning matters; why batch norm / layer norm / RMSNorm help (they implicitly precondition by keeping activations / gradients well-scaled).

### 8.2.4 Momentum methods *(important for deep learning)*

#### Heavy ball / momentum
$$m_t = \beta m_{t-1} + g_{t-1}, \quad \theta_t = \theta_{t-1} - \eta_t m_t$$

- Typical $\beta = 0.9$. EWMA of past gradients: $m_t = \sum_{\tau} \beta^\tau g_{t-\tau-1}$.
- Constant-gradient limit: scales gradient by $1/(1-\beta) = 10$ for $\beta=0.9$.
- **Why it matters**: massively reduces oscillation in narrow valleys; effectively averages noise from SGD — like a free large batch.

#### Nesterov accelerated gradient (NAG)
$$\tilde\theta_{t+1} = \theta_t + \beta(\theta_t - \theta_{t-1}), \quad \theta_{t+1} = \tilde\theta_{t+1} - \eta_t \nabla \mathcal{L}(\tilde\theta_{t+1})$$

- "Look-ahead" gradient. Provably faster than steepest descent for convex Lipschitz problems. In practice for deep nets: roughly tied with vanilla momentum, sometimes unstable.

**Modern LLM angle**: virtually nobody uses pure momentum SGD for LLMs anymore. AdamW dominates. But conceptually momentum is *inside* Adam (the $m_t$ term).

---

## 8.3 Second-order methods

Pretty much ignored in LLM training because storing/inverting a Hessian of $\sim 10^{11}$ parameters is impossible. **Still worth understanding** because (a) it explains why adaptive methods help, (b) recent methods like Shampoo/Muon are *approximate second-order*.

### 8.3.1 Newton's method
$$\theta_{t+1} = \theta_t - \eta_t H_t^{-1} g_t$$

- Derived from minimizing the quadratic Taylor approx around $\theta_t$.
- For linear regression: converges in one step ($w^* = (X^\top X)^{-1} X^\top y$, the OLS estimate).
- **Pitfalls**: needs $H \succ 0$; can move toward a maximum if Hessian indefinite. Use line search or trust region.

### 8.3.2 Quasi-Newton (BFGS, L-BFGS)

- Build up Hessian approx $B_t$ from gradient differences:
  - $s_t = \theta_t - \theta_{t-1}$, $y_t = g_t - g_{t-1}$
  - Rank-2 update: $B_{t+1} = B_t + \tfrac{y_t y_t^\top}{y_t^\top s_t} - \tfrac{(B_t s_t)(B_t s_t)^\top}{s_t^\top B_t s_t}$
- **Wolfe conditions** ensure positive-definiteness.
- **L-BFGS**: store only $M \approx 5\text{-}20$ pairs $(s_t, y_t)$, approximate $H^{-1}g$ via inner products. $O(MD)$ memory.
- **Where L-BFGS is used in ML today**: convex problems (sklearn logistic regression default), small fine-tuning regimes, GP hyperparameter optimization, scientific ML. **Not** for big nets — stochasticity ruins curvature estimates.

### 8.3.3 Trust region methods

- Pick a region $\mathcal{R}_t$ (typically ball of radius $r$) and minimize a model inside it.
- **Tikhonov damping**: $\delta = -(H + \lambda I)^{-1} g$. As $\lambda \to 0$ reduces to Newton; large $\lambda$ flips negative eigenvalues, becomes gradient-descent-like.
- **LLM connection**: TRPO/PPO use a KL-divergence trust region between old/new policies. This is one of the most important conceptual links between classical optimization and modern RLHF.

---

## 8.4 Stochastic gradient descent *(THE workhorse)*

This section is the heart of modern ML. Every LLM is trained with SGD-with-Adam-flavored variants. **Spend time on 8.4.1–8.4.6**.

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L}(\theta_t, z_t), \quad z_t \sim q$$

Gradient estimate just needs to be unbiased.

### 8.4.1 Finite sum / minibatch

- ERM: $\mathcal{L}(\theta) = \tfrac{1}{N}\sum_n \ell_n(\theta)$.
- **Minibatch**: $g_t \approx \tfrac{1}{|\mathcal{B}_t|}\sum_{n\in\mathcal{B}_t}\nabla \ell_n$.
- **Why SGD beats full-batch GD in practice**: even though SGD has sublinear convergence rate, per-step cost is $|\mathcal{B}|/N$ smaller. Also, full-batch wastes effort early when params are nowhere near optimum.
- **LLM scale**: batch sizes are $\sim$millions of tokens, distributed over thousands of GPUs. Microbatching + gradient accumulation, ZeRO/FSDP shard params/grads/optimizer states.

### 8.4.3 Learning rate schedules *(extremely important)*

- **Robbins-Monro condition**: $\sum \eta_t = \infty$, $\sum \eta_t^2 < \infty$.
- Common schedules:
  - **Piecewise constant** / step decay (still used, e.g., reduce-on-plateau).
  - **Exponential** $\eta_t = \eta_0 e^{-\lambda t}$ — usually too aggressive.
  - **Polynomial** $\eta_t = \eta_0(\beta t + 1)^{-\alpha}$; $\alpha = 0.5$ gives **square-root**.
  - **Linear warmup + cosine decay** (Fig 8.19a) — **the standard for LLM pretraining**. Warmup prevents instability from large early gradients in poorly-conditioned regions of loss landscape. Cosine smoothly ramps to 0.
  - **Cyclical / one-cycle (Smith)**, **SGDR** (warm restarts) — useful for image classification, less common for LLMs.

#### LR finder
- Smith's heuristic: sweep LR over orders of magnitude on a few minibatches, pick where loss is decreasing fastest. Used by `fast.ai`.

**Practical LLM frontier-lab advice**: peak LR scales as $\sim 1/\sqrt{d_{\text{model}}}$ or via $\mu$P (maximal update parameterization, Yang & Hu). Warmup ~1-2% of total tokens. Cosine to ~10% of peak. WSD (Warmup-Stable-Decay) is gaining popularity because it lets you decide when to stop.

### 8.4.4 Iterate averaging

- **Polyak-Ruppert**: $\bar\theta_t = \tfrac{1}{t}\sum_i \theta_i$. Asymptotically optimal rate matching second-order methods.
- **SWA (Stochastic Weight Averaging)**: equal-weight avg of late-training iterates + modified LR schedule. **Exploits flat minima** for better generalization.
- **Modern relevance**: EMA of weights is used everywhere — diffusion model training (massively improves sample quality), GAN training, RL value networks, semi-supervised mean teacher.

### 8.4.5 Variance reduction (SVRG, SAGA)

- SVRG: $g_t = \nabla \mathcal{L}_t(\theta_t) - \nabla \mathcal{L}_t(\tilde\theta) + \nabla \mathcal{L}(\tilde\theta)$ using a snapshot $\tilde\theta$. Provably linear convergence on convex problems.
- SAGA: stores all $N$ per-example gradients.
- **Honest assessment**: largely irrelevant for deep learning. Batchnorm, data augmentation, dropout all violate the finite-sum assumption (different randomness each step). Works fine for convex linear models on static data.

### 8.4.6 Preconditioned SGD *(the foundation of Adam et al.)*
$$\theta_{t+1} = \theta_t - \eta_t M_t^{-1} g_t$$

In practice $M_t$ is diagonal (cheap).

#### AdaGrad
$$s_{t,d} = \sum_{i=1}^t g_{i,d}^2, \quad \theta_{t+1,d} = \theta_{t,d} - \eta_t \tfrac{1}{\sqrt{s_{t,d}+\epsilon}} g_{t,d}$$
- Larger LR for rarely-occurring features (originally motivated by sparse-feature problems like word counts).
- **Problem**: denominator monotonically grows → effective LR decays too fast.

#### RMSProp (Hinton)
$$s_{t+1,d} = \beta s_{t,d} + (1-\beta) g_{t,d}^2$$
- EWMA instead of cumulative sum. $\beta \approx 0.9$.

#### AdaDelta
- Additional EWMA of squared updates; "units cancel" so $\eta_t = 1$ in principle. Rarely used now.

#### **Adam** — the king of LLM training
```
m_t = β₁ m_{t-1} + (1-β₁) g_t           # momentum (EWMA of gradients)
s_t = β₂ s_{t-1} + (1-β₂) g_t²           # EWMA of squared gradients
m̂_t = m_t / (1 - β₁ᵗ)                   # bias correction
ŝ_t = s_t / (1 - β₂ᵗ)
θ_t = θ_{t-1} - η_t m̂_t / (√ŝ_t + ε)
```
- Defaults: $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-6}$ (LLMs often use $10^{-8}$ or $\beta_2=0.95$ for stability with bf16).
- **Why Adam dominates LLM training**:
  - Robust to gradient scale variation across layers (early embedding vs late LM head can differ by orders of magnitude).
  - Bias correction prevents huge first-step jumps.
  - Combines momentum (smooth out noise) + adaptive scaling.
- **Variants you should know**:
  - **AdamW** (Loshchilov & Hutter): decouples weight decay from gradient update. **This is what's actually used for transformers, not vanilla Adam.** The distinction matters.
  - **AMSGrad, AdaBelief, Yogi, AdaBound, Lion, ADOPT**: various fixes for non-convergence issues. Lion (Google, evolved search) uses signed momentum — competitive with AdamW with half the optimizer state memory.
  - Murphy mentions ADOPT as a recent provable-convergence variant.

#### 8.4.6.4 Issues with adaptive learning rates
- Adam fails to converge even on simple convex problems (Reddi et al.) — usually a non-issue in practice because we early-stop anyway.

#### 8.4.6.5 Non-diagonal preconditioning
- **Full-matrix AdaGrad**: $M_t = [(G_t G_t^\top)^{1/2} + \epsilon I]^{-1}$. $O(D^2)$ — infeasible for LLMs.
- **Shampoo** (Gupta et al.): block-diagonal with Kronecker factorization, one factor per layer dimension. **Recently scaled to frontier LLMs** (Anil et al., DistributedShampoo).
- Not in the book but worth knowing:
  - **Muon** (Jordan et al. 2024): orthogonalize update via Newton-Schulz iteration on momentum matrix. Used to train modded-NanoGPT records, increasing adoption.
  - **K-FAC**: Kronecker-factored approximation to natural gradient — older but related idea.

**Frontier-lab honest take**: AdamW is the default; Shampoo/Muon are emerging contenders that give modest but real speedups (~1.3-2x) at scale and are being adopted. The optimizer matters less than the data and architecture for most decisions, but at billion-dollar training scales, even 10% speedups are huge.

---

## 8.5 Constrained optimization

Less central to LLM training but conceptually critical for RLHF, alignment, and theory.

### 8.5.1 Lagrange multipliers (equality constraints)
- At optimum on constraint surface, $\nabla \mathcal{L}$ is parallel to $\nabla h$:
$$\nabla \mathcal{L}(\theta^*) = \lambda^* \nabla h(\theta^*)$$
- **Lagrangian**: $L(\theta, \lambda) = \mathcal{L}(\theta) + \lambda h(\theta)$. Stationary point gives $D+m$ equations.

### 8.5.2 KKT conditions (inequality constraints)
- Generalized Lagrangian: $L(\theta, \mu, \lambda) = \mathcal{L}(\theta) + \sum_i \mu_i g_i(\theta) + \sum_j \lambda_j h_j(\theta)$.
- **KKT conditions**:
  1. **Primal feasibility**: $g \le 0, h = 0$.
  2. **Stationarity**: $\nabla \mathcal{L} + \sum \mu_i \nabla g_i + \sum \lambda_j \nabla h_j = 0$.
  3. **Dual feasibility**: $\mu \ge 0$.
  4. **Complementary slackness**: $\mu_i g_i = 0$ — either constraint inactive ($g_i<0$, $\mu_i = 0$) or active ($g_i=0$).
- For convex problems: KKT is sufficient and necessary.
- **Why LLM people should know this**:
  - **Lagrangian duality** underlies many alignment formulations (constrained RLHF, safe RL with cost constraints).
  - **DPO derivation**: uses KKT on a KL-constrained reward maximization to derive its closed-form loss.
  - **Soft constraints in finetuning**: KL penalty against base model is literally a Lagrangian relaxation.

### 8.5.3 Linear programming
- $\min c^\top \theta$ s.t. $A\theta \le b, \theta \ge 0$.
- **Simplex**: walks vertices of feasible polytope. Worst-case exponential, usually fast.
- **Interior point**: poly-time but often slower in practice.
- **ML uses**: robust linear regression, state estimation in graphical models, optimal transport (Sinkhorn is the relevant fast variant for ML/diffusion).

### 8.5.4 Quadratic programming
- $\min \tfrac{1}{2}\theta^\top H\theta + c^\top \theta$ s.t. $A\theta \le b$.
- **ML uses**: lasso (reformulated as QP), SVM dual, MPC in robotics/RL.

### 8.5.5 Mixed integer linear programming (MILP)
- ILP with some real-valued vars. NP-hard.
- **ML uses**: formal verification of NNs (e.g., adversarial robustness certification), neural architecture search.

---

## 8.6 Proximal gradient method

Composite objective $\mathcal{L} = \mathcal{L}_s + \mathcal{L}_r$ (smooth + rough).

$$\theta_{t+1} = \text{prox}_{\eta_t \mathcal{L}_r}(\theta_t - \eta_t \nabla \mathcal{L}_s(\theta_t))$$

where $\text{prox}_{\eta f}(\theta) = \arg\min_z \left[f(z) + \tfrac{1}{2\eta}\|z-\theta\|^2\right]$ — take a gradient step on smooth part, then "project" via prox operator.

### 8.6.1 Projected gradient descent
- For indicator $\mathcal{L}_r = I_\mathcal{C}$, prox = projection onto $\mathcal{C}$.
- **Box constraints**: clip elementwise. **Non-negativity**: $\max(\theta, 0)$.

### 8.6.2 Prox operator for $\ell_1$ → **soft thresholding**
$$\text{prox}_{\lambda |\cdot|}(\theta) = \text{sign}(\theta)\,(|\theta| - \lambda)_+$$
- Foundation of lasso, ISTA/FISTA algorithms.
- **Modern relevance**: sparse fine-tuning, structured pruning of LLMs (SparseGPT, Wanda use related ideas), MoE routing.

### 8.6.3 Prox operator for quantization
- $\mathcal{L}_r(\theta) = \min_{\theta_0 \in \mathcal{C}}\|\theta - \theta_0\|_1$ for quantized set $\mathcal{C}$.
- **Straight-through estimator** (STE): use $\nabla \mathcal{L}/\nabla \theta \approx \nabla \mathcal{L}/\nabla q(\theta)$. The trick behind **binary-connect**, ProxQuant, and modern QAT.
- **LLM relevance**: huge. QLoRA, GPTQ, AWQ, BitNet, 1.58-bit LLMs all use variants of STE / proximal quantization. Essential for deploying LLMs on edge / running 70B models on a single GPU.

### 8.6.4 Incremental / online proximal
- Online learning extension; Kalman-filtering view.

---

## 8.7 Bound optimization (MM / EM) *(important conceptually for variational methods)*

**MM = majorize-minimize (minimization) or minorize-maximize (maximization).** Build a surrogate $Q(\theta, \theta^t)$ that:
- Lower bounds $\ell$: $Q(\theta, \theta^t) \le \ell(\theta)$
- Tight at current iterate: $Q(\theta^t, \theta^t) = \ell(\theta^t)$

Then $\theta^{t+1} = \arg\max_\theta Q(\theta, \theta^t)$. Guarantees monotonic improvement:
$$\ell(\theta^{t+1}) \ge Q(\theta^{t+1}, \theta^t) \ge Q(\theta^t, \theta^t) = \ell(\theta^t)$$

**Why this matters for LLM research**:
- **Variational inference** (ELBO) is a direct application — VAEs, diffusion models, posterior approximation.
- **DPO** can be viewed through an MM lens (lower-bounding RLHF objective).
- **EM-style algorithms** still appear in self-training, pseudo-labeling, expert iteration in RL.

### 8.7.2 The EM algorithm

Goal: MLE/MAP for latent variable models, $\ell(\theta) = \sum_n \log \sum_{z_n} p(y_n, z_n | \theta)$.

#### ELBO derivation
Introduce $q_n(z_n)$, use Jensen:
$$\ell(\theta) \ge \sum_n \mathbb{E}_{q_n}[\log p(y_n, z_n|\theta)] + \mathbb{H}(q_n) \triangleq \text{Ł}(\theta, \{q_n\})$$

**This is the ELBO**. Optimizing it is the basis of variational inference.

#### E step
- Set $q_n^* = p(z_n|y_n, \theta^t)$ → makes the bound tight: $\text{Ł}(\theta, \{q_n^*\}) = \ell(\theta)$.
- $\text{Ł}(\theta, q_n) = -D_{KL}(q_n \| p(z_n|y_n, \theta)) + \log p(y_n|\theta)$ — explains why setting $q = p(z|y,\theta)$ is optimal.

#### M step
- Maximize expected complete-data log likelihood:
$$\theta^{t+1} = \arg\max_\theta \sum_n \mathbb{E}_{q_n^t}[\log p(y_n, z_n | \theta)]$$
- For **exponential families**: closed-form via matching expected sufficient statistics.

#### Variational EM
- If true posterior intractable, use approx $q(z_n | y_n; \theta^t)$ → non-tight bound. **This is amortized inference / VAEs.**

### 8.7.3 EM for GMM

E-step (compute responsibilities):
$$r_{nk}^{(t)} = \frac{\pi_k^{(t)} \mathcal{N}(y_n | \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{k'} \pi_{k'}^{(t)} \mathcal{N}(y_n | \mu_{k'}^{(t)}, \Sigma_{k'}^{(t)})}$$

M-step (weighted MLE updates):
- $\mu_k^{(t+1)} = \sum_n r_{nk}^{(t)} y_n / r_k^{(t)}$
- $\Sigma_k^{(t+1)} = \sum_n r_{nk}^{(t)} (y_n - \mu_k)(y_n - \mu_k)^\top / r_k^{(t)}$
- $\pi_k^{(t+1)} = r_k^{(t)} / N$
- where $r_k^{(t)} = \sum_n r_{nk}^{(t)}$.

#### MAP estimation for GMM
- MLE has **collapsing variance** problem: $\sigma_k \to 0$ on a single point → infinite likelihood.
- Solution: NIW prior, $\Sigma_k^{(t+1)} = (\breve{S} + \hat\Sigma_k)/(\breve\nu + r_k + D + 2)$.
- **In practice for deep learning**: same issue in mixture-of-Gaussians density estimation, normalizing flows; usually fixed by regularization, not full MAP.

#### Nonconvexity of NLL
- Mixture NLL has multiple modes (label switching → $K!$ symmetries; sometimes more).
- Finding global optimum NP-hard. K-means++ init helps.

---

## 8.8 Blackbox / derivative-free optimization

- BBO = no gradients available (e.g., hyperparameter tuning of an end-to-end pipeline).
- **Grid search**: simple but curse-of-dimensionality.
- Not covered in depth here, but in modern LLM practice:
  - **Bayesian optimization** (GP-based, e.g., Vizier) for HPO.
  - **Random search** (often beats grid for irrelevant dims).
  - **Evolutionary search**: how Google found Lion optimizer; AutoML-Zero.
  - **Population-based training (PBT)** in RLHF.
  - **LLM-as-optimizer** (OPRO, etc.): use an LLM to propose hyperparameters / prompts — emerging research direction.

---

## What actually matters for modern LLM research — TL;DR ranking

**Tier S — use daily**:
- SGD + minibatch (8.4.1)
- Momentum (8.2.4)
- Adam / AdamW (8.4.6.3) and its variants (Lion, ADOPT, Shampoo, Muon)
- LR schedules, warmup, cosine decay (8.4.3)
- EMA / SWA / Polyak averaging (8.4.4) — diffusion models, RL targets
- STE / proximal quantization (8.6.3) — LLM compression
- ELBO / variational EM (8.7.2) — VAEs, diffusion, variational inference

**Tier A — important conceptual background**:
- Convexity (8.1.3) — useful for theory and convex sub-problems
- Saddle points (8.1.1.1) — high-D non-convex intuition
- KKT conditions (8.5.2) — RLHF, DPO derivation, constrained optimization
- Trust region / TRPO connection (8.3.3)
- Subgradients (8.1.4.1) — ReLU, L1, gradient clipping
- Condition number / preconditioning (8.2.3) — explains norm layers

**Tier B — useful in specific scenarios**:
- L-BFGS (8.3.2) — convex problems, hyperparameter optimization, scientific ML, small fine-tuning
- Proximal gradient / soft thresholding (8.6) — sparse models, pruning
- EM for GMMs (8.7.3) — clustering, density modeling, mixture experts (loosely related)
- LP/QP (8.5.3-4) — optimal transport, SVMs, MPC

**Tier C — rarely used directly in deep learning**:
- Pure Newton's method (8.3.1) — too expensive
- SVRG/SAGA (8.4.5) — assumptions broken by batchnorm/dropout/aug
- Simplex algorithm (8.5.3.1)
- MILP (8.5.5) — verification only

## What Murphy under-covers (worth knowing beyond the book)

- **Distributed training**: data parallelism, ZeRO/FSDP, tensor parallelism, pipeline parallelism — all change optimizer behavior (sharded states, all-reduce timing).
- **Mixed precision**: bf16/fp8 training, loss scaling, master weights in fp32 — Adam's $\epsilon$ matters more than you'd think.
- **Gradient clipping**: $g \leftarrow g \cdot \min(1, c/\|g\|)$ — essential for transformer stability, isn't even mentioned.
- **Maximal update parameterization ($\mu$P)**: principled way to transfer LRs across scales (Yang & Hu) — important for hyperparameter transfer from small to large models.
- **Schedule-Free optimizers** (Defazio et al.): remove the need for LR schedules.
- **Second-order revival**: Shampoo, K-FAC, Muon, SOAP — actually getting wins at scale.
- **RLHF-specific optimization tricks**: PPO clipping (a form of trust region), DPO/IPO derivation via KKT, GRPO (group relative policy opt) baseline tricks.
