# MML Chapter 5 — Vector Calculus

> **TL;DR for LLM research relevance**: The single most important chapter in the book for modern ML/LLMs. **Sections 5.3 (Jacobians/vector-valued gradients), 5.6 (backprop + autodiff), and 5.7 (Hessians)** are the load-bearing topics. Sections 5.1 (univariate rules) and 5.4 (gradients of matrices via tensors) are largely subsumed by autodiff in practice but useful for derivation hygiene. 5.8 (Taylor / linearization) is the math underpinning NTK, lazy training, influence functions, and second-order optimization.

---

## Why vector calculus is the bedrock of ML
- **Training = optimization** of a loss $L(\theta)$ w.r.t. parameters $\theta \in \mathbb{R}^D$ via gradient descent / Adam / etc.
- **Gradient $\nabla_\theta L$** points in direction of steepest ascent — negate for descent.
- LLMs have $D \sim 10^{9}$–$10^{12}$ parameters, so any practical gradient algorithm must be **time/memory-efficient** — reverse-mode AD (backprop) is the only viable approach.
- Vector calc gives the language for: **gradients, Jacobians, Hessians, chain rule, Taylor expansions** — every framework (PyTorch, JAX, TF) is essentially a numerical implementation of this chapter.

---

## 5.1 Differentiation of Univariate Functions

> **Importance: LOW for LLMs.** You will never hand-compute these — autodiff does it. But you must internalize the **chain rule** intuition.

### Difference quotient & derivative
- **Difference quotient**: $\frac{\delta y}{\delta x} := \frac{f(x+\delta x)-f(x)}{\delta x}$ — slope of secant.
- **Derivative**: $\frac{df}{dx} := \lim_{h\to 0} \frac{f(x+h)-f(x)}{h}$ — slope of tangent, limit of secant.
- **Geometric meaning**: direction of steepest ascent.

### 5.1.1 Taylor series (univariate)
- **Taylor polynomial of degree $n$ at $x_0$**:
  $$T_n(x) := \sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k$$
- **Taylor series** ($n \to \infty$): exact for **analytic** functions.
- **Maclaurin series**: $x_0 = 0$.
- For polynomials of degree $\le n$, Taylor poly $T_n$ is **exact** (all higher derivatives vanish).
- Useful identities used everywhere: $\sin x = \sum (-1)^k \frac{x^{2k+1}}{(2k+1)!}$, $\cos x = \sum (-1)^k \frac{x^{2k}}{(2k)!}$.

### 5.1.2 Differentiation rules
- **Product**: $(fg)' = f'g + fg'$
- **Quotient**: $(f/g)' = (f'g - fg')/g^2$
- **Sum**: $(f+g)' = f' + g'$
- **Chain**: $(g \circ f)'(x) = g'(f(x)) f'(x)$ — **the rule that makes deep learning work.**

> **Why this matters**: every nonlinearity in a transformer (LayerNorm, softmax, GELU, attention scores) is just a composition. The chain rule is the only reason we can train depth-$L$ networks.

---

## 5.2 Partial Differentiation and Gradients

> **Importance: HIGH.** Foundational notation; sets up Jacobians and backprop.

### Partial derivative
For $f: \mathbb{R}^n \to \mathbb{R}$:
$$\frac{\partial f}{\partial x_i} = \lim_{h\to 0} \frac{f(\dots, x_i + h, \dots) - f(\mathbf{x})}{h}$$

### Gradient (Jacobian for scalar output)
$$\nabla_x f = \frac{df}{d\mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{1 \times n}$$

- **Book convention: gradient is a ROW vector** (numerator layout). This is the source of endless confusion across textbooks/papers.
- **Why row vector?**
  1. Generalizes consistently to vector-valued $f$ (Jacobian becomes $m \times n$ matrix).
  2. Multivariate chain rule becomes plain **matrix multiplication** with no transposes.

### 5.2.1 Multivariate product/sum/chain rules
- **Product**: $\frac{\partial}{\partial \mathbf{x}}(fg) = \frac{\partial f}{\partial \mathbf{x}} g + f \frac{\partial g}{\partial \mathbf{x}}$
- **Sum**: $\frac{\partial}{\partial \mathbf{x}}(f+g) = \frac{\partial f}{\partial \mathbf{x}} + \frac{\partial g}{\partial \mathbf{x}}$
- **Chain**: $\frac{\partial}{\partial \mathbf{x}}(g \circ f) = \frac{\partial g}{\partial f} \frac{\partial f}{\partial \mathbf{x}}$
  - **Mnemonic**: $\partial f$ "cancels" like a fraction — neighboring dims must match (matrix mult intuition).

### 5.2.2 Chain rule (multivariate)
For $f(x_1, x_2)$, $x_i = x_i(t)$:
$$\frac{df}{dt} = \frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2}\frac{\partial x_2}{\partial t} = \frac{\partial f}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial t}$$

For $f(\mathbf{x})$, $\mathbf{x}(s, t)$:
$$\frac{df}{d(s,t)} = \frac{\partial f}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial (s,t)}$$
— a $1\times n \times n\times 2 = 1\times 2$ row vector. **Pure matrix multiplication.**

### Gradient checking
- Numerical sanity check: $h = 10^{-4}$, finite-diff vs analytic; require $\sqrt{\sum_i (dh_i - df_i)^2 / \sum_i (dh_i + df_i)^2} < 10^{-6}$.
- **Modern usage**: `torch.autograd.gradcheck` / `jax.test_util.check_grads` — still used when writing custom CUDA kernels or custom autograd ops.

---

## 5.3 Gradients of Vector-Valued Functions — **THE BIG ONE**

> **Importance: CRITICAL.** This is the section that trips everyone up. Layout conventions matter and **disagree across sources** (PRML, Goodfellow's DL book, JAX docs, matrix cookbook).

### Setup
$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, $\mathbf{f}(\mathbf{x}) = [f_1(\mathbf{x}), \dots, f_m(\mathbf{x})]^\top \in \mathbb{R}^m$.

### Jacobian (Definition 5.6)
$$\mathbf{J} = \nabla_x \mathbf{f} = \frac{d\mathbf{f}}{d\mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

### Layout conventions — the eternal pain
| Layout | Shape of $d\mathbf{f}/d\mathbf{x}$ | Who uses it |
|---|---|---|
| **Numerator layout** (MML, this book) | $m \times n$ (output indexes rows) | MML, most ML papers, JAX docstrings (conceptually) |
| **Denominator layout** | $n \times m$ (transpose) | Many stats/optim books, matrix cookbook in places |

- **Numerator layout wins** because chain rule becomes left-to-right matrix multiplication with **no transposes**:
  $$\frac{d(g \circ f)}{d\mathbf{x}} = \underbrace{\frac{dg}{d\mathbf{f}}}_{p \times m} \underbrace{\frac{d\mathbf{f}}{d\mathbf{x}}}_{m \times n} \in \mathbb{R}^{p\times n}$$
- **Practical advice**: always write down shapes when deriving. If you find yourself transposing randomly, you have layout confusion.

### Shape cheat sheet
| $f$ type | $\frac{df}{d\mathbf{x}}$ shape |
|---|---|
| $\mathbb{R} \to \mathbb{R}$ | scalar |
| $\mathbb{R}^D \to \mathbb{R}$ | $1 \times D$ row |
| $\mathbb{R} \to \mathbb{R}^E$ | $E \times 1$ column |
| $\mathbb{R}^D \to \mathbb{R}^E$ | $E \times D$ matrix |

### Geometric interpretation
- **Jacobian determinant** $|\det \mathbf{J}|$ = local **volume scaling factor** when $\mathbf{f}$ transforms a region.
- Used in **change-of-variables formula** for probability densities:
  $$p_Y(\mathbf{y}) = p_X(\mathbf{f}^{-1}(\mathbf{y})) \, |\det \mathbf{J}_{\mathbf{f}^{-1}}(\mathbf{y})|$$
- **LLM relevance**:
  - **Normalizing flows** (RealNVP, Glow, Neural Spline Flows): the entire architectural game is **designing invertible layers with cheap Jacobian determinants** (triangular Jacobians $\to$ $\det$ = product of diagonal).
  - **Diffusion models**: continuous-time normalizing flows / probability-flow ODEs use Jacobian traces (Hutchinson estimator).
  - **Reparameterization trick** (VAEs, policy gradient with reparam): the differentiable transform $\mathbf{z} = g_\phi(\epsilon)$ uses the Jacobian implicitly via autodiff.

### Key example: linear map gradient
$\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x}$, $\mathbf{A} \in \mathbb{R}^{M \times N}$:
$$\frac{d\mathbf{f}}{d\mathbf{x}} = \mathbf{A} \in \mathbb{R}^{M \times N}$$

### Worked example: least-squares loss (Example 5.11)
Model $\mathbf{y} = \mathbf{\Phi}\boldsymbol\theta$; loss $L = \|\mathbf{e}\|^2$, $\mathbf{e} = \mathbf{y} - \mathbf{\Phi}\boldsymbol\theta$.
- $\frac{\partial L}{\partial \mathbf{e}} = 2\mathbf{e}^\top \in \mathbb{R}^{1 \times N}$
- $\frac{\partial \mathbf{e}}{\partial \boldsymbol\theta} = -\mathbf{\Phi} \in \mathbb{R}^{N \times D}$
- $\frac{\partial L}{\partial \boldsymbol\theta} = -2(\mathbf{y} - \mathbf{\Phi}\boldsymbol\theta)^\top \mathbf{\Phi} \in \mathbb{R}^{1\times D}$
- Chain rule by **matrix multiplication, no transposes** — that's the payoff of numerator layout.

---

## 5.4 Gradients of Matrices

> **Importance: MEDIUM.** Mostly relevant for analytic derivations (e.g., proving softmax gradient, attention gradient). In code, autodiff handles all of this.

### Two approaches when differentiating w.r.t. a matrix:
1. **Tensor approach**: gradient of $\mathbf{A} \in \mathbb{R}^{m \times n}$ w.r.t. $\mathbf{B} \in \mathbb{R}^{p \times q}$ is a 4-tensor $\mathbf{J} \in \mathbb{R}^{(m\times n)\times(p\times q)}$, $J_{ijkl} = \partial A_{ij}/\partial B_{kl}$.
2. **Flattening approach**: vectorize via $\text{vec}(\cdot)$ (stack columns), get $mn \times pq$ Jacobian matrix. Then chain rule becomes ordinary matrix mult; trade-off is reasoning about reshape/permute.

> **In practice**: PyTorch/JAX never materialize these 4-tensors. The autograd graph computes **vector-Jacobian products** directly (see 5.6).

### Examples
- **Ex 5.12** $\frac{d(\mathbf{A}\mathbf{x})}{d\mathbf{A}} \in \mathbb{R}^{M \times (M\times N)}$: only one row of $\mathbf{A}$ contributes to each output → sparse tensor whose nonzero "slice" is $\mathbf{x}^\top$.
- **Ex 5.13** $\frac{d(\mathbf{R}^\top \mathbf{R})}{d\mathbf{R}}$: full 4-tensor with structured entries.

---

## 5.5 Useful Identities for Computing Gradients

> **Importance: MEDIUM (reference).** Memorize a handful; look up the rest in the **Matrix Cookbook** (Petersen & Pedersen). These pop up constantly when reading ML papers.

| Identity | Note |
|---|---|
| $\frac{\partial}{\partial \mathbf{X}} \mathbf{f}(\mathbf{X})^\top = \left(\frac{\partial \mathbf{f}}{\partial \mathbf{X}}\right)^\top$ | transpose commutes with $d/dX$ |
| $\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{f}(\mathbf{X})) = \text{tr}(\partial \mathbf{f}/\partial \mathbf{X})$ | linearity |
| $\frac{\partial}{\partial \mathbf{X}} \det(\mathbf{f}(\mathbf{X})) = \det(\mathbf{f}) \text{tr}(\mathbf{f}^{-1} \partial \mathbf{f}/\partial \mathbf{X})$ | **Jacobi's formula** — used in flows |
| $\frac{\partial}{\partial \mathbf{X}} \mathbf{f}(\mathbf{X})^{-1} = -\mathbf{f}^{-1} (\partial \mathbf{f}/\partial \mathbf{X}) \mathbf{f}^{-1}$ | matrix inverse derivative |
| $\frac{\partial (\mathbf{a}^\top \mathbf{X}^{-1} \mathbf{b})}{\partial \mathbf{X}} = -(\mathbf{X}^{-1})^\top \mathbf{a}\mathbf{b}^\top (\mathbf{X}^{-1})^\top$ | Gaussian log-likelihood |
| $\frac{\partial \mathbf{x}^\top \mathbf{a}}{\partial \mathbf{x}} = \mathbf{a}^\top$ | |
| $\frac{\partial \mathbf{a}^\top \mathbf{X} \mathbf{b}}{\partial \mathbf{X}} = \mathbf{a}\mathbf{b}^\top$ | **bilinear form** |
| $\frac{\partial \mathbf{x}^\top \mathbf{B} \mathbf{x}}{\partial \mathbf{x}} = \mathbf{x}^\top(\mathbf{B} + \mathbf{B}^\top)$ | quadratic form |
| $\frac{\partial}{\partial \mathbf{s}}(\mathbf{x} - \mathbf{A}\mathbf{s})^\top \mathbf{W}(\mathbf{x} - \mathbf{A}\mathbf{s}) = -2(\mathbf{x} - \mathbf{A}\mathbf{s})^\top \mathbf{W}\mathbf{A}$ (symmetric $\mathbf{W}$) | weighted least squares |

- **Where these show up in LLMs**:
  - $\mathbf{a}\mathbf{b}^\top$ pattern → gradient of **attention logits** $\mathbf{Q}\mathbf{K}^\top$ w.r.t. $\mathbf{Q}$ or $\mathbf{K}$ is an outer product (rank-1).
  - $\det/\log\det$ derivative → **continuous normalizing flows**, **Bayesian inference** with Gaussian priors.
  - Matrix inverse derivative → **Newton's method**, **Gauss-Newton**, **natural gradient**, **K-FAC**.

---

## 5.6 Backpropagation and Automatic Differentiation — **THE OTHER BIG ONE**

> **Importance: CRITICAL.** This is the engine of all modern ML. If you internalize one thing in this chapter, make it this section.

### Why hand-derived gradients fail
Example: $f(x) = \sqrt{x^2 + \exp(x^2)} + \cos(x^2 + \exp(x^2))$.
- Hand-derived $df/dx$ is several times longer than $f$ itself → recomputes shared subexpressions.
- For deep nets, naive hand differentiation is **exponential** in depth; backprop makes it **linear**.

### 5.6.1 Backprop in deep nets
Network: $\mathbf{f}_i = \sigma_i(\mathbf{A}_{i-1}\mathbf{f}_{i-1} + \mathbf{b}_{i-1})$, loss $L = \|\mathbf{y} - \mathbf{f}_K\|^2$.

Chain rule unrolled:
$$\frac{\partial L}{\partial \boldsymbol\theta_i} = \frac{\partial L}{\partial \mathbf{f}_K} \cdot \frac{\partial \mathbf{f}_K}{\partial \mathbf{f}_{K-1}} \cdots \frac{\partial \mathbf{f}_{i+2}}{\partial \mathbf{f}_{i+1}} \cdot \frac{\partial \mathbf{f}_{i+1}}{\partial \boldsymbol\theta_i}$$

- **Key efficiency**: once $\partial L/\partial \mathbf{f}_{i+1}$ is computed, reuse it for all earlier layers.
- **Forward pass**: compute and **cache** $\mathbf{f}_0, \dots, \mathbf{f}_K$, $L$.
- **Backward pass**: walk gradients backward, $O(\text{depth})$ work.

### 5.6.2 Automatic differentiation (AD)
- AD ≠ symbolic ≠ numerical. AD is **exact to machine precision** by tracking elementary ops + chain rule on a **computation graph**.
- **Computation graph** (DAG): nodes = intermediate variables $x_i = g_i(x_{\text{Pa}(i)})$, edges = data dependencies.

#### Forward mode vs reverse mode
For composition $y = f_3(f_2(f_1(x)))$ with Jacobians $J_3 J_2 J_1$:
- **Forward mode (JVP)**: $((J_3)(J_2))(J_1)\mathbf{v}$ — pre-multiply right-to-left starting from input perturbation. Computes $J\mathbf{v}$ (one column of $J$ per pass). Cost $\propto \dim(\text{input})$.
- **Reverse mode (VJP) = backprop**: $\mathbf{u}^\top(J_3(J_2(J_1)))$ — post-multiply left-to-right starting from output cotangent. Computes $\mathbf{u}^\top J$ (one row of $J$ per pass). Cost $\propto \dim(\text{output})$.

| Mode | Best when | LLM use |
|---|---|---|
| **Reverse** | $\dim(\text{out}) \ll \dim(\text{in})$ — scalar loss, billions of params | **All gradient descent training.** |
| **Forward** | $\dim(\text{out}) \gg \dim(\text{in})$ — Jacobian of a function of few inputs | Hessian-vector products (mixed mode), sensitivity analysis, sometimes ODE solvers |

> **Why reverse mode = backprop is the backbone of every framework**:
> - LLM loss is a **scalar** ($m=1$), parameters are **billions** ($n \sim 10^{12}$).
> - Reverse mode computes the full gradient with **one** backward pass at ~2-3× the forward FLOPs.
> - Forward mode would need $n$ passes — infeasible.
> - **PyTorch** `loss.backward()` and **JAX** `jax.grad` are both reverse-mode AD.

#### JAX / PyTorch primitives
```python
# JAX exposes both modes explicitly
import jax
f = lambda x: jnp.sin(jnp.exp(x))
df_fwd = jax.jacfwd(f)   # forward-mode Jacobian
df_rev = jax.jacrev(f)   # reverse-mode Jacobian
grad_f = jax.grad(f)     # reverse-mode (scalar output)

# vjp / jvp: low-level primitives
y, vjp_fn = jax.vjp(f, x)   # returns (f(x), λu. u^T J)
y, jvp_out = jax.jvp(f, (x,), (v,))   # returns (f(x), J v)

# vmap: batched/parallel evaluation — independent of AD mode
batched_f = jax.vmap(f)

# Computing Jacobian: jacrev = vmap(vjp on one-hot); jacfwd = vmap(jvp on one-hot)
```
```python
# PyTorch
import torch
x = torch.tensor([1.0], requires_grad=True)
y = (x**2 + torch.exp(x**2)).sqrt() + torch.cos(x**2 + torch.exp(x**2))
y.backward()        # reverse mode
print(x.grad)

# Newer functorch / torch.func API mirrors JAX
from torch.func import grad, vjp, jvp, vmap, jacrev, jacfwd
```

#### vjp / jvp / vmap as composable primitives
- **VJP** $(J^\top u)$: the **adjoint** — building block of backprop.
- **JVP** $(Jv)$: the **tangent** — building block of forward-mode AD.
- **vmap**: vectorize over a batch axis, orthogonal to differentiation mode. Lets you write per-example code and get batched code for free.
- **Composition**: `jacrev = vmap(vjp, in_axes=basis_vectors)`, `hvp = grad(grad)` or `jvp ∘ grad` (mixed forward-over-reverse — cheapest Hessian-vector product).

### Computation graph example (Example 5.14)
$f(x) = \sqrt{x^2 + \exp(x^2)} + \cos(x^2 + \exp(x^2))$ via intermediates:
$a = x^2$, $b = \exp(a)$, $c = a + b$, $d = \sqrt{c}$, $e = \cos(c)$, $f = d + e$.

Local derivatives are trivial. Reverse-mode accumulation:
$$\frac{\partial f}{\partial c} = \frac{\partial f}{\partial d}\frac{\partial d}{\partial c} + \frac{\partial f}{\partial e}\frac{\partial e}{\partial c} = \frac{1}{2\sqrt{c}} - \sin(c)$$
…and so on backward through the DAG. **Same complexity as forward pass** (up to constant).

### General reverse AD
$$\frac{\partial f}{\partial x_i} = \sum_{j : x_i \in \text{Pa}(x_j)} \frac{\partial f}{\partial x_j} \frac{\partial g_j}{\partial x_i}$$
Sum over **children** in DAG. Implemented via topological order + accumulator buffers.

### Caveats
- Control flow (`for`, `if`) requires tracing or symbolic execution (JAX's `jit`, `lax.scan`, `lax.cond`).
- Non-differentiable ops (argmax, sorting, sampling) need surrogates: **straight-through estimator**, **Gumbel-softmax**, **reparameterization**, **score function (REINFORCE)** estimator.
- **Memory cost**: must store all activations for the backward pass → **gradient/activation checkpointing** trades compute for memory (recompute activations during backward). Critical for training large LLMs.

---

## 5.7 Higher-Order Derivatives

> **Importance: HIGH for advanced LLM research; MEDIUM for everyday practice.**

### Hessian
For $f: \mathbb{R}^n \to \mathbb{R}$:
$$\mathbf{H} = \nabla^2 f, \quad H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j} \in \mathbb{R}^{n \times n}$$

- **Schwarz/Clairaut**: if $f \in C^2$, $\partial^2 f/\partial x_i \partial x_j = \partial^2 f/\partial x_j \partial x_i$ → **$\mathbf{H}$ is symmetric**.
- Measures **local curvature**.
- For vector field $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, "Hessian" is an $(m \times n \times n)$ tensor.

### Why Hessians matter for LLM research
1. **Second-order optimization** — full Newton step $\mathbf{\theta} \leftarrow \mathbf{\theta} - \mathbf{H}^{-1}\nabla L$:
   - Storing/inverting $\mathbf{H}$ is $O(D^2)$ memory and $O(D^3)$ compute — impossible for LLMs.
   - **Approximations** keep it tractable:
     - **K-FAC** (Kronecker-Factored Approximate Curvature): approximates layer-wise Fisher info as Kronecker products. Used in distributed training, RL.
     - **Shampoo / Distributed Shampoo**: preconditions with block-diagonal Kronecker-factored statistics. Has been competitive with Adam on LLM pretraining (e.g., DeepMind's, MosaicML reports).
     - **GGN / Gauss-Newton**: $\mathbf{J}^\top \mathbf{H}_L \mathbf{J}$ — drops Hessian-of-network, keeps Hessian-of-loss.
     - **AdaHessian, Sophia**: diagonal Hessian estimates; Sophia (Liu et al. 2023) used for LLM pretraining.
2. **Hessian-vector products (HVP)** — $\mathbf{H}\mathbf{v}$ computable in $O(\text{forward pass})$ via:
   ```python
   # Pearlmutter trick: HVP = jvp(grad f, v) or grad(grad f · v)
   from torch.func import grad, jvp
   hvp = lambda f, x, v: jvp(grad(f), (x,), (v,))[1]
   ```
   - Used in **conjugate gradient** Newton solvers, **Lanczos** for top Hessian eigenvalues (curvature analysis).
3. **Influence functions** (Koh & Liang 2017, recent LLM training-data attribution work): per-example influence = $-\mathbf{H}^{-1} \nabla_\theta L(z_{\text{test}}) \cdot \nabla_\theta L(z_{\text{train}})$. Used for **TRAK**, **influence-based data selection for LLM finetuning**, and copyright/memorization analysis.
4. **Sharpness / loss landscape**: top eigenvalue of $\mathbf{H}$ = local sharpness. Connection to generalization:
   - **SAM** (Sharpness-Aware Minimization) uses first-order sharpness proxies.
   - **Edge of Stability** phenomenon: gradient descent operates with learning rate $> 2/\lambda_{\max}(\mathbf{H})$.
   - Curvature in LLM training: low-rank dominant Hessian spectrum, "lazy" vs "feature-learning" regimes.
5. **Bayesian deep learning**: **Laplace approximation** uses Hessian at MAP to get Gaussian posterior approximation $\mathcal{N}(\theta^*, \mathbf{H}^{-1})$. Used for LLM uncertainty quantification (Laplace-LoRA).

---

## 5.8 Linearization and Multivariate Taylor Series

> **Importance: HIGH — math underlying NTK / lazy training / linearization analyses of LLMs.**

### First-order (linearization)
$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + (\nabla_x f)(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)$$
- "Tangent plane" — local linear model.

### Multivariate Taylor series (Def. 5.7)
With $\boldsymbol\delta := \mathbf{x} - \mathbf{x}_0$:
$$f(\mathbf{x}) = \sum_{k=0}^\infty \frac{D_x^k f(\mathbf{x}_0)}{k!} \boldsymbol\delta^k$$
where $D_x^k f$ is the $k$-th **total derivative tensor** (order $k$) and $\boldsymbol\delta^k := \boldsymbol\delta \otimes \cdots \otimes \boldsymbol\delta$ is the $k$-fold outer product.

### Explicit first terms
- $k=0$: $f(\mathbf{x}_0)$ — constant.
- $k=1$: $\nabla_x f(\mathbf{x}_0) \boldsymbol\delta = \sum_i [\nabla f]_i \delta_i$.
- $k=2$: $\frac{1}{2}\boldsymbol\delta^\top \mathbf{H}(\mathbf{x}_0) \boldsymbol\delta = \frac{1}{2}\text{tr}(\mathbf{H} \boldsymbol\delta\boldsymbol\delta^\top)$ — quadratic form.
- $k=3$: $\frac{1}{6}\sum_{ijk} D^3_{ijk} \delta_i \delta_j \delta_k$ — 3-tensor contraction.

### Einsum implementations
```python
# Taylor terms as einsums
import numpy as np
# k=1: ∇f · δ
np.einsum('i,i', grad_f, delta)
# k=2: δ^T H δ
np.einsum('ij,i,j', hessian, delta, delta)
# k=3
np.einsum('ijk,i,j,k', third_deriv, delta, delta, delta)
```

### Why Taylor series matters for LLMs
1. **Neural Tangent Kernel (NTK)** (Jacot, Gabriel, Hongler 2018):
   - In infinite-width limit, $f_\theta(\mathbf{x})$ behaves like its **first-order Taylor expansion** around init $\theta_0$:
     $$f_\theta(\mathbf{x}) \approx f_{\theta_0}(\mathbf{x}) + \nabla_\theta f_{\theta_0}(\mathbf{x})^\top (\theta - \theta_0)$$
   - Training dynamics → **kernel regression** with kernel $K(\mathbf{x}, \mathbf{x}') = \nabla_\theta f(\mathbf{x})^\top \nabla_\theta f(\mathbf{x}')$.
   - The **linearization is the entire theory** — explains "lazy training" regime where params barely move.
2. **Lazy training / linearization for LLM finetuning**:
   - LoRA, prefix tuning are sometimes analyzed in a linearized regime where the pretrained $\theta_0$ is close enough that first-order Taylor is accurate.
   - **NTK eigenvalue analysis** predicts which directions in function space are learned fastest.
3. **Catastrophic forgetting & continual learning**: quadratic penalty terms (EWC, SI) are Taylor approximations of the original-task loss around finetuning init — uses Fisher info (Gauss-Newton approx of Hessian).
4. **Laplace approximation / extended Kalman filter**: linearize a nonlinear $f$ around the mode to propagate Gaussians.
5. **Unscented transform** (5.9 footnote): alternative to Taylor — sample sigma points, propagate, refit. No gradients needed.
6. **Trust-region methods / quadratic models in optimization**: every Newton-like optimizer is implicitly a 2nd-order Taylor model.

### Worked example (Example 5.15)
$f(x,y) = x^2 + 2xy + y^3$ at $(1,2)$: since $f$ is a degree-3 poly, the degree-3 Taylor expansion is **exact**:
$$f(x,y) = 13 + 6(x-1) + 14(y-2) + (x-1)^2 + 6(y-2)^2 + 2(x-1)(y-2) + (y-2)^3$$

---

## 5.9 Further Reading (book's pointers + my additions)

- **Petersen & Pedersen, "The Matrix Cookbook"** — bookmark this; reference for matrix derivatives.
- **Magnus & Neudecker** — rigorous matrix differential calculus (Kronecker products, vec operator).
- **Griewank & Walther, "Evaluating Derivatives"** — definitive AD textbook.
- **Baydin et al. (2018), "Automatic Differentiation in Machine Learning: a Survey"** — accessible AD overview.
- **JAX docs on autodiff cookbook** — vjp/jvp/vmap composition in practice.
- **Jacot et al. (2018) NTK**, **Lee et al. (2019) wide nets as linear models** — the linearization story.

### Integrals via Taylor (book's mention)
- $\mathbb{E}_x[f(\mathbf{x})] = \int f(\mathbf{x})p(\mathbf{x})d\mathbf{x}$ generally intractable.
- **Linearize $f$ around mean** → Gaussian integral closed-form. This is the **Extended Kalman Filter**.
- **2nd-order Taylor of $\log p$** around mode → **Laplace approximation**.

---

## Opinionated priority ranking for LLM researchers

| Section | Importance | Why |
|---|---|---|
| 5.6 Backprop & AD | **MUST KNOW COLD** | Engine of everything. Understand reverse vs forward, VJP/JVP, vmap, checkpointing. |
| 5.3 Jacobians (vector-valued grads) | **MUST KNOW COLD** | Shape conventions, layout pain, building block of every chain-rule derivation. |
| 5.2 Partial derivs + multivariate chain rule | **MUST KNOW** | The chain rule is the load-bearing fact. |
| 5.7 Higher-order / Hessian | **HIGH** | Second-order optimizers (Shampoo, Sophia, K-FAC), influence functions, sharpness, Laplace, Bayesian LLMs. HVPs are the key trick. |
| 5.8 Taylor / linearization | **HIGH** | NTK, lazy training, EWC, EKF, Laplace, trust regions. |
| 5.5 Useful identities | **REFERENCE** | Look up in matrix cookbook as needed. |
| 5.4 Matrix gradients | **REFERENCE / autodiff handles it** | Tensor reshaping mostly invisible in practice. |
| 5.1 Univariate calc | **LOW** | High school refresher. |

---

## Quick mental checklist for any gradient derivation
1. **Write down shapes** of every quantity before differentiating.
2. **Stick to one layout** (numerator throughout if you use MML conventions).
3. **Use chain rule by left-to-right matrix multiplication** with shapes matching.
4. **Verify against autodiff** (PyTorch / JAX `grad`) for at least one numeric input.
5. **If working with matrices/tensors**: prefer index notation (Einstein / einsum) for unambiguity.
6. **When in doubt**: derive on a tiny example ($2 \times 2$ inputs), then generalize.
