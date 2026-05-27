# Chapter 8: Backpropagation

> **Overall relevance to modern LLM research: CRITICAL (but mostly invisible).** Backpropagation is the algorithm that makes training neural networks computationally tractable. You almost never implement it by hand anymore — autodiff frameworks (PyTorch, JAX) handle it — but understanding what's happening under the hood is essential for debugging gradient issues, designing custom operations, understanding memory/compute tradeoffs, and building intuition about why certain architectures train well or badly. The autodiff section (8.2) is arguably the most practically important part, since it describes exactly what your framework is doing. The Jacobian and Hessian sections are increasingly relevant as people explore second-order methods, mechanistic interpretability, and model compression.

---

## 8.1 Evaluation of Gradients

**Relevance: HIGH.** This is the mathematical backbone. You need to understand this to know what your framework is computing, even if you never write the equations yourself.

### 8.1.1 Single-Layer Networks
- Start with the simplest case: linear model $y_k = \sum_i w_{ki} x_i$
- Sum-of-squares error for one data point: $E_n = \frac{1}{2}\sum_k (y_{nk} - t_{nk})^2$
- Gradient w.r.t. weight: $\frac{\partial E_n}{\partial w_{ji}} = (y_{nj} - t_{nj})x_{ni}$
	- **Local computation**: error signal $(y - t)$ at the output end times input $x$ at the input end
	- Same form appears with cross-entropy + sigmoid and cross-entropy + softmax (canonical link functions)
- This "output error times input activation" pattern is the seed of the full backprop algorithm

### 8.1.2 General Feed-Forward Networks

**This is the core derivation.** Everything else in the chapter builds on it.

- General feed-forward network: each unit computes weighted sum of inputs, then applies nonlinearity
	- **Pre-activation**: $a_j = \sum_i w_{ji} z_i$ (weighted sum, where $z_i$ are activations of connected units)
	- **Activation**: $z_j = h(a_j)$ (apply nonlinear activation function)
	- Biases folded in via a fixed input $z_0 = 1$
- **Forward propagation**: compute all $a_j$ and $z_j$ from inputs to outputs by successive application of these equations
- **Goal**: evaluate $\frac{\partial E_n}{\partial w_{ji}}$ for a single data point

#### The Chain Rule Decomposition
- $E_n$ depends on $w_{ji}$ only through $a_j$, so chain rule gives: $\frac{\partial E_n}{\partial w_{ji}} = \frac{\partial E_n}{\partial a_j} \cdot \frac{\partial a_j}{\partial w_{ji}}$
- Define **errors** (aka **deltas**): $\delta_j \equiv \frac{\partial E_n}{\partial a_j}$
- Since $\frac{\partial a_j}{\partial w_{ji}} = z_i$, we get the key result:

$$\frac{\partial E_n}{\partial w_{ji}} = \delta_j z_i$$

- **Interpretation**: gradient = error at output end $\times$ activation at input end
	- Identical structure to the single-layer case — just with $\delta$ replacing $(y - t)$
	- For biases (where $z = 1$): gradient is just $\delta_j$

#### Computing the Deltas
- **Output units** (with canonical link): $\delta_k = y_k - t_k$
- **Hidden units** — apply chain rule again, summing over all units $k$ that unit $j$ feeds into:

$$\delta_j = h'(a_j) \sum_k w_{kj} \delta_k$$

- **This is the backpropagation formula** — deltas flow backwards through the network
	- Multiply by activation derivative $h'(a_j)$, then take weighted sum of downstream deltas
	- Weights used are the **same weights** as forward pass, but indexed in reverse (sum over first index $k$ of $w_{kj}$, not second)

#### Algorithm 8.1: Backpropagation
1. **Forward pass**: compute $a_j$ and $z_j$ for all hidden and output units
2. **Error evaluation**: compute $\delta_k$ for output units
3. **Backward pass**: compute $\delta_j$ for all hidden units in reverse order using the backprop formula
4. **Gradient evaluation**: $\frac{\partial E_n}{\partial w_{ji}} = \delta_j z_i$ for each weight

- For batch/mini-batch: sum over data points $\frac{\partial E}{\partial w_{ji}} = \sum_n \frac{\partial E_n}{\partial w_{ji}}$

**Why this matters now**: This is literally what `loss.backward()` computes. Every time you train a transformer, this algorithm runs. The "backward pass takes about the same compute as the forward pass" intuition comes directly from this — you do one backward sweep through the same graph. The fact that each weight gradient is a simple local product ($\delta \times z$) is why gradient accumulation, distributed training, and pipeline parallelism are architecturally feasible.

### 8.1.3 A Simple Example
- Two-layer network with tanh hidden units, linear outputs, sum-of-squares error
- $h(a) = \tanh(a)$, with convenient derivative $h'(a) = 1 - h(a)^2 = 1 - z^2$
	- This is why tanh was popular historically: derivative expressed purely in terms of the output
- Forward: $a_j = \sum_i w_{ji}^{(1)} x_i$, $z_j = \tanh(a_j)$, $y_k = \sum_j w_{kj}^{(2)} z_j$
- Output deltas: $\delta_k = y_k - t_k$
- Hidden deltas: $\delta_j = (1 - z_j^2) \sum_k w_{kj}^{(2)} \delta_k$
- Gradients: $\frac{\partial E_n}{\partial w_{ji}^{(1)}} = \delta_j x_i$, $\frac{\partial E_n}{\partial w_{kj}^{(2)}} = \delta_k z_j$

**Modern note**: ReLU ($h'(a) = \mathbf{1}[a > 0]$) makes this even simpler — no vanishing derivative issue, just a binary gate. That simplicity is a big part of why ReLU won.

### 8.1.4 Numerical Differentiation

**Relevance: LOW for production, HIGH for debugging.** Nobody trains with finite differences, but everyone should use them to check their gradients.

- **Computational cost of backprop**: $\mathcal{O}(W)$ per data point (same order as forward pass)
	- One forward pass = $\mathcal{O}(W)$ (dominated by weight multiplications)
	- One backward pass = also $\mathcal{O}(W)$
	- This is the key insight: **gradients are essentially free** relative to the cost of evaluation
- **Finite differences**: $\frac{\partial E_n}{\partial w_{ji}} \approx \frac{E_n(w_{ji} + \epsilon) - E_n(w_{ji})}{\epsilon} + \mathcal{O}(\epsilon)$
	- Requires one forward pass per weight → $\mathcal{O}(W^2)$ total — completely impractical for real networks
- **Central differences**: $\frac{\partial E_n}{\partial w_{ji}} \approx \frac{E_n(w_{ji} + \epsilon) - E_n(w_{ji} - \epsilon)}{2\epsilon} + \mathcal{O}(\epsilon^2)$
	- Much more accurate ($\mathcal{O}(\epsilon^2)$ vs $\mathcal{O}(\epsilon)$) but 2x the cost
- **Tradeoff with $\epsilon$**: too large → truncation error dominates; too small → floating-point round-off dominates
	- Optimal $\epsilon$ for finite differences: where these two error sources cross (visible in Bishop's Figure 8.2)
	- Central differences shift this crossover, giving better accuracy at the same $\epsilon$

**Why this still matters**: `torch.autograd.gradcheck` uses exactly this technique. When implementing custom autograd functions, RLHF reward models, or novel loss functions, you should always verify your backward pass against numerical differences. This catches bugs that would otherwise silently corrupt training. The $\mathcal{O}(W^2)$ cost doesn't matter because you only run it on small test cases.

### 8.1.5 The Jacobian Matrix

**Relevance: HIGH and growing.** Jacobians show up everywhere in modern research — mechanistic interpretability, neural ODEs, normalizing flows, sensitivity analysis, and understanding information flow through networks.

- **Jacobian** $J_{ki} = \frac{\partial y_k}{\partial x_i}$: derivatives of outputs w.r.t. inputs
	- Measures **local sensitivity** of each output to each input
	- For a nonlinear network, $\mathbf{J}$ depends on the specific input point (unlike linear models)
- **Modular systems**: for a system composed of differentiable modules, chain rule via Jacobians:
$$\frac{\partial E}{\partial w} = \sum_{k,j} \frac{\partial E}{\partial y_k} \frac{\partial y_k}{\partial z_j} \frac{\partial z_j}{\partial w}$$
	- Jacobian of the intermediate module appears as the "bridge" connecting upstream and downstream gradients
- **Error propagation**: small input perturbations $\Delta x_i$ cause output changes $\Delta y_k \approx \sum_i J_{ki} \Delta x_i$
	- Only valid for small perturbations (linear approximation)
	- Must be recomputed at each new input point
- **Computing the Jacobian via backprop**:
	- Write $J_{ki} = \frac{\partial y_k}{\partial x_i} = \sum_j w_{ji} \frac{\partial y_k}{\partial a_j}$
	- Recursive formula: $\frac{\partial y_k}{\partial a_j} = h'(a_j) \sum_l w_{lj} \frac{\partial y_k}{\partial a_l}$
	- Start from output units: $\frac{\partial y_k}{\partial a_l} = \delta_{kl}$ (linear), $\delta_{kl}\sigma'(a_l)$ (sigmoid), $\delta_{kl}y_k - y_k y_l$ (softmax)
	- Backpropagate row by row: each row $k$ requires one backward pass
- **Numerical verification**: $\frac{\partial y_k}{\partial x_i} \approx \frac{y_k(x_i + \epsilon) - y_k(x_i - \epsilon)}{2\epsilon}$
	- Requires $2D$ forward passes for a network with $D$ inputs → $\mathcal{O}(DW)$ total

**Where Jacobians matter in practice**:
- **Normalizing flows**: require computing $\log|\det \mathbf{J}|$ — entire architecture families (RealNVP, Glow, etc.) are designed to make this cheap
- **Neural ODEs**: Jacobian of the dynamics function determines stability and is needed for the adjoint method
- **Mechanistic interpretability**: Jacobians of intermediate representations reveal how information flows through transformer layers — which input tokens influence which output positions
- **Input sensitivity / saliency**: $\|\mathbf{J}\|$ measures how sensitive the model is to input perturbations — used in adversarial robustness and attribution methods
- **Fine-tuning theory**: Fisher information matrix (closely related to $\mathbf{J}^\top \mathbf{J}$) determines which parameter directions matter most — used in elastic weight consolidation and related continual learning methods

### 8.1.6 The Hessian Matrix

**Relevance: MEDIUM-HIGH and increasing.** Full Hessians are impractical for LLMs, but Hessian-related quantities are critical for optimization, pruning, quantization, and understanding loss landscapes.

- **Hessian**: $H_{ij} = \frac{\partial^2 E}{\partial w_i \partial w_j}$ — second derivatives of the error w.r.t. all weight pairs
	- $W \times W$ matrix where $W$ = total parameters
	- For modern LLMs: $W$ = billions, so the full Hessian has $\sim W^2$ entries — completely infeasible to store or compute
- **Uses of the Hessian** (Bishop lists several):
	- Second-order optimization methods (Newton's method, natural gradient)
	- Bayesian treatments of neural networks (Laplace approximation)
	- **Reducing weight precision** (quantization) — knowing curvature tells you which weights can tolerate rounding
- **Computational costs**:
	- Naively: $\mathcal{O}(W^2)$ per data point via backprop extension
	- **Hessian-vector product** $\mathbf{v}^\top \mathbf{H}$: can be computed in $\mathcal{O}(W)$ via a single forward + backward pass (Pearlmutter, 1994)
		- This is the practical workhorse — you almost never need the full Hessian, just its action on vectors
	- Full Hessian: $\mathcal{O}(W^2)$ to evaluate, $\mathcal{O}(W^3)$ to invert

#### Approximations to the Hessian
- **Diagonal approximation**: only compute $\frac{\partial^2 E}{\partial w_i^2}$ — $\mathcal{O}(W)$ storage, $\mathcal{O}(W)$ to invert
	- Assumes parameters are independent — bad assumption in practice (Hessian has significant off-diagonal terms)
	- Used in some older methods (Becker & LeCun 1989)
- **Outer product approximation (Levenberg-Marquardt)**:
	- For sum-of-squares error: $\mathbf{H} = \nabla\nabla E = \sum_n \nabla y_n (\nabla y_n)^\top + \sum_n (y_n - t_n)\nabla\nabla y_n$
	- If network fits data well, residuals $(y_n - t_n)$ are small and uncorrelated → second term averages to zero
	- **Approximation**: $\mathbf{H} \simeq \sum_n \nabla a_n \nabla a_n^\top$ (outer product of gradients)
		- Only needs first derivatives — cheap to evaluate via standard backprop
		- Elements found in $\mathcal{O}(W^2)$ by simple outer products
	- For cross-entropy + sigmoid outputs: $\mathbf{H} \simeq \sum_n y_n(1 - y_n) \nabla a_n \nabla a_n^\top$
	- For cross-entropy + softmax: analogous result exists
	- **Important caveat**: only valid when the network has been trained well — second-derivative terms are not negligible for a general (undertrained) network
- **Sequential Hessian inverse updates**: using Woodbury identity $(M + vv^\top)^{-1} = M^{-1} - \frac{(M^{-1}v)(v^\top M^{-1})}{1 + v^\top M^{-1} v}$
	- Can incrementally update $\mathbf{H}^{-1}$ as each data point is processed
	- Initialize with $\mathbf{H} = \alpha \mathbf{I}$ (not sensitive to $\alpha$)

**Where Hessian information matters in modern practice**:
- **Model compression / quantization**: OBS (Optimal Brain Surgeon), OBQ, and GPTQ all use Hessian information to decide which weights can be quantized or pruned with minimal loss increase — this is how people fit 70B models onto consumer GPUs
- **Second-order optimizers**: K-FAC (Kronecker-Factored Approximate Curvature) approximates the Fisher information matrix block-diagonally — has shown promise for faster convergence in some LLM training setups
- **Sharpness-Aware Minimization (SAM)**: implicitly uses curvature information to find flat minima that generalize better
- **Influence functions**: use Hessian-vector products to trace model predictions back to training data points — important for data attribution and debugging
- **Loss landscape analysis**: Hessian eigenspectrum reveals the structure of the loss surface — bulk of eigenvalues near zero (most directions are flat), a few large eigenvalues (sharp directions)
- **Neural network pruning**: second-order Taylor approximation of the loss increase from removing a weight: $\Delta E \approx \frac{1}{2} H_{ii} w_i^2$ — tells you which weights to remove

---

## 8.2 Automatic Differentiation

**Relevance: CRITICAL. This is the single most important section in the chapter for modern practice.** Autodiff is what makes PyTorch, JAX, TensorFlow, and every other deep learning framework possible. Understanding forward vs. reverse mode is essential for knowing why training is efficient, why certain operations are memory-intensive, and how to write custom differentiable operations.

### Four Approaches to Evaluating Gradients

Bishop lays out four methods, which is a great way to frame the landscape:

1. **Manual derivation + implementation**: derive backprop equations by hand, code them separately
	- Error-prone, redundant (forward and backward coded separately), inflexible
	- How things were done for decades — painful for architecture exploration
2. **Numerical differentiation** (finite differences): only need forward pass code
	- $\mathcal{O}(W^2)$ cost — impractical for training
	- Limited accuracy (floating-point issues)
	- Useful for **gradient checking** only
3. **Symbolic differentiation**: computer algebra system applies calculus rules mechanically
	- Produces exact derivative expressions
	- **Expression swell**: derivative expressions grow exponentially in complexity
		- Example: $f(x) = u(x)v(x)$ → $f'(x) = u'(x)v(x) + u(x)v'(x)$ — redundant evaluation of $u(x)$ and $v(x)$
		- Nested products create exponential blowup
	- Can't handle control flow (loops, branches, recursion) — requires closed-form expressions
	- Bishop's example: two-layer network with softReLU $\zeta(a) = \ln(1 + \exp(a))$ — symbolic derivative of $\partial y/\partial w_1$ is already a monster expression with redundant subexpressions
4. **Automatic differentiation** (autodiff): the winner
	- Not finding a formula — generating **code** that computes the gradient alongside the function
	- Accurate to machine precision (like symbolic)
	- Exploits intermediate variables to avoid redundancy (unlike symbolic)
	- Handles control flow: loops, branches, recursion, procedure calls
	- Two modes: **forward** and **reverse**

### 8.2.1 Forward-Mode Automatic Differentiation

- Augment each intermediate variable $v_i$ (primal) with a **tangent variable** $\dot{v}_i = \frac{\partial v_i}{\partial x_j}$ (derivative w.r.t. one chosen input)
- Code evaluates tuples $(v_i, \dot{v}_i)$ in parallel during the forward pass
- **Tangent propagation rule** (chain rule on the evaluation trace):
$$\dot{v}_i = \sum_{j \in \text{pa}(i)} \dot{v}_j \frac{\partial v_i}{\partial v_j}$$
	- $\text{pa}(i)$ = parent nodes in the evaluation trace (variables that feed into $v_i$)

#### Example
- $f(x_1, x_2) = x_1 x_2 + \exp(x_1 x_2) - \sin(x_2)$
- **Evaluation trace** (computational graph): $v_1 = x_1, v_2 = x_2, v_3 = v_1 v_2, v_4 = \sin(v_2), v_5 = \exp(v_3), v_6 = v_3 - v_4, v_7 = v_5 + v_6$
- To compute $\partial f/\partial x_1$: set $\dot{v}_1 = 1, \dot{v}_2 = 0$ (seed)
- Tangent equations propagate forward: $\dot{v}_3 = v_1\dot{v}_2 + \dot{v}_1 v_2$, etc.
- Final $\dot{v}_7 = \partial f/\partial x_1$

#### Key Properties
- One forward pass computes **one column** of the Jacobian $\mathbf{J}$ (derivatives of all outputs w.r.t. one input)
- For $D$ inputs and $K$ outputs: need $D$ forward passes for the full $K \times D$ Jacobian
- **Efficient when $K \gg D$** (many outputs, few inputs) — each pass gets a full column
- Can compute Jacobian-vector product $\mathbf{J}\mathbf{r}$ in a single pass by setting $\dot{\mathbf{x}} = \mathbf{r}$

**When forward mode is used in practice**:
- **Per-example gradients**: when you need gradients of a scalar loss w.r.t. a small number of hyperparameters (not model weights)
- **Jacobian-vector products** in neural ODEs (forward sensitivity method)
- **Hessian-vector products**: combine forward mode (to get $\mathbf{v}^\top \nabla f$) with reverse mode (to differentiate that scalar) — $\mathcal{O}(W)$ computation for $\nabla^2 f \cdot \mathbf{v}$, avoiding the full $\mathcal{O}(W^2)$ Hessian
- JAX's `jvp` primitive — useful when you have few inputs relative to outputs

### 8.2.2 Reverse-Mode Automatic Differentiation

**This is the one that matters for training.** Reverse-mode autodiff is a generalization of backpropagation to arbitrary computational graphs.

- Augment each intermediate variable $v_i$ with an **adjoint variable** $\bar{v}_i = \frac{\partial f}{\partial v_i}$ (derivative of output w.r.t. that intermediate)
- **Adjoint propagation rule** (chain rule, working backwards):
$$\bar{v}_i = \sum_{j \in \text{ch}(i)} \bar{v}_j \frac{\partial v_j}{\partial v_i}$$
	- $\text{ch}(i)$ = children of node $i$ (variables that $v_i$ feeds into)
- Start from output: $\bar{v}_{\text{output}} = 1$, propagate backwards to inputs

#### Example (same function)
- $\bar{v}_7 = 1$ (seed)
- $\bar{v}_6 = \bar{v}_7 = 1$, $\bar{v}_5 = \bar{v}_7 = 1$
- $\bar{v}_4 = -\bar{v}_6 = -1$
- $\bar{v}_3 = \bar{v}_5 v_5 + \bar{v}_6$ (from $v_5 = \exp(v_3)$ and $v_6 = v_3 - v_4$)
- Continue back to get $\bar{v}_1 = \partial f/\partial x_1$ and $\bar{v}_2 = \partial f/\partial x_2$

#### Key Properties
- One backward pass computes **one row** of the Jacobian (derivatives of one output w.r.t. all inputs/parameters)
- For a scalar output (loss function): **a single backward pass gives ALL gradients**
	- This is why we can train networks with billions of parameters — one forward + one backward pass
- **Efficient when $D \gg K$** (many inputs/parameters, few outputs) — the standard training regime
- Computes **vector-Jacobian product** $\mathbf{r}^\top \mathbf{J}$ in a single pass

#### Forward vs. Reverse: The Critical Comparison

| | Forward Mode | Reverse Mode |
|---|---|---|
| **Computes** | One Jacobian column | One Jacobian row |
| **Efficient for** | Few inputs, many outputs ($K \gg D$) | Few outputs, many inputs ($D \gg K$) |
| **Passes needed for full Jacobian** | $D$ forward passes | $K$ backward passes |
| **For training** (scalar loss, $W$ params) | $W$ passes needed — **completely impractical** | **1 pass** — this is why it wins |
| **Memory** | Low — compute tangents alongside primals, discard | **High** — must store all intermediate values for backward pass |
| **Implementation** | Simpler — just augment forward code | Harder — need to reverse the computation |
| **Product computed** | $\mathbf{J}\mathbf{v}$ (Jacobian-vector) | $\mathbf{v}^\top\mathbf{J}$ (vector-Jacobian) |

- **Overhead guarantee**: for both modes, a single pass through the graph costs at most **6x** the cost of a function evaluation. In practice, closer to **2-3x**.

#### Memory vs. Compute Tradeoff
- Reverse mode must store **all intermediate primal values** (activations) for the backward pass
	- Forward mode doesn't need this — tangent and primal computed together, then discarded
- This is the fundamental source of the **memory wall** in training deep networks
	- For a transformer with $L$ layers, you need to store activations at every layer during forward pass
	- This is why **gradient checkpointing** (aka activation recomputation) exists: throw away some activations during forward pass, recompute them during backward pass — trade compute for memory
	- Also why **mixed-precision training** matters: storing activations in fp16/bf16 halves memory

**Hybrid forward + reverse for Hessian-vector products**:
- Want to compute $\nabla^2 f \cdot \mathbf{v} = \mathbf{H}\mathbf{v}$
- Step 1: forward mode with $\dot{\mathbf{x}} = \mathbf{v}$ to get the directional derivative $\mathbf{v}^\top \nabla f$ (a scalar)
- Step 2: reverse mode to differentiate that scalar w.r.t. parameters
- Total cost: $\mathcal{O}(W)$ — even though $\mathbf{H}$ is $W \times W$
- The full Hessian can also be computed via autodiff but costs $\mathcal{O}(W^2)$

**Why this is the most important section**:
- **PyTorch's `autograd`** = reverse-mode autodiff with dynamic computation graphs (define-by-run)
- **JAX's `grad`** = reverse-mode autodiff with functional transformations (can compose `grad`, `jvp`, `vmap`)
- **Understanding memory usage**: activation memory during training is a direct consequence of reverse mode's need to store intermediates — this is why inference is cheap but training is expensive
- **Custom autograd functions**: when you implement `torch.autograd.Function`, you're manually specifying the local Jacobian (the `backward` method) — understanding the chain rule decomposition here tells you exactly what that function needs to compute
- **Gradient accumulation / pipeline parallelism**: works because the backward pass decomposes into local operations that can be scheduled flexibly
- **Checkpointing strategies**: understanding the activation storage requirement of reverse mode is essential for fitting large models into limited GPU memory — e.g., in Megatron-LM, activation checkpointing at every transformer layer is standard

---

## Commentary: What Matters Most for LLM Research

### Tier 1: You use this every day (even if you don't realize it)
- **Reverse-mode autodiff** (8.2.2): This IS `loss.backward()`. Understanding the memory-compute tradeoff is essential for training large models.
- **The backpropagation algorithm** (8.1.2): The mathematical foundation. Knowing that gradients are $\delta_j z_i$ (local products) explains why gradient statistics are easy to collect, why gradient clipping works the way it does, and why certain layer designs have stable/unstable training.
- **$\mathcal{O}(W)$ gradient computation** (8.1.4): The most important complexity result in all of deep learning. Training is only feasible because gradients cost the same as a forward pass.

### Tier 2: You need this for specific research areas
- **Jacobian matrices** (8.1.5): Critical for normalizing flows, neural ODEs, mechanistic interpretability, adversarial robustness, and input attribution methods. Also relevant to understanding how information propagates through transformer layers.
- **Hessian approximations** (8.1.6): Essential for post-training compression (GPTQ, AWQ), pruning (OBS/OBD), influence functions, and second-order optimization. The outer product approximation $\mathbf{H} \approx \sum_n \nabla a_n \nabla a_n^\top$ is the conceptual ancestor of the Fisher information matrix and K-FAC.
- **Forward-mode autodiff** (8.2.1): Needed for efficient Jacobian-vector products, Hessian-vector products (combined with reverse mode), and per-example gradient computations. JAX makes this easy with `jvp`.

### Tier 3: Good to know, rarely directly needed
- **Numerical differentiation** (8.1.4): Use `gradcheck` when debugging custom ops. Otherwise never.
- **Simple backprop example** (8.1.3): Pedagogically useful. The tanh derivative trick $h'(a) = 1 - h(a)^2$ is a historical artifact — ReLU made this moot. But it illustrates why activation function choice affects gradient flow.
- **Symbolic differentiation** (8.2): Understanding why it fails (expression swell, no control flow) explains why autodiff was the right answer. But you'll never use sympy for training.

### The Big Picture

The reason this chapter exists is to answer one question: **how do we efficiently compute $\nabla_\mathbf{w} E$?** The answer — reverse-mode autodiff, which generalizes backpropagation — is arguably the single most important algorithmic insight in deep learning. It's what makes the whole field computationally tractable. Everything else (architectures, loss functions, data) is built on top of the assumption that we can cheaply differentiate through arbitrary computation.

If you're at a frontier lab, you probably interact with this machinery in three ways:
1. **Trusting it**: `loss.backward()` just works. But knowing the algorithm helps you debug when it doesn't (NaN gradients, memory OOMs, slow backward passes).
2. **Working around its constraints**: gradient checkpointing, mixed-precision storage, pipeline parallelism — all motivated by reverse mode's memory requirements.
3. **Extending it**: custom backward passes for novel operations, stop-gradient operators for RLHF/contrastive learning, higher-order derivatives for meta-learning or Hessian estimation.

The shift from manual backprop derivation (approach 1) to autodiff (approach 4) is what enabled the modern era of "just try it" architecture exploration. You can define an arbitrary computation in PyTorch and get gradients for free — this is why architecture search, novel attention mechanisms, and exotic loss functions are all so easy to prototype.
