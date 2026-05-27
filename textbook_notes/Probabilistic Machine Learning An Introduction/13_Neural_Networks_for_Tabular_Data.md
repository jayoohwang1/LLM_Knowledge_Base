# Chapter 13: Neural Networks for Tabular Data

Senior-researcher framing upfront: the chapter title is "tabular data" but it's really a *foundations of feedforward DNNs + autodiff + training tricks* chapter. ~80% of what's here is still load-bearing for modern LLM work: backprop / VJPs, activation choices (GELU/SiLU/Swish), residual connections, init (He/Xavier), vanishing/exploding gradients, weight decay, dropout, flat minima, over-parameterization, mixture-of-experts. The pure "MLP on tabular" content is the part that's *least* relevant to frontier LLM research — modern LLMs are basically stacks of MLP blocks inside transformers, so the MLP fundamentals translate, but actual tabular-data SOTA is still mostly GBDTs (XGBoost/LightGBM). Pay attention to: **autodiff (13.3)**, **residuals (13.4.4)**, **activation functions (13.4.3)**, **init (13.4.5)**, **regularization / flat minima (13.5)**, **MoE (13.6.2)**.

---

## 13.1 Introduction

- **Core idea**: chain linear models with learnable feature extractors.
	- Single linear model: $f(x;\theta) = W\phi(x) + b$ with hand-crafted $\phi$.
	- **Basis function expansion** with hand-designed $\phi$ is limiting.
	- Endow $\phi$ with its own params: $f(x;\theta) = W\phi(x;\theta_2)+b$.
	- Stack $L$ such functions: $f(x;\theta) = f_L(f_{L-1}(\cdots f_1(x)\cdots))$ — this is a **DNN**.
- **DNN = differentiable composition over a DAG**, MLP is the chain-graph special case (also called **FFNN**).
- **Tabular / structured data**: $N\times D$ design matrix, each column has a fixed semantic meaning (age, weight, etc.).
	- Contrast with **unstructured data** (images, text, graphs) where individual elements are meaningless without context.
	- *Honest take*: MLPs on raw tabular still get beaten by gradient-boosted trees in almost every Kaggle competition. Useful for embeddings tables and the FFN sublayer of transformers, less so as a standalone tabular model.
- **Footnote worth highlighting**: MLP-Mixer (Tolstikhin+21) shows MLPs *can* compete on images/text given massive data — they just lack the inductive bias. Echoes the "bitter lesson" / compute-scaling story behind LLMs.

---

## 13.2 Multilayer Perceptrons (MLPs)

### 13.2.1 XOR problem
- Single perceptron $f(x) = H(w^\top x + b)$ can't represent XOR (not linearly separable).
- Hand-built 2-hidden-unit MLP solves it: $h_1 = x_1\wedge x_2$, $h_2 = x_1\vee x_2$, $y = \overline{h_1}\wedge h_2$.
- Historical motivation for hidden layers + learning weights from data.

### 13.2.2 Differentiable MLPs
- Replace Heaviside with **differentiable activation** $\varphi$.
- **Pre-activation**: $a_l = b_l + W_l z_{l-1}$.
- **Activation**: $z_l = \varphi(a_l)$.
- Now compose $L$ layers and run backprop.

### 13.2.3 Activation functions (deep dive in 13.4.3)
- Linear activations collapse to a single linear model: $W_L\cdots W_1 x = W'x$.
- Sigmoid $\sigma(a) = 1/(1+e^{-a})$ — historical, smooth Heaviside.
- **Saturation**: sigmoid/tanh flatten for large $|a|$ → gradients vanish.
- **ReLU**: $\max(a,0)$ — non-saturating for positive inputs, key enabler of deep nets.
- **Neural implicit representations / coordinate-based reps**: input is coordinates $(x,y,z)$ or $t$, output is the signal value (NeRF, SIREN). MLPs have low-frequency bias → use $\sin$ activation or positional encodings to capture high freq.
	- *Why this matters now*: positional encoding intuition flows directly into RoPE / sinusoidal PE in transformers.

### 13.2.4 Example models
- **2D classification**: stacked layers carve up input space, see tensorflow playground.
- **MNIST MLP**: flatten $28\times28 \to 784$, two 128-unit hidden layers, 10-way softmax. ~118k params, 97.1% test acc in 2 epochs. CNNs do better with fewer params (inductive bias).
- **Text classification (IMDB)**: embedding $W_1\in\mathbb{R}^{V\times E}$ → **global average pooling** $\bar e = \frac{1}{T}\sum e_t$ → MLP. 86% IMDB val acc.
	- Most params in embedding matrix → pretraining word embeddings is leverage. Foreshadows pretraining-then-finetune paradigm that became LLMs.
- **Heteroskedastic regression**: shared **backbone** + two **heads** ($\mu, \sigma$). $p(y|x) = \mathcal{N}(y \mid f_\mu(x), \sigma_+(f_\sigma(x)))$.
	- Multi-head shared-backbone is *exactly* the pattern used in modern multitask LLMs / RLHF (policy + value heads, classifier heads).

### 13.2.5 Importance of depth
- **Universal approximation theorem**: 1 hidden layer MLP can approximate any smooth function (Hornik, Cybenko 1989).
- But **deep > shallow** empirically and theoretically (Hastad, Montufar, Raghu, Poggio).
- Reason: **compositional / hierarchical** structure — layer $l+1$ reuses features from layer $l$.
- Example: regex `*AA??CGCG??AA*` — layer 1 learns motifs, layer 2 composes them.
- *Modern view*: transformers are super deep (100+ layers in frontier models) precisely because compositional features keep paying off; depth scaling laws (Kaplan, Chinchilla) are downstream of this.

### 13.2.6 The "deep learning revolution"
- Catalysts: 2011 ASR (Dahl+), 2012 ImageNet (Krizhevsky+).
- Drivers: cheap **GPUs**, large labeled datasets (ImageNet 1.3M), open-source autodiff (TF, PyTorch, JAX/MXNet).
- *Bret Victor / Andrew Ng quote*: data = fuel, models = rockets.

### 13.2.7 Connections to biology
- **McCulloch-Pitts**: $h_k = H(w_k^\top x - b_k)$.
- ANNs differ from brains: backprop vs. local Hebbian updates, feedforward vs. recurrent feedback, simple summed-input neurons vs. dendritic computation, scale, single-function vs. multi-system.
- *Honest aside*: bio-plausibility is mostly irrelevant to scaling LLMs in practice. Mentioned for color, but not load-bearing for frontier research.

---

## 13.3 Backpropagation (MOST IMPORTANT SECTION)

This is the conceptual bedrock of every neural net library. If you understand VJP/JVP and reverse-mode AD, you understand 90% of why modern training infrastructure looks the way it does.

### 13.3.1 Forward vs reverse mode
- Setup: $f = f_4 \circ f_3 \circ f_2 \circ f_1: \mathbb{R}^n \to \mathbb{R}^m$.
- **Jacobian chain rule**: $J_f(x) = J_{f_4}(x_4)J_{f_3}(x_3)J_{f_2}(x_2)J_{f_1}(x_1)$.
- **JVP** (forward mode): $J_f(x) v$ — push tangent vector forward. Cheap when $n < m$. Cost $O(n^2)$ for chain.
- **VJP** (reverse mode): $u^\top J_f(x)$ — pull cotangent backward. Cheap when $m < n$ (e.g. scalar loss, millions of params). Cost $O(n^2)$ for chain but in the orientation we want.
- **Practical rule for DL**: loss is scalar ($m=1$), params are huge ($n$ big) → always use reverse mode = backprop.

```
Forward mode:                       Reverse mode:
v_j := e_j                          x_{k+1} = f_k(x_k)   # forward pass
for k = 1..K:                       u_i := e_i
  x_{k+1} = f_k(x_k)                for k = K..1:
  v_j := J_{f_k}(x_k) v_j             u_i^T := u_i^T J_{f_k}(x_k)
```

- *Frontier-lab relevance*: JAX is built around `jvp`/`vjp` primitives. Understanding which mode to use is critical for: per-example gradients (`vmap` of `grad`), Hessian-vector products (Pearlmutter trick: `jvp` of `grad`), influence functions, second-order optimizers. K-FAC / Shampoo / Sophia all rely on cheap HVPs.

### 13.3.2 Reverse mode for MLPs
- Loss is scalar so reverse mode dominates.
- For each layer $k$:
	- $g_k = u_{k+1}^\top \partial f_k / \partial \theta_k$ (parameter gradient)
	- $u_k^\top = u_{k+1}^\top \partial f_k / \partial x_k$ (continue propagating)
- Algorithm 13.3 (backprop for $K$-layer MLP): forward pass caches $x_k$, backward pass propagates adjoint $u$ and accumulates $g_k$.

### 13.3.3 VJPs for common layers
- **Cross-entropy with logits**: $z = -\sum_c y_c \log\,\text{softmax}(x)_c$. Beautiful Jacobian: $J = (p - y)^\top$. This is why softmax + CE is a numerical and computational sweet spot — it's the canonical loss for LLM next-token prediction.
- **Elementwise nonlinearity** $z = \varphi(x)$: $J = \text{diag}(\varphi'(x))$. VJP = elementwise multiply.
- **Linear layer** $z = Wx$:
	- VJP wrt input: $u^\top W$.
	- VJP wrt weights: $[u^\top \partial z/\partial W]_{1,:} = u x^\top$ (outer product). This $ux^\top$ rank-1 outer product is the foundational operation; for LoRA, this same outer-product structure is *parameterized* as the low-rank update.

### 13.3.4 Computation graphs
- General DAG of differentiable ops.
- Topological order forward, reverse topological order backward.
- For node $j$ with children $\text{Ch}(j)$: $\partial o / \partial x_j = \sum_{k \in \text{Ch}(j)} (\partial o/\partial x_k)(\partial x_k / \partial x_j)$.
- The **adjoint** $\partial o/\partial x_k$ is reused across all parents.
- **Static graphs** (old TF1) vs **dynamic / traced graphs** (PyTorch, JAX, TF eager).
	- *2025 reality*: JAX uses tracing + XLA compilation (`jit`); PyTorch 2 has `torch.compile` doing similar. Static analysis underpins TPU/GPU performance.
- "**Differentiable programming**" framing: deep learning = programming with differentiable ops.

---

## 13.4 Training neural networks

### 13.4.1 Learning rate tuning
- One line, points to Ch 8. In practice: cosine schedules + warmup are standard in LLM pretraining; AdamW with $\beta_1=0.9, \beta_2=0.95$ and small wd $\sim 0.1$ is the default config.

### 13.4.2 Vanishing and exploding gradients (IMPORTANT)
- Repeated Jacobian multiplication: if $J_l$ constant across layers, contribution from depth-$L$ to depth-$l$ is $J^{L-l}g_L$ → behavior dictated by **spectral radius** $\lambda$.
	- $\lambda > 1$: explode. $\lambda < 1$: vanish.
- **Gradient clipping**: $g' = \min(1, c/\|g\|) g$. Standard in LLM training (typically clip to 1.0).
- Mitigations:
	- Non-saturating activations (13.4.3).
	- Additive (residual) instead of multiplicative updates (13.4.4).
	- Normalization layers (LayerNorm/BatchNorm, ch 14).
	- Careful init (13.4.5).
- *LLM relevance*: every layer of a transformer has a residual + LayerNorm precisely to keep gradients well-behaved across 100+ layers.

### 13.4.3 Non-saturating activations (IMPORTANT)
- Sigmoid saturates → $\sigma'(a) = \sigma(a)(1-\sigma(a))$ vanishes when $z\approx 0$ or $1$. Gradient wrt weights $\partial z/\partial W = z(1-z)x^\top$ → dies when saturated.
- **ReLU**: $\max(a,0)$. Gradient is just $\mathbb{I}(a>0)$. Doesn't saturate on the positive side.
	- **Dead ReLU**: if weight init or update pushes pre-activation $a$ negative everywhere, neuron stuck off forever.
- **Leaky ReLU**: $\max(\alpha a, a)$, $0<\alpha<1$. Always has nonzero gradient.
- **PReLU**: leaky ReLU but $\alpha$ is learned.
- **ELU**: smooth at zero, $\alpha(e^a-1)$ for $a\le 0$.
- **SELU**: scaled ELU with magic constants → self-normalizing.
- **Swish / SiLU**: $a\sigma(\beta a)$. Discovered by neural architecture search.
- **GELU**: $a\Phi(a)$ where $\Phi$ is standard normal CDF.
	- Interpretation: stochastic dropout where keep prob = $\Phi(a)$, expected output = $a\Phi(a)$.
	- Approx: $\text{GELU}(a) \approx a\sigma(1.702 a)$.
- *Frontier-lab take*: **GELU and SwiGLU dominate modern LLMs**. GPT-2/3/BERT use GELU. PaLM/Llama use SwiGLU ($\text{Swish}(xW_1)\odot xW_2 W_3$, a gated variant). SwiGLU empirically beats GELU at scale. This is one of the few activation-function discoveries that *actually matters* for modern work.

### 13.4.4 Residual connections (CRITICAL)
- **Residual block**: $\mathcal{F}'(x) = \mathcal{F}(x) + x$.
- Inner $\mathcal{F}$ learns a *delta* / perturbation.
- Same param count, way easier to train.
- Gradient flow: $z_L = z_l + \sum_{i=l}^{L-1} \mathcal{F}_i(z_i)$, so $\partial \mathcal{L}/\partial \theta_l$ has a direct path to $\partial \mathcal{L}/\partial z_L$ that's independent of depth.
- *Why this matters more than any other architectural trick*: ResNet (He+16) made 100+ layer training tractable. Every transformer block is `x + Attention(LN(x))` followed by `x + MLP(LN(x))`. Without residuals, transformers don't scale. Pre-norm vs post-norm placement of LN is an active research topic for stability at scale.

### 13.4.5 Parameter initialization (IMPORTANT)
- Variance analysis for linear unit $o_i = \sum_j w_{ij} x_j$:
	- $\mathbb{E}[o_i] = 0$, $\mathbb{V}[o_i] = n_\text{in}\sigma^2 \gamma^2$.
	- Need $n_\text{in}\sigma^2 \approx 1$ to avoid blow-up forward.
	- Need $n_\text{out}\sigma^2 \approx 1$ to avoid blow-up backward.
- **Xavier / Glorot**: $\sigma^2 = 2/(n_\text{in}+n_\text{out})$. Good for linear/tanh/sigmoid/softmax.
- **LeCun**: $\sigma^2 = 1/n_\text{in}$. Good for SELU.
- **He**: $\sigma^2 = 2/n_\text{in}$. **Standard for ReLU and variants**, used in modern LLMs.
- **LSUV** (data-driven): orthogonal init, then rescale per-layer using empirical first minibatch activation variance.
- *Frontier-lab take*: at scale, init still matters a lot. **Muon / muP** (Yang et al., maximal update parameterization) goes further — re-parameterizes layers so the optimal learning rate is invariant to width, letting you tune on a small model and transfer. Critical for scaling experiments.

### 13.4.6 Parallel training (CRITICAL for LLMs)
- **Model parallelism**: split model across machines. Hard, lots of comms.
- **Data parallelism**: each machine has full model copy, processes different minibatch.
	- All-reduce sum local gradients $g_t = \sum_k g_t^k$, broadcast.
	- **Synchronous SGD**: wait for all-reduce. Standard.
	- **Asynchronous / Hogwild**: no wait, can work if updates are sparse.
- *Frontier-lab reality*: real LLM training is a combination of **data parallelism + tensor parallelism + pipeline parallelism + ZeRO/FSDP sharding of optimizer state + sequence parallelism + expert parallelism (for MoE)**. The book covers only the simplest case; for actually-deployed systems see Megatron-LM, DeepSpeed, FSDP. But understanding all-reduce as the basic primitive is the right place to start.

---

## 13.5 Regularization (statistical overfitting)

### 13.5.1 Early stopping
- Stop when validation error rises. Implicitly limits info transferable from train data to params.

### 13.5.2 Weight decay
- $\ell_2$ regularizer on weights ↔ Gaussian prior ↔ MAP.
- *LLM practice*: AdamW decouples weight decay from gradient update (Loshchilov & Hutter). Standard wd $\sim$ 0.1 for LLM pretraining.

### 13.5.3 Sparse DNNs
- $\ell_1$ / ARD on weights for compression.
- Not widely used because GPUs are optimized for dense matmul. **Block / group sparsity** can prune whole layers / heads.
- *Modern relevance*: structured pruning, attention-head pruning, MoE-style sparsity (active research). Unstructured weight sparsity still mostly bypassed by quantization for inference.

### 13.5.4 Dropout
- Randomly zero out outgoing connections from each neuron with prob $p$.
- Prevents co-adaptation; ~ ensemble of thinned networks.
- Test time: multiply weights by keep prob $1-p$ to match training-time expected activation.
- **Monte Carlo dropout**: sample $S$ dropout masks at test time → approximate posterior predictive: $p(y|x,\mathcal{D}) \approx \frac{1}{S}\sum_s p(y|x, \tilde W \epsilon^s + b)$.
- *Modern LLMs*: dropout was standard in BERT/early transformers but largely *dropped* in frontier pretraining — they're underparameterized relative to data (Chinchilla regime), so regularization helps less. Reappears in fine-tuning and small models. Still relevant for uncertainty estimation (MC dropout).

### 13.5.5 Bayesian neural networks
- Marginalize params: $p(y|x,\mathcal{D}) = \int p(y|x,\theta) p(\theta|\mathcal{D})\,d\theta$.
- Infinite ensemble of different-weight nets.
- Costly. Active in safety/calibration research.
- *Honest take*: full BNNs are still niche at frontier-LLM scale (cost prohibitive). Deep ensembles + temperature scaling are the practical workhorses. Laplace approximations on last layer are emerging.

### 13.5.6 Implicit regularization of (S)GD (CONCEPTUALLY IMPORTANT)
- **Flat vs sharp minima**: flat minima generalize better (Hochreiter & Schmidhuber 97).
- Intuition: flat = robust to noise in params = doesn't memorize irrelevant details.
- SGD's noise → biased toward flat minima = **implicit regularization**.
- Smith+21, Barrett-Dherin 21 continuous-time GD flow analysis:
	- $\tilde{\mathcal{L}}_\text{GD}(w) = \mathcal{L}(w) + \frac{\epsilon}{4}\|\nabla\mathcal{L}(w)\|^2$ — full-batch GD implicitly penalizes large gradient norm.
	- $\tilde{\mathcal{L}}_\text{SGD}(w) = \tilde{\mathcal{L}}_\text{GD}(w) + \frac{\epsilon}{4m}\sum_k \|\nabla\mathcal{L}_k - \nabla\mathcal{L}\|^2$ — SGD additionally penalizes gradient variance across minibatches.
- Methods that explicitly seek flat minima:
	- **Entropy SGD** (Chaudhari+17)
	- **Sharpness-Aware Minimization (SAM)** (Foret+21)
	- **Stochastic weight averaging (SWA)** (Izmailov+18)
- *Why this is hot*: SAM and its variants (ASAM, GSAM) keep appearing in vision/LM SOTA recipes. Edge of Stability (Cohen+21) builds on this. Loss-landscape geometry is one of the few principled angles into why DL generalizes.

### 13.5.7 Over-parameterized models (CONCEPTUALLY IMPORTANT)
- $P \gg N$. Trivially achieves zero training loss for classification.
- **Double descent** (Belkin, Nakkiran): test error U-shape for $P<N$, then dips again for $P\gg N$.
- Intuition: when $P\gg N$ there are many zero-loss interpolating solutions; SGD finds smooth ones.
- *LLM relevance*: pretraining objective is *almost interpolation* on a huge corpus; modern scaling laws live in the over-parameterized regime. Grokking, neural scaling laws, induction-head formation are all over-parameterized phenomena.

---

## 13.6 Other feedforward networks *

### 13.6.1 RBF networks
- $\phi(x) = [\mathcal{K}(x,\mu_1), \ldots, \mathcal{K}(x,\mu_K)]$ with Gaussian kernel.
- Non-parametric if $K=N$. Connection to **sparse kernel machines** and **Gaussian processes** (Ch 17).
- Bandwidth $\sigma$ controls wiggliness.
- *Modern relevance*: largely supplanted by deep learning, but RBF/GP intuition lives on in **neural tangent kernel (NTK)** theory and in some uncertainty-quantification methods.

### 13.6.2 Mixtures of Experts (CRITICAL FOR MODERN LLMs)
- **One-to-many functions**: standard regression to a unimodal target produces blurry averages (e.g. 3D pose from 2D image, colorization, video prediction).
- **MoE conditional mixture model**:
	- $p(y|x) = \sum_k p(y|x, z=k) p(z=k|x)$
	- Each $p(y|x,z=k)$ is an "expert", $p(z=k|x) = \text{Cat}(\text{softmax}(f_z(x)))$ is the **gating function**.
	- **Responsibility** $p(z=k|x)$ = soft assignment.
	- **Conditional computation**: at inference, route to the most likely expert(s), save compute.
- **Mixture of linear experts**: closed-form-ish, EM-trainable.
- **Mixture density networks (MDN)**: experts and gates are DNNs.
- **Hierarchical MoE (HME)**: tree of MoEs, soft decision tree.
- *Frontier-lab importance*: MoE is **the** scaling trick of the last 3 years.
	- **Switch Transformer, GShard, GLaM, Mixtral, DeepSeek-V3, GPT-4 (rumored)** all use sparse MoE in the FFN sublayer of transformers.
	- Top-1 / top-2 routing replaces soft mixture for compute efficiency.
	- Key challenges: load balancing (auxiliary loss), routing instability, expert collapse.
	- Active topics: fine-grained experts (DeepSeek), shared experts, expert parallelism across GPUs.
	- The Jacobs/Jordan 91 paper described in the book is the *direct ancestor* of every modern sparse MoE LLM.

---

## 13.7 Exercise 13.1: Backprop for a one-hidden-layer MLP

Concrete derivation worth internalizing as the canonical micro-example.

- Setup:
	- $z = Wx + b_1 \in \mathbb{R}^K$
	- $h = \text{ReLU}(z) \in \mathbb{R}^K$
	- $a = Uh + b_2 \in \mathbb{R}^C$
	- $\mathcal{L} = \text{CE}(y, \text{softmax}(a))$
- **Local gradients** (deltas):
	- $\delta_1 = \nabla_a \mathcal{L} = p - y \in \mathbb{R}^C$
	- $\delta_2 = \nabla_z \mathcal{L} = (U^\top \delta_1) \odot H(z) \in \mathbb{R}^K$
- **Parameter gradients**:
	- $\nabla_U \mathcal{L} = \delta_1 h^\top \in \mathbb{R}^{C\times K}$
	- $\nabla_{b_2} \mathcal{L} = \delta_1$
	- $\nabla_W \mathcal{L} = \delta_2 x^\top \in \mathbb{R}^{K\times D}$
	- $\nabla_{b_1} \mathcal{L} = \delta_2$
	- $\nabla_x \mathcal{L} = W^\top \delta_2$
- **Pattern**: param grad of a linear layer = (downstream delta) $\otimes$ (input). Backprop = repeated outer product + chain. Once you have this for 1 layer, all of backprop is just bookkeeping.

---

## Senior-researcher meta-summary: what to internalize for modern LLM work

**Tier 1 — directly load-bearing daily**:
- Reverse-mode AD / VJP intuition (13.3).
- Residual connections (13.4.4) — *the* architectural primitive of transformers.
- GELU/Swish/SwiGLU activations (13.4.3) — standard in every LLM.
- He / muP initialization (13.4.5).
- AdamW + weight decay (13.5.2).
- Gradient clipping for stability (13.4.2).
- Cross-entropy on softmax logits + its clean $(p-y)$ gradient (13.3.3.1) — the LLM loss.
- Data parallelism + all-reduce as the starting point for distributed training (13.4.6).
- MoE / gating / conditional computation (13.6.2) — the scaling frontier.

**Tier 2 — important conceptual scaffolding**:
- Universal approximation + depth-vs-width / compositionality (13.2.5).
- Vanishing/exploding gradients and spectral radius (13.4.2).
- Flat vs sharp minima, SAM, implicit regularization of SGD (13.5.6).
- Over-parameterization and double descent (13.5.7).

**Tier 3 — useful background, less frequently used**:
- Dropout (13.5.4) — rarely used in modern LLM pretraining, but everywhere in older / smaller models and in MC dropout for uncertainty.
- Bayesian NNs (13.5.5) — niche but reappears in calibration / safety.
- RBF nets (13.6.1) — historical, but kernels reappear in NTK theory.
- Heteroskedastic / shared-backbone+heads (13.2.4.4) — the multi-head pattern.

**Tier 4 — mostly skip for LLM work**:
- Tabular-data framing itself (LLMs aren't tabular).
- McCulloch-Pitts / biological-plausibility section (13.2.7).
- Detailed sparse-NN $\ell_1$ pruning (13.5.3) — superseded by quantization and structured pruning.

**Final honest take**: chapter 13 is one of the better "ground-floor" treatments for someone entering LLM research. The conceptual gap from "MLP for tabular" to "transformer for LLM" is much smaller than it looks: it's almost entirely (a) attention as a sequence-aware mixing layer, (b) LayerNorm everywhere, (c) bigger blocks, (d) MoE in the FFN. Master backprop, residuals, init, and activations cold, and you can read modern LLM papers fluently.
