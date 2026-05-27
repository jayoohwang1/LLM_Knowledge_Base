# Chapter 7: Gradient Descent

> **Overall relevance to modern LLM research: CRITICAL.** This is the chapter about how neural networks actually learn. Every concept here — SGD, momentum, Adam, learning rate schedules, batch/layer normalization, weight initialization — is used daily in training LLMs. If Chapter 6 was "what neural nets compute," Chapter 7 is "how we make them compute the right thing." There are no historical curiosities here; it's all live infrastructure.

---

## 7.1 Error Surfaces

**Relevance: Medium-High.** You rarely think about Hessians explicitly during training, but understanding the geometry of loss landscapes is essential for debugging training runs and choosing hyperparameters.

### Local Quadratic Approximation
- Taylor expand the error function around a point $\hat{\mathbf{w}}$:
$$E(\mathbf{w}) \approx E(\hat{\mathbf{w}}) + (\mathbf{w} - \hat{\mathbf{w}})^\top \mathbf{b} + \frac{1}{2}(\mathbf{w} - \hat{\mathbf{w}})^\top \mathbf{H}(\mathbf{w} - \hat{\mathbf{w}})$$
	- $\mathbf{b} = \nabla E|_{\hat{\mathbf{w}}}$ is the gradient
	- $\mathbf{H}$ is the **Hessian matrix** (second derivatives), $W \times W$ where $W$ = total number of parameters
- At a minimum $\mathbf{w}^*$ the gradient vanishes, so: $E(\mathbf{w}) \approx E(\mathbf{w}^*) + \frac{1}{2}(\mathbf{w} - \mathbf{w}^*)^\top \mathbf{H}(\mathbf{w} - \mathbf{w}^*)$

### Hessian Eigenstructure
- Hessian eigenvectors $\{\mathbf{u}_i\}$ with eigenvalues $\{\lambda_i\}$: $\mathbf{H}\mathbf{u}_i = \lambda_i \mathbf{u}_i$
- Decompose displacement from minimum: $\mathbf{w} - \mathbf{w}^* = \sum_i \alpha_i \mathbf{u}_i$
- Error becomes: $E(\mathbf{w}) = E(\mathbf{w}^*) + \frac{1}{2}\sum_i \lambda_i \alpha_i^2$
	- Each eigendirection contributes independently to the error
- **Classification of stationary points by eigenvalues**:
	- All $\lambda_i > 0$ → **local minimum** ($\mathbf{H}$ positive definite)
	- All $\lambda_i < 0$ → **local maximum**
	- Mixed signs → **saddle point**
- **Positive definiteness**: $\mathbf{H}$ is positive definite iff $\mathbf{v}^\top \mathbf{H} \mathbf{v} > 0$ for all $\mathbf{v}$, iff all eigenvalues are positive
- Contours of constant error = **ellipses** aligned with eigenvectors, axis lengths $\propto \lambda_i^{-1/2}$
	- Large eigenvalue = high curvature = short axis (loss changes rapidly in that direction)
	- Small eigenvalue = low curvature = long axis (loss changes slowly)

**Why this matters now**: Understanding the Hessian is key to understanding why optimization is hard:
- **Condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ determines how elongated the loss landscape is — high $\kappa$ means gradient descent oscillates across narrow valleys and makes slow progress along them
	- This is literally why Adam exists — to handle different curvatures per parameter
- **Saddle points** are far more common than local minima in high dimensions — in a space with $W$ parameters, a stationary point has roughly equal probability of each eigenvalue being positive or negative, so for large $W$ almost all stationary points are saddle points
	- This was a major theoretical insight (Dauphin et al. 2014): the problem isn't local minima, it's saddle points
- **Sharpness-aware minimization (SAM)** and related methods explicitly try to find minima where the Hessian eigenvalues are small (flat minima), which tend to generalize better
- **Loss landscape visualization** (Li et al. 2018) is a diagnostic tool for understanding training dynamics

---

## 7.2 Gradient Descent Optimization

**Relevance: CRITICAL.** This is the actual training algorithm. Everything here is used in practice.

### 7.2.1 Use of Gradient Information
- Without gradients: locating a minimum requires $\mathcal{O}(W^2)$ function evaluations (need $W(W+3)/2$ independent measurements to determine $\mathbf{b}$ and $\mathbf{H}$), each costing $\mathcal{O}(W)$ → total $\mathcal{O}(W^3)$
- With gradients: each gradient evaluation gives $W$ pieces of information, so minimum found in $\mathcal{O}(W^2)$ with $\mathcal{O}(W)$ gradient evaluations (via backprop, each costs $\mathcal{O}(W)$)
- **Key insight**: gradient information is essentially free via backpropagation — this is the fundamental reason gradient-based methods dominate

### 7.2.2 Batch Gradient Descent
- Update: $\mathbf{w}^{(\tau)} = \mathbf{w}^{(\tau-1)} - \eta \nabla E(\mathbf{w}^{(\tau-1)})$
- $\eta > 0$ = **learning rate**
- Requires processing the **entire training set** to compute one gradient → impractical for large datasets
- Also called **steepest descent** — steps in the direction of greatest error decrease

### 7.2.3 Stochastic Gradient Descent (SGD)
- Error decomposes as sum over data points: $E(\mathbf{w}) = \sum_{n=1}^{N} E_n(\mathbf{w})$
- **SGD update**: $\mathbf{w}^{(\tau)} = \mathbf{w}^{(\tau-1)} - \eta \nabla E_n(\mathbf{w}^{(\tau-1)})$ — one data point at a time
- One pass through all data = one **epoch**
- Also called **online gradient descent** (can handle streaming data)
- **Advantages over batch**:
	- Handles **data redundancy** efficiently — duplicating the dataset doesn't change SGD's cost per epoch, but doubles batch gradient cost
	- **Noise helps escape local minima** — a stationary point of the full-batch loss is generally not stationary for any individual $E_n$
	- Much faster wall-clock convergence for large datasets

### 7.2.4 Mini-batches
- Compromise: compute gradient on a **mini-batch** of $B$ data points
- Gradient estimation error scales as $\sigma/\sqrt{N}$ where $\sigma$ = data standard deviation, $N$ = batch size
	- **Diminishing returns**: increasing batch size 100x only reduces gradient noise 10x
- **Practical considerations**:
	- Powers of 2 batch sizes for hardware efficiency: 64, 128, 256, ...
	- **Shuffle data** between epochs — avoid correlations from data ordering
	- Reshuffle can also help escape local minima
- Algorithm is still called "SGD" even with mini-batches

**Why this matters now: CRITICAL.** Batch size is one of the most important hyperparameters in LLM training:
- **Large-batch training** is the norm for LLMs (batch sizes of millions of tokens) — needed to efficiently utilize GPU clusters
- **Gradient accumulation**: simulate large batches on limited hardware by accumulating gradients over multiple forward passes
- **Critical batch size** (McCandlish et al. 2018): there's an optimal batch size beyond which you get diminishing returns — this is directly from the $\sigma/\sqrt{N}$ scaling
- **Data parallelism**: distribute mini-batches across GPUs, average gradients — the backbone of distributed LLM training
- **Batch size warmup**: start with small batches for noisy exploration, increase for stable convergence — used in many LLM training recipes
- The relationship between batch size and learning rate (linear scaling rule, square root scaling) is a practical engineering concern for every large training run

### 7.2.5 Parameter Initialization
- **Symmetry breaking**: if all weights start identical, all hidden units compute the same thing and receive the same gradient → remain identical forever
	- Must initialize randomly to break symmetry
- Initialize from $\text{Uniform}[-\epsilon, \epsilon]$ or $\mathcal{N}(0, \epsilon^2)$

#### He Initialization (He et al. 2015b)
- For a layer with $M$ inputs and ReLU activation:
	- Pre-activation: $a_i^{(l)} = \sum_{j=1}^{M} w_{ij} z_j^{(l-1)}$, activation: $z_i^{(l)} = \text{ReLU}(a_i^{(l)})$
	- If weights $\sim \mathcal{N}(0, \epsilon^2)$ and previous layer outputs have variance $\lambda^2$:
		- $\mathbb{E}[a_i^{(l)}] = 0$
		- $\text{var}[z_j^{(l)}] = \frac{M}{2}\epsilon^2\lambda^2$ (factor of $1/2$ from ReLU killing negative half)
	- Want variance preserved across layers: set $\frac{M}{2}\epsilon^2 = 1$, giving:
$$\epsilon = \sqrt{\frac{2}{M}}$$
- **Biases**: typically initialized to small positive values for ReLU (ensures pre-activations start positive → non-zero gradients)
- **Transfer learning**: initialize from pre-trained weights instead of random — a separate (and often better) initialization strategy

**Why this matters now: HIGH.** Initialization affects whether training even starts:
- **He init** is the default for ReLU networks and is baked into most frameworks
- **Xavier/Glorot init** ($\epsilon = \sqrt{1/M}$, for sigmoid/tanh) is the other standard option
- For transformers specifically, there are custom initialization schemes:
	- GPT-style: scale residual path weights by $1/\sqrt{N}$ where $N$ = number of layers — prevents signal from growing with depth
	- **muP (Maximal Update Parameterization)** (Yang et al. 2022): a principled initialization + learning rate scheme that allows hyperparameter transfer across model scales — increasingly used at frontier labs
- Bad initialization can cause training to fail completely — one of the first things to check when a training run diverges
- The whole idea of **pre-training** can be viewed as a very good initialization for fine-tuning

---

## 7.3 Convergence

**Relevance: CRITICAL.** Momentum, learning rate schedules, and Adam are the core training algorithms used in every LLM training run.

### The Convergence Problem
- In the quadratic approximation near a minimum, gradient descent along Hessian eigendirection $i$:
$$\alpha_i^{(T)} = (1 - \eta\lambda_i)^T \alpha_i^{(0)}$$
- Converges iff $|1 - \eta\lambda_i| < 1$ for all $i$ → requires $\eta < 2/\lambda_{\max}$
- **Convergence rate governed by condition number**: with $\eta = 2/\lambda_{\max}$, convergence along the slowest direction scales as $(1 - 2\lambda_{\min}/\lambda_{\max})$ per step
	- If $\lambda_{\min} \ll \lambda_{\max}$ (poor conditioning), this is close to 1 → extremely slow convergence
- **The core problem**: gradient descent must use a learning rate small enough for the highest-curvature direction, which makes it painfully slow in low-curvature directions
	- Manifests as oscillation across narrow valleys with slow progress along them

### 7.3.1 Momentum
- Add a "velocity" term that accumulates past gradients:
$$\Delta\mathbf{w}^{(\tau-1)} = -\eta\nabla E(\mathbf{w}^{(\tau-1)}) + \mu\Delta\mathbf{w}^{(\tau-2)}$$
	- $\mu \in [0, 1)$ = **momentum coefficient**, typical value $\mu = 0.9$
- **In low-curvature regions** (consistent gradient direction): momentum terms accumulate, effective learning rate becomes $\eta/(1-\mu)$
	- With $\mu = 0.9$, this is a 10x speedup
- **In high-curvature regions** (oscillating gradients): successive momentum terms cancel out, effective rate stays near $\eta$
	- Oscillations are damped rather than amplified
- **Net effect**: accelerates along consistent gradient directions, dampens oscillations across narrow valleys — exactly what's needed for poorly conditioned problems

#### Nesterov Momentum (Nesterov 2004, Sutskever et al. 2013)
- Modification: evaluate gradient at the **lookahead position** $\mathbf{w}^{(\tau-1)} + \mu\Delta\mathbf{w}^{(\tau-2)}$ instead of current position:
$$\Delta\mathbf{w}^{(\tau-1)} = -\eta\nabla E(\mathbf{w}^{(\tau-1)} + \mu\Delta\mathbf{w}^{(\tau-2)}) + \mu\Delta\mathbf{w}^{(\tau-2)}$$
- Intuition: "look ahead to where momentum will take you, then correct from there"
- Better theoretical convergence for batch GD; less clear-cut advantage for SGD

**Honest take**: Plain momentum with $\mu = 0.9$ is embedded in Adam and basically every optimizer used in practice. Nesterov is occasionally used but not standard in LLM training. The concept of momentum is important to understand because it's a core component of Adam.

### 7.3.2 Learning Rate Schedule
- Using a fixed $\eta$ is suboptimal — want large $\eta$ early (fast exploration) and small $\eta$ late (fine convergence)
- **Common schedules**:
	- **Linear decay**: $\eta^{(\tau)} = (1 - \tau/K)\eta^{(0)} + (\tau/K)\eta^{(K)}$ — ramp from $\eta^{(0)}$ to $\eta^{(K)}$ over $K$ steps
	- **Power law decay**: $\eta^{(\tau)} = \eta^{(0)}(1 + \tau/s)^c$
	- **Exponential decay**: $\eta^{(\tau)} = \eta^{(0)} c^{\tau/s}$
- **Monitor the learning curve** (loss vs. iteration) to verify the schedule is working

**Why this matters now: EXTREMELY HIGH.** Learning rate scheduling is one of the most impactful hyperparameters for LLM training:
- **Cosine annealing** (not in Bishop, but the standard): $\eta^{(\tau)} = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi\tau/T))$ — smoothly decays from max to min following a cosine curve. This is the default for nearly all modern LLM pre-training runs.
- **Warmup**: linearly increase $\eta$ from 0 to $\eta_{\max}$ over the first ~1-5% of training, then decay — critical for stable training with Adam. Without warmup, the early Adam statistics ($s$ and $r$) are unreliable and large learning rates cause divergence.
- **Warmup + cosine decay** is the de facto standard schedule for LLM pre-training (GPT, LLaMA, Mistral, etc.)
- **WSD (Warmup-Stable-Decay)**: warmup → constant → rapid decay. Used in some training recipes. The "chinchilla-optimal" analysis showed the final cooldown phase matters a lot.
- **Learning rate is the single most important hyperparameter** — a 2x error in learning rate can be the difference between a successful training run and a wasted one
- **Learning rate transfer across scales**: if you can predict the optimal LR at large scale from small experiments, you save enormous compute. This is a major motivation for muP.

### 7.3.3 RMSProp and Adam

**Relevance: CRITICAL.** Adam is the default optimizer for LLM training.

#### AdaGrad (Duchi, Hazan, Singer 2011)
- Per-parameter adaptive learning rates based on historical gradient magnitudes:
$$r_i^{(\tau)} = r_i^{(\tau-1)} + \left(\frac{\partial E}{\partial w_i}\right)^2$$
$$w_i^{(\tau)} = w_i^{(\tau-1)} - \frac{\eta}{\sqrt{r_i^{(\tau)} + \delta}} \cdot \frac{\partial E}{\partial w_i}$$
- Parameters with large historical gradients get smaller learning rates, and vice versa
- **Problem**: $r_i$ monotonically increases → learning rate monotonically decreases → training eventually stalls

#### RMSProp (Hinton 2012)
- Fix AdaGrad by using **exponentially weighted moving average** of squared gradients:
$$r_i^{(\tau)} = \beta r_i^{(\tau-1)} + (1-\beta)\left(\frac{\partial E}{\partial w_i}\right)^2$$
$$w_i^{(\tau)} = w_i^{(\tau-1)} - \frac{\eta}{\sqrt{r_i^{(\tau)} + \delta}} \cdot \frac{\partial E}{\partial w_i}$$
- Typical $\beta = 0.9$ — forgets old gradients, so learning rate can recover
- Effectively normalizes each parameter's gradient by its recent RMS magnitude

#### Adam (Kingma and Ba 2014)
- Combines RMSProp (adaptive learning rates) with momentum (gradient smoothing):
- **First moment** (momentum): $s_i^{(\tau)} = \beta_1 s_i^{(\tau-1)} + (1-\beta_1)\frac{\partial E}{\partial w_i}$
- **Second moment** (RMSProp): $r_i^{(\tau)} = \beta_2 r_i^{(\tau-1)} + (1-\beta_2)\left(\frac{\partial E}{\partial w_i}\right)^2$
- **Bias correction** (critical for early steps when $s, r \approx 0$):
$$\hat{s}_i^{(\tau)} = \frac{s_i^{(\tau)}}{1 - \beta_1^\tau}, \quad \hat{r}_i^{(\tau)} = \frac{r_i^{(\tau)}}{1 - \beta_2^\tau}$$
- **Update**: $w_i^{(\tau)} = w_i^{(\tau-1)} - \eta\frac{\hat{s}_i^{(\tau)}}{\sqrt{\hat{r}_i^{(\tau)}} + \delta}$
- **Typical hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$ (but 0.95 increasingly common for LLMs), $\delta = 10^{-8}$
- Bias correction factor $1/(1-\beta^\tau) \to 1$ as $\tau \to \infty$ — only matters in early training

**Why Adam matters now: CRITICAL.** Adam (and its variants) is the optimizer behind virtually every LLM:
- **AdamW** (Loshchilov & Hutter 2019): decoupled weight decay — the actual standard. Adds weight decay directly to the parameter update rather than as L2 regularization in the loss. This is what PyTorch's `AdamW` implements and what every major LLM uses.
- **$\beta_2$ tuning**: for LLMs, $\beta_2 = 0.95$ is common instead of the default 0.999 — shorter memory for the second moment helps with the non-stationary gradient distributions in language modeling (the data distribution shifts as you iterate through the corpus)
- **Gradient clipping**: typically clip the global gradient norm to 1.0 before applying Adam — prevents catastrophic updates from anomalous batches. Not in Bishop but universally used.
- **8-bit Adam**: quantize the optimizer states ($s$ and $r$) to 8-bit to save memory — these states take 2x the memory of the model parameters, so this is a 2x savings on optimizer memory
- **LION, Sophia, Adalayer, Schedule-Free**: recent optimizer research trying to improve on Adam, but none have convincingly dethroned it yet for LLM pre-training
- **Adam's memory cost**: stores $s_i$ and $r_i$ per parameter → 2x parameter count in optimizer states. For a 70B model, that's 140B additional floats. This is why optimizer sharding (ZeRO, FSDP) is essential for large-scale training.

---

## 7.4 Normalization

**Relevance: CRITICAL.** Layer normalization is a non-negotiable component of every transformer. Understanding what it does and why it matters is essential.

### 7.4.1 Data Normalization
- Different input features may span wildly different ranges (e.g., height in meters vs. platelet count in hundreds of thousands)
- Creates elongated error contours → poor conditioning → slow convergence
- **Fix**: standardize each input to zero mean, unit variance before training:
$$\mu_i = \frac{1}{N}\sum_{n=1}^{N} x_{ni}, \quad \sigma_i^2 = \frac{1}{N}\sum_{n=1}^{N}(x_{ni} - \mu_i)^2$$
$$\tilde{x}_{ni} = \frac{x_{ni} - \mu_i}{\sigma_i}$$
- **Critical**: use the same $\mu_i$ and $\sigma_i$ (computed on training set) for validation/test data

**Honest take**: Data normalization is basic hygiene. For most modern DL workflows (especially NLP where inputs are discrete tokens → embeddings), this happens implicitly via the embedding layer. But for any task with continuous inputs (tabular data, scientific ML, RL observations), this is still essential.

### 7.4.2 Batch Normalization (Ioffe and Szegedy 2015)
- **Motivation**: the same scale-mismatch problem from inputs also occurs at hidden layers. As weights update, the distribution of activations shifts, making learning harder for subsequent layers.
- **Vanishing/exploding gradients**: gradient of loss w.r.t. early-layer weights involves a product of Jacobians across all layers:
$$\frac{\partial E}{\partial w_i} = \sum_m \cdots \sum_l \sum_j \frac{\partial z_m^{(1)}}{\partial w_i} \cdots \frac{\partial z_j^{(K)}}{\partial z_l^{(K-1)}} \frac{\partial E}{\partial z_j^{(K)}}$$
	- Product of many terms with magnitude $< 1$ → gradients vanish
	- Product of many terms with magnitude $> 1$ → gradients explode
	- Batch norm keeps activations in a stable range, mitigating both

#### Mechanism
- For each hidden unit $i$ in a layer, normalize the **pre-activations** across the mini-batch of size $K$:
$$\mu_i = \frac{1}{K}\sum_{n=1}^{K} a_{ni}, \quad \sigma_i^2 = \frac{1}{K}\sum_{n=1}^{K}(a_{ni} - \mu_i)^2$$
$$\hat{a}_{ni} = \frac{a_{ni} - \mu_i}{\sqrt{\sigma_i^2 + \delta}}$$
- **Learnable scale and shift**: restore representational capacity with per-unit parameters $\gamma_i$ and $\beta_i$:
$$\tilde{a}_{ni} = \gamma_i \hat{a}_{ni} + \beta_i$$
	- Seems like this could undo the normalization — but the key difference is that $\gamma_i$ and $\beta_i$ are independent learnable parameters, much simpler to optimize than the complex function that determines the un-normalized distribution
	- Can be viewed as an additional differentiable layer inserted after each hidden layer

#### Inference
- At test time, no mini-batch is available for statistics
- Use **moving averages** computed during training:
$$\overline{\mu}_i^{(\tau)} = \alpha\overline{\mu}_i^{(\tau-1)} + (1-\alpha)\mu_i$$
$$\overline{\sigma}_i^{(\tau)} = \alpha\overline{\sigma}_i^{(\tau-1)} + (1-\alpha)\sigma_i$$
- **Train/test discrepancy**: this is a fundamental asymmetry — the model behaves differently at training and inference time

#### Why It Works (Contested)
- **Original motivation**: reduces "internal covariate shift" — the idea that changing weights in earlier layers shifts the input distribution for later layers
- **Santurkar et al. (2018)**: argued covariate shift isn't the key factor; batch norm's main effect is **smoothing the loss landscape**, making optimization easier

**Why this matters / why it's been partially replaced**:
- Batch norm was revolutionary for CNNs (enabled training of very deep networks like ResNet)
- **Problems with batch norm**:
	- Statistics depend on other examples in the mini-batch → weird coupling between examples
	- Breaks down with small batch sizes (noisy statistics)
	- Doesn't work well for variable-length sequences (common in NLP)
	- Distributed training: need to synchronize batch statistics across GPUs or accept noisy per-GPU statistics
	- Train/test behavior mismatch causes subtle bugs
- **Result**: batch norm is still standard in vision (ConvNets), but **layer norm has replaced it** in transformers/LLMs

### 7.4.3 Layer Normalization (Ba, Kiros, Hinton 2016)
- Instead of normalizing across the mini-batch (batch norm), normalize across the **hidden units** for each data point separately:
$$\mu_n = \frac{1}{M}\sum_{i=1}^{M} a_{ni}, \quad \sigma_n^2 = \frac{1}{M}\sum_{i=1}^{M}(a_{ni} - \mu_n)^2$$
$$\hat{a}_{ni} = \frac{a_{ni} - \mu_n}{\sqrt{\sigma_n^2 + \delta}}$$
- Same learnable $\gamma_i$ and $\beta_i$ parameters as batch norm
- **Key differences from batch norm**:
	- Statistics computed per-example, not per-batch → no coupling between examples in a mini-batch
	- **Same computation at train and test time** — no moving averages needed
	- Works with **any batch size**, including batch size 1
	- Natural fit for **variable-length sequences**: each token normalized independently
- Originally introduced for RNNs where batch norm was impractical (sequence lengths vary across examples)

**Why this matters now: EXTREMELY HIGH.** Layer norm is *the* normalization for transformers:
- **Every transformer uses layer norm** — GPT, BERT, LLaMA, all of them. It's in every attention block and every MLP block.
- **Pre-norm vs. post-norm**: original transformer (Vaswani 2017) applied layer norm *after* the residual connection (post-norm). GPT-2 and most modern models apply it *before* (pre-norm), which stabilizes training and allows deeper models.
- **RMSNorm** (Zhang & Sennrich 2019): a simplified variant that only divides by the RMS (no mean subtraction): $\hat{a}_i = a_i / \sqrt{\frac{1}{M}\sum_j a_j^2 + \delta}$. Used in LLaMA, Mistral, and most modern LLMs. Slightly faster and works just as well — removing the mean centering has minimal effect.
- **DeepNorm** (Wang et al. 2022): modification of post-norm for training very deep transformers (1000+ layers). Scales residual connections by a factor $\alpha$ dependent on depth.
- **QK-norm**: normalizing query and key vectors before the attention dot product — helps prevent attention logit growth in very deep or long-context models
- Layer norm is also a key component in **understanding transformer internals** — the "residual stream" view of transformers treats each layer as reading from and writing to a normalized stream of hidden states

---

## Summary: What to Prioritize

### Must know cold (used in every training run):
1. **Adam / AdamW** — the optimizer; understand the first/second moment mechanics, bias correction, and how it provides per-parameter adaptive learning rates
2. **Learning rate schedules** — warmup + cosine decay is the standard; learning rate is the most important hyperparameter
3. **Layer normalization** — core transformer component; understand Pre-norm vs Post-norm, RMSNorm
4. **Mini-batch SGD** — the computational backbone; batch size selection, gradient accumulation, data parallelism
5. **He initialization** — default for ReLU networks; understand why variance preservation matters across layers

### Important conceptual foundations:
6. **Momentum** — embedded in Adam, explains the first moment
7. **Condition number / Hessian eigenstructure** — explains why adaptive methods are needed
8. **Vanishing/exploding gradients** — the problem that normalization and careful initialization solve
9. **Batch normalization** — still used in vision; understanding the contrast with layer norm is useful

### Less directly used but good to know:
10. **AdaGrad** — historical ancestor of RMSProp → Adam
11. **Nesterov momentum** — minor variant, occasionally used
12. **Batch gradient descent** — nobody uses this in practice, but useful to understand the SGD → mini-batch spectrum

### Key things Bishop doesn't cover that you should also know:
- **AdamW** (decoupled weight decay) — the actual optimizer used in practice
- **Cosine annealing** — the dominant schedule
- **Gradient clipping** — universally used, prevents training instabilities
- **RMSNorm** — the layer norm variant most LLMs actually use
- **muP** — principled hyperparameter transfer across scales
- **Gradient accumulation** — essential for simulating large batches on limited hardware
- **Mixed precision training** (fp16/bf16) — not an optimizer, but deeply interacts with gradient scaling and normalization
