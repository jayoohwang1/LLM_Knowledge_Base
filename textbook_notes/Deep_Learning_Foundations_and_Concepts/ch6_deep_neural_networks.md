# Chapter 6: Deep Neural Networks

> **Overall relevance to modern LLM research: HIGH.** This chapter covers the foundational building blocks — activation functions, loss functions, representation learning, transfer learning, contrastive learning — that literally everything in modern deep learning is built on. Some sections are "you need to know this cold" and others are more "historically interesting but you won't use this directly." I'll be blunt about which is which.

---

## 6.1 Limitations of Fixed Basis Functions

**Relevance: Medium.** You won't directly use fixed basis functions anywhere in modern practice, but understanding *why* they fail is the conceptual origin story for why we use neural nets at all. Good for building intuition.

### 6.1.1 The Curse of Dimensionality
- **Core problem**: for a polynomial of order $M$ in $D$ dimensions, number of coefficients grows as $\mathcal{O}(D^M)$ — combinatorial explosion
- Grid-based classifiers similarly blow up — dividing input space into regular cells, the number of cells grows exponentially with $D$
	- Most cells end up empty, can't classify test points that fall in them
	- Fundamentally: fixed basis functions chosen independently of the problem can't scale to even moderately high dimensions
- **Bellman (1961)** coined the term — applies to polynomial regression, classification, basically everything with fixed pre-defined features

**Why this matters now**: The curse of dimensionality is why feature engineering died and representation learning won. Every time someone asks "why can't we just use hand-crafted features?" this is the answer. It's also the deep reason behind why scaling laws work — neural nets learn to exploit manifold structure that fixed methods can't.

### 6.1.2 High-Dimensional Spaces
- **Hypersphere volume concentration**: fraction of volume in thin shell $r \in [1-\epsilon, 1]$ is $1 - (1-\epsilon)^D$ — approaches 1 for large $D$ even with tiny $\epsilon$
	- In high-D, almost all the volume of a hypersphere is concentrated near the surface
- **Gaussian mass concentration**: probability mass of a Gaussian in $D$ dimensions concentrates in a thin shell at radius $\hat{r} \simeq \sqrt{D}\sigma$
	- The mode (at origin) is far from where the mass actually lives
	- This is deeply counterintuitive if you think in 2D/3D
- **Upside of high dimensions**: data that's inseparable in low-D can become linearly separable in higher-D (the kernel trick intuition, also why random projections work)

**Honest take**: This is one of those things that's useful to internalize once and then you never explicitly reason about it again. But it does come up when thinking about why cosine similarity works well in high-D embedding spaces (everything's on a shell), and why nearest neighbor search is hard at scale.

### 6.1.3 Data Manifolds
- **Key insight**: real data lives on low-dimensional manifolds embedded in high-D space
	- Images of a digit with varying position/orientation = 3D manifold in pixel space
	- The manifold is highly nonlinear
- Number of required basis functions should scale with **manifold dimensionality**, not ambient dimensionality — massive savings
- Natural images occupy a tiny fraction of pixel space — random pixel images look nothing like real images
	- Adjacent pixels in natural images are strongly correlated
	- This is why conv nets and diffusion models work — they exploit spatial structure on the image manifold

**Why this matters now: VERY HIGH.** The manifold hypothesis is arguably the single most important conceptual foundation for modern deep learning. It explains:
- Why autoencoders and VAEs learn compressed representations
- Why diffusion models can generate realistic images by learning the score function on the data manifold
- Why LLM embeddings cluster semantically — text lives on a low-D meaning manifold
- Why retrieval-augmented generation works — you can do nearest neighbor search in embedding space because similar meanings are nearby on the manifold

### 6.1.4 Data-Dependent Basis Functions
- **History**: hand-crafted features (domain-specific basis functions) were the mainstream ML approach for decades
	- Superseded by learning features from data — the deep learning revolution
- **Radial basis functions (RBFs)**: one basis function per data point, $\phi_n(\mathbf{x}) = \exp(-\|\mathbf{x} - \mathbf{x}_n\|^2 / s^2)$
	- Computationally unwieldy for large datasets, needs heavy regularization
- **SVMs**: select a subset of data-centered basis functions automatically during training
	- Effective number of basis functions still grows with dataset size
	- Don't produce probabilistic outputs, don't naturally handle multiclass
	- **Superseded by deep nets** which scale much better with large datasets and learn hierarchical representations

**Honest take**: RBFs and SVMs are historically important but essentially dead in modern practice. The only remnant: the conceptual idea that your "features" should be adapted to data rather than fixed. That idea won — it's just that neural nets are the mechanism now.

---

## 6.2 Multilayer Networks

**Relevance: CRITICAL.** This is the literal architecture of everything. Even though modern architectures are far more sophisticated, the core computational pattern — linear transform, nonlinear activation, repeat — is unchanged.

### 6.2.1 Parameter Matrices
- **Two-layer network**: input $\rightarrow$ hidden $\rightarrow$ output
	- First layer: $a_j^{(1)} = \sum_{i=0}^{D} w_{ji}^{(1)} x_i$ (linear combination + bias)
	- Activation: $z_j^{(1)} = h(a_j^{(1)})$ (nonlinear transformation)
	- Second layer: $a_k^{(2)} = \sum_{j=0}^{M} w_{kj}^{(2)} z_j^{(1)}$
	- Output activation: $y_k = f(a_k^{(2)})$
- Compact matrix form: $\mathbf{y}(\mathbf{x}, \mathbf{w}) = f(\mathbf{W}^{(2)} h(\mathbf{W}^{(1)} \mathbf{x}))$
	- Bias absorbed into weight matrices via augmented input $x_0 = 1$
- $f(\cdot)$ and $h(\cdot)$ applied elementwise

**Why this matters now**: This exact pattern — matmul, nonlinearity, matmul, nonlinearity — is the computational backbone of every transformer layer. The MLP blocks in transformers are literally this. Understanding it in its simplest form helps you reason about what each component contributes.

### 6.2.2 Universal Approximation
- **Theorem** (Funahashi 1989, Cybenko 1989, Hornik et al. 1989, Leshno et al. 1993): a two-layer network with a wide range of activation functions can approximate any continuous function on a compact subset of $\mathbb{R}^D$ to arbitrary accuracy
	- Also holds for functions between finite-dimensional discrete spaces
- **Caveats** — the theorem only says such a network *exists*:
	- May require exponentially many hidden units
	- Says nothing about whether gradient descent can *find* it
	- No free lunch theorem: no universal learning algorithm
- **Depth wins over width**: Montufar et al. (2014) showed deep nets divide input space into regions that grow exponentially with depth but only polynomially with width
	- This is the theoretical justification for going deep

**Honest take**: Universal approximation is one of those results that's reassuring but not practically useful. Nobody designs architectures based on it. The depth-vs-width result is much more actionable — it's part of why we stack 100+ layers in modern models rather than making 2-layer networks with billions of hidden units. But even that's oversimplified: in practice, the right architecture (attention, convolution, etc.) matters way more than raw depth.

### 6.2.3 Hidden Unit Activation Functions
- **Linear activations are useless**: composition of linear transforms = single linear transform. A network of linear layers has no more representational power than one layer.
	- Exception: **bottleneck networks** with fewer hidden units than inputs/outputs → implements PCA (rank-deficient linear mapping)

#### Sigmoid / Tanh (historical)
- **Logistic sigmoid**: $\sigma(a) = \frac{1}{1+\exp(-a)}$, output in $(0,1)$
	- Inspired by biological neurons
	- Equivalent to tanh up to linear rescaling
- **Tanh**: $\tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}}$, output in $(-1, 1)$
	- Hard tanh: $h(a) = \max(-1, \min(1, a))$ — piecewise linear approximation
- **Fatal flaw**: gradients vanish exponentially for large $|a|$ — kills training of deep networks

**Honest take**: Sigmoid/tanh are mostly historical now. You'll still see sigmoid as an output gate (e.g., in LSTMs, gating mechanisms in MoE routers) and tanh occasionally in specific architectural choices, but they're not used as hidden activations in any modern architecture I'm aware of.

#### ReLU and Variants (this is what you actually use)
- **ReLU**: $h(a) = \max(0, a)$
	- Simple, computationally cheap (just a comparison), works with low-precision arithmetic
	- Much less sensitive to weight initialization than sigmoid/tanh
	- Enabled training of genuinely deep networks (Krizhevsky, Sutskever, Hinton 2012 — AlexNet)
	- **Dead neurons**: units with $a < 0$ for all inputs get zero gradient, permanently inactive
- **Leaky ReLU**: $h(a) = \max(0, a) + \alpha \min(0, a)$, $0 < \alpha < 1$
	- Nonzero gradient for negative inputs — fixes dead neuron problem
	- Variant: $\alpha = -1$ gives $h(a) = |a|$ (absolute value activation)
	- Parametric ReLU: learn $\alpha$ per-unit during training
- **Softplus**: $h(a) = \ln(1 + \exp(a))$ — smooth approximation to ReLU
	- Non-zero gradient everywhere but more expensive to compute

**Why this matters now: CRITICAL.** ReLU is still the default in most MLPs. But the frontier has moved:
- **GeLU** (Gaussian Error Linear Unit) is the dominant activation in transformers (GPT, BERT, etc.) — not mentioned in Bishop but basically a smooth ReLU
- **SwiGLU** (Swish-Gated Linear Unit) is what most modern LLMs actually use (LLaMA, PaLM, etc.) — a gated variant that multiplies two linear projections with a Swish activation
- **SiLU/Swish**: $h(x) = x \cdot \sigma(\beta x)$ — mentioned in exercises (6.5) — became ReLU's main competitor
- The trend is toward smooth, gated activations over hard ReLU, but the core insight (simple nonlinearities + depth = expressiveness) is unchanged

### 6.2.4 Weight-Space Symmetries
- Multiple distinct weight vectors map to the same input-output function
- **Sign-flip symmetry**: for tanh (odd function), flipping signs of all weights into and out of a hidden unit preserves the mapping → $2^M$ equivalent weight vectors for $M$ hidden units
- **Permutation symmetry**: reordering hidden units gives the same function → $M!$ equivalent orderings
- Total symmetry factor: $M! \cdot 2^M$ per layer (for tanh; ReLU has different symmetries)
- **Practical consequence**: mostly irrelevant for SGD-based training, but matters for Bayesian approaches where you integrate over weight space

**Honest take**: This is a fun theoretical curiosity that almost never matters in practice. The one place it shows up: when people try to do model merging or weight averaging (e.g., model soups, federated learning), permutation symmetry means you can't just average weights naively — you need to align the neurons first. There's been recent work on this (e.g., Git Re-Basin, Ainsworth et al. 2022).

---

## 6.3 Deep Networks

**Relevance: CRITICAL.** This section covers the conceptual foundations that make deep learning actually work: hierarchical representations, transfer learning, and contrastive learning. These are the ideas, not just the math.

### General Deep Architecture
- Extend to $L$ layers: $\mathbf{z}^{(l)} = h^{(l)}(\mathbf{W}^{(l)} \mathbf{z}^{(l-1)})$ with $\mathbf{z}^{(0)} = \mathbf{x}$, $\mathbf{z}^{(L)} = \mathbf{y}$
- **Depth advantage** (Montufar et al. 2014): number of linear regions grows exponentially with depth, polynomially with width
	- Same function that takes exponentially many units in 2 layers can be represented by a polynomially-sized deep network

### 6.3.1 Hierarchical Representations
- Deep nets learn **compositional hierarchies**: early layers detect simple features (edges), later layers combine them into complex concepts (eyes → faces → identities)
- This mirrors the structure of many real-world problems — objects are composed of parts, which are composed of sub-parts
- Can think of it as **compositional inductive bias**: higher-level objects = compositions of lower-level objects
	- Exponential gain: at each stage, many ways to combine components from previous stage

**Why this matters now: CRITICAL.** This is the entire thesis of deep learning. In LLMs specifically:
- Early layers learn token-level syntax patterns
- Middle layers learn semantic relationships and factual associations
- Late layers learn task-specific reasoning patterns
- This hierarchical structure is what mechanistic interpretability research is trying to reverse-engineer (circuits, induction heads, etc.)
- It's also why techniques like layer freezing, progressive training, and LoRA (targeting specific layers) work — different layers encode different levels of abstraction

### 6.3.2 Distributed Representations
- Each hidden unit = a "feature"; $M$ binary units can represent $2^M$ distinct combinations
	- Exponentially more expressive than local/one-hot representations
- Example: face recognition network with 8 binary features (glasses, hat, beard...) → 256 combinations from just 8 units
	- In practice, features are continuous and even more expressive
- Networks can learn that features combine independently (compositional generalization)

**Why this matters now: HIGH.** Distributed representations are literally what embeddings are. Every time you use word2vec, BERT embeddings, or LLM hidden states, you're working with distributed representations. The key insight — that a 768-dimensional vector can represent an astronomically large number of concepts through combinations of features — is why embeddings work so well for search, clustering, and transfer.

### 6.3.3 Representation Learning
- Neural nets learn **transformations** of data that make downstream tasks easier
	- Final hidden layer = learned representation where classes are linearly separable
	- The learned representation is the **embedding space**
- **Unsupervised representation learning**: exploit unlabeled data
	- Autoencoders: learn compressed representation by reconstructing input
	- Historically: unsupervised pre-training enabled first successful deep networks (before batch norm, residual connections, etc.)
	- Now: unsupervised pre-training is the dominant paradigm (LLM pre-training on text, DINO/MAE for images)
- Pre-training + supervised fine-tuning became the standard recipe
	- Later discovered that with proper techniques (batch norm, residual connections, good initialization), supervised training from scratch also works
	- But pre-training on large unlabeled data still gives better representations

**Why this matters now: EXTREMELY HIGH.** This is arguably the single most important idea in modern AI:
- **LLM pre-training** = learning representations of language by predicting next tokens on massive text corpora
- **Foundation models** = the idea that one big pre-trained representation can serve many downstream tasks
- **Embedding APIs** (OpenAI, Cohere, etc.) are literally selling learned representations
- The quality of the learned representation is what separates good models from great ones — it's the whole game

### 6.3.4 Transfer Learning
- Train on task B (data-rich) → transfer representations to task A (data-scarce)
	- Requirements: same input modality, shared low-level features, enough commonality
- **Pre-training**: learn general representations on large dataset
	- Early layers learn transferable features (edges, textures in vision; syntax, common phrases in language)
	- Later layers become task-specific
- **Fine-tuning**: adapt pre-trained network to new task
	- Full fine-tuning: update all parameters with small learning rate
	- Feature extraction: freeze early layers, only retrain final layers
	- Efficiency trick: forward pass through frozen pre-trained network once, then train small network on cached representations
- **Multitask learning** (Caruana 1997): jointly train on multiple related tasks
	- Shared early layers, task-specific heads
	- Each task benefits from data of other tasks
- **Meta-learning / learning to learn**: learn to generalize to entirely new tasks
	- Few-shot learning: classify new classes from very few examples
	- One-shot learning: single example per class

**Why this matters now: EXTREMELY HIGH.** Transfer learning is the backbone of modern AI deployment:
- **Fine-tuning LLMs**: SFT, RLHF, DPO — all forms of transfer learning from pre-trained base models
- **Parameter-efficient fine-tuning**: LoRA, QLoRA, adapters — fine-tuning only a tiny fraction of parameters, made possible because the pre-trained representations are already good
- **In-context learning**: arguably a form of implicit transfer/meta-learning where the model adapts to new tasks from the prompt alone
- **Multitask learning**: instruction tuning trains on many tasks simultaneously — exactly this idea at scale
- The entire foundation model paradigm (train once, adapt many times) is transfer learning taken to its logical conclusion

### 6.3.5 Contrastive Learning
- Learn representations by pulling **positive pairs** close and pushing **negative pairs** apart in embedding space
- Unlike supervised learning: loss is defined between pairs of inputs, not per-input labels
- Uses earlier layer activations as the embedding, not the final output

#### InfoNCE Loss
$$E(\mathbf{w}) = -\ln \frac{\exp\{\mathbf{f_w}(\mathbf{x})^\top \mathbf{f_w}(\mathbf{x}^+)\}}{\exp\{\mathbf{f_w}(\mathbf{x})^\top \mathbf{f_w}(\mathbf{x}^+)\} + \sum_{n=1}^{N} \exp\{\mathbf{f_w}(\mathbf{x})^\top \mathbf{f_w}(\mathbf{x}_n^-)\}}$$
- Resembles cross-entropy: positive pair cosine similarity = logit for correct class, negative pair similarities = logits for incorrect classes
- Representations normalized to unit hypersphere: $\|\mathbf{f_w}(\mathbf{x})\| = 1$
- Negative pairs are crucial — without them, trivial solution maps everything to same point

#### Flavors of Contrastive Learning
- **Instance discrimination (self-supervised)**: positive pair = anchor + augmented version of same image
	- Augmentations: rotation, cropping, color jitter, etc.
	- No labels needed — pure self-supervised
- **Supervised contrastive learning** (Khosla et al. 2020): positive pair = two images of same class
	- Avoids treating semantically similar images as negatives
	- Often outperforms cross-entropy classification
- **Cross-modal contrastive learning (CLIP)** (Radford et al. 2021): positive pair = image + its text caption
	- Two separate encoders: $\mathbf{f_w}$ for images, $\mathbf{g_\theta}$ for text
	- Loss is symmetric — image should be close to its caption AND caption close to its image
	- Weakly supervised: scraped image-caption pairs from internet
	- Loss: $$E(\mathbf{w}) = -\frac{1}{2}\ln \frac{\exp\{\mathbf{f_w}(\mathbf{x}^+)^\top \mathbf{g_\theta}(\mathbf{y}^+)\}}{\exp\{\mathbf{f_w}(\mathbf{x}^+)^\top \mathbf{g_\theta}(\mathbf{y}^+)\} + \sum_n \exp\{\mathbf{f_w}(\mathbf{x}_n^-)^\top \mathbf{g_\theta}(\mathbf{y}^+)\}} - \frac{1}{2}\ln \frac{\exp\{\mathbf{f_w}(\mathbf{x}^+)^\top \mathbf{g_\theta}(\mathbf{y}^+)\}}{\exp\{\mathbf{f_w}(\mathbf{x}^+)^\top \mathbf{g_\theta}(\mathbf{y}^+)\} + \sum_m \exp\{\mathbf{f_w}(\mathbf{x}^+)^\top \mathbf{g_\theta}(\mathbf{y}_m^-)\}}$$
	- Generalizes to any paired modalities

**Why this matters now: EXTREMELY HIGH.** Contrastive learning is everywhere:
- **CLIP** is foundational to multimodal AI — used in DALL-E, Stable Diffusion (as the text encoder), GPT-4V's vision capabilities, and basically every vision-language model
- **Embedding models** for RAG (retrieval-augmented generation) are trained with contrastive objectives — you're literally using InfoNCE every time you do semantic search
- **DPO (Direct Preference Optimization)** for RLHF can be seen as a contrastive objective over preferred/dispreferred completions
- **SimCLR, DINO, BYOL** for self-supervised vision pre-training — compete with or beat supervised pre-training
- **Sentence-BERT, E5, GTE** — the text embedding models powering every RAG pipeline are all trained with contrastive losses
- Hard negative mining, in-batch negatives, temperature scaling — these practical details of contrastive training are critical engineering knowledge

### 6.3.6 General Network Architectures
- Networks don't have to be sequential stacks — any directed acyclic graph (DAG) of units works
- Each unit computes: $z_k = h\left(\sum_{j \in \mathcal{A}(k)} w_{kj} z_j + b_k\right)$ where $\mathcal{A}(k)$ = ancestors of node $k$
- Must be **feed-forward** (no cycles) for deterministic outputs
	- Recurrent networks have cycles — covered later

**Honest take**: This is setting up for the fact that modern architectures (ResNets with skip connections, transformers with attention connecting all positions, U-Nets, etc.) are all DAGs that don't look like simple stacks. The key constraint is acyclicity for feed-forward; recurrence relaxes this.

### 6.3.7 Tensors
- Data and parameters naturally represented as multi-dimensional arrays (tensors)
	- Image batch: 4D tensor $x_{ijkn}$ (row, column, channel, batch)
	- Scalars, vectors, matrices are special cases
- GPUs are optimized for tensor operations — this is why deep learning took off when GPU computing became accessible

**Honest take**: Tensors are just the data structure. You need to be fluent in thinking about shapes and broadcasting (torch.einsum is your friend), but the concept itself is straightforward.

---

## 6.4 Error Functions

**Relevance: HIGH.** You need to know these cold — they're the objectives that every model optimizes.

### 6.4.1 Regression
- Assume Gaussian noise: $p(t|\mathbf{x}, \mathbf{w}) = \mathcal{N}(t | y(\mathbf{x}, \mathbf{w}), \sigma^2)$
- **Negative log-likelihood → sum-of-squares error**: $$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^{N}\{y(\mathbf{x}_n, \mathbf{w}) - t_n\}^2$$
- Output activation = **identity** (linear output), so $y_k = a_k$
- Gradient has the clean form: $\frac{\partial E}{\partial a_k} = y_k - t_k$
- Noise variance estimated post-training: $\sigma^{2*} = \frac{1}{N}\sum_n \{y(\mathbf{x}_n, \mathbf{w}^*) - t_n\}^2$
- Multiple outputs: assume conditionally independent targets → sum-of-squares over all outputs, noise variance $\sigma^{2*} = \frac{1}{NK}\sum_n \|\mathbf{y}(\mathbf{x}_n, \mathbf{w}^*) - \mathbf{t}_n\|^2$

### 6.4.2 Binary Classification
- Output activation = **logistic sigmoid**: $y(\mathbf{x}, \mathbf{w}) = p(\mathcal{C}_1 | \mathbf{x})$
- **Negative log-likelihood → cross-entropy error**: $$E(\mathbf{w}) = -\sum_{n=1}^{N}\{t_n \ln y_n + (1-t_n)\ln(1-y_n)\}$$
- Cross-entropy trains faster and generalizes better than sum-of-squares for classification (Simard et al. 2003)
- Label noise: can extend to handle probability $\epsilon$ of incorrect labels
- Multiple independent binary classifications: $K$ sigmoid outputs, sum of $K$ binary cross-entropies

### 6.4.3 Multiclass Classification
- Output activation = **softmax**: $y_k(\mathbf{x}, \mathbf{w}) = \frac{\exp(a_k(\mathbf{x}, \mathbf{w}))}{\sum_j \exp(a_j(\mathbf{x}, \mathbf{w}))}$
	- Outputs sum to 1, all non-negative — valid probability distribution
	- Invariant to adding constant to all pre-activations → degeneracy removed by regularization
- **Negative log-likelihood → multiclass cross-entropy**: $$E(\mathbf{w}) = -\sum_{n=1}^{N}\sum_{k=1}^{K} t_{kn} \ln y_k(\mathbf{x}_n, \mathbf{w})$$
- Gradient: $\frac{\partial E}{\partial a_k} = y_k - t_k$ — same elegant form as regression and binary classification

**The pairing principle**: there's a natural canonical link between output activation and loss function:

| Task | Output Activation | Loss Function |
|------|------------------|---------------|
| Regression | Identity (linear) | Sum-of-squares (MSE) |
| Binary classification | Sigmoid | Binary cross-entropy |
| Multiclass classification | Softmax | Categorical cross-entropy |

**Why this matters now: CRITICAL.**
- **LLM pre-training** uses softmax + cross-entropy over the vocabulary — this is literally the next-token prediction objective
- **Logit lens / tuned lens** interpretability techniques work by applying the final softmax to intermediate layer representations — understanding the output layer is essential
- **Temperature scaling** in LLM sampling directly manipulates the softmax: $y_k \propto \exp(a_k / T)$
- **Label smoothing** (widely used in LLM training) modifies the target distribution away from hard one-hot — a regularization technique on the cross-entropy loss
- **Distillation** uses soft targets (teacher's softmax outputs) as training signal — KL divergence between teacher and student softmax distributions
- Understanding that MSE and cross-entropy come from different distributional assumptions (Gaussian vs. Bernoulli/categorical) helps you choose the right loss for novel tasks

---

## 6.5 Mixture Density Networks

**Relevance: Medium-Low for most LLM work, but conceptually interesting and has niche applications.**

### Core Problem: Multimodal Conditional Distributions
- Standard regression assumes unimodal Gaussian $p(t|\mathbf{x})$ — fails badly for **inverse problems** where mapping is one-to-many
	- Robot kinematics example: two joint angle solutions ("elbow up" / "elbow down") for same end-effector position
	- Least-squares regression averages the modes → predicts the mean, which may not correspond to any valid solution
- Many real-world problems are multimodal: multiple valid outputs for same input

### 6.5.1 Conditional Mixture Distributions
- Model $p(\mathbf{t}|\mathbf{x})$ as a mixture of Gaussians with input-dependent parameters:
$$p(\mathbf{t}|\mathbf{x}) = \sum_{k=1}^{K} \pi_k(\mathbf{x}) \mathcal{N}(\mathbf{t} | \boldsymbol{\mu}_k(\mathbf{x}), \sigma_k^2(\mathbf{x}))$$
- **Heteroscedastic**: variance depends on input — more general than fixed-variance models
- All mixture parameters ($\pi_k$, $\boldsymbol{\mu}_k$, $\sigma_k^2$) are outputs of the neural network
- Network outputs for $K$ components with $L$-dim targets:
	- Mixing coefficients: $K$ outputs through **softmax** (sum to 1)
	- Variances: $K$ outputs through **exp** (positive)
	- Means: $K \times L$ outputs through **identity** (unconstrained)
	- Total: $(L+2)K$ outputs vs usual $L$
- Related to but distinct from **mixture-of-experts (MoE)**: MoE has independent expert networks; MDN shares the backbone and only the output heads differ

### 6.5.2 Gradient Optimization
- Error function: $E(\mathbf{w}) = -\sum_n \ln\left\{\sum_k \pi_k(\mathbf{x}_n, \mathbf{w}) \mathcal{N}(\mathbf{t}_n | \boldsymbol{\mu}_k(\mathbf{x}_n, \mathbf{w}), \sigma_k^2(\mathbf{x}_n, \mathbf{w}))\right\}$
- **Posterior responsibilities**: $\gamma_{nk} = \frac{\pi_k \mathcal{N}_{nk}}{\sum_l \pi_l \mathcal{N}_{nl}}$ — soft assignment of each data point to mixture components
- Gradients w.r.t. output pre-activations:
	- Mixing coefficients: $\frac{\partial E_n}{\partial a_k^\pi} = \pi_k - \gamma_{nk}$
	- Means: $\frac{\partial E_n}{\partial a_{kl}^\mu} = \gamma_{nk} \left\{\frac{\mu_{kl} - t_{nl}}{\sigma_k^2}\right\}$
	- Variances: $\frac{\partial E_n}{\partial a_k^\sigma} = \gamma_{nk}\left\{L - \frac{\|\mathbf{t}_n - \boldsymbol{\mu}_k\|^2}{\sigma_k^2}\right\}$

### 6.5.3 Predictive Distribution
- **Conditional mean**: $\mathbb{E}[\mathbf{t}|\mathbf{x}] = \sum_k \pi_k(\mathbf{x}) \boldsymbol{\mu}_k(\mathbf{x})$ — reduces to standard regression if $K=1$
	- For multimodal distributions, the mean may not be a useful prediction (falls between modes)
- **Conditional mode**: take the mean of the most probable component $\arg\max_k \pi_k(\mathbf{x})$ — often more useful for multimodal problems
	- No closed-form; requires numerical iteration or argmax over components
- **Conditional variance**: $s^2(\mathbf{x}) = \sum_k \pi_k(\mathbf{x})\left\{\sigma_k^2(\mathbf{x}) + \|\boldsymbol{\mu}_k(\mathbf{x}) - \sum_l \pi_l(\mathbf{x})\boldsymbol{\mu}_l(\mathbf{x})\|^2\right\}$
	- Captures both within-component variance and between-component variance

**Why this matters now**: MDNs are a niche tool, but the concepts reappear:
- **Mixture-of-Experts (MoE)** in modern LLMs (Mixtral, Switch Transformer) shares the gating/routing idea — softmax over experts is exactly the mixing coefficient mechanism
- **Diffusion models** effectively learn a complex multimodal distribution, solving the same fundamental problem MDNs address but much more flexibly
- **Multi-token prediction** objectives (Meta's recent work) output multiple possible next tokens — conceptually similar to predicting a mixture
- The idea of **heteroscedastic models** (input-dependent uncertainty) shows up in calibration research and Bayesian deep learning
- **Flow matching** and other generative approaches handle multimodality better than MDNs, which is partly why MDNs aren't widely used — but the problem they identify (regression can't handle multimodality) is still very real

---

## Summary: What to Prioritize

### Concepts to internalize deeply (you'll use these every day):
1. **Representation learning** — the whole foundation model paradigm
2. **Contrastive learning / InfoNCE** — embedding models, CLIP, RAG
3. **Softmax + cross-entropy** — LLM training objective
4. **Transfer learning / fine-tuning** — SFT, LoRA, adaptation
5. **Hierarchical representations** — interpretability, layer-wise analysis
6. **ReLU and its descendants** — architecture design
7. **Manifold hypothesis** — why embeddings and generative models work

### Concepts worth knowing but less directly used:
8. **Universal approximation** — theoretical reassurance, depth-vs-width
9. **Mixture density networks** — conceptual ancestor of MoE routing
10. **Curse of dimensionality** — foundational intuition

### Concepts that are mostly historical:
11. **Fixed basis functions / RBFs / SVMs** — superseded by deep learning
12. **Weight-space symmetries** — theoretical curiosity (except for model merging)
13. **Sigmoid/tanh as hidden activations** — replaced by ReLU/GeLU/SwiGLU
