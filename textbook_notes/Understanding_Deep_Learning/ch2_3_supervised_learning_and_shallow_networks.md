# Ch 2–3: Supervised Learning & Shallow Neural Networks

> **Relevance verdict**: These are foundational chapters. Nothing here is novel if you've trained models before, but the formalism is worth internalizing because it's the exact same framework that scales up to LLM pretraining. Chapter 2 is a quick pass; Chapter 3 has more meat, especially the universal approximation theorem and the piecewise-linear geometry of ReLU networks.

---

## Chapter 2 — Supervised Learning

### 2.1 Supervised Learning Overview

- **Core abstraction**: model $\mathbf{y} = \mathbf{f}[\mathbf{x}, \boldsymbol{\phi}]$ maps inputs to outputs, parameterized by $\boldsymbol{\phi}$
	- The equation defines a *family* of functions; $\boldsymbol{\phi}$ selects one member
	- Computing $\mathbf{y}$ from $\mathbf{x}$ = **inference**
	- Finding $\boldsymbol{\phi}$ from data = **training** / **learning**
- **Training data**: $I$ pairs $\{\mathbf{x}_i, \mathbf{y}_i\}$ — assumed structured/tabular (fixed-size, fixed-order vectors)
- **Loss function** $L[\boldsymbol{\phi}]$: scalar measuring mismatch between predictions and targets
	- Training = finding $\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\text{argmin}} \left[ L[\boldsymbol{\phi}] \right]$
- **Generalization**: evaluate on held-out **test data** to check if model learned the real relationship vs. just memorized training quirks

> **Why this matters for LLMs**: Next-token prediction is exactly this framework. $\mathbf{x}$ = token sequence so far, $\mathbf{y}$ = probability distribution over next token, $\boldsymbol{\phi}$ = all transformer weights, $L$ = cross-entropy loss. The entire pretraining pipeline is just supervised learning on (context, next_token) pairs scraped from the internet. Fine-tuning (SFT) is the same thing with curated pairs. Even RLHF starts from this base and adds a reward signal on top.

### 2.2 Linear Regression Example

- **1D linear model**: $y = \phi_0 + \phi_1 x$ — two params (intercept + slope)
- **Least-squares loss**: $L[\boldsymbol{\phi}] = \sum_{i=1}^{I} (\phi_0 + \phi_1 x_i - y_i)^2$
	- Squaring makes direction of deviation irrelevant
	- Loss surface is a function of $\boldsymbol{\phi}$ — visualizable as a 2D surface for this simple case
	- Optimal params sit at the minimum of this surface

### 2.2.3–2.2.4 Training & Testing

- **Training**: initialize params randomly, iteratively "walk downhill" on loss surface via gradient descent
	- Linear regression has a closed-form solution (normal equations), but gradient descent is introduced because it generalizes to all complex models where no closed form exists
- **Testing**: evaluate on separate test set
	- **Underfitting**: model too simple to capture true relationship (high bias)
	- **Overfitting**: model memorizes training noise, fails on new data (high variance)

> **Practical relevance**: At scale, you almost never think about closed-form solutions. Everything is iterative optimization. But the mental model of "walking downhill on a loss surface" is exactly right, and the loss landscape geometry becomes a huge topic when you get to LLM-scale training (loss spikes, saddle points, learning rate warmup to avoid sharp regions, etc.).

### Notes Section — Key Distinctions

- **Loss vs. cost vs. objective function**: technically, loss = per-sample, cost = aggregated over dataset (can include regularization terms), objective = any function being optimized. In practice, people use these interchangeably.
- **Discriminative vs. generative models**:
	- **Discriminative**: $\mathbf{y} = \mathbf{f}[\mathbf{x}, \boldsymbol{\phi}]$ — directly predict output from input
	- **Generative**: $\mathbf{x} = \mathbf{g}[\mathbf{y}, \boldsymbol{\phi}]$ — model how data was created, invert to predict
	- In practice, discriminative models dominate because flexible discriminative models + lots of data beats the inductive bias advantage of generative models

> **Modern context**: This discriminative vs. generative distinction has gotten blurry. Autoregressive LLMs are technically generative models (they model $P(\text{text})$), but when you use them for classification or QA they act discriminatively. Diffusion models are generative. The real insight: at frontier scale, the distinction matters less than having the right training objective and enough data.

---

## Chapter 3 — Shallow Neural Networks

> **Importance**: This chapter introduces the machinery that scales to everything we use today. The shallow network is a one-hidden-layer MLP. It's not what we use in practice (we use deep networks), but understanding the geometry of a single layer is prerequisite to understanding what stacking layers does. The MLP block inside every transformer layer is literally a wider, deeper version of what's described here.

### 3.1 Neural Network Example

- **Shallow network** (1 input, 1 output, 3 hidden units):
$$y = \phi_0 + \phi_1 \text{a}[\theta_{10} + \theta_{11}x] + \phi_2 \text{a}[\theta_{20} + \theta_{21}x] + \phi_3 \text{a}[\theta_{30} + \theta_{31}x]$$
- **Three-step computation**:
	1. Compute linear functions of input: $\theta_{\bullet 0} + \theta_{\bullet 1}x$ (one per hidden unit)
	2. Apply activation function $\text{a}[\bullet]$ to each — produces **hidden units** $h_1, h_2, h_3$
	3. Take weighted sum of hidden units + bias: $y = \phi_0 + \phi_1 h_1 + \phi_2 h_2 + \phi_3 h_3$
- **10 parameters** total for this tiny network

#### ReLU Activation

- $\text{ReLU}[z] = \max(0, z)$ — clips negative values to zero
- Most common activation in practice by far
- Simple, cheap to compute, no saturation for positive inputs

> **Why ReLU still matters**: ReLU is still the default in most MLP layers inside transformers (though many modern LLMs use variants like GeLU or SiLU/Swish). The key property isn't the specific function — it's that it's a cheap nonlinearity that doesn't saturate on the positive side, so gradients flow well. GeLU is a smooth approximation of ReLU that empirically trains slightly better for transformers. SwiGLU (used in LLaMA, Gemma, etc.) uses a gated variant. But they all serve the same role: inject nonlinearity into the MLP blocks.

#### 3.1.1 Neural Network Intuition — Piecewise Linear Functions

- With ReLU, the network produces **continuous piecewise linear functions**
	- 3 hidden units → up to 4 linear regions (each ReLU contributes one "joint")
	- Each linear region corresponds to a different **activation pattern** (which hidden units are active vs. clipped)
- **Slopes of each region** determined by which units are active and their weights
	- e.g., in a region where $h_1$ and $h_3$ are active but $h_2$ is clipped: slope = $\theta_{11}\phi_1 + \theta_{31}\phi_3$

> **Why this geometry matters for LLMs**: The piecewise-linear picture gives you the right mental model for what MLP layers in transformers are doing. Each MLP block partitions its input space into linear regions and applies different linear transformations in each. Mechanistic interpretability work (Anthropic's superposition papers, etc.) literally studies which features activate which "neurons" (hidden units) in transformer MLPs. The activation pattern picture from this chapter is exactly that, just at massive scale. The key finding from interpretability research: individual neurons often aren't monosemantic (they don't correspond to one clean concept), leading to sparse autoencoder work that tries to decompose activations into interpretable directions.

#### 3.1.2 Depicting Neural Networks

- Standard diagram: input layer → hidden layer → output layer
- Arrows = weights (slope params), hidden circles = neurons
- Biases typically omitted from diagrams

### 3.2 Universal Approximation Theorem ★★★

- Generalize to $D$ hidden units: $h_d = \text{a}[\theta_{d0} + \theta_{d1}x]$, output $y = \phi_0 + \sum_{d=1}^{D} \phi_d h_d$
- $D$ hidden units → at most $D$ joints → at most $D+1$ linear regions
- **Theorem**: for any continuous function on a compact subset of $\mathbb{R}$, there exists a shallow network (with enough hidden units) that approximates it to arbitrary precision
	- More hidden units = more linear regions = finer approximation
	- Generalizes to multi-dimensional inputs/outputs

> **Honest take on UAT**: The universal approximation theorem is one of those results that's important to know and completely useless in practice. Yes, a single hidden layer can approximate anything — but it might need exponentially many hidden units to do so. The theorem says nothing about:
> - How many units you actually need (could be astronomical)
> - Whether gradient descent will find the right parameters
> - Whether the resulting model will generalize
> 
> Deep networks (Ch 4) are better because they can express the same functions with exponentially fewer parameters via hierarchical composition. This is the real reason we use depth — not because shallow networks can't theoretically approximate things, but because they need way too many parameters to do so in practice. The UAT is a nice existence proof but tells you nothing about learnability.
>
> **Where UAT-adjacent thinking shows up in LLM research**: Width vs. depth tradeoffs. There's active research on whether making transformer layers wider (more hidden dim in MLP) vs. deeper (more layers) is more parameter-efficient. Scaling laws (Chinchilla, etc.) study this empirically. The theoretical backing comes from approximation theory results like UAT and its depth-efficient generalizations.

### 3.3 Multivariate Inputs and Outputs

#### Multiple Outputs

- Use a different linear combination of hidden units for each output dimension:
$$y_j = \phi_{j0} + \sum_{d=1}^{D} \phi_{jd} h_d$$
- Hidden units are **shared** across outputs — joint positions are the same, only slopes/offsets differ per output
- This is weight sharing / representation sharing — the hidden layer learns features useful for all outputs

> **Direct LLM connection**: This is exactly the structure of the output layer in a language model. The hidden layer (last transformer layer's output) is shared, and the "unembedding" matrix projects to vocabulary-sized logits. Each vocabulary token is a different output that shares the same internal representation. The fact that outputs share the hidden representation is why token embeddings capture semantic relationships — similar tokens use similar combinations of the same hidden features.

#### Multiple Inputs

- Each hidden unit receives a linear combination of all inputs: $h_d = \text{a}[\theta_{d0} + \sum_{i=1}^{D_i} \theta_{di} x_i]$
- With 2D inputs, each hidden unit defines a **hyperplane** in input space
	- ReLU clips one side of the hyperplane to zero
	- Result is a surface made of **convex polygonal regions**, each with its own linear function
- With $D_i$ input dimensions, regions become **convex polytopes** in $\mathbb{R}^{D_i}$

#### Number of Linear Regions Scales Rapidly

- With $D$ hidden units and $D_i$ input dimensions: up to $2^{D_i}$ regions when $D \geq D_i$ (axis-aligned case)
- In practice, typically *more* than $2^{D_i}$ because hyperplanes aren't axis-aligned
- Example from text: $D = 500$, $D_i = 100$ → more than $10^{107}$ possible regions with only 51,001 parameters
	- This is tiny by modern standards

> **Scale context**: Modern LLMs have hidden dims of 4096–16384 and MLP intermediate dims of 11008–65536. The number of linear regions in a *single* MLP layer is incomprehensibly large. This is partly why these models can memorize vast amounts of factual knowledge — the input space is partitioned into an astronomical number of regions, each with its own linear behavior. This connects to the "LLMs as databases" intuition and to research on knowledge localization (where facts are stored in the network).

### 3.4 Shallow Networks: General Case

- **General shallow network**: $\mathbf{x} \in \mathbb{R}^{D_i} \to \mathbf{y} \in \mathbb{R}^{D_o}$ with $D$ hidden units
$$h_d = \text{a}\left[\theta_{d0} + \sum_{i=1}^{D_i} \theta_{di} x_i\right]$$
$$y_j = \phi_{j0} + \sum_{d=1}^{D} \phi_{jd} h_d$$
- **Parameters** $\boldsymbol{\phi} = \{\theta_{\bullet\bullet}, \phi_{\bullet\bullet}\}$
- Activation function must be nonlinear — otherwise the whole thing collapses to a linear map (composition of linear functions is linear)
- With ReLU: input space divided into convex polytopes, each containing a different linear function per output

### 3.5 Terminology

| Term | Meaning |
|------|---------|
| **Input layer** | The inputs $\mathbf{x}$ |
| **Hidden layer** | The hidden units $h_d$ (after activation) |
| **Output layer** | The outputs $\mathbf{y}$ |
| **Neurons** / **hidden units** | Individual elements of hidden layer |
| **Pre-activations** | Values before activation function ($\theta_{d0} + \sum \theta_{di} x_i$) |
| **Activations** | Values after activation function ($h_d$) |
| **MLP** (multi-layer perceptron) | Any network with at least one hidden layer |
| **Shallow network** | One hidden layer |
| **Deep network** | Multiple hidden layers |
| **Feed-forward** | Acyclic computation graph (no loops) |
| **Fully connected** | Every element in one layer connects to every element in next |
| **Weights** | Slope parameters (arrows in diagrams) |
| **Biases** | Offset parameters |

> **Terminology in LLM context**: When people say "MLP layer" in a transformer, they mean the feed-forward network block (the two linear projections with a nonlinearity in between). "Fully connected" is sometimes used interchangeably with "dense" — as opposed to sparse architectures like mixture-of-experts (MoE), where only a subset of parameters are activated per input. The term "neurons" is used heavily in mechanistic interpretability research.

### 3.6 Summary

- Shallow nets: linear functions of input → activation function → linear combination for output
- Divide input space into piecewise linear regions
- Universal approximation: can approximate any continuous function with enough hidden units
- But: no guarantee on efficiency, learnability, or generalization

### Notes — "Neural" Networks

- The name is a historical artifact — the connection to biological neurons is superficial
- Dense connectivity resembles mammalian neural connectivity, but the computation is nothing like how brains work
- **Not a useful analogy** for understanding modern deep learning

> **Honest opinion**: The biological analogy actively misleads people. Brains have recurrent dynamics, spike-timing-dependent plasticity, no backpropagation, etc. The "neuron" terminology persists and is fine as jargon, but don't let it shape your intuitions about how these systems work. Think of neural networks as parameterized function approximators optimized by gradient descent — that's the useful abstraction.

---

## Cross-Cutting Themes: What Matters for LLM Research

### Concepts from Ch 2–3 That You'll Use Constantly

1. **Supervised learning as function approximation** — pretraining, SFT, reward model training are all instances of this
2. **Loss minimization by gradient descent** — the entire training pipeline
3. **Piecewise linear geometry of ReLU networks** — the right mental model for what MLP layers compute
4. **Activation patterns** — central to mechanistic interpretability (which neurons fire for which inputs)
5. **Shared representations across outputs** — why the same hidden states can be projected to answer different questions (this is what the unembedding matrix does)

### Concepts That Are Good Background But Won't Come Up Daily

- Universal approximation theorem — know it exists, don't lean on it
- Specific linear regression mechanics — too simple to transfer directly
- Discriminative vs. generative model distinction — increasingly blurred at scale
- Exact count of linear regions — interesting theoretically, not actionable

### Active Research Areas That Build on These Foundations

- **Mechanistic interpretability** (Anthropic, DeepMind): understanding which neurons/features activate for which inputs, decomposing MLP layers into interpretable directions, superposition hypothesis
- **Scaling laws** (Chinchilla, Kaplan et al.): how performance scales with parameters, data, and compute — directly relates to width/depth tradeoffs from Ch 3
- **Sparse Mixture-of-Experts** (Switch Transformer, Mixtral, DeepSeek): instead of one big dense MLP, route inputs to different "expert" sub-networks. The geometry argument from Ch 3 explains why this works — different experts can specialize in different regions of input space
- **Knowledge editing / localization**: research on where factual knowledge is stored in MLPs, whether you can surgically edit it (ROME, MEMIT). Relies on the per-region linear function picture from Ch 3
- **GeLU/SwiGLU activation functions**: modern replacements for ReLU in transformers. Same role (inject nonlinearity), smoother gradients, empirically better training dynamics at scale
