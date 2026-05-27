# Ch 6-7: Fitting Models, Gradients & Initialization

> These two chapters cover the optimization toolkit for training neural networks: how to minimize loss functions (Ch 6) and how to efficiently compute gradients + initialize parameters so training actually works (Ch 7). This is the bread and butter of making models learn. Almost everything here is directly load-bearing in modern LLM training pipelines.

---

## Importance to Modern LLM Research: Overview

**The optimizer and its hyperparameters are arguably the second most important design choice after the architecture itself for frontier LLM training.** The difference between a good and bad optimizer configuration can be the difference between a stable 3-month training run and a diverged waste of millions of dollars in compute. Backpropagation is so fundamental it's invisible -- you call `loss.backward()` and move on -- but understanding it deeply matters when you're debugging gradient flow in 100+ layer transformers, designing new architectures, or working on memory-efficient training. Initialization interacts with every architectural choice (residual connections, normalization layers, attention scaling) and getting it wrong can silently degrade performance even when training doesn't outright fail.

---

## Ch 6: Fitting Models

### 6.1 Gradient Descent

- **Core loop**: compute $\frac{\partial L}{\partial \phi}$, then update $\phi \leftarrow \phi - \alpha \cdot \frac{\partial L}{\partial \phi}$
	- $\alpha$ is learning rate (or found via line search)
	- Terminate when gradient magnitude is small enough
- **Linear regression**: loss is convex (single global minimum), GD always works
	- Convex = every chord lies above the function
	- Hessian $\mathbf{H}[\phi]$ positive definite $\Rightarrow$ convex
- **Non-convex losses** (the real world): loss surfaces for neural nets have local minima and saddle points
	- **Local minima**: gradient is zero, loss increases in all directions, but not the global best
	- **Saddle points**: gradient is zero, but loss increases in some directions and decreases in others
	- GD can get stuck at either -- no guarantee of finding global minimum
	- Near saddle points the surface is flat, so gradient is tiny and you might erroneously stop

> **Relevance**: In practice, for large overparameterized models (like LLMs), local minima are less of a concern than you'd think -- most local minima are roughly equally good (Ch 20 discusses this). Saddle points are more of a nuisance but SGD + momentum handles them. The loss landscape of large transformers is surprisingly well-behaved in the directions that matter, which is part of why deep learning works at all.

### 6.2 Stochastic Gradient Descent (SGD) [IMPORTANT]

- **Key idea**: instead of computing gradient over full dataset, use a random subset (minibatch/batch) at each step
	- Update: $\phi_{t+1} \leftarrow \phi_t - \alpha \cdot \sum_{i \in \mathcal{B}_t} \frac{\partial \ell_i[\phi_t]}{\partial \phi}$
	- $\mathcal{B}_t$ = set of indices in current batch

- **Batches and epochs**
	- Batches drawn without replacement from dataset
	- **Epoch** = one full pass through the training data
	- Batch size can range from 1 sample to full dataset (full-batch = regular GD)

- **Why SGD is better than GD in practice**
	- Each iteration is cheaper (only process a subset)
	- The noise from random batches can **escape local minima** -- a batch's loss surface differs from the full loss surface, so "downhill" on the batch can be "uphill" on the full loss, jumping you out of bad basins
	- Less likely to get stuck at saddle points (some batches will have significant gradient even if full gradient is near zero)
	- **Implicit regularization**: SGD tends to find parameters that generalize better (see Ch 9.2)
	- Can be viewed as deterministic GD on a constantly changing loss function (one per batch)

- **Learning rate schedules**: start high (explore), decay over time (fine-tune)
	- Common: decrease by constant factor every $N$ epochs

> **Relevance [CRITICAL]**: SGD (and its variants) is *the* algorithm for training everything in deep learning. For LLMs specifically:
> - **Batch size** is a critical hyperparameter. Larger batches = more stable gradients but worse generalization (the noise is the feature, not the bug). The ratio of batch size to learning rate is key -- scaling laws research (Goyal et al., 2018; Smith et al., 2018) showed you should scale LR linearly with batch size up to a point.
> - Modern LLM training uses very large batches (millions of tokens) for throughput, but this requires careful LR tuning and warmup to compensate for the reduced noise.
> - **Data ordering/curriculum** effects: because SGD processes data in batches sequentially, the order matters more than people think. This connects to curriculum learning and data mixing strategies for LLM pretraining.

### 6.3 Momentum [IMPORTANT]

- **SGD + momentum**: update is weighted combination of current gradient and previous direction
	- $\mathbf{m}_{t+1} \leftarrow \beta \cdot \mathbf{m}_t + (1-\beta) \sum_{i \in \mathcal{B}_t} \frac{\partial \ell_i[\phi_t]}{\partial \phi}$
	- $\phi_{t+1} \leftarrow \phi_t - \alpha \cdot \mathbf{m}_{t+1}$
	- $\beta \in [0,1)$ controls smoothing
- **Effect**: exponential moving average of all past gradients
	- Aligned gradients across steps $\Rightarrow$ effective LR increases (accelerates through consistent directions)
	- Oscillating gradients $\Rightarrow$ terms cancel out (damps oscillation in valleys)
	- Smoother trajectory, faster convergence

- **Nesterov accelerated momentum**: evaluate gradient at the *predicted* next position $\phi_t - \alpha\beta \cdot \mathbf{m}_t$ instead of current position
	- "Look ahead" before computing gradient -- corrects the path from momentum alone
	- Slightly better in practice for convex problems; mixed results for deep learning

> **Relevance**: Momentum ($\beta = 0.9$) is standard in virtually all LLM training configs. It's baked into Adam. Nesterov momentum is used less often in the transformer world -- Adam's adaptive learning rates subsume much of its benefit.

### 6.4 Adam (Adaptive Moment Estimation) [CRITICAL]

- **Problem with fixed step size**: GD makes big steps where gradients are large (should be cautious) and small steps where gradients are small (should explore more)
	- When loss surface is much steeper in one direction than another, no single LR works well for both

- **Gradient normalization idea**: normalize gradients so you move a fixed distance $\alpha$ in each coordinate direction
	- $\phi_{t+1} \leftarrow \phi_t - \alpha \cdot \frac{\mathbf{m}_{t+1}}{\sqrt{\mathbf{v}_{t+1}} + \epsilon}$
	- This alone oscillates and doesn't converge

- **Adam**: add momentum to *both* the gradient estimate and the squared gradient estimate
	- $\mathbf{m}_{t+1} \leftarrow \beta \cdot \mathbf{m}_t + (1-\beta) \frac{\partial L[\phi_t]}{\partial \phi}$ (first moment / mean of gradients)
	- $\mathbf{v}_{t+1} \leftarrow \gamma \cdot \mathbf{v}_t + (1-\gamma) \left(\frac{\partial L[\phi_t]}{\partial \phi}\right)^2$ (second moment / variance of gradients)
	- **Bias correction** (crucial for early training):
		- $\tilde{\mathbf{m}}_{t+1} \leftarrow \frac{\mathbf{m}_{t+1}}{1 - \beta^{t+1}}$
		- $\tilde{\mathbf{v}}_{t+1} \leftarrow \frac{\mathbf{v}_{t+1}}{1 - \gamma^{t+1}}$
	- Update: $\phi_{t+1} \leftarrow \phi_t - \alpha \cdot \frac{\tilde{\mathbf{m}}_{t+1}}{\sqrt{\tilde{\mathbf{v}}_{t+1}} + \epsilon}$
	- Default hyperparams: $\alpha = 0.001$, $\beta = 0.9$, $\gamma = 0.999$, $\epsilon = 10^{-8}$

- **Why Adam works well**
	- Per-parameter adaptive learning rates -- compensates for different gradient magnitudes across layers
	- Less sensitive to initial LR choice (avoids the pathological cases in Fig 6.9a-b)
	- Makes good progress in every direction of parameter space
	- Doesn't need complex LR schedules to work reasonably well

- **Adam variants and related optimizers**
	- **AdaGrad**: cumulative squared gradient for per-parameter LR, but LR decays to zero over time
	- **RMSProp**: fixes AdaGrad by using exponential moving average of squared gradients
	- **AdamW** (Loshchilov & Hutter, 2019): fixes Adam's interaction with L2 regularization -- **decoupled weight decay**
	- **AMSGrad**: theoretical fix for Adam's non-convergence on convex functions (Reddi et al., 2018)
	- **YOGI**: another convergent variant

- **SGD vs Adam debate**
	- Wilson et al. (2017): SGD+momentum can find *better* minima (better generalization) than Adam
	- But this is likely an artifact of using Adam's default hyperparams -- with tuned hyperparams, Adam matches SGD (Choi et al., 2019)
	- **SWATS** (Keskar & Socher, 2017): start with Adam, switch to SGD for final generalization performance
	- **Learning rate warmup**: gradually increase LR over first few thousand iterations to avoid instability from noisy early statistics

> **Relevance [CRITICAL]**: **Adam (specifically AdamW) is the dominant optimizer for LLM training.** Full stop. Here's why this matters:
> - Every major LLM (GPT, LLaMA, Claude, Gemini, etc.) uses AdamW or a close variant
> - **AdamW vs Adam**: the decoupled weight decay in AdamW is important -- regular Adam's L2 regularization is effectively scaled by the adaptive learning rate, which defeats the purpose. AdamW fixes this. If you're training an LLM, use AdamW.
> - **The $\beta_1, \beta_2$ hyperparameters matter a lot for LLM stability**: common LLM configs use $\beta_1 = 0.9, \beta_2 = 0.95$ (not the default 0.999). Lower $\beta_2$ makes the optimizer more responsive to recent gradient magnitudes, which helps with the non-stationarity of language modeling loss landscapes.
> - **LR warmup is essential** for large model training -- without it, the optimizer statistics are garbage early on and you get instability or divergence. Typical: linear warmup over 1-2K steps.
> - **Cosine LR schedule** (not discussed in book) is standard: warmup then cosine decay to ~10% of peak LR. This has become the default for pretraining.
> - **Newer optimizers for LLMs**: Lion (Chen et al., 2023), Sophia (Liu et al., 2023) use sign-based updates or second-order info; Shampoo/SOAP use preconditioned gradients. These show promise but AdamW remains the safe default.
> - **The optimizer state is expensive**: Adam stores $\mathbf{m}$ and $\mathbf{v}$ for every parameter -- that's 2x the model size in optimizer states alone. For a 70B model in fp32, that's ~560GB just for optimizer states. This drives techniques like mixed-precision training, 8-bit Adam, and optimizer state partitioning (ZeRO).

### 6.5 Training Algorithm Hyperparameters

- Choice of optimizer, batch size, LR schedule, momentum coefficients = **hyperparameters**
- Distinct from model parameters -- not learned by gradient descent
- Choosing them is "more art than science" -- hyperparameter search (Ch 8)

> **Relevance**: For LLM pretraining, hyperparameter tuning is done on smaller proxy models and then transferred via scaling laws (the "muP" / maximal update parametrization approach). You can't grid-search hyperparameters when a single run costs millions of dollars.

---

## Ch 7: Gradients and Initialization

### 7.1 Problem Setup

- Network: $\mathbf{h}_k = \mathbf{a}[\mathbf{f}_{k-1}]$, $\mathbf{f}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k$ for layers $k = 1, \ldots, K$
- Parameters $\phi = \{\boldsymbol{\beta}_0, \boldsymbol{\Omega}_0, \ldots, \boldsymbol{\beta}_K, \boldsymbol{\Omega}_K\}$ (biases and weight matrices)
- Need to compute $\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k}$ and $\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k}$ for all layers $k$ and all batch elements $i$
- Two problems addressed: (1) efficient gradient computation, (2) parameter initialization

### 7.2 Computing Derivatives -- Key Intuitions

- **Observation 1**: effect of changing a weight is scaled by the activation at the source hidden unit
	- $\Rightarrow$ need to store all activations during forward pass
- **Observation 2**: changing a parameter causes a ripple effect through all subsequent layers to the loss
	- Same downstream derivatives are needed for parameters at the same or earlier layers $\Rightarrow$ compute once and reuse
	- Work backward from the loss to reuse computations

### 7.3 Toy Example (Scalar Backprop)

- Model: $\text{f}[x, \phi] = \beta_3 + \omega_3 \cdot \cos[\beta_2 + \omega_2 \cdot \exp[\beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x]]]$
- **Forward pass**: compute and store intermediate variables $f_0, h_1, f_1, h_2, f_2, h_3, f_3, \ell_i$ sequentially
- **Backward pass #1**: compute $\frac{\partial \ell_i}{\partial f_k}$ and $\frac{\partial \ell_i}{\partial h_k}$ from end to start using chain rule
	- Each derivative = (local derivative) $\times$ (previously computed downstream derivative)
	- E.g., $\frac{\partial \ell_i}{\partial h_3} = \frac{\partial f_3}{\partial h_3} \cdot \frac{\partial \ell_i}{\partial f_3}$
- **Backward pass #2**: compute parameter derivatives
	- $\frac{\partial \ell_i}{\partial \beta_k} = \frac{\partial f_k}{\partial \beta_k} \cdot \frac{\partial \ell_i}{\partial f_k} = 1 \cdot \frac{\partial \ell_i}{\partial f_k}$
	- $\frac{\partial \ell_i}{\partial \omega_k} = \frac{\partial f_k}{\partial \omega_k} \cdot \frac{\partial \ell_i}{\partial f_k} = h_k \cdot \frac{\partial \ell_i}{\partial f_k}$
	- Weight gradient is proportional to the activation it multiplies (stored in forward pass)

### 7.4 Backpropagation Algorithm [CRITICAL]

- **Forward pass**: compute and store all pre-activations and activations
	- $\mathbf{f}_0 = \boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}_i$
	- $\mathbf{h}_k = \mathbf{a}[\mathbf{f}_{k-1}]$ for $k \in \{1, \ldots, K\}$
	- $\mathbf{f}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k$ for $k \in \{1, \ldots, K\}$
	- $\ell_i = \text{l}[\mathbf{f}_K, \mathbf{y}_i]$

- **Backward pass #1**: compute $\frac{\partial \ell_i}{\partial \mathbf{f}_k}$ for all layers, starting from $\frac{\partial \ell_i}{\partial \mathbf{f}_K}$
	- Key recurrence: $\frac{\partial \ell_i}{\partial \mathbf{f}_{k-1}} = \mathbb{I}[\mathbf{f}_{k-1} > 0] \odot \left(\boldsymbol{\Omega}_k^T \frac{\partial \ell_i}{\partial \mathbf{f}_k}\right)$
		- $\mathbb{I}[\mathbf{f}_{k-1} > 0]$ is the ReLU derivative: 1 where pre-activation > 0, else 0
		- $\boldsymbol{\Omega}_k^T$ propagates gradients back through the weight matrix
		- $\odot$ is pointwise multiplication (masking by ReLU derivative)
	- Pattern: alternate between (i) multiplying by $\boldsymbol{\Omega}_k^T$ and (ii) thresholding based on stored $\mathbf{f}_{k-1}$

- **Backward pass #2**: compute parameter gradients
	- $\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} = \frac{\partial \ell_i}{\partial \mathbf{f}_k}$ (bias gradient = the pre-activation gradient itself)
	- $\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k} = \frac{\partial \ell_i}{\partial \mathbf{f}_k} \mathbf{h}_k^T$ (weight gradient = outer product of pre-activation gradient and input activations)
	- First layer: $\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_0} = \frac{\partial \ell_i}{\partial \mathbf{f}_0} \mathbf{x}_i^T$

- **Computational cost**: dominated by matrix multiplications ($\boldsymbol{\Omega}$ forward, $\boldsymbol{\Omega}^T$ backward)
	- Backprop is roughly **2x the cost of the forward pass** (one matmul forward, one transpose matmul backward, plus outer products for weight gradients)
- **Memory cost**: must store all intermediate $\mathbf{f}_k$ and $\mathbf{h}_k$ values -- **this is the bottleneck**, not compute

> **Relevance [CRITICAL]**: Backprop is the foundation of all neural network training. For LLMs:
> - **Memory is the bottleneck**: for a transformer with billions of parameters, storing activations for every layer and every token in the batch is enormous. This directly limits batch size and sequence length.
> - **Gradient checkpointing** (activation recomputation): trade compute for memory by only storing activations at every $N$th layer, recomputing the rest during backward pass. Standard in LLM training -- typically 2x forward pass cost but massive memory savings. FlashAttention also reduces memory by not materializing the full attention matrix.
> - **Mixed-precision training**: forward pass in fp16/bf16, accumulate gradients in fp32. Cuts memory ~2x and exploits tensor cores. bf16 is preferred for LLMs because it has the same exponent range as fp32 (no overflow issues).

### 7.4.2 Algorithmic Differentiation (Autograd)

- In practice, you never implement backprop by hand -- frameworks (PyTorch, JAX) do it automatically
- Each operation (linear transform, ReLU, loss) knows its own derivative
- Framework builds a **computational graph** and applies chain rule automatically = **reverse-mode algorithmic differentiation**
- Forward-mode differentiation exists but is less efficient for neural nets (where #outputs << #parameters)
- Extends to **arbitrary acyclic computational graphs**, not just sequential networks

> **Relevance**: Understanding autograd is essential for writing custom training loops, implementing new loss functions, or debugging gradient issues. JAX's functional autodiff (`jax.grad`, `jax.vjp`) is particularly popular in research because it composes cleanly with `jax.vmap`, `jax.pmap`, etc. for per-example gradients and distributed training.

### 7.4.3 Extension to Arbitrary Computational Graphs

- Backprop works for any DAG, not just sequential models
- Models with branching structures (e.g., residual connections, multi-head attention) work fine
- PyTorch/TensorFlow handle this automatically

### 7.5 Parameter Initialization [IMPORTANT]

- **Why it matters**: pre-activations $\mathbf{f}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{a}[\mathbf{f}_{k-1}]$ are computed layer by layer
	- If weights have variance too small ($\sigma^2 \sim 10^{-5}$): activations shrink exponentially through layers $\Rightarrow$ **vanishing activations** (and vanishing gradients in backward pass)
	- If weights have variance too large ($\sigma^2 \sim 10^5$): activations grow exponentially $\Rightarrow$ **exploding activations** (and exploding gradients in backward pass)
	- Both cause values to exceed floating point range $\Rightarrow$ training fails

- **The math** (for ReLU networks):
	- Input pre-activations have variance $\sigma_f^2$
	- After one layer with weights $\boldsymbol{\Omega}$ (dimension $D_h$, variance $\sigma_\Omega^2$), biases = 0:
		- Mean of next layer: $\mathbb{E}[f'_i] = 0$
		- Variance of next layer: $\sigma_{f'}^2 = \frac{1}{2} D_h \sigma_\Omega^2 \sigma_f^2$
		- The $\frac{1}{2}$ comes from ReLU clipping half the distribution to zero
	- For variance to stay stable across layers: need $\frac{1}{2} D_h \sigma_\Omega^2 = 1$

#### 7.5.1 He Initialization (Forward Pass)

- Set $\sigma_\Omega^2 = \frac{2}{D_h}$ where $D_h$ is the fan-in (dimension of source layer)
- Biases initialized to zero
- Keeps variance of activations stable through forward pass

#### 7.5.2 Initialization for Backward Pass

- Same logic applied to gradients: backward pass multiplies by $\boldsymbol{\Omega}^T$
- For stable gradient magnitudes: $\sigma_\Omega^2 = \frac{2}{D_{h'}}$ where $D_{h'}$ is the fan-out (dimension of destination layer)

#### 7.5.3 Xavier/Glorot Initialization (Compromise)

- Can't satisfy both forward and backward simultaneously when layers have different sizes
- **Compromise**: $\sigma_\Omega^2 = \frac{4}{D_h + D_{h'}}$ (average of fan-in and fan-out)
- Glorot & Bengio (2010) originally derived this for sigmoid/tanh activations (without the factor of 2 for ReLU)

> **Relevance [IMPORTANT]**: Initialization is foundational but modern architectures have largely *engineered around* the raw initialization problem:
> - **LayerNorm/RMSNorm**: transformers use normalization after (or before) each sublayer, which re-normalizes activations and gradients regardless of initialization. This is a huge part of why transformers are easier to train than vanilla deep nets.
> - **Residual connections**: skip connections mean gradients have a direct path backward, mitigating vanishing gradients. But you still need to initialize the residual branches carefully.
> - **Scaling factors in attention**: the $\frac{1}{\sqrt{d_k}}$ in scaled dot-product attention is an initialization-style fix -- it prevents the softmax from saturating by keeping the variance of the dot products at O(1).
> - **muP (Maximal Update Parametrization)**: Yang et al. (2022) showed that with the right parametrization, optimal hyperparameters transfer across model scales. This is essentially a systematic theory of how initialization, learning rate, and width interact. Increasingly used at frontier labs for LLM training.
> - **In practice** for transformers: use the framework defaults (usually Glorot/Xavier or He) + normalization layers, and you're fine. The cases where initialization really bites you are when you're doing something non-standard (e.g., very deep networks without normalization, novel architectures).

### 7.5 Notes Section -- Additional Important Topics

#### Reducing Memory Requirements

- **Gradient checkpointing** (Chen et al., 2016a): only store activations at every $N$ layers during forward pass, recompute the rest during backward pass
	- ~2x forward pass compute cost, but dramatically less memory
- **Micro-batching** (Huang et al., 2019): subdivide batch into smaller sub-batches, aggregate gradients
	- Enables large effective batch sizes on limited GPU memory
- **Reversible networks** (Gomez et al., 2017): compute previous layer activations from current ones $\Rightarrow$ no need to cache anything during forward pass
	- Connects to normalizing flows (Ch 16)

#### Distributed Training

- **Data parallelism**: each GPU has full model, processes different data subset, gradients aggregated
	- **Synchronous**: aggregate all gradients before updating (consistent but has synchronization bottleneck)
	- **Asynchronous** (e.g., Hogwild, Recht et al., 2011): update central model whenever a node finishes -- stale gradients but faster in practice
- **Pipeline model parallelism**: different layers on different GPUs, pass activations between them
	- Obvious disadvantage: GPUs idle for most of the cycle ("pipeline bubble")
	- Solutions: micro-batching to keep pipeline full (GPipe, PipeDream)
- **Tensor model parallelism**: a single layer's computation is split across GPUs
	- E.g., Megatron-LM (Shoeybi et al., 2019): split attention heads and MLP across GPUs
- **Combined approaches**: Narayanan et al. (2021b) trained 1T parameter model on 3072 GPUs combining tensor, pipeline, and data parallelism

> **Relevance [CRITICAL]**: **Distributed training is arguably the most important systems-level topic for LLM research.** You cannot train frontier models on a single GPU. The entire field is shaped by the constraints of distributing training across thousands of accelerators:
> - **ZeRO** (Rajbhandari et al., 2020): partitions optimizer states, gradients, and parameters across GPUs rather than replicating them. ZeRO-3 enables training models that don't fit in any single GPU's memory. This is what DeepSpeed uses.
> - **FSDP** (Fully Sharded Data Parallelism): PyTorch's native implementation of ZeRO-style sharding.
> - **Ring-AllReduce** vs **tree-AllReduce**: communication patterns for gradient aggregation. The communication cost scales with model size, not number of GPUs, making it efficient at scale.
> - **The interplay between batch size, gradient accumulation, data parallelism, and learning rate** is one of the most important practical considerations. Large batch training (Goyal et al., 2018) requires linear LR scaling + warmup.

#### Other Initialization Approaches

- **Layer-sequential unit-variance initialization** (LSUV, Mishkin & Matas, 2016): initialize weights as orthonormal, then pass data through and scale each layer to unit variance empirically
- **GradInit** (Zhu et al., 2021): learn scaling factors per layer to maximize loss decrease
- **Activation normalization / ActNorm**: learn offset and scale per layer from first batch (used in normalizing flows)
- **Fixup** (Zhang et al., 2019a): initialization scheme for ResNets without normalization
- **T-Fixup** (Xu et al., 2021b): Fixup adapted for transformers (DTFixup)

### 7.6 Example Training Code (PyTorch)

```python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# define input size, hidden layer size, output size
D_i, D_k, D_o = 10, 40, 5
# create model with two hidden layers
model = nn.Sequential(
    nn.Linear(D_i, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o))

# He initialization of weights
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)
model.apply(weights_init)

# choose least squares loss function
criterion = nn.MSELoss()
# construct SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)
# object that decreases learning rate by half every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# create 100 random data points and store in data loader class
x = torch.randn(100, D_i)
y = torch.randn(100, D_o)
data_loader = DataLoader(TensorDataset(x,y), batch_size=10, shuffle=True)

# loop over the dataset 100 times
for epoch in range(100):
    epoch_loss = 0.0
    # loop over batches
    for i, data in enumerate(data_loader):
        # retrieve inputs and labels for this batch
        x_batch, y_batch = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        # backward pass
        loss.backward()
        # SGD update
        optimizer.step()
        # update statistics
        epoch_loss += loss.item()
    # print error
    print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')
    # tell scheduler to consider updating learning rate
    scheduler.step()
```

- Key insight: **all backprop details are hidden in `loss.backward()`** -- one line of code

---

## Summary: What Actually Matters for LLM Work

### Tier 1 -- You use this every day at a frontier lab:
- **AdamW optimizer**: the default. Know its hyperparameters ($\beta_1, \beta_2$, $\epsilon$, weight decay) and how they affect training stability
- **Learning rate scheduling**: warmup + cosine decay. The warmup period, peak LR, and decay schedule are among the most impactful hyperparameters
- **Batch size / gradient accumulation**: directly tied to training throughput and generalization
- **Distributed training** (data/tensor/pipeline parallelism): you cannot escape this at scale
- **Gradient checkpointing**: standard memory optimization for training large models
- **Mixed-precision training**: bf16 forward/backward, fp32 master weights and optimizer states

### Tier 2 -- Important foundation, shapes how you think:
- **Backpropagation**: understanding the chain rule through the network helps you reason about gradient flow, design architectures, and debug training
- **Vanishing/exploding gradients**: the fundamental failure mode that motivates residual connections, normalization layers, and careful initialization
- **He/Xavier initialization**: the defaults work because of this math, but normalization layers do most of the heavy lifting in modern architectures
- **SGD and momentum**: the conceptual foundation even if you use Adam. Understanding why noise helps (escaping local minima, implicit regularization) informs many decisions

### Tier 3 -- Good to know, occasionally relevant:
- **Loss surface properties** (convexity, saddle points, local minima): more theoretical but informs understanding of why overparameterized models are trainable
- **Nesterov momentum**: rarely used directly in LLM training
- **Line search**: not used in practice for neural networks (too expensive per step)
- **Hessian analysis**: second-order methods are making a comeback (Sophia optimizer) but still niche
- **Forward-mode autodiff**: useful for specific research (e.g., computing Hessian-vector products) but not for standard training
