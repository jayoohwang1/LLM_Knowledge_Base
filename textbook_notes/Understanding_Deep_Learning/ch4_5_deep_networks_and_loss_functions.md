# Understanding Deep Learning - Chapters 4 & 5: Deep Neural Networks and Loss Functions

## Practical Importance Rating

**Chapter 4 (Deep Neural Networks)**: Foundational but mostly background knowledge at this point. The "why depth works" intuitions are useful for architecture decisions, but nobody is hand-designing network depth from first principles anymore -- you're using established architectures or scaling laws. **Know the intuitions, don't sweat the math.**

**Chapter 5 (Loss Functions)**: **Critically important.** Loss function design is one of the most impactful levers in modern ML. The maximum likelihood framework is the backbone of how we train basically everything -- LLMs, diffusion models, classifiers. Cross-entropy loss literally trains every autoregressive language model. Understanding loss functions deeply lets you debug training, design new objectives, and understand why models behave the way they do.

---

## Chapter 4: Deep Neural Networks

### 4.1 Composing Neural Networks

- **Core idea**: deep nets = composing shallow nets, output of one becomes input of next
	- First network "folds" input space back onto itself so multiple inputs map to same intermediate value
	- Second network applies its function to the folded space, gets duplicated at every fold point
	- Two 3-unit shallow nets composed: first creates 3 linear regions, second creates 3 more, composition yields up to 9 regions
	- Same principle extends to higher dimensions
- **The folding metaphor**: first net collapses input space, second net's function gets replicated wherever space was folded on top of itself
	- Actually a useful mental model for thinking about how features get reused across layers

> **Relevance to LLMs**: This folding/composition view is a nice way to think about why transformer layers progressively build more abstract representations. Each layer can be thought of as "folding" the representation space so that semantically similar things get mapped together, then subsequent layers operate on those grouped representations.

### 4.2 From Composing to Deep Networks

- Composing two shallow networks = special case of a two-hidden-layer deep network
- **Key insight**: the composed version has constrained parameters (outer product structure $[\theta'_{11}, \theta'_{21}, \theta'_{31}]^T[\phi_1, \phi_2, \phi_3]$), while the general deep network has unconstrained weight matrix $\psi_{ij}$
	- Deep net is strictly more expressive than composition -- more free parameters per connection
- Substituting first network output into second network and simplifying yields standard deep network equations

### 4.3 Deep Neural Networks -- General Formulation

- **How computation flows through a deep ReLU net** (step by step):
	1. First hidden layer: linear functions of input, passed through ReLU -- creates piecewise linear functions with joints
	2. Second layer pre-activations: linear combinations of first layer activations -- new piecewise linear functions with joints at same positions
	3. Second layer ReLU: clips these, adds new joints
	4. Output: linear combination of clipped functions
- Each layer adds complexity by clipping and recombining

#### 4.3.1 Hyperparameters

- **Width** $D_k$: number of hidden units per layer $k$
- **Depth** $K$: number of hidden layers
- **Capacity**: total number of hidden units across all layers
- These are chosen before training (hyperparameters), not learned

### 4.4 Matrix Notation

- **General deep network** with $K$ layers:
$$\mathbf{h}_1 = \mathbf{a}[\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}]$$
$$\mathbf{h}_k = \mathbf{a}[\boldsymbol{\beta}_{k-1} + \boldsymbol{\Omega}_{k-1} \mathbf{h}_{k-1}]$$
$$\mathbf{y} = \boldsymbol{\beta}_K + \boldsymbol{\Omega}_K \mathbf{h}_K$$
- $\boldsymbol{\Omega}_k \in \mathbb{R}^{D_{k+1} \times D_k}$ are weight matrices, $\boldsymbol{\beta}_k$ are bias vectors
- Parameters $\phi = \{\boldsymbol{\beta}_k, \boldsymbol{\Omega}_k\}_{k=0}^{K}$
- Alternating pattern: linear transform, nonlinearity, linear transform, nonlinearity...
	- This is still the backbone of every neural net, transformers included

### 4.5 Shallow vs. Deep Neural Networks -- **Important Section**

#### 4.5.1 Universal Approximation

- Both shallow and deep nets can approximate any continuous function given enough capacity
- Deep nets with two hidden layers can replicate any shallow net (second layer computes identity)
- **Depth version**: with ReLU activations, a network with $D_i + 4$ units per layer can approximate any function to arbitrary accuracy if enough layers available

#### 4.5.2 Number of Linear Regions per Parameter -- **Key Result**

- **Shallow net**: $D+1$ linear regions from $3D+1$ parameters (linear scaling)
- **Deep net**: up to $(D+1)^K$ regions from $3D + 1 + (K-1)D(D+1)$ parameters (exponential scaling!)
- Deep networks are exponentially more expressive per parameter
	- With $K=5$ layers, $D=10$ units each: 10,801 params can create $>10^{40}$ linear regions
- **BUT** -- the extra regions contain complex dependencies and symmetries from the folding structure
	- Only useful if real-world functions have similar compositional structure
	- Spoiler: they mostly do, which is why deep learning works

> **Relevance to scaling LLMs**: This is the theoretical underpinning for why scaling depth has historically been so effective. The exponential growth in expressivity per parameter is part of why a 70B parameter deep model can represent far more complex functions than a hypothetical shallow model with the same parameter count. In practice though, modern LLMs tend to be wide AND deep -- the scaling laws work suggests there's an optimal depth-to-width ratio for a given compute budget.

#### 4.5.3 Depth Efficiency

- Some functions require exponentially more hidden units in a shallow net vs. a deep one
- Not all functions fall into this category, but enough real-world functions do to make depth worthwhile

#### 4.5.4 Large, Structured Inputs

- Fully connected layers impractical for high-dimensional inputs (images: ~$10^6$ pixels)
- Need to process local regions in parallel, then integrate -- this requires depth
- Motivates convolutional architectures and, by extension, the local-then-global processing pattern

> **Relevance to LLMs**: This is exactly the motivation behind attention mechanisms in transformers. You can't have fully connected layers operating on raw token sequences of length 100K+. Attention provides a structured way to selectively process and integrate information, and the depth allows progressive abstraction -- early layers capture syntax/local patterns, later layers capture semantics/long-range dependencies.

#### 4.5.5 Training and Generalization

- Moderately deep networks are usually easier to train than shallow ones
	- Over-parameterized models have many roughly equivalent solutions that are easy to find
- But very deep networks get hard to train again (vanishing/exploding gradients -- ch. 11 on residual connections)
- Deep nets generalize better empirically -- not fully understood theoretically

---

## Chapter 5: Loss Functions -- **HIGH IMPORTANCE**

### Overview: The Probabilistic Framing

- **Paradigm shift**: instead of thinking of networks as computing outputs $\mathbf{y}$, think of them as computing parameters $\boldsymbol{\theta}$ of probability distributions $Pr(\mathbf{y}|\boldsymbol{\theta})$ over outputs
- This is the foundation of how basically every modern model is trained
- Loss function = negative log-likelihood of the data under the model's predicted distribution

> **This is the single most important idea in these two chapters for LLM research.** Every autoregressive LM is trained by predicting $Pr(\text{next token} | \text{context})$ and minimizing cross-entropy loss. Understanding this probabilistic framing is essential for understanding training dynamics, calibration, mode collapse, and basically everything about how LLMs work.

### 5.1 Maximum Likelihood -- **Critical Foundation**

#### 5.1.1 Computing Distributions over Outputs

- Choose a parametric distribution $Pr(\mathbf{y}|\boldsymbol{\theta})$ over the output domain
- Network predicts distribution parameters: $\boldsymbol{\theta} = \mathbf{f}[\mathbf{x}, \phi]$
- **Example**: for regression, choose normal distribution -- network predicts mean $\mu$, variance $\sigma^2$ could be fixed or also predicted

#### 5.1.2 Maximum Likelihood Criterion

- Choose params $\phi$ to maximize joint probability of training data:
$$\hat{\phi} = \underset{\phi}{\text{argmax}} \prod_{i=1}^{I} Pr(\mathbf{y}_i | \mathbf{x}_i)$$
- **Key assumptions**:
	- Same distribution family for all data points
	- Conditional independence: $Pr(\mathbf{y}_1, ..., \mathbf{y}_I | \mathbf{x}_1, ..., \mathbf{x}_I) = \prod_i Pr(\mathbf{y}_i | \mathbf{x}_i)$
- The product of probabilities = the **likelihood** of the parameters

> **In LLM training**: the independence assumption technically doesn't hold within a sequence (tokens are obviously not independent), but the autoregressive factorization $Pr(x_1, ..., x_T) = \prod_t Pr(x_t | x_{<t})$ sidesteps this beautifully by conditioning each prediction on all previous tokens. Each conditional IS independent given the context. This is one of the most elegant aspects of autoregressive modeling.

#### 5.1.3 Maximizing Log-Likelihood

- Product of many small probabilities = numerical underflow
- **Solution**: maximize sum of log-probabilities instead (log is monotonic, same optimum)
$$\hat{\phi} = \underset{\phi}{\text{argmax}} \sum_{i=1}^{I} \log Pr(\mathbf{y}_i | \mathbf{f}[\mathbf{x}_i, \phi])$$
- Sums are numerically stable, gradients decompose nicely

#### 5.1.4 Minimizing Negative Log-Likelihood (NLL)

- Convention: frame as minimization (gradient descent minimizes)
- **Loss function**: $L[\phi] = -\sum_i \log Pr(\mathbf{y}_i | \mathbf{f}[\mathbf{x}_i, \phi])$
- This is the loss function you see everywhere -- the NLL

> **This is literally what trains GPT, Claude, Llama, etc.** The per-token NLL averaged over the training corpus is the training loss. When people talk about "perplexity" of a language model, it's $\exp(\text{average NLL})$. Lower perplexity = better model. The entire scaling laws literature (Kaplan et al., Chinchilla, etc.) plots this quantity as a function of compute/data/params.

#### 5.1.5 Inference

- At inference time, return the mode of the predicted distribution: $\hat{\mathbf{y}} = \text{argmax}_{\mathbf{y}} Pr(\mathbf{y} | \mathbf{f}[\mathbf{x}, \hat{\phi}])$
- For normal distribution: mode = mean $\mu$
- For categorical distribution: mode = argmax over probabilities

> **In LLM inference**: this corresponds to greedy decoding (always pick the most probable token). In practice, we almost never do pure greedy decoding -- we use temperature scaling, top-k, top-p/nucleus sampling, etc. to trade off between quality and diversity. The distribution the model predicts is much richer than just its mode.

### 5.2 Recipe for Constructing Loss Functions

**The recipe** (used for every example in the chapter):
1. Choose probability distribution $Pr(\mathbf{y}|\boldsymbol{\theta})$ over prediction domain
2. Set network to predict distribution parameters: $\boldsymbol{\theta} = \mathbf{f}[\mathbf{x}, \phi]$
3. Minimize negative log-likelihood: $\hat{\phi} = \underset{\phi}{\text{argmin}} \left[-\sum_i \log Pr(\mathbf{y}_i | \mathbf{f}[\mathbf{x}_i, \phi])\right]$
4. Inference: return full distribution or its mode

- This recipe is universal -- pick your domain, pick your distribution, crank through the math

### 5.3 Example 1: Univariate Regression

#### 5.3.1 Least Squares from Maximum Likelihood

- **Distribution**: univariate normal $Pr(y|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(y - \mu)^2}{2\sigma^2}\right]$
- Network predicts mean: $\mu = \mathbf{f}[\mathbf{x}, \phi]$, variance $\sigma^2$ fixed
- NLL simplifies to (after dropping constants):
$$L[\phi] = \sum_{i=1}^{I} (y_i - \mathbf{f}[\mathbf{x}_i, \phi])^2$$
- **Least squares loss falls out of normal distribution assumption** -- not an arbitrary choice!
- Inference: $\hat{y} = \mathbf{f}[\mathbf{x}, \hat{\phi}]$ (mode of normal = mean)

#### 5.3.3 Estimating Variance

- Can also learn $\sigma^2$ by treating it as a parameter to optimize alongside $\phi$
- Gives you a measure of model uncertainty "for free"

#### 5.3.4 Heteroscedastic Regression

- **Homoscedastic**: constant variance everywhere (standard)
- **Heteroscedastic**: variance depends on input -- network predicts both $\mu$ and $\sigma^2$
	- Need to ensure $\sigma^2 > 0$: pass second output through squaring function
	- $\mu = f_1[\mathbf{x}, \phi]$, $\sigma^2 = f_2[\mathbf{x}, \phi]^2$
- Loss becomes input-dependent weighting:
$$L = -\sum_i \left(\log\left[\frac{1}{\sqrt{2\pi f_2[\mathbf{x}_i, \phi]^2}}\right] - \frac{(y_i - f_1[\mathbf{x}_i, \phi])^2}{2 f_2[\mathbf{x}_i, \phi]^2}\right)$$

> **Relevance to modern research**: Heteroscedastic models are increasingly relevant for uncertainty quantification in LLMs and for reward modeling in RLHF. Knowing when the model is uncertain vs. confident is crucial for calibration, abstention ("I don't know"), and active learning. Some recent work on conformal prediction and calibration for LLMs builds on exactly this kind of input-dependent uncertainty modeling.

### 5.4 Example 2: Binary Classification -- **Important**

- **Distribution**: Bernoulli with parameter $\lambda \in [0,1]$ (probability of class 1)
$$Pr(y|\lambda) = (1-\lambda)^{1-y} \cdot \lambda^y$$
- Network output can be any real number, but $\lambda$ must be in $[0,1]$
- **Solution**: pass through **logistic sigmoid** $\text{sig}[z] = \frac{1}{1 + \exp[-z]}$
	- Maps $\mathbb{R} \to (0,1)$, smooth, monotonic
	- $\text{sig}[0] = 0.5$, positive inputs $\to$ probabilities $> 0.5$
- $\lambda = \text{sig}[\mathbf{f}[\mathbf{x}, \phi]]$
- **Binary cross-entropy loss**:
$$L[\phi] = \sum_i -(1-y_i)\log[1 - \text{sig}[\mathbf{f}[\mathbf{x}_i, \phi]]] - y_i \log[\text{sig}[\mathbf{f}[\mathbf{x}_i, \phi]]]$$
- Inference: $\hat{y} = 1$ if $\lambda > 0.5$, else $\hat{y} = 0$

> **Where this shows up in LLM research**: Binary classification is everywhere in the LLM pipeline. Reward models in RLHF are often trained with binary cross-entropy (preferred vs. rejected response). Toxicity classifiers, relevance filters, constitutional AI classifiers -- all binary classification with this loss. The sigmoid + BCE pattern is also the foundation of contrastive losses like in CLIP.

### 5.5 Example 3: Multiclass Classification -- **THE Most Important Section**

- **Distribution**: categorical with $K$ parameters $\lambda_1, ..., \lambda_K$ (must be non-negative, sum to 1)
$$Pr(y = k) = \lambda_k$$
- Network has $K$ outputs $f_k[\mathbf{x}, \phi]$ -- unconstrained real numbers
- **Softmax function** maps arbitrary vector to valid probability distribution:
$$\text{softmax}_k[\mathbf{z}] = \frac{\exp[z_k]}{\sum_{k'=1}^{K} \exp[z_{k'}]}$$
	- Exponentials ensure positivity
	- Denominator ensures sum to 1
- **Multiclass cross-entropy loss**:
$$L[\phi] = -\sum_{i=1}^{I} \left(f_{y_i}[\mathbf{x}_i, \phi] - \log\left[\sum_{k'=1}^{K} \exp[f_{k'}[\mathbf{x}_i, \phi]]\right]\right)$$
- Inference: $\hat{y} = \underset{k}{\text{argmax}} \; Pr(y = k | \mathbf{f}[\mathbf{x}, \hat{\phi}])$

> **This is THE loss function for language modeling.** Next-token prediction in every autoregressive LM is multiclass classification where $K$ = vocabulary size (typically 32K-256K). The model outputs logits for each token in the vocab, softmax converts to probabilities, and cross-entropy loss penalizes assigning low probability to the correct next token. When someone says a model was "trained with cross-entropy loss" -- this is what they mean. Every single forward pass during LLM training computes this.

> **Temperature scaling** in LLM sampling is literally modifying the softmax: $\text{softmax}_k[\mathbf{z}/T]$. High temperature $T$ flattens the distribution (more random), low $T$ sharpens it (more deterministic). This falls directly out of the softmax formulation.

> **The log-sum-exp** in the loss ($\log \sum_k \exp[f_k]$) is a famous numerical computation -- in practice you subtract $\max_k f_k$ before exponentiating to avoid overflow. This "log-sum-exp trick" shows up everywhere in ML engineering.

### 5.6 Multiple Outputs

- For multiple predictions, assume independence across output dimensions:
$$Pr(\mathbf{y} | \mathbf{f}[\mathbf{x}, \phi]) = \prod_d Pr(y_d | \mathbf{f}_d[\mathbf{x}, \phi])$$
- NLL becomes sum over both training examples AND output dimensions:
$$L[\phi] = -\sum_i \sum_d \log Pr(y_{id} | \mathbf{f}_d[\mathbf{x}_i, \phi])$$
- Can even mix output types (e.g., von Mises for direction + exponential for magnitude)
	- Independence assumption makes the joint loss just a sum of individual losses

> **In practice for LLMs**: the independence assumption across sequence positions is sidestepped by the autoregressive factorization, but for multi-task models (e.g., predicting text + bounding boxes + classifications), this additive loss structure is exactly how you combine objectives. Multi-modal models like those doing text + image generation use weighted sums of losses over different output modalities.

### 5.7 Cross-Entropy Loss -- **Important Theoretical Connection**

- **Cross-entropy** comes from minimizing KL divergence between empirical data distribution and model distribution
- **KL divergence**: $D_{KL}[q||p] = \int q(z)\log[q(z)]dz - \int q(z)\log[p(z)]dz$
	- First term = negative entropy of data (constant w.r.t. model params)
	- Second term = cross-entropy (the part we optimize)
- Empirical distribution = weighted sum of Dirac deltas at data points
- Plugging in and simplifying: minimizing KL divergence = minimizing NLL
	- **Cross-entropy loss and NLL are the same thing**, just derived from different perspectives
	- NLL: "maximize probability of data"
	- Cross-entropy: "minimize distance between model and empirical distribution"

> **Both perspectives are useful.** The NLL view is more intuitive for training. The KL/cross-entropy view is essential for understanding distillation (student-teacher training), variational inference (VAEs), and KL penalties in RLHF. When you see a "KL penalty" term in PPO for RLHF, it's penalizing the KL divergence between the fine-tuned policy and the reference policy -- directly from this framework.

### 5.8 Additional Topics from Notes

- **Focal loss**: down-weights well-classified examples, helps with class imbalance
	- Originally from object detection (Lin et al. 2017), useful anywhere you have imbalanced classes
- **Robust regression**: Laplace distribution assumption gives mean absolute error instead of MSE
	- More robust to outliers
- **Quantile regression**: predict quantiles rather than mean, useful for uncertainty estimation
- **Mixture density networks**: output parameters of mixture of Gaussians for multimodal predictions
- **Learning to rank**: Plackett-Luce model for ranking tasks (listwise, pointwise, pairwise approaches)

> **Focal loss** has found use in LLM training for handling the long tail of rare tokens. **Mixture density networks** are conceptually related to mixture-of-experts architectures -- though MoE routes inputs to different experts rather than mixing outputs, the probabilistic intuition is similar.

---

## Distribution-to-Loss Cheat Sheet

| **Data Type** | **Domain** | **Distribution** | **Loss** | **Modern Use** |
|---|---|---|---|---|
| Continuous, unbounded | $y \in \mathbb{R}$ | Normal | Least squares (MSE) | Regression heads, diffusion models |
| Continuous, robust | $y \in \mathbb{R}$ | Laplace | Mean absolute error (L1) | Robust training, style transfer |
| Binary discrete | $y \in \{0,1\}$ | Bernoulli | Binary cross-entropy | Reward models, classifiers, CLIP |
| Multiclass discrete | $y \in \{1,...,K\}$ | Categorical | Multiclass cross-entropy | **LLM pre-training**, classification |
| Counts | $y \in \{0,1,2,...\}$ | Poisson | Poisson NLL | Event prediction |
| Bounded continuous | $y \in [0,1]$ | Beta | Beta NLL | Proportion prediction |
| Circular | $y \in (-\pi, \pi]$ | von Mises | von Mises NLL | Direction/angle prediction |
| Ranking | $y \in \text{Perm}$ | Plackett-Luce | Listwise ranking loss | Search ranking, recommendations |

---

## Key Takeaways for LLM Research

1. **Cross-entropy (NLL) is king.** It's the loss function for autoregressive LM training and it comes from a principled probabilistic framework. Understanding it deeply -- including its information-theoretic interpretation as KL divergence -- is non-negotiable.

2. **The softmax-to-categorical pipeline** is the most practically important construction in these chapters. Logits $\to$ softmax $\to$ probabilities $\to$ cross-entropy loss. This is every forward pass in LLM training.

3. **Depth matters for expressivity** but the specific exponential-regions-per-parameter argument is less relevant than the empirical observation that deep models learn better hierarchical representations. The real reason depth works in LLMs is the ability to compose simple operations (attention + FFN) into progressively more abstract representations.

4. **The probabilistic framing enables everything downstream**: uncertainty quantification, sampling strategies (temperature, top-k, nucleus), distillation, RLHF's KL penalties, and calibration. If you only understand the loss as "a number that goes down during training," you're missing the entire toolkit.

5. **Loss function design is an active research area**: focal loss for imbalanced training, DPO/preference losses as alternatives to RL-based RLHF, contrastive losses for embeddings, auxiliary losses for load balancing in MoE -- all follow the same principled framework of "pick a probabilistic model, derive the NLL."
