# Chapter 3: Probability — Multivariate Models

**TL;DR from a frontier-lab perspective**: this chapter is the bread-and-butter probability that every ML practitioner uses, but for *modern LLM research* most of the content is foundational rather than load-bearing. The pieces that actually matter day-to-day in current LLM work: **exponential family** (the entire transformer head is essentially a softmax = exponential family), **mixture models / latent variables** (mixture-of-experts, mixture-of-depths, speculative decoding samplers), **Markov chains / autoregressive factorization** (this *is* the LLM modeling assumption), **conditional independence + PGM intuition** (used implicitly when thinking about long-context attention, RLHF reward modeling, etc.), and **maxent derivation** (justifies softmax + RLHF KL-regularized policies). The heavy Gaussian math (MVN conditioning, linear-Gaussian systems, Kalman) shows up far less in pure language modeling but is everywhere in **diffusion models**, **Bayesian deep learning**, **flow matching**, **uncertainty quantification**, and the recent wave of **continuous-time / state-space models** (Mamba's roots, S4).

---

## 3.1 Joint distributions for multiple random variables

### 3.1.1 Covariance
- **Definition**: $\text{Cov}[X,Y] \triangleq \mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$.
- **Covariance matrix** of vector $\boldsymbol{x}\in\mathbb{R}^D$: $\boldsymbol{\Sigma} = \mathbb{E}[(\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^\top]$, symmetric PSD.
- **Key identity**: $\mathbb{E}[\boldsymbol{x}\boldsymbol{x}^\top] = \boldsymbol{\Sigma} + \boldsymbol{\mu}\boldsymbol{\mu}^\top$.
- **Linear transform**: $\text{Cov}[\mathbf{A}\boldsymbol{x}+\boldsymbol{b}] = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top$. Used constantly in stats and Bayesian DL (e.g. propagating uncertainty through a linear layer in Laplace approximations).

### 3.1.2 Correlation
- **Pearson**: $\rho = \text{Cov}[X,Y]/\sqrt{\mathbb{V}[X]\mathbb{V}[Y]}\in[-1,1]$.
- **Only captures linearity**. Correlation = 0 even for $Y = X^2$ where $X\sim U(-1,1)$.
- **Correlation matrix**: $\text{corr}(\boldsymbol{x}) = (\text{diag}(\mathbf{K}_{xx}))^{-1/2}\mathbf{K}_{xx}(\text{diag}(\mathbf{K}_{xx}))^{-1/2}$.

### 3.1.3 Uncorrelated ≠ independent
- Independent $\Rightarrow$ uncorrelated. **Converse false**. For real dependence-detection use **mutual information** (Ch. 6).
- **Why this matters for LLMs**: every "linear probing" study leans on this. Just because a probe can't linearly recover a feature doesn't mean it isn't there — MI / nonlinear probes are the right tool.

### 3.1.4 Correlation ≠ causation
- Spurious correlation, hidden common causes (ice cream vs. crime via weather).
- **Modern relevance**: causal-flavored eval matters for **interpretability / mech interp** (does ablating circuit X actually *cause* the behavior?) and **RLHF reward hacking** (model exploits correlations in reward signal that don't reflect true preferences).

### 3.1.5 Simpson's paradox
- Trends within subgroups reverse when pooled (Iris sepal width vs length, COVID CFR Italy vs China).
- **Direct relevance to LLM eval**: aggregate benchmark scores can hide subgroup reversals (e.g. average MMLU up, but math down). Always slice your evals.

---

## 3.2 The multivariate Gaussian (MVN) — important but more for *generative / Bayesian* DL than LLMs

### 3.2.1 Definition
$$\mathcal{N}(\boldsymbol{y}|\boldsymbol{\mu},\boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\left[-\tfrac{1}{2}(\boldsymbol{y}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{y}-\boldsymbol{\mu})\right]$$

- **Three flavors of covariance**:
    - **Full**: $D(D+1)/2$ params, arbitrarily oriented ellipsoidal contours.
    - **Diagonal**: $D$ params, axis-aligned.
    - **Spherical / isotropic**: $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$, one parameter.
- **Why isotropic matters**: this is the noise model used in **DDPM / score-based diffusion**, **rectified flow**, **flow matching**, **VAEs**. Isotropic Gaussian noise in pixel/latent space is the workhorse of modern image/video generation.

### 3.2.2 Mahalanobis distance
- $\Delta^2 = (\boldsymbol{y}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{y}-\boldsymbol{\mu})$.
- **Geometric picture via eigendecomp** $\boldsymbol{\Sigma}=\sum_d\lambda_d\boldsymbol{u}_d\boldsymbol{u}_d^\top$: contours of constant density are ellipsoids; eigenvectors give axes, eigenvalues give lengths.
- **Use cases**:
    - **OOD detection** for deployed models — Mahalanobis distance on penultimate features is still a competitive baseline.
    - **Whitening** of activations / gradients (preconditioned optimizers, K-FAC, Shampoo all rely on this geometry).

### 3.2.3 Marginals and conditionals of an MVN — **important formula**
Partition $\boldsymbol{y}=(\boldsymbol{y}_1,\boldsymbol{y}_2)$ with block $\boldsymbol{\mu},\boldsymbol{\Sigma}$, precision $\boldsymbol{\Lambda}=\boldsymbol{\Sigma}^{-1}$.

- **Marginals**: just take the corresponding blocks: $p(\boldsymbol{y}_1)=\mathcal{N}(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_{11})$.
- **Conditional**:
$$p(\boldsymbol{y}_1|\boldsymbol{y}_2) = \mathcal{N}(\boldsymbol{\mu}_{1|2},\boldsymbol{\Sigma}_{1|2})$$
$$\boldsymbol{\mu}_{1|2}=\boldsymbol{\mu}_1+\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\boldsymbol{y}_2-\boldsymbol{\mu}_2)$$
$$\boldsymbol{\Sigma}_{1|2}=\boldsymbol{\Sigma}_{11}-\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$
- **Conditional mean is linear in $\boldsymbol{y}_2$, conditional covariance is constant** (this is *the* property that makes Gaussians tractable).
- **Where this shows up**:
    - **Kalman filter / smoother** — drives state-space models, recently revived (S4, Mamba — though Mamba's selectivity breaks linear-Gaussian structure, the LTI variants are exact LGS).
    - **Gaussian processes** for hyperparameter optimization, Bayesian optimization (still standard for LLM hyperparam sweeps and arch search).
    - **DDPM posterior** $p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$ is derived from exactly these formulas.
    - **Linear attention / random features** approximations.

### 3.2.4 Conditioning a 2d Gaussian — concrete intuition
- Observing one component **shrinks** the conditional mean of the other toward $\rho(y_2-\mu_2)$ and **reduces variance** by factor $(1-\rho^2)$.

### 3.2.5 Missing value imputation
- Use $p(\boldsymbol{y}_h|\boldsymbol{y}_v,\boldsymbol{\theta})$ to fill missing entries. **Posterior mean** = MMSE estimate. **Multiple imputation** = sampling from posterior, gives downstream uncertainty.
- **Practical analog in modern ML**: masked autoencoders (MAE) and BERT-style masked LM are literally a deep-net version of this, but with a learned non-Gaussian conditional.

---

## 3.3 Linear Gaussian systems — the conjugacy engine

Setup:
$$p(\boldsymbol{z})=\mathcal{N}(\boldsymbol{\mu}_z,\boldsymbol{\Sigma}_z),\quad p(\boldsymbol{y}|\boldsymbol{z})=\mathcal{N}(\mathbf{W}\boldsymbol{z}+\boldsymbol{b},\boldsymbol{\Sigma}_y)$$

### 3.3.1 Bayes rule for Gaussians — **memorize this**
$$\boldsymbol{\Sigma}_{z|y}^{-1} = \boldsymbol{\Sigma}_z^{-1} + \mathbf{W}^\top\boldsymbol{\Sigma}_y^{-1}\mathbf{W}$$
$$\boldsymbol{\mu}_{z|y} = \boldsymbol{\Sigma}_{z|y}\left[\mathbf{W}^\top\boldsymbol{\Sigma}_y^{-1}(\boldsymbol{y}-\boldsymbol{b}) + \boldsymbol{\Sigma}_z^{-1}\boldsymbol{\mu}_z\right]$$
- **Posterior precision = prior precision + data precision** (additive in precision space).
- **Gaussian prior + Gaussian likelihood ⇒ Gaussian posterior** ⇒ **conjugate prior**.
- **Marginal likelihood**: $p(\boldsymbol{y})=\mathcal{N}(\mathbf{W}\boldsymbol{\mu}_z+\boldsymbol{b},\,\boldsymbol{\Sigma}_y+\mathbf{W}\boldsymbol{\Sigma}_z\mathbf{W}^\top)$.

### 3.3.2 Derivation via completing the square
- Vector identity: $\boldsymbol{x}^\top\mathbf{A}\boldsymbol{x}+\boldsymbol{x}^\top\boldsymbol{b}+c = (\boldsymbol{x}-\boldsymbol{h})^\top\mathbf{A}(\boldsymbol{x}-\boldsymbol{h})+k$, $\boldsymbol{h}=-\tfrac{1}{2}\mathbf{A}^{-1}\boldsymbol{b}$, $k=c-\tfrac{1}{4}\boldsymbol{b}^\top\mathbf{A}^{-1}\boldsymbol{b}$.
- The trick behind half the derivations in ML — diffusion ELBO terms, variational inference, GP marginal likelihood.

### 3.3.3 Inferring an unknown scalar — shrinkage intuition
- Posterior: $\lambda_N = \lambda_0 + N\lambda_y$, $\mu_N = (N\lambda_y\bar{y} + \lambda_0\mu_0)/\lambda_N$.
- **Posterior mean is a convex combination of prior and MLE**, weighted by relative precision.
- **Shrinkage form**: $\mu_1 = \mu_0 + (y-\mu_0)\frac{\Sigma_0}{\Sigma_y+\Sigma_0}$.
- **Strong prior → posterior near prior; weak prior → posterior near MLE**.
- **Sequential updating** trivially works (Kalman style).
- **Modern relevance**: conceptual basis of **EMA in training** (Polyak averaging, EMA teacher in self-distillation, DDPM/EMA weights at inference), **Bayesian fine-tuning**, **continual learning regularizers** (EWC = a Laplace-Gaussian posterior used as prior for the next task).

### 3.3.4–3.3.5 Vector inference / sensor fusion
- Multiple noisy observations: posterior precision sums; posterior mean weighted by sensor precisions.
- Each sensor contributes more in directions it's more reliable.
- **Modern relevance**: **multi-modal fusion** (vision + language reward models, CLIP-like training), **mixture-of-experts gating**, **ensembling**. The lesson — combine sources weighted by precision — survives even when the math doesn't.

---

## 3.4 The exponential family — **one of the most important sections for LLM research**

### 3.4.1 Definition
$$p(\boldsymbol{y}|\boldsymbol{\eta}) = h(\boldsymbol{y})\exp[\boldsymbol{\eta}^\top\mathcal{T}(\boldsymbol{y}) - A(\boldsymbol{\eta})]$$
- $\boldsymbol{\eta}$: **natural / canonical parameters**.
- $\mathcal{T}(\boldsymbol{y})$: **sufficient statistics**.
- $A(\boldsymbol{\eta}) = \log Z(\boldsymbol{\eta})$: **log partition function** — convex in $\boldsymbol{\eta}$.
- $h(\boldsymbol{y})$: base measure.
- **Minimal representation**: $\boldsymbol{\eta}$ uniquely identifies the distribution; no linear redundancy in $\mathcal{T}$.

### 3.4.2 Bernoulli example
- $\text{Ber}(y|\mu) = \exp[y\log(\mu/(1-\mu)) + \log(1-\mu)]$.
- $\eta = \log(\mu/(1-\mu)) = $ **logit**.
- $\mu = \sigma(\eta) = 1/(1+e^{-\eta})$.
- **This is exactly what every classifier head does**: logit-to-prob = exponential-family canonical link inverse.
- **Categorical / softmax** is the multivariate version. **Every token-level prediction in an LLM is exponential family**.

### 3.4.3 Log partition = cumulant generating function — **huge property**
$$\nabla A(\boldsymbol{\eta}) = \mathbb{E}[\mathcal{T}(\boldsymbol{y})], \quad \nabla^2 A(\boldsymbol{\eta}) = \text{Cov}[\mathcal{T}(\boldsymbol{y})]$$
- **Consequences**:
    - $A$ is convex $\Rightarrow$ log likelihood is **concave in $\boldsymbol{\eta}$** $\Rightarrow$ unique global MLE.
    - **Direct link to softmax loss gradient**: $\nabla \log Z = \mathbb{E}[\text{softmax}]$. Gradient of cross-entropy = (predicted prob) − (one-hot target) falls straight out.
- **Why this matters constantly**:
    - **RLHF / DPO** reward modeling: Bradley-Terry is exponential family; $\log Z$ derivatives give expected reward differences.
    - **Energy-based models, score matching**: $A(\eta)$ being intractable is the whole reason these methods exist.
    - **InfoNCE / contrastive learning**: literally a softmax / log-partition over negatives.

### 3.4.4 Maximum entropy derivation — **conceptually big**
- Given moment constraints $\int p(\boldsymbol{x})f_k(\boldsymbol{x})d\boldsymbol{x}=F_k$, the **maximum entropy** distribution is:
$$p(\boldsymbol{x}) \propto q(\boldsymbol{x})\exp\left(-\sum_k\lambda_k f_k(\boldsymbol{x})\right)$$
- **This is exactly an exponential family**. Lagrange multipliers = natural parameters; $f_k$ = sufficient statistics.
- **Examples**: mean+second moment → Gaussian; mean on $[0,\infty)$ → exponential; nothing → uniform.
- **Why this is huge for LLMs**:
    - **RLHF KL-regularized RL**: the optimal policy under KL-to-reference + reward maximization is $\pi^*(y|x)\propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$ — *literally* a max-entropy / exponential-family policy. This is the foundation of **DPO**, **IPO**, **GRPO**, and basically every modern alignment algorithm.
    - **Softmax temperature sampling**: $p(y)\propto \exp(\text{logits}/T)$ is exactly a maxent reweighting.
    - **EBM / Boltzmann policies** in RL.
    - **Diffusion guidance**: classifier-free guidance $\propto p(\boldsymbol{x})^{1-w}p(\boldsymbol{x}|c)^w$ is exponential reweighting of densities.

---

## 3.5 Mixture models — **very important, high modern relevance**

### 3.5.1 Gaussian mixture models (GMM)
$$p(\boldsymbol{y}|\boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{y}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$$
- **Universal approximator** for smooth densities.
- **Latent variable form**: $z\sim\text{Cat}(\boldsymbol{\pi})$, $\boldsymbol{y}|z=k\sim p_k$.
- **Responsibility** (posterior over cluster id):
$$r_{nk}=p(z_n=k|\boldsymbol{y}_n,\boldsymbol{\theta})=\frac{\pi_k p(\boldsymbol{y}_n|\boldsymbol{\theta}_k)}{\sum_{k'}\pi_{k'}p(\boldsymbol{y}_n|\boldsymbol{\theta}_{k'})}$$
- **Hard vs soft clustering**: argmax vs full posterior.
- With uniform $\pi$, $\Sigma_k=\mathbf{I}$ → **K-means**: $z_n=\arg\min_k\|\boldsymbol{y}_n-\hat{\boldsymbol{\mu}}_k\|^2$.

```python
# soft assignment (responsibilities) = softmax over log-likelihoods
log_resp = log_pi[None, :] + log_gauss_pdf(y, mu, Sigma)  # (N, K)
r = softmax(log_resp, axis=-1)
```

- **Modern LLM relevance — very high**:
    - **Mixture of Experts (MoE)**: GShard, Switch, Mixtral, DeepSeek-MoE — conditional mixtures where gate computes $\pi_k(x)$ (usually top-k for sparsity).
    - **Mixture of Depths** (Raposo et al.): token-level routing to different depths.
    - **Speculative decoding** uses mixture-style sampling: draft + verifier; posterior over which token to accept ≈ responsibility.
    - **Mixture-of-softmaxes** (Yang et al. 2017) breaks the softmax bottleneck.
    - **VQ-VAE / discrete latent LMs** — discrete latent z, like a BMM with neural emissions.

### 3.5.2 Bernoulli mixture models (BMM)
- For binary data: $p(\boldsymbol{y}|z=k,\boldsymbol{\theta})=\prod_d \mu_{dk}^{y_d}(1-\mu_{dk})^{1-y_d}$.
- Fit via EM. MNIST demo: clusters into digit-like prototypes without labels.
- **Modern relevance**: less direct, but conceptually linked to **discrete latent codes**, **binary tokenization**, **probabilistic circuits**.

---

## 3.6 Probabilistic graphical models (PGMs) — **conceptually essential**

### 3.6.1 Representation
- **Bayesian network / DAG**: nodes = RVs, edges = direct dependencies.
- **Ordered Markov property**: $Y_i \perp \mathbf{Y}_{\text{pred}(i)\setminus \text{pa}(i)} | \mathbf{Y}_{\text{pa}(i)}$.
- **Joint factorization**: $p(\mathbf{Y}_{1:N}) = \prod_i p(Y_i|\mathbf{Y}_{\text{pa}(i)})$.
- **CPDs / CPTs**: each node has a conditional distribution / table.

**Water sprinkler example**: $p(C,S,R,W)=p(C)p(S|C)p(R|C)p(W|S,R)$. Demonstrates **explaining away** (Berkson's paradox): wet grass + sprinkler on reduces $p(\text{rain})$.

**Markov chain — the LLM bedrock**:
$$p(\boldsymbol{y}_{1:T}) = \prod_{t=1}^T p(y_t|\boldsymbol{y}_{1:t-1})$$
- **No-assumption form**: full chain rule; param count explodes with $t$.
- **First-order Markov**: $p(y_t|y_{t-1})$ — **transition kernel / state-transition matrix** $A_{jk}=p(y_t=k|y_{t-1}=j)$.
- **Homogeneous / stationary** = **parameter tying** (same transition all $t$).
- **$M$-th order**: trigram if $M=2$, bigram if $M=1$.
- Large vocab → table infeasible → **low-rank** or **neural** parameterization → **neural language model**.

**This section IS the LLM**:
- Transformer LMs make zero Markov assumption (condition on full history) — the no-assumption chain rule, but with neural parameterization that shares params across positions = deep version of parameter tying.
- Mamba / S4 / linear attention reintroduce a Markovian *latent state* (RNN flavor), trading exactness for $O(T)$ compute.
- KV-cache = caching the conditioning context.
- Causal masking enforces the autoregressive factorization.

### 3.6.2 Inference
- Standard PGM ops: marginal $p(Y_i|Y_j=y_j)$.
- **Explaining away / Berkson's paradox**: observing a common effect couples its causes (negative correlation induced).
- Sprinkler ex: $p(R=1|W=1)=0.7079$, $p(R=1|W=1,S=1)=0.3204$.
- **Modern relevance**: implicit in reasoning chains, mech interp ablations.

### 3.6.3 Learning
- Treat unknown CPD params as latent nodes; do Bayesian inference.
- **iid + exchangeable** = standard ML setup.
- **Plate notation**: visual shorthand for repeated nodes.
- GMM as PGM: global $\boldsymbol{\pi},\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k$ plate of $K$, per-datapoint $z_n,\boldsymbol{y}_n$ plate of $N$.

---

## Synthesis: what to actually internalize for current LLM research

### Tier 1 — used directly, every day
- **Exponential family + log-partition gradients** → softmax, cross-entropy, RLHF, contrastive loss. If you don't have this in your bones you'll re-derive things badly.
- **Max-entropy / KL-regularized objectives** → DPO, GRPO, PPO+KL, classifier-free guidance, temperature sampling. **The single most important conceptual thread for alignment work.**
- **Markov / autoregressive factorization** → underlies LMs, diffusion (reverse Markov chain), state-space models.
- **Mixture models + responsibilities** → MoE, speculative decoding, gated routing.

### Tier 2 — adjacent / supporting
- **MVN conditioning + linear-Gaussian Bayes rule** → diffusion derivations (DDPM posterior, score matching), GPs for hyperparam search, Laplace approximations.
- **Sensor fusion / precision-weighted averaging** → EMA, ensembling, multimodal fusion.
- **Conditional independence intuition** → long-context dependencies, attention sparsity, interp circuit analysis.

### Tier 3 — useful background
- **Mahalanobis distance** → OOD detection, K-FAC / second-order optimization geometry.
- **Simpson's paradox / explaining away** → eval design, interp sanity checks.
- **PGM plate notation** → for reading classical Bayesian ML papers.
- **Bernoulli mixture** → niche unless you work on discrete latents.

### Honest take
- Gaussian-heavy parts (3.2, 3.3) feel beautiful and are crucial for **diffusion + Bayesian DL + uncertainty + state-space models**, but you can do strong LLM research without inverting a covariance matrix. They become essential the moment you touch image/audio/video gen or want to put error bars on anything.
- **Exponential family + maxent** is criminally under-taught in DL courses but gets reused constantly in modern alignment math. Spend time here.
- **Mixture models / latent variables** got a massive revival via MoE — EM intuition (responsibilities = soft assignments) maps directly onto gating networks.
- **PGMs as a formalism** are out of fashion, but the *thinking style* (factorize, identify conditional independencies, reason about posteriors) is exactly what good researchers use when designing training pipelines, reward models, world models, multi-agent setups.
