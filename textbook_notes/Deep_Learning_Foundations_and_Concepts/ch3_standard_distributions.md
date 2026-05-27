# DLFC Chapter 3: Standard Distributions

**Source:** Deep Learning Foundations and Concepts — Christopher Bishop & Hugh Bishop (2024), pp. 66–105

---

## Overview & What Actually Matters

Chapter 3 catalogs the standard parametric distributions (discrete, Gaussian, periodic, exponential family) and closes with nonparametric methods. If you're working on LLMs or generative models, the honest breakdown is:

- **Multinomial distribution + softmax (3.1.3, 3.4):** Absolutely critical. This is the output layer of every autoregressive LM. You use it every day whether you think about it or not.
- **Exponential family / natural parameters / sufficient statistics (3.4):** The theoretical backbone that explains *why* softmax, cross-entropy, and logistic regression take the forms they do. Understanding this deeply pays off in loss design and distribution modeling.
- **Multivariate Gaussian — conditional & marginal (3.2.4–3.2.6):** The linear-Gaussian machinery is the skeleton of Gaussian processes, Kalman filters, VAEs with Gaussian encoders, and diffusion models. The conditional/marginal formulas are used constantly.
- **Mixtures of Gaussians (3.2.9):** The conceptual ancestor of mixture-of-experts (MoE) architectures. The "responsibilities" / posterior mixing weights are exactly the gating mechanism.
- **MLE and its bias (3.2.7):** You do MLE all day in deep learning. The covariance bias discussion foreshadows why we need Bessel's correction and more generally why naive MLE can underestimate uncertainty.
- **Sequential estimation (3.2.8):** The online update formula is the skeleton of SGD and running-mean estimators (BatchNorm, Adam's moment estimates).
- **Nonparametric methods (3.5):** Kernel density estimation and KNN are conceptually relevant to retrieval-augmented generation and in-context learning, but the specific techniques here don't scale and aren't used directly in modern LLM work.
- **Periodic variables / von Mises (3.3):** Niche. Relevant if you work on rotary position embeddings (RoPE uses the same circular geometry) but you don't need the von Mises distribution per se.

---

## 3.1 Discrete Variables

### 3.1.1 Bernoulli Distribution

- $\text{Bern}(x|\mu) = \mu^x(1-\mu)^{1-x}$, $x \in \{0,1\}$
- $\mathbb{E}[x] = \mu$, $\text{var}[x] = \mu(1-\mu)$
- **MLE:** $\mu_{\text{ML}} = m/N$ (fraction of heads) — the sample mean
- Log-likelihood depends on data only through $\sum_n x_n$ — a **sufficient statistic**

**Relevance:** Binary classification output neurons. The binary cross-entropy loss is literally $-\ln p(\mathcal{D}|\mu)$ for Bernoulli observations. Every time you train a binary classifier with BCE loss, you're doing MLE on a Bernoulli model.

### 3.1.2 Binomial Distribution

- $\text{Bin}(m|N,\mu) = \binom{N}{m}\mu^m(1-\mu)^{N-m}$
- Counts number of successes in $N$ i.i.d. Bernoulli trials
- $\mathbb{E}[m] = N\mu$, $\text{var}[m] = N\mu(1-\mu)$

**Relevance:** Not heavily used directly in LLM work. Shows up occasionally in combinatorial arguments about token frequencies and sampling theory.

### 3.1.3 Multinomial Distribution [CRITICAL]

This is the one that matters most for language models.

- **One-hot encoding:** Variable with $K$ states represented as $\mathbf{x} \in \{0,1\}^K$ with $\sum_k x_k = 1$
	- Exactly how we encode tokens in every LM vocabulary
- **Categorical distribution:** $p(\mathbf{x}|\boldsymbol{\mu}) = \prod_{k=1}^K \mu_k^{x_k}$
	- Parameters $\mu_k \geq 0$, $\sum_k \mu_k = 1$
	- $\mathbb{E}[\mathbf{x}|\boldsymbol{\mu}] = \boldsymbol{\mu}$
- **MLE:** $\mu_k^{\text{ML}} = m_k / N$ where $m_k = \sum_n x_{nk}$ is the count of class $k$
	- Sufficient statistics are the counts $\{m_k\}$ — this is why token frequency statistics tell you so much
- **Multinomial distribution:** Joint distribution of counts $(m_1, \ldots, m_K)$:
$$\text{Mult}(m_1,\ldots,m_K|\boldsymbol{\mu},N) = \binom{N}{m_1 \cdots m_K} \prod_{k=1}^K \mu_k^{m_k}$$

**Why this is critical for LLMs:**
- The output of every autoregressive language model is a categorical distribution over the vocabulary at each position
- The training loss (cross-entropy) is the negative log-likelihood of the observed token under this categorical distribution: $-\sum_k y_k \ln \mu_k$ where $y$ is the one-hot target
- Token frequency distributions in natural language (Zipf's law) are empirical multinomials — understanding their structure is essential for vocabulary design, tokenizer construction, and understanding why rare tokens are harder to learn
- **Sampling strategies** (top-k, top-p/nucleus, temperature scaling) are all manipulations of this categorical distribution at inference time. Temperature $\tau$ replaces $\mu_k$ with $\mu_k^{1/\tau} / \sum_j \mu_j^{1/\tau}$

---

## 3.2 The Multivariate Gaussian

### 3.2.1 Geometry of the Gaussian

- Univariate: $\mathcal{N}(x|\mu,\sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left\{-\frac{1}{2\sigma^2}(x-\mu)^2\right\}$
- Multivariate: $\mathcal{N}(\mathbf{x}|\boldsymbol{\mu},\boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}$
- **Mahalanobis distance:** $\Delta^2 = (\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ — reduces to Euclidean when $\boldsymbol{\Sigma} = \mathbf{I}$
- Constant-density surfaces are ellipsoids, axes along eigenvectors $\mathbf{u}_i$ of $\boldsymbol{\Sigma}$, half-lengths $\lambda_i^{1/2}$
- **Eigendecomposition of covariance:** $\boldsymbol{\Sigma} = \sum_i \lambda_i \mathbf{u}_i\mathbf{u}_i^T$
	- Defines a rotated coordinate system $y_i = \mathbf{u}_i^T(\mathbf{x}-\boldsymbol{\mu})$ where the Gaussian factorizes into $D$ independent univariate Gaussians
	- This is PCA — the eigenvectors of the data covariance define the principal components
- $\boldsymbol{\Sigma}$ must be **positive definite** (all $\lambda_i > 0$) for a proper density
	- Positive semi-definite (some $\lambda_i = 0$) gives a degenerate Gaussian confined to a subspace — shows up in latent variable models

**Research connections:**
- The eigendecomposition is literally what PCA does to the data covariance. In LLM interpretability, PCA on activation spaces identifies dominant directions of variation.
- The Mahalanobis distance is used in OOD detection methods: points far from the training distribution in Mahalanobis distance are flagged as out-of-distribution.
- Understanding the geometry of high-dimensional Gaussians is essential for diffusion models, where you start from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ and progressively transform toward the data distribution.

### 3.2.2 Moments

- $\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}$ (verified by symmetry under the substitution $\mathbf{z} = \mathbf{x}-\boldsymbol{\mu}$)
- $\mathbb{E}[\mathbf{x}\mathbf{x}^T] = \boldsymbol{\mu}\boldsymbol{\mu}^T + \boldsymbol{\Sigma}$
- $\text{cov}[\mathbf{x}] = \mathbb{E}[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{x}-\mathbb{E}[\mathbf{x}])^T] = \boldsymbol{\Sigma}$

Straightforward but important: the second-moment matrix $\mathbb{E}[\mathbf{x}\mathbf{x}^T]$ and the covariance $\boldsymbol{\Sigma}$ differ by the outer product of the mean. This distinction matters when computing running statistics (BatchNorm, layer norm).

### 3.2.3 Limitations [Important for Intuition]

Two fundamental issues with Gaussians:

- **Too many parameters:** Full covariance has $D(D+1)/2$ parameters — quadratic in $D$
	- **Diagonal covariance:** $\boldsymbol{\Sigma} = \text{diag}(\sigma_i^2)$ — $2D$ params, axis-aligned ellipsoids
	- **Isotropic covariance:** $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$ — $D+1$ params, spherical contours
	- This is exactly the tradeoff you hit in diffusion models: DDPM uses isotropic Gaussian noise ($\boldsymbol{\Sigma} = \beta_t\mathbf{I}$) because full covariance would be intractable at image resolution
- **Unimodal:** A single Gaussian can't represent multimodal distributions
	- Solved by mixtures (Section 3.2.9) or latent variable models (Chapter 16)

**Practical relevance:** In VAEs, the encoder typically outputs a diagonal Gaussian $q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ — this is a direct compromise between expressiveness and tractability. The isotropic prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ is the simplest choice. The tension between these (prior too simple, posterior too constrained) drives a lot of modern variational inference research.

### 3.2.4 Conditional Distribution [IMPORTANT]

Given a joint Gaussian $\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with $\mathbf{x} = (\mathbf{x}_a, \mathbf{x}_b)$:

- **Precision matrix:** $\boldsymbol{\Lambda} \equiv \boldsymbol{\Sigma}^{-1}$ (inverse covariance)
- **Conditional is Gaussian:** $p(\mathbf{x}_a|\mathbf{x}_b) = \mathcal{N}(\mathbf{x}_a|\boldsymbol{\mu}_{a|b}, \boldsymbol{\Sigma}_{a|b})$

**Expressed via precision (simpler):**
- $\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Lambda}_{aa}^{-1}$
- $\boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{aa}^{-1}\boldsymbol{\Lambda}_{ab}(\mathbf{x}_b - \boldsymbol{\mu}_b)$

**Expressed via covariance:**
- $\boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\mathbf{x}_b - \boldsymbol{\mu}_b)$
- $\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$

Key observations:
- The conditional mean is a **linear function** of $\mathbf{x}_b$ — this is a **linear-Gaussian model**
- The conditional covariance is **independent of $\mathbf{x}_b$** — conditioning changes the mean but not the spread
- The Schur complement $\boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$ shows up everywhere

**"Completing the square" technique:** Bishop uses this extensively — given a quadratic form in the exponent, read off the precision (second-order coefficient matrix) and mean (from the linear term). This algebraic trick is the workhorse for deriving Gaussian conditional/marginal results throughout the book.

**Why this matters enormously:**
- **Gaussian Processes (GPs):** The GP posterior is exactly this conditional — you observe function values at training points ($\mathbf{x}_b$) and condition to get predictions at test points ($\mathbf{x}_a$). The posterior mean is a linear combination of observations, variance reduces where you have data.
- **Diffusion models:** The reverse diffusion step $p(\mathbf{x}_{t-1}|\mathbf{x}_t)$ in DDPM is derived using exactly this conditional Gaussian machinery. The forward process is a linear-Gaussian model, and Bayes' rule for Gaussians gives the reverse.
- **Kalman filters / state-space models:** Mamba and other state-space models use this linear-Gaussian conditioning under the hood.

### 3.2.5 Marginal Distribution

- $p(\mathbf{x}_a) = \int p(\mathbf{x}_a, \mathbf{x}_b)\,d\mathbf{x}_b = \mathcal{N}(\mathbf{x}_a|\boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa})$
- Marginalization just reads off the corresponding block of the covariance matrix
- Simpler in covariance form (vs conditional being simpler in precision form)

**Derivation approach:** Complete the square on $\mathbf{x}_b$ in the joint exponent, integrate out the Gaussian in $\mathbf{x}_b$ (which just gives a constant), and identify what's left as a Gaussian in $\mathbf{x}_a$.

### 3.2.6 Bayes' Theorem for Gaussians [IMPORTANT]

The **linear-Gaussian model** setup:
- Prior: $p(\mathbf{x}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1})$
- Likelihood: $p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y}|\mathbf{A}\mathbf{x}+\mathbf{b}, \mathbf{L}^{-1})$

Then:
- **Marginal:** $p(\mathbf{y}) = \mathcal{N}(\mathbf{y}|\mathbf{A}\boldsymbol{\mu}+\mathbf{b}, \mathbf{L}^{-1}+\mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^T)$
	- Mean propagates linearly, covariance sums (noise + projected prior uncertainty)
- **Posterior:** $p(\mathbf{x}|\mathbf{y}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\Sigma}\{\mathbf{A}^T\mathbf{L}(\mathbf{y}-\mathbf{b})+\boldsymbol{\Lambda}\boldsymbol{\mu}\}, \boldsymbol{\Sigma})$
	- where $\boldsymbol{\Sigma} = (\boldsymbol{\Lambda}+\mathbf{A}^T\mathbf{L}\mathbf{A})^{-1}$
	- **Precisions add** in the posterior — the precision of the posterior is the prior precision plus the data precision
	- This "precisions add" rule is the Gaussian analog of Bayesian updating

**Why this is a workhorse result:**
- **Special case $\mathbf{A}=\mathbf{I}$:** Convolution of two Gaussians — means add, covariances add. This is why the forward diffusion process (repeatedly adding Gaussian noise) produces a Gaussian with known mean and variance at any timestep $t$.
- **Bayesian linear regression:** Set $\mathbf{x}$ = weights, $\mathbf{y}$ = observations, $\mathbf{A}$ = design matrix. The posterior over weights is exactly this formula.
- **VAE reparameterization:** The encoder $q(\mathbf{z}|\mathbf{x})$ and prior $p(\mathbf{z})$ are both Gaussians, and the KL divergence between them has a closed form because of these conjugacy results.
- **Diffusion reverse process:** DDPM derives $p(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ using exactly this Bayes' rule for Gaussians, then marginalizes over $\mathbf{x}_0$ using the model's estimate.

### 3.2.7 Maximum Likelihood

- Log-likelihood: $\ln p(\mathbf{X}|\boldsymbol{\mu},\boldsymbol{\Sigma}) = -\frac{ND}{2}\ln(2\pi) - \frac{N}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}\sum_n (\mathbf{x}_n-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_n-\boldsymbol{\mu})$
- **Sufficient statistics:** $\sum_n \mathbf{x}_n$ and $\sum_n \mathbf{x}_n\mathbf{x}_n^T$ — only these two summaries of the data matter
- **MLE for mean:** $\boldsymbol{\mu}_{\text{ML}} = \frac{1}{N}\sum_n \mathbf{x}_n$ (sample mean — unbiased)
- **MLE for covariance:** $\boldsymbol{\Sigma}_{\text{ML}} = \frac{1}{N}\sum_n (\mathbf{x}_n-\boldsymbol{\mu}_{\text{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\text{ML}})^T$ (biased!)
- **Bias:** $\mathbb{E}[\boldsymbol{\Sigma}_{\text{ML}}] = \frac{N-1}{N}\boldsymbol{\Sigma}$ — systematically underestimates variance
	- Bessel's correction: divide by $N-1$ instead of $N$

**Practical relevance:** The bias of MLE covariance estimation is small for large $N$ but conceptually important — it's a concrete example of how MLE can be overconfident. This same phenomenon (MLE overfitting to the training set) is why we need regularization, early stopping, and all the other tricks in deep learning. The sufficient statistics result also explains why batch statistics in BatchNorm work: you only need the mean and (co)variance of activations, not the individual values.

### 3.2.8 Sequential Estimation

- Online update for the mean: $\boldsymbol{\mu}_{\text{ML}}^{(N)} = \boldsymbol{\mu}_{\text{ML}}^{(N-1)} + \frac{1}{N}(\mathbf{x}_N - \boldsymbol{\mu}_{\text{ML}}^{(N-1)})$
- General pattern: **new estimate = old estimate + learning_rate * (observation - old estimate)**
- Contribution of each new point shrinks as $1/N$

**This is the template for basically all of online learning:**
- **SGD:** $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ — same structure, gradient plays the role of error signal
- **Adam's running moments:** $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ is an exponential moving average version of this, with fixed "learning rate" $(1-\beta_1)$ instead of $1/N$
- **BatchNorm running statistics:** The running mean/variance tracked during training use exactly this exponential moving average pattern
- **Robbins-Monro conditions** ($\sum \eta_t = \infty$, $\sum \eta_t^2 < \infty$) for convergence of stochastic approximation generalize the $1/N$ schedule here

### 3.2.9 Mixtures of Gaussians [IMPORTANT]

- $p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$
- **Mixing coefficients:** $\pi_k \geq 0$, $\sum_k \pi_k = 1$ — they are probabilities
- Sufficient Gaussians can approximate any continuous density to arbitrary accuracy
- **Latent variable interpretation:** $\pi_k = p(k)$ is the prior on component, $\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = p(\mathbf{x}|k)$ is the class-conditional
- **Responsibilities (posterior):** $\gamma_k(\mathbf{x}) = p(k|\mathbf{x}) = \frac{\pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_l \pi_l \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_l,\boldsymbol{\Sigma}_l)}$
- MLE has no closed-form solution due to $\ln\sum$ in the log-likelihood — requires EM (Chapter 15) or gradient methods

**Why this matters — the direct ancestor of MoE:**
- **Mixture of Experts (MoE)** in transformers (GShard, Switch Transformer, Mixtral) is structurally this: a gating network computes mixing weights $\pi_k(\mathbf{x})$ that are input-dependent (unlike the fixed $\pi_k$ here), and each expert computes a component output. The responsibilities $\gamma_k$ are the soft routing weights.
- The key difference from classical GMMs: in MoE, the mixing coefficients are input-dependent (computed by a learned router), and the components are neural networks rather than Gaussians. But the probabilistic structure — prior over components, posterior responsibilities — is identical.
- **Latent variable perspective:** Understanding GMMs as latent variable models (discrete latent $k$ selects the component) is the conceptual bridge to VAEs (continuous latent $\mathbf{z}$) and diffusion models (sequential latent $\mathbf{x}_{1:T}$).
- **EM algorithm connection:** The E-step computes responsibilities, M-step updates parameters. This alternating optimization pattern shows up in many modern training procedures.

---

## 3.3 Periodic Variables

### 3.3.1 Von Mises Distribution

- Standard Gaussian on angular data doesn't work — results depend on choice of origin
- **Invariant mean estimation:** Map observations $\theta_n$ to unit vectors $\mathbf{x}_n = (\cos\theta_n, \sin\theta_n)$, average the vectors, take the angle of the result: $\bar{\theta} = \tan^{-1}\left\{\frac{\sum_n \sin\theta_n}{\sum_n \cos\theta_n}\right\}$
- **Von Mises distribution:** $p(\theta|\theta_0, m) = \frac{1}{2\pi I_0(m)} \exp\{m\cos(\theta-\theta_0)\}$
	- $\theta_0$ = mean direction, $m$ = concentration (analogous to precision)
	- Normalized by $I_0(m)$, the zeroth-order modified Bessel function
	- Derived by conditioning an isotropic 2D Gaussian onto the unit circle
- MLE: $\theta_0^{\text{ML}}$ is the circular mean (above), $m$ found by inverting $A(m) = I_1(m)/I_0(m)$

**Relevance to LLMs — lower than you'd think, but nonzero:**
- **RoPE (Rotary Position Embeddings):** Uses the same unit-circle geometry — positions are encoded as rotations $(\cos m\theta, \sin m\theta)$ applied to query/key pairs. The inner product between rotated vectors depends on the angular difference, exactly like the von Mises kernel $\cos(\theta - \theta_0)$. You don't need the full von Mises distribution theory, but understanding why circular/periodic representations work well for encoding position comes from this same math.
- Beyond RoPE, periodic variables aren't a major concern in mainstream LLM research. If you work on time-series forecasting with LLMs or scientific ML with periodic phenomena, this becomes more relevant.

---

## 3.4 The Exponential Family [IMPORTANT]

### General Form

$$p(\mathbf{x}|\boldsymbol{\eta}) = h(\mathbf{x})\,g(\boldsymbol{\eta})\exp\{\boldsymbol{\eta}^T\mathbf{u}(\mathbf{x})\}$$

- $\boldsymbol{\eta}$ = **natural parameters** (the "right" parameterization)
- $\mathbf{u}(\mathbf{x})$ = **sufficient statistics**
- $g(\boldsymbol{\eta})$ = normalizing coefficient (ensures integration to 1)
- $h(\mathbf{x})$ = base measure

### Key Members

**Bernoulli → logistic sigmoid:**
- Natural parameter: $\eta = \ln\frac{\mu}{1-\mu}$ (the **log-odds** or **logit**)
- Inverse: $\mu = \sigma(\eta) = \frac{1}{1+\exp(-\eta)}$ — the **logistic sigmoid**
- This is why binary classifiers use sigmoid activations — it's the natural link function

**Multinomial → softmax:**
- Natural parameters: $\eta_k = \ln\frac{\mu_k}{1-\sum_j \mu_j}$
- Inverse: $\mu_k = \frac{\exp(\eta_k)}{1+\sum_j \exp(\eta_j)}$ — the **softmax** function
- This is literally the output layer of every language model
- The normalization $g(\boldsymbol{\eta}) = (1+\sum_k \exp(\eta_k))^{-1}$ is the log-partition function whose gradient gives the expected sufficient statistics

**Gaussian:**
- Natural parameters: $\boldsymbol{\eta} = (\mu/\sigma^2, -1/2\sigma^2)^T$
- Sufficient statistics: $\mathbf{u}(x) = (x, x^2)^T$ — need both the sum and sum-of-squares

**Why exponential families are the hidden backbone of ML:**
- **Cross-entropy loss = negative log-likelihood of exponential family.** When you minimize cross-entropy for a categorical output, you're doing MLE in the multinomial exponential family. The softmax output + cross-entropy loss is not an arbitrary design choice — it's the unique consistent combination derived from the exponential family structure.
- **Natural gradients:** The Fisher information matrix (related to $-\nabla^2 \ln g(\boldsymbol{\eta})$) defines a Riemannian metric on parameter space. Natural gradient descent (Amari) follows this geometry. Modern optimizers like K-FAC approximate this.
- **Sufficient statistics tell you what to compute.** The fact that the MLE depends on data only through $\sum_n \mathbf{u}(\mathbf{x}_n)$ means you never need to revisit individual data points once you have the aggregate statistics. This is the theoretical justification for computing batch statistics.
- **Generalized linear models (GLMs):** The link between natural parameters and means via the sigmoid/softmax is the foundation of GLMs, which are the simplest "neural networks" (single linear layer + appropriate activation + appropriate loss).

### 3.4.1 Sufficient Statistics

- MLE condition: $-\nabla \ln g(\boldsymbol{\eta}_{\text{ML}}) = \frac{1}{N}\sum_n \mathbf{u}(\mathbf{x}_n)$
- Left side = $\mathbb{E}[\mathbf{u}(\mathbf{x})]$ under the model — so **MLE matches model moments to data moments**
- This moment-matching interpretation connects to:
	- Method of moments estimation
	- The idea that neural networks learn to match feature statistics (style transfer matches Gram matrices, which are second moments of features)
	- **Score matching** and other implicit density estimation methods

---

## 3.5 Nonparametric Methods

### 3.5.1 Histograms

- Partition space into bins of width $\Delta$, estimate density as $p_i = n_i / (N\Delta_i)$
- Bin width $\Delta$ controls bias-variance tradeoff: too small → noisy, too large → over-smooth
- **Curse of dimensionality:** In $D$ dimensions with $M$ bins per dimension, need $M^D$ total bins — exponential scaling makes histograms useless beyond 2-3 dimensions
- Main value: quick visualization, data can be discarded after histogram computation

**Honestly:** You won't use histograms for density estimation in any modern ML pipeline. But the curse of dimensionality lesson is profound — it's the fundamental reason why we need structured models (neural networks) rather than brute-force nonparametric approaches for high-dimensional data.

### 3.5.2 Kernel Density Estimation

- General formula: $p(\mathbf{x}) = \frac{1}{N}\sum_{n=1}^N \frac{1}{h^D} k\left(\frac{\mathbf{x}-\mathbf{x}_n}{h}\right)$
- Common kernel: Gaussian $k(\mathbf{u}) \propto \exp(-\|\mathbf{u}\|^2/2)$
	- Places a Gaussian bump centered on every data point, averages them
- Bandwidth $h$ = smoothing parameter (same bias-variance tradeoff as bin width)
- No training computation, but inference cost is $O(N)$ per query — doesn't scale

**Conceptual connections (the ideas scale even if the method doesn't):**
- **Attention as kernel smoothing:** Self-attention $\text{Attn}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{q}\mathbf{K}^T/\sqrt{d})\mathbf{V}$ is structurally a kernel density estimate / Nadaraya-Watson regression. The softmax weights play the role of the kernel, keys are the "data points", values are the function values. This is not just an analogy — several papers (Tsai et al., 2019; Katharopoulos et al., 2020) formalize attention as kernel methods.
- **Retrieval-augmented generation (RAG):** Nearest-neighbor lookup in an embedding space + soft weighting by similarity is conceptually kernel density estimation in learned feature space.

### 3.5.3 Nearest-Neighbours

- Fix $K$, grow a sphere around query point until it contains $K$ points, estimate density as $p(\mathbf{x}) = K/(NV)$
- Adaptive smoothing: small $V$ in dense regions (high resolution), large $V$ in sparse regions
- **KNN classifier:** Classify by majority vote of $K$ nearest training points
	- $K=1$: error rate $\leq 2 \times$ Bayes optimal error rate as $N \to \infty$ — remarkable theoretical guarantee
- Both KNN and KDE require storing entire training set — $O(N)$ space and query time

**Conceptual connections:**
- **In-context learning as implicit KNN:** There's a line of research suggesting that transformer in-context learning implicitly performs something like nearest-neighbor lookup in an internal representation space. The examples in the prompt serve as "training data" for an implicit nonparametric predictor.
- **kNN-LM (Khandelwal et al., 2020):** Explicitly augments a neural LM with a KNN lookup over cached representations from the training set. Interpolates between the parametric LM distribution and a nonparametric KNN distribution. This is literally the synthesis Bishop describes in the chapter introduction: "deep learning combines the efficiency of parametric models with the generality of nonparametric methods."
- **Approximate nearest neighbor search** (FAISS, ScaNN, HNSW) makes KNN tractable at scale and is critical infrastructure for RAG systems.

---

## Chapter-Level Takeaways for LLM Research

**The big picture Bishop sets up (and states explicitly at the end):** Parametric models are efficient but inflexible. Nonparametric models are flexible but don't scale. Deep learning gets the best of both worlds — neural networks are parametric (fixed parameter count at inference), but with enough parameters they can approximate the flexibility of nonparametric methods. This framing is surprisingly prescient for understanding scaling laws: as you add more parameters, you're buying more of that nonparametric flexibility while keeping the computational structure of a parametric model.

**What to internalize from this chapter:**
1. The multinomial + softmax + cross-entropy pipeline is not ad hoc — it falls out uniquely from exponential family theory
2. Gaussian conditional/marginal/Bayes formulas are the engine behind GPs, diffusion models, VAEs, and Kalman-style state-space models
3. Mixtures of Gaussians → mixture of experts: same latent variable structure, same responsibilities/gating
4. Sequential estimation → online learning → SGD/Adam
5. KDE / KNN → attention-as-kernel-smoothing, RAG, kNN-LM

**What you can safely skim:** The detailed derivations of partitioned matrix inversions (3.60-3.75) — know the results, trust the algebra. The von Mises derivation details unless you're specifically working on positional encodings. The binomial distribution.
