# MML Chapter 6: Probability and Distributions

**Source:** Mathematics for Machine Learning — Deisenroth, Faisal, Ong (2020), pp. 178–224

---

## Overview

Chapter 6 builds the probabilistic foundation for all of Part II (regression, PCA, GMMs, SVMs). The central idea is that probability generalizes logical reasoning to handle uncertainty — both in data and in model parameters. The chapter covers the construction of probability spaces, discrete and continuous distributions, the rules of probability, summary statistics, the Gaussian distribution in depth, the exponential family, and change-of-variable techniques.

---

## 6.1 Construction of a Probability Space

### 6.1.1 Philosophical Basis

Probability theory extends Boolean logic to *plausible reasoning*. Jaynes (2003) identifies three desiderata: plausibilities are real numbers, they follow common sense, and reasoning must be consistent (non-contradictory, honest, reproducible). Satisfying these uniquely determines the rules of probability up to a monotone transformation.

Two interpretations:
- **Bayesian**: probability = degree of belief (subjective prior knowledge)
- **Frequentist**: probability = limiting relative frequency of events

### 6.1.2 Probability and Random Variables

A **probability space** has three components:
- **Sample space Ω**: all possible outcomes of an experiment
- **Event space A**: subsets of Ω (power set for discrete cases)
- **Probability P**: function A → [0,1] satisfying P(Ω) = 1

A **random variable** X: Ω → T is a function mapping outcomes to a *target space* T (usually ℝ or ℝ^D). The **distribution** (or *law*) of X is the function P_X(S) = P(X ∈ S) that assigns probabilities to subsets of T.

### 6.1.3 Probability vs. Statistics

- *Probability*: given a model, derive what happens
- *Statistics*: given observations, infer the underlying model
- Machine learning sits close to statistics: construct a model that best explains observed data, while caring about **generalization** to unseen data

---

## 6.2 Discrete and Continuous Probabilities

### 6.2.1 Discrete Probabilities

For a discrete target space T, the **probability mass function (pmf)** P(X = x) gives point probabilities. For two random variables X and Y, the **joint probability** is:

$$P(X = x_i, Y = y_j) = \frac{n_{ij}}{N}$$

- **Marginal probability**: sum (integrate) over the other variable → p(x) = Σ_y p(x, y)
- **Conditional probability**: p(y | x) = p(x, y) / p(x)

Discrete distributions model **categorical variables** (e.g., token identities).

### 6.2.2 Continuous Probabilities

A **probability density function (pdf)** f: ℝ^D → ℝ satisfies:
1. f(x) ≥ 0 for all x
2. ∫ f(x)dx = 1

The probability of an interval is P(a ≤ X ≤ b) = ∫_a^b f(x)dx. The **cumulative distribution function (cdf)** F_X(x) = P(X ≤ x) is the integral of the pdf.

Key subtlety: for continuous variables, P(X = x) = 0 for any specific value; probability only makes sense for intervals. The pdf itself can exceed 1.

### 6.2.3 Contrasting Discrete and Continuous

| | Discrete | Continuous |
|---|---|---|
| Point probability | P(X = x) — pmf | 0 (measure zero) |
| Density | pmf | pdf f(x) |
| Interval probability | Not standard | P(a ≤ X ≤ b) — cdf |

---

## 6.3 Sum Rule, Product Rule, and Bayes' Theorem

All of probability theory reduces to two rules:

**Sum rule (marginalization)**:
$$p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{y})\, d\mathbf{y}$$

**Product rule**:
$$p(\mathbf{x}, \mathbf{y}) = p(\mathbf{y} \mid \mathbf{x})\, p(\mathbf{x})$$

**Bayes' theorem** follows directly from the product rule (applying it in both orderings):

$$\underbrace{p(\mathbf{x} \mid \mathbf{y})}_{\text{posterior}} = \frac{\overbrace{p(\mathbf{y} \mid \mathbf{x})}^{\text{likelihood}}\, \overbrace{p(\mathbf{x})}^{\text{prior}}}{\underbrace{p(\mathbf{y})}_{\text{evidence}}}$$

- **Prior** p(x): belief about latent variable x before observing data
- **Likelihood** p(y | x): how probable is the data y given x
- **Evidence** p(y) = ∫ p(y | x) p(x) dx: marginal likelihood, normalizes the posterior
- **Posterior** p(x | y): updated belief after observing y

The evidence is often intractable to compute (requires integrating over all x), which motivates approximate inference techniques.

> **LLM connection**: Bayesian reasoning is the conceptual backbone of language modeling. A language model implicitly defines p(token | context). RLHF and fine-tuning can be viewed as updating a prior (pre-trained model) toward a posterior (aligned model) given new data (human feedback). Bayesian model selection (Section 8.6) is directly related to how researchers evaluate language model architectures.

---

## 6.4 Summary Statistics and Independence

### 6.4.1 Means and Covariances

**Expected value** of a function g of X:
$$\mathbb{E}_X[g(x)] = \int g(x)\, p(x)\, dx \quad \text{(continuous)}$$

The expected value is a **linear operator**: E[ag + bh] = aE[g] + bE[h].

**Mean** (Definition 6.4): E_X[x] = μ ∈ ℝ^D — the "center of mass" of the distribution.

**Variance** (Definition 6.7):
$$\mathbb{V}_X[\mathbf{x}] = \mathbb{E}_X[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top]$$

The D×D **covariance matrix** Σ has:
- Diagonal: variances of each dimension
- Off-diagonal: cross-covariances Cov[x_i, x_j]
- Always symmetric and positive semidefinite

**Correlation**: normalized covariance, corr[x, y] = Cov[x, y] / √(V[x] V[y]) ∈ [-1, 1]

Three equivalent expressions for scalar variance:
1. Standard: V[x] = E[(x - μ)²]
2. Raw-score: V[x] = E[x²] − (E[x])² (useful for one-pass computation)
3. Pairwise differences: (1/N²) Σ_{i,j} (x_i - x_j)² = 2 × raw-score expression

**Empirical mean and covariance** (finite dataset of N samples):
$$\bar{\mathbf{x}} = \frac{1}{N}\sum_{n=1}^N \mathbf{x}_n, \qquad \boldsymbol{\Sigma} = \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n - \bar{\mathbf{x}})(\mathbf{x}_n - \bar{\mathbf{x}})^\top$$

Note: the N denominator gives a biased estimate; N-1 gives the unbiased correction.

### 6.4.4 Sums and Affine Transformations

For y = Ax + b with X having mean μ and covariance Σ:
- E[y] = Aμ + b
- V[y] = AΣA^T

This is critical for understanding how neural network layers transform distributions.

### 6.4.5 Statistical Independence

X and Y are **statistically independent** iff p(x, y) = p(x)p(y).

Implications:
- p(y | x) = p(y)  and  p(x | y) = p(x)
- V[x + y] = V[x] + V[y]
- Cov[x, y] = 0

**Important caveat**: zero covariance does NOT imply independence (covariance only captures linear dependence). Example: if Y = X², then Cov[X, Y] = 0 even though Y is entirely determined by X.

**Conditional independence** (Definition 6.11): X ⊥⊥ Y | Z iff p(x, y | z) = p(x | z)p(y | z). Used extensively in graphical models (Chapter 8.5).

### 6.4.6 Inner Products of Random Variables

Random variables as vectors in a function space:
- Inner product: ⟨X, Y⟩ = Cov[x, y] (for zero-mean variables)
- "Length" of X: ||X|| = σ(x) (standard deviation)
- Angle between X and Y: cos θ = corr[x, y] (correlation = cosine similarity)

Uncorrelated random variables are **orthogonal** in this space — the Pythagorean theorem applies: V[X + Y] = V[X] + V[Y].

---

## 6.5 Gaussian Distribution

The Gaussian (normal) distribution is the most important distribution in ML due to:
1. Analytically tractable marginals and conditionals
2. Closed-form Bayesian updates
3. The Central Limit Theorem: sums of i.i.d. variables converge to Gaussian
4. Linear/affine transformations preserve Gaussianity

**Univariate Gaussian**:
$$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Multivariate Gaussian** X ~ N(μ, Σ), x ∈ ℝ^D:
$$p(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = (2\pi)^{-D/2} |\boldsymbol{\Sigma}|^{-1/2} \exp\!\left(-\tfrac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

**Standard normal**: μ = 0, Σ = I.

### 6.5.1 Marginals and Conditionals of Gaussians are Gaussians

Given a joint Gaussian:
$$p(\mathbf{x}, \mathbf{y}) = \mathcal{N}\!\left(\begin{bmatrix}\boldsymbol{\mu}_x \\ \boldsymbol{\mu}_y\end{bmatrix},\, \begin{bmatrix}\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\ \boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}\end{bmatrix}\right)$$

**Marginal** (sum rule): p(x) = N(x | μ_x, Σ_xx)

**Conditional** (product rule + Bayes):
$$p(\mathbf{x} \mid \mathbf{y}) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_{x|y},\, \boldsymbol{\Sigma}_{x|y})$$
$$\boldsymbol{\mu}_{x|y} = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}(\mathbf{y} - \boldsymbol{\mu}_y)$$
$$\boldsymbol{\Sigma}_{x|y} = \boldsymbol{\Sigma}_{xx} - \boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx}$$

The conditional mean is a linear function of the observed y. The **Schur complement** Σ_{x|y} = Σ_xx − Σ_xy Σ_yy^{-1} Σ_yx shows variance is always reduced by conditioning.

Applications: Kalman filter (state estimation), Gaussian processes, latent linear Gaussian models (PPCA).

### 6.5.2 Product of Gaussian Densities

The product of two Gaussians N(x | a, A) · N(x | b, B) is an unnormalized Gaussian cN(x | c, C) where:
$$\mathbf{C} = (\mathbf{A}^{-1} + \mathbf{B}^{-1})^{-1}, \quad \mathbf{c} = \mathbf{C}(\mathbf{A}^{-1}\mathbf{a} + \mathbf{B}^{-1}\mathbf{b})$$

This is used when computing Bayesian posteriors: posterior ∝ likelihood × prior.

### 6.5.3 Sums and Linear Transformations

- If X ~ N(μ_x, Σ_x) and Y ~ N(μ_y, Σ_y) are **independent**, then:  
  X + Y ~ N(μ_x + μ_y, Σ_x + Σ_y)

- Linear transformation y = Ax:  
  p(y) = N(y | Aμ, AΣA^T)

- Any linear/affine transformation of a Gaussian is Gaussian.

### 6.5.4 Sampling from Multivariate Gaussians

To sample from N(μ, Σ):
1. Sample z ~ N(0, I) (standard normal)
2. Compute Cholesky decomposition: Σ = AA^T
3. Return x = Az + μ

> **LLM connection — Gaussian distributions are foundational to many LLM techniques:**
>
> - **Variational autoencoders (VAEs)**: the latent space is modeled as a Gaussian with reparameterization z = μ + σ·ε, ε ~ N(0, I). This allows backpropagation through sampling. Related to Section 6.5.4.
>
> - **Diffusion models (e.g., DDPM)**: the forward process gradually adds Gaussian noise; the reverse denoising process learns to invert this. The Gaussian closure properties (sums remain Gaussian) make the math tractable.
>
> - **Gaussian noise in training**: dropout, weight noise, and gradient noise are often modeled as Gaussian perturbations. The linear transformation result (Section 6.5.3) explains why such perturbations remain benign.
>
> - **Attention score distributions**: while softmax outputs are not Gaussian, the pre-softmax logits often approximately follow Gaussian distributions, motivating scaled dot-product attention (dividing by √d_k to control variance).

---

## 6.6 Conjugacy and the Exponential Family

### Desiderata for Distributions in ML

1. Closure under Bayes' rule (applying operations returns same family type)
2. Fixed number of parameters regardless of data size
3. Maximum likelihood estimation behaves well

The **exponential family** satisfies all three.

### 6.6.1 Conjugacy

**Definition 6.13**: A prior is *conjugate* for a likelihood if the posterior is in the same family as the prior.

This allows Bayesian updates to be done by simply updating distribution parameters — no integration needed.

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Bernoulli / Binomial | Beta | Beta |
| Gaussian (mean) | Gaussian | Gaussian |
| Gaussian (variance) | Inverse-Gamma | Inverse-Gamma |
| Multinomial | Dirichlet | Dirichlet |

**Beta-Bernoulli conjugacy**: if θ ~ Beta(α, β) and we observe h heads in N flips:
$$p(\theta \mid h) \propto \text{Beta}(\theta \mid h + \alpha, N - h + \beta)$$

The parameters α, β act as "pseudo-counts" — prior observations of the outcome.

> **LLM connection**: The **Dirichlet-Multinomial** conjugacy is the probabilistic basis for topic models (LDA). While modern LLMs don't use topic models directly, understanding conjugacy helps interpret mixture models and the "prior" built into pretrained weights.

### 6.6.2 Sufficient Statistics

A **sufficient statistic** φ(x) captures all information in the data x needed to estimate θ. Formally, p(x | θ) = h(x) g_θ(φ(x)) (Fisher-Neyman factorization).

For Gaussians: sufficient statistics are the sample mean and sample covariance — once computed, the raw data adds no further information about μ and σ².

### 6.6.3 Exponential Family

The exponential family is a parameterized class of distributions of the form:
$$p(\mathbf{x} \mid \boldsymbol{\theta}) = h(\mathbf{x})\exp\!\big(\langle\boldsymbol{\theta}, \boldsymbol{\phi}(\mathbf{x})\rangle - A(\boldsymbol{\theta})\big)$$

- **φ(x)**: sufficient statistics (features of the data)
- **θ**: natural parameters
- **A(θ)**: log-partition function (normalization)
- **h(x)**: base measure

Examples:
- Gaussian: φ(x) = [x, x²]^T, θ = [μ/σ², -1/(2σ²)]^T
- Bernoulli: θ = log(μ/(1-μ)) — the **logit/sigmoid** relationship  
  (μ = 1/(1 + exp(−θ)) — the **sigmoid function**)
- Binomial, Beta, Dirichlet, Poisson, Gamma all belong

**Key properties**:
1. Maximum likelihood estimation of θ sets the empirical sufficient statistics equal to their expected values
2. The log-likelihood is concave → efficient optimization (Chapter 7)
3. Every exponential family member has a conjugate prior within the exponential family
4. Finite-dimensional sufficient statistics — model complexity doesn't grow with data

> **LLM connection — exponential family is deeply embedded in modern deep learning:**
>
> - **Softmax output layer**: the categorical distribution (generalization of Bernoulli to K classes) is an exponential family member. The softmax p(y=k | x) = exp(θ_k^T x) / Z is directly the exponential family form, with natural parameters θ_k^T x and log-partition function log Z.
>
> - **Cross-entropy loss**: the standard training loss for language models is the negative log-likelihood of the categorical distribution. Its concavity (a property of all exponential family log-likelihoods) helps explain why gradient descent converges reliably.
>
> - **Sigmoid activation**: the natural-parameter-to-mean-parameter transformation for the Bernoulli is the sigmoid function (eq. 6.118). This is the probabilistic reason sigmoid is the "right" activation for binary outputs.
>
> - **Normalizing flows**: generative models that apply invertible transformations to simple base distributions (often Gaussian). The change-of-variable formula (Section 6.7) is the mathematical engine.
>
> - **Temperature scaling and top-k/top-p sampling**: modifying the natural parameters θ by a temperature τ (divide by τ) directly controls the sharpness of the categorical distribution. This is standard in LLM decoding.

---

## 6.7 Change of Variables / Inverse Transform

### 6.7.1 Distribution Function Technique

Given X with pdf f_X(x) and a transformation Y = U(X), find the pdf of Y by:
1. Find F_Y(y) = P(Y ≤ y)
2. Differentiate: f_Y(y) = d/dy F_Y(y)

### 6.7.2 Change-of-Variables Formula

For an invertible function U with univariate X:
$$f_Y(y) = f_X(U^{-1}(y)) \cdot \left|\frac{d}{dy} U^{-1}(y)\right|$$

The absolute value of the derivative adjusts for how U stretches or compresses the probability mass.

**Multivariate case (Theorem 6.16)**: for y = U(x) invertible and differentiable:
$$f_Y(\mathbf{y}) = f_X(U^{-1}(\mathbf{y})) \cdot \left|\det\!\left(\frac{\partial}{\partial \mathbf{y}} U^{-1}(\mathbf{y})\right)\right|$$

The **Jacobian determinant** measures how U distorts volume.

**Probability integral transform (Theorem 6.15)**: If Y = F_X(X) where F_X is the cdf of X, then Y ~ Uniform[0,1]. This gives a universal sampling algorithm:
1. Sample u ~ Uniform[0,1]
2. Return x = F_X^{-1}(u)

> **LLM connection — change of variables is central to generative modeling:**
>
> - **Normalizing flows**: learn an invertible neural network f_θ that maps simple noise z ~ p_z(z) to data x = f_θ(z). Training maximizes log p(x) = log p_z(f_θ^{-1}(x)) + log|det J|, where J is the Jacobian of f_θ^{-1}. The change-of-variable formula (eq. 6.143/6.144) IS the normalizing flow objective.
>
> - **Reparameterization trick** (VAEs): sampling z ~ N(μ, σ²) is equivalent to sampling ε ~ N(0,1) and computing z = μ + σ·ε. This is a change of variables that enables backpropagation through sampling.
>
> - **Diffusion models**: the forward noising process is a sequence of change-of-variable steps, each adding controlled Gaussian noise. The reverse process learns the inverse Jacobian effectively.
>
> - **Inverse transform sampling** underpins nucleus sampling and temperature sampling in LLMs: sample a uniform draw, then apply the inverse cdf of the token distribution.

---

## Key Takeaways and Connections

| Concept | Core Idea | LLM Relevance |
|---|---|---|
| Bayes' theorem | Invert likelihood to get posterior | Foundation of RLHF, Bayesian fine-tuning |
| Gaussian distribution | Closed-form marginals/conditionals | VAEs, diffusion models, attention scaling |
| Exponential family | Unified framework for named distributions | Softmax, cross-entropy, temperature |
| Sufficient statistics | Data compression without information loss | Feature representations in transformers |
| Conjugate priors | Posterior has same form as prior | Bayesian interpretations of fine-tuning |
| Change of variables | Transform distributions via invertible maps | Normalizing flows, reparameterization trick |
| Independence / Cov | Measuring and encoding variable relationships | Attention patterns, independence assumptions |
| Sum/product rules | Two rules that generate all of probability | KL divergence, ELBO, all of variational inference |

---

## Connections to LLM Research: Summary

The probabilistic framework in this chapter underlies virtually all of modern deep learning for language:

1. **Autoregressive language modeling** defines p(token | context) using the product rule: p(sentence) = ∏_t p(token_t | token_1,...,token_{t-1}).

2. **The softmax output** is the exponential family (categorical distribution) with natural parameters equal to the logits.

3. **Cross-entropy training** minimizes negative log-likelihood of the categorical — a direct application of MLE for exponential family models.

4. **Variational inference / ELBO** (used in VAEs and some LLM regularization schemes) decomposes the intractable evidence p(y) into a tractable objective using the KL divergence between distributions — built entirely on the sum rule and product rule.

5. **RLHF** treats the reward model as a likelihood and uses Bayes' theorem to update the policy (prior) toward the posterior over desirable behavior.

6. **Uncertainty quantification** in LLMs uses entropy of the output distribution (a function of the sufficient statistics of the categorical) to estimate model confidence.

7. **Diffusion models** (e.g., used in image generation; related in structure to audio/text generation) use Gaussian noise addition and the change-of-variables Jacobian to define a tractable training objective.
