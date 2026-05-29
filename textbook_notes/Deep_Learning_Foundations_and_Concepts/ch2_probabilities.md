# DLFC Chapter 2: Probabilities

**Source:** Deep Learning Foundations and Concepts — Christopher Bishop & Hugh Bishop (2024), pp. 25–58

---

## Overview

Chapter 2 is the probabilistic backbone of the entire book and, arguably, the most important foundational chapter for LLM research. It builds from the sum and product rules through Bayes' theorem, the Gaussian distribution and maximum likelihood, density transformations, information theory (entropy, KL divergence, mutual information), and finally the Bayesian perspective on learning. Nearly every concept here reappears — often in disguised form — in transformer training, RLHF, diffusion models, and variational inference.

---

## 2.1 The Rules of Probability

### 2.1.1–2.1.2 Sum Rule, Product Rule

The two axioms from which everything else follows:

- **Sum rule (marginalization):** $p(X) = \sum_Y p(X, Y)$
- **Product rule:** $p(X, Y) = p(Y|X)\, p(X)$

These are derived from a frequency-counting argument on a joint distribution table (Figure 2.4). The joint probability $p(X = x_i, Y = y_j) = n_{ij} / N$; the marginal $p(X = x_i) = c_i / N$ is obtained by summing over $Y$; the conditional $p(Y = y_j | X = x_i) = n_{ij} / c_i$ is the fraction within a column.

**LLM connection:** Autoregressive language modeling is literally a repeated application of the product rule. A sequence probability is factored as:

$$p(x_1, \ldots, x_T) = p(x_1) \cdot p(x_2|x_1) \cdot p(x_3|x_1, x_2) \cdots p(x_T|x_1,\ldots,x_{T-1})$$

Each transformer forward pass computes one conditional factor in this chain.

### 2.1.3 Bayes' Theorem

From the symmetry $p(X, Y) = p(Y, X)$ and the product rule:

$$p(Y|X) = \frac{p(X|Y)\, p(Y)}{p(X)}$$

The denominator $p(X) = \sum_Y p(X|Y)p(Y)$ serves as a normalization constant.

**LLM connections beyond the book:**
- **In-context learning has a Bayesian interpretation.** Xie et al. (2022) showed that transformer in-context learning can be understood as implicit Bayesian inference: the model maintains a posterior over latent "concepts" (tasks) given the prompt, and each new in-context example updates this posterior. The pretraining distribution over documents provides the prior.
- **Posterior inference in RLHF.** The reward model in RLHF can be seen through a Bayesian lens: human preference labels are noisy observations, and the reward model estimates a posterior over the quality of responses.
- **Bayesian model selection** is conceptually what happens when we choose between model sizes or training configurations — we're implicitly weighing evidence for different model classes, though in practice we use validation loss rather than computing the marginal likelihood.

### 2.1.5 Prior and Posterior Probabilities

Bishop illustrates with the medical screening example: a test with 90% sensitivity and 97% specificity still yields only a 23% positive predictive value when prevalence is 1%. The key insight: the prior $p(C) = 0.01$ strongly modulates the posterior.

**Research connection:** This base-rate neglect problem is directly relevant to LLM calibration research. When an LLM assigns high confidence to a rare class, the calibrated probability should account for the base rate. This is also the intuition behind why rare failure modes (jailbreaks, hallucinations on uncommon topics) are hard to detect even with high-accuracy classifiers.

### 2.1.6 Independent Variables

$p(X, Y) = p(X)p(Y)$ implies $p(Y|X) = p(Y)$ — observing $X$ tells you nothing about $Y$.

**Research connection:** The conditional independence assumption in autoregressive models is that each token is independent of all future tokens given the past. This is trivially satisfied by construction (causal masking), but the question of what conditional independencies a transformer *learns* internally is a core question in mechanistic interpretability.

---

## 2.2 Probability Densities

### Key Definitions

For continuous variables, probability is defined via densities:
- $p(x) \geq 0$, $\int_{-\infty}^{\infty} p(x)\,dx = 1$
- Probability of an interval: $p(x \in (a,b)) = \int_a^b p(x)\,dx$
- CDF: $P(z) = \int_{-\infty}^z p(x)\,dx$, with $P'(x) = p(x)$

For multivariate: $p(\mathbf{x})\,d\mathbf{x}$ is the probability in an infinitesimal volume. The sum and product rules become integrals.

### 2.2.1 Example Distributions

- **Uniform** on $(c, d)$: $p(x) = \frac{1}{d - c}$
- **Exponential:** $p(x|\lambda) = \lambda \exp(-\lambda x),\quad x \geq 0$
- **Laplace:** $p(x|\mu, \gamma) = \frac{1}{2\gamma} \exp\!\left(-\frac{|x - \mu|}{\gamma}\right)$
- **Dirac delta:** $p(x|\mu) = \delta(x - \mu)$ — used to construct the empirical distribution $p(x|\mathcal{D}) = \frac{1}{N} \sum_n \delta(x - x_n)$

**Research connection:** The Laplace distribution is the implicit prior when using $L_1$ regularization (analogous to how Gaussian prior gives $L_2$/weight decay). Some modern training recipes explore $L_1$-style sparsity penalties on activations — directly related to the Laplace distribution's heavier tails compared to the Gaussian.

### 2.2.2 Expectations and Covariances

- **Expectation:** $\mathbb{E}[f] = \sum_x p(x) f(x)$ (discrete) or $\int p(x) f(x)\,dx$ (continuous)
- **Monte Carlo approximation:** $\mathbb{E}[f] \approx \frac{1}{N} \sum_n f(x_n)$ — this is the foundation of all stochastic gradient methods
- **Conditional expectation:** $\mathbb{E}_x[f|y] = \sum_x p(x|y) f(x)$
- **Variance:** $\text{var}[f] = \mathbb{E}\!\left[(f - \mathbb{E}[f])^2\right] = \mathbb{E}[f^2] - \mathbb{E}[f]^2$
- **Covariance:** $\text{cov}[x, y] = \mathbb{E}[xy] - \mathbb{E}[x]\mathbb{E}[y]$

**LLM connection:** Every gradient computation in SGD is a Monte Carlo estimate of $\mathbb{E}[\nabla \mathcal{L}]$. The variance of this estimate determines training stability. Techniques like gradient accumulation, large batch training, and gradient clipping all manage the variance of this Monte Carlo estimator. The bias-variance tradeoff in gradient estimation is why batch size matters.

---

## 2.3 The Gaussian Distribution

### 2.3.1 Mean and Variance

$$\mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

- $\mathbb{E}[x] = \mu$ (first moment)
- $\mathbb{E}[x^2] = \mu^2 + \sigma^2$ (second moment)
- $\text{var}[x] = \sigma^2$
- Precision: $\beta = 1/\sigma^2$

**Research connection:** The Gaussian is not just a convenient choice — it arises as the maximum entropy distribution given fixed mean and variance (shown in Section 2.5.4). This is why it appears as the noise distribution in diffusion models: it maximizes entropy subject to the variance schedule, meaning it's the "least informative" noise you can add. This is also why Gaussian initialization of weights (He/Xavier) is natural — it's the maximum entropy distribution with the right variance scale.

### 2.3.2 Likelihood Function

Given i.i.d. data $\mathbf{x} = (x_1, \ldots, x_N)$:

$$p(\mathbf{x}|\mu, \sigma^2) = \prod_n \mathcal{N}(x_n|\mu, \sigma^2)$$

Log-likelihood:

$$\ln p(\mathbf{x}|\mu, \sigma^2) = -\frac{1}{2\sigma^2} \sum_n (x_n - \mu)^2 - \frac{N}{2} \ln \sigma^2 - \frac{N}{2} \ln(2\pi)$$

MLE solutions:
- $\mu_\text{ML} = \frac{1}{N} \sum_n x_n$ (sample mean)
- $\sigma^2_\text{ML} = \frac{1}{N} \sum_n (x_n - \mu_\text{ML})^2$ (sample variance)

**LLM connection:** The cross-entropy loss used in LLM training is a log-likelihood objective. When we minimize cross-entropy, we are performing maximum likelihood estimation on the parameters of a categorical distribution (softmax output) over the vocabulary. The connection: for Gaussian-distributed continuous data, MLE gives MSE loss; for categorically-distributed discrete data, MLE gives cross-entropy loss. Both are instances of the same principle.

### 2.3.3 Bias of Maximum Likelihood

- $\mathbb{E}[\mu_\text{ML}] = \mu$ (unbiased)
- $\mathbb{E}[\sigma^2_\text{ML}] = \frac{N-1}{N} \sigma^2$ (biased — systematically underestimates variance)
- Bessel's correction: use $\frac{1}{N-1}$ instead of $\frac{1}{N}$

**Research connection:** MLE bias becomes severe in overparameterized models. In LLMs, the bias of MLE manifests as overfitting — the model assigns high likelihood to training data but underestimates the variance of the true distribution, leading to overconfident predictions. This is the probabilistic explanation for why large models memorize training data and why regularization (weight decay, dropout) is necessary.

### 2.3.4 Linear Regression (Probabilistic View)

Assuming Gaussian noise: $p(t|x, \mathbf{w}, \sigma^2) = \mathcal{N}(t|y(x, \mathbf{w}), \sigma^2)$. The log-likelihood reduces to the sum-of-squares error function:

$$E(\mathbf{w}) = \frac{1}{2} \sum_n \{y(x_n, \mathbf{w}) - t_n\}^2$$

**Key insight:** The MSE loss function is not an arbitrary choice — it is the MLE objective under the assumption of Gaussian noise. Similarly, cross-entropy loss is MLE under a categorical (multinomial) assumption. This means choosing a loss function is equivalent to choosing a noise model.

---

## 2.4 Transformation of Densities

### Change of Variables (1D)

For $x = g(y)$:

$$p_y(y) = p_x(g(y)) \left|\frac{dg}{dy}\right|$$

The Jacobian factor $|dg/dy|$ accounts for how the transformation stretches or compresses probability mass. Crucially, the mode of a density is not invariant under nonlinear transformations — the mode of $p_y(y)$ is generally not at $g^{-1}(\hat{x})$ where $\hat{x}$ is the mode of $p_x(x)$.

### Multivariate Generalization

For $\mathbf{x} = g(\mathbf{y})$ in $D$ dimensions:

$$p_y(\mathbf{y}) = p_x(\mathbf{x})\, |\det \mathbf{J}|, \quad \text{where } J_{ij} = \frac{\partial g_i}{\partial y_j}$$

**LLM connections beyond the book:**
- **Normalizing flows** (Chapter 18) are built entirely on this formula. A flow is a sequence of invertible transformations, each with a tractable Jacobian determinant, that maps a simple base distribution (Gaussian) to a complex data distribution.
- **The reparameterization trick** in VAEs is a change of variables: instead of sampling $z \sim q(z|x)$, write $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. This allows gradients to flow through the sampling operation. The same trick appears in policy gradient methods (REINFORCE vs. pathwise gradients).
- **Flow matching** (used in Stable Diffusion 3 and similar models) defines a time-dependent vector field that transports a noise distribution to a data distribution. The ODE perspective is a continuous version of the change-of-variables formula.
- **The Jacobian determinant** appears in the training objective of continuous normalizing flows (neural ODEs). Computing it efficiently via the Hutchinson trace estimator is a key algorithmic contribution.

---

## 2.5 Information Theory

> 🎨 **Interactive visual companion:** [`ch2.5_information_theory_visual.html`](./ch2.5_information_theory_visual.html) — a 3Blue1Brown-style explorable for this section (surprise, entropy, coding, microstates, differential/max entropy, KL divergence + forward-vs-reverse, mutual information). Includes a tabbed **deep-dive** working through the full step-by-step computations of **RLHF/PPO, DPO, knowledge distillation, and the VAE ELBO** with color-coded interactive toys. Drag the bars and push the sliders; open in a browser.

### 2.5.1 Entropy

Information content of observing outcome $x$: $h(x) = -\log_2 p(x)$

**Entropy** — the expected information content:

$$H[x] = -\sum_x p(x) \ln p(x)$$

Properties:
- $H \geq 0$ for discrete distributions
- $H$ is maximized by the uniform distribution ($H = \ln M$ for $M$ states)
- Noiseless coding theorem (Shannon, 1948): entropy is the lower bound on average code length

**LLM connections:**
- **Perplexity** $= \exp(H) = \exp(-\text{mean log-likelihood})$. When we report perplexity on a test set, we are exponentiating the cross-entropy, which is an upper bound on the true entropy of the language. A perplexity of 20 means the model is "as uncertain as if choosing uniformly among 20 tokens" at each step.
- **Tokenization and entropy.** BPE tokenization can be understood as an approximate entropy coding: frequent subwords get short tokens (analogous to short codewords for probable events). The information-theoretic optimality of the tokenizer directly affects model efficiency — a suboptimal tokenizer wastes bits.
- **Entropy regularization** in RL: PPO adds an entropy bonus to encourage exploration by preventing the policy from collapsing to a deterministic strategy. The entropy of the action distribution is explicitly optimized.
- **Temperature scaling** directly controls the entropy of the output distribution. $T > 1$ increases entropy (more uniform, more "creative"); $T < 1$ decreases entropy (more peaked, more "deterministic"). At $T \to 0$, we get argmax (greedy decoding); at $T \to \infty$, we get uniform sampling.

### 2.5.2 Physics Perspective

Entropy arises from counting microstates (multiplicity $W = N! / \prod_i n_i!$). Using Stirling's approximation in the limit:

$$H = -\sum_i p_i \ln p_i$$

This connects to the Boltzmann distribution in statistical mechanics. The maximum entropy state is the equilibrium state.

**Research connection:** Energy-based models (EBMs) in ML directly use the Boltzmann distribution: $p(\mathbf{x}) \propto \exp(-E(\mathbf{x})/T)$. The partition function (normalization constant) is intractable in high dimensions, which is why EBMs are hard to train and why contrastive divergence, score matching, and noise-contrastive estimation were developed as alternatives.

### 2.5.3 Differential Entropy

For continuous variables:

$$H[x] = -\int p(x) \ln p(x)\,dx$$

Key differences from discrete entropy: can be negative (e.g., for narrow Gaussians with $\sigma^2 < \frac{1}{2\pi e}$), and is not invariant under change of variables.

Gaussian differential entropy: $H[x] = \frac{1}{2}\!\left(1 + \ln(2\pi\sigma^2)\right)$

### 2.5.4 Maximum Entropy

The Gaussian is the maximum entropy distribution subject to fixed mean and variance. This is shown via calculus of variations with Lagrange multipliers:

Maximize $-\int p(x) \ln p(x)\,dx$ subject to normalization, fixed mean $\mu$, and fixed variance $\sigma^2$.

Setting the functional derivative to zero gives $p(x) = \exp\!\left\{-1 + \lambda_1 + \lambda_2 x + \lambda_3 (x - \mu)^2\right\}$, which is the Gaussian.

**Research connection:** Maximum entropy provides a principled justification for distribution choices in generative models. When you have constraints (e.g., moments) but no other information, the Gaussian is the least biased choice. This principle extends to why the softmax output (which is the maximum entropy distribution subject to linear constraints from logits) is the natural choice for classification heads.

### 2.5.5 Kullback-Leibler Divergence (CRITICAL SECTION)

[Six (and a half) intuitions for KL divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence)
[KL-Divergence as an Objective Function](https://timvieira.github.io/blog/kl-divergence-as-an-objective-function/)

$$\text{KL}(p \| q) = -\int p(x) \ln\frac{q(x)}{p(x)}\,dx$$

Properties:
- $\text{KL}(p \| q) \geq 0$ (proved via Jensen's inequality on $-\ln$, which is convex)
- $\text{KL}(p \| q) = 0$ iff $p = q$
- **Asymmetric:** $\text{KL}(p \| q) \neq \text{KL}(q \| p)$ in general

**Connection to MLE:** Minimizing $\text{KL}(p_\text{data} \| q_\theta)$ over $\theta$ is equivalent to maximizing the log-likelihood:

$$\text{KL}(p \| q) \approx \frac{1}{N} \sum_n \left\{-\ln q(x_n|\theta) + \ln p(x_n)\right\}$$

The second term is independent of $\theta$, so minimizing KL = maximizing log-likelihood. This is the theoretical justification for why cross-entropy training works.

**LLM connections — this is where it really matters:**
- **RLHF / PPO:** The KL penalty in RLHF is $\beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})$, preventing the policy $\pi_\theta$ from drifting too far from the reference model $\pi_\text{ref}$. The asymmetry matters: $\text{KL}(\pi_\theta \| \pi_\text{ref})$ penalizes the policy for putting mass where the reference doesn't, which prevents mode-seeking behavior and encourages the fine-tuned model to stay within the "support" of the reference.
- **DPO (Direct Preference Optimization):** The entire DPO derivation starts from the KL-constrained reward maximization problem: $\max_\pi \mathbb{E}[r(x, y)] - \beta \cdot \text{KL}(\pi \| \pi_\text{ref})$. The closed-form solution $\pi^*(y|x) \propto \pi_\text{ref}(y|x) \exp(r(x,y)/\beta)$ is what allows DPO to bypass explicit reward modeling.
- **Knowledge distillation:** Training a student model to match a teacher's output distribution minimizes $\text{KL}(p_\text{teacher} \| q_\text{student})$. The choice of direction matters: $\text{KL}(\text{teacher} \| \text{student})$ is "mean-seeking" (student covers all teacher modes), while $\text{KL}(\text{student} \| \text{teacher})$ is "mode-seeking" (student focuses on teacher's peaks). Most distillation uses the former.
- **VAE training:** The ELBO contains a KL term: $\text{KL}(q(z|x) \| p(z))$, which regularizes the encoder's approximate posterior toward the prior. When this KL collapses to zero ("posterior collapse"), the latent space becomes uninformative — a major failure mode in VAE-based text generation.
- **Forward vs. Reverse KL:**
  - $\text{KL}(p \| q)$ = "forward KL" — mode-covering, used in MLE/distillation
  - $\text{KL}(q \| p)$ = "reverse KL" — mode-seeking, used in variational inference
  - In RLHF, using $\text{KL}(\pi \| \pi_\text{ref})$ is reverse KL: it prevents the new policy from exploring regions where the reference has low probability, which is a safety-motivated choice.
- **f-divergences:** KL is one member of the $f$-divergence family. Recent work explores alternatives like $\chi^2$ divergence (used in some distributional robustness formulations) and Jensen-Shannon divergence (the GAN training objective is related to JSD).

### 2.5.6 Conditional Entropy

$$H[y|x] = -\iint p(y, x) \ln p(y|x)\,dy\,dx$$

Chain rule: $H[x, y] = H[y|x] + H[x]$

**Research connection:** Conditional entropy $H[Y|X]$ measures the remaining uncertainty in $Y$ after observing $X$. In language modeling, $H[\text{next\_token} | \text{context}]$ is exactly what we're trying to minimize. The gap between $H[Y|X]$ (with full context) and $H[Y]$ (unconditional) quantifies how much the context helps — this is the mutual information, discussed next.

### 2.5.7 Mutual Information

$$I[x, y] = \text{KL}(p(x, y) \| p(x)p(y)) = H[x] - H[x|y] = H[y] - H[y|x]$$

- $I \geq 0$, with equality iff $x$ and $y$ are independent
- Symmetric: $I[x, y] = I[y, x]$
- Measures the reduction in uncertainty about one variable given the other

**LLM connections beyond the book:**
- **Contrastive learning (InfoNCE/CLIP):** The InfoNCE loss is a lower bound on mutual information between representations. CLIP maximizes $I(\text{image\_repr}, \text{text\_repr})$, learning aligned representations. This is why CLIP embeddings are useful for text-to-image retrieval — they maximize the information shared between modalities.
- **Information bottleneck:** The information bottleneck framework (Tishby et al., 2000) says optimal representations should maximize $I(\text{representation}, \text{output})$ while minimizing $I(\text{representation}, \text{input})$. This provides a theoretical lens for understanding what transformer layers learn: early layers compress input (reduce $I$ with input), later layers extract task-relevant features (maximize $I$ with output).
- **Probing and representational analysis:** When we probe intermediate LLM representations for linguistic properties (syntax, semantics), we're estimating mutual information between the representation and the linguistic label. High $I$ means the representation encodes that property.
- **Pointwise mutual information (PMI):** $\text{PMI}(x, y) = \ln \frac{p(x, y)}{p(x)p(y)}$ is the log-ratio version. Word2Vec's skip-gram with negative sampling implicitly factorizes a PMI matrix. PMI analysis of attention patterns reveals which token pairs have surprising co-attention.

---

## 2.6 Bayesian Probabilities

> 🎨 **Interactive visual companion:** [`ch2.6_bayesian_probabilities_visual.html`](./ch2.6_bayesian_probabilities_visual.html) — explorable lessons for **2.6.1** (posterior = likelihood × prior; point estimate vs full predictive with click-to-add Bayesian regression; weight averaging & mode connectivity for ensembles/soups/SWA) and **2.6.2** (regularization as MAP with L1/L2 geometry; **AdamW vs Adam+L2** coupled-decay trajectories; **LoRA** as a low-rank prior via SVD truncation). Drag and slide; open in a browser.

### 2.6.1 Model Parameters

The Bayesian framework treats model parameters $\mathbf{w}$ as random variables with a prior $p(\mathbf{w})$. After observing data $\mathcal{D}$, the posterior is:

$$p(\mathbf{w}|\mathcal{D}) = \frac{p(\mathcal{D}|\mathbf{w})\, p(\mathbf{w})}{p(\mathcal{D})}$$

where $p(\mathcal{D}|\mathbf{w})$ is the likelihood and $p(\mathcal{D}) = \int p(\mathcal{D}|\mathbf{w})\, p(\mathbf{w})\,d\mathbf{w}$ is the model evidence (marginal likelihood).

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

**LLM connection:** In standard LLM training, we do MLE (or MAP with weight decay), finding a single point estimate $\mathbf{w}_\text{ML}$. The Bayesian perspective suggests we should integrate over $\mathbf{w}$, but this is intractable for billion-parameter models. However:
- **Stochastic Weight Averaging (SWA)** and **model soups** (averaging checkpoints) approximate Bayesian model averaging by combining multiple points in weight space.
- **Ensembles** of LLMs can be interpreted as a discrete approximation to the Bayesian posterior integral.
- **LoRA ensembles** — training multiple LoRA adapters and averaging predictions — is a computationally feasible approximation to Bayesian inference in the fine-tuning regime.

### 2.6.2 Regularization as MAP

Taking negative logs of the posterior:

$$-\ln p(\mathbf{w}|\mathcal{D}) = -\ln p(\mathcal{D}|\mathbf{w}) - \ln p(\mathbf{w}) + \text{const}$$

With a Gaussian prior $p(\mathbf{w}|s) = \prod_i \mathcal{N}(w_i|0, s^2)$:

$$-\ln p(\mathbf{w}|\mathcal{D}) = -\ln p(\mathcal{D}|\mathbf{w}) + \frac{1}{2s^2} \sum_i w_i^2 + \text{const}$$

This is exactly the loss function with $L_2$ regularization (weight decay). The regularization coefficient $\lambda = \frac{1}{2s^2}$ is determined by the prior variance.

**LLM connections:**
- **AdamW** uses decoupled weight decay, which is exactly this MAP objective where the prior is a zero-mean Gaussian. The weight decay coefficient controls how tight the prior is — larger decay = smaller prior variance = stronger belief that weights should be near zero.
- **The difference between $L_2$ regularization in Adam vs. decoupled weight decay in AdamW matters.** In standard Adam with $L_2$, the regularization gradient is scaled by the adaptive learning rate, effectively weakening the regularization for parameters with large gradients. AdamW applies weight decay separately, maintaining the intended prior strength uniformly.
- **LoRA** can be interpreted as imposing a low-rank structural prior on the weight update: $\Delta W = BA$, where $B$ and $A$ are low-rank matrices. This is a much stronger prior than isotropic Gaussian — it constrains updates to a low-dimensional subspace, which is empirically justified by the observation that fine-tuning updates have low intrinsic dimensionality (Aghajanyan et al., 2021).

### 2.6.3 Bayesian Machine Learning

Full Bayesian prediction integrates over parameters:

$$p(t|x, \mathcal{D}) = \int p(t|x, \mathbf{w})\, p(\mathbf{w}|\mathcal{D})\,d\mathbf{w}$$

This automatically handles model complexity: the marginal likelihood $p(\mathcal{D}) = \int p(\mathcal{D}|\mathbf{w})\, p(\mathbf{w})\,d\mathbf{w}$ is typically highest for models of intermediate complexity (Occam's razor).

**Research connections beyond the book:**
- **Bayesian deep learning at scale** remains an open problem. Approaches like MC Dropout (Gal & Ghahramani, 2016), Laplace approximation (Daxberger et al., 2021), and SWAG (Maddox et al., 2019) provide cheap uncertainty estimates, but none fully solve the problem for LLM-scale models.
- **Uncertainty quantification** in LLMs is critical for deployment: knowing when the model "doesn't know" is essential for safe systems. The Bayesian framework provides the principled answer (epistemic uncertainty from the posterior over $\mathbf{w}$, aleatoric uncertainty from the likelihood), but practical methods (conformal prediction, ensemble disagreement, token-level entropy) are approximations.
- **Scaling laws** can be viewed through a Bayesian lens: as you add more data, the posterior over parameters concentrates, and the predictive distribution improves. The power-law relationship between compute/data and loss may reflect properties of the Bayesian posterior in high dimensions.

---

## Summary: Priority Ranking for LLM Research

| Section | Topic | Priority | Key Application |
|---------|-------|----------|----------------|
| 2.5.5 | KL Divergence | **Critical** | RLHF, DPO, distillation, VAEs |
| 2.5.1 | Entropy | **Critical** | Perplexity, temperature, entropy regularization |
| 2.5.7 | Mutual Information | **Very High** | Contrastive learning, CLIP, probing, info bottleneck |
| 2.1.3 | Bayes' Theorem | **Very High** | In-context learning, posterior inference |
| 2.6.2 | Regularization as MAP | **Very High** | Weight decay, AdamW, LoRA priors |
| 2.3.2 | Likelihood / MLE | **High** | Cross-entropy loss derivation |
| 2.1.1–2 | Sum & Product Rules | **High** | Autoregressive factorization |
| 2.4 | Density Transformation | **High** | Normalizing flows, reparameterization trick |
| 2.2.2 | Expectations | **High** | SGD as Monte Carlo, gradient variance |
| 2.5.4 | Maximum Entropy | **Moderate** | Justifies Gaussian/softmax choices |
| 2.3 | Gaussian Distribution | **Moderate** | Initialization, diffusion noise, weight priors |
| 2.6.3 | Bayesian ML | **Moderate** | Uncertainty quantification, model selection |
| 2.3.3 | MLE Bias | **Moderate** | Overfitting intuition |
| 2.5.6 | Conditional Entropy | **Moderate** | Chain rule for information |
| 2.3.4 | Linear Regression | **Low** | Background for understanding loss functions |
| 2.5.2 | Physics Perspective | **Low** | Energy-based models context |

---

## Cross-Chapter Forward References

- **Ch 7 (Gradient Descent):** SGD is Monte Carlo estimation of $\mathbb{E}[\nabla \mathcal{L}]$ (Section 2.2.2). Adam's adaptive rates relate to per-parameter variance estimation.
- **Ch 9 (Regularization):** Weight decay = Gaussian prior (Section 2.6.2). Double descent relates to MLE bias in overparameterized regimes (Section 2.3.3).
- **Ch 12 (Transformers):** Autoregressive factorization = product rule (Section 2.1.2). Cross-entropy training = KL minimization (Section 2.5.5). Temperature sampling controls entropy (Section 2.5.1).
- **Ch 15–16 (Latent Variables):** ELBO derivation uses KL divergence (Section 2.5.5). EM algorithm connects to KL minimization.
- **Ch 18 (Normalizing Flows):** Built on the change-of-variables formula (Section 2.4).
- **Ch 19 (Autoencoders / VAEs):** Reparameterization trick uses density transformation (Section 2.4). KL regularization in VAE training (Section 2.5.5).
- **Ch 20 (Diffusion Models):** Gaussian noise schedule (Section 2.3). Forward process KL divergence. Score matching connects to gradient of log-density.
