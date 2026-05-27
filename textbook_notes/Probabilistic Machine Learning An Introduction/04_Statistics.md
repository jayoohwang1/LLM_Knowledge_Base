# Chapter 4: Statistics

**Casual TL;DR from a frontier-lab perspective**: Chapter 4 is the "how do we actually fit models to data" chapter. The MLE/NLL/cross-entropy story (4.2) is the spine of every LLM pretraining run — if you only deeply internalize one section, make it that one. The regularization stuff (4.5) — weight decay, early stopping, more-data-as-regularizer — is the next tier; modern training recipes are basically a mix of these knobs. The Bayesian sections (4.6) matter much less for LLM pretraining day-to-day, but are huge in **uncertainty-aware downstream applications** (active learning, safety, scientific ML, RLHF-style reward modeling, conformal-adjacent reasoning, data attribution). The frequentist sampling/CI stuff (4.7) is mostly an "interview / evaluation" tool — you need it when reporting eval numbers and arguing about error bars on benchmarks. Bias-variance is conceptually load-bearing but the modern double-descent picture (not in this chapter) supersedes the classic story.

---

## 4.1 Introduction

- **Model fitting / training** = picking $\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$.
- Output is either a **point estimate** $\hat{\boldsymbol{\theta}}$ or a full posterior $p(\boldsymbol{\theta}|\mathcal{D})$ — the latter is "inference" in stats (and "prediction" in DL, annoyingly).
- Two camps: **Bayesian** vs **frequentist**.

---

## 4.2 Maximum Likelihood Estimation (MLE) — **THE central section for LLMs**

- $\hat{\boldsymbol{\theta}}_{\text{mle}} \triangleq \arg\max_{\boldsymbol{\theta}} p(\mathcal{D}|\boldsymbol{\theta})$.
- iid data $\Rightarrow$ likelihood factorizes: $p(\mathcal{D}|\boldsymbol{\theta}) = \prod_n p(\boldsymbol{y}_n | \boldsymbol{x}_n, \boldsymbol{\theta})$.
- Work in log space: $\ell(\boldsymbol{\theta}) = \sum_n \log p(\boldsymbol{y}_n | \boldsymbol{x}_n, \boldsymbol{\theta})$.
- **NLL** (negative log likelihood) is the standard cost function: $\text{NLL}(\boldsymbol{\theta}) = -\sum_n \log p(\boldsymbol{y}_n | \boldsymbol{x}_n, \boldsymbol{\theta})$.
	- **Why this matters for LLMs**: next-token prediction loss IS the conditional NLL of $p(y_t | y_{<t}, \boldsymbol{\theta})$. Pretraining = MLE on a giant autoregressive categorical model. Every line in this section maps directly to GPT-style training.
- **Unconditional vs joint vs conditional** MLE — language models are conditional on prefix.

### 4.2.2 Justification for MLE

- **MAP with uniform prior**: $\hat{\boldsymbol{\theta}}_{\text{map}} = \hat{\boldsymbol{\theta}}_{\text{mle}}$ when $p(\boldsymbol{\theta}) \propto 1$.
- **KL-minimization to empirical distribution**: define empirical $p_{\mathcal{D}}(\boldsymbol{y}) = \frac{1}{N} \sum_n \delta(\boldsymbol{y} - \boldsymbol{y}_n)$. Then
	- $D_{\text{KL}}(p_{\mathcal{D}} \| q_{\boldsymbol{\theta}}) = -\mathbb{H}(p_{\mathcal{D}}) - \frac{1}{N}\sum_n \log p(\boldsymbol{y}_n|\boldsymbol{\theta}) = \text{const} + \text{NLL}(\boldsymbol{\theta})$.
	- **Minimizing NLL $\equiv$ minimizing forward KL to empirical $\equiv$ cross-entropy minimization**.
	- **Crucial intuition for LLMs**: pretraining is matching model dist to data dist in *forward* KL → "**mode-covering**" behavior. This is why base LMs assign nonzero mass to weird/rare patterns and are hard to make "narrow". Reverse-KL (RL fine-tuning, RLHF) is mode-seeking → narrower behavior. This forward-vs-reverse KL asymmetry is *the* mental model for understanding post-training.
- **Use cases**:
	- Pretraining LLMs (next-token CE).
	- Speech recognition, machine translation, diffusion model training (denoising MLE).
	- Variational autoencoders use a variational lower bound on the MLE.

### 4.2.3 MLE for Bernoulli

- NLL = $-[N_1 \log\theta + N_0 \log(1-\theta)]$ → $\hat{\theta} = N_1/(N_0+N_1)$.
- $N_0, N_1$ are **sufficient statistics**.
- Toy but mirrors logistic-regression loss; binary classification heads use this loss.

### 4.2.4 MLE for categorical

- NLL = $-\sum_k N_k \log \theta_k$ with simplex constraint → $\hat{\theta}_k = N_k/N$.
- **This IS the softmax cross-entropy** at each token position for an LM. Lagrange-multiplier derivation is the same logic that gives you the softmax as the maxent distribution.

### 4.2.5 MLE for univariate Gaussian

- $\hat{\mu} = \bar{y}$, $\hat{\sigma}^2 = \frac{1}{N}\sum (y_n - \bar{y})^2 = s^2 - \bar{y}^2$.
- Note the **MLE variance is biased**; unbiased estimate uses $N-1$.
- Relevant for: regression losses (MSE = Gaussian NLL with fixed $\sigma$), batchnorm/layernorm statistics, score-matching/diffusion noise schedules.

### 4.2.6 MLE for multivariate Gaussian

- $\hat{\boldsymbol{\mu}} = \bar{\boldsymbol{y}}$, $\hat{\boldsymbol{\Sigma}} = \frac{1}{N} \mathbf{Y}^\top \mathbf{C}_N \mathbf{Y}$ (centered scatter / $N$).
- Derivation uses **trace trick**: $\ell = \frac{N}{2}\log|\boldsymbol{\Lambda}| - \frac{1}{2}\text{tr}[\mathbf{S}_{\bar{y}}\boldsymbol{\Lambda}]$ with $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ (precision matrix).
- **Warning**: $\boldsymbol{\Sigma}$ has $O(D^2)$ params; **singular if $N < D$**, ill-conditioned even when $N>D$. → shrinkage (see 4.5).
- **Relevance**: Gaussian fits show up in feature covariance analysis (e.g., **CKA**, neural representation similarity), Laplace approximations of NN posteriors, **Fisher-based** optimizers (K-FAC, natural gradient, EWC, second-order methods), and Gaussian process surrogates. The "covariance is singular when $D > N$" issue is the whole reason **shrinkage / regularization** matters for representation-learning analyses on small probe sets.

### 4.2.7 MLE for linear regression

- $\text{NLL}(\boldsymbol{w}) \propto \text{RSS}(\boldsymbol{w}) = \|\mathbf{X}\boldsymbol{w} - \boldsymbol{y}\|_2^2$.
- **OLS** closed form: $\hat{\boldsymbol{w}}_{\text{mle}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \boldsymbol{y}$.
- **Why this matters surprisingly often for LLM-era work**: linear probes on hidden states, **representation engineering** (steering vectors via linear regression), influence-function approximations, low-rank weight fitting (DoRA/LoRA initialization), reward-model fitting in some setups. Knowing OLS cold is non-negotiable.

---

## 4.3 Empirical Risk Minimization (ERM)

- Generalize MLE to **arbitrary loss** $\ell$: $\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N}\sum_n \ell(\boldsymbol{y}_n, \boldsymbol{\theta}; \boldsymbol{x}_n)$.
- Special case $\ell = -\log p$ → MLE.
- **0-1 loss** = misclassification rate, but **non-smooth, NP-hard** to optimize → use **surrogate losses**.
- **Surrogates** (binary, $\tilde{y} \in \{-1,+1\}$, margin $z = \tilde{y}\eta$):
	- **Log loss**: $\log(1+e^{-z})$ — used in logistic regression, used everywhere as **binary CE**.
	- **Hinge loss**: $\max(0, 1-z)$ — SVMs, some adversarial training objectives.
	- **Exp loss**: $e^{-z}$ — AdaBoost.
- **Modern relevance**: Almost everything in deep learning uses surrogate losses. Knowing that **CE is a convex upper bound on 0-1 loss** is the standard "why we train with log loss" answer. Margin-based losses show up in **contrastive learning** (triplet/margin losses), DPO/IPO/SimPO-style preference losses, and reward modeling.

---

## 4.4 Other Estimation Methods *

### 4.4.1 Method of Moments (MOM)

- Match theoretical moments $\mu_k = \mathbb{E}[Y^k]$ to empirical moments.
- Simple but statistically less efficient than MLE; can produce **inconsistent / invalid** estimates (e.g., uniform-dist example gives bounds tighter than data).
- **When it matters**: classic baseline for spectral / tensor methods for **latent variable models** (LDA, topic models, HMMs — see AHK12 reference). Less relevant for modern LLM stack but historically important.

### 4.4.2 Online (recursive) estimation

- Streaming updates: $\boldsymbol{\theta}_t = f(\boldsymbol{\theta}_{t-1}, \boldsymbol{y}_t)$.
- **Running mean**: $\hat{\mu}_t = \hat{\mu}_{t-1} + \frac{1}{t}(y_t - \hat{\mu}_{t-1})$.
- **EWMA** (exponentially weighted moving average): $\hat{\mu}_t = \beta \hat{\mu}_{t-1} + (1-\beta) y_t$.
	- Bias correction: $\tilde{\mu}_t = \hat{\mu}_t / (1 - \beta^t)$ (KB15).
- **HUGELY relevant for LLM training**:
	- **Adam / AdamW** use EWMA of gradient and gradient-squared (the bias correction term `1-beta^t` in Adam is literally Eq 4.88).
	- **EMA of model weights** is now standard in diffusion training, **stable RL** (SAC target networks), **self-distillation** (BYOL, DINO target networks).
	- **Loss curves** for monitoring training are usually EMA-smoothed.

```python
# Adam-style EWMA with bias correction (exactly Eq 4.82, 4.88)
m_t = beta1 * m_prev + (1 - beta1) * grad
v_t = beta2 * v_prev + (1 - beta2) * grad**2
m_hat = m_t / (1 - beta1**t)   # bias correction
v_hat = v_t / (1 - beta2**t)
theta -= lr * m_hat / (sqrt(v_hat) + eps)
```

---

## 4.5 Regularization — **second-most-important section for modern practice**

- Pure MLE/ERM **overfits**: 3 heads in 3 flips → $\hat{\theta}_{\text{mle}} = 1$, predicts no future tails.
- Add penalty: $\mathcal{L}(\boldsymbol{\theta}; \lambda) = [\frac{1}{N}\sum_n \ell] + \lambda C(\boldsymbol{\theta})$.
- $C(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta})$ → **MAP estimation** = MLE + log-prior penalty.

### 4.5.1 MAP for Bernoulli (add-one smoothing / Laplace)

- Beta$(a,b)$ prior → $\hat{\theta}_{\text{map}} = (N_1 + a - 1)/(N + a + b - 2)$.
- $a=b=2$ gives **add-one smoothing**: $(N_1+1)/(N+2)$ — fixes black-swan / zero-count problem.
- **Used in**: n-gram LMs (interpolation/back-off), naive Bayes text classifiers, RLHF reward calibration. Modern LLMs implicitly do this via softmax temperature / label smoothing.

### 4.5.2 MAP for multivariate Gaussian — **shrinkage estimator**

- Inverse-Wishart prior → $\hat{\boldsymbol{\Sigma}}_{\text{map}} = \lambda \boldsymbol{\Sigma}_0 + (1-\lambda)\hat{\boldsymbol{\Sigma}}_{\text{mle}}$.
- Common: shrink off-diagonals only (Ledoit-Wolf, sklearn `LedoitWolf`).
- Fixes singular MLE when $N < D$.
- **Where used**: financial portfolio covariance, **neural-net Fisher / Hessian estimation** (KFAC and friends always need damping = shrinkage), **probe analysis**, **Laplace approximations of NN posteriors**.

### 4.5.3 Weight decay / Ridge regression

- $\hat{\boldsymbol{w}}_{\text{map}} = \arg\min \text{NLL}(\boldsymbol{w}) + \lambda \|\boldsymbol{w}\|_2^2$.
- Equivalent to **Gaussian zero-mean prior** on weights.
- For linear regression = **ridge regression**.
- **The most-used regularizer in deep learning, period.**
	- AdamW decouples weight decay from gradient update — empirically critical for transformers.
	- Recent work (μP, scaling laws) studies weight-decay scaling carefully.
- Note: only penalize **weight magnitudes**, not biases / noise variances.

### 4.5.4 Validation-set tuning

- Split train/val (e.g. 80/20), fit on train for each $\lambda$, pick best on val.
- Define **regularized risk** $R_\lambda$ and **validation risk** $R^{\text{val}}_\lambda$.
- After picking $\lambda^*$, **refit on full data**.
- LLM analog: hyperparameter sweeps, choice of LR / WD / dropout etc.

### 4.5.5 Cross-validation

- $K$-fold CV: $R^{\text{cv}}_\lambda = \frac{1}{K} \sum_k R_0(\hat{\boldsymbol{\theta}}_\lambda(\mathcal{D}_{-k}), \mathcal{D}_k)$.
- $K=N$ = LOO-CV.
- **One-standard-error rule**: pick simplest model whose risk is within 1 SE of best.
- **Practical reality at frontier labs**: CV is rare for LLM training (single training run is too expensive). Used for **eval reliability** (bootstrapped benchmarks), **probing experiments**, **classical baselines in scientific ML**.

### 4.5.6 Early stopping

- Train loss decreases monotonically; val loss starts increasing → stop.
- "Limits how far parameters move from init" — implicit $\ell_2$ regularization around initialization.
- **Effective and free**. Used everywhere from small finetunes to dev-loss-based checkpoint selection. The IMDB plot (Fig 4.8) is the canonical picture.

### 4.5.7 Using more data — **THE LLM mantra**

- More data → less overfitting (assuming non-redundant).
- **Learning curves**: test error vs $N$ → asymptote at **Bayes error / noise floor**.
- For underspecified models (deg 1 on deg 2 data) error remains high → **underfitting**.
- **Most important section here for modern AI**: scaling-laws (Chinchilla, Hoffmann et al.) are a quantitative version of "use more data". Synthetic data, **data curation, deduplication, quality filtering** are direct descendants — the whole field of LLM data is "regularize by getting more / better data".

---

## 4.6 Bayesian Statistics *

Posterior: $p(\boldsymbol{\theta} | \mathcal{D}) = p(\boldsymbol{\theta}) p(\mathcal{D}|\boldsymbol{\theta}) / p(\mathcal{D})$.
Posterior predictive (BMA): $p(\boldsymbol{y}|\boldsymbol{x}, \mathcal{D}) = \int p(\boldsymbol{y}|\boldsymbol{x}, \boldsymbol{\theta}) p(\boldsymbol{\theta}|\mathcal{D}) d\boldsymbol{\theta}$.

**Honest take**: Almost no production LLM is "Bayesian" over weights. But Bayesian *thinking* shows up everywhere: **uncertainty estimation, active learning, exploration in RL, ensembling-as-approximate-Bayes, conformal prediction, calibration**. Worth understanding but you almost never write down a posterior over an LLM's parameters.

### 4.6.1 Conjugate priors

- $p(\boldsymbol{\theta}) \in \mathcal{F}$, posterior $\in \mathcal{F}$ — closed-form updates.
- All from **exponential family**. Pedagogically clean, practically useful for **simple hierarchical models, online learning, MAB/Thompson sampling**.

### 4.6.2 Beta-Binomial — canonical conjugate model

- Likelihood Bern$(\theta)$: $p(\mathcal{D}|\theta) = \theta^{N_1}(1-\theta)^{N_0}$.
- Beta prior: $p(\theta) = \text{Beta}(\theta | \breve{\alpha}, \breve{\beta})$.
- **Posterior**: Beta$(\breve{\alpha} + N_1, \breve{\beta} + N_0)$ — add data counts to "pseudo-counts".
- Posterior mean: $\bar{\theta} = \hat{\alpha}/\hat{N}$.
- Posterior var $\approx \hat{\theta}(1-\hat{\theta})/N$ — uncertainty shrinks at $1/\sqrt{N}$.
- **Posterior predictive** = $\hat{\alpha}/(\hat{\alpha}+\hat{\beta})$; with Beta(1,1) prior → $(N_1+1)/(N+2)$ = **Laplace's rule of succession**.
- Marginal likelihood: $p(\mathcal{D}) = \binom{N}{N_1} B(a+N_1, b+N_0)/B(a,b)$.
- **Mixtures of conjugates** also conjugate — useful for multi-modal priors ("fair OR loaded coin").
- **Use cases in modern ML**: **Thompson sampling for multi-armed bandits** (ad routing, A/B testing, LLM prompt selection, RLHF preference exploration); **Bayesian calibration of binary classifiers**; reward model uncertainty in RLHF.

### 4.6.3 Dirichlet-Multinomial

- $K$-ary version of Beta-Bernoulli.
- Dirichlet$(\boldsymbol{\theta} | \breve{\boldsymbol{\alpha}})$ over simplex; posterior $= $ Dir$(\breve{\boldsymbol{\alpha}} + \mathbf{N})$.
- $\alpha < 1$ → spiky/sparse, $\alpha \gg 1$ → concentrated; uniform $\alpha = 1$.
- Marginal likelihood = ratio of beta functions.
- **Used in**: LDA topic models (Dirichlet over topic distributions), naive Bayes smoothing, **Bayesian neural net softmax outputs**, exploration in discrete-action RL, MoE routing analysis. Dirichlet-sparsity prior is the conceptual ancestor of "sparse mixture of experts" routing.

### 4.6.4 Gaussian-Gaussian model

- Known $\sigma^2$, Gaussian prior on $\mu$ → Gaussian posterior.
- Posterior precision adds: $\hat{\lambda} = \breve{\lambda} + N\kappa$.
- Posterior mean = **precision-weighted average** of prior mean and data mean.
- $N=1$: $\hat{m} = (1-w)y + w \breve{m}$, $w = \breve{\lambda}/\hat{\lambda}$ — **shrinkage** toward prior.
- SEM $= s/\sqrt{N}$, 95% credible interval $\approx \bar{y} \pm 2 s/\sqrt{N}$.
- Multivariate version: same with $\boldsymbol{\Sigma}^{-1}$, prior covariance.
- **Where used**: Kalman filtering (recursive Gaussian update!), state estimation, **Bayesian linear regression** (also gives ridge regression posterior), **Laplace approximations**, **last-layer Bayesian NNs** (cheap uncertainty for LLM finetunes by treating last layer as Bayesian linear regression).

### 4.6.5 Beyond conjugate priors

- **Noninformative / flat / Jeffreys priors** — "minimally informative". No unique definition.
- **Hierarchical priors**: $\boldsymbol{\xi} \to \boldsymbol{\theta} \to \mathcal{D}$. Hyperparameters get their own prior. Useful when you have **groups / tasks / subpopulations** — directly maps to **multi-task / meta-learning** and **hierarchical Bayes for personalization**.
- **Empirical Bayes (ML-II)**: $\hat{\boldsymbol{\xi}} = \arg\max p(\mathcal{D}|\boldsymbol{\xi}) = \arg\max \int p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta}|\boldsymbol{\xi})d\boldsymbol{\theta}$.
	- **Type-II maximum likelihood**.
	- Hierarchy of "Bayesian-ness": MLE → MAP → ML-II → MAP-II → full Bayes.
	- **Used in**: GP hyperparameter tuning (length scales etc.), **automatic relevance determination (ARD)**, **PAC-Bayes bounds** (some have ML-II flavor).

### 4.6.6 Credible intervals

- $C_\alpha(\mathcal{D}) = (\ell, u)$ with $P(\ell \leq \theta \leq u | \mathcal{D}) = 1-\alpha$.
- **Central interval**: $(1-\alpha)/2$ mass in each tail.
- **HPD / HDI** (highest posterior density): all points inside have higher density than outside. Narrower than central interval; can be disconnected for multimodal posteriors.
- **Why care**: Bayesian eval reporting, **uncertainty-aware decision-making**, scientific ML. Modern conformal prediction is a frequentist cousin that gives **distribution-free** coverage — more practical for LLMs.

### 4.6.7 Bayesian machine learning

- Conditional models $p(\boldsymbol{y}|\boldsymbol{x}, \boldsymbol{\theta})$ + posterior on $\boldsymbol{\theta}$.
- **Plugin approximation** = $\delta(\boldsymbol{\theta} - \hat{\boldsymbol{\theta}})$ = standard DL.
	- Suffers from **overconfidence** — see the famous "ReLU networks are overconfident far from data" story.
- Bayesian logistic regression toy example (iris): plugin gives sharp sigmoid, Bayes gives band of curves with 95% CI on **decision boundary**.
- Monte Carlo posterior predictive: $p(y|x, \mathcal{D}) \approx \frac{1}{S}\sum_s p(y|x, \boldsymbol{\theta}^s)$.
- **Where it matters in 2025-ish**:
	- **Deep ensembles** = poor man's Bayesian deep learning, used everywhere for uncertainty (autonomous driving, medical, weather, AI safety eval).
	- **MC dropout, SWAG, Laplace approximations** for cheap NN uncertainty.
	- **Last-layer Bayes** for LLM finetuning uncertainty (recent NeurIPS line of work).
	- **Active learning / curiosity / exploration** in RLHF: uncertainty over reward model used to guide labeling.
	- **Bayesian optimization** for hyperparameter search.

### 4.6.8 Computational issues — approximate inference

- Exact posterior intractable except for conjugate cases.
- **Grid approximation**: enumerate $\theta$ grid, compute unnormalized posterior; doesn't scale past 2-3 dims.
- **Laplace / quadratic approximation**:
	- $p(\boldsymbol{\theta}|\mathcal{D}) \approx \mathcal{N}(\hat{\boldsymbol{\theta}}, \mathbf{H}^{-1})$ where $\mathbf{H}$ is Hessian of NLL at MAP.
	- Cheap (just optimize + compute Hessian / diagonal of Hessian).
	- Modern revival: **Laplace approximation for neural networks** (Daxberger et al., Immer et al.) — used to get post-hoc uncertainty estimates for pretrained models.
- **Variational inference (VI)**:
	- $q^* = \arg\min_{q \in \mathcal{Q}} D(q, p)$, typically reverse KL → **ELBO** maximization.
	- Foundation of **VAEs**, normalizing flows, amortized inference.
	- Reverse-KL ⇒ **mode-seeking** (recall the LLM intuition).
- **MCMC**:
	- Sampling from posterior without normalization constant.
	- **HMC** uses $\nabla \log p$ to accelerate.
	- Slow but unbiased asymptotically; used in **probabilistic programming languages** (Stan, PyMC, NumPyro), Bayesian neural net experiments at small scale.
- **Frontier-lab honest take**: For LLMs we essentially never do real Bayesian inference. **Laplace and deep ensembles** are the only methods that survive at scale. VI matters via VAEs / diffusion training. HMC is a "research curiosity" for foundation models.

---

## 4.7 Frequentist Statistics *

- View **data** as random, **parameters** as fixed; build **sampling distribution** of an estimator $\hat{\Theta}(\mathcal{D})$.
- Box quote: "...much less difficulty with an approach via likelihood and Bayes' theorem."
- Still essential for **evaluation reporting**.

### 4.7.1 Sampling distributions

- Resample $\mathcal{D}^{(s)} \sim \boldsymbol{\theta}^*$, apply estimator, get a distribution over estimates.

### 4.7.2 Gaussian approximation of MLE — asymptotic normality

- $\text{SamplingDist}(\hat{\Theta}^{\text{mle}}, \boldsymbol{\theta}^*) \to \mathcal{N}(\cdot | \boldsymbol{\theta}^*, (N\mathbf{F}(\boldsymbol{\theta}^*))^{-1})$.
- **Fisher information matrix** $\mathbf{F}(\boldsymbol{\theta}) = \mathbb{E}[\nabla \log p \cdot \nabla \log p^\top]$.
- Equivalently: $\mathbf{F} = -\mathbb{E}[\nabla^2 \log p]$ = expected Hessian of NLL.
- **Fisher information is everywhere in modern ML**:
	- **Natural gradient**, K-FAC, Shampoo — use Fisher (or empirical Fisher) as preconditioner.
	- **Elastic Weight Consolidation (EWC)** in continual learning — penalize moving away from previous tasks proportional to Fisher.
	- **Influence functions, data attribution** rely on Fisher / Hessian inverse.
	- **PAC-Bayes / generalization bounds** use Fisher-like quantities.
	- **Information geometry of LLMs** — flat vs sharp minima.

### 4.7.3 Bootstrap

- Resample $\mathcal{D}^{(s)}$ with replacement, refit $\hat{\boldsymbol{\theta}}^s = \hat{\Theta}(\mathcal{D}^{(s)})$ for $s = 1, ..., B$ ($B \approx 10000$).
- $\approx 0.632 N$ unique points per resample.
- **"Poor man's posterior"** — bootstrap distribution often resembles posterior under flat prior.
- **Critical for modern ML eval**: every "error bar" on a benchmark (MMLU, HELM, lm-eval-harness) is essentially a bootstrap estimate. Big releases now report bootstrapped CIs on eval scores.

### 4.7.4 Confidence intervals

- $\Pr(\theta^* \in I(\tilde{\mathcal{D}}) | \tilde{\mathcal{D}} \sim \theta^*) \geq 1-\alpha$.
- Common form: $\hat{\theta} \pm z_{\alpha/2} \hat{\text{se}}$, $z_{0.025} = 1.96$.
- Or bootstrap percentile CI.

### 4.7.5 CIs are not credible

- Coverage is over **repeated sampling of data**, not "prob true param in this interval".
- Several pathological examples (the (39,39),(39,40),(40,40) trick; Wald interval on Bernoulli with $N=1, y=1$ → CI = (0,0)).
- Bayesian credible intervals don't have this pathology.
- **Practical**: when reporting a single number ± error, you're almost always implicitly making a Bayesian claim ("we're 95% sure the true value is here"), which is what credible intervals actually say.

### 4.7.6 Bias-variance tradeoff

- $\text{bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}(\mathcal{D})] - \theta^*$.
- MLE for Gaussian mean: unbiased. MLE for variance: biased ($\mathbb{E}[\hat{\sigma}^2_{\text{mle}}] = \frac{N-1}{N}\sigma^2$); $N-1$ correction gives unbiased.
- **Variance of estimator**: $\mathbb{V}[\hat{\theta}] = \mathbb{E}[\hat{\theta}^2] - \mathbb{E}[\hat{\theta}]^2$.
- **Cramer-Rao**: $\mathbb{V}[\hat{\theta}] \geq 1/(N\mathbf{F}(\theta^*))$ — MLE asymptotically saturates this → **asymptotically optimal**.
- **MSE decomposition**: $\text{MSE} = \text{variance} + \text{bias}^2$.
- MAP estimators trade bias for variance — often beats MLE in MSE if prior is well-chosen.
- **Bias-variance for classification**: combines **multiplicatively**, not additively. Wrong-side-of-boundary bias is good or bad depending on sign → use cross-validated expected loss, not bias/variance directly.
- **Modern caveat**: In the over-parameterized deep-learning regime, the classical U-curve is wrong — **double descent** (test error decreases past the interpolation threshold). This chapter doesn't cover that, but the classical story is still load-bearing for small-scale / linear-probe work and for understanding why **larger models can be more data-efficient**.

---

## What actually matters for modern LLM research (tier list)

**S-tier (use every day):**
- **NLL / cross-entropy / MLE as KL-to-empirical** (4.2.2): the entire pretraining objective. Forward-vs-reverse KL intuition shapes how we think about SFT vs RL fine-tuning.
- **Weight decay / $\ell_2$ regularization** (4.5.3): in every optimizer.
- **EWMA / Adam-style updates with bias correction** (4.4.2): every optimizer step.
- **Early stopping & validation/eval-loss monitoring** (4.5.6): standard training hygiene.
- **More data = better** (4.5.7): scaling laws.

**A-tier (regularly relevant):**
- **OLS / ridge regression** (4.2.7, 4.5.3): linear probes, representation analysis, last-layer Bayes.
- **Bootstrap / CIs for eval** (4.7.3-4.7.4): benchmark reporting.
- **Fisher information / Hessian** (4.7.2): natural gradient, EWC, influence functions, data attribution.
- **Laplace approximation** (4.6.8.2): post-hoc NN uncertainty.
- **MAP / smoothing / shrinkage** (4.5.1-4.5.2): label smoothing, regularized covariance, reward-model calibration.

**B-tier (matters for specific subareas):**
- **Beta-Bernoulli / Dirichlet-multinomial** (4.6.2-4.6.3): Thompson sampling, bandits, LDA, calibration, RLHF exploration.
- **Hierarchical / empirical Bayes** (4.6.5): multi-task, personalization, ARD.
- **Credible intervals / HDI** (4.6.6): probabilistic eval, scientific ML.
- **VI / MCMC** (4.6.8.3-4.6.8.4): VAEs, diffusion, probabilistic programming.
- **Bias-variance tradeoff** (4.7.6): classical conceptual scaffolding; superseded by double-descent / scaling laws for big models.

**C-tier (good to know, rarely used directly):**
- **Method of moments** (4.4.1): historical, spectral LV methods.
- **CIs are not credible** caveats (4.7.5): philosophical clarity but doesn't change daily practice much.
- **Surrogate loss zoo (hinge, exp)** (4.3.2): mostly use log/CE in deep learning, hinge for some contrastive setups.
