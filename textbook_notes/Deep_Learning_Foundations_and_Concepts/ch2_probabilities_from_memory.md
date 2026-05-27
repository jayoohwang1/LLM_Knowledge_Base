# Ch 2: Probabilities — Bishop & Bishop (2024), pp. 25–58 *(from memory)*

> **Disclaimer**: written without re-reading the source. Pagination/sectioning may drift; the substance is what I'd expect this chapter to cover. Voice = senior applications researcher giving you the "what actually matters" cut.

---

## TL;DR — what to care about as an LLM researcher

| Concept | Used in modern LLM work? | Where it shows up |
|---|---|---|
| **Sum / product rules, chain rule of probability** | **Constantly** | Autoregressive factorization $p(x_1,...,x_T) = \prod_t p(x_t \mid x_{<t})$ — the entire next-token paradigm |
| **Cross-entropy** | **Literally the loss** | NLL training, distillation, reward modeling |
| **KL divergence** | **Massively** | PPO/GRPO trust regions, distillation, variational inference, mode-collapse diagnostics |
| **Entropy** | **High** | Sampling temperature, exploration bonuses, "uncertainty" proxies, calibration |
| **Bayes' theorem** | Conceptual mostly | Posterior interpretations of fine-tuning; in-context learning as Bayesian updating |
| **Gaussians (univariate/multivariate)** | Background tool | Diffusion noise, VAE priors, weight init, RLHF reward modeling (Bradley-Terry is logistic but Gaussian shows up adjacent), DPO derivations |
| **MLE bias / variance estimator bias** | Low-medium | Mentioned in scaling-law fits, not load-bearing for pretraining |
| **Change of variables / Jacobians** | Medium | Normalizing flows, diffusion's continuous-time formulation, sampler analysis |
| **Mutual information** | Low for LLMs directly, high in repr-learning literature | InfoNCE/contrastive (CLIP-era), interpretability ("info content of features"), MDL framing of compression-as-intelligence |
| **Bayesian regularization (MAP)** | Mostly conceptual today | Weight decay $\equiv$ Gaussian prior; not how anyone *thinks* during pretraining but useful framing |

**One sentence**: if you remember nothing else from this chapter, internalize **cross-entropy = NLL = log-loss**, **KL constraints prevent policy collapse**, and **the chain rule of probability is why autoregressive LMs work**.

---

## 2.1 The Rules of Probability

The foundational sum/product rules. Tiny in pages, enormous in implication.

### 2.1.1 — Motivating medical-test example
Bishop usually opens with the classic "test is 99% accurate, disease is 1% prevalent — what's $p(\text{disease} \mid \text{positive})$?" setup to motivate Bayes. The punchline: most positives are false positives because the **base rate dominates**. This is the same intuition pump people hammer when explaining why an LLM hallucinating a rare fact is *expected* — the prior over "this exact entity exists in training data" is tiny.

### 2.1.2 — Sum and product rules
- **Sum rule (marginalization)**: $p(X) = \sum_Y p(X, Y)$, continuous: $p(x) = \int p(x, y)\,dy$
- **Product rule**: $p(X, Y) = p(Y \mid X)\,p(X) = p(X \mid Y)\,p(Y)$
- **Chain rule (iterated product rule)**: $p(x_1, ..., x_T) = \prod_t p(x_t \mid x_1, ..., x_{t-1})$

> **Why this is the most important equation in modern AI**: every autoregressive model — GPT-x, Llama, Claude, etc. — is just *parameterizing the chain rule with a neural net*. There is no other ML insight that has been monetized harder. You learn $p_\theta(x_t \mid x_{<t})$, then sample sequentially, and you get language, code, math, image tokens (VQ-VAE / RQ), audio (AudioLM), video (Sora-style token streams) — all the same trick. Knowing the chain rule is *trivial*. Knowing that **the entire generative-AI economy is one inductive bias** (factorize the joint left-to-right, fit each conditional with a transformer) is the move.

### 2.1.3 — Bayes' theorem
$$p(Y \mid X) = \frac{p(X \mid Y)\,p(Y)}{p(X)}$$
- $p(Y)$ = **prior**, $p(X \mid Y)$ = **likelihood**, $p(Y \mid X)$ = **posterior**, $p(X) = \sum_Y p(X \mid Y)p(Y)$ = **evidence / marginal likelihood**.
- Posterior $\propto$ likelihood $\times$ prior — the engineering version.

**LLM-relevant uses**:
- **In-context learning as Bayesian inference**: there's a popular line (Xie et al., 2022; Mike Lewis / Anthropic-adjacent work) framing ICL as the model implicitly maintaining a posterior over latent tasks/concepts given the prompt, then predicting from $p(x_{t+1} \mid x_{<t}, \text{prompt})$. Whether or not it's *mechanistically* Bayes, the lens explains why more demos help (sharper posterior) and why irrelevant context hurts (likelihood pollution).
- **Fine-tuning intuition**: SFT/RLHF can be read as posterior updates from a pretrained prior. DPO's derivation literally drops out of "what's the posterior over policies given preference data with a KL-to-reference prior."
- **Calibration**: well-calibrated model probs are well-calibrated *posteriors* over the next token.

### 2.1.4–2.1.6 — Priors, posteriors, independence
- **Independence**: $p(X, Y) = p(X)p(Y) \iff p(X \mid Y) = p(X)$.
- **Conditional independence**: $X \perp Y \mid Z$ is the workhorse of graphical models. Less central to LLMs (which use a fully autoregressive, non-factorized joint over tokens) but matters for understanding why **bag-of-words** and **n-gram** models fail and why **attention** is needed (long-range conditional dependencies are not factorizable by position).

---

## 2.2 Probability Densities

Quick refresher. Continuous variables → PDFs $p(x) \ge 0$, $\int p(x)\,dx = 1$, CDFs $P(z) = \int_{-\infty}^z p(x)\,dx$.

### 2.2.2 — Expectations and covariances
- $\mathbb{E}[f] = \int f(x)\,p(x)\,dx$ (or $\sum$). LLM analog: **expected reward**, **expected logprob**, **expected loss** — every objective function in RLHF/PPO is an expectation, and **Monte Carlo estimation of expectations is half of what an RL trainer is doing.**
- $\text{var}[f] = \mathbb{E}[(f - \mathbb{E}[f])^2]$, $\text{cov}[X, Y] = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$. Multivariate: $\boldsymbol{\Sigma}$.
- **Conditional expectation**: $\mathbb{E}_y[f(y) \mid x] = \int f(y)\,p(y \mid x)\,dy$. This is *exactly* what a transformer's softmax output gives you when you take the expected value over the next-token distribution.

> **In practice**: when you see $\hat{g} = \frac{1}{N}\sum_i g(x_i),\ x_i \sim p$ in a PPO/GRPO codebase, that's a Monte Carlo expectation. Variance of this estimator is your enemy — drives the entire field of variance-reduced policy gradients (baselines, GAE, importance sampling corrections). Bishop doesn't go there but this is where the bridge is.

---

## 2.3 The Gaussian Distribution

The most-belabored distribution in any stats book. For LLMs *directly*, it's background; for everything *around* LLMs (diffusion, VAEs, init schemes, optimizer analysis) it's load-bearing.

### Univariate
$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

### Multivariate (D-dim)
$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}\,|\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

- $\boldsymbol{\Sigma}$ symmetric PSD; eigendecomposition gives axes of the contour ellipsoid, eigenvalues are squared semi-axis lengths.
- **Mahalanobis distance**: $\Delta^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ — Bishop usually emphasizes the geometric picture (rotated, scaled coordinate system).

### 2.3.2 — MLE for Gaussian
Given i.i.d. $\{x_n\}_{n=1}^N$:
- $\mu_{\text{ML}} = \frac{1}{N}\sum_n x_n$ (sample mean — unbiased)
- $\sigma^2_{\text{ML}} = \frac{1}{N}\sum_n (x_n - \mu_{\text{ML}})^2$ (**biased**; biases toward smaller variance because we used the estimated mean)
- Unbiased correction: divide by $N-1$ instead of $N$.

### 2.3.3 — Bias of MLE
Bishop makes a thing of MLE being biased in general (overfitting connection — the *intuition pump* for why frequentist MLE without regularization can lie to you, leading into the Bayesian section). For modern deep learning this is *cute but not load-bearing*: we have 10T tokens, the bias-from-mean correction is in the noise. Where it matters: **scaling laws fits with small N**, **eval suites with few items** (e.g., MMLU subcategories), and **reward model uncertainty** — there the difference between MLE and posterior matters.

### 2.3.4 — Gaussian linear regression
- Likelihood: $p(t \mid \mathbf{x}, \mathbf{w}, \beta) = \mathcal{N}(t \mid \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}), \beta^{-1})$
- MLE $\equiv$ least squares: $\nabla_{\mathbf{w}} \log L = 0 \Rightarrow \mathbf{w}_{\text{ML}} = (\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top \mathbf{t}$ (normal equations).
- **Punchline most worth remembering**: *"squared error loss falls out of assuming Gaussian noise."* Symmetric story: cross-entropy loss falls out of assuming categorical/Bernoulli output. **Loss functions = negative log-likelihoods under a particular output distribution.** This is one of the single most useful framings in deep learning.

> **Where Gaussians actually appear in modern LLM-stack work**:
> - **Diffusion models** (image gen, increasingly used for video, structured generation, even diffusion-LMs like SEDD/MD4): noise is exactly $\mathcal{N}(0, I)$, the entire forward/reverse process is Gaussian transitions.
> - **VAEs** (latent-diffusion's encoder, audio codecs): Gaussian variational posteriors $q(z \mid x) = \mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$.
> - **Weight init**: Kaiming/Xavier are Gaussians with carefully chosen variance. Initialization is where Gaussians silently make or break a training run.
> - **Optimizer analysis**: stochastic noise in SGD is often modeled as Gaussian for theory (SDE limit) — Mandt et al., Smith et al., the whole "SGD as approx Bayesian inference" line.
> - **RLHF reward modeling**: not strictly Gaussian (Bradley-Terry is logistic), but Gaussian reward priors show up in some posterior reward-modeling work.
> - **Mechanistic interp**: Gaussian assumptions about residual stream activations underlie SAE training stability tricks.

---

## 2.4 Transformation of Densities

Short section, surprisingly important downstream.

If $y = f(x)$ with $f$ monotonic invertible, and $x \sim p_x$, then:
$$p_y(y) = p_x(f^{-1}(y)) \left|\frac{df^{-1}}{dy}\right|$$
Multivariate, $\mathbf{y} = \mathbf{f}(\mathbf{x})$:
$$p_y(\mathbf{y}) = p_x(\mathbf{x})\,|\det \mathbf{J}_{\mathbf{f}^{-1}}(\mathbf{y})|$$
where $\mathbf{J}$ is the Jacobian.

> **Why care**:
> - **Normalizing flows** (RealNVP, Glow, NICE, neural spline flows): *the* application. You design $f$ to have a tractable Jacobian determinant so you can compute exact likelihoods. Less hot in pure-LLM land but very alive in scientific ML (lattice QCD, molecular gen).
> - **Continuous-time diffusion / score-based models**: the reverse SDE involves a change of variables; Jacobians show up in probability-flow ODE formulations and in flow-matching (Lipman et al., 2023). Flow-matching is currently *the* hot ground-truth-likelihood-free training paradigm — see Stable Diffusion 3, Flux, Movie Gen.
> - **Reparameterization trick**: $z = \mu + \sigma \epsilon$ is a change of variables. Underlies VAE training, low-variance gradient estimators, and DDPM sampling.
> - **Geometric DL**: equivariant nets reason about how densities transform under group actions.

If you're going to forget one paragraph of this chapter, **don't forget this one** — change-of-variables is the math under both flow models and diffusion samplers.

---

## 2.5 Information Theory

**The most important section of the chapter for LLM people.** Everything here directly powers training losses, evaluation metrics, and RL fine-tuning.

### 2.5.1 — Entropy
Discrete:
$$H[X] = -\sum_x p(x)\log p(x)$$
- Units: bits ($\log_2$) or nats ($\ln$). Default is nats in ML.
- Interpretation: expected information content (Shannon), expected codeword length under optimal code, "spread" of a distribution.
- Max entropy for discrete with $K$ outcomes: uniform, $H = \log K$.
- Min entropy: delta, $H = 0$.

**LLM uses**:
- **Sampling temperature** $\tau$: $p_\tau(x) \propto \exp(\text{logit}/\tau)$. Low $\tau$ → low entropy (greedy-ish, repetition risk). High $\tau$ → high entropy (creative, incoherent). Modern decoding (top-p, min-p, typical sampling) is essentially entropy management.
- **Entropy bonuses in RL** (PPO entropy coefficient): prevents policy collapse during RLHF. When entropy crashes to zero, you've memorized one response per prompt — usually a sign your reward model is being exploited or your KL is set wrong.
- **Diagnostic**: tracking per-token entropy during training is a free signal — sudden drops correlate with overfitting / mode collapse; sudden rises with training instability.

### 2.5.2 — Physics view (Boltzmann)
Bishop usually motivates entropy via statistical mechanics (multinomial occupations → Stirling → Shannon). Skim-worthy unless you're into the EBM / statistical-physics-of-DL angle (Mehta, Bahri, etc.). Cute connection: the partition function $Z = \sum_x e^{-E(x)/T}$ recurs everywhere — softmax is literally a Boltzmann distribution.

### 2.5.3 — Differential entropy
Continuous version: $h[X] = -\int p(x) \log p(x)\,dx$.
- **Warning**: not invariant under change of variables, can be negative. People often write $H$ and mean $h$ — be careful.
- **Max-entropy distribution under variance constraint = Gaussian.** Useful fact: justifies Gaussian noise in maximum-entropy settings (least-informative prior with known variance).

### 2.5.4 — Maximum entropy
Lagrangian principle: maximize entropy subject to moment constraints → exponential-family distribution. Big deal historically (Jaynes, MaxEnt modeling, early NLP log-linear models like MEMM/CRF). **Modern LLMs basically dropped this framework** in favor of overparameterized neural net likelihoods, but the principle survives: softmax + cross-entropy is a max-ent classifier conditional on features.

### 2.5.5 — KL Divergence — **the single most important inequality in the chapter for modern ML**

$$\text{KL}(p \,\|\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = \mathbb{E}_{p}[\log p - \log q]$$

- Non-negative ($\text{KL} \ge 0$), zero iff $p = q$. Asymmetric.
- $\text{KL}(p\,\|\,q) \ne \text{KL}(q\,\|\,p)$ — the asymmetry matters: forward KL is "mean-seeking" (covers $p$'s support), reverse KL is "mode-seeking" (concentrates on one mode).

**Decomposition (the key identity)**:
$$\text{KL}(p \,\|\, q) = H(p, q) - H(p) = \underbrace{-\mathbb{E}_p[\log q]}_{\text{cross-entropy}} - \underbrace{(-\mathbb{E}_p[\log p])}_{\text{entropy of }p}$$

If $p$ is fixed (e.g., the data distribution), minimizing $\text{KL}(p \,\|\, q)$ over $q$ is **the same as minimizing cross-entropy** — which is **the same as maximum likelihood** with empirical data.

> **This identity — KL ↔ cross-entropy ↔ MLE — is the entire reason you train LLMs with cross-entropy loss.** It is not arbitrary. You are minimizing KL from data to model.

**LLM-direct uses** (this is the chapter section that actually pays your salary):
1. **Pretraining loss**: cross-entropy = forward KL from empirical data dist to model. Perplexity $= e^{H(p, q)/T}$.
2. **RLHF KL constraint**: PPO objective is $\mathbb{E}[\,r(x, y) - \beta\,\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\,]$. The KL stops the policy drifting from the SFT base — without it you get reward hacking / gibberish that scores high. Tuning $\beta$ is half the art of post-training.
3. **DPO**: derived by analytically solving the KL-constrained reward-max problem; the closed form $\log \pi^*/\pi_{\text{ref}} \propto r$ is exactly the Boltzmann posterior under a KL constraint to the reference. The "magic" of DPO is just calculus on KL.
4. **GRPO / RLOO / REINFORCE-Leave-One-Out**: same KL trust region, different baseline scheme.
5. **Distillation**: $\text{KL}(\pi_{\text{teacher}} \,\|\, \pi_{\text{student}})$. Soft targets > hard targets because they carry the *full* distribution's info (dark knowledge — Hinton 2015). Direction matters: reverse KL (mode-seeking) for capability transfer; forward KL (mean-seeking) for diverse outputs. People are still arguing about which to use.
6. **Variational inference (VAEs, ELBO)**: ELBO $= \log p(x) - \text{KL}(q(z|x) \,\|\, p(z|x))$. The whole VAE training story.
7. **Drift detection / sycophancy / safety**: track KL between deployment policy and trusted reference; spikes flag distribution shift.
8. **Speculative decoding**: acceptance prob based on ratio $\pi_{\text{target}}/\pi_{\text{draft}}$ — formally a KL-related concept.

If you write one section of this chapter on a sticky note: **this one**.

### 2.5.6 — Conditional entropy
$$H[Y \mid X] = -\sum_{x, y} p(x, y) \log p(y \mid x) = H[X, Y] - H[X]$$
- Chain rule of entropy mirrors chain rule of probability.
- Per-token training loss of an LM is *exactly* an empirical estimate of $H[\text{token}_t \mid \text{context}_{<t}]$.

### 2.5.7 — Mutual Information
$$I[X; Y] = \text{KL}(p(x, y) \,\|\, p(x)p(y)) = H[X] - H[X \mid Y] = H[Y] - H[Y \mid X]$$
- Measures how much knowing $Y$ reduces uncertainty about $X$.
- Symmetric, non-negative, zero iff independent.

**Use cases**:
- **InfoNCE** (van den Oord 2018, CLIP, SimCLR, all contrastive SSL): lower bound on MI between two views. CLIP is *the* poster child — text-image MI maximization gave us the embedding space that powers all modern multimodal models.
- **Mechanistic interpretability**: MI between SAE features and behaviors; "information bottleneck" framings.
- **Estimation is hard**: MI between high-dim continuous vars is notoriously brittle to estimate (MINE, NWJ bounds). Trust the bounds, not the absolute numbers.

---

## 2.6 Bayesian Probabilities

The frequentist-vs-Bayesian sermon. Bishop is a card-carrying Bayesian and wants you to be one too; modern frontier-lab practice is mostly *unapologetically frequentist with regularization* but the Bayesian framing remains the cleanest mental model for several things.

### 2.6.1 — Bayesian view of model parameters
$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w})\,p(\mathbf{w})}{p(\mathcal{D})}$$
- Treat parameters as random variables with a prior, update to posterior given data.
- Predict by **marginalizing**: $p(y \mid x, \mathcal{D}) = \int p(y \mid x, \mathbf{w})\,p(\mathbf{w} \mid \mathcal{D})\,d\mathbf{w}$.

**Honest reality check for LLMs**: nobody is doing full posterior inference over a 70B parameter model. The integral is intractable; even approximate methods (Laplace, SWAG, ensembling) hit a wall at this scale. **What we actually do**: train one MAP estimate (= MLE + weight decay), maybe average a few checkpoints (SWA / soup), maybe ensemble across seeds for safety-critical eval. Bayesian deep learning as a research field has been mostly displaced from frontier work, but **it's the right frame for thinking about**:
- Uncertainty quantification / "does the model know it doesn't know?"
- Why scaling laws are smooth (implicit ensembling).
- Conformal prediction & calibration.

### 2.6.2 — Regularization $\equiv$ MAP
- Adding $\lambda \|\mathbf{w}\|^2$ to MLE $\equiv$ MAP with Gaussian prior $p(\mathbf{w}) = \mathcal{N}(0, \lambda^{-1} I)$.
- L1 $\equiv$ Laplace prior. Dropout $\equiv$ approximate Bayesian model averaging (Gal & Ghahramani 2016).

> **Practical takeaway**: every regularizer is secretly a prior. AdamW's decoupled weight decay is the modern workhorse. If you ever wondered "why $10^{-1}$ vs $10^{-4}$ weight decay matters so much" — that's a prior-strength choice in disguise.

### 2.6.3 — Bayesian ML
Bishop's pitch: Bayesian methods naturally handle small data, give calibrated uncertainty, do automatic model selection via marginal likelihood. **True, and also why they lost the deep learning fight** — frontier LLM training has *abundant* data, doesn't care much about formal uncertainty (we use eval suites), and does model selection via scaling laws + hyperparam sweeps, not marginal likelihoods. Bayes is *correct* but *expensive*. The compromise is "Bayesian-flavored" tricks: weight decay, MC dropout, ensembling, deep kernel learning, SAM (sharpness-aware minimization $\sim$ flat posterior region).

---

## What I'd skip on a re-read

- Detailed derivation of MLE bias for Gaussian variance — know that it's biased, move on.
- Boltzmann derivation of Shannon entropy — cute, not load-bearing.
- The medical screening worked example, *if* you already get Bayes — 1 page of intuition for something you've seen 50 times.

## What I'd dwell on

- **Chain rule of probability + autoregressive factorization** (re-read section, internalize).
- **Loss = NLL = expected log-density under data dist** (the Gaussian-regression-MLE-equivalence story generalizes everywhere).
- **All of 2.5 — entropy, KL, MI** — every part of this section ends up in a modern training recipe.
- **Change of variables** — short but powers flow / diffusion math.

---

## Bridges to later chapters / future reading

- Ch 3 (Standard Distributions): exponential family generalization, conjugate priors, categorical/Bernoulli/Beta-Binomial — this is where the LM-relevant categorical comes in formally.
- Ch 4–5 (Regression/Classification): cross-entropy + Gaussian-noise stories made fully concrete.
- Ch 17 (probably) on Diffusion / Ch on VAEs: change-of-variables and KL machinery deployed.
- For RLHF specifically: read Ouyang et al. 2022 (InstructGPT), Rafailov et al. 2023 (DPO), Schulman et al. 2017 (PPO) with the KL identity from 2.5.5 firmly in hand.
- For ICL-as-Bayes: Xie et al. 2022 "An Explanation of In-context Learning as Implicit Bayesian Inference."
