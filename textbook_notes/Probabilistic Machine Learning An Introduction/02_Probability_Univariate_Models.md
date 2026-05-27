# Chapter 2: Probability — Univariate Models

Foundations chapter. Mostly review for anyone in ML, but a few subtle things actually matter a lot for modern LLM work (softmax/temperature, log-sum-exp, cross-entropy via logits, KL/entropy intuitions that show up in Ch 6, change of variables for flows/diffusion, CLT for noise modeling). I'll flag the load-bearing stuff as we go.

---

## 2.1 Introduction — What is probability?

- **Two interpretations**: **frequentist** (long-run frequency of repeatable events) vs **Bayesian** (degree of belief / uncertainty representation).
	- Murphy adopts **Bayesian** throughout. Rules of probability are interpretation-agnostic though.
	- **Why it matters for LLMs**: when you talk about a model's "calibration" or "epistemic uncertainty over generations", you're implicitly Bayesian. Pretraining loss is also a Bayesian quantity (negative log marginal of data under model).
- **Types of uncertainty** (this distinction is everywhere in modern ML):
	- **Epistemic / model uncertainty**: due to ignorance of params or model. **Reducible** with more data.
	- **Aleatoric / data uncertainty**: intrinsic variability. **Irreducible**.
	- **Use cases**: active learning (only useful to query high-epistemic points), safe RL, hallucination detection, ensemble-based UQ in LLMs (e.g., semantic entropy for hallucination), Bayesian deep learning.
- **Probability as extension of logic** (Jaynes view). Useful to ground discussions of model "reasoning".

### 2.1.3 Rules — bread and butter

- $\Pr(A \wedge B) = \Pr(A,B)$; independence: $\Pr(A,B)=\Pr(A)\Pr(B)$.
- **Union**: $\Pr(A \vee B) = \Pr(A) + \Pr(B) - \Pr(A \wedge B)$.
- **Conditional**: $\Pr(B|A) \triangleq \Pr(A,B)/\Pr(A)$.
- **Conditional independence (CI)**: $A \perp B | C \iff \Pr(A,B|C)=\Pr(A|C)\Pr(B|C)$. Backbone of graphical models, also implicitly assumed in transformer attention factorizations (causal mask = CI of future tokens given past).

---

## 2.2 Random variables

- **Discrete rv**: pmf $p(x) \triangleq \Pr(X=x)$, sums to 1.
- **Continuous rv**: cdf $P(x) \triangleq \Pr(X \le x)$, pdf $p(x) \triangleq dP/dx$.
	- Density can exceed 1 — common gotcha.
- **Quantiles**: inverse cdf $P^{-1}(q)$. Used in **quantile regression**, conformal prediction, sampling tricks.
- **Joint / marginal / conditional**: $p(x)=\sum_y p(x,y)$ (sum rule); $p(x,y)=p(x)p(y|x)$ (product rule).
- **Chain rule of probability**: $p(x_{1:D})=\prod_d p(x_d | x_{1:d-1})$.
	- **THIS IS LITERALLY THE LLM OBJECTIVE.** Autoregressive language modeling factorizes $p(\text{tokens})$ exactly this way. Every next-token prediction loss term comes from this rule. If you internalize one thing in this chapter, this is it.
	- Diffusion models also factorize as a (reverse-time) chain.
- **Independence vs CI**: unconditional independence is rare; CI captures the "mediated through Z" structure that defines graphical models, and informally justifies why deep nets work (latent representations render inputs conditionally near-independent given features).

### Moments (2.2.5)

- **Mean**: $\mathbb{E}[X] = \int x\, p(x)\,dx$. Linear: $\mathbb{E}[aX+b]=a\mathbb{E}[X]+b$.
	- $\mathbb{E}[\sum X_i] = \sum \mathbb{E}[X_i]$ — always (even dependent).
	- $\mathbb{E}[\prod X_i] = \prod \mathbb{E}[X_i]$ — only if independent.
- **Variance**: $\mathbb{V}[X] = \mathbb{E}[X^2] - \mu^2$. $\mathbb{V}[aX+b] = a^2 \mathbb{V}[X]$. Sum of independents: variances add.
- **Mode**: $\arg\max p(x)$. Multimodal distributions: mode is a poor summary (huge issue in generative modeling — MLE of a mixture-distribution data with unimodal model => mean of modes => blurry samples).
- **Conditional moments — really useful identities**:
	- **Law of total expectation**: $\mathbb{E}[X]=\mathbb{E}_Y[\mathbb{E}[X|Y]]$.
	- **Law of total variance**: $\mathbb{V}[X]=\mathbb{E}_Y[\mathbb{V}[X|Y]] + \mathbb{V}_Y[\mathbb{E}[X|Y]]$.
		- Within-cluster + between-cluster variance decomposition. Used in mixture model analysis, hierarchical Bayes, gradient variance decomposition (e.g., REINFORCE control variates, Rao-Blackwellization).

### 2.2.6 Limitations of summary statistics

- **Anscombe's quartet / Datasaurus dozen**: same mean/variance/correlation, totally different distributions. Always visualize. Modern echo: don't trust a single eval metric for LLMs (e.g., MMLU score vs actual capability). Use distributions of outcomes, error analysis.

---

## 2.3 Bayes' rule

Core identity. Posterior $\propto$ prior $\times$ likelihood:

$$p(H=h|Y=y) = \frac{p(H=h)\,p(Y=y|H=h)}{p(Y=y)}$$

- **Prior** $p(H)$: prior belief, **likelihood** $p(Y|H)$ as function of $h$ (not a distribution!), **marginal likelihood / evidence** $p(Y)$ (normalizer, also = model selection criterion).
- Examples in chapter: COVID test (counter-intuitive role of base rate), Monty Hall.
- **Why this matters for LLMs**:
	- **Posterior inference / Bayesian deep learning** is hard at scale but conceptually central to UQ.
	- **In-context learning is implicit Bayesian inference** (Xie et al., 2022 view): model treats prompt as evidence, performs implicit posterior over latent task/concept.
	- **RLHF/DPO** is reweighting a base policy with a reward — can be derived as posterior under reward-as-likelihood (KL-constrained RL ≡ Bayesian posterior).
	- **Diagnostic-test-style base-rate reasoning** failures are exactly the kind of reasoning LLMs are bad at; relevant for benchmarks like LSAT/MedQA.

### 2.3.3 Inverse problems

- Forward model $p(y|h)$ easy, inverse $p(h|y)$ usually ill-posed and needs prior.
- Applications: 3D from 2D images, NLU (intent from text), inverse rendering. Diffusion-based posterior sampling (DPS, etc.) is currently hot in inverse problems literature.

---

## 2.4 Bernoulli & binomial

- **Bernoulli**: $\text{Ber}(y|\theta) = \theta^y(1-\theta)^{1-y}$. Single binary outcome.
- **Binomial**: $\text{Bin}(s|N,\theta)=\binom{N}{s}\theta^s(1-\theta)^{N-s}$. Count of successes in $N$ trials.
- **Sigmoid / logistic function** $\sigma(a)=1/(1+e^{-a})$ — maps logit $a$ to probability.
	- Properties (memorize): $\frac{d}{dx}\sigma = \sigma(1-\sigma)$; $1-\sigma(x) = \sigma(-x)$; inverse is $\text{logit}(p)=\log\frac{p}{1-p}$; softplus $\sigma_+(x)=\log(1+e^x)$, derivative = $\sigma(x)$.
	- **Where you actually see this**: gating in LSTMs/GRUs (historical), gates in MoE routers, SwiGLU/GeGLU activations (softplus relatives), binary classification heads, reward models in RLHF (often binary preferences), DPO loss is literally a log-sigmoid of reward differences.
- **Binary logistic regression**: $p(y|x)=\text{Ber}(y|\sigma(w^\top x+b))$. Foundation for everything.

---

## 2.5 Categorical & multinomial — **EXTREMELY IMPORTANT FOR LLMs**

This section is short in the book but punches way above its weight for our work.

- **Categorical**: $\text{Cat}(y|\theta)=\prod_c \theta_c^{\mathbb{I}(y=c)}$, $C-1$ free params (sum to 1 constraint).
	- One-hot encoding $\mathbf{e}_c$ for class $c$ — used everywhere for token embeddings (input is one-hot index into embedding matrix).
- **Multinomial**: counts over $N$ categorical trials. Used in topic models (LDA), bag-of-words, n-gram LMs.
- **Softmax** $\text{softmax}(a)_c = e^{a_c}/\sum_{c'} e^{a_{c'}}$. Inputs $a$ = **logits** (generalize log-odds).
	- **This is the output layer of every LLM.** Logits $a = Wh + b$ over vocab of size $\sim$100K-300K, softmax over them gives next-token distribution.
	- **Temperature**: $\text{softmax}(a/T)$. $T\to 0$ ⇒ argmax (greedy); $T\to\infty$ ⇒ uniform.
		- **Direct use in sampling**: temperature, top-k, top-p (nucleus), min-p — all manipulate this distribution.
		- Used in **distillation** (soft targets at $T>1$ from teacher).
		- Used in **annealing** for combinatorial optimization, Gumbel-softmax for differentiable sampling.
		- Name comes from **Boltzmann distribution** in stat physics — relevant for **energy-based models** and **score-based diffusion**.
- **Multiclass logistic regression**: $p(y|x)=\text{Cat}(y|\text{softmax}(Wx+b))$.
	- Note **over-parameterization** (two weight vectors for binary case differ by constant): you can fix one class's logit to 0. In LLMs we ignore this but it shows up in interpretability work (logit lens, etc.).

### 2.5.4 Log-sum-exp trick — **CRITICAL IN PRACTICE**

$$\log \sum_c \exp(a_c) = m + \log\sum_c \exp(a_c - m),\quad m = \max_c a_c$$

- Without it, $\exp(1000) = \inf$ in fp32/fp64. Pretty much every deep learning framework uses this in their softmax/cross-entropy/loss.
- **Cross-entropy from logits directly** is the numerically stable, framework-standard form (`F.cross_entropy(logits, target)` in PyTorch). The chapter shows binary case:

```python
# binary CE from logit a, label y in {0,1}
# log p_1 = -lse([0, -a]),  log p_0 = -lse([0, a])
# loss = -y * log_p1 - (1-y) * log_p0
```

- **Where it shows up in modern code**: every transformer's loss head, MoE gating (top-k softmax), attention softmax (with scaling by $1/\sqrt{d}$), **online softmax** in FlashAttention is literally an LSE running update.
- If you're going to remember one numerical trick, make it this one.

---

## 2.6 Univariate Gaussian — workhorse

- pdf: $\mathcal{N}(y|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(y-\mu)^2)$.
- cdf is the **error function** $\Phi$; inverse cdf = **probit**.
- **Precision** $\lambda = 1/\sigma^2$ — natural parameter in exponential family form; relevant in Bayesian linear regression and Gaussian processes.

### 2.6.3 Regression with Gaussian noise

- **Homoscedastic**: $p(y|x)=\mathcal{N}(y|w^\top x + b, \sigma^2)$ — standard linear regression. MLE = least squares.
- **Heteroscedastic**: variance depends on input, $\sigma(x)^2 = \text{softplus}(w_\sigma^\top x)^2$.
	- **Big deal for modern uncertainty quantification**: predicting variance lets a regression model "know what it doesn't know". Used in continuous control, RL value functions (distributional RL), Bayesian deep learning, calibration of LLM probabilities.
- **Uncertainty over $y$ (predictive)** vs **uncertainty over $\theta$ (epistemic)** distinction reappears.

### 2.6.4 Why Gaussian is everywhere

- Two interpretable parameters (mean, variance).
- **Central Limit Theorem** ⇒ default for noise.
- **Maximum entropy** under fixed mean+variance (least committal).
- Simple math, conjugate updates, closed-form many things.
- **In LLM context**: weight init, activation analysis (often empirically Gaussian-ish post-LayerNorm), diffusion model noise ($\mathcal{N}(0,I)$), VAE priors/posteriors, gradient noise modeling.

### 2.6.5 Dirac delta as limit

$\lim_{\sigma\to 0} \mathcal{N}(y|\mu,\sigma^2) = \delta(y-\mu)$. Sifting property $\int f(y)\delta(x-y)dy = f(x)$.
- Used in empirical distributions, point estimates, MAP. Key for understanding **deterministic vs stochastic** policies in RL.

### 2.6.6 Truncated Gaussian — used in positive-support models, RL action bounding.

---

## 2.7 Other common univariate distributions

Brief tour. Most matter less day-to-day but a few are useful.

- **Student-t** $\mathcal{T}(\mu, \sigma^2, \nu)$: heavy-tailed, **robust to outliers**. As $\nu \to \infty$ ⇒ Gaussian. Common: $\nu=4$.
	- **Use cases**: robust regression, financial returns, attention with heavy-tailed gradients, **t-SNE** uses Student-t in low-dim (heavy tail = avoid crowding).
- **Cauchy** ($\nu=1$): mean undefined, infinite variance. Pathological cases / robust Bayesian priors.
- **Half-Cauchy**: positive-real heavy-tailed prior — used in **horseshoe priors** (sparsity in Bayesian regression).
- **Laplace** $\propto \exp(-|y-\mu|/b)$: heavy-tailed, sharp peak. Equivalent to **L1 regularization** (sparse linear regression / LASSO comes from MAP under Laplace prior). Also used in robust regression. Relevant for: sparse attention, pruning, weight quantization analysis.
- **Beta** on $[0,1]$: $\propto x^{a-1}(1-x)^{b-1}$. Conjugate prior to Bernoulli/Binomial.
	- Use cases: Thompson sampling (bandits → RLHF exploration), beta-policies in continuous control, modeling probabilities/rates.
- **Gamma** on $\mathbb{R}_+$: flexible positive-real. Special cases:
	- **Exponential**: inter-arrival of Poisson process.
	- **Chi-squared**: sum of squared standard normals — appears in test statistics, gradient norms.
	- **Inverse-Gamma**: conjugate prior for Gaussian variance.
- **Empirical distribution** $\hat p_N(x) = \frac{1}{N}\sum_n \delta_{x^{(n)}}(x)$: foundation for **Monte Carlo**, bootstrapping, replay buffers in RL, training data as a distribution.

---

## 2.8 Transformations of random variables — **important for flows and diffusion**

### Discrete case

$p_Y(y) = \sum_{x: f(x)=y} p_X(x)$. Trivial.

### Continuous — change of variables

- **Scalar bijection**: $p_y(y) = p_x(g(y)) |dg/dy|$ where $g = f^{-1}$.
- **Multivariate bijection**: $p_y(\mathbf{y}) = p_x(\mathbf{g}(\mathbf{y})) |\det \mathbf{J}_g(\mathbf{y})|$ — **the Jacobian determinant**.
	- **THIS IS THE FOUNDATION OF NORMALIZING FLOWS**. Coupling layers (RealNVP, Glow), autoregressive flows (MAF/IAF), continuous flows (FFJORD), and the change-of-variables form of the log-likelihood for flow-based generative models all reduce to this formula. Diffusion models in continuous time (score-based SDEs) also rely on it via Fokker-Planck / instantaneous change of variables.
	- Even **rectified flow** and **flow matching** (Lipman et al., 2022, very current) are grounded here.
- Polar coordinate example $|\det J| = r$ shows up everywhere (Box-Muller sampling, polar integrals for Gaussian normalizer in Exercise 2.12).

### 2.8.4 Moments of linear transformation

- $\mathbb{E}[\mathbf{A}\mathbf{x}+\mathbf{b}] = \mathbf{A}\boldsymbol{\mu}+\mathbf{b}$; $\text{Cov}[\mathbf{A}\mathbf{x}+\mathbf{b}] = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top$.
- $\mathbb{V}[X+Y]=\mathbb{V}[X]+\mathbb{V}[Y]+2\text{Cov}[X,Y]$ — sum's variance with correlation.
- Crucial for Gaussian propagation through linear layers (e.g., signal propagation analyses, GP kernels, NTK theory, batchnorm/layernorm analysis).

### 2.8.5 Convolution theorem

- Sum of independent rvs: $p_{X+Y} = p_X \circledast p_Y$.
- Discrete: $p(y=j) = \sum_k p(x_1=k)p(x_2=j-k)$.
- **Sum of independent Gaussians is Gaussian**: $\mathcal{N}(\mu_1,\sigma_1^2) \circledast \mathcal{N}(\mu_2,\sigma_2^2) = \mathcal{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$.
	- Underlies diffusion forward process: $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$ — closed form because Gaussians compose under linear combo.

### 2.8.6 Central limit theorem

- $S_N = \sum_n X_n$, $Z_N = (S_N - N\mu)/(\sigma\sqrt{N}) \xrightarrow{d} \mathcal{N}(0,1)$.
- **Why we use Gaussians for noise** in physics, ML, statistics.
- **Practical**: gradient noise from minibatch SGD is approximately Gaussian by CLT (for large batches) — leads to the SDE view of SGD, Langevin dynamics, connection to Bayesian posteriors.

### 2.8.7 Monte Carlo approximation

- Approximate $p(y)$ for $y = f(x)$ by sampling $x \sim p(x)$, taking $f(x_s)$, building empirical distribution.
- **Foundation of**:
	- All sampling from LLMs (token sampling).
	- MCMC for Bayesian posteriors (HMC, NUTS).
	- Variational inference's reparameterization gradients.
	- Diffusion sampling (DDPM/DDIM/EDM/flow matching solvers).
	- RL policy gradients (REINFORCE), MCTS, value estimation.
	- Self-consistency / majority-vote LLM reasoning (Wang et al.).
- One of the most practically important tools in the entire book — though the chapter only spends half a page on it (more in later chapters / Murphy book 2).

---

## What to actually internalize for modern LLM/frontier-lab work

Rough priority ranking based on what comes up in practice:

1. **Chain rule of probability** + **categorical/softmax** + **log-sum-exp/cross-entropy from logits** — this IS the LLM objective and loss. Understand it cold including numerical-stability tricks.
2. **Temperature in softmax** + sampling — directly controls all generation behavior; understand $T\to 0$ vs $T\to\infty$ limits.
3. **Bayes' rule + base-rate reasoning** — for thinking about RLHF/DPO as posterior, in-context learning as implicit inference, calibration, uncertainty.
4. **Change of variables (Jacobian)** — for flows, diffusion, any generative model with explicit density.
5. **Convolution of Gaussians + CLT** — for diffusion forward processes, SGD noise, signal propagation.
6. **Epistemic vs aleatoric uncertainty** — for UQ, hallucination, calibration, active learning.
7. **Conditional independence** — for graphical thinking, causal masks, separability.
8. **Heavy-tailed distributions (Student-t, Laplace)** — robust modeling, regularization interpretations (L1 ↔ Laplace), t-SNE.
9. **Beta / Gamma / etc** — useful for Bayesian-flavored work but rare in core LLM training.
10. **Anscombe / Datasaurus** — keep in mind when reading benchmark summaries; visualize don't summarize.

The rest (mean/variance/mode definitions, basic cdf/pdf relations, Bernoulli/Binomial) is just bedrock you should be able to write down without thinking. If any of that feels rusty, fix it before tackling the rest of the book.
