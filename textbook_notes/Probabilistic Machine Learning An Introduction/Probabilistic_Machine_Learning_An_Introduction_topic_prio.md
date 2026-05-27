# Probabilistic Machine Learning: An Introduction — Topic Priority Guide

> Kevin Murphy, MIT Press 2022 (3rd printing Jan 2025). ~800 pages.
> Notes from the perspective of: if you're working on frontier LLMs today, what in this book actually pays rent?

## TL;DR — Honest Take on the Book

- **What it is**: the standard graduate-level "modern ML" textbook. Encyclopedic, well-written, slightly outdated in NN coverage (the 2022 cutoff is real — RLHF, scaling laws, MoE post-Switch, modern post-training, none of that).
- **What it's great for**: getting probabilistic intuition down cold. Bayes, KL, entropy, MLE, calibration, decision theory. This is the lens that lets you read papers like *"Direct Preference Optimization"* or *"Bayesian Active Learning"* and not feel lost.
- **What it's weak for**: anything you'd want to know about the post-2022 LLM stack. Transformer chapter is fine but light. No RLHF, no chinchilla/scaling, no MoE depth, no diffusion, no flash attention, no inference systems. Pair with *Sebastian Raschka's Build a LLM From Scratch* or papers for that.
- **How I'd use it**: keep it as a reference. Use the foundations (Parts I + II + Ch 8) as the "math you should be able to do in your sleep". Skim DNN chapters. Read transformer + self-supervised + dimensionality reduction sections carefully.

## Priority Tier List

**S-tier — directly load-bearing for LLM work**
- Ch 6 **Information Theory** — cross-entropy *is* the LLM loss, KL *is* RLHF/DPO, perplexity *is* the eval. If you can't derive these in your sleep you have a hole.
- Ch 8 **Optimization** — SGD, Adam(W) (preconditioned SGD), LR schedules, momentum. Every choice you make at training time lives here.
- Ch 15 **Neural Networks for Sequences** — attention + transformers. The whole point.
- Ch 19 **Learning with Fewer Labeled Examples** — transfer learning, self-supervised pre-training, fine-tuning. This *is* the LLM paradigm.

**A-tier — you need solid intuition here**
- Ch 2/3 **Probability** — sampling, distributions, exponential family, mixture models. Sampling especially: temperature, top-k, top-p are reparametrizations of categorical sampling.
- Ch 4 **Statistics** — MLE, MAP, regularization, Bayesian basics, bias-variance. The conceptual scaffolding.
- Ch 5 **Decision Theory** — calibration, ROC, Bayes risk, model selection, NHST. Used heavily in eval.
- Ch 7 **Linear Algebra** — SVD, eigendecomp, matrix calculus. The substrate of every weight matrix you'll ever look at.
- Ch 13 **NN for Tabular Data** — backprop, init, regularization. Backprop derivation is non-negotiable.
- Ch 20 **Dimensionality Reduction** — autoencoders, VAEs, word embeddings. VAE for the diffusion crowd, word2vec/contextual embeddings for the rep-learning crowd.

**B-tier — useful, often used, but not where you'll live**
- Ch 14 **NN for Images** — relevant for vision-language models, ViT lineage.
- Ch 1 **Introduction** — fine framing, skim.
- Ch 9/10 **LDA, Logistic Regression** — logistic regression matters because it *is* the final softmax head of every classifier you build.
- Ch 17 **Kernel Methods** — GPs come back via NTK / Bayesian DL literature. SVMs less so.

**C-tier — read if you need it**
- Ch 11 **Linear Regression** — classical, well-trodden. Ridge / Lasso ideas show up in regularization.
- Ch 12 **GLMs** — pretty classical, mostly subsumed by NNs.
- Ch 16 **Exemplar-based Methods** — KNN/metric learning has a quiet comeback in RAG and embedding-based retrieval.
- Ch 18 **Trees, Forests, Boosting** — still wins on tabular but irrelevant to LLM training.
- Ch 21 **Clustering** — useful as exploratory tooling.
- Ch 22 **Recommender Systems** — useful if you do personalization, not core to LLM training.

**D-tier — skip unless niche use case**
- Ch 23 **Graph Embeddings** — niche. GNNs matter in some specialized domains but not in mainstream LLM work.

---

# Chapter-by-Chapter

## Ch 1: Introduction (S-skim)

Standard "what is ML" framing — supervised/unsupervised/RL, datasets, the three regimes. Skip if you've been around the block. Two things worth flagging:

- **No free lunch theorem** (§1.2.4): used rhetorically a lot. The actual content: averaged over *all* possible problems no algorithm is best. In practice this matters basically never — we don't sample uniformly from problem space, we have inductive biases that work.
- **Self-supervised learning** (§1.3.3) gets a one-paragraph treatment here but the whole modern LLM paradigm *is* self-supervised. Read it as a teaser.

## Part I: Foundations

### Ch 2: Probability — Univariate Models (A-tier)

The fundamentals. If you only remember a few things:

- **Bayes' rule** (§2.3): $p(h|e) = p(e|h)p(h)/p(e)$. Comes up everywhere: posterior inference, RLHF as a posterior update, in-context learning interpreted as Bayesian inference, calibration.
- **Distributions you must know cold**:
  - **Bernoulli / Categorical** — every classifier output, every next-token distribution.
  - **Gaussian** — universal default for continuous quantities; reparam trick relies on it.
  - **Student-t** — heavy tails. Worth knowing for robust regression and as a Gaussian-mixture stand-in.
  - **Laplace** — $\ell_1$ prior, sparsity. Connects to lasso.
- **Softmax** (§2.5.2): $\text{softmax}(x)_i = e^{x_i}/\sum_j e^{x_j}$. The most important nonlinearity in deep learning. Temperature $T$ rescales logits: $\text{softmax}(x/T)$. Worth deriving the gradient by hand once.
- **Log-sum-exp trick** (§2.5.4): numerically stable softmax/cross-entropy. Burned into every framework but you should know why $\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$ where $m = \max_i x_i$.
- **Transformations of RVs / change of variables** (§2.8): foundation of **normalizing flows**. Diffusion models use a noisier cousin (reverse SDEs) but the Jacobian-determinant intuition is the same.
- **CLT** (§2.8.6): mostly relevant because *initialization* uses Gaussian-by-CLT reasoning (Xavier/Kaiming).
- **Monte Carlo approximation** (§2.8.7): the entire foundation of sampling-based inference, RL value estimation, evaluation pipelines.

**Where it bites in practice**:
- Temperature sampling, top-k/p, beam search — all categorical-distribution gymnastics.
- KL between two Gaussians appears constantly (VAE ELBO, PPO penalties, distillation between teacher/student).
- Numerical issues with softmax in long-context attention.

### Ch 3: Probability — Multivariate Models (A-tier)

Where it gets concrete. Key bits:

- **Multivariate Gaussian** (§3.2): mean vector + covariance matrix. Marginals and conditionals close under affine ops — this is *why* Gaussians dominate ML.
  - **Mahalanobis distance**: $\sqrt{(x-\mu)^T \Sigma^{-1}(x-\mu)}$. Geometry of "how unlikely" in covariance-aware way.
  - **OOD detection** literature uses this directly.
- **Exponential family** (§3.4): the unifying lens. Bernoulli, Categorical, Gaussian, Beta, Dirichlet, Gamma — all $p(x|\theta) = h(x)\exp(\eta(\theta)^T T(x) - A(\theta))$.
  - Why care: many "tricks" (natural gradients, conjugate priors, sufficient statistics, MLE on exp-fam = moment matching) generalize beautifully.
  - **Maxent derivation** (§3.4.4): exp-fam = max entropy distribution given moment constraints. This is the basis of the "softmax is the right thing" argument.
- **Mixture models** (§3.5): GMMs, Bernoulli mixtures. Mostly useful conceptually — modern generation uses learned mixture-of-Gaussian outputs in some places (e.g. WaveNet-style speech, mixture density networks).
- **PGMs / graphical models** (§3.6): less central than they used to be. Worth knowing the d-separation / Markov blanket vocab. The big idea (factorize joint distribution by conditional independence) underlies autoregressive LMs trivially: $p(x_1,...,x_T) = \prod_t p(x_t | x_{<t})$.

**Where it bites in practice**:
- Diffusion models are basically Gaussian Markov chains with learned reverse kernels.
- KL between Gaussians has a closed form ($\frac{1}{2}[\text{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1-\mu_0)^T\Sigma_1^{-1}(\mu_1-\mu_0) - k + \log\frac{|\Sigma_1|}{|\Sigma_0|}]$) — memorize at least the spherical case.
- Mixture-of-experts uses the same gating math as Gaussian mixtures, just with neural gates.

### Ch 4: Statistics (A-tier)

The conceptual frame for "how do we estimate things".

- **MLE** (§4.2): $\hat\theta_{\text{MLE}} = \arg\max_\theta p(D|\theta)$. Next-token-prediction = MLE on the data distribution. This is the *entire* LLM pretraining objective. Stop and stare at this connection if it's not crisp.
  - Equivalent to minimizing forward KL: $\arg\min_\theta D_{\text{KL}}(p_{\text{data}} \| p_\theta)$ which is why models trained this way are *mode-covering* (try to put mass everywhere data has mass).
- **MAP / regularization** (§4.5): adds prior. Weight decay is a Gaussian prior. $\ell_1$ penalty is a Laplace prior. Crucial intuition.
  - **Early stopping** is also implicitly regularization — connects to bias-variance.
- **Bayesian statistics** (§4.6): conjugate priors, posterior predictive. Not used directly in big LLMs but the Bayesian frame for in-context learning (Xie et al. 2022, Wang et al. 2024) makes this important.
  - **Beta-binomial**, **Dirichlet-multinomial**, **Gaussian-Gaussian** — three conjugate pairs to know.
- **Frequentist** (§4.7): bootstrap, confidence intervals.
  - **Bias-variance tradeoff** (§4.7.6): central organizing principle but modern DL kind of breaks it (overparameterized models defy classical predictions — see "double descent" literature). Still worth knowing for the vocabulary.
- **Bootstrap** (§4.7.3): the most useful frequentist tool. Use it whenever you want a confidence interval on an eval metric and the analytical form is gnarly.

**Where it bites in practice**:
- Every loss function you write is either MLE or MLE + regularizer.
- "Why do we need temperature?" — because MLE trains mode-covering models, sampling at $T=1$ is high entropy; lowering $T$ moves toward MAP/mode.
- Bayesian model averaging arguments come up in ensembling/SWA literature.

### Ch 5: Decision Theory (A-tier)

Underused chapter — read it carefully.

- **Bayes risk** (§5.1): pick the action minimizing expected loss under the posterior. The "right" thing to do under uncertainty.
- **Classification with cost-sensitive loss** (§5.1.2): when false positives ≠ false negatives. Relevant for safety classifiers, eval grader design.
- **ROC / PR curves** (§5.1.3-4): worth being able to interpret in your sleep. PR is much more meaningful for imbalanced data — most safety/eval problems are imbalanced.
- **Calibration / probabilistic prediction** (§5.1.6): a model is calibrated if among predictions of confidence $p$, the actual frequency is $p$. **Huge topic for LLMs** — base models are well-calibrated; RLHF *destroys* calibration (the OpenAI GPT-4 system card chart is the canonical reference).
- **Bayesian model selection** (§5.2.2): marginal likelihood (evidence) — connects to information criteria (AIC, BIC, WAIC). Mostly used for hyperparameter selection in smaller models; in deep learning we use validation loss.
- **Occam's razor** (§5.2.3): simpler models with similar fit win. The Bayesian Occam factor explains this. Worth reading even if you don't use it directly.
- **NHST / p-values** (§5.5): you'll need this every time you read an experiment paper. Section 5.5.4 ("p-values considered harmful") and 5.5.5 ("why isn't everyone a Bayesian") are actually opinionated and worth the read.

**Where it bites in practice**:
- "Is my eval result significant?" → bootstrap CI + paired t-test or sign test.
- Calibration metrics: ECE, reliability diagrams. Every paper releasing a safety classifier should show these.
- "How do I pick which model to ship?" → expected reward under posterior over user preferences, basically Bayes risk in disguise.

### Ch 6: Information Theory (S-tier — read like your job depends on it)

If you skip everything else in the book, do not skip this chapter.

- **Entropy** $H(p) = -\sum_x p(x)\log p(x)$ (§6.1.1): measures uncertainty.
- **Cross entropy** $H(p,q) = -\sum_x p(x)\log q(x)$ (§6.1.2): **this is the LLM loss**. Pretraining minimizes $H(p_{\text{data}}, p_\theta)$ per-token, averaged over the corpus. Internalize this.
- **Perplexity** $\text{PPL} = e^{H(p,q)}$ (§6.1.5): the most-quoted LM eval metric. Roughly "effective branching factor". A model with PPL 10 is "as confused as if it had to choose uniformly among 10 options".
- **KL divergence** $D_{\text{KL}}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$ (§6.2):
  - **Non-symmetric**. Crucial distinction.
  - **Forward KL** $D_{\text{KL}}(p_{\text{data}} \| p_\theta)$: mode-covering. Standard MLE.
  - **Reverse KL** $D_{\text{KL}}(p_\theta \| p_{\text{data}})$: mode-seeking. Used in RL fine-tuning (PPO, GRPO) and VAE.
  - $D_{\text{KL}}(p \| q) = H(p,q) - H(p)$ — cross-entropy minus entropy. When $p$ is fixed (data), minimizing CE = minimizing KL.
  - Read §6.2.6 carefully. Mode-seeking vs mode-covering is one of the deepest practical distinctions in generative modeling.
- **Mutual information** (§6.3): $I(X;Y) = D_{\text{KL}}(p(x,y) \| p(x)p(y))$. Foundation of:
  - **InfoNCE / contrastive learning** (CLIP, SimCLR) — lower bound on MI.
  - **Information bottleneck** (Tishby) — interpretive lens for representation learning.
- **Sufficient statistics** (§6.3.9): connects back to exp-fam.
- **Fano's inequality** (§6.3.10): theoretical limits on classification given limited information.

**Where it bites in practice — extensive coverage because this matters**:

- **Pretraining objective is cross-entropy on tokens**. Literally. The "loss" line in your wandb run is mean per-token CE.
- **PPO/GRPO** in RLHF use a KL penalty term: $\mathcal{L} = -\mathbb{E}_{\pi_\theta}[r(s,a)] + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$. The $\beta$ controls how far you drift from the SFT model. Tuning this is half of RLHF practice.
- **DPO** (Direct Preference Optimization) is derived from the KL-regularized RL objective in closed form — read the DPO paper after you understand reverse KL.
- **Distillation** is forward-KL minimization between teacher and student logits. Temperature softening (Hinton 2015) makes the targets less mode-collapsed.
- **Knowledge distillation loss**: $\mathcal{L} = \alpha H(y_{\text{true}}, p_s) + (1-\alpha) T^2 D_{\text{KL}}(p_t^{(T)} \| p_s^{(T)})$ where $p^{(T)}$ is temperature-softened.
- **InfoNCE objective** (van den Oord 2018): for $N$ samples one of which is a positive, $\mathcal{L} = -\log \frac{e^{f(x,y^+)/\tau}}{\sum_i e^{f(x,y_i)/\tau}}$. This is the entire CLIP loss. It lower-bounds MI between $x$ and $y^+$.
- **Mutual information estimators** (MINE, NWJ, JSD bounds) come up in disentanglement and representation analysis.
- **Bits per byte / bits per character** is just another transformation of entropy used to compare models across tokenizers.

If you really want to understand this chapter, derive the following by hand:
1. CE = entropy + KL (when underlying $p$ is fixed, minimizing CE = minimizing KL).
2. KL between two univariate Gaussians.
3. Why softmax + CE has nice gradient $\partial \mathcal{L}/\partial z = p - y$ (predicted minus one-hot).
4. The DPO objective from the KL-regularized RL objective (this requires §6.2 + Ch 8).

### Ch 7: Linear Algebra (A-tier)

Murphy's version is a pretty good crash course. Hit the highlights:

- **Norms** ($\ell_2$, $\ell_1$, $\ell_\infty$, Frobenius, spectral) (§7.1.3): you'll use these constantly. Spectral norm = largest singular value — used in spectral normalization (GAN stabilization, Lipschitz constraints).
- **Matrix structure** (§7.1.5): symmetric / PSD / orthogonal / unitary. Attention's QK matrix isn't symmetric, but the kernel $QK^T$ at score level often gets compared to symmetric variants.
- **Matrix multiplication views** (§7.2): inner-product view, outer-product view, column-space view. Worth being fluent in all three — different mental models help in different contexts (e.g. low-rank approximation reasoning).
- **Eigendecomposition (EVD) / SVD** (§7.4, §7.5): **the single most useful linear algebra concept for ML**.
  - Every $m \times n$ matrix $A = U\Sigma V^T$.
  - PCA = eigendecomp of covariance / SVD of centered data matrix.
  - **LoRA**: $W \mapsto W + BA$ where $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times d}$ — low-rank update, motivated by the empirical observation that fine-tuning updates *are* low-rank.
  - **Truncated SVD** (§7.5.5): keep top-$k$ singular values. Foundation of model compression, ALBERT-style factorization.
  - **Spectral analysis of attention/MLP weights**: papers like Yang's tensor programs / muP use eigenvalue distributions to set widths.
- **Matrix calculus** (§7.8): you need this to do backprop derivations. Murphy's notation conventions (numerator vs denominator layout) are a pain — pick one and stick with it.
  - Gradients of common quantities: $\nabla_x x^T A x = (A + A^T)x$, $\nabla_x \|x\|_2^2 = 2x$, $\nabla_A \text{tr}(AB) = B^T$.
- **Solving systems** (§7.7): rarely used directly in DL training, but appears in numerical methods for second-order optimizers (K-FAC, Shampoo).

**Where it bites in practice**:
- Understanding why attention is $O(n^2)$ — it's a matmul $QK^T$ of two $n \times d$ matrices.
- LoRA, DoRA, GaLore, and friends — all linear algebra tricks for parameter-efficient finetuning.
- Quantization (int8, fp4) lives in the linear algebra layer and needs you to understand rank, norms, and conditioning.
- Singular value distributions of LLM weight matrices have become a research subject (e.g. "outliers" in attention weights).

### Ch 8: Optimization (S-tier)

The other "do not skip" chapter, alongside Information Theory.

- **First-order methods** (§8.2): GD step $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.
  - **Momentum** (§8.2.4): $v \leftarrow \beta v + \nabla \mathcal{L}$; $\theta \leftarrow \theta - \eta v$. Crucial.
  - **Convergence rates**: convex Lipschitz $O(1/\sqrt{T})$, strongly convex $O(1/T)$, Nesterov $O(1/T^2)$. DL is none of these and we don't care about formal rates much in practice.
- **Second-order methods** (§8.3): Newton, BFGS, L-BFGS. Almost never used in LLM training (too memory hungry). Trust region methods reappear in RL (TRPO).
- **SGD** (§8.4): the workhorse.
  - **Application to finite sums** (§8.4.1): pretraining is just SGD on a giant finite (or streaming) sum.
  - **LR schedules** (§8.4.3): the most impactful hyperparameter after learning rate itself.
    - **Warmup + cosine**: the dominant pattern. Warmup avoids early-training instability with Adam; cosine gives smooth annealing.
    - **WSD (warmup-stable-decay)**: emerging alternative used in continued pretraining and small-budget runs.
  - **Preconditioned SGD** (§8.4.6): the most important section in this chapter. Includes AdaGrad, RMSProp, Adam, AdamW.
    - **Adam**: $m \leftarrow \beta_1 m + (1-\beta_1)g$; $v \leftarrow \beta_2 v + (1-\beta_2)g^2$; $\theta \leftarrow \theta - \eta \hat{m}/(\sqrt{\hat{v}} + \epsilon)$. The default.
    - **AdamW**: decouples weight decay from the gradient update. The actual default for LLM training.
    - **Lion** (Google 2023, not in book): sign-of-momentum, no second moment. Memory savings, similar perf.
    - **Shampoo / Distributed Shampoo / SOAP** (2023-24, not in book): full-matrix preconditioning that actually scales. Several frontier labs use variants.
    - **Muon** (2024, not in book): newton-schulz orthogonalization on the gradient. Hot in 2025.
- **Constrained optimization** (§8.5): Lagrangians, KKT. Less directly relevant for LLM training but appears in RL (TRPO, constrained policy opt).
- **Proximal methods** (§8.6): foundation of $\ell_1$ optimization (ISTA). Not central to LLM training.
- **Bound optimization / EM** (§8.7): EM algorithm is the canonical example. Used in VAE ELBO (sort of), and in classical mixture models.
- **Blackbox / derivative-free** (§8.8): evolutionary methods, CMA-ES. Niche but reappears in hyperparameter search, prompt optimization.

**Where it bites in practice — extensive coverage because every training run lives here**:

- **Learning rate** is the most important hyperparameter by far. Wrong by 3× and your model is dead.
- **muP / mup transfer** (Yang & Hu 2022): scales the LR per-layer so that LR transfers across model sizes. Big practical win.
- **Gradient clipping**: not in this chapter but ubiquitous in LLM training. Standard is global norm clip at 1.0.
- **Mixed precision**: bf16/fp16 forward, fp32 accumulate. Master weights in fp32. Loss scaling for fp16.
- **ZeRO / FSDP / TP / PP**: parallelization strategies. All implementations of "SGD distributed across many devices" with different tradeoffs in communication and memory.
- **Batch size scaling laws**: critical batch size grows with training progress (McCandlish et al. 2018). Beyond critical BS you waste compute.
- **EMA of weights**: not in this chapter but standard for stable evaluation. Polyak averaging from §8.4.4 is the formal version.
- **Sharpness-aware minimization (SAM)**: not in this chapter. Looks for flat minima. Sometimes used at end of training.

Hot research areas:
- **Optimizer design at scale** — Muon, SOAP, Sophia, Lion. Most new optimizer papers ship "Shampoo with a budget you can afford".
- **Second-order methods that scale** — K-FAC, Shampoo. The hope is to recover Newton-like rates without the $O(d^2)$ memory.
- **Learning rate schedules and annealing** — WSD, "schedule-free" optimizers (Defazio 2024).
- **Adam vs SGD distinction**: Adam's effective LR per parameter is data-dependent. Why does Adam work for transformers but SGD doesn't? Active research (Kunstner et al. 2023 attribute it to noise / heavy-tailed gradients).

## Part II: Linear Models

### Ch 9: Linear Discriminant Analysis (B-tier — skim)

GDA, naive Bayes, gen-vs-discriminative debate. The generative-vs-discriminative argument used to be a big deal; modern LLMs are *both* (autoregressive = generative, but we use them discriminatively all the time for classification via prompting / few-shot). Worth reading §9.4 for vocabulary. Skip the rest.

Honest take: naive Bayes was the spam filter of 2005. The conceptual point — modeling $p(x|y)p(y)$ vs $p(y|x)$ — comes up in generative classification and energy-based models, but you don't need this chapter for that.

### Ch 10: Logistic Regression (B-tier)

The model under every classification softmax. Worth reading because:
- It's the simplest case where you can derive gradient + IRLS + Laplace approx by hand.
- The softmax layer at the top of every LM head *is* multinomial logistic regression on top of the hidden state.
- **MAP estimation** (§10.2.7) → weight decay
- **Standardization** (§10.2.8) → layer norm / RMS norm cousin

**Where it bites in practice**:
- Reward models in RLHF are scalar-head transformers — basically logistic regression on top of transformer features (binary preference).
- Linear probes on hidden states are logistic regression. Useful for interpretability ("does layer X linearly encode property Y").
- Bi-tempered loss (§10.4.2): noise-robust loss. Sometimes used in label-noisy settings.

### Ch 11: Linear Regression (C-tier)

Classical content. Skip unless you specifically want:
- **Ridge regression** (§11.3) — weight decay derivation in closed form.
- **Connection between ridge and PCA** (§11.3.2) — useful intuition.
- **Lasso** (§11.4) — $\ell_1$ regularization. Conceptually still important (e.g. SAE sparsity penalties), even if Lasso itself isn't used in DL.
- **ARD** (§11.7.7) — Bayesian relevance determination. Conceptual ancestor of "feature importance" stuff.

You will never train a linear regression model in production. But ridge intuitions transfer.

### Ch 12: Generalized Linear Models (C-tier — skim)

The unifying frame: $\mathbb{E}[y|x] = g^{-1}(w^T x)$ where $g$ is a link function, $y$ is exp-fam. Pretty but rarely directly useful. Worth one read to internalize the pattern.

## Part III: Deep Neural Networks

### Ch 13: Neural Networks for Tabular Data (A-tier for the basics)

Despite the name, this is where Murphy introduces MLPs and backprop. Don't skip.

- **XOR problem** (§13.2.1): classical motivation for hidden layers.
- **Activation functions** (§13.2.3): ReLU, GELU, SiLU/Swish. SiLU and GELU dominate transformers. SwiGLU (Shazeer 2020, not in Murphy) is in basically every modern LLM.
- **Importance of depth** (§13.2.5): empirical and theoretical.
- **Backpropagation** (§13.3): non-negotiable to understand. Derive it once by hand.
  - **Forward vs reverse mode** (§13.3.1): we use reverse for ML because $\nabla_\theta \mathcal{L}$ has 1 output, many inputs.
  - **Vector-Jacobian products** (§13.3.3): mental model behind every autograd implementation.
  - **Computation graphs** (§13.3.4): how PyTorch / JAX represent the program.
- **Training tricks** (§13.4):
  - **LR tuning** (§13.4.1): see Ch 8.
  - **Vanishing/exploding gradients** (§13.4.2): why pre-LayerNorm + residuals + careful init are the standard recipe.
  - **Non-saturating activations** (§13.4.3): ReLU and friends.
  - **Residual connections** (§13.4.4): the most important architecture innovation since backprop. Every transformer block has $x \mapsto x + \text{Attn}(LN(x))$.
  - **Parameter init** (§13.4.5): Xavier, Kaiming, etc. **muP** (2022) is the modern frontier — Murphy mentions but doesn't dive in.
- **Regularization** (§13.5):
  - **Weight decay** — basically everyone uses it. AdamW handles it correctly.
  - **Dropout** — less common in LLMs (interferes with residual stream interpretability, and is often empirically unnecessary at scale).
  - **Bayesian NNs** (§13.5.5): nice but expensive. SWAG, MC dropout are practical hacks.
  - **Regularization effects of SGD** (§13.5.6) — *very* interesting research area (implicit regularization, flat minima).
  - **Overparameterized models** (§13.5.7) — double descent, lottery ticket, modern generalization theory.
- **Mixture of experts** (§13.6.2): one paragraph here. Modern MoE (Switch, GShard, Mixtral, DeepSeek-V3, Qwen3-MoE) is a *huge* topic. The book undersells how central this has become. Active routing + sparse activation = current scaling frontier.

**Where it bites in practice**:
- Every transformer block is MLPs + attention.
- Pre-norm vs post-norm: pre-norm is standard now; post-norm is harder to train. Worth knowing why (gradient flow).
- Activation function ablations (Geglu, SwiGLU vs ReLU vs GELU): SwiGLU wins for transformers and nobody has a fully satisfying explanation.

### Ch 14: Neural Networks for Images (B-tier)

CNN content is mostly relevant for vision-language models now (CLIP, LLaVA, Llama 3-V, etc.). Highlights:

- **Convolutional layers** (§14.2.1): worth understanding the parameter sharing / locality inductive bias.
- **Normalization** (§14.2.4): BatchNorm (not used in transformers), LayerNorm (the standard), GroupNorm (sometimes for vision).
- **Image classification architectures** (§14.3): LeNet → AlexNet → VGG → ResNet. Worth knowing for context. ResNet's residual connection is the ancestor of every transformer.
- **ViT** is barely mentioned (book is 2022). It's just "transformer on patches" — read the original An et al. 2020 paper.
- **Generating images by inverting CNNs** (§14.6): kind of dated. Modern image generation is diffusion (DDPM, DDIM, EDM, Rectified Flow, Flow Matching). None of that is in this book. Use *Tony Duan's Diffusion Models from Scratch* or the Stable Diffusion / EDM papers.

### Ch 15: Neural Networks for Sequences (S-tier — read carefully)

The headline chapter for anyone doing LLMs.

- **RNNs** (§15.2): conceptually still important even if we don't use them. Worth understanding for:
  - **State-space models** (Mamba, S4, RWKV) — recurrent in nature. **Active and important** research area for long context and constant-memory inference. Murphy doesn't cover this.
  - **Linear attention / linear transformers** — equivalent to a specific RNN.
- **LSTMs / GRUs** (§15.2.7): gating mechanism. Conceptual ancestor of the gating in MoE.
- **Beam search** (§15.2.8): the canonical sequence decoding method. For modern LLMs we mostly use sampling (top-k, top-p, temperature) for open-ended generation; beam search hangs on in translation and constrained decoding.
- **1d CNNs / causal 1d CNNs** (§15.3): WaveNet lineage. Still used in audio (e.g. some text-to-speech).
- **Attention** (§15.4):
  - **Attention as soft dictionary lookup** (§15.4.1): the right mental model. Query against keys, weighted sum of values.
  - $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$.
  - **Parametric attention** (§15.4.3): the version used in transformers.
- **Transformers** (§15.5):
  - **Self-attention** (§15.5.1): every token attends to every other token (or causally to previous ones).
  - **Multi-head attention** (§15.5.2): run $h$ attention operations in parallel with different projections, concat, project.
  - **Positional encoding** (§15.5.3): sin/cos in the original transformer. Modern: **RoPE** (Su et al. 2021), **ALiBi**, **NoPE**. RoPE dominates. Murphy doesn't cover RoPE.
  - **Comparing transformers, CNNs, RNNs** (§15.5.5): worth reading. Transformers have $O(n^2)$ attention cost but parallel training (vs sequential RNN).
- **Efficient transformers** (§15.6): Linformer, Performer, Longformer-style sparse patterns. Largely superseded in practice:
  - **FlashAttention 1/2/3** (Dao et al. 2022-24, not in book): tiling + recomputation. The dominant attention kernel.
  - **Ring attention** (Liu et al. 2023): scales context across devices.
  - **Sliding window / striped / local-global** patterns still used (Gemma, Mistral).
  - **State-space models** (Mamba, Mamba-2): genuinely different paradigm.
- **Language models and unsupervised representation learning** (§15.7):
  - **Non-generative LMs** (§15.7.1): BERT, encoder models. Still important for retrieval (BGE, E5) and as small classification backbones.
  - **Generative LMs** (§15.7.2): GPT-style. The entire frontier.

**What's missing (huge gaps)**:
- **Scaling laws** (Kaplan et al. 2020, Chinchilla, Hoffmann et al. 2022) — completely absent. *This is the central organizing fact of modern LLM research.* Optimal compute allocation between params and tokens is Chinchilla; later refinements (e.g. for inference-cost-aware training) abound.
- **Mixture of experts** at scale — Switch, GShard, Mixtral, DeepSeekMoE, the MoE design space. Hot.
- **RLHF / RLAIF / DPO** — absent. Read Stiennon 2020, Ouyang 2022 (InstructGPT), Bai 2022 (Constitutional AI), Rafailov 2023 (DPO), Shao 2024 (GRPO).
- **Chain of thought / reasoning** — absent. Read Wei 2022, then o1 system card, then DeepSeek-R1.
- **Inference-time compute scaling** — absent. Test-time training, tree search over CoT, self-consistency, MCTS-style reasoning.
- **Long context techniques** — RoPE scaling (linear / NTK-aware / dynamic / YaRN), Ring attention, ALiBi, sliding window.
- **Efficient inference** — KV caching, speculative decoding (Leviathan 2023), continuous batching, paged attention (vLLM).
- **Tokenization** — BPE, WordPiece, SentencePiece, byte-level. Not in book. Matters more than people realize.

**Bottom line**: read this chapter for the attention/transformer foundations. Everything *after* the basics, supplement with papers.

## Part IV: Nonparametric Models

### Ch 16: Exemplar-based Methods (B-tier)

KNN + distance metric learning + KDE. The reason this is B-tier and not C-tier:

- **Deep metric learning** (§16.2.2): the conceptual ancestor of contrastive embeddings. Triplet loss, margin loss, InfoNCE — same family.
- **KNN in embedding space** is the entire substrate of **RAG** (retrieval-augmented generation), **dense retrieval** (Karpukhin et al. DPR, ColBERT, BGE, E5), and **vector databases** (FAISS, Pinecone, Weaviate). Sufficient to make this chapter worth reading.
- **Curse of dimensionality** (§16.1.2): worth knowing because vector search in 768- or 1024-dim spaces requires approximate methods (HNSW, IVF) and the geometry is unintuitive.
- **KDE** (§16.3): mostly classical statistics. Skip.

### Ch 17: Kernel Methods (B-tier)

- **Mercer kernels** (§17.1): kernel trick.
- **Gaussian Processes** (§17.2): the canonical Bayesian nonparametric.
  - GPs come back in **neural tangent kernel (NTK)** theory — infinite-width networks behave like GPs. NTK predictions about training dynamics are an active research strand.
  - Bayesian optimization (using GPs) is the standard for hyperparameter search at moderate scale.
- **SVMs** (§17.3): historical. Largely irrelevant for LLM work. Worth knowing the duality argument once because dual formulations recur (e.g. kernelized attention).
- **Relevance vector machines** (§17.4): obscure now.

**Where it bites in practice**:
- NTK / lazy training regime: theoretical predictions about wide NNs.
- Bayesian optimization via GPs for hparam search.
- Kernel attention / linear attention reformulations.

### Ch 18: Trees, Forests, Bagging, Boosting (C-tier)

XGBoost / LightGBM / CatBoost still dominate tabular, but irrelevant for LLM training. If you're at a frontier lab doing LLMs, you might *use* boosted trees once a year for some ad-hoc dataset analysis.

**Useful ideas**:
- **Boosting** (§18.5): conceptual model of "fit residuals iteratively". Connects loosely to chain-of-thought / iterative refinement methods.
- **Feature importance** (§18.6.1): SHAP and friends. Used in interpretability of structured-data models.
- **Bagging** (§18.3): ensembling. Still important for high-stakes deployments and for understanding deep ensembles.

## Part V: Beyond Supervised Learning

### Ch 19: Learning with Fewer Labeled Examples (S-tier)

This chapter contains the entire modern LLM paradigm in disguise.

- **Data augmentation** (§19.1): for vision still important; for text we mostly use synthetic data and paraphrasing now.
- **Transfer learning** (§19.2):
  - **Fine-tuning** (§19.2.1): the entire post-pretraining workflow. Full FT, partial FT, LoRA, QLoRA, DoRA, IA3, prefix tuning, prompt tuning.
  - **Adapters** (§19.2.2): the conceptual ancestor of LoRA. Houlsby 2019.
  - **Unsupervised pre-training (self-supervised)** (§19.2.4): **THE** modern paradigm. Next-token prediction, masked language modeling, contrastive (SimCLR, MoCo, DINO, BYOL).
- **Semi-supervised learning** (§19.3): less central for modern LLM pretraining (we have basically unlimited unsupervised text), but **back in vogue** for:
  - Weakly-supervised training on web-scraped pairs (CLIP).
  - Self-training (pseudo-labeling) for distillation pipelines.
  - **Deep generative models for SSL** (§19.3.6): VAE + classifier hybrids. Mostly historical.
- **Active learning** (§19.4): which data should I label next?
  - Modern incarnation: **data selection** for pretraining (DSIR, DataComp, Doge, Multi-armed bandit data mixing), curriculum design, hard-example mining in fine-tuning.
  - **Information-theoretic AL** ties back to Ch 6.
- **Meta-learning** (§19.5): MAML and friends. Less central now — in-context learning is the de-facto meta-learner.
- **Few-shot learning** (§19.6): the original meaning was "low-shot classification". Now "few-shot" usually means *in-context* few-shot (Brown 2020 / GPT-3). Distinct mechanism.

**Where it bites in practice — extensive coverage**:

- **The entire LLM stack is**: self-supervised pre-training (§19.2.4) + supervised fine-tuning (instruction tuning) + RLHF/DPO + inference-time prompting (few-shot, CoT). Every step is in this chapter or descended from it.
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA is the dominant approach. Reduces trainable params by 10000×. Quantized variants (QLoRA, Dettmers 2023) further reduce memory.
- **In-context learning**: emergent few-shot capability. Theoretical work tries to explain it as Bayesian inference (Xie 2022), implicit gradient descent (von Oswald 2023), or implicit meta-learning. None of these stories are complete.
- **Continued pretraining / domain adaptation**: fine-tuning on domain-specific corpora before instruction tuning. Standard for specialized models (biomedicine, code, finance).
- **Distillation as transfer learning**: train a small student on a large teacher's outputs. Used widely for serving cost reduction (e.g. Gemma 2 was distilled from Gemini, Llama 3 8B → 1B distillations, etc.).
- **Data mixing / data curriculum**: which proportions of web text vs code vs math vs books? Active research (DoReMi, DataComp-LM).
- **Synthetic data** (Phi series, Anthropic / OpenAI internal pipelines): generate training data with a strong model. Quietly half the gains since 2023.

### Ch 20: Dimensionality Reduction (A-tier)

Mixed chapter. The relevant parts are:

- **PCA** (§20.1): worth knowing cold. Used for:
  - Activation analysis ("how many dimensions does this representation use?").
  - Approximation of attention / weight matrices.
  - Diagnostics.
- **Probabilistic PCA / Factor Analysis** (§20.2): bridge to VAE.
- **Autoencoders** (§20.3):
  - **Bottleneck / denoising / contractive** (§20.3.1-3): conceptual ancestors.
  - **Sparse autoencoders** (§20.3.4): **EXTREMELY HOT** in 2024–25. Anthropic's interpretability program (Bricken 2023 "Towards Monosemanticity", Templeton 2024 "Scaling Monosemanticity") uses SAEs on transformer activations to find interpretable features. This is one of the most active areas at frontier labs right now.
  - **Variational autoencoders** (§20.3.5): the foundation of **latent diffusion** (Stable Diffusion uses a VAE to compress images before diffusion). ELBO derivation is also a critical concept.
- **Manifold learning** (§20.4): mostly classical (Isomap, LLE, Laplacian eigenmaps).
  - **t-SNE** (§20.4.10): the dominant visualization technique for high-dim embeddings. UMAP (not in book) is the modern alternative.
- **Word embeddings** (§20.5):
  - **Word2vec / GloVe**: conceptual ancestors of LLM embeddings.
  - **Contextual embeddings** (§20.5.6): ELMo / BERT. The route from static embeddings to contextual ones is short but conceptually critical.

**Where it bites in practice**:
- **Sparse autoencoders for interpretability**: one of the most-funded areas in alignment research. Read Bricken et al. 2023 and Templeton et al. 2024 carefully after this chapter.
- **VAEs for latent diffusion**: Stable Diffusion's encoder is an SD-VAE. KL between encoder posterior and prior is the regularization.
- **Embedding models** (BGE, E5, OpenAI text-embedding-3): power retrieval, search, RAG.
- **Activation steering**: subtract / add directions in residual stream to control behavior. Conceptually like vector arithmetic in word2vec.

### Ch 21: Clustering (C-tier)

K-means, GMM, spectral clustering, hierarchical. Useful as exploratory tools.

- **K-means** (§21.3): you'll use it for diagnostic clustering of embeddings.
- **GMM** (§21.4.1): worth knowing for the EM example.
- **Spectral clustering** (§21.5): connects to graph Laplacian / GNN content. Worth a skim.

Not central to LLM work. Skim.

### Ch 22: Recommender Systems (C-tier)

If you do personalization / search ranking, this matters. For pure LLM training, almost irrelevant.

- **Matrix factorization** (§22.1.3): the workhorse. ALS, SVD, BPR.
- **Bayesian personalized ranking** (§22.2.1): the canonical implicit-feedback method.
- **Exploration-exploitation** (§22.4): bandits, Thompson sampling — relevant if you do A/B testing of models or active data selection.

### Ch 23: Graph Embeddings (D-tier)

GNNs are important in some specialized fields (drug discovery, fraud detection, knowledge graphs) but irrelevant to mainstream LLM work. Skip unless you specifically need it. The connection to transformers (which are GNNs on complete graphs with positional info) is intellectually nice but not practically actionable.

---

# What to Actually Do With This Book

## If you have a weekend
- Ch 6 (Information Theory) start to finish.
- Ch 8 §8.4 (SGD) and §8.4.6 (preconditioned SGD / Adam).
- Ch 15 §15.4-15.5 (attention + transformers).
- Ch 19 §19.2 (transfer learning) + §19.2.4 (self-supervised).

## If you have a week
- Add: Ch 2-3 (probability) skim, Ch 4 (statistics) read, Ch 5 (decision theory) read.
- Add: Ch 7 (linear algebra, especially SVD).
- Add: Ch 13 (NN basics, especially backprop) — fast read if you know it.
- Add: Ch 20 §20.3 (autoencoders + VAE) + §20.5 (word embeddings).

## If you're trying to fill foundational gaps
- Add Ch 17 §17.2 (GPs — for NTK / BayesOpt).
- Add Ch 16 (KNN + metric learning — for retrieval / RAG).
- Add Ch 19 §19.4 (active learning — for data selection).

## Pair the book with these for current frontier
- **Attention / FlashAttention**: Vaswani 2017, Dao 2022, Dao 2023.
- **Scaling**: Kaplan 2020, Hoffmann 2022 (Chinchilla), Sardana 2023 (Beyond Chinchilla for inference).
- **RLHF / Post-training**: Stiennon 2020, Ouyang 2022 (InstructGPT), Rafailov 2023 (DPO), Shao 2024 (GRPO), Bai 2022 (CAI).
- **MoE**: Shazeer 2017, Fedus 2021 (Switch), Jiang 2024 (Mixtral), DeepSeek-V3 tech report.
- **Reasoning**: Wei 2022 (CoT), o1 system card, DeepSeek-R1.
- **SSMs**: Gu 2023 (Mamba), Dao 2024 (Mamba-2).
- **Interpretability**: Bricken 2023, Templeton 2024, Marks 2024 (sparse feature circuits).
- **Long context**: Su 2021 (RoPE), Peng 2023 (YaRN), Liu 2023 (Ring attention).
- **Inference**: Leviathan 2023 (speculative decoding), Kwon 2023 (vLLM / PagedAttention).
- **Diffusion** (since Murphy skips it): Ho 2020 (DDPM), Karras 2022 (EDM), Lipman 2023 (Flow Matching).

## Things to remember even if you forget everything else

1. **Pretraining = MLE = minimizing cross-entropy = minimizing forward KL to the data distribution.**
2. **PPO/GRPO add a reverse-KL penalty to a learned reward objective.**
3. **DPO is the closed-form solution of the KL-regularized RL objective when the optimal policy is parameterized correctly.**
4. **Attention is a soft, learned associative memory: softmax(QK^T/√d) V.**
5. **LoRA exploits the empirical low-rank structure of fine-tuning updates.**
6. **AdamW = Adam + decoupled weight decay; almost all LLMs use it.**
7. **Calibration of LLMs is destroyed by RLHF.**
8. **Sparse autoencoders are the current best lever for mechanistic interpretability of LLM features.**
9. **Self-supervised pretraining + targeted post-training is the entire stack.**
10. **You will never run out of ways to use Bayes' rule.**
