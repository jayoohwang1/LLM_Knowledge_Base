# Chapter 5: Decision Theory

Quick orienting note before the bullets: this chapter is the conceptual bridge between "I have a posterior" and "what do I actually output / which model do I pick / how do I evaluate". For frontier LLM work, the bits that matter most in practice are: **proper scoring rules / cross-entropy / log-loss** (this is literally the training objective), **KL divergence** (RLHF, distillation, variational stuff), **calibration-adjacent ideas like reject option** (selective prediction, abstention, "I don't know"), **classification metrics** (eval benchmarks, reward model accuracy), **ERM + structural risk** (the whole modern training paradigm), and **Bayesian model selection / Occam's razor intuition** (scaling laws, model comparison). The Bayesian decision theory framing also underlies modern thinking about LLM agents that take actions under uncertainty. Less critical for current LLM research: minimax, admissibility, Neyman-Pearson, VC dimension. Detailed notes below.

## 5.1 Bayesian Decision Theory

- **Core setup**: turn beliefs $p(H|x)$ into **actions** $a \in \mathcal{A}$ via a **loss function** $\ell(h, a)$.
	- **Posterior expected loss / risk**: $\rho(a|x) \triangleq \mathbb{E}_{p(h|x)}[\ell(h,a)] = \sum_h \ell(h,a) p(h|x)$.
	- **Optimal policy / Bayes estimator**: $\pi^*(x) = \arg\min_a \rho(a|x)$.
	- Equivalent **utility formulation**: $U(h,a) = -\ell(h,a)$, max expected utility (MEU).
- **COVID drug example**: state = (age, disease status), action = (nothing, drug), loss in QALY units.
	- **Preferences**: any consistent set converts to ordinal cost scale (decision theory theorem).
- **Risk neutrality assumed**: indifferent between \$50 sure vs 50/50 \$0/\$100. Real agents are often **risk averse**; generalizes to risk-sensitive decision theory (not pursued).
	- **LLM relevance**: RL-from-AI-feedback and agentic systems often need risk-sensitive objectives (CVaR-style); recent safety work explicitly considers worst-case rather than expected.

### 5.1.2 Classification problems

- **Zero-one loss** $\ell_{01}(y^*, \hat{y}) = \mathbb{I}(y^* \neq \hat{y})$.
	- Optimal: **MAP** $\pi(x) = \arg\max_y p(y|x)$.
	- Practical note: this is what greedy decoding in LLMs implicitly does at each token, but the loss we *train* with is log-loss, not 0-1. That mismatch is part of why temperature/sampling matters.
- **Cost-sensitive classification**: general loss matrix $\ell_{ij}$.
	- Binary with $\ell_{00}=\ell_{11}=0$, $\ell_{10}=c\ell_{01}$: threshold becomes $1/(1+c)$ instead of $1/2$.
	- **LLM relevance**: refusal tuning (false-refuse vs unsafe-allow have very different costs); content moderation; selective classification.
- **Reject option / Chow's rule**: extra "don't know" action with cost $\lambda_r$ vs error cost $\lambda_e$.
	- Optimal: predict $y^* = \arg\max p(y|x)$ iff $p^* > \lambda^* = 1 - \lambda_r/\lambda_e$, else reject.
	- IBM **Watson** Jeopardy example: only buzzed in when confident.
	- **LLM relevance — very important**: this is the formal framework for **abstention / "I don't know"** which is one of the biggest open problems in LLM reliability. Selective prediction, calibration-aware decoding, knowing what you don't know — all variants of Chow's rule but with the harder problem of estimating $p^*$ in the first place (LLM confidences are often badly calibrated).

### 5.1.3 ROC curves

- **Class confusion matrix**: TP, FP, TN, FN. Standard definitions:
	- **TPR / sensitivity / recall** = TP/P
	- **FPR / fallout / type I rate** = FP/N
	- **TNR / specificity** = TN/N
	- **FNR / miss / type II rate** = FN/P
	- **Precision (PPV)** = TP/$\hat{P}$, **NPV** = TN/$\hat{N}$.
- **ROC**: TPR vs FPR as threshold $\tau$ swept. Hugs top-left = good.
- **AUC**: area under ROC. **EER**: equal error rate (FPR=FNR), intersection with anti-diagonal.
- **Class imbalance**: ROC insensitive to it (TPR, FPR are within-class ratios). PR curves are sensitive — useful in imbalanced regimes like retrieval.

### 5.1.4 Precision-recall curves

- Used when "negatives" not well-defined (object detection, retrieval).
- $\mathcal{P}(\tau) = TP/(TP+FP)$, $\mathcal{R}(\tau) = TP/(TP+FN)$.
- **AP (average precision)**: area under interpolated PR curve.
- **mAP**: mean over multiple PR curves (different classes/queries).
- Used heavily in detection (COCO mAP), retrieval, and increasingly in **RAG eval** (retrieval-precision @ K).
- **F-scores**: weighted harmonic mean. $F_1 = 2\mathcal{P}\mathcal{R}/(\mathcal{P}+\mathcal{R})$, $F_\beta$ weights recall by $\beta^2$.
	- Why harmonic? It punishes imbalance — predicting all positive at prevalence $10^{-4}$ gives arithmetic mean ~50% but harmonic mean ~0.02%.
- **Class imbalance affects PR**: $\text{Prec} = \frac{TPR}{TPR + (1/r) FPR}$ with $r = P/N$ — drops as positives become rarer. Don't naively average APs across imbalanced subproblems.
- **LLM relevance**: most LLM eval benchmarks (HumanEval pass@k, MMLU accuracy, etc.) collapse to simple accuracy because they're balanced or single-answer. But for **agent / tool-use evals**, **harmful-content classifiers**, **reward model accuracy**, **retrieval-augmented systems**, PR-style metrics dominate.

### 5.1.5 Regression losses

- **L2 / squared / quadratic**: $\ell_2(h,a) = (h-a)^2 \Rightarrow$ optimal = **posterior mean** $\mathbb{E}[h|x]$ (MMSE).
- **L1 / absolute**: $\ell_1 = |h-a| \Rightarrow$ optimal = **posterior median** (robust to outliers).
- **Huber**: quadratic for $|r|\le\delta$, linear beyond. Robust + differentiable.
	- **LLM relevance**: most generation isn't framed this way, but Huber-style losses show up in **value heads for RL** (PPO, AWR), in **reward modeling** (sometimes pairwise with margin), and **diffusion model training** has L2 baked in. Recent work uses L1-style for robustness vs noisy preference data.

### 5.1.6 Probabilistic prediction problems — **THE central section for modern ML**

- Action = a full distribution $q(Y|x)$. Loss compares to true $p(Y|x)$.
- **KL divergence**: $D_{KL}(p\|q) = \sum_y p(y) \log \frac{p(y)}{q(y)} = -\mathbb{H}(p) + \mathbb{H}_{ce}(p,q)$.
	- Asymmetric, $\ge 0$, =0 iff $p=q$.
	- Minimizing KL wrt $q$ = minimizing **cross-entropy** $\mathbb{H}_{ce}(p,q) = -\sum_y p(y)\log q(y)$.
	- **One-hot special case**: $\mathbb{H}_{ce}(\delta(Y=c), q) = -\log q(c)$ = **log loss**.
- **This is the LLM training objective.** Next-token prediction is exactly cross-entropy / log-loss against the empirical (one-hot) target. The fact that this is a **proper scoring rule** is why it works — it incentivizes calibrated probabilities, not just argmax.
- **Proper scoring rule**: $\ell(p,q) \ge \ell(p,p)$ with equality iff $p=q$. Cross-entropy qualifies via $D_{KL}\ge 0$.
- **Brier score**: $\text{BS}(p,q) = \frac{1}{N}\sum_{n,c}(q_{nc} - p_{nc})^2$. Squared error on probability vectors.
	- Also proper, less sensitive to rare extremes than log-loss (no $\log(0)$ blowup).
	- **Brier skill score**: $1 - \text{BS}/\text{BS}_\text{ref}$, normalizes vs baseline.
	- Use case: weather forecasting, calibration eval. Less common in LLM-land but the right tool for evaluating **token-level calibration** without the log-loss instability.
- **Why this section matters most for LLM research**:
	- **Pretraining**: log-loss / cross-entropy.
	- **Distillation**: KL$(p_\text{teacher} \| p_\text{student})$ — Hinton-style.
	- **RLHF (PPO)**: KL penalty against reference policy to prevent reward hacking. Forward vs reverse KL choice has huge practical implications (reverse KL = mode-seeking, forward = mode-covering).
	- **DPO and variants**: derived directly from a KL-regularized RL objective.
	- **Variational methods / VAEs / diffusion ELBO**: KL terms everywhere.
	- **Calibration research**: proper scoring rules are the right diagnostic; ECE alone misses things.

## 5.2 Choosing the "right" model — **High signal for ML research generally**

### 5.2.1 Bayesian hypothesis testing

- $M_0$ vs $M_1$. With 0-1 loss and uniform prior, pick $M_1$ iff **Bayes factor** $B_{1,0} = p(\mathcal{D}|M_1)/p(\mathcal{D}|M_0) > 1$.
- Like a **likelihood ratio** but integrates out parameters $\Rightarrow$ Occam-style automatic complexity penalty.
- **Jeffreys scale**: BF>10 strong, >100 decisive. Bayesian replacement for p-values.
- Coin example: $p(\mathcal{D}|M_0) = (1/2)^N$, $p(\mathcal{D}|M_1) = B(\alpha_1+N_1, \alpha_0+N_0)/B(\alpha_1,\alpha_0)$.

### 5.2.2 Bayesian model selection

- $\hat{m} = \arg\max_m p(m|\mathcal{D}) = \arg\max_m p(\mathcal{D}|m)p(m)$.
- **Marginal likelihood / evidence**: $p(\mathcal{D}|m) = \int p(\mathcal{D}|\theta, m) p(\theta|m) d\theta$.
- Example: polynomial regression — with $N=5$, degree 1 wins; with $N=30$, degree 2 wins. Evidence adapts to data size.

### 5.2.3 Occam's razor — **conceptually crucial**

- Complex models spread prior mass thinly → averaged likelihood is lower in "good" parameter region.
- **Conservation of probability mass**: $\sum_{\mathcal{D}'} p(\mathcal{D}'|m) = 1$ → complex models must spread predictions thin → can't concentrate as much mass on actual observation.
- **LLM relevance — high**: scaling laws are essentially empirical Occam's razor in reverse: when you have enough data, the "right" model class is bigger than you'd naively expect. The connection between **model size, data size, and effective capacity** in modern scaling work (Chinchilla, etc.) is exactly this tradeoff. Also relevant to **mechanistic interpretability** — simpler circuits favored by training dynamics is an implicit prior.

### 5.2.4 Connection between cross-validation and marginal likelihood

- $\log p(\mathcal{D}|m) = \sum_n \log p(y_n|x_n, \mathcal{D}_{1:n-1}, m)$ — sequential / prequential view.
- With plug-in approx: $\approx \sum_n \log p(y_n|x_n, \hat{\theta}_m(\mathcal{D}_{1:n-1}))$ — similar to **LOO-CV**.
- **LLM relevance**: this is why **held-out validation log-likelihood** is the right metric — it's a frequentist proxy for marginal likelihood. Modern pretraining eval literally watches val loss curves. Also undergirds **next-token prediction as model evaluation**.

### 5.2.5 Information criteria

- Form: $\mathcal{L}(m) = -\log p(\mathcal{D}|\hat{\theta},m) + C(m)$.
- **BIC**: $C = \frac{D_m}{2}\log N$. Approximates log marginal likelihood via Laplace.
	- Derivation involves Hessian / Occam factor $\log|\mathbf{H}|$ approximated as $D_m \log N$.
- **AIC**: $C = 2D_m$. Penalizes less, no $N$ dependence. Frequentist origin.
- **MDL**: identical form to BIC, motivated by coding theory (two-part code).
- **WAIC (Watanabe-Akaike)**: handles **singular models** where Fisher info matrix is degenerate.
	- $\text{WAIC} = -2\text{LPPD} + 2C$ with $\text{LPPD} = \sum_n \log \mathbb{E}_\theta[p(y_n|\theta)]$ over posterior.
	- $C = \sum_n \mathbb{V}_\theta[\log p(y_n|\theta)]$ = posterior variance of log-likelihood.
	- **Big deal for deep nets**: NNs are singular (overparam, non-identifiable params), so BIC/AIC formally don't apply. WAIC does.
	- **LLM relevance**: theoretical underpinning for why "effective parameters" $\neq$ raw param count in LLMs; connects to recent work on overparam, double descent, and posterior-based generalization bounds.

### 5.2.6 Posterior inference over effect sizes / Bayesian significance testing

- **Effect size vs null**: rather than asking "is $\Delta = 0$" (point null = straw man), ask "is $|\Delta| > \epsilon$".
- **ROPE** (region of practical equivalence): $R = [-\epsilon, \epsilon]$. Three meaningful hypotheses partition $\mathbb{R}$.
- **Bayesian t-test**: $d_i = e_i^1 - e_i^2$, $d_i \sim \mathcal{N}(\Delta, \sigma^2)$, posterior for $\Delta$ is Student-$t$. Compute $p(|\Delta|>\epsilon|d)$ directly.
- **Bayesian $\chi^2$-test**: rates from binomials with Beta posteriors; integrate.
- **LLM relevance**: model comparison benchmarks (one LLM beats another by 0.3% on MMLU — is that real?) is exactly this problem. Most current eval reports point estimates without uncertainty. Calls for paired Bayesian comparison have been growing in eval literature. Recent EleutherAI/Anthropic eval papers explicitly do this kind of analysis.

## 5.3 Frequentist Decision Theory

- **Risk**: $R(\theta, \delta) = \mathbb{E}_{p(x|\theta)}[\ell(\theta, \delta(x))]$. Averages over data, conditions on truth (flip of Bayesian).
- Gaussian mean estimation example with 5 estimators (MLE, median, fixed, posterior mean with two priors). MSE = variance + bias².
	- MLE risk = $\sigma^2/N$. Constant estimator: 0 variance, bias² = $(\theta^* - \theta_0)^2$.
	- "Best" depends on unknown $\theta^*$ — no uniformly best estimator.

### 5.3.1 Bayes risk vs maximum risk

- **Bayes risk**: $R_{\pi_0}(\delta) = \int R(\theta,\delta) \pi_0(\theta) d\theta$. Minimizer = Bayes estimator = same as posterior risk minimizer.
- **Maximum risk**: $R_\max(\delta) = \sup_\theta R(\theta,\delta)$. Minimizer = **minimax estimator**.
	- Minimax = Bayes under **least favorable prior**.
	- Overly conservative for non-adversarial settings.
	- **LLM relevance**: robustness research (adversarial training, jailbreak resistance) is essentially minimax. RLHF with adversarial red-teamers approximates this.

### 5.3.2 Consistent estimators

- $\hat{\theta}(\mathcal{D}) \to \theta^*$ in probability as $N \to \infty$. MLE is consistent (under identifiability).
- Unbiased $\neq$ consistent (example: $\delta = x_N$ is unbiased for mean but doesn't converge).
- In practice the model is always misspecified, so consistency is more of a sanity check than a goal — what we really want is to minimize KL to the data distribution.

### 5.3.3 Admissible estimators

- $\delta_1$ **dominates** $\delta_2$ if $R(\theta,\delta_1) \le R(\theta,\delta_2)$ for all $\theta$. **Admissible** = not dominated.
- **Wald**: admissible $\Leftrightarrow$ Bayes (under conditions).
- **Stein's paradox**: sample mean is **inadmissible** in $d \ge 3$ dimensions under squared loss — James-Stein shrinkage strictly dominates.
	- **LLM relevance**: shrinkage / regularization is fundamental. Modern weight decay, prior-regularized fine-tuning (e.g., KL to base model in RLHF), and LoRA's implicit low-rank prior all leverage this intuition.
- Concept of limited practical value — useless constant estimators are also admissible.

## 5.4 Empirical Risk Minimization — **The whole basis of modern deep learning**

### 5.4.1 Empirical risk

- **Population risk**: $R(f, p^*) = \mathbb{E}_{p^*(x,y)}[\ell(y, f(x))]$. Can't compute.
- **Empirical risk**: $R(f, \mathcal{D}) = \frac{1}{N}\sum_n \ell(y_n, f(x_n))$ — plug in empirical distribution.
- **ERM**: $\hat{f}_\text{ERM} = \arg\min_{f\in\mathcal{H}} R(f, \mathcal{D})$.
- **THIS IS WHAT WE ALL DO**: pretraining, fine-tuning, supervised heads — all ERM with cross-entropy. Worth internalizing the formalism even if it feels obvious.

### 5.4.1.1 Approximation vs estimation error

- $f^{**}$ = best function over all; $f^*_\mathcal{H}$ = best in hypothesis class; $f^*_\mathcal{D}$ = ERM solution.
- Decomposition:
$$\mathbb{E}_\mathcal{D}[R(f^*_\mathcal{D}) - R(f^{**})] = \underbrace{R(f^*_\mathcal{H}) - R(f^{**})}_{\mathcal{E}_\text{app}(\mathcal{H})} + \underbrace{\mathbb{E}_\mathcal{D}[R(f^*_\mathcal{D}) - R(f^*_\mathcal{H})]}_{\mathcal{E}_\text{est}(\mathcal{H},N)}$$
- **Approximation error**: how well can class $\mathcal{H}$ represent truth.
- **Estimation error**: cost of finite data, finding best-in-class.
- Bigger $\mathcal{H}$ → lower approx, higher estimation (the classical bias-variance from a different angle).
- **LLM relevance — very high**: scaling laws are essentially: bigger model + more data shifts the optimal point of this tradeoff. Modern deep nets defy classical bias-variance because of implicit regularization from SGD — **estimation error doesn't blow up** as expected (double descent etc.).
- **Generalization gap**: $R(f) - R(f, \mathcal{D}_\text{train}) \approx R(f, \mathcal{D}_\text{test}) - R(f, \mathcal{D}_\text{train})$.
	- The headline number in every eval — train-test gap = the entire generalization question.

### 5.4.2 Regularized risk + structural risk

- **Regularized empirical risk**: $R_\lambda(f, \mathcal{D}) = R(f, \mathcal{D}) + \lambda C(f)$.
	- Log loss + neg log prior = **MAP estimation**.
	- In LLMs: weight decay (Gaussian prior), KL-to-reference (RLHF), L2 penalty on adapter weights, etc.
- **Naive bilevel** $\hat\lambda = \arg\min_\lambda \min_\theta R_\lambda$ fails — picks $\lambda=0$ (optimism of training error).
- **Structural risk minimization** (Vapnik): need an honest estimate of population risk to pick $\lambda$.

### 5.4.3 Cross-validation

- Holdout 80/20, or K-fold, or LOO ($K=N$).
- $R^\text{cv}_\lambda = \frac{1}{K}\sum_k R_0(\hat\theta_\lambda(\mathcal{D}_{-k}), \mathcal{D}_k)$.
- Pick $\hat\lambda = \arg\min R^\text{cv}_\lambda$, refit on full data.
- **LLM relevance**: nobody K-fold-CVs a 70B model. But: held-out val sets, **multi-task averaged eval**, **subset-based hyperparam search at small scale → transfer to large** are all CV-flavored. Also basis of $\mu$P-style hyperparameter prediction.

### 5.4.4 Statistical learning theory / PAC

- Goal: bound generalization error w.h.p.
- **PAC learnable**: probably ($1-\delta$) approximately ($\epsilon$) correct.
- **Theorem 5.4.1** (finite $\mathcal{H}$): $P(\max_h |R(h) - R(h,\mathcal{D})| > \epsilon) \le 2\dim(\mathcal{H}) e^{-2N\epsilon^2}$.
	- Proof = Hoeffding + union bound. Bound is loose but conveys: gap $\downarrow$ with $N$, $\uparrow$ with $\dim(\mathcal{H})$.
- **VC dimension**: effective complexity of infinite-hypothesis classes.
- **Honest take for LLM research**: classical SLT is mostly **vacuous for deep nets** — VC dim bounds give nonsense numbers. Recent practical generalization bounds for DNNs (Jiang+ 2020, PAC-Bayes flat minima, NTK-based) are more relevant but still mostly explain rather than predict. Don't expect to derive scaling laws from VC dim.

## 5.5 Frequentist Hypothesis Testing (starred section — lower priority)

### 5.5.1 Likelihood ratio test

- $\frac{p(\mathcal{D}|H_0)}{p(\mathcal{D}|H_1)}$. Optimal under 0-1 loss + uniform prior.
- **Simple vs compound hypothesis**: parameters fully specified vs need to be optimized/integrated.
- For compound: **maximum likelihood ratio**: $\frac{\max_{\theta \in H_0} p_\theta(\mathcal{D})}{\max_{\theta \in H_1} p_\theta(\mathcal{D})}$.

### 5.5.2 Type I/II errors and Neyman-Pearson lemma

- **Type I** ($\alpha$): reject $H_0$ when true. **Type II** ($\beta$): accept $H_0$ when $H_1$ true.
- **Power** = $1-\beta$ = TPR.
- **Neyman-Pearson lemma**: likelihood ratio is most powerful test at fixed $\alpha$.

### 5.5.3 NHST and p-values

- $\text{pval} = \Pr(\text{test}(\tilde\mathcal{D}) \ge \text{test}(\mathcal{D}) | \tilde\mathcal{D} \sim H_0)$.
- **Wald statistic**: $(\hat\theta - \theta_0)/\hat{se}(\hat\theta)$, asymptotically Normal.
- **Permutation test**: nonparametric, sample random reorderings.

### 5.5.4 p-values considered harmful

- Common fallacy: small p-value $\Rightarrow$ $H_1$ likely. This is **induction**, not deduction, so needs Bayes.
- Toy example: 200 trials, $\alpha=0.05$, $\beta=0.2$, 90% prior of ineffective. Result "significant" → $p(H_0|\text{sig}) = 0.36$ — much more than 5%.
- **My honest take**: p-values are mostly dead in modern ML research. We compare means with confidence intervals or do Bayesian effect-size analysis. Worth understanding for reading older lit + medical/scientific ML applications.

### 5.5.5 Why isn't everyone a Bayesian?

- Frequentist inference violates **likelihood principle** (depends on hypothetical future data).
- Historical: computation was the barrier; today's compute makes Bayesian methods tractable.
- Modeling-assumption sensitivity: Bayesian is only as good as the prior + model, but the same applies to frequentist (sampling distribution = model assumption).
- xkcd 1132 (sun explosion cartoon): the punchline of the chapter.
- **LLM lens**: the field is mostly point-estimate frequentist (one big MAP/MLE-ish model) but **uncertainty quantification** is a major open problem and Bayesian deep learning is making a comeback (e.g., **Laplace approximations for LLM confidence**, **ensemble methods**, **conformal prediction**).

---

## Senior-researcher takeaways (what I actually use vs read once)

- **Use every day**:
	- Cross-entropy / log-loss as the training objective + KL as the regularizer/diagnostic. Section 5.1.6 is the most important section in the chapter.
	- ERM + structural risk + cross-validation (5.4) — this is the deep learning playbook.
	- Generalization gap framing — every benchmark report is implicitly arguing about $R(f) - R(f, \mathcal{D}_\text{train})$.
- **Reach for in specific problems**:
	- Reject option / Chow's rule when working on **selective prediction / abstention / calibration**.
	- Bayes factors / model selection when comparing architectures or doing **paired model evals**.
	- Bayesian t-test / ROPE when reporting whether model A really beats model B.
	- F-scores / PR curves for **retrieval, detection, safety classifiers, reward model eval**.
	- Brier score for **calibration** measurement (less brittle than log-loss).
- **Worth reading once, rarely apply**:
	- Minimax, admissibility, Stein's paradox (conceptual gloss for why regularization works).
	- Neyman-Pearson, NHST, p-values (read code from older papers; don't write new ones).
	- VC dimension / classical PAC (intuition only — bounds are vacuous for modern nets).
- **Underrated bridge concepts for current research**:
	- **WAIC** for singular models (relevant to deep net generalization bounds).
	- **Marginal likelihood ↔ LOO-CV** (justifies val-loss as the metric).
	- **Bayesian Occam's razor** (intuition for scaling laws).
	- **KL = proper scoring rule + asymmetric** (everything in RLHF, distillation, VI).
