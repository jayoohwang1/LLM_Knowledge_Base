# Chapters 4 & 5 — Single-Layer Networks: Regression & Classification

> **Researcher take:** These two chapters are the bedrock you usually absorb by osmosis if you've taken ML courses — but Bishop's treatment is unusually clean and the probabilistic framing pays dividends later. Ch 4 is mostly background; Ch 5 contains two things that are genuinely load-bearing for LLM work: **softmax + cross-entropy** (the LM head) and **decision theory** (the calibration/safety framing). If you already know logistic regression cold, skim Ch 4 for the bias-variance section and spend more time on 5.2 and 5.4.

---

## Chapter 4 — Linear Regression

### 4.1 Linear Regression

#### **Core Setup**
- **Goal:** predict continuous target $t$ from $D$-dim input $\mathbf{x}$ via function $y(\mathbf{x}, \mathbf{w})$ parameterized by $\mathbf{w}$
- **Linear model:** $y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + \ldots + w_D x_D$ — linear in both $\mathbf{w}$ *and* $\mathbf{x}$
- The hard constraint is linearity in $\mathbf{x}$; it's this that makes it too weak for most real problems

#### 4.1.1 Basis Functions — The Pre-Deep Learning Workaround

- **Idea:** extend to $y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j \phi_j(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x})$, where $\phi_j$ are fixed nonlinear **basis functions**
  - Still linear in $\mathbf{w}$ → closed-form MLE; not linear in $\mathbf{x}$ → can fit nonlinear data
  - $\phi_0(\mathbf{x}) = 1$ always (absorbs the bias $w_0$)
- **Examples of basis functions:**
  - **Polynomial:** $\phi_j(x) = x^j$ — global support, horrible extrapolation
  - **Gaussian (RBF):** $\phi_j(x) = \exp\left\{-\frac{(x - \mu_j)^2}{2s^2}\right\}$ — localized in input space; normalization irrelevant since $w_j$ absorbs the scale
  - **Sigmoidal:** $\phi_j(x) = \sigma\!\left(\frac{x - \mu_j}{s}\right)$ — S-shaped, localized transition around $\mu_j$
  - **Fourier:** $e^{i\omega x}$ — global, captures frequency structure; leads to wavelets when you want both space and frequency locality

> **Modern relevance — LOW.** The entire deep learning revolution is about *learning* these basis functions from data instead of hand-crafting them. This is literally the punchline of Ch 6 (data-dependent basis functions via deep networks). Ch 4 is useful as the "what we were stuck with before" picture. One nontrivial survival: **sinusoidal positional encodings in transformers** are a special-case Fourier basis — not for features but for position. And the RBF intuition shows up in kernel methods and Gaussian processes (still used for calibration and small-data settings).

#### 4.1.2 Likelihood Function

- Assume **Gaussian noise model:** $t = y(\mathbf{x}, \mathbf{w}) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$
  - So $p(t | \mathbf{x}, \mathbf{w}, \sigma^2) = \mathcal{N}(t | y(\mathbf{x}, \mathbf{w}), \sigma^2)$
- Joint likelihood over $N$ i.i.d. points: $p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \sigma^2) = \prod_{n=1}^N \mathcal{N}(t_n | \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_n), \sigma^2)$
- Log-likelihood: $\ln p(\mathbf{t}|\mathbf{X}, \mathbf{w}, \sigma^2) = -\frac{N}{2}\ln\sigma^2 - \frac{N}{2}\ln(2\pi) - \frac{1}{\sigma^2} E_D(\mathbf{w})$
  - where $E_D(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \{t_n - \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_n)\}^2$ is the **sum-of-squares error**
  - **Key insight:** maximizing log-likelihood under Gaussian noise ≡ minimizing MSE. This is why MSE is the "right" loss under Gaussian noise assumptions — it's not arbitrary, it's MLE.

#### 4.1.3 Maximum Likelihood — Normal Equations

- Setting $\nabla_\mathbf{w} \ln p = 0$ gives the **normal equations:**
  $$\mathbf{w}_\text{ML} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{t} = \boldsymbol{\Phi}^\dagger \mathbf{t}$$
  - $\boldsymbol{\Phi}$ is the $N \times M$ **design matrix** with $\Phi_{nj} = \phi_j(\mathbf{x}_n)$
  - $\boldsymbol{\Phi}^\dagger \equiv (\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top$ is the **Moore-Penrose pseudoinverse**
- MLE for $\sigma^2$: $\sigma^2_\text{ML} = \frac{1}{N}\sum_{n=1}^N \{t_n - \mathbf{w}^\top_\text{ML}\boldsymbol{\phi}(\mathbf{x}_n)\}^2$ — residual variance

> **Modern relevance — LOW for inference, HIGH for intuition.** You never solve the normal equations for anything interesting (scales as $O(M^3)$ for matrix inversion). But the pseudoinverse framing resurfaces when you think about **linear probing** (fitting a linear classifier on frozen embeddings) or **activation patching** (causal mediation analysis in mech interp). The design matrix intuition also helps when thinking about in-context learning — the model is essentially doing something like a fast approximate gradient descent in the forward pass.

#### 4.1.4 Geometry of Least Squares

- View $\mathbf{t} \in \mathbb{R}^N$ and each basis function $\phi_j(\mathbf{x}_n)$ as a vector $\boldsymbol{\varphi}_j \in \mathbb{R}^N$
- $\mathbf{y} = \boldsymbol{\Phi}\mathbf{w}$ lives in the $M$-dimensional subspace $\mathcal{S}$ spanned by the $\boldsymbol{\varphi}_j$
- **Least-squares solution** = orthogonal projection of $\mathbf{t}$ onto $\mathcal{S}$
  - This gives the geometric intuition: we're finding the closest point in the model's reachable subspace to the target
- Near-collinearity among basis vectors → numerical instability; SVD is the robust solution

#### 4.1.5 Sequential Learning

- Full-batch computation of $\boldsymbol{\Phi}^\dagger$ requires all data upfront; infeasible at scale
- **LMS (least-mean-squares) / online gradient descent:**
  $$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \eta(t_n - (\mathbf{w}^{(\tau)})^\top \boldsymbol{\phi}_n)\boldsymbol{\phi}_n$$
- This is the origin of SGD — one data point at a time, update rule is literally gradient of the single-point squared error

#### 4.1.6 Regularized Least Squares — Weight Decay

- Total loss: $E_D(\mathbf{w}) + \lambda E_W(\mathbf{w})$ where $E_W = \frac{1}{2}\mathbf{w}^\top\mathbf{w}$ (L2 / **weight decay**)
- Closed form: $\mathbf{w} = (\lambda\mathbf{I} + \boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\mathbf{t}$
  - $\lambda\mathbf{I}$ makes the matrix invertible even when $\boldsymbol{\Phi}^\top\boldsymbol{\Phi}$ is singular — ridge regression as numerical fix
  - **Bayesian interpretation:** L2 regularization = Gaussian prior $p(\mathbf{w}) \propto \exp(-\frac{\lambda}{2}\mathbf{w}^\top\mathbf{w})$; MAP estimate under this prior = regularized MLE

> **Modern relevance — HIGH.** Weight decay is used in virtually every LLM training run (via AdamW). The subtle point Bishop glosses over: in Adam, L2 regularization ≠ decoupled weight decay because the gradient scaling in Adam changes the effective regularization magnitude per parameter. **AdamW** (Loshchilov & Hutter 2019) fixes this by decoupling the weight decay from the gradient update. This distinction matters in practice — L2 regularization in Adam is not the same as weight decay.

#### 4.1.7 Multiple Outputs

- For $K$ outputs: $\mathbf{y}(\mathbf{x}, \mathbf{W}) = \mathbf{W}^\top \boldsymbol{\phi}(\mathbf{x})$ with $\mathbf{W}$ an $M \times K$ parameter matrix
- Under isotropic Gaussian noise, the problem **decouples** — each output has an independent regression problem with the same pseudoinverse $\boldsymbol{\Phi}^\dagger$
- This decoupling is a special property of the Gaussian + isotropic assumption; general covariance noise still decouples (proof in Ex 4.7) because MLE for the mean of a multivariate Gaussian is independent of the covariance

---

### 4.2 Decision Theory

- **Two stages of prediction:**
  1. **Inference:** determine $p(t|\mathbf{x})$ from training data
  2. **Decision:** use $p(t|\mathbf{x})$ to choose a specific action $f(\mathbf{x})$ that minimizes **expected loss**
- **Expected loss:** $\mathbb{E}[L] = \iint L(t, f(\mathbf{x})) p(\mathbf{x}, t)\, d\mathbf{x}\, dt$
- **Squared loss** $L(t, f(\mathbf{x})) = \{f(\mathbf{x}) - t\}^2$: optimal decision is the **conditional mean**
  $$f^*(\mathbf{x}) = \mathbb{E}_t[t|\mathbf{x}] = \int t\, p(t|\mathbf{x})\, dt$$
  - This is the **regression function** — the theoretically optimal predictor under squared loss
- **Minkowski loss** $L_q = |f(\mathbf{x}) - t|^q$:
  - $q = 2$: conditional mean (squared loss)
  - $q = 1$: conditional median (L1 loss)
  - $q \to 0$: conditional mode

> **Modern relevance — MEDIUM but conceptually important.** The inference/decision split is how you should think about LLMs for deployment. The model outputs $p(\text{token}|\text{context})$; temperature, top-p, etc. are the **decision rule** applied to that distribution. Choosing the right loss during *training* also matters: MSE vs. cross-entropy for language modeling, and the entire RLHF machinery is about designing a loss that matches what you actually want to optimize (human preference ≈ a complex loss matrix). The result that **the optimal squared-loss predictor is $\mathbb{E}[t|\mathbf{x}]$** is also the theoretical justification for why language models trained with cross-entropy (a proper scoring rule) should produce well-calibrated probabilities.

---

### 4.3 Bias–Variance Trade-off ⭐ (Important conceptually, nuanced for modern scale)

- **Setup:** imagine training on many different datasets $\mathcal{D}$ of size $N$; how does expected loss decompose?
- **Decomposition:**
  $$\text{expected loss} = (\text{bias})^2 + \text{variance} + \text{noise}$$
  - $(\text{bias})^2 = \int \{\mathbb{E}_\mathcal{D}[f(\mathbf{x};\mathcal{D})] - h(\mathbf{x})\}^2 p(\mathbf{x})\, d\mathbf{x}$
    - how far is the **average prediction** from the true regression function?
  - $\text{variance} = \int \mathbb{E}_\mathcal{D}[\{f(\mathbf{x};\mathcal{D}) - \mathbb{E}_\mathcal{D}[f(\mathbf{x};\mathcal{D})]\}^2]\, p(\mathbf{x})\, d\mathbf{x}$
    - how much does the prediction vary across different datasets?
  - noise = irreducible error from $\text{var}[t|\mathbf{x}]$
- **Classical intuition:**
  - **High regularization** ($\lambda$ large): stiff model → high bias, low variance (underfits)
  - **Low regularization** ($\lambda$ small): flexible model → low bias, high variance (overfits)
  - Optimal $\lambda$ balances both; the bias-variance sum is minimized somewhere in between
- **Critical caveat:** this analysis is frequentist and assumes you're averaging over datasets. In practice you have one dataset. If you had more data, you'd just combine it and use a more complex model → the trade-off is a data-scarcity artifact.
- **Bayesian perspective:** marginalize over parameters (rather than data sets) → bias-variance doesn't arise. The Bayesian average prediction is just the posterior predictive mean.

> **Modern relevance — IMPORTANT to understand where it breaks down.** The classical bias-variance trade-off says: use a model that's "just complex enough" — not too simple (high bias), not too complex (high variance). This **breaks down completely** at LLM scale. We train models with far more parameters than data points and they generalize better than smaller models. This is the **double descent** phenomenon (Ch 9.3.2): there are actually *two* descent curves in test error as model size grows — a classical U-shaped curve, then another improvement once the model is massively overparameterized. Modern LLMs are firmly in the second regime where classical bias-variance intuitions are backward. The practical implication: don't early-stop to the classical bias-variance optimum in the overparameterized setting; you may want to train longer and let the model interpolate.

---

## Chapter 5 — Single-Layer Networks: Classification

> **Researcher take:** Chapters 4 and 5 come as a pair but 5 punches much higher for LLM relevance. The key things to own from this chapter: (1) **softmax + cross-entropy** and the fact that the gradient simplifies beautifully (Sec 5.4.4), (2) the **three-way distinction** between discriminant functions, discriminative models, and generative models (Sec 5.2.4) — this maps directly onto the GPT/BERT/decoder-only landscape, and (3) the **decision theory framing** for thinking about calibration and safety (Sec 5.2).

---

### 5.1 Discriminant Functions

#### **Setup**
- Three broad approaches to classification:
  1. **Discriminant function:** directly maps $\mathbf{x} \to$ class label (bypasses probabilities)
  2. **Discriminative probabilistic:** models $p(\mathcal{C}_k|\mathbf{x})$ directly
  3. **Generative probabilistic:** models $p(\mathbf{x}|\mathcal{C}_k)$ and $p(\mathcal{C}_k)$, uses Bayes to get posteriors
- Focus of 5.1: linear discriminants — decision surfaces are hyperplanes

#### 5.1.1 Two Classes

- $y(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + w_0$: assign to $\mathcal{C}_1$ if $y(\mathbf{x}) \geq 0$, else $\mathcal{C}_2$
- **Geometry:**
  - $\mathbf{w}$ is the **normal to the decision hyperplane** (perpendicular to the surface)
  - $w_0$ controls the **offset** of the hyperplane from the origin: surface passes through origin iff $w_0 = 0$
  - Signed distance of any point $\mathbf{x}$ from decision surface: $r = y(\mathbf{x}) / \|\mathbf{w}\|$
- Compact form: augment with $x_0 = 1$ so $y(\mathbf{x}) = \tilde{\mathbf{w}}^\top\tilde{\mathbf{x}}$

#### 5.1.2 Multiple Classes ($K > 2$)

- **One-vs-rest:** $K-1$ classifiers, each distinguishes $\mathcal{C}_k$ from not-$\mathcal{C}_k$ → **ambiguous regions** in input space (a point can be "not $\mathcal{C}_1$" and "not $\mathcal{C}_2$" simultaneously)
- **One-vs-one:** $K(K-1)/2$ classifiers → also has ambiguous regions
- **Correct approach:** single $K$-class discriminant with $K$ linear functions $y_k(\mathbf{x}) = \mathbf{w}_k^\top\mathbf{x} + w_{k0}$
  - Assign to $\mathcal{C}_k$ if $y_k > y_j$ for all $j \neq k$
  - Decision regions are **singly connected and convex** (key geometric property: any line between two points in region $\mathcal{R}_k$ stays in $\mathcal{R}_k$)
  - Decision boundary between $\mathcal{C}_k$ and $\mathcal{C}_j$: $(\mathbf{w}_k - \mathbf{w}_j)^\top\mathbf{x} + (w_{k0} - w_{j0}) = 0$ — a hyperplane

#### 5.1.3 1-of-K Coding (One-Hot Encoding)

- For $K$ classes, target $\mathbf{t}$ is a $K$-vector with $t_j = 1$ if class is $\mathcal{C}_j$, else 0
- Conditional expectation of $\mathbf{t}$ given $\mathbf{x}$ = posterior class probabilities: $\mathbb{E}[\mathbf{t}|\mathbf{x}] = (p(\mathcal{C}_1|\mathbf{x}), \ldots, p(\mathcal{C}_K|\mathbf{x}))^\top$

> **Modern relevance — EXTREMELY HIGH.** This is exactly the token representation in language models. Vocabulary of size $V$ → 1-of-$V$ one-hot targets. The model outputs a distribution over the vocabulary at each position. Cross-entropy loss is computed against the one-hot target. This is the fundamental correspondence: language modeling = sequential multiclass classification with a vocabulary-sized 1-of-K scheme at every position.

#### 5.1.4 Least Squares for Classification — Why It Fails

- Applying the regression least-squares formalism to classification with 1-of-K targets:
  - **Constraint satisfaction:** $\sum_k y_k(\mathbf{x}) = 1$ preserved by the LS solution (follows from constraint $\mathbf{a}^\top\mathbf{t}_n + b = 0$)
  - **But outputs can lie outside $[0,1]$** → can't interpret as probabilities
  - **Highly sensitive to outliers:** squared loss penalizes correctly-classified far-away points heavily, distorting the decision boundary
- Why LS fails: squared loss ≡ MLE under Gaussian noise; binary targets are Bernoulli, not Gaussian → wrong noise model → bad behavior

> **Modern relevance — MEDIUM (mostly as a cautionary tale).** Nobody uses least-squares for classification. But the *reason* it fails (wrong noise model → wrong loss function) is a principle that carries forward: if your output distribution doesn't match your loss function's implicit assumption, you'll get pathological gradients. This comes up in modern settings: e.g., using MSE for token-level generation vs. cross-entropy, or reward model choices in RLHF. The "wrong loss" problem is alive and well.

---

### 5.2 Decision Theory ⭐⭐ (More important than it looks for deployment)

#### 5.2.1 Misclassification Rate

- To minimize probability of mistake: assign $\mathbf{x}$ to class with **highest posterior probability** $p(\mathcal{C}_k|\mathbf{x})$
- Optimal decision rule = MAP classification (given correct posteriors)

#### 5.2.2 Expected Loss — Loss Matrices

- **Loss matrix** $L_{kj}$: loss incurred when true class is $\mathcal{C}_k$ but we predict $\mathcal{C}_j$
  - Example: $L_{\text{cancer, normal}} = 100$, $L_{\text{normal, cancer}} = 1$ — asymmetric costs
- **Expected loss:** $\mathbb{E}[L] = \sum_k\sum_j \int_{\mathcal{R}_j} L_{kj}\, p(\mathbf{x}, \mathcal{C}_k)\, d\mathbf{x}$
- **Optimal rule:** assign $\mathbf{x}$ to class $j$ that minimizes $\sum_k L_{kj} p(\mathcal{C}_k|\mathbf{x})$
  - Crucially: to apply a *different* loss matrix, you just plug in new $L_{kj}$ — you don't retrain, just change the decision function at inference time (requires knowing posterior probabilities; impossible with a pure discriminant function)

> **Modern relevance — HIGH for safety/alignment.** This is the formal framework behind **reward modeling** and **RLHF**. The loss matrix captures what types of errors are more costly. In LLM deployment: refusing a harmful request costs 1 (mild inconvenience), generating harmful content costs 100 (serious harm). RLHF can be understood as training a model whose decisions approximately minimize expected loss under a human-specified loss function. The **reject option** (next section) maps directly to the **abstention/refusal** problem in safe deployment.

#### 5.2.3 The Reject Option

- **Idea:** for inputs where the posterior probability $\max_k p(\mathcal{C}_k|\mathbf{x})$ is below a threshold $\theta$, don't classify — send to human/higher-authority
- At $\theta = 1$: reject everything; $\theta < 1/K$: reject nothing
- With loss matrix: reject incurs cost $\lambda$; reject when this is cheaper than expected classification loss

> **Modern relevance — HIGH.** This is the theoretical backbone of **abstention** in LLM safety. When a model is uncertain (low confidence in any answer, or the request is ambiguous/harmful), the optimal decision may be to abstain/refuse rather than guess. Calibration work: if model probabilities are well-calibrated, you can set $\theta$ to achieve a target false-positive rate on harmful content detection. Also connects to speculative decoding: a fast draft model "rejects" tokens when the main model would disagree — similar threshold-based mechanism.

#### 5.2.4 Inference and Decision — The Three Approaches ⭐

The three levels of increasing complexity, in decreasing order:

**(a) Generative models**
- Model $p(\mathbf{x}|\mathcal{C}_k)$ (class-conditional density) + $p(\mathcal{C}_k)$ (class prior)
- Use Bayes: $p(\mathcal{C}_k|\mathbf{x}) = \frac{p(\mathbf{x}|\mathcal{C}_k)p(\mathcal{C}_k)}{p(\mathbf{x})}$
- Most parameters, requires modeling the full input distribution
- Advantage: know $p(\mathbf{x})$ → can detect OOD samples where predictions are unreliable
- Can generate synthetic data from $p(\mathbf{x}|\mathcal{C}_k)$

**(b) Discriminative probabilistic models**
- Model $p(\mathcal{C}_k|\mathbf{x})$ directly (don't model the input density)
- Fewer parameters, avoids wasting capacity modeling irrelevant variation in $p(\mathbf{x}|\mathcal{C}_k)$
- Better empirical classification performance in many settings

**(c) Discriminant functions**
- Map $\mathbf{x}$ directly to a class label, bypassing probabilities entirely
- Most constrained; can't adjust decisions when loss matrix changes

**Why you want posterior probabilities:**
- **Minimize risk:** can swap loss matrices at inference without retraining
- **Reject option:** requires calibrated probabilities to implement a principled threshold
- **Compensate for class imbalance:** if training distribution has different class frequencies than deployment, you can correct posteriors using Bayes — multiply by deployment priors, divide by training priors, renormalize. This requires probabilities; impossible with a pure discriminant function.
- **Combining models:** if you have separate models for different input modalities (image + blood test), you can combine them using $p(\mathcal{C}_k|\mathbf{x}_I, \mathbf{x}_B) \propto \frac{p(\mathcal{C}_k|\mathbf{x}_I)p(\mathcal{C}_k|\mathbf{x}_B)}{p(\mathcal{C}_k)}$ (naive Bayes conditional independence assumption). Requires posteriors from each model.

> **Modern relevance — EXTREMELY HIGH conceptually.** This three-way distinction maps almost perfectly onto the modern LLM landscape:
> - **(a) Generative:** GPT-style decoder-only autoregressive LLMs — they model $p(\mathbf{x})$ and can generate samples. They get posterior class probabilities "for free" via Bayes, but the hard part is that $p(\mathbf{x})$ is over the entire input space (exponentially complex).
> - **(b) Discriminative:** BERT-style masked LMs — directly model $p(\text{label}|\mathbf{x})$ for downstream tasks. More parameter-efficient for classification tasks, but can't generate freely.
> - **(c) Discriminant:** fine-tuned classifiers trained with hard labels and argmax output — no probabilities, can't adjust for different deployment contexts.
> 
> The argument for why discriminative beats generative is a classic ML result (Ng & Jordan 2002): discriminative models need fewer samples to fit well because they don't waste capacity modeling the input distribution. But generative models (LLMs) win in the infinite-data regime because the $p(\mathbf{x})$ modeling forces better representations. This is part of why pretraining at massive scale and then fine-tuning (which is discriminative) works so well.

#### 5.2.5 Classifier Accuracy — Evaluation Metrics

- **Confusion matrix:** $N_{TP}, N_{FP}, N_{TN}, N_{FN}$ (for two-class problems)
- **Accuracy:** $(N_{TP} + N_{TN}) / N$ — useless for imbalanced classes
- **Precision** $= N_{TP} / (N_{TP} + N_{FP})$: of those predicted positive, how many are?
- **Recall** $= N_{TP} / (N_{TP} + N_{FN})$: of all true positives, how many did we catch?
- **False positive rate** $= N_{FP} / (N_{FP} + N_{TN})$: of all true negatives, how many did we wrongly predict positive?
- **F-score** $= 2 \times \text{precision} \times \text{recall} / (\text{precision} + \text{recall})$: harmonic mean of precision and recall

> **Modern relevance — HIGH for applied/safety work.** LLM evaluation constantly runs into the class imbalance problem: harmful content is rare, refusals are common. Accuracy is meaningless; F1 or AUC is more informative. The precision/recall trade-off (vary classification threshold, trace out the PR curve) is exactly what safety teams do to set refusal thresholds — where on the curve to operate is a policy decision about how many false positives (over-refusals) you're willing to accept in exchange for fewer false negatives (harmful outputs).

#### 5.2.6 ROC Curve

- **ROC curve:** true positive rate vs. false positive rate as threshold varies from $-\infty$ to $\infty$
- Perfect classifier = top-left corner; random classifier = diagonal
- **AUC (area under the curve):** $\text{AUC} = 0.5$ (random) to $1.0$ (perfect)
- AUC is threshold-independent: characterizes the classifier's discriminative ability across all operating points

> **Modern relevance — HIGH for reward modeling and safety evaluation.** AUC is the standard metric for binary classification problems like "is this output harmful" or "is this a correct answer". Reward model training is exactly a binary classification problem (chosen vs. rejected response) and AUC is how you validate the reward model before using it for PPO. PR-AUC is often more informative than ROC-AUC for heavily imbalanced safety settings.

---

### 5.3 Generative Classifiers

#### 5.3.1 Continuous Inputs — Gaussian Class-Conditional Densities

- Assume $p(\mathbf{x}|\mathcal{C}_k) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$ with a **shared covariance** $\boldsymbol{\Sigma}$
- For two classes, posterior:
  $$p(\mathcal{C}_1|\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + w_0)$$
  where $\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ and $w_0 = -\frac{1}{2}\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2 + \ln\frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)}$
- **Key mechanism:** the quadratic terms in $\mathbf{x}$ from the Gaussian exponents cancel when covariance is shared → **linear decision boundary** in $\mathbf{x}$
- For $K$ classes: $a_k(\mathbf{x}) = \mathbf{w}_k^\top\mathbf{x} + w_{k0}$ with $\mathbf{w}_k = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k$ → this is **Linear Discriminant Analysis (LDA)**
- Separate covariances $\boldsymbol{\Sigma}_k$: quadratic terms don't cancel → **Quadratic Discriminant Analysis (QDA)**, curved decision boundaries

> **Modern relevance — LOW for direct use, HIGH for conceptual insight.** Nobody uses LDA/QDA for LLM tasks. But the key insight — that linear decision boundaries arise naturally when you combine Gaussians with the same covariance via Bayes — reappears in **linear probing** theory. The theoretical argument for why linear probes on LLM embeddings work is that the embeddings approximately cluster as Gaussians with shared covariance per class (isotropic within class), making the classification problem "linearly separable" in representation space. Also: the logit $a(\mathbf{x})$ from a generative classifier with Gaussian assumptions is a **Mahalanobis distance** — exactly the structure used in some OOD detection methods.

#### 5.3.2 Maximum Likelihood Solution

- For shared-covariance Gaussians, MLE gives:
  - **Class prior:** $\hat{\pi} = N_k / N$ (fraction of training data in each class)
  - **Class means:** $\hat{\boldsymbol{\mu}}_k = \frac{1}{N_k}\sum_{n \in \mathcal{C}_k} \mathbf{x}_n$ (within-class means)
  - **Shared covariance:** $\hat{\boldsymbol{\Sigma}} = \frac{N_1}{N}\mathbf{S}_1 + \frac{N_2}{N}\mathbf{S}_2$ (weighted average of within-class scatter matrices)
- Limitation: Gaussian MLE for covariance is not robust to outliers — a single outlier can massively perturb $\hat{\boldsymbol{\Sigma}}$

#### 5.3.3 Discrete Features — Naive Bayes

- Binary features $x_i \in \{0, 1\}$: if full joint distribution over $D$ features, need $2^D - 1$ parameters per class — exponentially bad
- **Naive Bayes assumption:** features are conditionally independent given class:
  $$p(\mathbf{x}|\mathcal{C}_k) = \prod_{i=1}^D \mu_{ki}^{x_i}(1 - \mu_{ki})^{1-x_i}$$
- Reduces to $D$ parameters per class — tractable
- The class-conditional log-likelihood $a_k(\mathbf{x})$ is **linear in the features** $x_i$
- Despite obviously wrong independence assumption, naive Bayes works remarkably well in practice

> **Modern relevance — MEDIUM.** Naive Bayes itself is rarely used in LLM-era pipelines. But the conditional independence factorization is the foundational assumption behind:
> - **Autoregressive modeling:** $p(x_1, \ldots, x_T) = \prod_t p(x_t | x_{<t})$ — the chain rule factorization is the sequence-level analog, without the independence assumption (each step conditions on all history).
> - **Modality fusion:** when combining different input streams (text + image + audio), conditional independence per modality given the latent is a common modeling choice.
> - **Bayesian classifier combination** (Sec 5.2.4): combining models for different input features requires a conditional independence assumption between the feature types.

#### 5.3.4 Exponential Family — Generality of Logistic/Softmax

- For class-conditional densities from the **exponential family** (Gaussian, Bernoulli, Poisson, etc.):
  $$p(\mathbf{x}|\boldsymbol{\lambda}_k, s) = \frac{1}{s}h\!\left(\frac{1}{s}\mathbf{x}\right)g(\boldsymbol{\lambda}_k)\exp\!\left\{\frac{1}{s}\boldsymbol{\lambda}_k^\top\mathbf{x}\right\}$$
  - shared scale parameter $s$ across classes
- The resulting posterior for two-class problem: $p(\mathcal{C}_1|\mathbf{x}) = \sigma(a(\mathbf{x}))$ with $a(\mathbf{x})$ **linear in** $\mathbf{x}$
- For $K$-class: $p(\mathcal{C}_k|\mathbf{x}) \propto \exp(a_k(\mathbf{x}))$ with $a_k$ linear in $\mathbf{x}$ → **softmax**
- This is a remarkable generality result: for *any* exponential family class-conditional density (with shared scale), the optimal Bayesian classifier has a logistic/softmax form — logistic regression is not ad hoc, it emerges from a generative model

> **Modern relevance — HIGH conceptually.** This result is the theoretical justification for why softmax is the "right" output layer for classification. It's not just a convenient normalization — it's the optimal posterior transformation when the classes have exponential family distributions. This connects deep learning outputs to probabilistic models. It also connects to the **Boltzmann distribution** in statistical physics: softmax is the Gibbs distribution over discrete states, with energies given by the logits.

---

### 5.4 Discriminative Classifiers ⭐⭐ (Core LLM machinery)

#### 5.4.1 Activation Functions

- For classification: transform linear output $\mathbf{w}^\top\mathbf{x} + w_0$ via nonlinear $f(\cdot)$ to get probabilities in $(0, 1)$:
  $$y(\mathbf{x}, \mathbf{w}) = f(\mathbf{w}^\top\mathbf{w} + w_0)$$
- $f$ is the **activation function** (ML terminology); $f^{-1}$ is the **link function** (statistics terminology)
- Decision boundaries where $\mathbf{w}^\top\mathbf{x} = \text{const}$ → **always linear in** $\mathbf{x}$, regardless of $f$
- Class of models: **generalized linear models (GLMs)**
- Nonlinearity in $f$ makes them nonlinear in $\mathbf{w}$ → no closed-form MLE; need gradient descent

#### 5.4.2 Fixed Basis Functions

- Apply the linear discriminant to feature vectors $\boldsymbol{\phi}(\mathbf{x})$ instead of raw $\mathbf{x}$
- Decision boundary is linear in $\boldsymbol{\phi}$-space → can be nonlinear in $\mathbf{x}$-space
- Can't remove class overlap (classes that overlap in $\mathbf{x}$ may overlap in $\boldsymbol{\phi}$); can only increase it, not remove it
- Fixed basis functions still have fundamental limitations → deep learning learns adaptive ones

#### 5.4.3 Logistic Regression ⭐⭐ (Essential)

- **Two-class model:** $p(\mathcal{C}_1|\boldsymbol{\phi}) = y = \sigma(\mathbf{w}^\top\boldsymbol{\phi})$
  - $M$ parameters in $\mathbf{w}$ — linear in $M$, vs. $O(M^2)$ for the Gaussian generative model
- **Cross-entropy loss** (negative log-likelihood):
  $$E(\mathbf{w}) = -\ln p(\mathbf{t}|\mathbf{w}) = -\sum_{n=1}^N \{t_n \ln y_n + (1-t_n)\ln(1-y_n)\}$$
- **Gradient:** use $\frac{d\sigma}{da} = \sigma(1-\sigma)$, derivative of sigmoid cancels in gradient:
  $$\nabla E(\mathbf{w}) = \sum_{n=1}^N (y_n - t_n)\boldsymbol{\phi}_n$$
  - **Beautifully simple form:** gradient is the sum of errors $(y_n - t_n)$ weighted by features $\boldsymbol{\phi}_n$
  - Same structural form as gradient of sum-of-squares error in linear regression (4.12) — not a coincidence, explained by canonical link functions (Sec 5.4.6)
- **No closed form:** logistic regression error function is convex (no local minima!) but nonlinear → gradient descent or IRLS (iteratively reweighted least squares)
- **Collapse to Heaviside under separability:** if data is linearly separable, MLE pushes $\|\mathbf{w}\| \to \infty$ — the sigmoid becomes a step function and posterior probabilities are 0 or 1 on training data. Continuum of solutions (any separating hyperplane gives same posteriors). **Regularization is necessary** to select a unique solution.

> **Modern relevance — CRITICAL.** The LM head of every language model is logistic regression: a linear layer followed by softmax, optimized with cross-entropy loss against one-hot targets. The gradient form $\nabla E = \sum_n (y_n - t_n)\boldsymbol{\phi}_n$ — where $\boldsymbol{\phi}_n$ is the representation, $y_n$ is the model's probability, $t_n$ is the target — is exactly the gradient flowing back through the output layer of every LLM. The collapse-to-Heaviside under separability problem also reappears in **reward model training**: if you can perfectly separate preferred vs. rejected responses, the reward model diverges. This is why reward model training needs regularization and early stopping.

#### 5.4.4 Multi-class Logistic Regression (Softmax) ⭐⭐ (THE most important section in this chapter)

- **Softmax function** (normalized exponential):
  $$p(\mathcal{C}_k|\boldsymbol{\phi}) = y_k(\boldsymbol{\phi}) = \frac{\exp(a_k)}{\sum_j \exp(a_j)}, \quad a_k = \mathbf{w}_k^\top\boldsymbol{\phi}$$
  - Properties: outputs sum to 1, all in $(0,1)$, monotone in $a_k$
  - If $a_k \gg a_j$ for all $j \neq k$: $y_k \approx 1$ → softmax approximates argmax
  - Temperature scaling: replace $a_k$ with $a_k / T$ → $T \to 0$ gives hard argmax, $T \to \infty$ gives uniform
- **Softmax gradient:** $\frac{\partial y_k}{\partial a_j} = y_k(I_{kj} - y_j)$
- **Cross-entropy loss for $K$ classes:**
  $$E(\mathbf{w}_1, \ldots, \mathbf{w}_K) = -\sum_{n=1}^N\sum_{k=1}^K t_{nk}\ln y_{nk}$$
  - where $\mathbf{t}_n$ is one-hot: $t_{nk} = 1$ iff $n \in \mathcal{C}_k$
- **Gradient w.r.t. $\mathbf{w}_j$:**
  $$\nabla_{\mathbf{w}_j} E = \sum_{n=1}^N (y_{nj} - t_{nj})\boldsymbol{\phi}_n$$
  - Again: error $(y_{nj} - t_{nj})$ times the input feature $\boldsymbol{\phi}_n$. Identical structure to binary case and to linear regression gradient.

> **Modern relevance — ABSOLUTELY CRITICAL.** This is the training objective for language modeling. The softmax + cross-entropy combination with gradient $(y - t)\phi$ is everywhere:
> - **Next-token prediction:** $K = |\mathcal{V}|$ (vocabulary size), $\boldsymbol{\phi}_n$ = hidden state at position $n$, $t_{nk}$ = next token one-hot. The entire pretraining loss is this.
> - **Temperature in generation:** sampling at temperature $T$ = dividing logits $a_k$ by $T$ before softmax. Top-k and top-p are decision rules applied to the resulting probability distribution (Sec 5.2 territory).
> - **Logit lens:** the "logit lens" interpretability technique (nostalgebraist 2020) directly interprets intermediate hidden states as logits $a_k = \mathbf{w}_k^\top h$, asking "what would the model predict if we applied the output head to this intermediate representation?"
> - **Calibration:** if the model is well-calibrated, $y_k = p(\mathcal{C}_k|\boldsymbol{\phi})$ should match empirical class frequencies. LLMs are often overconfident (need temperature scaling to calibrate).
> - **Numerical stability:** never compute softmax naively — subtract $\max_j a_j$ first: $\exp(a_k - \max_j a_j) / \sum_j \exp(a_j - \max_j a_j)$. Same math, no overflow.

#### 5.4.5 Probit Regression

- Alternative activation: cumulative Gaussian (probit function) $\Phi(a) = \int_{-\infty}^a \mathcal{N}(\theta|0,1)\, d\theta$
- Arises from a **noisy threshold model**: assign class based on whether $a_n = \mathbf{w}^\top\boldsymbol{\phi}_n + \epsilon$ exceeds 0, where $\epsilon \sim \mathcal{N}(0,1)$
- Very similar shape to logistic sigmoid in the middle but **different tails:**
  - Logistic sigmoid: tails decay as $\exp(-|x|)$ — lighter than Gaussian
  - Probit: tails decay as $\exp(-x^2)$ — more Gaussian tails
  - Consequence: logistic is **more robust to outliers** (assigns more probability mass to extreme inputs); probit is more sensitive
- In practice: logistic and probit give very similar results except at the tails

> **Modern relevance — LOW.** You'll essentially never see probit regression in LLM work. The logistic sigmoid + softmax has won. The one place probit-like thinking shows up: **signal detection theory** in psychophysics/cognitive science, which occasionally influences how we think about model "perception" thresholds in multimodal systems. Skip this section unless you specifically need it.

#### 5.4.6 Canonical Link Functions ⭐ (Elegant but often overlooked)

- **Key observation:** for regression (Gaussian + identity link), binary classification (Bernoulli + logistic), and multiclass (categorical + softmax), the gradient all has the same form:
  $$\nabla E(\mathbf{w}) = \frac{1}{s}\sum_{n=1}^N (y_n - t_n)\boldsymbol{\phi}_n$$
- This is **not a coincidence** — it's a theorem about exponential family distributions
- When the target distribution is from the exponential family and you choose the **canonical link function** (the specific activation function that pairs naturally with that distribution), the gradient simplifies to this form
- **Canonical pairings:**
  - Gaussian $p(t|\eta, s)$ + identity link → squared loss, gradient = $(y-t)\boldsymbol{\phi}$
  - Bernoulli $p(t|\eta)$ + logistic sigmoid → cross-entropy, gradient = $(y-t)\boldsymbol{\phi}$
  - Categorical $p(t|\eta)$ + softmax → cross-entropy, gradient = $(y-t)\boldsymbol{\phi}$
- The simplification $f'(\psi(y))\psi'(y) = 1$ (when you choose $f^{-1}(y) = \psi(y)$) cancels the derivative of the activation function in the gradient — that's why the sigmoid's derivative disappears in the logistic regression gradient

> **Modern relevance — MEDIUM but conceptually valuable.** Understanding canonical links explains why:
> 1. Cross-entropy is the "right" loss for classification (not MSE), and squared loss is "right" for Gaussian regression — these aren't arbitrary, they're canonically paired
> 2. The gradient always has the clean $(y-t)\phi$ form regardless of activation type — this is why backpropagation through the output layer is always the same structural computation
> 3. When you use a non-canonical combination (e.g., MSE for token generation instead of cross-entropy), you get the sigmoid derivative factor in the gradient, which can create optimization difficulties near saturated units
>
> This also unifies the understanding of training dynamics: in all well-designed output layers, the gradient scales with the magnitude of the error $(y-t)$, which is desirable (larger errors → larger updates).

---

## Cross-Chapter Synthesis for LLM Research

### What Actually Matters

| Topic | Location | LLM Relevance | Why |
|---|---|---|---|
| Softmax + cross-entropy | 5.4.4 | **Critical** | The LM training objective |
| Gradient $(y-t)\phi$ | 5.4.3/4 | **Critical** | LM head gradient; canonical link intuition |
| Inference vs. decision split | 5.2.4 | **Critical** | Generative/discriminative model landscape |
| Expected loss / loss matrix | 5.2.2 | **High** | RLHF; reward modeling; safety trade-offs |
| Reject option | 5.2.3 | **High** | Model abstention; refusal policies |
| Weight decay / L2 regularization | 4.1.6 | **High** | AdamW; understanding vs. L2 in Adam |
| Bias-variance trade-off | 4.3 | **High (conceptual)** | Know where it breaks (double descent) |
| Precision/recall/AUC | 5.2.5-6 | **High** | Safety evaluation; reward model validation |
| Separability → collapse | 5.4.3 | **Medium** | Reward model regularization; overconfidence |
| Naive Bayes / cond. independence | 5.3.3 | **Medium** | Autoregressive factorization; modality fusion |
| Exponential family → softmax | 5.3.4 | **Medium** | Justification for softmax; Boltzmann connection |
| Canonical link functions | 5.4.6 | **Medium** | Why cross-entropy not MSE; gradient structure |
| Decision theory (conditional mean) | 4.2 | **Medium** | Calibration; sampling as decision making |
| Basis functions | 4.1.1 | **Low** | Historical context; pos. encoding niche |
| Normal equations / pseudoinverse | 4.1.3 | **Low** | Linear probing implementation; analytic solutions |
| LDA/QDA | 5.3.1 | **Low** | OOD detection; linear probe theory |
| Probit regression | 5.4.5 | **Very Low** | Skip |

### The Big Picture Connection

These chapters establish two foundational pillars:

**Pillar 1: The training objective.** The output of any language model is multiclass logistic regression (softmax) over the vocabulary, trained with cross-entropy loss against one-hot targets. The gradient is $(y-t)\phi$. Temperature controls the sharpness of the distribution at inference time. Everything in LLM training ultimately connects back to this.

**Pillar 2: The probabilistic framework for deployment.** The inference/decision split (5.2.4) and the expected loss framework (5.2.2) provide a principled language for thinking about: How do you set refusal thresholds? How do you combine signals from multiple classifiers? How do you compensate for class imbalance in evaluation? These are daily problems in applied LLM work, and the decision theory chapter provides the vocabulary.

The bias-variance trade-off (4.3) is important precisely *because* modern scale has broken it — understanding the classical result is necessary to appreciate why double descent is surprising and what it implies about the overparameterized regime we operate in.
