# Ch 8-9: Measuring Performance & Regularization

> These two chapters cover the fundamentals of generalization theory and the practical toolkit for improving it. Honestly, the bias-variance framework is one of those things everyone learns but that becomes less directly useful at frontier scale -- the overparameterized regime and double descent are where the real action is for modern LLMs. The regularization chapter is a grab bag: some techniques (dropout, weight decay, label smoothing, data augmentation) are load-bearing infrastructure at every lab, while others (Bayesian inference, explicit L2 in isolation) are more academically interesting than practically deployed at scale. I'll flag what actually matters as we go.

---

## Chapter 8: Measuring Performance

### 8.1 Training a Simple Model
- **Setup**: MNIST-1D dataset, 10 classes, $D_i = 40$ inputs, network with two hidden layers of $D = 100$ units each, softmax output
- **Key observation**: training error goes to zero (~4000 steps) but test error only drops to ~40% -- classic overfitting
- **Loss vs error divergence**: test *loss* starts increasing even as test *error* stays flat
	- Model becomes increasingly confident about its (wrong) predictions
	- Pre-softmax activations get pushed to extreme values
	- This is a side effect of cross-entropy loss + softmax -- the loss always rewards more confidence on training data

> **LLM relevance: LOW for this specific example** but the confidence calibration issue is huge. Modern LLMs are notoriously poorly calibrated -- they'll say wrong things with high confidence. This connects directly to work on calibration, RLHF reward hacking, and why temperature scaling at inference matters.

### 8.2 Sources of Error

#### 8.2.1 Noise, Bias, and Variance
- **Three sources of test error**:
	- **Noise** ($\sigma^2$): inherent stochasticity in data generation, mislabeled data, unobserved explanatory variables
		- Irreducible -- fundamental ceiling on performance
		- Sometimes noise is absent (deterministic but computationally expensive functions)
	- **Bias**: model can't represent the true function even with optimal parameters
		- Too few parameters, wrong architecture, wrong inductive bias
	- **Variance**: sensitivity to the particular training set sampled
		- Different training sets $\rightarrow$ different learned parameters $\rightarrow$ different predictions
		- Also includes variance from stochastic optimization

#### 8.2.2 Mathematical Formulation
- For regression with least squares loss, test error decomposes cleanly:
$$\mathbb{E}_{\mathcal{D}}[\mathbb{E}_y[L[x]]] = \underbrace{\mathbb{E}_{\mathcal{D}}\left[(\text{f}[x, \phi[\mathcal{D}]] - f_\mu[x])^2\right]}_{\text{variance}} + \underbrace{(f_\mu[x] - \mu[x])^2}_{\text{bias}} + \underbrace{\sigma^2}_{\text{noise}}$$
- Where $f_\mu[x] = \mathbb{E}_{\mathcal{D}}[\text{f}[x, \phi[\mathcal{D}]]]$ is the expected model output across all possible training sets
- **Clean additive decomposition only holds for regression with squared loss** -- for classification and other losses the interaction is more complex

> **LLM relevance: MODERATE.** The decomposition itself isn't directly used in LLM research, but the conceptual framework matters. When people talk about "scaling laws" they're implicitly reasoning about how bias (model too small) and variance (not enough data) change with scale. The noise term connects to data quality work -- cleaning pretraining corpora reduces the noise floor.

### 8.3 Reducing Error

#### 8.3.1 Reducing Variance
- **More training data** $\rightarrow$ lower variance, almost always improves test performance
- Averages out noise, ensures input space is well-sampled

#### 8.3.2 Reducing Bias
- **Increase model capacity**: more hidden units, more layers
- More capacity $\rightarrow$ more flexible function class $\rightarrow$ can fit true function more closely

#### 8.3.3 Bias-Variance Trade-off
- **Classic view**: increasing capacity reduces bias but increases variance
- Optimal capacity minimizes their sum
- **Overfitting**: model fits training data well but fits noise rather than signal -- extra capacity goes to modeling noise
- For fixed data, there's a sweet spot

> **LLM relevance: LOW in isolation.** The classical bias-variance trade-off is the thing that modern deep learning spectacularly violates. It's useful as background for understanding why double descent was surprising, but nobody at a frontier lab is doing bias-variance analysis to choose model size. Scaling laws (Chinchilla, etc.) are the practical replacement.

### 8.4 Double Descent ★★★

- **The big finding**: test error does NOT follow the classical U-shaped bias-variance curve for deep networks
- Three regimes:
	- **Classical/under-parameterized regime**: bias-variance trade-off holds, test error has U-shape
	- **Critical regime**: model has just enough capacity to memorize training data -- performance is *worst* here
	- **Modern/over-parameterized regime**: test error decreases again, eventually surpassing the classical optimum

#### 8.4.1 Explanation
- **Why performance is worst at the critical point**: model has just barely enough capacity to fit training data, so it's forced to contort itself through every point $\rightarrow$ erratic, non-smooth interpolation
- **Why over-parameterized models improve**: with excess capacity, model can choose *how* to interpolate between data points
	- **Inductive bias** determines which interpolation the model selects
	- More capacity $\rightarrow$ smoother interpolation $\rightarrow$ better generalization
- **Curse of dimensionality makes this critical**: in high-D, data is extremely sparse (40D MNIST-1D with 10K examples: $10^{40}$ bins but only $10^4$ data points -- one point per $10^{36}$ bins)
- **What encourages smoothness in the overparameterized regime?** Two hypotheses:
	1. Network initialization encourages smoothness, model never departs
	2. Training algorithm "prefers" converging to smooth functions (implicit regularization)
- **Epoch-wise double descent**: same pattern when you fix capacity and increase training iterations (Nakkiran et al., 2021)
- **Effective model capacity**: depends not just on architecture but on training algorithm and duration

> **LLM relevance: CRITICAL.** This is arguably the most important concept in these two chapters for understanding modern LLMs. Every frontier model is massively overparameterized relative to the classical regime. The fact that performance keeps improving in the overparameterized regime is *why scaling works*. Double descent also explains:
> - Why we don't worry about overfitting in the traditional sense for large pretrained models
> - Why scaling laws are smooth power laws rather than U-curves
> - Why "just make it bigger" has been such a dominant strategy
> - The connection to grokking (delayed generalization after memorization)
> - Why data quality matters so much -- noisy labels make the critical regime more pronounced
>
> Research areas: neural scaling laws (Kaplan et al., Hoffmann et al./Chinchilla), grokking, lottery ticket hypothesis, understanding implicit bias of SGD

### 8.5 Choosing Hyperparameters
- **Validation set**: third data split beyond train/test -- used for model selection
	- Train on training set, evaluate candidates on validation set, report final performance on test set
	- Critical to never use test set for selection -- leads to overfitting to the test set
- **k-fold cross-validation**: partition train+val into $K$ subsets, rotate which is held out
	- Useful when data is limited
	- Final performance: average predictions from all $K$ models on the true test set
- **Hyperparameter search is expensive**: must train a full model per configuration
- **Search methods**:
	- Random search (Bergstra & Bengio, 2012)
	- Bayesian optimization (Gaussian processes, Snoek et al., 2012)
	- Tree-Parzen Estimators (TPE) (Bergstra et al., 2011)
	- SMAC (random forests) (Hutter et al., 2011)
	- Hyperband (multi-armed bandit, early stopping bad configs) (Li et al., 2017)
	- BOHB = Hyperband + TPE (Falkner et al., 2018)

> **LLM relevance: HIGH.** Hyperparameter search is enormously expensive for LLMs because each eval requires a full training run (or at least a significant one). This is why:
> - Scaling laws are so valuable -- they let you predict performance from small runs
> - $\mu$P (maximal update parameterization) exists -- transfer hyperparameters from small to large models
> - Most labs do extensive sweeps on small models and then scale up with those hyperparameters
> - The community has converged on relatively standard recipes (AdamW, cosine schedule, specific LR ranges)

### 8.5+ Notes Section Highlights

#### Capacity
- **Representational capacity**: what the model *could* represent with all possible parameter values
- **Effective capacity**: what it actually reaches given the optimization algorithm
- **VC dimension**: max number of training examples a binary classifier can label arbitrarily -- formal capacity measure
- **Rademacher complexity**: expected empirical performance on random labels

#### Real-World Performance / Distribution Shift
- **Three types of shift** (all critical for deployed LLMs):
	- **Covariate shift**: input distribution $P(\mathbf{x})$ changes
	- **Prior shift**: output distribution $P(\mathbf{y})$ changes
	- **Concept shift**: relationship $P(\mathbf{y}|\mathbf{x})$ changes
- **Data drift**: real-world data changes over time $\rightarrow$ deployed models degrade

> **LLM relevance: VERY HIGH.** Distribution shift is one of the biggest practical problems for deployed LLMs. Models are trained on data up to a cutoff date and then deployed into a world that keeps changing. This connects to:
> - Continual/online learning
> - RAG (retrieval-augmented generation) as a partial solution to knowledge staleness
> - Domain adaptation for specialized applications
> - Evaluation contamination -- when test sets leak into training data

---

## Chapter 9: Regularization

### 9.1 Explicit Regularization

- **Core idea**: add a penalty term $g[\phi]$ to the loss function to bias toward preferred solutions:
$$\hat{\phi} = \underset{\phi}{\text{argmin}} \left[\sum_{i=1}^{I} \ell_i[\mathbf{x}_i, \mathbf{y}_i] + \lambda \cdot g[\phi]\right]$$
- $\lambda$ controls regularization strength
- **Probabilistic interpretation**: regularization term = negative log prior $Pr(\phi)$, making this MAP estimation rather than MLE

#### 9.1.2 L2 Regularization (Weight Decay) ★★★
- Penalize sum of squared weights: $\lambda \sum_j \phi_j^2$
- Also called **Tikhonov regularization**, **ridge regression**, **Frobenius norm regularization**
- Applied to weights only, not biases
- **Effect**: encourages smaller weights $\rightarrow$ smoother output function
	- Smaller weights $\rightarrow$ output varies less $\rightarrow$ less overfitting
	- In the limit ($\lambda \rightarrow \infty$), all weights $\rightarrow 0$, output = constant (just the bias)
- **Two ways it helps**:
	1. Overfitting regime: trades off data fidelity vs smoothness, reduces variance at cost of bias
	2. Over-parameterized regime: encourages smooth interpolation in data-sparse regions

##### Weight Decay vs L2 Regularization
- **For SGD**: weight decay $\equiv$ L2 regularization with $\lambda = \lambda'/2\alpha$
- **For Adam**: they are NOT equivalent because Adam has per-parameter learning rates
- **AdamW** (Loshchilov & Hutter, 2019): implements weight decay correctly for Adam, shown to improve performance

> **LLM relevance: CRITICAL.** Weight decay (via AdamW) is used in literally every modern LLM training run. It's not optional -- it's part of the standard recipe. The AdamW distinction is important: the original Adam paper's L2 regularization doesn't work the same way, and AdamW was a meaningful improvement. Typical values are in the 0.01-0.1 range for LLM pretraining.

#### Other Norms
- **L0 regularization**: penalizes number of non-zero weights $\rightarrow$ pruning
	- Hard to optimize (non-differentiable)
- **L1 regularization / LASSO**: penalizes absolute values, encourages sparsity
	- Constant derivative regardless of weight magnitude $\rightarrow$ actually pushes small weights to zero (unlike L2)
- **Elastic net**: L1 + L2 together
- **Spectral norm regularization**: Lipschitz constant of network bounded by product of spectral norms of weight matrices
	- Bartlett et al. (2017), Neyshabur et al. (2018) -- add terms encouraging smaller spectral norms
	- Gouk et al. (2021) -- constrain Lipschitz constant directly

> **LLM relevance: HIGH for sparsity.** L1/L0-style regularization connects to the massive push toward sparse models:
> - Pruning (magnitude pruning, SparseGPT, Wanda)
> - Mixture of Experts (MoE) as a form of structured sparsity
> - Sparse attention patterns
> - Post-training quantization implicitly does something related (sets small weights to zero)
> Spectral norm stuff is more niche but shows up in GAN training and some stability analyses.

### 9.2 Implicit Regularization ★★★

#### 9.2.1 Implicit Regularization in Gradient Descent
- **Key insight**: discrete gradient descent doesn't just minimize the loss -- it implicitly minimizes a *modified* loss:
$$\tilde{L}_{GD}[\phi] = L[\phi] + \frac{\alpha}{4}\left\|\frac{\partial L}{\partial \phi}\right\|^2$$
- The extra term penalizes the squared gradient norm -- avoids regions where loss surface is steep
- Doesn't change the position of minima (gradients are zero there) but changes which minimum the optimizer reaches
- **Larger learning rate $\alpha$** $\rightarrow$ stronger implicit regularization $\rightarrow$ better generalization
	- This explains the empirical finding that larger learning rates (up to the point of instability) generalize better

#### 9.2.2 Implicit Regularization in SGD
- SGD adds another term beyond gradient descent's implicit regularization:
$$\tilde{L}_{SGD}[\phi] = L[\phi] + \frac{\alpha}{4}\left\|\frac{\partial L}{\partial \phi}\right\|^2 + \frac{\alpha}{4B}\sum_{b=1}^{B}\left\|\frac{\partial L_b}{\partial \phi} - \frac{\partial L}{\partial \phi}\right\|^2$$
- Extra term = variance of batch gradients
- **Favors solutions where all batches agree on the gradient direction**
	- i.e., solutions that fit *all* the data uniformly well, not just some subsets
- **Smaller batch sizes** $\rightarrow$ more gradient variance $\rightarrow$ stronger implicit regularization
- This may explain why SGD generalizes better than full-batch GD, and smaller batches generalize better than larger ones

> **LLM relevance: VERY HIGH.** This is one of the deepest insights in these chapters. Implicit regularization explains many empirical observations in LLM training:
> - Why learning rate warmup and decay schedules matter so much
> - Why there's an optimal batch size for a given compute budget (gradient noise is a feature, not a bug)
> - The connection to sharpness-aware minimization (SAM) which explicitly penalizes sharp minima
> - Why the learning rate / batch size ratio matters (linear scaling rule)
> - Large learning rates pushing optimization toward flatter minima -- this connects to the "edge of stability" phenomenon
>
> In practice, labs carefully tune the batch size / learning rate relationship. The Chinchilla paper showed that for compute-optimal training, you want the right balance between tokens seen and model size, but the batch size also matters for this implicit regularization reason.

### 9.3 Heuristics to Improve Performance

#### 9.3.1 Early Stopping
- Stop training before full convergence -- model captures coarse structure but hasn't overfit to noise
- **Equivalent to L2 regularization**: weights start small (initialization) and don't have time to grow large
	- Under quadratic approximation: effective regularization weight $\lambda \approx 1/(\tau\alpha)$ where $\tau$ = stopping time, $\alpha$ = learning rate
- **Practical advantage**: only one hyperparameter (stopping time), can select without retraining by monitoring validation loss

> **LLM relevance: MODERATE.** Early stopping is used everywhere but it's not the primary regularization mechanism for LLMs. In practice, large LLMs are often trained for a fixed compute budget determined by scaling laws, and you stop when you hit your budget. But validation loss is still monitored, and the best checkpoint is often not the final one. More relevant for fine-tuning, where overfitting happens fast with small datasets.

#### 9.3.2 Ensembling
- Train multiple models, average predictions
- **Combining**: mean of outputs (regression) or mean of pre-softmax logits (classification)
- **Methods for diversity**:
	- Different random initializations
	- **Bagging** (bootstrap aggregating): resample training data with replacement
	- Different hyperparameters
	- Different model families
- Reduces variance by averaging out idiosyncratic errors

> **LLM relevance: MODERATE.** Ensembling is well-understood to work but rarely used for frontier LLMs due to cost -- you can't easily ensemble models that each cost $100M+ to train. However, the concept appears in:
> - Mixture of Experts (MoE) can be viewed as a form of conditional ensembling
> - Self-consistency in chain-of-thought reasoning (sample multiple reasoning paths, majority vote)
> - Reward model ensembles in RLHF
> - Model soups / weight averaging (Wortsman et al., 2022) -- average weights of models fine-tuned with different hyperparameters
> - Speculative decoding uses a small model as a draft that a large model verifies -- not ensembling per se but related

#### 9.3.3 Dropout ★★
- Randomly zero out hidden units (typically 50%) at each training step
- Makes network less dependent on any single unit, encourages distributed representations
- **Eliminates "kinks"**: when multiple hidden units conspire to create a local feature that reduces training loss but doesn't generalize, dropping one of them breaks the conspiracy, and subsequent gradient steps remove it
- **At test time**: use all units but multiply weights by $(1 - p_{\text{dropout}})$ (**weight scaling inference rule**)
- **Monte Carlo dropout**: run network multiple times with random dropout at test time, combine results
	- Approximates Bayesian inference -- gives uncertainty estimates for free

> **LLM relevance: MODERATE-HIGH.** Dropout was critical in the pre-transformer era but its role has evolved:
> - Original transformers used dropout and it's still used in many implementations
> - GPT-3 used dropout rate of 0.1
> - Some modern architectures (e.g., PaLM, LLaMA) dropped dropout entirely for pretraining, relying on other regularization
> - Still very commonly used in fine-tuning where datasets are smaller
> - MC dropout for uncertainty estimation is used in some production systems
> - **DropPath / Stochastic Depth** (dropping entire layers in residual networks) is arguably more important now than standard dropout

#### 9.3.4 Applying Noise
- **Input noise**: smooths learned function, equivalent to penalizing derivatives of network output w.r.t. input
	- **Adversarial training**: find worst-case input perturbations, train on those
- **Weight noise**: encourages convergence to wide, flat minima where small parameter changes don't matter
- **Label smoothing** ★★★: instead of hard targets (0 or 1), use mixture:
	- True label gets probability $1 - \rho$, other classes share $\rho$ equally
	- Discourages overconfident predictions (pre-softmax activations going to extremes)
	- Equivalent to modifying loss to be cross-entropy against a smoothed distribution

> **LLM relevance: VERY HIGH for label smoothing.** Label smoothing is used in virtually every modern model. For LLMs:
> - Prevents the model from being pathologically overconfident
> - Improves calibration of predicted probabilities
> - Used in the original Transformer paper and almost all subsequent work
> - Knowledge distillation (training a smaller model on a larger model's soft outputs) is conceptually related -- the soft targets from the teacher are like an extreme, data-dependent form of label smoothing
>
> Adversarial training connects to robustness research and adversarial attacks on LLMs (jailbreaking, prompt injection).

#### 9.3.5 Bayesian Inference
- Treat parameters as random variables, compute posterior $Pr(\phi|\{\mathbf{x}_i, \mathbf{y}_i\})$ via Bayes' rule
- Prediction = infinite weighted ensemble over all parameter values:
$$Pr(\mathbf{y}|\mathbf{x}, \{\mathbf{x}_i, \mathbf{y}_i\}) = \int Pr(\mathbf{y}|\mathbf{x}, \phi) Pr(\phi|\{\mathbf{x}_i, \mathbf{y}_i\}) d\phi$$
- **Elegant but impractical** for large networks -- can't represent or integrate over the full posterior
- All current methods are approximations (variational inference, MCMC, etc.)

> **LLM relevance: LOW for direct application, HIGH for conceptual influence.** You will never do full Bayesian inference over a 70B parameter model. But the ideas show up in:
> - Variational inference frameworks
> - Uncertainty quantification research
> - The connection between temperature in sampling and posterior sharpness
> - Bayesian model averaging as the theoretical justification for ensembling
> - Laplace approximation for LLM uncertainty (recent work)

#### 9.3.6 Transfer Learning and Multi-Task Learning ★★★
- **Transfer learning**: pretrain on a large secondary task, then adapt to the target task
	- Remove final layer(s), add new output layer(s)
	- Either freeze pretrained weights or fine-tune entire model
	- **Principle**: network learns good internal representations from the secondary task
- **Multi-task learning**: train on multiple tasks simultaneously with shared backbone
	- Each task gets its own output head
	- Tasks share a common understanding of the input

> **LLM relevance: ABSOLUTELY CRITICAL.** This is the single most important concept for modern LLMs in these two chapters. The entire LLM paradigm is transfer learning:
> - **Pretraining** (next-token prediction on massive corpora) = the "secondary task"
> - **Fine-tuning** (instruction tuning, RLHF, task-specific adaptation) = adaptation to the target
> - The reason LLMs work at all is that next-token prediction forces the model to learn rich representations of language, knowledge, and reasoning
> - **Multi-task learning** shows up as instruction tuning on diverse tasks, FLAN-style training, and the general finding that training on more diverse tasks improves generalization
> - **LoRA, QLoRA, adapters** = modern parameter-efficient versions of the "freeze backbone, train new layers" idea
> - Foundation models are fundamentally a transfer learning paradigm

#### 9.3.7 Self-Supervised Learning ★★★
- **Generative self-supervised learning**: mask part of input, predict the missing part
	- Language: mask tokens, predict them (BERT-style MLM)
	- Images: mask patches, predict them (MAE)
- **Contrastive self-supervised learning**: learn that similar pairs are related, dissimilar pairs aren't
	- Images: SimCLR, CLIP
	- Text: next-sentence prediction, sentence similarity

> **LLM relevance: FOUNDATIONAL.** LLM pretraining *is* generative self-supervised learning. Next-token prediction is the purest form of "predict the missing part." This is the method that enables training on unlimited unlabeled data, which is the key insight that made LLMs possible. Contrastive learning shows up in CLIP (connecting text and images) and in embedding models like those used for retrieval/RAG.

#### 9.3.8 Data Augmentation ★★
- Transform inputs while preserving labels (rotation, flipping, color jitter for images; synonym substitution, back-translation for text)
- Goal: teach invariance to irrelevant transformations
- Effectively increases dataset size

> **LLM relevance: MODERATE for pretraining (the corpus is already huge), HIGH for fine-tuning.** In the LLM world, data augmentation takes different forms:
> - Back-translation for multilingual models
> - Paraphrasing training examples
> - **Synthetic data generation** (using a stronger model to generate training data for a weaker one) is arguably the modern version of data augmentation and is enormously important
> - Test-time augmentation (asking the same question multiple ways and aggregating) relates to prompt engineering

---

## Summary: What Matters Most for Modern LLM Research

### Tier 1 -- Load-Bearing Concepts
1. **Transfer learning / self-supervised pretraining** -- the entire LLM paradigm
2. **Double descent / overparameterization** -- why scaling works at all
3. **Implicit regularization of SGD** -- why training dynamics matter (batch size, LR, schedules)
4. **Weight decay (AdamW)** -- standard recipe, always used
5. **Label smoothing** -- standard recipe for training

### Tier 2 -- Important and Widely Used
6. **Dropout** -- still used in fine-tuning, some pretraining
7. **Data augmentation / synthetic data** -- huge for fine-tuning, RLHF data generation
8. **Distribution shift** -- critical for deployed systems
9. **Sparsity (L1/L0, pruning)** -- growing importance with efficiency focus

### Tier 3 -- Good to Know, Less Directly Used
10. **Bias-variance decomposition** -- useful mental model, less directly applied
11. **Ensembling** -- too expensive for full LLMs, but ideas show up in MoE, self-consistency
12. **Bayesian inference** -- theoretically elegant, practically limited at scale
13. **Early stopping** -- used but not the primary tool
14. **Hyperparameter search methods** -- relevant but dominated by scaling law predictions and $\mu$P
