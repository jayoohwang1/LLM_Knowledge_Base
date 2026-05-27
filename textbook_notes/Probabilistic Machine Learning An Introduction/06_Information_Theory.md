# Chapter 6: Information Theory

**Frontier-lab take**: This is one of *the* most practically important chapters for modern LLM work. Cross-entropy is literally the loss function. KL divergence shows up in RLHF, distillation, variational inference, policy gradients. Mutual information underpins representation learning. Perplexity is the bread-and-butter LM evaluation metric. If you only deeply internalize one math chapter, make it this one. Most of the chapter is starred ("advanced") in the book, but for LLM research it's all core.

---

## 6.1 Entropy

- **Definition (discrete)**: $\mathbb{H}(X) \triangleq -\sum_{k=1}^K p(X=k) \log_2 p(X=k) = -\mathbb{E}_X[\log p(X)]$
  - Units: **bits** if $\log_2$, **nats** if $\ln$.
  - **Interpretation**: average uncertainty / minimal number of bits needed to encode samples from $p$.
- **Bounds**:
  - **Max entropy** = uniform distribution: $\mathbb{H}(X) = \log K$.
  - **Min entropy** = 0 for any delta function.
- **Binary entropy function**: $\mathbb{H}(\theta) = -[\theta \log_2 \theta + (1-\theta)\log_2(1-\theta)]$, peaks at 1 bit when $\theta = 0.5$.

### Why it matters for LLMs
- **Token-level entropy** of the model's next-token distribution is widely used as an *uncertainty signal*: high entropy means the model is unsure (good for active learning, hallucination detection, speculative decoding gating, "thinking" allocation).
- **Entropy regularization** in RL/RLHF: adding $\beta \mathbb{H}(\pi(\cdot|s))$ to the objective prevents policy collapse. Modern variants (entropy bonuses in GRPO/PPO for reasoning models) directly use this.
- **Sampling temperature** is essentially a knob on the entropy of the output distribution.

### 6.1.2 Cross entropy
- $\mathbb{H}_{ce}(p, q) \triangleq -\sum_k p_k \log q_k$.
- **Shannon source coding theorem**: $\mathbb{H}_{ce}(p,q) \geq \mathbb{H}(p)$, equality iff $q = p$.
- **This is the LLM training loss**. Literally. Next-token prediction = minimize $-\sum_n \log q_\theta(x_n | x_{<n})$ which is cross-entropy between empirical data distribution and model.
- Distillation losses are also cross-entropy (or KL, equivalent up to a constant) between teacher and student distributions over the vocab.

### 6.1.3 Joint entropy
- $\mathbb{H}(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$.
- **Subadditivity**: $\mathbb{H}(X,Y) \leq \mathbb{H}(X) + \mathbb{H}(Y)$, equality iff independent.
- **Lower bound**: $\mathbb{H}(X,Y) \geq \max\{\mathbb{H}(X), \mathbb{H}(Y)\}$.

### 6.1.4 Conditional entropy
- $\mathbb{H}(Y|X) \triangleq \mathbb{E}_{p(X)}[\mathbb{H}(p(Y|X))] = \mathbb{H}(X,Y) - \mathbb{H}(X)$.
- **Conditioning reduces entropy on average**: $\mathbb{H}(Y|X) \leq \mathbb{H}(Y)$.
  - But for a *specific* observation, entropy can go up — i.e., a particular data point can confuse you. Only the expectation is monotone.
- **Chain rule**: $\mathbb{H}(X_1, \dots, X_n) = \sum_i \mathbb{H}(X_i | X_{<i})$.
  - This is *exactly* the decomposition autoregressive LMs exploit. Modeling $p(x_{1:n}) = \prod_i p(x_i | x_{<i})$ as a sum of conditional entropies is the whole game.

### 6.1.5 Perplexity — **critical for LLMs**
- $\text{perplexity}(p) \triangleq 2^{\mathbb{H}(p)}$.
- For empirical $p_\mathcal{D}$ and model $p$:
  - $\text{perplexity}(p_\mathcal{D}, p) = 2^{\mathbb{H}_{ce}(p_\mathcal{D}, p)} = \sqrt[N]{\prod_n \frac{1}{p(x_n)}}$ (geometric mean of inverse predictive probs).
- **Interpretation: effective branching factor** — number of equally-likely next tokens the model is "choosing between".
- **In practice**:
  - Standard pretraining metric (e.g., WikiText, C4 ppl). Going from 20 to 10 ppl is a *huge* win.
  - **Not directly comparable across tokenizers** (different vocab sizes / segmentations). Always check tokenizer when comparing PPL between papers.
  - **Bits-per-byte (BPB)** = $\mathbb{H}_{ce} / \text{bytes-per-token}$ is tokenizer-invariant; preferred in recent comparisons (Chinchilla, scaling-law papers).
  - PPL gets weird on instruct/chat models with templated prompts; researchers increasingly favor downstream evals.
- **Lower bound**: PPL of any model $\geq 2^{\mathbb{H}(p^*)}$ where $p^*$ is the true data process (irreducible entropy).

### 6.1.6 Differential entropy (continuous) — less central for LLMs
- $h(X) = -\int p(x) \log p(x) dx$. Can be **negative** (densities can exceed 1).
- **Gaussian**: $h(\mathcal{N}(\mu, \Sigma)) = \frac{1}{2}\ln[(2\pi e)^d |\Sigma|]$.
- Mostly used in VAE/flow/diffusion derivations and in continuous-latent representation analysis. Not something you compute often directly in LLM-land, but you'll see it in ELBO derivations.

---

## 6.2 KL divergence — **core ingredient in modern LLM training**

- **Definition**: $D_{KL}(p \| q) \triangleq \sum_k p_k \log \frac{p_k}{q_k} = \mathbb{H}_{ce}(p, q) - \mathbb{H}(p)$.
- **Information inequality**: $D_{KL}(p \| q) \geq 0$, equality iff $p = q$. Proof via Jensen's inequality on concave $\log$.
- **NOT a metric**: asymmetric, no triangle inequality. Just a divergence.
- **Interpretation**: extra bits paid when coding $p$-data with a $q$-optimized code.

### 6.2.3 Gaussian KL
- Closed form, used constantly in VAEs:
  $D_{KL}(\mathcal{N}_1 \| \mathcal{N}_2) = \frac{1}{2}\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1) - D + \log\frac{\det\Sigma_2}{\det\Sigma_1}\right]$
- Scalar version drops out of the standard ELBO derivation.

### 6.2.5 KL and MLE — **the foundational equivalence**
- Minimizing $D_{KL}(p_\mathcal{D} \| q_\theta)$ wrt $\theta$ $\equiv$ maximizing log-likelihood. The empirical KL reduces to $-\frac{1}{N}\sum_n \log q(x_n) + C$.
- **Caveat the book makes (worth pondering)**: MLE puts all mass on training data, which is a brittle empirical distribution. Mitigations:
  - **Kernel density / smoothing**: hard in high dim.
  - **Data augmentation**: huge in vision; in LLMs it shows up as synthetic data generation, paraphrasing, instruction tuning data scaling. Why "rephrase the web" / SynthLLM-style work matters.

### 6.2.6 Forward vs reverse KL — **this distinction is everywhere in modern LLM RL**
- **Forward KL** $D_{KL}(p \| q)$: **mode-covering / zero-avoiding** / mean-seeking. Penalizes $q$ being 0 where $p > 0$ infinitely.
  - **M-projection** (moment projection).
  - **Used in**: MLE / supervised next-token training (data is $p$, model is $q$), distillation when you want student to cover teacher.
- **Reverse KL** $D_{KL}(q \| p)$: **mode-seeking / zero-forcing**. Penalizes $q$ putting mass where $p = 0$.
  - **I-projection** (information projection).
  - **Used in**: variational inference (ELBO maximization is reverse-KL minimization), RLHF/DPO regularization terms $\beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$, GAN-like objectives.
- **Frontier relevance — huge**:
  - RLHF/PPO: KL penalty between policy and SFT reference is *reverse* KL (or "k1"/"k2"/"k3" estimators in practice; see Schulman's blog post — also surfaces in TRL and GRPO).
  - DPO derivation is built on the reverse-KL constrained RL objective.
  - **On-policy distillation** vs **off-policy distillation** roughly corresponds to reverse vs forward KL. Mode collapse in distilled small models often traceable to reverse-KL choice.
  - Recent papers (e.g., MiniLLM, GKD) explicitly study mode coverage vs mode seeking tradeoffs.

---

## 6.3 Mutual Information — **representation learning's backbone**

- **Definition**: $\mathbb{I}(X; Y) \triangleq D_{KL}(p(x,y) \| p(x)p(y)) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$.
- **Always non-negative**, 0 iff independent.
- **Identities** (memorize):
  - $\mathbb{I}(X;Y) = \mathbb{H}(X) - \mathbb{H}(X|Y) = \mathbb{H}(Y) - \mathbb{H}(Y|X)$
  - $\mathbb{I}(X;Y) = \mathbb{H}(X) + \mathbb{H}(Y) - \mathbb{H}(X,Y)$
  - $\mathbb{I}(X;Y) = \mathbb{H}(X,Y) - \mathbb{H}(X|Y) - \mathbb{H}(Y|X)$
- **Information diagram**: Venn-diagram intuition (Fig 6.4). $\mathbb{I}$ = intersection, $\mathbb{H}(X|Y)$ = $X$-only crescent, $\mathbb{H}(X,Y)$ = union.

### Why MI matters in modern ML
- **Self-supervised / contrastive learning**: SimCLR, CPC, InfoNCE, MoCo all justified as MI lower bounds (specifically the InfoNCE bound $\mathbb{I}(X;Y) \geq \log N - \mathcal{L}_{\text{NCE}}$). The reason contrastive pretraining works at all.
- **Mutual-information bottleneck** / **information bottleneck**: $\min \mathbb{I}(X;Z) - \beta \mathbb{I}(Z;Y)$. Theoretical lens on deep learning generalization (Tishby).
- **CLIP / multimodal alignment**: image-text alignment is effectively MI maximization between modalities.
- **Probing / interpretability**: measuring MI between hidden states and labels to diagnose what models "know" at each layer.
- **Caveats** (important and current): MI estimators in high dim are notoriously bad. MINE, InfoNCE, etc. provide bounds but tight estimation is hard. Don't trust naive plug-in estimates of MI from neural net activations.

### 6.3.4 Conditional MI
- $\mathbb{I}(X;Y|Z) = \mathbb{H}(X|Z) + \mathbb{H}(Y|Z) - \mathbb{H}(X,Y|Z)$.
- **Chain rule for MI**: $\mathbb{I}(Z_1, \dots, Z_N; X) = \sum_n \mathbb{I}(Z_n; X | Z_{<n})$.
- Relevant in **causal probing** and decomposing what each layer/token contributes.

### 6.3.5 MI as generalized correlation
- For jointly Gaussian $(X,Y)$ with correlation $\rho$: $\mathbb{I}(X;Y) = -\frac{1}{2}\log(1 - \rho^2)$.
- $\rho = \pm 1 \Rightarrow \infty$ MI (deterministic), $\rho = 0 \Rightarrow 0$ MI (independence).
- Captures **nonlinear** dependence that Pearson misses — useful for diagnostic plots.
- **KSG estimator**: KNN-based MI estimator (sklearn `mutual_info_regression`). Reasonable default for low-dim.

### 6.3.6–6.3.7 Normalized MI, MIC
- **NMI**: $\mathbb{I}(X;Y) / \min(\mathbb{H}(X), \mathbb{H}(Y)) \in [0,1]$.
- **MIC** (maximal information coefficient): searches over 2D grids for the discretization that maximizes normalized MI. "Equitable" — same score for equally-noisy relationships of any functional form. Called "a correlation for the 21st century" (Reshef et al. 2011).
- **In practice**: MIC is more of a stats/EDA tool, rarely shows up in deep learning papers, but worth knowing.

### 6.3.8 Data processing inequality (DPI) — **fundamental**
- **Theorem**: $X \to Y \to Z$ Markov chain $\Rightarrow \mathbb{I}(X;Y) \geq \mathbb{I}(X;Z)$.
- **Intuition**: post-processing (functions, noisy channels) can only destroy information about the source.
- **LLM relevance — huge**:
  - **Distillation upper bound**: a student trained on teacher outputs cannot have more information about the true data than the teacher's outputs do. Justifies why you can't infinitely distill.
  - **Tokenization / representation choices**: any deterministic encoding step is upper-bounded by what's in the raw signal.
  - **RAG / retrieval**: retrieved context $Z$ filtered from corpus $Y$ can only have less info than the full corpus.
  - **Privacy / DP**: motivates why noise injection bounds adversary's information.

### 6.3.9 Sufficient statistics
- $s(\mathcal{D})$ sufficient if $\mathbb{I}(\theta; s(\mathcal{D})) = \mathbb{I}(\theta; \mathcal{D})$.
- **Minimal sufficient statistic**: maximal compression preserving all info about $\theta$.
- Example: count of heads in $N$ Bernoulli trials.
- **Modern angle**: representation learning often implicitly searches for minimal sufficient statistics for downstream tasks. Information bottleneck formalizes this.

### 6.3.10 Fano's inequality
- Bounds error probability $P_e$ in classification by conditional entropy:
  $P_e \geq \frac{\mathbb{H}(Y|X) - 1}{\log |\mathcal{Y}|}$
- **Implication**: minimizing $\mathbb{H}(Y|X)$ (= maximizing $\mathbb{I}(X;Y)$) lower-bounds the achievable error. Justifies **MI-based feature selection**.
- Useful for theoretical analyses of LLM capability ceilings given input information.

---

## Bonus content: things the chapter doesn't fully cover but you should know

- **Jensen-Shannon divergence**: symmetric, bounded version of KL. $\text{JSD}(p,q) = \frac{1}{2}D_{KL}(p\|m) + \frac{1}{2}D_{KL}(q\|m)$ where $m = (p+q)/2$. Used in GANs (original), some RL setups.
- **Total variation, Wasserstein**: alternative divergences. Wasserstein gives non-vanishing gradients when supports don't overlap (W-GAN motivation), increasingly relevant in flow matching / diffusion.
- **f-divergences**: general family containing KL, reverse KL, JSD, Hellinger, $\chi^2$. Many GAN variants live here.

---

## Quick reference: where each concept shows up in LLM research

| Concept | LLM use case |
|---|---|
| **Cross entropy** | Pretraining loss, every next-token objective |
| **Perplexity / BPB** | Standard LM eval, scaling laws |
| **Forward KL** | MLE, distillation (mode-covering) |
| **Reverse KL** | RLHF/PPO/DPO regularizer, ELBO/VAE, mode-seeking distillation |
| **Token-level entropy** | Uncertainty, sampling temperature, RL entropy bonus, speculative decoding |
| **Conditional entropy chain rule** | Autoregressive factorization |
| **Mutual information** | Contrastive SSL (InfoNCE, CPC), CLIP, info bottleneck |
| **DPI** | Distillation bounds, RAG limits, privacy |
| **Sufficient statistics** | Representation learning, IB framework |
| **Fano's inequality** | Theoretical error bounds, capability analysis |
| **Gaussian KL** | VAEs, diffusion / flow matching ELBOs |

---

## Code snippets you'll actually write

```python
# Cross-entropy loss (the LLM loss)
import torch.nn.functional as F
# logits: (B, T, V), targets: (B, T)
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
ppl = loss.exp()  # perplexity

# Token-level entropy (uncertainty signal)
probs = logits.softmax(dim=-1)
token_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)  # (B, T)

# KL divergence for RLHF / distillation
# Forward KL: D_KL(p || q), p = teacher/reference, q = student/policy
kl_fwd = (p * (p.log() - q.log())).sum(-1)
# Reverse KL: D_KL(q || p) — used in RLHF KL penalty
kl_rev = (q * (q.log() - p.log())).sum(-1)
# Schulman's k3 estimator (low variance, on-policy):
# k3 = (p/q - 1) - log(p/q), unbiased for D_KL(q || p)
```

---

## TL;DR / final senior-researcher take
- **Must-know cold**: cross-entropy, perplexity, forward/reverse KL distinction, MI definition + identities, DPI.
- **Conceptual lens you should carry around**: every loss is some divergence; every architecture is some channel obeying DPI; every representation can be analyzed as a (lossy) sufficient statistic.
- **Where the action currently is** (2024–2026): RL post-training and the choice of KL direction/estimator; data-efficient training and MI-based SSL objectives; calibration and entropy-based uncertainty for reasoning models / agentic systems; theoretical analyses of distillation via DPI and IB.
- Chapter is short but information-dense (pun intended). Worth re-reading every 6 months — you'll notice new things each time as the field shifts.
