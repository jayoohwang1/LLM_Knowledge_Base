# Chapter 19: Learning with Fewer Labeled Examples

> **TL;DR for the impatient**: This chapter is the *prehistory of modern LLM training* in disguise. Strip away the 2020-era CNN/CIFAR framing and what's left is the recipe used today: **self-supervised pretraining + (parameter-efficient) fine-tuning + pseudo-labels/self-training + meta-learning-style few-shot eval**. The single most important section for modern frontier work is **19.2.4 (Unsupervised/self-supervised pretraining)** and **19.2.2 (Adapters)** — those two ideas underpin nearly everything post-GPT-3. Active learning (19.4) and meta-learning (19.5) matter less than they did in 2020 but are quietly creeping back via data selection and ICL.

---

## Big-picture framing (why this chapter still matters in 2026)

- **The chapter's premise** (massive params $\gg$ labels) was the 2018-2020 framing for vision. The LLM era *radicalized* it: we now have $\sim10^{12}$ params and zero "task labels" — we use the data itself as supervision.
- **Mental remapping** for modern LLM work:
	- **Data augmentation (19.1)** → modern equivalents: **synthetic data generation**, paraphrasing, instruction back-translation, augmentation in RL rollouts (rejection sampling, best-of-N), Mixup-style methods rarely used directly but the *philosophy* (vicinal risk minimization, label-preserving perturbations) lives in things like DPO data construction.
	- **Transfer learning + fine-tuning (19.2)** → SFT, continued pretraining, domain-adaptive pretraining (DAPT/TAPT), instruction tuning.
	- **Adapters (19.2.2)** → **LoRA, QLoRA, DoRA, IA³, prefix/prompt tuning** — the entire PEFT zoo descends directly from this section.
	- **Self-supervised pretraining (19.2.4)** → the *raison d'être* of modern foundation models: next-token prediction, masked LM, contrastive (CLIP), JEPA, etc.
	- **Semi-supervised / self-training (19.3)** → modern **iterative self-improvement**: STaR, ReST, RFT, self-rewarding LMs, distillation from a stronger teacher, Constitutional AI bootstrap, rejection sampling fine-tuning.
	- **Active learning (19.4)** → **data curation/selection** (DSIR, DoReMi, deduplication, importance sampling for pretraining corpora), preference data selection for RLHF.
	- **Meta-learning (19.5)** → **in-context learning** is the modern incarnation; ICL is essentially MAML's "fast adaptation" but realized via attention rather than gradient steps.
	- **Few-shot (19.6)** → **few-shot prompting**, ICL evaluation suites (MMLU few-shot, BBH, etc.).
	- **Weakly supervised (19.7)** → **RLHF/DPO with noisy preferences**, distant supervision-style synthetic labeling.

---

## 19.1 Data augmentation

- **Core idea**: apply label-preserving transformations to inputs to inflate effective dataset size.
- **Vision examples**: random crops, zooms, flips, color jitter; AutoAugment (RL/Bayesian-optimized augmentation policies).
- **NLP**: synonym replacement, char dropout, back-translation; in practice **largely supplanted by sheer pretraining scale** for LLMs, though it lives on in:
	- **RL training**: paraphrasing prompts, format randomization, multi-turn rollouts as implicit augmentation.
	- **Multimodal**: CLIP-style training relies heavily on crops + color distortion.
	- **Robustness/jailbreak training**: adversarial paraphrases, character perturbations.

### 19.1.2 Theoretical justification — vicinal risk minimization

- Standard ERM uses empirical distribution $p_\mathcal{D}(x,y) = \frac{1}{N}\sum_n \delta(x-x_n)\delta(y-y_n)$.
- Augmentation replaces it with a **smoothed** distribution $p_\mathcal{D}(x,y|A) = \frac{1}{N}\sum_n p(x|x_n, A)\delta(y-y_n)$ — i.e., we integrate the loss over a *neighborhood* of each training point.
- Framing: augmentation = **injecting prior knowledge** about invariances algorithmically. Whatever symmetry $A$ encodes is what you've told the model is irrelevant.
- **Modern take**: for LLMs, the "augmentation" prior is much weaker because you can't easily articulate semantic invariances in text. This is why **synthetic data from a stronger model** (which *does* know semantics) is the dominant form of "augmentation" today — see Phi-series, Llama 3 post-training, DeepSeek-R1's distillation, etc.

---

## 19.2 Transfer learning — **the most important section**

- **Premise**: pretrain on a large source dataset $\mathcal{D}_p$, fine-tune on a small target $\mathcal{D}_q$. The shared-parameter approach to leverage structural similarity between tasks.
- This is literally the LLM playbook: pretrain on the internet, post-train on whatever you care about.

### 19.2.1 Fine-tuning

- **Recipe**: train classifier $p(y|x,\theta_p)$ on source, swap output head, fine-tune on target with low LR to avoid catastrophic forgetting.
- **"Model surgery"**: chop final layer, replace with new head $W_3 h(x;\theta_1) + b_3$.
	- If $\theta_1$ frozen → convex linear probe. Useful in long-tail regimes (rare classes).
	- If $\theta_1$ unfrozen → full fine-tune with smaller LR.
- **Modern equivalents**:
	- **SFT (Supervised Fine-Tuning)**: full-parameter fine-tune on instruction/response pairs. Catastrophic forgetting is real — hence mixing in pretraining data, replay buffers, "model soups".
	- **Continued pretraining / DAPT / TAPT**: domain-adaptive pretraining (BioMedLM, Code Llama) — same recipe, just in next-token loss.
	- **Linear probing vs full FT**: still relevant — Kumar et al. (2022) "fine-tuning can distort pretrained features" showed LP-then-FT often wins for OOD. Modern echo: keep LM head untrainable in some PEFT recipes.

### 19.2.2 Adapters — **the seed of PEFT**

- **Motivation**: full FT is slow, takes lots of memory, and you get a new full-sized model per task. Adapters insert small trainable modules into a frozen backbone.
- **Transformer adapter (Houlsby 2019)**: two bottleneck MLPs per transformer block — one after MHA, one after FFN. With dim $D$ and bottleneck $M$: $O(DM)$ params/layer. Empirically gets full-FT performance with ~1-10% of params.
	- Skip connections + zero-init → starts as identity, no degradation at init.
- **CNN adapter (Rebuffi)**: 1×1 convs added in series or parallel.
	- Series: $\rho(x) = x + \text{diag}_1(\alpha) \otimes x = \text{diag}_1(I + \alpha) \otimes x$ — **low-rank multiplicative perturbation**.
	- Parallel: $y = f \otimes x + \text{diag}_1(\alpha) \otimes x$ — **low-rank additive perturbation**.

#### Why this section matters NOW (PEFT landscape)

- **LoRA** (Hu et al. 2021): $W' = W + BA$ where $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times d}$, $r \ll d$. Direct descendant of the "parallel low-rank additive perturbation". Train only $A, B$; freeze $W$.
- **QLoRA**: LoRA on top of 4-bit NF4-quantized base weights → fine-tune 65B on a single 48GB GPU.
- **DoRA**: decompose $W = m \cdot \frac{V}{\|V\|}$ (magnitude + direction), only LoRA the direction.
- **IA³**: rescale activations elementwise (just learned multipliers).
- **Prefix tuning / prompt tuning / P-tuning v2**: prepend learnable vectors to attention K/V — analog of input-side adapters.
- **Practical reality at frontier labs**:
	- LoRA is *the* default for fine-tuning research experiments — fast, mergeable, multiple "skills" can be stored as small adapter files.
	- **Serving**: vLLM and SGLang support **multi-LoRA serving** — one base model + N adapters → N personalized models.
	- For *final* production models, full SFT is usually used because you can afford the compute and the few-% quality gap matters.
	- Hot research: **adapter composition** (TIES, DARE, model merging), **MoE-of-LoRAs**, **LoRA for RLHF** (lower memory PPO).

### 19.2.3 Supervised pre-training

- **Vision**: ImageNet pre-training was the de facto standard. Murphy notes it's been found to be more of a "speedup trick" than essential — train from scratch long enough and you catch up (He et al. 2019).
- **NLP**: pretrain on NLI (e.g., InferSent) was a thing, but **completely supplanted by self-supervised**.
- **Modern relevance**: limited — except for things like **classification-style reward models** trained from human preferences, which is "supervised pretraining" of a critic.

### 19.2.4 Unsupervised/self-supervised pretraining — **THE foundational idea**

- Use unlabeled data; create labels algorithmically from the input itself.
- Three families:

#### 19.2.4.1 Imputation / cloze tasks ("fill-in-the-blank")

- Partition $x = (x_h, x_v)$, predict $\hat{x}_h = f(x_v, x_h = 0)$.
- **This is literally MLM (BERT) and the entire encoder family**.
- Variants: span corruption (T5), permutation LM (XLNet), denoising autoencoding (BART).
- **Causal LM** (GPT-family) is also an imputation task: predict next token given prefix — strictly easier to scale and serve, hence the dominant paradigm.
- **Why this is the most important idea in modern ML**: every frontier LLM is a giant next-token cloze model. All of post-training is dessert on top of this cake.

#### 19.2.4.2 Proxy / pretext tasks

- Self-defined transformations as labels: predict rotation (Gidaris), jigsaw, colorization, relative patch position.
- Largely **superseded** by contrastive and masked-modeling methods. Of historical interest.

#### 19.2.4.3 Contrastive tasks

- **Idea**: make augmented views of same example close; push different examples apart.
- **SimCLR** (19.2.4.4): two augmentations $x_1 = t_1(x), x_2 = t_2(x)$; minimize InfoNCE:
$$J = F(t_1(x))^\top F(t_2(x)) - \log \sum_{x^-_i \in N(x)} \exp[F(x^-_i)^\top F(t_1(x))]$$
- Interpretable as a **conditional energy-based model** $p(x_2|x_1) \propto \exp(F(x_2)^\top F(x_1))$, with the partition function approximated by negatives in the batch. Contrastive learning ≈ approximate MLE of an EBM.
- **MoCo**: memory bank of negatives via EMA encoder — handles small batches.
- **CLIP (19.2.4.5)**: contrastive between (image, text) pairs. 400M web pairs, batch ~32k, ResNet/ViT image encoder + transformer text encoder. The logits matrix $L_{ij} = I_i^\top T_j$, symmetric CE loss.
	- **Zero-shot classification**: convert class labels to prompts ("a photo of a {class}"), pick highest cosine.
	- **Prompt engineering** is *required* — "a photo of a {x}, a type of food" beats just "{x}" by a lot.
- **Why this matters**:
	- CLIP is the **backbone of every modern multimodal model**: LLaVA, GPT-4V's vision tower, Flamingo's frozen vision encoder, SigLIP (improved CLIP with sigmoid loss) used in Gemini and PaliGemma.
	- The **InfoNCE objective** reappears in: DPR/ColBERT (retrieval), SimCSE (sentence embeddings), text embeddings (E5, BGE, OpenAI text-embedding-3), preference learning (DPO is a kind of pairwise contrastive loss).
	- **JEPA** (LeCun's V-JEPA, I-JEPA): non-contrastive predictive SSL, currently fashionable — descends from this lineage but avoids negatives.

### 19.2.5 Domain adaptation

- Source domain $\mathcal{X}_s$, target $\mathcal{X}_t$, **same label set** $\mathcal{Y}$ but no target labels.
- **Domain-adversarial training** (Ganin): train features that fool a domain classifier. $\min_\phi \max_\theta$ objective with gradient reversal.
- **Modern relevance**: less direct, but conceptually present in:
	- **Cross-lingual transfer** (train on English data, evaluate on Swahili).
	- **Sim-to-real** for robotics policies.
	- **Distribution shift evaluation**, OOD generalization research.
	- For LLMs, "domain adaptation" usually just means continued pretraining on target corpus — no adversarial machinery.

---

## 19.3 Semi-supervised learning

- **Setting**: small labeled $\mathcal{D}_\ell$, large unlabeled $\mathcal{D}_u$, both sampled from same distribution.
- Modern version: you have a big pretrained model, some gold human-labeled data, and a much larger pile of un/weakly-labeled data — how to use the latter?

### 19.3.1 Self-training and pseudo-labeling — **comes back in a big way for LLMs**

- **Recipe**: train on labeled data → predict on unlabeled → treat predictions as labels → retrain.
- Two variants:
	- **Offline**: relabel all of $\mathcal{D}_u$, retrain to convergence, repeat. Used in Noisy Student (Xie+20), state-of-the-art ImageNet.
	- **Online**: pseudo-label per minibatch, train against immediately.
- **Confirmation bias**: model's own errors reinforce → use **selection metrics** (threshold on max prob, only keep confident ones).
- **Modern LLM analogues**:
	- **STaR (Zelikman 2022)**: generate chain-of-thought, keep ones that produce correct answer, fine-tune on the rationales. Self-training over CoTs.
	- **ReST / ReST-EM (DeepMind)**: rejection-sample model outputs scored by a reward model, fine-tune on accepted samples. Offline self-training.
	- **RFT (Rejection-sampling Fine-Tuning)**: Llama 2 used this for math.
	- **Self-Rewarding LMs** (Yuan+24): model both generates and grades; iteratively boostraps.
	- **DeepSeek-R1 cold-start → R1-Zero pipeline**: heavy use of self-generated reasoning traces filtered by correctness/format.
- **Confirmation bias is the central failure mode**: in LLM-land this manifests as mode collapse, reward hacking, hallucinated CoTs that happen to land on the right answer. Mitigations: verifier-in-the-loop (executable tests, math checkers), diverse sampling, KL constraint to ref policy (PPO/DPO).

### 19.3.2 Entropy minimization

- Loss: $\mathcal{L} = -\sum_c p_\theta(y=c|x) \log p_\theta(y=c|x)$ — minimized when output is one-hot.
- **Cluster assumption**: decision boundary should pass through low-density regions; entropy minimization pushes it there.
- **MixMatch** sharpens the target with temperature $1/T$ — interpolates between entropy min ($T=1$) and hard self-training ($T \to 0$).
- **Input-output MI** justification: maximize $\mathcal{I}(y; x)$, which decomposes into avg conditional entropy (push down) + marginal entropy (push up — encourages class balance).
- **Modern echo**: low-entropy outputs ↔ confident generations. Temperature-tuning at inference is dual to this. **Reward modeling** sometimes regularizes toward low-entropy peaked preferences.

### 19.3.3 Co-training and 19.3.4 Label propagation

- **Co-training**: two views, two models, each labels for the other when confident. Tri-Training extends.
- **Label propagation**: build similarity graph on $\mathcal{D}_\ell \cup \mathcal{D}_u$, propagate labels via transition matrix.
- **Niche relevance** today: graph-based propagation appears in **embedding-space pseudo-labeling**, RAG retrieval expansion. Co-training's spirit lives on in **ensemble distillation** and **multi-view RLHF** (multiple reward models).

### 19.3.5 Consistency regularization — important

- **Idea**: $p_\theta(y|x) \approx p_\theta(y|x')$ where $x' \sim q(x'|x)$ is a perturbed input. Penalize $\|p_\theta(y|x) - p_\theta(y|x')\|^2$ or $D_{KL}$.
- Full SSL objective: supervised CE on labeled + consistency on unlabeled.
- **VAT (virtual adversarial training)**: find worst-case perturbation $\delta = \arg\max_\delta D_{KL}(p(y|x) \| p(y|x+\delta))$ — adversarial example for consistency.
- **Modern parallels**:
	- **Adversarial robustness training** (Madry-style) — same math, different framing.
	- **Self-consistency decoding** (Wang+22): sample multiple CoTs, majority-vote — consistency-as-inference.
	- **Constitutional AI revision rounds**: model rewrites own output, both rewrites should agree with stated principles.
	- **DPO/IPO**: implicit consistency between policy and reference model via KL.

### 19.3.6 Deep generative models for SSL (starred / less important)

- **VAE-M2** (Kingma): $p(x,y) = p(y)\int p(x|y,z)p(z)\,dz$. Unlabeled data marginalizes over $y$ via inference net $q(y|x)$ that doubles as discriminative classifier.
- **GAN-SSL**: critic outputs C+1 classes (C real + 1 "fake").
- **Normalizing flows for SSL**: invertible $f: \mathcal{X} \to \mathcal{Z}$, latent prior is class-conditional Gaussian mixture.
- **Relevance**: largely sidelined for LLMs. Diffusion models for synthetic labeled data generation is the modern flavor, and that's more about augmentation than SSL.

### 19.3.7 Combining SSL + SSL + distillation (Chen+20c)

- **Pipeline**: SimCLR on unlabeled → fine-tune on small labeled → self-train (pseudo-label unlabeled) → **distill** into smaller student.
- **Distillation loss**: $\mathcal{L}(T) = -\sum_{x_i \in \mathcal{D}} \sum_y p^T(y|x_i; \tau) \log p^S(y|x_i; \tau)$, with temperature $\tau$ (softens teacher).
- **This is THE blueprint for the modern LLM ecosystem**:
	- **Pretraining** (Chinchilla, Llama 3 — self-supervised on web).
	- **Post-training** (SFT + RLHF/DPO — small labeled prefs).
	- **Synthetic data + distillation** (Llama-3.1 instruct distilled into 8B, DeepSeek-R1 → R1-Distill-7B/14B/32B/70B, Phi-3 fully distilled from GPT-4).
- **Distillation today**:
	- **Hard distillation**: train student on teacher's outputs (just SFT, basically).
	- **Soft distillation**: KL on full vocabulary distribution (used in MiniLLM, GKD).
	- **On-policy distillation**: student samples, teacher scores → student updates (better than offline because no distribution shift).
	- **Speculative decoding** is *inference-time* distillation: draft model proposes, large model verifies.
- **Why temperature matters in distillation**: high $\tau$ smooths the teacher's distribution, revealing "dark knowledge" — the relative scores between wrong answers carry useful signal.

---

## 19.4 Active learning — modest relevance, growing again

- **Goal**: query labels on the most informative points to minimize labels needed.
- Three flavors: query synthesis, **pool-based** (most studied), stream-based.
- **Related problems**: Bayesian optimization (find $x^* = \arg\min f$), experimental design (estimate $\theta$ with few queries).

### 19.4.1 Decision-theoretic — value of information

- $U(x) = \mathbb{E}_{p(y|x,\mathcal{D})}[\min_a (\rho(a|\mathcal{D}) - \rho(a|\mathcal{D}, (x,y)))]$. Expensive — one-step lookahead per candidate.

### 19.4.2 Information-theoretic — BALD

- Utility = info gain about parameters: $U(x) = \mathbb{H}(p(\theta|\mathcal{D})) - \mathbb{E}_{p(y|x,\mathcal{D})}[\mathbb{H}(p(\theta|\mathcal{D},x,y))]$.
- By symmetry of MI: $U(x) = \mathbb{H}(p(y|x,\mathcal{D})) - \mathbb{E}_{p(\theta|\mathcal{D})}[\mathbb{H}(p(y|x,\theta))]$.
- **Interpretation**: prefer $x$ where (1) the marginal predictive is uncertain AND (2) different model samples *confidently disagree*. This is **BALD** (Bayesian Active Learning by Disagreement, Houlsby+12) — pick examples where the *ensemble disagrees*, not just where any single model is unsure (which avoids inherently ambiguous examples).
- **Maximum entropy sampling** is just the first term — gets stuck on ambiguous examples.

### 19.4.3 Batch active learning — BatchBALD

- Selecting B examples greedily is suboptimal but **submodular** (Krause-Guestrin), so greedy is near-optimal (constant-factor).

### Modern relevance — **data selection for pretraining/post-training**

- The most live application of active learning ideas in 2026 is **data curation**, not online querying:
	- **DSIR** (Xie+23): importance sampling for pretraining data based on hashed n-gram features.
	- **DoReMi**: learn domain weights for pretraining via group DRO.
	- **Llama 3 data quality filters**, **FineWeb-Edu**: classifier-based filtering.
	- **Preference data selection for RLHF**: which prompts to collect human preferences on? BALD-style disagreement among reward model ensembles is used in practice.
	- **Active learning for evals**: which eval examples are most diagnostic? (Adaptive benchmarking.)
- **Reasoning-trace selection**: when training on synthetic CoTs, BALD-like disagreement filters identify the most informative ones.
- **Honest take**: classical pool-based active learning is rarely used directly in LLM labs — the bottleneck shifted from "labels are expensive" to "compute is expensive" and "data quality is everything". But the *concepts* (info gain, disagreement, diversity, submodular batch selection) all power modern curation pipelines.

---

## 19.5 Meta-learning — important conceptually, less in practice

- **Idea**: learn the learning algorithm. $\theta = A(\mathcal{D}; \phi)$, learn $\phi$ over a distribution of tasks.
- "Learning to learn" — Thrun & Pratt 1997.

### 19.5.1 MAML

- **Hierarchical Bayes view**: each task's params $\theta_j \sim p(\theta_j | \xi)$, meta-learn the prior $\xi$.
- **Empirical Bayes objective**: $\xi^* = \arg\max_\xi \frac{1}{J}\sum_j \log p(\mathcal{D}^j_\text{valid} | \hat\theta_j(\xi, \mathcal{D}^j_\text{train}))$.
- **Fast adaptation**: $\hat\theta_j$ = K gradient steps of inner SGD starting at $\xi$. Equivalent to MAP under Gaussian prior centered at $\xi$ with prior strength controlled by step count (Grant+18).
- Outer loop: backprop through inner loop's gradient steps → second-order gradients (or first-order approx, **FOMAML/Reptile**).

### Modern relevance — **ICL is the new meta-learning**

- **Hot take**: MAML-style gradient-based meta-learning largely died around 2020. **In-context learning** ate its lunch.
- **Why ICL is meta-learning in disguise**:
	- Pretraining on a giant mixture of sequences is *exactly* sampling tasks from a task distribution.
	- The transformer learns to do gradient-descent-like updates *in the forward pass* (Akyürek+22, von Oswald+23 — "transformers learn in-context by gradient descent").
	- Few-shot prompting "support set" = MAML's $\mathcal{D}_\text{train}^j$; "query" = $\mathcal{D}_\text{valid}^j$. No outer SGD because the meta-optimization happened during pretraining.
- **Practical implications**:
	- The reason GPT-3 *worked* is that next-token pretraining on diverse data implicitly meta-learns a universal adaptation algorithm.
	- Chain-of-thought, scratchpads, ReAct, etc. are all forms of *learned* inference-time adaptation.
- Other meta-learning ideas that *did* survive: **hyperparameter learning** (μP, learning-rate schedules), **AutoML** at industrial scale.

---

## 19.6 Few-shot learning

- **FSL**: predict from few labeled examples; **one-shot** = single example; **zero-shot** = none.
- **Eval**: C-way N-shot — distinguish C classes with N examples each.

### 19.6.1 Matching networks

- Learn a distance/attention metric on auxiliary data, use nearest-neighbor with support set:
$$p_\theta(y|x, \mathcal{S}) = \mathbb{I}\left(y = \sum_{n \in \mathcal{S}} a_\theta(x, x_n; \mathcal{S}) y_n\right)$$
- Attention kernel: $a(x, x_n; \mathcal{S}) = \frac{\exp(c(f(x), g(x_n)))}{\sum_{n'} \exp(c(f(x), g(x_{n'})))}$ where $c$ is cosine sim.
- Trained by episode sampling — sample task → sample support+query → optimize.
- **Spiritual ancestor of**: ICL (attention over examples), kNN-LM, retrieval-augmented generation.

### Modern relevance

- **Few-shot prompting** is the dominant FSL paradigm. "Here are 3 examples, now do this one." Works because ICL.
- **Retrieval-augmented few-shot**: pick the most relevant in-context examples from a pool (e.g., kATE selectors, BM25/dense retrievers feeding ICL).
- **Fine-grained classification** (the original motivating use case — face/product recognition) is still a real industrial problem; matching-net-style metric learning underlies face recognition systems.

---

## 19.7 Weakly supervised learning

- **Weak labels**: distribution over labels (label smoothing), bag-level labels (MIL), distant supervision.
- **Label smoothing**: target = $(1-\epsilon)$ on observed label + $\epsilon/(C-1)$ uniform. Regularization that prevents overconfidence; used everywhere in classifier training, in language modeling, in reward models.
- **Multi-instance learning**: only bag-level labels; bag positive iff any instance positive: $y_n = \vee_b y_{n,b}$.
- **Distant supervision**: harvest training data via heuristic rules from a knowledge base (e.g., relations from Wikidata + sentences mentioning the entities). Noisy but scalable.
- **Modern echoes**:
	- **RLHF/DPO**: preferences are weak labels on outcomes; we don't know *why* one response is preferred.
	- **Process reward models** vs **outcome reward models**: ORM is weakly supervised at the trajectory level (like MIL bag); PRM gives step-level labels (stronger).
	- **Synthetic data with heuristic verifiers**: distant supervision reborn — "any reasoning trace that ends at the right number is good", which is noisy but useful at scale (used in math/code training).
	- **Label noise handling**: matters a lot for preference data; methods like cDPO, IPO, KTO are partly motivated by acknowledging that preference labels are noisy.

---

## What to actually internalize for frontier LLM work (opinionated)

1. **Self-supervised pretraining (19.2.4) is the master idea**. Cloze tasks (causal LM in particular) scale better than any supervised signal. The chapter doesn't fully convey this — it was written just as GPT-3 was landing — but the seed is here.

2. **Adapters (19.2.2) → PEFT** is the most directly used concept. If you're doing any kind of fine-tuning research, you should reflexively reach for LoRA before full FT. It's nearly free; the only reason not to use it is when the last 1% matters.

3. **Self-training + distillation pipeline (19.3.7)** is *exactly* the modern post-training recipe. Internalize this loop: pretrain → fine-tune on gold → generate synthetic → filter → distill. Every frontier lab is iterating on variants of this.

4. **CLIP-style contrastive (19.2.4.5)** runs all of multimodal. If you work on VLMs, you live in this paradigm. SigLIP > CLIP now; same idea, sigmoid loss instead of softmax.

5. **Consistency regularization (19.3.5)** as a lens for KL constraints in RLHF/DPO, self-consistency decoding, and adversarial robustness. The math is the same.

6. **Meta-learning (19.5) is mostly subsumed by ICL** — but the framing ("learn algorithms, not just functions") is still useful when thinking about *why* ICL works and *what* limits its generalization.

7. **Active learning ideas (19.4) re-emerge as data curation**. BALD-style disagreement, info gain, submodular batch selection — these are tools you reach for when building data pipelines, not when annotating images.

8. **Don't bother memorizing**: deep generative SSL (VAE-M2, GAN-SSL, flow-SSL — 19.3.6), co-training, label propagation, supervised pretraining (19.2.3). Useful to know they exist; unlikely to use directly.

9. **What this chapter is missing** (because of 2020 vintage):
	- Instruction tuning + RLHF (the post-training paradigm that emerged in 2022).
	- DPO/IPO/KTO and the offline preference optimization family.
	- Process vs outcome reward models, verifiers, RL on reasoning (R1-style).
	- Mixture-of-experts as a parameter-efficient *scaling* method (cousin of adapters).
	- Test-time compute scaling (best-of-N, MCTS, o1-style chain-of-thought training).
	- Modern data quality filtering (FineWeb, DCLM, etc.).
- **Read this chapter as the *prequel*** to all of that. The concepts here are necessary background; the actual frontier moved on but never abandoned the foundations.
