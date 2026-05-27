# Chapter 16: Exemplar-based Methods

- **Big picture**: nonparametric / instance-based / memory-based learning — keep training data around, define model via similarity $s(x, x')$ or distance $d(x, x')$.
- **Why now (LLM agents)**: agents are increasingly defined by their memory store + retrieval policy, not by their weights. Every idea in this chapter is a candidate component of an agentic memory stack. Treat the LLM as the "Bayes-optimal classifier" sitting on top of a KNN / KDE / metric-learned substrate.

---

## 16.1 K Nearest Neighbor (KNN) Classification

**Technical summary**
- **Predictor**: $p(y=c|x, \mathcal{D}) = \frac{1}{K} \sum_{n \in N_K(x, \mathcal{D})} \mathbb{I}(y_n = c)$.
- **Hyperparameters**: $K$ (neighborhood size), distance $d(x, x')$ (often Mahalanobis $d_M(x, \mu) = \sqrt{(x-\mu)^T M (x-\mu)}$).
- **Bayes optimality**: as $N \to \infty$ KNN error $\le 2 \times$ Bayes error (very strong asymptotic guarantee).
- **K=1 = Voronoi tessellation**, zero train error, overfits.
- **As $K$ grows**: decision boundary smooths, train error rises, test error U-shaped.

### Agentic LLM hypothesis
- **Trajectory retrieval policies**: store agent trajectories $\tau_n$ with outcome labels (success/fail, reward, tool-used). At decision time, retrieve top-$K$ most similar past trajectories and let LLM vote / chain-of-thought over them. Effectively KNN-classification over a trajectory manifold.
- **K as exploration knob**: small $K$ = exploit best-matching past episode (high variance, sensitive to noisy memory entries). Large $K$ = ensemble over similar episodes, smoother but slower adapting. Could be **annealed** during agent rollouts (small $K$ once confident, large $K$ when uncertain).
- **Bayes-optimal floor**: the $2\times$ Bayes bound suggests that *for fixed embedding*, even unboundedly large LLMs querying an unboundedly large memory bank cannot beat $2\times$ optimal. This bounds what can be gained by scaling RAG corpus alone — the embedding/metric is the bottleneck, not the corpus size.
- **Voronoi as policy partition**: agent policies become piecewise constant over state space partitioned by stored exemplars. Possibly useful as an interpretability tool: "which past episode is my agent currently 'living inside'?"

### 16.1.2 Curse of dimensionality

**Technical summary**
- To cover fraction $p$ of unit cube uniformly, need edge length $e_D(p) = p^{1/D}$.
- $D=10, p=0.01 \Rightarrow e=0.63$ — "neighbors" span 63% of each axis, no longer local.
- **Volume grows exponentially with $D$** $\Rightarrow$ nearest neighbors are not actually near.

### Agentic LLM hypothesis
- **Huge speculative payoff here**. Modern agent memory uses 768–4096-d embeddings. The curse implies that vanilla cosine-NN retrieval over a million-entry agent memory is *not* actually retrieving "nearby" memories in any geometric sense — all distances concentrate.
- **Implication 1 — Subspace metrics for agents**: agents probably need *task-conditioned* metrics that only attend to task-relevant subspace (e.g., for tool use, only the "what does this tool do" subspace, not the surface text). Connects directly to §16.2 metric learning.
- **Implication 2 — Hierarchical memory**: instead of a flat embedding store, use coarse-to-fine retrieval (think Matryoshka embeddings, IVF + PQ). Each level operates in lower effective dimension.
- **Implication 3 — Episodic vs semantic memory split (cognitive science analog)**: high-D embeddings should be reserved for episodes; semantic facts should be compressed into lower-D structured form. Curse predicts naive episodic recall fails at scale.
- **Implication 4 — Why pure embedding RAG plateaus**: empirical observation that RAG quality saturates with corpus size is partly the curse. Need either parametric assumptions (LLM reranker) or learned task-specific metrics.

### 16.1.3 Speed/memory (ANN, LSH, k-d trees, FAISS)

**Technical summary**
- Exact NN intractable above ~10-d $\Rightarrow$ approximate NN.
- **Partitioning**: k-d tree (axis-aligned splits), clustering (anchor points / IVF).
- **Hashing**: **locality-sensitive hashing (LSH)** — designs hash $h$ s.t. $\Pr[h(x) = h(x')] \propto s(x, x')$. Learned variants now standard.
- **FAISS** for production scale.

### Agentic LLM hypothesis
- **LSH for tool selection**: tools live in a discrete vocabulary but their *semantics* are continuous (descriptions + past invocation contexts). LSH bucket = "tool affordance class". At inference, hash the agent's current intent, retrieve candidate tools from the bucket, then LLM does fine-grained selection. $O(1)$ tool routing for agent suites with thousands of tools.
- **LSH-coded memory keys as discrete tokens**: feed hash codes back as tokens into the LLM context, letting the model do symbolic reasoning over memory locations (think Differentiable Neural Computer reborn). Hashes are short and tokenizable; embeddings are not.
- **Learned hash functions trained on agent outcomes**: replace cosine-LSH with LSH whose collisions correlate with *reward-equivalent* states. The hash buckets become a learned discretization of agent state space — basically learned options or skills.
- **Coreset pruning of agent memory**: just as sparse kernel machines keep only critical exemplars, agent memory should prune to "support trajectories" — those that change downstream behavior. Connects to MBR-style trajectory filtering.

### 16.1.4 Open set recognition

**Technical summary**
- Closed world assumption breaks in reality. Need **novelty / OOD detection**, **incremental / continual / life-long learning**, **few-shot** classification.
- KNN naturally suits this: just add the new exemplar to gallery, do **face verification**-style matching, declare OOD if no match within threshold.
- **Entity resolution / entity linking**, **multi-object tracking (random finite sets)** are open-world variants.

### Agentic LLM hypothesis
- **Massive speculative payoff**. Agents that operate for days/weeks *must* handle open-set: new tools appear, new users, new failure modes.
- **Exemplar-based continual learning**: instead of fine-tuning weights (catastrophic forgetting), append new exemplars to memory. The KNN bound says we are at worst $2\times$ Bayes — so a memory-only agent can be theoretically competitive without weight updates.
- **Novelty as gating signal for human-in-the-loop**: if no exemplar within distance $\tau$, escalate. This is exactly the "I don't know, ask" capability current LLMs lack — calibrated abstention via distance threshold over agent memory.
- **Entity resolution for agent memory hygiene**: "Is this user the same as the one I talked to last week?" "Is this tool error the same as that one?" Without entity resolution, agent memory accumulates duplicates that distort KNN votes.
- **Random finite set tracking for multi-task agents**: agent juggling N tasks needs to track which "task object" each observation belongs to — formally the same as multi-object tracking. Underexplored framing for agentic planning.

---

## 16.2 Learning Distance Metrics

**Technical summary**
- Mahalanobis $d_M(x, x') = \sqrt{(x-x')^T M (x-x')}$ with $M \succ 0$.
- For high-D / structured inputs: learn embedding $e = f(x)$, compute distance in embedding space $\Rightarrow$ **deep metric learning (DML)**.

### Agentic LLM hypothesis (umbrella)
- **The metric is the policy.** In RAG / agentic setups, the retriever's metric defines what "similar context" means, which determines what the LLM sees, which determines what it does. Learning the metric end-to-end against agent reward is the highest-leverage research direction this chapter suggests.

### 16.2.1 Linear / convex methods

#### 16.2.1.1 Large margin nearest neighbors (LMNN)
- **Pull**: $\mathcal{L}_{\text{pull}}(M) = \sum_i \sum_{j \in N_i} d_M(x_i, x_j)^2$ (target neighbors close).
- **Push**: hinge on impostors $\mathcal{L}_{\text{push}}(M) = \sum_i \sum_{j \in N_i} \sum_l \mathbb{I}(y_i \ne y_l)[m + d_M(x_i, x_j)^2 - d_M(x_i, x_l)^2]_+$.
- Total $(1-\lambda)\mathcal{L}_{\text{pull}} + \lambda \mathcal{L}_{\text{push}}$, convex via SDP; or factor $M = W^T W$ and SGD (nonconvex but low-rank).

#### 16.2.1.2 Neighborhood components analysis (NCA)
- **Stochastic 1-NN**: $p_{ij}^W = \frac{\exp(-\|Wx_i - Wx_j\|^2)}{\sum_{l \ne i} \exp(-\|Wx_i - Wx_l\|^2)}$.
- Maximize expected LOO 1-NN accuracy $J(W) = \sum_i \sum_{j: y_j = y_i} p_{ij}^W$. Gradient on $W$.
- **NCA is supervised SNE** (foreshadows t-SNE).

#### 16.2.1.3 Latent coincidence analysis (LCA)
- Generative: $z = Wx + \text{noise}$, $p(y=1|z, z') = \exp(-\|z-z'\|/(2\kappa^2))$.
- Fit by EM, closed-form E-step, weighted least squares M-step. More stable than NCA's SGD.

### Agentic LLM hypothesis
- **LMNN for agent memory**: target neighbors = past trajectories with same outcome. Impostors = trajectories that *looked* similar but had different outcomes. Train $M$ so that "looks similar AND ended similarly" ≠ "looks similar but ended differently". This is essentially RLHF-style reranking but for retrieval, not generation.
- **NCA as differentiable retriever**: the softmax form $p_{ij}$ is exactly the attention pattern. Modern attention-as-retrieval (Memorizing Transformer, kNN-LM) is rediscovering NCA. The chapter cite [Gol+05] predates these by 15 years.
  - **Concrete project**: train embeddings with NCA-style objective using *agent task success* as the supervisory label, not "same class".
- **LCA's EM stability**: an underused observation — EM with closed-form E-step doesn't need step sizes. For online agent memory updating where each new episode triggers a retriever update, EM-style updates may beat SGD which needs careful LR tuning per task.
- **Margin $m$ as confidence calibration**: in LMNN, $m$ controls how decisively distinct classes must separate. For agents, $m$ could be tied to risk tolerance — high-stakes domains demand bigger margins between "do tool A" and "do tool B" exemplar clusters.

### 16.2.2 Deep metric learning (DML)

**Technical summary**
- Learn $e = f(x; \theta) \in \mathbb{R}^L$, normalize $\hat e = e / \|e\|_2$ (everything lives on hypersphere).
- Distances: $\|\hat e_i - \hat e_j\|^2 = 2 - 2 \hat e_i^T \hat e_j$ (Euclidean and cosine equivalent on sphere).
- Push similar pairs close, dissimilar far. Works without labels if pairs are otherwise defined (self-supervised).
- **Caveat**: many recent DML claims don't replicate vs. old baselines [MBL20; Rot+20].

### Agentic LLM hypothesis
- **Hypersphere geometry for agent state**: forcing all agent state embeddings onto a sphere bounds the partition function and gives well-behaved similarities — useful for stable retrieval under distribution shift.
- **Self-supervised DML on agent trajectories**: define similar pairs as (state, next-state) or (state, state+small-perturbation), dissimilar as random trajectories. Yields embedding where local dynamics are captured — useful for model-based agent planning.
- **Recent DML claims overstated**: cautionary tale for current RAG benchmarks. Old simple methods (cross-entropy + last-layer features) often win. Likely true for agent memory too — fancy contrastive objectives may not beat well-tuned cosine over LM embeddings.

### 16.2.3 Classification losses
- Train a $C$-way classifier; reuse penultimate features as embedding. $O(NC)$, scales. Caveat: only learns decision boundary, not within-class compactness.
- **Agent angle**: pre-train embedding with massive auxiliary classification (e.g., "which tool was used here?"), use penultimate features for retrieval. Cheap and scalable.

### 16.2.4 Ranking losses

#### 16.2.4.1 Pairwise / contrastive / Siamese
- $\mathcal{L} = \mathbb{I}(y_i=y_j) d(x_i, x_j)^2 + \mathbb{I}(y_i \ne y_j) [m - d(x_i, x_j)]_+^2$.
- $O(N^2)$ pairs. Same encoder both branches = **Siamese**.

#### 16.2.4.2 Triplet loss
- $\mathcal{L} = [d_\theta(x, x^+)^2 - d_\theta(x, x^-)^2 + m]_+$.
- $O(N^3)$ naively; minibatch in practice.

#### 16.2.4.3 N-pairs / InfoNCE / NT-Xent
- $\mathcal{L} = -\log \frac{\exp(\hat e^T \hat e^+)}{\exp(\hat e^T \hat e^+) + \sum_k \exp(\hat e^T \hat e^-_k)}$.
- **Same as InfoNCE (CPC), NT-Xent (SimCLR)**. Temperature scales sphere radius.

### Agentic LLM hypothesis
- **Triplets from agent self-play**: anchor = current step, positive = step from a successful trajectory in similar state, negative = step from a failing trajectory. This is "metric learning from RL trajectories" — could replace value-function-based ranking in offline RL with simple triplet objective.
- **InfoNCE for tool retrieval**: anchor = task description, positive = correct tool, negatives = wrong tools. This is the standard dense retriever objective. The chapter notes the equivalence of N-pairs / InfoNCE / NT-Xent — a unifying framework that says current retriever training is just $N$-way classification in disguise.
- **Temperature as exploration**: NT-Xent temperature $\tau$ flattens/sharpens. For agents, low $\tau$ = sharp retrieval (commit), high $\tau$ = diffuse retrieval (consider many options). Could be adapted per-step like Thompson sampling.
- **Pairwise vs triplet incomparability**: chapter notes pairwise objectives leave +/- magnitudes uncalibrated. Same issue plagues RLHF reward models (Bradley-Terry). Hint: triplet/list-wise losses may give more stable agent reward models.

### 16.2.5 Speedups

#### 16.2.5.1 Hard / semi-hard negative mining
- **Hard negative**: $d(x_a, x_n) < d(x_a, x_p)$ with $y_n \ne y_a$.
- **Semi-hard**: $d(x_a, x_p) < d(x_a, x_n) < d(x_a, x_p) + m$. (FaceNet uses this.)
- In practice, mined within minibatch $\Rightarrow$ need large batches OR maintain candidate pool that's continually refreshed as metric evolves.

#### 16.2.5.2 Proxy methods
- Each class represented by a learned proxy $c_c \in \mathbb{R}^L$. Distance to proxy is a surrogate. $O(NP)$ where $P \sim C$.
- Soft triple loss generalizes (multiple prototypes per class).

#### 16.2.5.3 Upper-bound on triplet loss via proxies
- Triangle inequality gives $\|\hat e_i - \hat e_j\| \le \|\hat e_i - c_{y_i}\| + \|\hat e_j - c_{y_i}\|$ etc.
- Upper-bounds reduce $O(N^3)$ to $O(NC)$.
- Centroids should be orthonormal (one-hot) to make bound tight.

#### 16.2.5.4 Inter/intra-class diagnostics
- $\pi_{\text{intra}}$ avg same-class distance, $\pi_{\text{inter}}$ avg centroid distance. Ratio $\pi_{\text{intra}} / \pi_{\text{inter}}$ predicts retrieval quality.

### Agentic LLM hypothesis
- **Hard negative mining for agent training**: high-leverage. Mine "trajectories that look like success but failed" — these are the impostors agents most need to disambiguate. Resembles failure-case mining in autonomous driving (rare events).
  - **Curriculum**: start with easy negatives (random) then ramp to semi-hard then hard, paralleling how human apprentices learn.
- **Proxies = learned skill prototypes**: a proxy per class = one prototype per skill/task type. Agent memory could collapse from millions of episodes to $C$ prototypes per task family, queried in $O(C)$.
  - **Caveat**: proxies lose episode-specific detail. Hybrid: proxy for routing + episode retrieval within proxy cluster.
- **Orthogonal proxies for tool/intent vocabulary**: one-hot-like proxies = maximum-margin separation. Concrete suggestion: pre-allocate orthogonal anchor directions in embedding space for major tool categories, fine-tune embeddings to align there.
- **$\pi_{\text{intra}}/\pi_{\text{inter}}$ as memory quality metric**: cheap unsupervised diagnostic — does this agent's memory have well-separated clusters? Run periodically as memory grows; if ratio degrades, trigger reclustering / prototype refresh.

### 16.2.6 Other DML tricks
- Minibatch construction: pick $B/n$ classes, $N_c=2$ each is a decent default.
- Pretrain on ImageNet, fine-tune with DML loss.
- Data augmentation as the only way to make similar pairs in self-supervised settings.
- **Spherical embedding constraint (SEC)**: regularize so all embeddings have similar norm. Acts like batchnorm for DML.

### Agentic LLM hypothesis
- **Episode augmentation**: paraphrase past agent observations / replay with perturbed inputs to generate "similar pairs" for self-supervised metric updates. The agent's memory then learns invariances on the fly.
- **SEC for embedding stability across long horizons**: as agent embeddings drift over weeks of operation, norm regularization could prevent the magnitude collapse problem in continual learning systems.

---

## 16.3 Kernel Density Estimation (KDE)

**Technical summary**
- Nonparametric *generative* model $p(x)$.
- **Density kernel** $\mathcal{K}: \mathbb{R} \to \mathbb{R}_+$ with $\int \mathcal{K} = 1$, even.
- Common kernels: boxcar, Gaussian ($\mathcal{K}(x) = (2\pi)^{-1/2} e^{-x^2/2}$), Epanechnikov ($\propto (1-x^2) \mathbb{I}(|x| \le 1)$, optimal MSE), tri-cube.
- **Bandwidth $h$**: $\mathcal{K}_h(x) = \frac{1}{h} \mathcal{K}(x/h)$. Controls complexity — too small = spiky, too large = oversmoothed.
- **RBF generalization** to $\mathbb{R}^D$: $\mathcal{K}_h(x) \propto \mathcal{K}_h(\|x\|)$.

### 16.3.2 Parzen window estimator
- $p(x|\mathcal{D}) = \frac{1}{N} \sum_n \mathcal{K}_h(x - x_n)$.
- One kernel per data point. No fitting (except bandwidth). Stores all data $\Rightarrow$ memory + slow eval.

### 16.3.3 Bandwidth selection
- For 1d Gaussian data: $h = \sigma (4/(3N))^{1/5}$ (Silverman's rule).
- Robust: $\hat\sigma = 1.4826 \cdot \text{MAD}$.
- In $D$ dimensions: per-dim $h_d$, then $h = (\prod h_d)^{1/D}$.

### 16.3.4 From KDE to KNN classification
- **Balloon KDE**: instead of fixed $h$, grow volume $V(x)$ around test point until $K$ points enclosed.
- Class conditional $p(x|y=c) = N_c(x) / (N_c V(x))$.
- Class prior $N_c / N$, plug into Bayes $\Rightarrow$ recovers $p(y=c|x) = K_c / K$, i.e., KNN classification.
- **Unification**: KNN is a balloon-bandwidth KDE-based generative classifier.

### Agentic LLM hypothesis (high speculative potential)
- **KDE for agent uncertainty estimation** — this is the most underexploited idea in the chapter for agents.
  - At inference time, compute $p(x_{\text{query}}|\mathcal{D}_{\text{memory}}) = \frac{1}{N} \sum_n \mathcal{K}_h(x_{\text{query}} - x_n)$ in embedding space. **Low density = OOD = abstain or escalate**.
  - This is a principled alternative to "confidence via token probability" which is well-known to be miscalibrated.
  - **Per-class density**: $p(x|\text{task}=t)$ for each task type $t$; classify the user's request as "the agent has seen this kind of thing before" or "novel".
- **Adaptive bandwidth = adaptive caution**: in dense regions of memory (familiar territory), small $h$ — agent trusts specific past episodes. In sparse regions, large $h$ — agent generalizes more broadly or asks for help.
- **KDE for value uncertainty in agentic RL**:
  - Estimate $p(s, a)$ over visited state-action pairs.
  - Density-based **pseudo-counts** for exploration bonuses (this idea exists in deep RL; mostly forgotten in LLM-agent literature).
  - Reward shaping: $r' = r + \beta / \sqrt{p(s,a)}$ encourages exploration of low-density agent states.
- **Balloon KDE for adaptive context length**: at inference, "grow" the context retrieval radius until $K$ relevant docs are gathered. Differs from fixed top-$K$ — automatically adjusts to query difficulty. Could yield better tradeoffs than fixed top-K RAG.
- **Bandwidth via MAD over distance distribution**: $\hat h \propto \text{MAD}(\{d(x_q, x_n)\})$ per query — a robust, data-driven retrieval radius.

### 16.3.5 Kernel regression (Nadaraya-Watson)

**Technical summary**
- $\mathbb{E}[y|x, \mathcal{D}] = \sum_n y_n w_n(x)$, where $w_n(x) = \frac{\mathcal{K}_h(x - x_n)}{\sum_{n'} \mathcal{K}_h(x - x_{n'})}$.
- Just *softmax over distances* weighting stored targets — **this is attention**.
- **Connection to GP regression** (Ch 17).

#### 16.3.5.2 Variance estimator
- $\mathbb{V}[y|x, \mathcal{D}] = \sigma^2 + \sum_n w_n(x) y_n^2 - \mu(x)^2$.
- Gives epistemic-ish uncertainty for free.

#### 16.3.5.3 Locally weighted regression (LOWESS / LOESS)
- Fit local linear model around each test point: $\mu(x) = \min_\beta \sum_n [y_n - \beta^T \phi(x_n)]^2 \mathcal{K}_h(x - x_n)$.
- Used in scatterplot trend lines.

### Agentic LLM hypothesis (huge speculative potential)
- **Nadaraya-Watson IS attention**: $w_n(x)$ is softmax over $-d(x, x_n)/h$, $\sum w_n y_n$ is the value aggregation. Transformers are doing N-W kernel regression with learned features.
  - **Implication**: every result about kernel regression (bias, variance, bandwidth selection, asymptotic rates) applies to attention. There's a whole 1960s–80s nonparametric statistics literature waiting to be re-applied to attention analysis.
  - **Concrete project**: derive variance estimators for transformer attention using §16.3.5.2 formulas $\Rightarrow$ free uncertainty estimates without ensembling.
- **N-W variance for agent confidence**: each forward pass already computes attention weights. With one extra pass over $y_n^2$, get a free variance estimate. Cheap calibration for agentic decisions.
- **Locally weighted regression for in-context fine-tuning**: the LLM, given a few in-context examples, is doing something LOWESS-like — fitting a local model around the query. This frames ICL as nonparametric local regression in representation space.
  - **Bandwidth = effective context length / temperature**. Tuning $h$ analogous to tuning how broadly the model attends.
- **Local linear correction over retrieved exemplars**: after retrieving $K$ similar past episodes, fit a local linear model (over outcomes) to predict the outcome of the current query, weighted by similarity — a more principled aggregation than majority vote.
- **Curse of dimensionality bites kernel regression too**: N-W convergence rate is $O(N^{-2/(4+D)})$ — for $D=768$ embeddings, exponentially many samples needed. Suggests that LLM attention works only because the *effective* dimension of relevant features is much smaller than $D$ (sparse / low-rank structure).
  - **Research question**: can we estimate the effective dimension of an agent's task and use that to set retrieval policy?

---

## Cross-cutting Speculative Themes

- **Agent memory is just KNN/KDE with a learned metric**. Three knobs: (i) embedding $f_\theta$, (ii) similarity/kernel, (iii) bandwidth/K. Each independently tunable; this chapter gives the theory for all three.
- **Curse of dimensionality is the silent killer of embedding-based agent memory**. Either (a) reduce effective dim via metric learning / Matryoshka, (b) introduce parametric structure (LLM as reranker), or (c) accept that "nearest" memory isn't really near and instrument accordingly.
- **Attention $\equiv$ Nadaraya-Watson kernel regression** is the conceptual rosetta stone. All the kernel-regression machinery (variance, bandwidth, LOWESS) is transferrable to transformer analysis.
- **Exemplar-based meta-learning / fast adaptation**: KNN naturally few-shot. An agent with a continually growing exemplar store + good learned metric ≈ MAML without gradient updates. Likely a more scalable path to long-horizon adaptation than weight-update-based continual learning.
- **Calibrated abstention via KDE density**: principled "I don't know" signal — measure $p(x_q | \text{memory})$ and threshold. Underused in current LLM agent stacks.
- **LSH-coded memory as discrete tokens for symbolic agent reasoning**: bridge between continuous retrieval and symbolic planning.
- **Hardest-negative mining for agent training**: failure-similar-to-success episodes are the highest-information training data. Mirrors how humans learn from near-misses.
- **Open-set is the actual problem**: agents in the wild are perpetual open-set learners. The exemplar framework natively supports this; weight-update-based learning does not.
