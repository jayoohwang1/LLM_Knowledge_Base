# Chapter 22: Recommender Systems

- **Setting**: predict items (movies, books, ads, products) to users from past interaction signal + optional side info.
- **Data structure**: $Y_{ui} \in \mathbb{R}$ rating matrix, $M \times N$ users × items. Hugely sparse (~99% missing on Netflix).
- **Relational data**: $u, i$ indices are arbitrary symbols; what matters is the bipartite graph of interactions, not feature vectors.
- **Missingness assumption**: in this chapter assume MAR (missing at random), though in reality missingness is often informative (users avoid items they predict they'll dislike).
- **Why this chapter matters for the LLM-agent era**: every piece of the recsys playbook (latent factors, pairwise ranking, FMs, neural CF, bandit feedback loops) maps onto problems that LLM agent systems are starting to face but tend to re-invent badly. This entire field has spent 20+ years solving sparse-feedback / cold-start / exploration problems on huge interaction matrices — which is exactly what you get when you have millions of users × hundreds of tools × thousands of prompts × dozens of models.

---

## 22.1 Explicit Feedback

- **Explicit feedback**: user supplies a numerical rating (Likert 1-5, thumbs up/down).
- **Datasets**: Netflix Prize (100M ratings, retired for privacy), MovieLens (1M-25M), Jester, BookCrossing.
- **Netflix Prize**: $1M prize, claimed 2009 by BellKor's Pragmatic Chaos using an ensemble centered on matrix factorization.

### 22.1.2 Collaborative Filtering

- **Original CF idea**: predict via weighted average of similar users' ratings.
  - $\hat{Y}_{ui} = \sum_{u' : Y_{u',i} \neq ?} \text{sim}(u, u') \, Y_{u',i}$
  - **Similarity**: set overlap of rated items, e.g. Jaccard on $S_u = \{Y_{u,i} \neq ? : i \in \mathcal{I}\}$.
- **Failure mode**: sparsity kills similarity estimates — most user pairs share zero items.
- **Fix**: learn dense embeddings, compute sim in latent space (next section).

#### Agentic LLM Hypothesis

- **Tool-agent CF**: "users" = agents (or agent personas / system prompts), "items" = tools/APIs. A new agent can be bootstrapped by averaging the tool-success ratings of similar agents.
  - Sim could be cosine over system-prompt embeddings or behavioral fingerprints.
- **Prompt CF**: "users" = task instances, "items" = prompt templates / few-shot exemplars. Recommend a prompt template for a new task by similarity to historically-handled tasks.
- **Memory retrieval as CF**: if a user has a long-lived memory store, retrieving relevant past episodes is structurally CF — "users who had this context found these memories useful."
- The set-overlap sim formulation is a dead end in any high-dim agent setting — the embedding-based fix is what generalizes.

---

### 22.1.3 Matrix Factorization

- **Matrix completion view**: $\mathcal{L}(\mathbf{Z}) = \sum_{ij : Y_{ij} \neq ?} (Z_{ij} - Y_{ij})^2 = \|\mathbf{Z} - \mathbf{Y}\|_F^2$, ill-posed without rank constraint.
- **Low-rank assumption**: $\mathbf{Z} = \mathbf{U}\mathbf{V}^\top \approx \mathbf{Y}$, $\mathbf{U} \in \mathbb{R}^{M \times K}$, $\mathbf{V} \in \mathbb{R}^{N \times K}$.
- **Prediction**: $\hat{y}_{ui} = \boldsymbol{u}_u^\top \boldsymbol{v}_i$.
- **Optimization**:
  - **Full SVD** if matrix fully observed.
  - **ALS** (alternating least squares): fix $\mathbf{V}$, solve for $\mathbf{U}$ in closed form; alternate.
  - **SGD**: sample observed $(u,i)$, update along gradient.
- **Biases**: $\hat{y}_{ui} = \mu + b_u + c_i + \boldsymbol{u}_u^\top \boldsymbol{v}_i$ — captures per-user and per-item baselines (some users always rate harshly; some movies are universally popular).
- **Regularized SGD updates** (Simon Funk's seminal Netflix entry):
  - $b_u \leftarrow b_u + \eta (e_{ui} - \lambda b_u)$
  - $c_i \leftarrow c_i + \eta (e_{ui} - \lambda c_i)$
  - $\boldsymbol{u}_u \leftarrow \boldsymbol{u}_u + \eta (e_{ui} \boldsymbol{v}_i - \lambda \boldsymbol{u}_u)$
  - $\boldsymbol{v}_i \leftarrow \boldsymbol{v}_i + \eta (e_{ui} \boldsymbol{u}_u - \lambda \boldsymbol{v}_i)$
  - $e_{ui} = y_{ui} - \hat{y}_{ui}$.

#### 22.1.3.1 PMF (Probabilistic MF)

- **Generative**: $p(y_{ui} = y) = \mathcal{N}(y | \mu + b_u + c_i + \boldsymbol{u}_u^\top \boldsymbol{v}_i, \sigma^2)$.
- NLL equivalent to MSE loss; advantage is principled extensions (Poisson for counts, negative binomial, etc.).
- Symmetric version of exponential-family PCA.

#### Examples

- **Netflix K=2 visualization**: factor axes ended up roughly = (lowbrow vs. highbrow) × (indie/serious vs. blockbuster). Wizard of Oz sits at origin (average movie).
- **MovieLens K=50**: predicts genre coherence (action/film-noir clusters) even though genre was never given as input — pure CF picks up content structure.

#### Agentic LLM Hypothesis (high speculative potential)

- **User × LLM × Task tensor factorization**: 3-mode tensor of (user, model/agent variant, task). Latent factors give per-user agent affinity vectors and per-agent task-competency vectors. Personalized routing collapses to dot product in $K$-dim space.
  - Solves "which model for which user on which task" without enumerating combos.
  - $K=2$ visualization of agents could look like (helpful-vs-blunt) × (creative-vs-precise), analogous to the Netflix indie/blockbuster axes.
- **User-prompt MF for personalized agents**: $\boldsymbol{u}_u^\top \boldsymbol{v}_p$ predicts whether prompt template $p$ will satisfy user $u$; learn from thumbs-up signals. Bias terms $b_u$ (grumpy users) and $c_p$ (universally well-rated prompts) are exactly the right inductive bias — and probably more important than the interaction term in low-data regimes.
- **Latent skill factorization**: user-question × LLM-response correctness matrix factors into latent "skill" dims (math, code, reasoning chains). Could power adaptive evals — recommend the next eval question that maximizes information about a model's latent skill vector (active learning meets MF).
- **MF as cheap router**: instead of training a heavy gating MLP, learn $K=8$-dim user and task embeddings, route via $\arg\max_a \boldsymbol{u}^\top \boldsymbol{v}_a$. The Netflix lesson: bias terms + low-rank interaction beat fancy architectures with less data.
- **Per-tool baseline biases as a "tool popularity prior"**: most agent frameworks lack a clean way to say "this tool just works more often." Bias terms $c_i$ literally encode this; should be added to any tool-selection scoring head.
- **Implicit ALS for warm-start memory**: each new user starts with $\boldsymbol{u}_u$ initialized from a small calibration session, then ALS-updated as feedback arrives. Cheap personalization without finetuning.

---

### 22.1.4 Autoencoders (AutoRec)

- **Idea**: replace bilinear $\boldsymbol{u}^\top \boldsymbol{v}$ with a nonlinear AE over the rating vector.
- **Item-based AutoRec**: $f(\boldsymbol{y}_{:,i}; \boldsymbol{\theta}) = \mathbf{W}^\top \varphi(\mathbf{V} \boldsymbol{y}_{:,i} + \boldsymbol{\mu}) + \boldsymbol{b}$.
- Train only on observed entries:
  - $\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^N \sum_{u: y_{u,i} \neq ?} (y_{u,i} - f(\boldsymbol{y}_{:,i};\boldsymbol{\theta})_u)^2 + \frac{\lambda}{2}(\|\mathbf{W}\|_F^2 + \|\mathbf{V}\|_F^2)$.
- Beats fancier RBMs and LLORMA despite simplicity.

#### Agentic LLM Hypothesis

- **Behavior-vector autoencoder**: encode the sparse vector of a user's interactions across all agents/tools/prompts into a dense $\mathbb{R}^K$ "user state." Use it as a context token prefix for personalization at inference.
- **Tool-usage autoencoder for cold start**: when a new tool appears, encode the sparse vector of who has used similar tools to predict its likely adoption pattern.
- Less interesting than MF for agent settings — AE is usually a worse fit than MF when the matrix is genuinely low-rank, and the interpretability of latent factors is sacrificed.

---

## 22.2 Implicit Feedback

- **Implicit feedback**: positives only — user watched/clicked/purchased item $i$. Everything else is "unlabeled" (not necessarily negative).
- **Triplet formulation**: $y_n = (u, i, j)$ with $(u,i)$ positive and $(u,j)$ negative-or-unlabeled.

### 22.2.1 Bayesian Personalized Ranking (BPR)

- **Ranking loss**: model should score $i$ above $j$ for user $u$.
- **Bernoulli over preference**: $p(y_n = (u,i,j) | \boldsymbol{\theta}) = \sigma(f(u, i; \boldsymbol{\theta}) - f(u, j; \boldsymbol{\theta}))$.
- **MAP objective**: $\mathcal{L}(\boldsymbol{\theta}) = \sum_{(u,i,j) \in \mathcal{D}} \log \sigma(f(u,i) - f(u,j)) - \lambda \|\boldsymbol{\theta}\|^2$.
- **Dataset**: $\mathcal{D} = \{(u,i,j) : i \in \mathcal{I}_u^+, j \in \mathcal{I} \setminus \mathcal{I}_u^+\}$.
- **Negative sampling**: $|\mathcal{I} \setminus \mathcal{I}_u^+|$ is huge → subsample.
- **Pairwise item-preference matrix** $\mathbf{Y}_u$ has entries $+$, $-$, $?$ (unknown); only $+/-$ pairs contribute.
- **Hinge variant**: $\mathcal{L} = \max(m - f(u,i) + f(u,j), 0)$, margin $m \geq 0$.

#### Agentic LLM Hypothesis (very high speculative potential)

- **BPR is the structural cousin of DPO**. RLHF/DPO's pairwise sigmoid log-likelihood
  $\log \sigma(r_\theta(x, y_+) - r_\theta(x, y_-))$
  is *literally* BPR with $u = $ prompt, $i = $ chosen response, $j = $ rejected. The recsys community had this loss in 2009 — DPO (2023) is BPR with an implicit-reward parameterization.
  - Suggests latent inheritance opportunities: regularization tricks, negative-sampling strategies (popularity-weighted negatives → "hard negative" rejected responses), warm-start initialization, biases for "always-preferred" responses.
- **Pairwise tool preference learning**: when an agent chooses tool $i$ over tool $j$ and the trajectory succeeds, that's a BPR triplet. Cheap to log, no scalar reward needed.
- **Pairwise trajectory ranking** instead of scalar reward models: train a reward model to predict $\sigma(R(\tau_+) - R(\tau_-))$ from execution traces. Avoids reward-hacking pathologies of scalar regression.
- **CoT BPR**: when chain-of-thought $A$ beats CoT $B$ on a math problem (one is correct, one isn't), generate (problem, $A$, $B$) triplet. Train a reasoning-strategy ranker. Likely better than verifier-style scalar scoring because the noise is symmetric.
- **Implicit-from-edits**: when a user accepts response A but edits it into A', the pair (A', A) is a BPR-style positive/negative even without explicit thumbs.
- **Hinge variant for safety**: margin $m$ on (safe, unsafe) response pairs gives a hard separation guarantee — closer to constitutional-AI hard constraints than to soft RLHF.
- **Subsampled negatives matter**: the recsys lesson is that *which* negatives you sample dominates result quality. Popularity-corrected negatives (don't compare against responses no one would ever produce) is a transferable trick to RLHF — the agent-domain version is "don't rank against trajectories the policy would never sample."

---

### 22.2.2 Factorization Machines (FMs)

- **Motivation**: AutoRec is nonlinear but asymmetric in users/items; FMs are symmetric and accept arbitrary side features.
- **Linear FM model** (degree 2):
  - $f(\boldsymbol{x}) = \mu + \sum_{i=1}^D w_i x_i + \sum_{i=1}^D \sum_{j=i+1}^D (\boldsymbol{v}_i^\top \boldsymbol{v}_j) x_i x_j$
  - $\boldsymbol{x} \in \mathbb{R}^D$ is e.g. $[\text{one-hot}(u), \text{one-hot}(i), \text{side features}]$.
  - Each feature $i$ gets a latent $\boldsymbol{v}_i \in \mathbb{R}^K$; pairwise interactions are inner products.
- **Naive $O(KD^2)$, but rewriteable to $O(KD)$**:
  - $\sum_{i<j}(\boldsymbol{v}_i^\top \boldsymbol{v}_j) x_i x_j = \frac{1}{2}\sum_{k=1}^K \left[(\sum_i v_{ik} x_i)^2 - \sum_i v_{ik}^2 x_i^2\right]$.
  - For sparse $\boldsymbol{x}$, complexity $\to$ linear in nonzeros. One-hot user+item recovers original MF at $O(K)$.
- **Generality**: any loss (MSE for explicit, BPR for implicit) plugs in.
- **DeepFM**: $f(\boldsymbol{x}) = \sigma(\text{FM}(\boldsymbol{x}) + \text{MLP}(\boldsymbol{x}))$ — FM captures explicit memorization of feature interactions, MLP generalizes (the "wide and deep" idea).

#### Agentic LLM Hypothesis (very high speculative potential)

- **FMs are tailor-made for context-dependent tool use**. The agent input is naturally
  $\boldsymbol{x} = [\text{one-hot}(\text{user}), \text{one-hot}(\text{tool}), \text{one-hot}(\text{task-type}), \text{embeddings of recent actions}, \text{time-of-day}, \dots]$
  — sparse, high-D, heterogeneous. FM's pairwise inner-product structure
  - captures `(user × tool)`, `(tool × task-type)`, `(prev-tool × next-tool)` interactions in one model
  - shares parameters across rare combinations (the cold-start fix)
  - costs only $O(KD)$ for sparse $\boldsymbol{x}$ — way cheaper than a transformer pass for a fast routing layer.
- **FM-based tool router** as a pre-LLM gate: predict success probability of `(prompt-features, tool-id, agent-state)` tuple in microseconds; only call the LLM if the gate is uncertain.
- **FM for hyperparameter recommendation**: features = (model-id, prompt-style, temperature-bucket, max-tokens-bucket, task-domain); target = success rate. Each new model gets a $\boldsymbol{v}$ that lets you predict good hparams from sparse historical data.
- **DeepFM = wide-and-deep agent**: rare exact combos (user-X-on-tool-Y) handled by FM memorization; generalization to unseen combos handled by MLP. Mirrors hippocampus/cortex split.
- **FM with sequential side info** (Caser / Conv Sequence Embedding): treat last $M$ actions as an "image" → embed via convnet. Direct analog for agent action sequences — embed last $M$ tool calls, use convolution over the sequence to predict next tool. Way cheaper than an LLM head doing the same job.
- **Cross-feature explainability**: FM gives interpretable $(\boldsymbol{v}_i^\top \boldsymbol{v}_j)$ scores per feature pair — for an agent system, this is a tractable audit trail for "why did you pick this tool" without LLM-generated post-hoc rationalization.

---

### 22.2.3 Neural Matrix Factorization (NeuMF / NCF)

- **Two pathways combined**:
  - **GMF pathway** (bilinear): $\boldsymbol{z}_{ui}^1 = \mathbf{P}_{u,:} \odot \mathbf{Q}_{i,:}$ (elementwise product of separate user/item embeddings).
  - **MLP pathway**: $\boldsymbol{z}_{ui}^2 = \text{MLP}([\tilde{\mathbf{U}}_{u,:}, \tilde{\mathbf{V}}_{i,:}])$ (concatenation, separate embeddings).
- **Output**: $f(u,i;\boldsymbol{\theta}) = \sigma(\boldsymbol{w}^\top [\boldsymbol{z}_{ui}^1, \boldsymbol{z}_{ui}^2])$.
- Train with implicit binary labels or BPR.

#### Agentic LLM Hypothesis

- **Two-head router**: same "GMF + MLP" split for tool selection. The bilinear head gives sample-efficient learning on common (user, tool) pairs; MLP head exploits side info / context for hard cases.
- **Hybrid embedding learning**: use *separate* embedding tables for the two pathways (per the NeuMF paper) — agent systems tend to share one embedding for everything, which is empirically suboptimal here. Cheap thing to try.
- Less novel than FMs for this use case, but the architectural recipe (cheap bilinear + expressive deep, both with their own embeddings) is a generally-undervalued pattern in agent-routing literature.

---

## 22.3 Leveraging Side Information (Cold Start)

- **Cold-start problem**: new user / new item has no interaction history → CF fails.
- **Side info sources**:
  - Items: titles, images, prices, categorical metadata.
  - Users: query history (search), browsing/cookies (e-commerce), social graph.
- **FM framework naturally accepts side info** by expanding $\boldsymbol{x}$ beyond two one-hots.
- **Sequential / contextual signal**: order of last interactions matters. **Caser** (Convolutional Sequence Embedding Rec) embeds last $M$ items as a $M \times K$ "image," uses 2D convolution.

#### Agentic LLM Hypothesis

- **Cold start for new tools/models** is the practical bottleneck in agent ecosystems — every new MCP server / new LLM release needs a side-info-based prior on how it'll be used. Item embedding from textual description (tool docstring, model card) gives a free initialization.
- **User cold start from system-prompt embedding**: a new user's $\boldsymbol{u}_u$ initialized from an LLM embedding of their persona/system prompt. Then MF updates from there.
- **Caser-style action-sequence convolution** for next-action prediction in agent traces: extremely cheap, no transformer needed, captures local motifs like "search → fetch → summarize."
- **Tool docs as item content embedding**: the "image / title / cover" analog for tools is the API spec + examples. Encoder-only model on docs → tool embedding → seeds the recsys.

---

## 22.4 Exploration-Exploitation Tradeoff

- **Key observation**: recsys training data is a *consequence* of past recommendations — feedback loop. Items the system never showed have no data; items it always shows look great.
- **YouTube example**: slate of $k$ videos shown; user clicks one → positive signal. Counterfactual ("would they have liked video 100?") is unanswerable without exploration.
- **Solution direction**: RL / bandits, optimize long-term reward, take exploratory actions.
- Chapter explicitly punts details to the sequel book.

#### Agentic LLM Hypothesis (extremely high speculative potential)

- **The agent action loop is a bandit problem in disguise**. Every step picks an action (tool, sub-prompt, model, depth-of-search) from a large discrete set, observes a noisy success signal, and the training data for future picks is conditioned on past picks. Recsys's pathologies (popularity bias, feedback loops, filter bubbles) reappear:
  - **Popularity bias** → "the agent always uses web-search because the training set is dominated by traces where web-search was tried first."
  - **Filter bubble** → the agent narrows to a small repertoire of behaviors; never learns that an unused tool would have worked.
- **Thompson sampling for tool selection**:
  - Maintain a posterior over each tool's success rate (per task type).
  - At decision time, sample a value from each posterior, pick argmax.
  - Naturally interpolates explore/exploit, near-optimal regret, embarrassingly cheap.
  - Concrete recipe: Beta posterior on success/failure counts per (tool, task-cluster). Far better than $\epsilon$-greedy and *far* better than "always pick the LLM's first instinct."
- **Thompson sampling for model routing**: posterior over $p(\text{success} | \text{model}, \text{task-embedding})$; for each request, sample → route. Solves the model-bakeoff cold-start problem online.
- **Test-time compute as a multi-armed bandit**: arms = {single sample, k-sample majority vote, tree-of-thought depth d, MCTS budget b, verifier-guided beam}. Reward = quality / cost. Per-task-difficulty bandit picks the arm. Way more principled than current "always use the heaviest budget" or hand-tuned thresholds.
- **CoT strategy bandits**: arms = {direct answer, step-by-step, decomposed, self-critique, plan-then-execute, sketch-first-then-fill}. Online bandit over which CoT pattern to use, contextual on prompt embedding.
- **LinUCB / contextual bandit for tool gating**: linear model of $E[\text{success} | \text{context}, \text{tool}]$ with confidence bounds → pick action maximizing UCB. Lightweight, online-learnable, beats static routers.
- **Counterfactual / off-policy eval**: recsys community has 10+ years of IPS (inverse propensity scoring) and doubly-robust estimators for evaluating a new policy from logs collected under an old policy. Agent-system evaluation is structurally identical — every new prompt/system change needs counterfactual quality estimation from existing logs. Transferring IPS to "what if we'd used model B instead of model A on these 1M traces" is a free win.
- **Exposure / position bias** (recsys's deep wound): the *position* in the slate dramatically affects click. Direct analog: order in which tools are presented to the LLM in the function-calling list affects which gets chosen (positional bias is documented). Recsys's debiasing literature applies almost verbatim.
- **Long-horizon credit assignment** = the RL bridge. Recsys observation that "you may have to wait long to detect if a policy change is beneficial" is the *exact* pathology of agent eval — episode-level reward, sparse, delayed. Off-policy RL from logged agent trajectories is the natural framework. RecSim and Bot+13's framework on feedback loops are directly importable.

---

## Cross-Cutting Speculative Threads

- **The "everything is a recommendation problem" frame** is underrated in agent research:
  - Choosing a tool → recommending an item.
  - Choosing a model → recommending a service.
  - Choosing next CoT step → recommending an action.
  - Retrieving a memory → recommending a document.
  - Picking a few-shot exemplar → recommending an example.
  - Selecting next eval question → recommending a probe.
- **Latent factor models will be revived as cheap routing layers** in front of LLM calls; the $O(K)$ inner product is the right unit for billions of decisions/sec.
- **Pairwise loss generalization**: BPR's pairwise sigmoid is showing up everywhere (DPO, IPO, KTO, SimPO). Expect more recsys-style refinements: weighted BPR, popularity-aware negatives, hard-negative mining schedules.
- **Bandits are the missing layer in agent stacks**: between the LLM (slow, expensive, capable) and rule-based routing (cheap, dumb), there's room for online-learning bandit/contextual-bandit modules that the recsys community has perfected.
- **Cold start is the central practical problem** in fast-moving agent ecosystems with constantly new tools/models. Content-based bootstrapping from docs + LLM-generated synthetic interactions to warm up embeddings is the natural recipe.
- **Feedback loops are the central safety/reliability problem**: agents trained on their own logs will spiral into degenerate behavior just like recsys did into filter bubbles. The recsys community's debiasing tools (IPS, exposure modeling, randomized exploration) should be standard hygiene.
