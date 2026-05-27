# Chapter 21: Clustering

**Framing**: Unsupervised structure discovery. Two input modes - feature vectors $\mathcal{X} = \mathbb{R}^D$ or pairwise dissimilarity matrices $D_{ij} \geq 0$. Evaluation is the persistent thorn (Jain & Dubes quote: "black art ... only accessible to those true believers who have experience and great courage").

**Why care for LLM agents?** Agents generate massive trajectory/CoT/tool-use logs. We have no labels, no clear reward decomposition, and we want to discover skills/concepts/modes/subtasks. Clustering is the natural toolkit but most agent research currently uses ad-hoc embedding + KMeans. The richer machinery here (HAC, GMMs, spectral, biclustering, DP mixtures) is largely untapped territory.

---

## 21.1 Evaluating Clustering Output

- **Internal vs external criteria**: internal (cohesion/separation) is limited; external uses ground-truth labels.
- **Purity**: $p_i = \max_j p_{ij}$ where $p_{ij} = N_{ij}/N_i$, overall $\sum_i \frac{N_i}{N} p_i$.
  - Trivially 1 by putting each point in its own cluster - no penalty on $K$.
- **Rand index**: $R = (TP+TN)/(TP+FP+FN+TN)$ over pairs.
  - **Adjusted Rand**: $(R - E[R])/(\max R - E[R])$ corrects for chance.
- **Mutual information** between partitions $U,V$: $\mathbb{I}(U,V) = \sum_{ij} p_{UV}(i,j) \log\frac{p_{UV}(i,j)}{p_U(i)p_V(j)}$.
  - **NMI** $= \mathbb{I}(U,V)/((\mathbb{H}(U)+\mathbb{H}(V))/2)$, in $[0,1]$.
  - **Variation of information** is a metric variant.

### Agentic LLM hypothesis
- **External-label proxy for unlabeled agent trajectories**: use task success/reward as the "label" and measure NMI between cluster id and success. Skills that meaningfully predict outcomes will have high NMI - automatic skill validation without labels.
- **Rand index over rollouts**: pairs of trajectories that the human/judge LLM considered "same strategy" vs the clustering's pairwise assignments. Cheap evaluation of unsupervised strategy discovery.
- **Variation of information as a regularizer**: when comparing two different LLM clustering schemes (e.g., embeddings from different layers or different judge LLMs), VI gives a stable distance in the space of partitions - useful for ensembling clusterings of agent behavior.
- **Multi-judge purity**: have $K$ judge-LLMs label trajectories, then measure purity across judges as a noisy-label consistency metric. Clusters with high cross-judge purity are likely capturing real structure rather than judge idiosyncrasy.

---

## 21.2 Hierarchical Agglomerative Clustering (HAC)

- Input: $N\times N$ dissimilarity $D_{ij}$. Output: **dendrogram** (binary tree).
- Bottom-up: start with $N$ singletons; greedily merge most similar pair.
- $O(N^3)$ naively, $O(N^2 \log N)$ with priority queue, $O(N^2)$ for single link.
- **Linkage variants**:
  - **Single link** (nearest neighbor): $d_{SL}(G,H) = \min_{i\in G, i'\in H} d_{i,i'}$. Tree = minimum spanning tree. Can violate **compactness** (chaining - large diameters).
  - **Complete link** (furthest neighbor): $\max$ instead of $\min$. Compact, small-diameter clusters.
  - **Average link**: average of all pairs. Preferred in practice; not invariant to monotonic transforms of $d$ (unlike single/complete).
- **Extensions**: parallel sub-cluster construction [Mon+21]; online HAC that reconsiders past merges - useful for streaming **entity discovery**.

### Agentic LLM hypothesis
- **Hierarchical skill/task taxonomies**: agglomerate over trajectory embeddings to induce a tree of skills. Cut at different heights $\Rightarrow$ skill granularity knob. E.g., top-level "code", "web", "math"; deeper "regex code", "sql code"; deeper still "select-with-join-sql". Library can be queried at the appropriate granularity for the task at hand.
- **Online HAC for streaming entity/concept discovery**: directly maps to LLM agent memory. As an agent observes new entities (people, files, APIs), online HAC clusters mentions into entities with the ability to *reconsider* past merges when contradictory evidence arrives. This is exactly the "should I split this memory record?" problem in agent long-term memory.
- **Single link = chaining**: useful when you *want* chains. E.g., trajectory connectivity in a sparse-reward environment - single link finds the maximum-spanning structure linking near-solutions. Hand off to a planner: "follow this chain of trajectories".
- **Average link for behavioral cloning data curation**: cluster demonstrations, cut at moderate height, train one specialist per cluster (mixture-of-experts BC). Average link's compromise nature avoids the chaining pathology of single link while remaining cheap.
- **Dendrogram as curriculum scaffold**: traverse dendrogram leaves-to-root for curriculum learning. Train RL agent on within-cluster tasks first, then merge clusters and re-train at coarser level. Curriculum structure emerges from data, not from human design.
- **Speculative**: HAC over *tool sequences* used by agents - linkage criteria over sequence-edit-distance. Discovers macro-operators bottom-up.

---

## 21.3 K-means

- Alternating: $z_n^* = \arg\min_k \|x_n - \mu_k\|^2$ then $\mu_k = \frac{1}{N_k}\sum_{n: z_n=k} x_n$.
- Minimizes **distortion** $J(M,Z) = \|X - ZM^\top\|_F^2$. Non-convex, sensitive to init.
- $O(NKT)$. Voronoi tessellation of input space.

### 21.3.3 Vector Quantization
- Replace each $x_n$ with codebook index $z_n \in \{1..K\}$. Encoding cost $O(N\log_2 K + KDB)$ bits.
- Deep tie to density estimation / lossy compression.

### 21.3.4 K-means++
- Pick first center uniformly, subsequent centers w.p. $\propto D_{t-1}(x)^2$ (squared distance to closest existing center). **Farthest point heuristic**. Guaranteed $O(\log K)$-competitive with optimal.

### 21.3.5 K-medoids
- Center = actual datapoint with min average dissimilarity to cluster (medoid). Works on arbitrary $D_{ij}$ - no $\mathbb{R}^D$ requirement. **PAM** algorithm, $O(N^2 KT)$. **Voronoi iteration** is faster heuristic.

### 21.3.7 Choosing $K$
- **Distortion on validation set**: monotonically decreases - useless (K-means is degenerate density model).
- **BIC of GMM**: U-shaped curve, picks correct $K$.
- **Silhouette coefficient**: $sc(i) = (b_i - a_i)/\max(a_i,b_i)$ where $a_i$ = mean intra-cluster dist, $b_i$ = mean dist to next-closest cluster. In $[-1, +1]$. **Silhouette diagram** shows per-instance scores.
- **Incremental growing**: split highest-weight cluster (perturb centroid, halve scores), prune small/narrow.
- **Sparse priors** (variational Bayes) to kill off components.

### Agentic LLM hypothesis - HIGH SPECULATIVE VALUE
- **Online VQ over agent observation embeddings = symbolic world model**: at each step the agent encodes its observation into a codebook index. The codebook becomes a discrete "concept vocabulary" learned online. Compare to VQ-VAE in world models (DreamerV2/V3 use a related trick). For LLM agents, this could discretize tool outputs, web pages, file contents into a compact set of "states" the LLM can reason over symbolically. Crucially the codebook can grow (K-means++ style farthest-point insertion when an observation is too novel).
- **K-medoids for trajectory clustering with non-metric similarity**: agent trajectories don't live in $\mathbb{R}^D$ in any canonical way. With an LLM-as-judge similarity (or learned trajectory embedding distance), K-medoids gives you canonical *exemplar trajectories* per cluster. These exemplars become **few-shot demonstrations** auto-curated from agent's own past - "show me an example of what cluster-7 behavior looks like" returns the medoid trajectory.
- **K-means++ for exploration**: in RL/agent exploration, choose next subgoal as the state farthest from all currently-visited states (in embedding space). This is literally the K-means++ initialization scheme repurposed as an intrinsic motivation signal. Provable $O(\log K)$ coverage guarantee.
- **Silhouette as confidence/abstention signal**: when an agent classifies a new situation into its skill library, compute its silhouette score against the library clusters. Low or negative silhouette $\Rightarrow$ "this situation is between clusters" $\Rightarrow$ ask for help / search / take cautious action. Cheap calibration without any explicit confidence model.
- **Vector quantization of CoT tokens / reasoning steps**: each reasoning step gets a discrete code. Now CoT becomes a sequence over a finite vocabulary of "reasoning moves". Enables: n-gram models of reasoning, frequency-based pruning of bad reasoning patterns, planning over discrete codes rather than tokens. Speculative bridge to neurosymbolic reasoning.
- **Distortion-as-novelty for active data curation**: when collecting agent data, compute average distortion to current codebook. High distortion examples are novel and should be kept for training. Low distortion examples are redundant - subsample them. This is a clean unsupervised version of coreset selection.
- **K-medoids for prompt-bank curation**: medoid prompts per cluster of user queries serve as canonical "this is what users in this segment ask" examples for prompt engineering / SFT data.

---

## 21.4 Mixture Models (GMM, Mixtures of Bernoullis)

### 21.4.1 GMM
- $p(x|\theta) = \sum_k \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$.
- **Responsibility** $r_{nk} = p(z_n=k|x_n,\theta)$ (Bayes rule).
- Hard cluster: $\hat z_n = \arg\max_k r_{nk}$.
- **K-means = EM-GMM** with $\Sigma_k = I$, $\pi_k = 1/K$, hard E-step.
- Covariance variants: full / tied / diagonal / spherical. Full needed when off-diagonal structure exists.
- **Label switching / non-identifiability**: permuting labels leaves likelihood unchanged. Posterior is symmetric - bad for HMC. Fix with **ordering constraint** $\mu_1 < \mu_2$ or invertible transforms. Doesn't scale to $D > 1$.
- **Bayesian model selection** for $K$: WAIC / BIC over candidate $K$.

### 21.4.2 Mixture of Bernoullis
- For binary data: $p(y|z=k) = \prod_d \mu_{dk}^{y_d}(1-\mu_{dk})^{1-y_d}$. Cluster binarized MNIST etc.

### Agentic LLM hypothesis - HIGH SPECULATIVE VALUE
- **Soft skill assignment via responsibilities**: instead of hard-routing a task to a single skill/expert/persona, use $r_{nk}$ as a soft mixture weight. This is exactly what MoE LLMs do at the parameter level - but here we'd do it at the **skill/policy** level. A query gets soft-assigned to skill clusters; the agent runs a weighted mixture of policies. Allows graceful interpolation between skills for novel queries.
- **Mixture of Bernoullis over tool-use vectors**: each agent run is a binary vector "which tools were called". Mixture of Bernoullis discovers latent agent "modes" (the search-heavy mode, the code-only mode, the read-and-summarize mode). $\mu_{dk}$ = prob that tool $d$ is used in mode $k$. Beautifully interpretable.
- **Full-covariance GMM captures correlated reasoning failures**: in CoT embedding space, full-covariance clusters can pick up directional "failure modes" (e.g., "miscount in this oriented subspace = hallucination"). Spherical/diagonal covariance misses this. Useful for failure-mode discovery in evals.
- **Label-switching mitigation via reference embeddings**: when comparing GMM-style skill libraries across training runs, label switching breaks comparisons. Fix: align clusters by matching centroids to fixed reference embeddings (e.g., embeddings of human-written skill names). Skill libraries become comparable across training runs / agents.
- **Posterior responsibility as a confidence signal for agent action**: if $\max_k r_{nk} \ll 1$, the agent is genuinely uncertain which skill to apply. Trigger reflection / deliberation. Inverse correlated with response time in humans - cognitively well-motivated.
- **EM-style self-training of agent skills**: E-step = soft assign past trajectories to skill cluster, M-step = fine-tune a per-cluster LoRA adapter on responsibility-weighted data. This is the obvious EM/MoE bridge for personalized/specialized agents - probably underexplored compared to hard-routing.
- **WAIC for skill-library sizing**: principled "how many skills should my agent have?" answer using WAIC over GMM fits in skill-embedding space. Sidesteps the "just pick $K=8$" arbitrariness.

---

## 21.5 Spectral Clustering

- Build similarity graph $W$ (often $k$-NN sparsified), Gaussian kernel weights $W_{ij} = \exp(-\frac{1}{2\sigma^2}\|x_i-x_j\|^2)$.
- **Cut** objective splits off singletons; **Normalized cut** fixes this:
$$\text{Ncut}(S_1,...,S_K) = \frac{1}{2}\sum_k \frac{\text{cut}(S_k, \bar S_k)}{\text{vol}(S_k)}$$
- NP-hard exactly; relax via graph Laplacian.
- **Graph Laplacian** $L = D - W$. **Theorem**: eigenvectors of $L$ with eigenvalue 0 are spanned by indicator vectors $\mathbf{1}_{S_k}$ of connected components.
- **Algorithm**: smallest $K$ eigenvectors of $L_{sym} = I - D^{-1/2} W D^{-1/2}$, stack as rows of $U$, normalize rows to unit norm, K-means on rows.
- **Connections**:
  - **kPCA**: largest eigenvectors of $W$ = smallest of $I-W$.
  - **Random walks**: $P = D^{-1}W$ is stochastic, $L_{rw} = I - D^{-1}W$. Ncut $\approx$ partition that random walk rarely crosses.
  - Same as **Laplacian eigenmaps**.
- Captures non-convex / non-Gaussian clusters (spirals where K-means fails).

### Agentic LLM hypothesis - HIGH SPECULATIVE VALUE
- **Spectral clustering on attention graphs**: treat token-token attention from a layer as $W$, run spectral clustering. The clusters are "tokens the model is jointly thinking about". Potentially: emergent constituent/coref discovery, automatic chunking for retrieval, or identifying "circuits" in mechanistic interpretability. Very natural since attention is already a graph.
- **Trajectory graph spectral clustering**: build a graph over trajectories where edges weight LLM-judged similarity. Spectral clustering finds non-convex skill modes that K-means in embedding space misses. Spiral-shaped skill manifolds (e.g., "increasingly complex versions of the same skill") are exactly the case K-means fails on.
- **Random walks $\equiv$ Ncut interpretation for agent state graphs**: build graph where nodes = agent states (or replay buffer entries), edges = transition frequency. Ncut finds **bottleneck states** = options/subgoals. This is literally the **option discovery** problem in RL. The book's random-walk identity (Ncut = $p(\bar S | S) + p(S|\bar S)$) IS the bottleneck-option discovery objective. Spectral clustering of agent state graphs = automatic option/subgoal mining.
- **Hierarchical RL with spectral options**: recursively spectral-cluster state graphs to get nested options. Top-level Ncut $\to$ macro-regions; within-region spectral $\to$ sub-options. Fully unsupervised RL hierarchy from graph topology.
- **Spectral clustering of agent population**: nodes = different agents (different prompts/checkpoints/personas), edges = behavioral similarity on a benchmark. Spectral partitions reveal "behavior families" - useful for diverse-agent ensembles, debate, multi-agent simulation seed selection.
- **Laplacian eigenmap features as skill embeddings**: skip K-means, just use the bottom Laplacian eigenvectors of a trajectory similarity graph as low-dimensional skill features. Geometric/connectivity-aware embeddings ready for downstream training.
- **Spectral graph clustering on LLM thought graphs**: ToT/GoT (tree/graph of thoughts) produces an actual graph. Apply spectral clustering to find "reasoning communities" - subgraphs of thoughts that mutually support each other. Use cluster structure to prune / reweight aggregation.
- **Bottleneck = handoff**: in multi-agent systems, low-conductance edges (bottlenecks) found by spectral clustering are natural **agent handoff points** - tasks where one agent finishes and another should pick up.

---

## 21.6 Biclustering / Coclustering

- Cluster rows *and* columns of $X \in \mathbb{R}^{N_r \times N_c}$ jointly. Standard in bioinformatics (genes x conditions) and collaborative filtering (users x movies).
- **Basic model** [Kem+06]: latent row cluster $u_i$, latent column cluster $v_j$, $X_{ij} \sim p(\cdot|\theta_{u_i, v_j})$.
- **Limitation**: each row in exactly one cluster - but objects can play multiple roles depending on which features you look at.
- **Crosscat / multi-clust / nested partitioning**: column clusters $v_j$ first; then *within each column cluster $l$*, an independent row partitioning $u_{i,l}$. Row $i$ belongs to different row clusters depending on the column-feature subset.
  - Animals dataset example: column partition A (taxonomic features) splits animals into mammals/reptiles/birds/invertebrates; column partition C (ecological features) splits same animals into prey/land-predators/sea-predators/air-predators. Same animals, different clusterings.
- Typically use Dirichlet processes for $N_u, N_v$.

### Agentic LLM hypothesis - VERY HIGH SPECULATIVE VALUE
- **Biclustering tool-call matrices**: $X_{ij} = 1$ iff agent $i$ called tool $j$ (or task $i$ uses tool $j$). Discover joint clusters of (task-types, tool-bundles). Output: tasks that share tool-recipes. Directly usable for:
  - Tool subset selection per task type (only show relevant tools).
  - Tool recommendation for novel tasks (find nearest task-cluster, suggest its tool-bundle).
  - Macro-tool synthesis (cluster of tools always co-occurring $\Rightarrow$ wrap as a single composite tool).
- **Crosscat for multi-view task taxonomy**: same task can be classified differently along different feature dimensions.
  - Column-cluster 1 = (latency, cost) features $\to$ tasks group as fast/cheap vs slow/expensive.
  - Column-cluster 2 = (domain) features $\to$ tasks group as code/math/web.
  - Column-cluster 3 = (failure mode) features $\to$ tasks group as hallucination-prone vs format-prone.
  - A single task has *three different cluster labels*, one per feature-view. This is closer to how humans actually organize work than single-view clustering.
- **User x prompt biclustering**: biclustering of user-by-prompt interaction matrices for product analytics on chat assistants. Joint clusters of (user-segment, prompt-archetype). Personalization without hard segmentation.
- **Biclustering CoT steps x problem types**: rows = problems, columns = atomic reasoning step types (extracted via VQ over CoT - tie-in to 21.3.3). Reveals which reasoning step bundles are diagnostic of which problem clusters. Auto-discovery of solution templates.
- **Biclustering attention head x input-pattern**: in mech interp, find joint clusters of (heads, input contexts where they fire). Crosscat lets the same head belong to different functional clusters depending on which input features you look at - which is empirically how heads behave (polysemantic).
- **Crosscat = the right model for multi-aspect skill clustering**: agent skills are multi-aspect (manner, domain, difficulty, risk). Each aspect induces its own partition. Crosscat is the only model in this chapter that natively handles this, and it appears underused in ML agent research.

---

## Dirichlet Process Mixtures (not deeply covered in 21, but referenced for crosscat)

- DP mixtures: infinite mixture model; effective $K$ determined from data via concentration param $\alpha$.
- Chinese restaurant process / stick-breaking representation.
- Used in crosscat to avoid pre-specifying $N_u, N_v$.

### Agentic LLM hypothesis - HIGHEST SPECULATIVE VALUE
- **Latent skill library = DP mixture**: every agent trajectory either (a) belongs to an existing skill cluster with probability $\propto N_k$ ("rich get richer") or (b) instantiates a new skill cluster with probability $\propto \alpha$. This is *exactly* the right inductive bias for **lifelong skill discovery**:
  - Most new trajectories fit existing skills (model reuses).
  - Occasionally a truly novel trajectory spawns a new skill (model expands).
  - No need to set $K$ in advance. $\alpha$ controls the rate of skill invention.
- **DP for memory/knowledge graphs**: agent memory items are clustered into entities/topics, but the number of entities grows with experience. Pure DP mixture; concentration $\alpha$ tunes "how readily a new memory becomes its own thing".
- **DP for tool synthesis**: cluster observed (task, tool-call-sequence) pairs into "tool templates". CRP prior: most new tasks fit a template; occasionally a task needs a brand new template (which becomes a candidate macro-tool).
- **DP mixtures for agent population modeling**: in multi-agent simulations, each agent's behavior is drawn from a DP mixture of "behavior types". New types emerge as the population is observed.
- **Hierarchical DPs (HDP) for cross-task transfer**: top-level DP over "global" skills shared across tasks, task-level DP picks a subset of global skills (with potentially new local skills). Maps cleanly onto multi-task / meta-RL setups where some skills are universal and some are task-specific. The "share-skills-across-tasks-but-allow-novel-ones" inductive bias is HDP, full stop.
- **DP-CRP as natural growth rule for KV-cache compression**: incoming tokens either fit an existing summary cluster or spawn a new one. Concentration parameter $\alpha$ controls compression ratio. Provably consistent unlike heuristic streaming summarizers.
- **Nonparametric Bayes is exactly what online agents need**: agents face an open-world setting - any fixed-$K$ model is misspecified by assumption. DP/HDP/IBP machinery is decades old and largely abandoned in the deep era, but the inductive bias matches LLM agents *better than anything else in the standard toolkit*. Strong candidate for a comeback in agent research.

---

## Cross-cutting Speculative Ideas

- **Pipeline**: agent rollouts $\to$ embed (LLM encoder) $\to$ HAC for hierarchy + GMM for soft assignment + DP for unbounded $K$ + biclustering for (task x tool) co-structure $\to$ output: structured skill library with multi-view partitions.
- **Evaluation crisis carries over**: Jain & Dubes quote applies *more* to skill clustering than to gene clustering - we have even fewer ground-truth labels. Need to develop LLM-judge-based external criteria (purity vs judge labels, Rand vs judge pairwise comparisons) as the standard evaluation tooling for agent-skill clustering papers.
- **Clustering for unsupervised reward shaping**:
  - Cluster all trajectories; reward = inverse-distance to nearest successful trajectory's cluster centroid.
  - Or: shape reward $\propto$ entropy-reduction in cluster assignment (encourages committing to a clear strategy mode).
- **Spectral + biclustering combo**: spectral cluster the trajectory similarity graph, then biclustering inside each spectral cluster across (trajectory, action-types) - hybrid hierarchical structure discovery.
- **Most underrated technique for agents**: crosscat/nested partitioning. Most overrated/already-saturated: vanilla K-means on embeddings.
- **Most dead-end**: complete-link HAC for agent data; the diameter constraint is unmotivated when trajectory similarity is high-dim and noisy. Single-link and average-link are more salvageable.
