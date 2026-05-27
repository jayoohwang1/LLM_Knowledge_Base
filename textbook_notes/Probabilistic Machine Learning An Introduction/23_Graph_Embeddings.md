# Chapter 23: Graph Embeddings

Note: chapter covers classical **Graph Representation Learning (GRL)** circa 2014–2020. The technical content is fairly digestible; the more interesting question is *where do these ideas show up next in LLM-agent land*. Each section gets a tight technical summary + a speculative **Agentic LLM hypothesis** subsection.

## 23.1 Introduction & Motivation

- **Goal of GRL**: learn low-dim vector reps $\mathbf{Z} \in \mathbb{R}^{N \times L}$ of nodes/edges/graphs from $\mathbf{A} \in \{0,1\}^{N\times N}$ (adj) or weighted $\mathbf{W}$, possibly with node features $\mathbf{X}\in\mathbb{R}^{N\times D}$.
- **Why hard**: graphs are **non-Euclidean**, no canonical node ordering, neighborhoods have varying size/structure. CNN priors (shift-invariance, fixed neighborhoods) break.
- **Geometric Deep Learning (GDL)**: umbrella term for DL on non-Euclidean domains (graphs, manifolds).
- **Two axes**:
  - **Unsupervised vs supervised** (preserve graph structure vs predict labels).
  - **Transductive** (fixed graph, predict missing pieces, e.g. social network user attrs) **vs inductive** (generalize to unseen graphs, e.g. molecule classification).
- **Non-Euclidean embedding spaces**: hyperbolic (continuous trees), spherical, mixed-curvature.

### Agentic LLM hypothesis
- Every meaningful agent substrate is a graph: **tool dependency graph**, **call graph of subagents**, **file system DAG**, **API surface graph**, **knowledge graph** of facts, **AST/CFG** of code, **plan-DAG** of subgoals. Most LLM systems flatten these into token sequences (token-serialized adjacency lists, JSON dumps) — this is *exactly* the Euclidean-vs-non-Euclidean mismatch the chapter opens with.
- Bet: as agents scale, we'll see explicit **graph-native conditioning** mechanisms in LLMs — not just retrieval over passages but retrieval+aggregation over typed relational structure, possibly via cross-attention to **graph encoder outputs** (Z) rather than raw text.
- **Inductive vs transductive** maps cleanly onto agents: per-agent fine-tuned graph models (transductive over a fixed corp KB) vs a generalist agent that ingests a new tool graph zero-shot (inductive). The latter is where GNN-style permutation-invariant encoders matter, because tool graphs differ per deployment.

## 23.2 GraphEDM: Encoder/Decoder Framework

- General template (Chami et al. 2021):
  - **Encoder**: $\mathbf{Z} = \text{ENC}(\mathbf{W}, \mathbf{X}; \Theta^E)$
  - **Structure decoder**: $\widehat{\mathbf{W}} = \text{DEC}(\mathbf{Z}; \Theta^D)$ — reconstructs similarity matrix.
  - **Classification decoder**: $\hat{y}^S = \text{DEC}(\mathbf{Z}; \Theta^S)$ — task head.
- **Total loss**: $\mathcal{L} = \alpha \mathcal{L}_{\text{SUP}}^S + \beta \mathcal{L}_{G,\text{RECON}} + \gamma \mathcal{L}_{\text{REG}}$.
  - $\alpha = 0$ ⇒ unsupervised; $\alpha \neq 0$ ⇒ (semi-)supervised.

### Agentic LLM hypothesis
- The encoder/decoder split is suggestive: agents could maintain a **persistent graph encoder** that compresses an evolving world-state graph into $\mathbf{Z}$, with two heads:
  - **Reconstruction head** = unsupervised auxiliary keeping embeddings faithful to ground-truth structure (acts as a regularizer against hallucination of relations).
  - **Task head** = action-policy / next-tool predictor from $\mathbf{Z}$.
- The **multi-objective loss** maps to **mixture-of-objectives RLHF-style training** for agents: combine task reward, structural fidelity reward (does the agent's internal world model match observed environment graph?), and a regularization term.
- Speculative pattern: train a small graph-encoder LoRA that lives alongside the LLM, with a `[GRAPH]` token routing attention to its Z. Reconstruction loss = "given Z, the LLM can answer queries about edges" → forces fidelity.

## 23.3 Shallow Graph Embeddings

Shallow = encoder is just an **embedding lookup table** $\mathbf{Z} = \Theta^E$. Categorical node ID → vector. Transductive only. Graph structure enters only via the loss.

### 23.3.1–23.3.2 Distance-based Euclidean

- **MDS**: $d_1 = \|s(\mathbf{W}) - \widehat{\mathbf{W}}\|_F^2$, $\widehat{W}_{ij} = \|\mathbf{Z}_i - \mathbf{Z}_j\|$.
- **Laplacian eigenmaps**: $\min \text{tr}(\mathbf{Z}^\top \mathbf{L} \mathbf{Z})$ s.t. $\mathbf{Z}^\top \mathbf{D} \mathbf{Z} = \mathbf{I}$. Equivalent to $\sum_{ij} W_{ij}\|\mathbf{Z}_i-\mathbf{Z}_j\|^2$ — connected nodes pulled close.

### 23.3.3 Non-Euclidean (Hyperbolic) Embeddings

- **Poincaré embeddings** (Nickel & Kiela): $d_{\text{Poincaré}}(\mathbf{Z}_i,\mathbf{Z}_j) = \text{arcosh}(1 + 2\frac{\|\mathbf{Z}_i-\mathbf{Z}_j\|^2}{(1-\|\mathbf{Z}_i\|^2)(1-\|\mathbf{Z}_j\|^2)})$.
  - Distance blows up near the boundary → infinite "room" for leaves of a tree. Volume of ball grows exponentially with radius (matches branching factor).
- **Lorentz model**: better numerical stability than Poincaré.
- **Mixed-curvature product spaces** (Gu et al.): ring-of-trees and other heterogeneous structures.

#### Agentic LLM hypothesis (extensive — high speculative value)
- **Hierarchies are everywhere in agent state**: task decomposition trees, file systems, ontology trees, code module hierarchies, organizational structures. Forcing these into Euclidean token positions is lossy — Euclidean distortion of a tree grows with depth.
- **Hyperbolic embeddings for plan trees**: an agent that decomposes a goal into a deep subgoal tree could keep subgoal embeddings in hyperbolic space so distance between sibling subgoals stays small while parent-child gaps remain interpretable.
- **Hyperbolic positional encodings** for LLMs that process tree-structured inputs (e.g., HTML DOM agents, AST-conditioned code agents). Could literally replace RoPE with a Poincaré ball coordinate.
- **Tool taxonomies**: tools naturally taxonomize into hierarchies (file ops → read, write, search, …). A hyperbolic embedding gives a tool-selector LLM a natural "drill down" geometry.
- Strong bet: hyperbolic side-channels appear in long-horizon agents within 1–2 years where hierarchy is the bottleneck (long agentic episodes, deep task decomposition).

### 23.3.4 Matrix Factorization

- Decoder = dot product: $\widehat{\mathbf{W}} = \mathbf{Z}\mathbf{Z}^\top$.
- **Graph factorization (Ahmed et al.)**: $\mathcal{L} = \sum_{(i,j)\in E}(W_{ij} - \widehat{W}_{ij})^2$, complexity $O(M)$ (sparse).
- **GraRep**: handles directed graphs via source $\mathbf{Z}^s$ and target $\mathbf{Z}^t$ embeddings; preserves $k$-hop neighborhoods via $\mathbf{D}^{-k}\mathbf{W}^k$. Not scalable since $\mathbf{D}^{-1}\mathbf{W}$ becomes dense.

#### Agentic LLM hypothesis
- **Asymmetric embeddings for directed agent relations**: "agent A delegates to agent B" is not symmetric. In multi-agent systems with role hierarchies (manager/worker), use GraRep-style $\mathbf{Z}^s, \mathbf{Z}^t$ for source-of-message and recipient roles.
- $k$-hop factorization is interesting for **reasoning chains** — a node's $k$-hop neighborhood is its $k$-step reachable causal set; embed reasoning steps so a downstream policy can read off "what's reachable in 3 steps".
- Dot-product decoders are essentially the same operation as **attention-logit computation**; transformers are already doing implicit graph factorization on the complete attention graph (more on this in §23.4).

### 23.3.5 Skip-gram / Random Walk Methods

- **DeepWalk** (Perozzi et al.): sample truncated random walks, treat as "sentences", train skip-gram (word2vec) on them.
  - Objective: $\mathcal{L} = -\sum \log P(w_{k-i}|w_k)$.
  - GraphEDM view: $s(\mathbf{W}) = \mathbb{E}_q[(\mathbf{D}^{-1}\mathbf{W})^q]$, $q \sim$ Categorical over walk lengths.
- **node2vec** (Grover & Leskovec): biased random walks with parameters $p$ (return) and $q$ (in/out), trading off BFS-like (structural similarity) vs DFS-like (community) exploration.
- **Implicit matrix factorization** (Levy & Goldberg, Qiu et al. NetMF): all these skip-gram methods provably factorize a particular matrix — unifies DeepWalk/LINE/node2vec under one MF framework.

#### Agentic LLM hypothesis (extensive — high speculative value)
- **Random walks as "synthetic agent trajectories"**: DeepWalk's insight = "walks on a graph are like sentences." Inverse: **agent trajectories through a tool/state graph are sentences**, and an LLM trained on them learns skip-gram-like context for what comes next.
  - This is essentially what **trajectory pretraining** does (e.g., behavior cloning over sampled tool-use traces). Pretrained LLM + sampled walks over the tool graph = node2vec-for-tools.
- **node2vec's $p,q$ biases** map onto **exploration policy** for an agent generating its own training data:
  - $q < 1$ (DFS-like, explore far) ↔ long-horizon task exploration.
  - $q > 1$ (BFS-like, stay local) ↔ structurally similar peer tools.
  - Could literally tune walk biases to generate balanced synthetic agent rollouts.
- **Curriculum from random walks on the doc graph**: take a wiki/codebase, build a graph of cross-references, generate random-walk "reading orders" and pretrain on them. Likely matches what some retrieval-pretraining schemes already approximate.
- **Two-rep trick** (center vs context): an agent could maintain a "speaker" representation (what it emits) vs "listener" representation (what it expects) — relevant for multi-agent communication channels.
- **NetMF** unification — skip-gram = MF — implies LLM next-token-prediction over walks ≈ low-rank factorization of a walk-statistics matrix. So pretraining-on-trajectories has a clean MF interpretation; might suggest principled regularizers.

### 23.3.6 Supervised Shallow: Label Propagation

- **LP** (Zhu et al.): smooth labels over graph; $\mathcal{L} = \sum W_{ij}\|\hat{y}_i - \hat{y}_j\|^2$.
- **Label spreading**: normalized variant with $\hat{y}_i/\sqrt{D_i}$.
- Works when graph is "consistent": neighbors share labels.

#### Agentic LLM hypothesis
- **Trust/credibility propagation across knowledge graphs**: if a retrieved fact $i$ is verified, propagate confidence to closely related facts. Classical PageRank-like trust propagation but with explicit per-fact embeddings.
- **Tool-success propagation**: if one API call worked, structurally similar APIs probably work too — let an agent boot-strap reliability estimates over its tool graph.

## 23.4 Graph Neural Networks (GNNs)

### 23.4.1 Message Passing GNNs

- **Original GNN** (Scarselli 2009): fixed-point iteration $\mathbf{Z}^{t+1} = \text{ENC}(\mathbf{X}, \mathbf{W}, \mathbf{Z}^t; \Theta^E)$ until convergence. Requires contraction mapping (Banach fixed point).
- **GGSNN** (Li et al.): removes contraction constraint; uses GRU; fixed number of steps; good for sequential/temporal graphs.
- **MPNN framework** (Gilmer et al.): the canonical abstraction.
  - Message: $\mathbf{m}_i^{\ell+1} = \sum_{j: (v_i,v_j)\in E} f^\ell(\mathbf{H}_i^\ell, \mathbf{H}_j^\ell)$
  - Update: $\mathbf{H}_i^{\ell+1} = h^\ell(\mathbf{H}_i^\ell, \mathbf{m}_i^{\ell+1})$
  - After $\ell$ layers, node sees $\ell$-hop neighborhood.
- **GraphNet** (Battaglia): extends MPNN with explicit edge + global representations.

#### Agentic LLM hypothesis (extensive — highest speculative value)
- **Transformers are GNNs on the complete graph with attention as the message function** — this is the load-bearing connection. The MPNN equation $\mathbf{m}_i = \sum_j f(\mathbf{H}_i, \mathbf{H}_j)$ is *literally* attention if $f$ is an attention-weighted value lookup.
  - Implication: when you sparsify attention (sliding window, BigBird, etc.), you're choosing an explicit edge set — i.e., choosing the GNN's graph. Agentic workloads with known structure (tool graphs, code ASTs, KG fragments) should use **structure-aware sparse attention**, which is exactly an MPNN.
- **Reasoning chains as DAGs amenable to MPNN**: ToT/GoT (Tree-of-Thought, Graph-of-Thought) constructs an explicit reasoning graph at inference time. Currently nodes are scored independently or via cross-thought attention. A real GNN over the thought-graph — message passing between partial reasoning states — could let "lemmas" propagate to nodes that need them, similar to belief propagation. Bet: GoT becomes literal GNN-of-Thought.
- **World models as graphs**: imagine an agent maintaining a typed entity-relation graph of its environment, with **MPNN updates** triggered by each observation (the message function = "given new percept, update entity reps that share an edge"). Recurrent MPNN ≈ neural Kalman filter on graph state.
- **Multi-agent communication graphs**: each round of agent communication is one MPNN layer; aggregation = how an agent integrates received messages. GraphNet's global node = "shared blackboard." Fixed number of message-passing rounds = fixed cooperation depth.
- **GGSNN (gated, sequential)** is interesting for **temporal/streaming agents** where the graph mutates: GRU-based updates handle the "graph is evolving" case naturally.
- **Edge representations (GraphNet)**: agents reasoning about *relations* (not just entities) benefit from explicit edge embeddings — e.g., "the relationship between Alice and Bob is collegial" embedded as an edge vector, updated as new evidence comes in.

### 23.4.2 Spectral Graph Convolutions

- Use spectral domain of $\mathbf{L}$.
- **Spectrum-based** (Bruna 2014): explicit eigendecomp, domain-dependent (filters tied to a specific graph).
- **Spectrum-free** (GCN, Kipf & Welling): uses Chebyshev/polynomial approximations of spectral filters; doesn't actually diagonalize $\mathbf{L}$, but needs full $\mathbf{W}$ → doesn't scale.
- **GCN** layer: $\mathbf{H}^{\ell+1} = \sigma(\hat{\mathbf{D}}^{-1/2}\hat{\mathbf{W}}\hat{\mathbf{D}}^{-1/2}\mathbf{H}^\ell\Theta^\ell)$ (normalized adjacency × features × weights).

#### Agentic LLM hypothesis
- Spectral methods are mostly a dead end for general agents — domain dependency kills inductive use. But:
- **Spectral analysis of agent communication graphs** as a *diagnostic*: low-frequency eigenvectors reveal communities of cooperating agents; spectral gap = how well-mixed information becomes. Could be a useful "agent-org" health metric.
- **Heat-kernel / diffusion-based positional encodings** (close cousin to spectral methods, e.g., GraphTransformer PEs) likely transfer to LLM agents that need to encode "how far apart" two nodes are in a known graph — these have shown up in graph transformer literature already and could become standard for graph-conditioned LLM prompts.

### 23.4.3 Spatial Graph Convolutions

#### 23.4.3.1 GraphSAGE (sampling-based)

- Samples fixed-size neighborhood per node: $\mathbf{H}_i^{\ell+1} = \sigma(\Theta_1^\ell \mathbf{H}_i^\ell + \Theta_2^\ell \text{AGG}(\{\mathbf{H}_j^\ell : v_j \in \text{Sample}(\text{nbr}(v_i), q)\}))$.
- AGG = mean / pool / LSTM. Inductive — generalizes to new nodes/graphs.

##### Agentic LLM hypothesis
- **Sampled neighborhood aggregation** maps onto **retrieval**: for a given query node (context, agent state), sample $q$ relevant neighbors from a huge KB graph, aggregate. This is essentially RAG-as-GNN-layer.
  - Suggests stacking *multiple* RAG steps with intermediate state (= multilayer SAGE) instead of one-shot retrieval — agent learns multi-hop retrieval as multilayer message passing with sampling.
- Practical: GraphRAG-style methods are already doing crude versions; a proper GraphSAGE-style stack with learned aggregators (not just LLM summarization) is an obvious next step.

#### 23.4.3.2 Graph Attention Networks (GAT)

- Each node attends over all neighbors with learned softmax weights: $\alpha_{ij} = \text{softmax}_j(a(\mathbf{H}_i, \mathbf{H}_j))$, $\mathbf{H}_i' = \sigma(\sum_j \alpha_{ij} \mathbf{W}\mathbf{H}_j)$.
- Inductive; weights per-edge.

##### Agentic LLM hypothesis (extensive — highest speculative value)
- **Tool selection over a capability graph via GAT**: an agent's tool-selection step is exactly graph attention — given current context, attend over neighbors (candidate tools) with learned scores. Most current tool-use LLMs do this implicitly via prompt-and-softmax; doing it with an explicit GAT over a *capability graph* gives:
  - Better generalization across tool sets (inductive).
  - Per-edge structure features (this tool depends-on that one, this tool composes-with this one).
  - Multi-head = simultaneous attention along different relation types (data-flow, control-flow, type-compat).
- **Retrieval re-ranking as GAT**: candidate documents form a neighborhood; agent learns attention weights conditioned on query + doc relations.
- **Multi-agent attention**: in a multi-agent system, each agent attends over peer messages weighted by relationship — GAT directly models attention as a function of (sender, receiver, relation), generalizes "transformer-on-tokens" to "transformer-on-agents."
- Already happening in nascent form (e.g., GraphFormer-style architectures), but agentic LLM tooling has not absorbed it yet. Strong bet for the next 1–2 years.

#### 23.4.3.3 MoNet (geometric)

- Patches via parametric Gaussian kernels over pseudo-coordinates $\mathbf{U}^s$. Generalizes mesh CNNs.

##### Agentic LLM hypothesis
- Niche but possibly useful for **embodied/spatial agents** (robotics, 3D world models) where node features are physical coordinates. Less load-bearing for symbolic agents.

### 23.4.4 Non-Euclidean GNNs (Hyperbolic)

- **HGCN / HGNN**: do convolution in tangent space at origin (Euclidean approximation), map back to hyperbolic manifold. Improves embeddings of hierarchical graphs.

#### Agentic LLM hypothesis
- Combine with the hyperbolic hypothesis from §23.3.3: a **hyperbolic GNN reasoner over a plan tree** could be much more efficient at long-horizon planning than a flat-Euclidean transformer.
- Tangent-space trick (compute Euclidean, map back) is a clean recipe: an LLM with mostly-standard layers but periodic exponential/log maps could maintain a hyperbolic latent.

## 23.5 Deep Graph Embeddings (GNN-based, unsupervised)

### 23.5.1.1 SDNE (Structural Deep Network Embedding)

- Autoencoder on adjacency rows; preserves 1st+2nd order proximity. Loss: $\|(s(\mathbf{W})-\widehat{\mathbf{W}})\odot \mathbb{I}(s(\mathbf{W})>0)\|_F^2 + \alpha \sum s(\mathbf{W})_{ij}\|\mathbf{Z}_i-\mathbf{Z}_j\|^2$.
- The indicator-mask term: only penalize reconstruction on actually-present edges (handles sparsity).

### 23.5.1.2 (V)GAE — (Variational) Graph Autoencoders

- **GAE**: encoder = GCN, decoder = $\sigma(\mathbf{Z}\mathbf{Z}^\top)$; loss = cross-entropy on adjacency.
- **VGAE**: $\mathbf{Z}$ is latent with prior $\mathcal{N}(0,\mathbf{I})$; train via ELBO.

### 23.5.1.3 Graphite

- Iterative encoder-decoder: alternates $\widehat{\mathbf{W}}^{(k)} = \mathbf{Z}^{(k)}\mathbf{Z}^{(k)\top}/\|\mathbf{Z}^{(k)}\|^2 + \mathbf{1}\mathbf{1}^\top/N$ and $\mathbf{Z}^{(k+1)} = \text{GCN}(\widehat{\mathbf{W}}^{(k)}, \mathbf{Z}^{(k)})$.
- More expressive decoder via iterative refinement.

### 23.5.1.4 Contrastive (Deep Graph Infomax, GMI)

- **DGI**: maximize MI between node reps $\mathbf{Z}_i$ and a graph-summary $\mathcal{R}(\mathbf{Z})$ (readout). Trains a discriminator $\mathcal{D}$ on real vs row-shuffled-feature negatives.
  - $\min_\Theta -\sum_i \log\mathcal{D}(\mathbf{Z}_i, \mathcal{R}(\mathbf{Z})) - \sum_j \log(1 - \mathcal{D}(\mathbf{Z}_j^-, \mathcal{R}(\mathbf{Z}^-)))$.
- **GMI** (Graphical MI): max MI between node and its **neighbors**, not whole graph.

#### Agentic LLM hypothesis (deep autoencoders + contrastive)
- **Agent state autoencoder**: world-state graph → Z → reconstruct. Z becomes a compact, persistent agent memory. Cheaper to attend over than full history.
- **VGAE-style stochastic latents** offer a way to do **Bayesian agent state**: maintain uncertainty over the world graph, sample plausible reconstructions for planning rollouts.
- **DGI / GMI for self-supervised agent pretraining**: maximize MI between local context (sub-graph of recent observations / tool calls) and global episode summary. Hits the same "local-global consistency" objective that probably underlies good agent memory.
- Contrastive readouts $\mathcal{R}$ over agent state are essentially what current "summary tokens" / "scratchpad summarization" do heuristically — could be replaced with a learned readout.

### 23.5.2 Semi-supervised (SemiEmb, Planetoid)

- **SemiEmb**: MLP encoder + LP-style graph regularizer on intermediate layers.
- **Planetoid**: split embedding into structural $\mathbf{Z}^c$ (random-walk-based) and feature $\mathbf{Z}^F$ (from features); jointly optimize.

#### Agentic LLM hypothesis
- **Decoupling structural vs feature embeddings** is suggestive for agents: keep "what this thing IS" (feature embedding from descriptor) separate from "where it SITS in the graph" (structural embedding from walks). Then a tool that's never been seen but has a known description can still be placed by feature reps, while frequent tools get sharper structural reps.

## 23.6 Applications

- **Unsupervised**: graph reconstruction, **link prediction** (predict missing edges — KG completion, recommendation, fraud detection), clustering (community detection), visualization (t-SNE/PCA on top of Z).
- **Supervised**: node classification (often semi-supervised, transductive), graph classification (inductive; molecular property prediction).

### Agentic LLM hypothesis (applications)
- **Link prediction over agent-relevant graphs**:
  - Predict which tool will be needed next (link from current state node to tool node).
  - Predict which subgoal connects to which — basically learned planning heuristics.
  - **KG completion** for grounding: an agent answering a relational question can use a learned KG-completion model as an oracle.
- **Fraud-detection-style anomaly detection** on agent action graphs: spot adversarial / off-policy behavior by detecting edges (action-flows) that look anomalous under the learned embedding.
- **Graph classification = whole-episode evaluation**: a graph rep of a whole agent episode (entities touched, tools called, sequence DAG) classified for success/failure → learned reward model that operates on structure, not text.

## Cross-cutting speculation: where graph methods plug into the agent stack

### 1. Knowledge graph embeddings (TransE / DistMult / ComplEx) for grounded reasoning

(Chapter mentions KG link prediction but doesn't dive into translational embeddings; worth covering since prompt asked.)

- **TransE**: $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$ for (head, relation, tail) triples. Train by margin loss vs negative samples.
- **DistMult**: bilinear score $\langle \mathbf{h}, \mathbf{r}, \mathbf{t} \rangle = \sum_k h_k r_k t_k$. Symmetric only.
- **ComplEx**: same but in $\mathbb{C}$, asymmetric relations representable.
- **RotatE**: relations as rotations in complex plane.

**Agentic LLM hypothesis**:
- **Grounded RAG via KG completion**: when an agent retrieves a fact $(h, r, ?)$, score candidates using the embedding score plus LLM verification. Hybrid symbolic+neural retrieval.
- **TransE-style structure in LLM embedding spaces**: there's evidence LLMs already learn approximate $\mathbf{e}_{\text{Paris}} - \mathbf{e}_{\text{France}} \approx \mathbf{e}_{\text{Berlin}} - \mathbf{e}_{\text{Germany}}$ patterns (word2vec analogy hangover). Explicitly fine-tuning to enforce $\mathbf{h}+\mathbf{r}\approx\mathbf{t}$ across millions of KG triples could give LLMs much sharper relational grounding.
- **ComplEx for asymmetric agent relations**: "delegates-to", "imports-from", "depends-on" all asymmetric — symmetric embeddings (DistMult) waste capacity here.

### 2. Code as graph (AST / CFG / data-flow)

- Not explicitly in chapter but cleanly fits MPNN/GAT applications.

**Agentic LLM hypothesis (extensive)**:
- Code-LLMs operating on **flat token streams** lose structure that's free: AST gives parent/child/sibling, CFG gives reachability, data-flow gives use-def chains.
- A code-agent with a **GNN side-channel** over the AST/CFG, attending into the LLM via cross-attention, could:
  - Localize bugs by aggregating data-flow paths (message passing along def-use edges).
  - Plan refactors as graph rewrites (graph-classification of "this subgraph is anti-pattern X").
  - Do typed tool/API selection: tool graph = call graph; select call sites by GAT.
- Bet: hybrid GNN-Transformer code agents become standard for repo-scale tasks where pure-context-window LLMs choke.

### 3. Web/filesystem as graph for web/computer-use agents

- Web is a hyperlinked DAG; filesystems are trees; UIs are DOM trees.
- **Agentic LLM hypothesis**:
  - DOM-aware browsing agent uses a tree GNN over the DOM to score "clickability" / "relevance" of UI elements, conditioned on goal embedding.
  - Filesystem agent uses hyperbolic embedding of directory tree (tree → hyperbolic is natural) for efficient "where should I look" prediction.

### 4. Transformers as GNNs on complete graphs — the central duality

- Stated explicitly in modern lit (e.g., Joshi 2020): self-attention = MPNN with all edges present and attention as $f$.
- **Implication for agent design**:
  - When the agent's state has known structure, the *complete* attention graph wastes compute on impossible/irrelevant edges. **Structure-aware attention masks** = explicit GNN.
  - Mixture: keep dense attention for free-form text, sparse structured attention for graph-substrate (tool graph, KG fragment, AST).
- Strong long-term thesis: **agent transformers will become explicitly graph-structured** as workloads expose more structure, especially for very long horizons (where dense attention is infeasible and structure is the natural compression).

### 5. Structured prediction & DAG outputs

- Many agent outputs are structured (plans = DAGs, code = trees, KGs = graphs). Chapter's "graph classification + pooling" stuff is relevant for **scoring** structured outputs; **generation** needs autoregressive graph models (not deeply covered here, but adjacent).
- **Agentic LLM hypothesis**: a generative graph model (graph-VAE, autoregressive graph gen) for plan generation, where the LLM produces a plan-DAG token-by-token but the structural validity is enforced by graph priors.

## TL;DR speculative roadmap

- **Near-term (1–2 years)**: graph-attention-conditioned tool selection; graph-RAG with proper GNN aggregation instead of LLM-summarization; AST/CFG-conditioned code agents; hyperbolic positional encodings for tree-structured agent inputs.
- **Medium-term (2–4 years)**: hybrid GNN-Transformer architectures with persistent graph-state encoders; KG-completion-grounded RAG; GoT/ToT replaced with learned MPNN-over-thought-graph; multi-agent systems with explicit GAT-based message routing.
- **Speculative / longer**: hyperbolic agent latent spaces for long-horizon planning; variational world-graph models with stochastic latents for Bayesian planning; mixed-curvature embeddings of agent state.
- **Probably dead ends**: pure spectral methods (domain-dependence), heavy non-Euclidean exotica (mixed-curvature) absent strong empirical wins, shallow-only methods (will be superseded by GNN+LLM hybrids).
