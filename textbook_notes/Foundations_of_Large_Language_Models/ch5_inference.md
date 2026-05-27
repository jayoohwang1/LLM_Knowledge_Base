# Chapter 5: Inference

**Goal**: $\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} \Pr(\mathbf{y}|\mathbf{x})$ — classical formulation since speech recognition / SMT.

- **Two sub-problems**:
	- **Model computation**: efficiently compute $\Pr(\mathbf{y}|\mathbf{x})$.
	- **Search**: efficiently find (sub-)optimal $\hat{\mathbf{y}}$ (exhaustive search is exponential).
- **Why it matters now**: long-context prompts + chain-of-thought + inference-time scaling (o1, R1) make inference compute critical.

---

## 5.1 Prefilling and Decoding

### 5.1.1 Preliminaries

- **Notation**:
	- $\mathbf{x} = x_0 \ldots x_m$, $x_0 = \langle\text{SOS}\rangle$ (input prompt, $m+1$ tokens).
	- $\mathbf{y} = y_1 \ldots y_n$ (response).
	- $\mathbf{y}_{<i} = y_1 \ldots y_{i-1}$.
	- $[\mathbf{x},\mathbf{y}] = x_0 \ldots x_m y_1 \ldots y_n$ (concatenation, a.k.a. $\text{seq}_{\mathbf{x},\mathbf{y}}$).
- **Decoder-only modeling**: $\log\Pr(\mathbf{y}|\mathbf{x}) = \log\Pr([\mathbf{x},\mathbf{y}]) - \log\Pr(\mathbf{x})$.
	- Each side computed via chain rule: $\log\Pr(\mathbf{x}) = \sum_{j=1}^m \log\Pr(x_j|\mathbf{x}_{<j})$.
	- In practice, just compute $\log\Pr(\mathbf{y}|\mathbf{x}) = \sum_{i=1}^n \log\Pr(y_i|\mathbf{x},\mathbf{y}_{<i})$ directly.
- **Architecture** (single Transformer decoder layer):
	- $\Pr(\cdot|\mathbf{x},\mathbf{y}_{<i}) = \text{Softmax}(\mathbf{HW}^o)_{m+i}$, with $\mathbf{H} = \text{Dec}([\mathbf{x},\mathbf{y}_{<i}])$.
	- $\mathbf{W}^o \in \mathbb{R}^{d\times|V|}$: output projection.
- **Self-attention**: $\text{Att}(\mathbf{q}_{i'},\mathbf{K},\mathbf{V}) = \text{Softmax}(\mathbf{q}_{i'}\mathbf{K}^T/\sqrt{d})\mathbf{V}$.
	- $O(L \cdot \text{len}^2)$ time for sequence of length $\text{len}$ across $L$ layers — **quadratic in length**.
- **KV cache** $(\mathbf{K},\mathbf{V})$: store K/V pairs from prior tokens to avoid recomputation.
	- Update: $\mathbf{K} \leftarrow \text{Append}(\mathbf{K},\mathbf{k}_{i'})$, $\mathbf{V} \leftarrow \text{Append}(\mathbf{V},\mathbf{v}_{i'})$.
	- Cache = $\{\text{cache}^1, \ldots, \text{cache}^L\}$ per layer.

### 5.1.2 Two-phase Framework

- **Prefilling**: encode $\mathbf{x}$ all at once, build KV cache.
	- $\text{cache} = \text{Dec}_{\text{kv}}(\mathbf{x})$.
	- All queries packed into matrix $\mathbf{Q}$, single masked self-attn: $\text{Att}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Softmax}(\mathbf{Q}\mathbf{K}^T/\sqrt{d} + \mathbf{Mask})\mathbf{V}$.
	- Causal mask: entry $(i,j) = -\infty$ if $i<j$.
	- Like BERT-style encoding but **unidirectional** (LM mask).
	- **Compute-bound** — bottleneck = GPU FLOPs, not memory bandwidth.
- **Decoding**: autoregressive token-by-token using cached KV.
	- $\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} \Pr(\mathbf{y}|\text{cache})$.
	- **Memory-bound** — frequent KV cache reads dominate.
	- Cost grows with generated length; usually more expensive than prefilling overall.

| Aspect | Prefilling | Decoding |
|---|---|---|
| **Goal** | Set up context $\mathbf{x}$ | Generate $\mathbf{y}$ |
| **Visibility** | All tokens at once | Sequential |
| **Context use** | Build cache | Use cache |
| **Bottleneck** | Compute-bound | Memory-bound |
| **Cost** | High | Very High |

### 5.1.3 Decoding Algorithms

- **Search tree**: each node = prefix; children = vocab extensions.
	- BFS over levels = left-to-right generation.
	- $Y_i = \text{Prune}(Y_{i-1} \times V)$, full space $\mathcal{Y} = \bigcup_{i} \Psi(Y_i)$.
	- Goal: keep $|Y_i| \ll |Y_{i-1}|\cdot|V|$ to avoid exponential blowup.

#### Greedy Decoding

- At each step pick top-1: $y_i^{\text{top1}} = \arg\max_{y_i\in V}\log\Pr(y_i|\mathbf{x},\mathbf{y}_{<i})$.
	- Accumulated log-prob: $\log\Pr(\mathbf{y}|\mathbf{x}) = \log\Pr(\mathbf{y}_{<i}|\mathbf{x}) + \log\Pr(y_i|\mathbf{x},\mathbf{y}_{<i})$ — first term fixed wrt $y_i$.
- **Pros**: fast, simple. **Cons**: suboptimal — kills good prefixes early.

#### Beam Decoding

- Maintain top-$K$ partial sequences (**beam width** $K$).
	- $\{y_i^{\text{top1}},\ldots,y_i^{\text{topK}}\} = \arg\text{TopK}_{y_i\in V}\Pr(y_i|\mathbf{x},\mathbf{y}_{<i})$.
	- Typically $K=2$ or $K=4$ suffices in LLMs.
- Large beam widths give diminishing returns.

#### Sampling-based Decoding

- Greedy/beam = deterministic → low diversity, bad for creative tasks.
- **Top-$k$ sampling**: restrict to top-$k$ tokens $\overline{V}_i = \{y_i^{\text{top1}},\ldots,y_i^{\text{topk}}\}$, renormalize, sample.
	- $\overline{\Pr}(y_i|\cdot) = \Pr(y_i|\cdot) / \sum_{y_j\in\overline{V}_i}\Pr(y_j|\cdot)$.
- **Top-$p$ (nucleus) sampling** [Holtzman 2020]: smallest set with cumulative prob $\geq p$.
	- Adapts size to distribution sharpness.
- **Temperature** $\beta$: $\overline{\Pr}(y_i|\cdot) = \exp(u_{y_i}/\beta)/\sum_j\exp(u_{y_j}/\beta)$.
	- Low $\beta$ → sharper, more deterministic (top-$p$ with $p=1,\beta\to 0$ = greedy).
	- High $\beta$ → uniform, more diverse.

#### Decoding with Penalty Terms

- Modify objective: $\hat{\mathbf{y}} = \arg\max[\Pr(\mathbf{y}|\mathbf{x}) - \lambda\cdot\text{Penalty}(\mathbf{x},\mathbf{y})]$.
- **Related**: MBR decoding [Kumar & Byrne 2004] — minimize expected risk over output distribution.
- **Penalty types**:
	- **Repetition penalty**: discourage repeated tokens/phrases.
	- **Length penalty**: hit target length (summarization).
	- **Diversity penalty**: in beam search, push hypotheses apart.
	- **Constraint-based**: terminology/tone matching.
- Can also operate on **hidden states** [Su et al. 2022] — penalize representation similarity to past tokens (anti-degeneration).

#### Speculative Decoding

- **Idea**: small fast **draft model** $\Pr_q$ proposes, large **verification model** $\Pr_p$ checks in parallel.
- **Algorithm** [Leviathan et al. 2023]:
	1. Draft generates $\tau$ tokens autoregressively: $\hat{y}_{i+t} = \arg\max\Pr_q(\cdot|\ldots\hat{y}_{i+t-1})$.
	2. Verification model evaluates all $\tau$ in **parallel** (one forward pass): $p(\hat{y}_{i+t})$.
	3. Accept/reject per token:
		- Accept if $q(\hat{y}_{i+t}) \leq p(\hat{y}_{i+t})$.
		- Else reject with prob $1 - p/q$.
		- $n_a = \min\{t-1 \mid r_t > p(\hat{y}_{i+t})/q(\hat{y}_{i+t})\}$, $r_t \sim U(0,1)$.
	4. Use verification model to predict 1 more token after the $n_a$ accepted ones.
- **Trade-off**: draft too small → low accept rate, small $n_a$; too large → no speedup.

#### Stopping Criteria

- $\langle\text{EOS}\rangle$ token emission.
- Max length cap.
- Time/compute budget (chatbots).
- Probability-threshold stopping.
- Repetition detection → halt on loops.

### 5.1.4 Evaluation Metrics

- **Quality**: perplexity, F1, robustness (adversarial/OOD), fluency/coherence/relevance/diversity, human eval, fairness/ethics.
- **Efficiency** [Nvidia 2025]:
	- **Request Latency**: end-to-end request→response time.
	- **Throughput**: tokens or requests per second.
	- **TTFT** (Time to First Token): mostly prefilling + first-token gen.
	- **ITL** (Inter-token Latency): per-token decoding speed.
	- **TPS** (Tokens Per Second): generation rate.
	- **Resource Utilization**: CPU/GPU/memory.
- Also: **energy efficiency**, **cost efficiency** (deployment-scale).

---

## 5.2 Efficient Inference Techniques

### 5.2.1 More Caching

- **Response-level cache**: hash table mapping $\mathbf{x} \to \mathbf{y}$ — bypass LLM on exact-match queries.
- **Prefix cache**: store $(\mathbf{x}_{<i}, \text{cache}_{<i})$ pairs.
	- For new $\mathbf{x}'$ sharing prefix $\mathbf{x}'_{<k} = \mathbf{x}_{<k}$, load $\text{cache}_{<k}$, only compute remainder.
	- Hash prefix tokens for $O(1)$ lookup.
	- Manage memory with **LRU** etc.

### 5.2.2 Batching

- Process $B$ sequences in one forward pass for GPU utilization.
- **Padding** (e.g., left-pad short seqs) to equalize lengths in batch.
- **Throughput vs latency trade-off**:
	- Small batch → low latency, underutilized GPU.
	- Large batch → high throughput, higher latency.
- **Length bucketing**: group similar-length seqs to reduce padding waste.
- **Disaggregation** [Wu 2023a, Patel 2024, Zhong 2024]: separate prefill GPU and decode GPU.
	- Concat short seqs into long ones for prefill → maximize compute.
	- Trade-off: must transfer KV cache between devices → needs high-BW low-latency interconnect.

#### Scheduling

- System has **Scheduler** (queues/dispatches) + **Inference Engine** (executes).
- **Request-level scheduling**: form batch, run to completion, then next batch. No mid-flight changes [Timonin et al. 2022].
- **Iteration-based scheduling**: scheduler interacts with engine at **every token step** — dynamic batch.

#### Continuous Batching

- Iteration-based scheduling method from **Orca** [Yu et al. 2022].
- Iteration = one prefill step OR one decode step.
- Per iteration: add new arrivals; remove completed seqs.
- **Prefilling-prioritized**: add new requests ASAP → higher throughput, may extend earlier-request latency.
- Contrast: standard batching = **decoding-prioritized** [Agrawal et al. 2024] — finish current batch first.
- **Cost**: continuous reorganization → memory fragmentation, overhead.

#### PagedAttention

- vLLM [Kwon et al. 2023]. Inspired by **OS paging**.
- KV cache split into fixed-size non-contiguous **pages** instead of one contiguous block.
- **Benefits**:
	- Eliminates internal/external fragmentation (no need for one big block per sequence).
	- Enables dynamic sequence growth without reallocation.
	- Parallel reads/writes across blocks possible.
- **Cost**: non-contiguous access overhead — usually small if paging well-designed.

#### Chunked Prefilling

- Problem: in iter-level scheduling, long prefills block other seqs' decoding → latency spikes.
- **Solution** [Agrawal et al. 2023]: split prefill into chunks $P_{x,1}, P_{x,2}, \ldots$, each = one iteration.
- Decoding of other seqs can interleave with chunked prefill → reduce idle time.
- **Cost**: extra memory (intermediate KV) + scheduling complexity; loses single-pass parallelism.

### 5.2.3 Parallelization

- Reuses pre-training parallelism (model, tensor, pipeline; MoE expert sharding).
- **Inference-specific challenges**:
	- Variable-length sequences (no pre-prepared batches).
	- Load balancing across devices with bursty arrivals.
	- Heterogeneous hardware + tight latency SLAs.
- **Serving frameworks**: vLLM, TensorRT-LLM, TGI [Pope 2023, Li 2024a].

### 5.2.4 Remarks

- Key trade-offs are about balancing several axes:
	- **Speed vs accuracy**: quantization/pruning/distillation gain speed, may lose quality.
	- **Memory-compute trade-off**:
		- KV cache: trades memory for less recompute.
		- Chunked/windowed attention: trades context for less memory.
	- Extends to **memory-compute-accuracy triangle** with FP16/INT8 precision.
- Additional dimensions:
	- **Throughput vs latency**.
	- **Generalization vs specialization** (general LLM vs many fine-tuned ones).
	- **Energy efficiency vs performance**.

---

## 5.3 Inference-time Scaling

- Scaling laws apply beyond pretraining → fine-tuning + inference [Snell 2025].
- **Inference-time (test-time) compute scaling** categories:
	- **Context scaling** (5.3.1).
	- **Search scaling** (5.3.2).
	- **Output ensembling** (5.3.3).
	- **Generating + verifying thinking paths** (5.3.4).

### 5.3.1 Context Scaling

- Inject more useful info into the prompt.
- **Few-shot prompting**: in-context examples → implicit task learning.
- **CoT prompting**: intermediate reasoning steps → better complex reasoning.
- **RAG**: dynamically retrieve external docs → grounded, up-to-date.
- **Challenges**:
	- Finite context window.
	- **"Lost in the middle"** — long contexts attend poorly to middle content.
	- Need careful selection/structuring of context.
- Detailed methods in Ch. 2/3.

### 5.3.2 Search Scaling

- Two aspects:
	- **Scale output length**: long thinking paths help reasoning (o1, R1).
	- **Scale search space**: e.g., increase beam width $K$.
- **Structured search**:
	- **Tree of Thoughts** [Yao 2024]: navigate tree/graph of reasoning steps, each node = partial solution.
	- **MCTS-inspired decoding**: stochastic exploration scored by heuristics/reward.
- **Diminishing returns**: bigger search ↔ exponentially more compute; sweet spot exists.

### 5.3.3 Output Ensembling

- Combine multiple outputs to reduce individual model error.
- **Methods**:
	- Average next-token distributions across models.
	- **Majority voting** on discrete answers.
	- Re-ranking with separate scorer / meta-learner.
- **Cost**: latency + complexity of managing multiple models.
- Benefits **diminish past a threshold** in ensemble size.
- Best with **diverse** members (different data/arch/objective).
- Scaling also = robustness + exploration (not just quality):
	- Variance reduction.
	- Increased chance of novel/superior solutions.

### 5.3.4 Generating and Verifying Thinking Paths

- Reasoning is a **dynamic process** (trial/error, backtracking) — simple CoT often insufficient.
- **Two paradigms**:
	- **Training-free**: prompting + search at inference only.
	- **Training-based**: SFT/RL/distillation to bake in reasoning.

#### 5.3.4.1 Solution-level Search with Verifiers

- Solution = sequence of reasoning steps: $\mathbf{y} = (a_1, a_2, \ldots, a_{n_r})$.
- **Components**:
	- **Search algorithm**: generates candidates $\mathcal{D}_c = \{\mathbf{y}_1,\ldots,\mathbf{y}_K\}$.
	- **Verifier** $V(\mathbf{y})$: scores each candidate.
	- Output: $\hat{\mathbf{y}} = \arg\max_{\mathbf{y}\in\mathcal{D}_c} V(\mathbf{y})$.
- **Verifier types**:
	- **Tool verifiers**: theorem provers, compilers, unit tests.
	- **Learned reward models**: trained on preference data.
	- **LLM-as-verifier**: prompted evaluator (possibly stronger LLM).
	- **Heuristic (majority vote)**: pick most frequent answer.
- **Parallel scaling** [Brown 2024, Snell 2024]:
	- Sample $K$ independent solutions (vary temperature), pick best — essentially **best-of-$N$**.
- **Sequential scaling** [Gou 2024, Zhang 2024]:
	- Iterative critique-refine: $\mathbf{y}_{k+1} = \text{Refine}(\mathbf{x}, \mathbf{y}_k, \text{Feedback}(\mathbf{y}_k))$.
	- Verifier guides generation (not just final pick) — **self-refinement** [Shinn 2023, Madaan 2024].
- **Tree-structured search** over complete solutions: expand a node = generate variant/alternative solutions.

#### 5.3.4.2 Step-level Search with Verifiers

- Verify/prune at **intermediate steps** $\mathbf{a}_{\leq i} = (a_1,\ldots,a_i)$, not just full solutions.
- **Why**: one bad step poisons rest of chain; early pruning saves compute.
- **Components**:
	- **Node**: partial path $\mathbf{a}_{\leq i}$.
	- **Expansion**: LLM samples next step candidates $\{a_{i+1}^{(1)},\ldots,a_{i+1}^{(M)}\}$.
	- **Verification**: $V(\cdot)$ scores each step in context of $\mathbf{a}_{\leq i}$ and $\mathbf{x}$.
	- **Search policy**: extend/prune partial paths; backtrack on low scores.
- **Comparison to beam search**: beam = step-level with implicit verifier = output prob.
	- Explicit verifiers can be much more sophisticated than raw LM probs.
- **Process Reward Models (PRMs)**:
	- Score each step (vs **Outcome Reward Models (ORMs)** scoring only final answer).
	- More fine-grained supervisory signal.
	- **Costly**: requires step-level human annotations.
	- **Synthetic alternative**: use a strong teacher LLM to generate reasoning paths + alternative steps + evaluate them → automatic step-level labels.
	- PRMs can generalize from one domain to others with little retraining.
- **Caveats**: frequent verification (esp. LLM-based) → compute/latency; bad verifier can mislead search.

#### 5.3.4.3 Encouraging Long Thinking

- Direct ways to make the model think longer:
	- Prompt explicitly: "think step by step", "deliberate further".
	- Decoding-level: raise token limits, penalize short outputs.
	- Multi-stage generation: incremental builds.

#### 5.3.4.4 Training-based Scaling

- Improve reasoning by **modifying parameters** (complementary to training-free).
- **Methods**:
	- **Fine-tuning on reasoning data**: math/code/logic datasets with step-by-step solutions.
	- **RL for reasoning**:
		- Verifier ⇒ reward model. LLM = policy.
		- Outcome reward model (ORM) for final answer; PRM for steps.
		- Even simple rule-based rewards (e.g., bonus for longer outputs) work.
	- **Knowledge distillation**: small student mimics big teacher's reasoning traces → cheap deployment.
	- **Iterative refinement (self-improve loop)**:
		- Generate solutions → verify (human/auto) → keep correct → fine-tune → repeat.
- **Pros**: better inherent reasoning, more efficient inference (less search), better generalization.
- **Cons**: dataset curation cost, RL/FT compute cost, risk of overfitting to training reasoning style.

---

## 5.4 Summary

- Inference for sequential data: tradition from speech recognition / SMT (beam search, pruning — Wozengraft 1961, Viterbi 1967, Forney 1972, Koehn 2010).
- Deep models initially shifted focus away from inference (simple search worked).
- **LLMs re-centered inference** for two reasons:
	- **High deployment cost** — efficiency critical for high-concurrency/low-latency serving.
	- **Long sequences** in reasoning/math; scaling inference itself improves quality.
- Modern inference = systems engineering as much as ML (scheduling, paging, kernel design beyond NLP).
- Hands-on practice is the practical path to mastery.
