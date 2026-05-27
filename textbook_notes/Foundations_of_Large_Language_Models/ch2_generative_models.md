# Chapter 2: Generative Models

> History: language modeling = predict next token from prior context. Shannon (1951) → $n$-gram (dominant pre-2010) → Bengio et al. 2003 (FFN LM, distributed reps, word embeddings, breaks curse of dimensionality) → Word2Vec (~2012) → LSTM LMs → Transformer LMs → BERT-style pretraining → modern generative LLMs (GPT-family). Scaling token-prediction yields generalist behavior, motivating LLMs as foundation models.

---

## 2.1 A Brief Introduction to LLMs

- **Goal**: model $\Pr(x_0,\ldots,x_m)=\prod_{i=0}^{m}\Pr(x_i|x_0,\ldots,x_{i-1})$ (chain rule); log form: $\log\Pr = \sum_i \log\Pr(x_i|x_{<i})$.
  - $x_0 = \langle s\rangle$ start token. Convention: $\Pr(x_0|\cdot)=1$.
- **Generation**: $\hat{x}_i = \arg\max_{x_i\in\mathcal{V}}\Pr(x_i|x_{<i})$, append, repeat → autoregressive left-to-right decoding.

### 2.1.1 Decoder-only Transformers

- **Input**: token embedding $\mathbf{x}_i$ + position embedding → $\mathbf{e}_i \in \mathbb{R}^{d_e}$.
- **Stack**: $L$ Transformer blocks, each = self-attention sub-layer + FFN sub-layer.
- **Sub-layer architectures**:
	- **Post-norm**: $\text{output} = \text{LNorm}(F(\text{input}) + \text{input})$
	- **Pre-norm**: $\text{output} = \text{LNorm}(F(\text{input})) + \text{input}$ — **preferred for deep LLMs** (more stable).
- **QKV attention** (masked for causal LM):
$$\text{Att}_{\text{qkv}}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \text{Mask}\right)\mathbf{V}$$
	- **Mask**: $0$ if $i\leq k$, $-\infty$ otherwise → enforces left-context only.
- **Multi-head**: $F(\mathbf{H}) = \text{Merge}(\text{head}_1,\ldots,\text{head}_\tau)\mathbf{W}^{\text{head}}$, with $\text{head}_j = \text{Att}_{\text{qkv}}(\mathbf{H}\mathbf{W}_j^q, \mathbf{H}\mathbf{W}_j^k, \mathbf{H}\mathbf{W}_j^v)$, $\mathbf{W}_j^\cdot \in \mathbb{R}^{d\times d/\tau}$.
- **Output**: $\text{Softmax}(\mathbf{H}^L \mathbf{W}^o)$, $\mathbf{W}^o\in\mathbb{R}^{d\times|\mathcal{V}|}$.
- **Scale of "large"**: depth + width both big. Sample sizes (Table 2.2):
	- **GPT-1**: 117M, $L=12$, $d=768$. **GPT-2**: 1.5B, $L=48$, $d=1600$. **GPT-3**: 175B, $L=96$, $d=12288$.
	- **LLaMA2 70B**: $L=80$, $d=8192$, 64 heads. **LLaMA3.1 405B**: $L=126$, $d=16384$, 128/8 heads (GQA, 8 KV heads).
	- **DeepSeek-V3 671B**, **Falcon 180B**, **Gemma2 37B**, **Qwen2.5 72B**, **Mistral 7B**.
	- LLaMA3/Gemma2/Qwen2.5 all use **GQA** (queries-heads > KV-heads).

### 2.1.2 Training LLMs

- **Objective**: maximize log-likelihood
$$\hat{\theta} = \arg\max_\theta \sum_{\mathbf{x}\in\mathcal{D}} \sum_{i=1}^m \log\Pr_\theta(x_i|x_{<i})$$
- Standard SGD/Adam on next-token prediction loss; "embarrassingly standard" but unstable + expensive at scale.
- **Pain points**: distributed systems, 100s-1000s of GPUs, multiple training runs needed, instability in deep/wide nets → architecture tweaks.

### 2.1.3 Fine-tuning LLMs

- **Inference framing**: given prompt $\mathbf{x}$, generate $\mathbf{y}$: $\hat{\mathbf{y}} = \arg\max_\mathbf{y} \sum_{i=1}^n \log\Pr(y_i|x_0,\ldots,x_m,y_1,\ldots,y_{i-1})$.
- **Prompt-as-task**: replace placeholders in templates (e.g. "Is this sentence grammatical? Answer: ___").
	- Multiple template styles: QA, instruction-like, code-like.
- **Two routes**:
	- **Bake into pretraining**: add instruction data into pretrain corpus. Effective but expensive + lots of labeled data needed.
	- **Instruction fine-tuning (SFT)**: continue training pretrained LLM on (instruction, response) pairs — *de facto* standard.
- **SFT setup**:
	- Sample = $[\mathbf{y}_{\text{sample}}, \mathbf{x}_{\text{sample}}]$. Compute forward over full seq, **backprop loss only on $\mathbf{y}_{\text{sample}}$**.
	- Loss: $\mathcal{L}_{\hat{\theta}^+}(\text{sample}) = -\log\Pr_{\hat{\theta}^+}(\mathbf{y}_{\text{sample}}|\mathbf{x}_{\text{sample}})$.
	- $\tilde{\theta} = \arg\max_{\hat{\theta}^+} \sum_{\text{sample}\in\mathcal{D}_{\text{tune}}} \mathcal{L}$.
- **Data sizes**: tune data orders of magnitude smaller than pretrain (10s-100s of K samples; some work shows < 1k high-quality samples suffices).
- **Scaling # of fine-tuning tasks** helps (Chung et al. 2022).
- **Zero-shot learning** emerges: generalize to unseen tasks. Distinguishes generative LLMs from task-specific fine-tuned BERT.
- Heavy engineering: LR, batch size, # steps — many runs and evals.

### 2.1.4 Aligning LLMs with the World

- **Alignment** = guide LLM behavior to match human intentions/values (truthful, unbiased, harmless).
- **Two phases after pretrain**:
	- **SFT** (covered above) — straightforward but limited.
	- **Learning from human feedback** — model may still be unfactual/biased after SFT.
- **RLHF** (Ouyang et al. 2022):
	- **Agent**: LLM = policy $\Pr(\mathbf{y}|\mathbf{x})$.
	- **Reward model** $R_\omega(\mathbf{x},\mathbf{y})$: proxy environment, gives scalar.
	- **Pipeline**:
		1. Build initial policy via pretrain + SFT.
		2. Sample multiple outputs $\{\mathbf{y}_1,\ldots,\mathbf{y}_K\}$ per prompt, collect human preferences (rankings preferred over scalar ratings — easier annotation).
		3. **Train reward model** (same arch as LLM, smaller; uses end-token rep + linear head):
$$\text{Loss}_\omega = -\mathbb{E}_{(\mathbf{x},\mathbf{y}_{k_1},\mathbf{y}_{k_2})}\log\text{Sigmoid}(R_\omega(\mathbf{x},\mathbf{y}_{k_1}) - R_\omega(\mathbf{x},\mathbf{y}_{k_2}))$$ (Bradley-Terry pairwise ranking loss).
		4. **RL fine-tune policy**: $\tilde{\theta} = \arg\max_{\hat{\theta}^+} \mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\mathcal{D}_{\text{rlft}}} R_{\hat{\omega}}(\mathbf{x}, \mathbf{y}_{\hat{\theta}^+})$. Use **PPO** in practice for stability.
- **Why preference learning over supervised**:
	- Easier to *recognize* good output than to *produce* it.
	- RL **explores** sample space beyond labeled data via sampling → can discover policies not present in annotated data.

### 2.1.5 Prompting LLMs

- Prompt = entire model input. Prompting = adapting LLM without retraining → "prompt engineering".
- **Roles/personas**: "You are a researcher in developmental psychology..." → shapes generation style.
- **Tasks**: code debugging, dialog (multi-turn → history in prompt).
- **In-context learning (ICL)**: include demos in prompt, condition on them.
- **Chain-of-Thought (COT) prompting** (Wei et al. 2022c):
	- Decompose reasoning into intermediate steps in demos → model mimics reasoning structure.
	- Effective on math reasoning (e.g. GSM8K).
	- **One-shot COT** = 1 demo with reasoning; **few-shot COT** = several; **zero-shot COT** = append "Let's think step by step." (Kojima et al. 2022).
- **Zero/one/few-shot** terminology more general than COT — refers to # demos.

---

## 2.2 Training at Scale

### 2.2.1 Data Preparation

- Pretraining: trillions of tokens. Examples (Table 2.3):
	- GPT-3 175B: **0.5T**, Falcon-180B: **3.5T**, LLaMA2-65B: **1-1.4T**, PaLM-450B: **0.78T**, Gemma-7B: **6T**.
	- Mix: webpages + books + code + Wikipedia + papers + news + Q&A.
- **Issues**:
	- **Quality**: raw web is dirty (errors, toxicity, AI-generated content, fabrications). Filtering can drop ~90% of web-scraped data (Penedo et al. 2023).
	- **Diversity**:
		- Include code → boosts reasoning + COT, not just programming.
		- Multilingual → single model handles many langs; weak on low-resource.
	- **Bias**: e.g. gender bias ("nurses" → female). Balance across gender/ethnicity/dialect; English-centric → English-biased values.
	- **Privacy**: LLMs can memorize/regurgitate sensitive data. Anonymization + leak-detection systems + fine-tune to refuse.

### 2.2.2 Model Modifications

Larger LLMs → unstable training → need architecture changes.

#### 2.2.2.1 Layer Normalization

- **Pre-norm preferred for deep Transformers**: $\text{output} = \text{LNorm}(F(\text{input})) + \text{input}$.
- **Standard LayerNorm**: $\text{LNorm}(\mathbf{h}) = \alpha \cdot \frac{\mathbf{h}-\mu}{\sigma+\epsilon} + \beta$.
- **RMSNorm** (Zhang & Sennrich 2019, used in LLaMA): no recentering, only rescaling:
$$\text{LNorm}(\mathbf{h}) = \alpha \cdot \frac{\mathbf{h}}{\sigma_{\text{rms}}+\epsilon} + \beta, \quad \sigma_{\text{rms}} = \sqrt{\tfrac{1}{d}\sum_k h_k^2}$$

#### 2.2.2.2 FFN Activations

- **Standard FFN**: $\text{FFN}(\mathbf{h}) = \sigma(\mathbf{h}\mathbf{W}_h + \mathbf{b}_h)\mathbf{W}_f + \mathbf{b}_f$. Wide $d_h$ helpful at scale but hard to train.
- **ReLU**: $\max(0,\mathbf{h})$ — baseline.
- **GeLU**: $\mathbf{h}\Phi(\mathbf{h})$ — smoothed ReLU, weights by Gaussian CDF percentile. Used in BERT, GPT-3, BLOOM.
- **GLU family** (gated linear units):
	- $\sigma_{\text{glu}}(\mathbf{h}) = \sigma(\mathbf{h}\mathbf{W}_1+\mathbf{b}_1) \odot (\mathbf{h}\mathbf{W}_2+\mathbf{b}_2)$.
	- **GeGLU**: GeLU as inner $\sigma$ (used in Gemma).
	- **SwiGLU**: Swish ($\mathbf{h}\odot\text{Sigmoid}(c\mathbf{h})$) as inner $\sigma$. Used in **PaLM and LLaMA**.

#### 2.2.2.3 Removing Bias Terms

- Drop biases in LayerNorm, QKV projections, FFN.
- $\text{FFN}(\mathbf{h}) = \sigma(\mathbf{h}\mathbf{W}_h)\mathbf{W}_f$. Improves training stability (Chowdhery et al. 2022). Used in LLaMA, Gemma.

#### 2.2.2.4 Other

- Replace sinusoidal positional encoding with **rotary positional embeddings (RoPE)** → see §2.3.5.
- Other tricks: increasing batch size during training, mixed precision, careful LR schedules.

### 2.2.3 Distributed Training

Four parallelism strategies (Narayanan et al. 2021; Fedus et al. 2022):

- **Data Parallelism**:
	- Split minibatch $\mathcal{D}_{\text{mini}}$ → $\{\mathcal{D}^1,\ldots,\mathcal{D}^N\}$, replicate model on $N$ workers, sum grads:
$$\frac{\partial L(\mathcal{D}_{\text{mini}})}{\partial \theta_t} = \sum_{k=1}^N \frac{\partial L(\mathcal{D}^k)}{\partial \theta_t}$$
	- Nearly $N$-fold speedup if comms cheap.
- **Model Parallelism** (layer-wise / pipeline-naive):
	- Assign consecutive layer groups to different workers. Forward then backward in sequence.
	- **Problem**: only one worker active at a time → devices idle.
- **Tensor Parallelism**:
	- Split a single matrix multiply across devices. Slice $\mathbf{W}_h = [\mathbf{W}_h^1, \ldots, \mathbf{W}_h^M]$ along columns:
$$\mathbf{h}\mathbf{W}_h = [\mathbf{h}\mathbf{W}_h^1, \ldots, \mathbf{h}\mathbf{W}_h^M]$$
	- Each device computes one sub-matrix product. Two-level: matrix → sub-matrices (across GPUs) → tile-based parallelism (within GPU).
- **Pipeline Parallelism** (Harlap 2018, Huang 2019):
	- Split batch into **micro-batches**, pipeline them through layer-stage workers → overlapping forward/backward.
	- Trade-off: small micro-batches → low GPU util, task-switching overhead.
- **Bottlenecks**:
	- **Communication cost** dominates as nodes grow.
	- **Sync** stragglers — async helps but → stale grads, non-guaranteed convergence.
	- **Fault tolerance** at scale (node failures).
- **Mixed precision** (Micikevicius 2018):
	- FP16/FP8 for forward/backward, FP32/FP64 for parameter updates.
	- **Gradient accumulation** key — but FP-add is non-associative → small numerical drift across nodes; overflow/underflow at low precision.

### 2.2.4 Scaling Laws

- **Empirical**: deep nets' performance ~ power-law in data size (Hestnes 2017). Curve has 3 phases:
	1. Slow reduction (low data),
	2. Power-law reduction (sweet spot),
	3. Convergence to **irreducible error**.
- **Emergent abilities** (Wei et al. 2022b): some capabilities appear only past a model-size threshold.
- **Power-law forms** (let $x$ = quantity of interest, $\mathcal{L}$ = test loss):
	- Simple: $\mathcal{L}(x) = ax^b$.
	- **Kaplan et al. 2020**: $\mathcal{L}(N) = (N/8.8\!\times\!10^{13})^{-0.076}$, $\mathcal{L}(D) = (D/5.4\!\times\!10^{13})^{-0.095}$ ($N$=params, $D$=data).
	- With irreducible error: $\mathcal{L}(x) = ax^b + \epsilon_\infty$.
	- Joint: $\mathcal{L}(N,D) = aN^b + cD^d + \epsilon_\infty$.
	- **Chinchilla** (Hoffmann et al. 2022): $\mathcal{L}(N,D) = \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}} + 1.69$.
- Power-law forms monotonic → cannot capture double-descent. More sophisticated families exist (Alabdulmohsin 2022, Caballero 2023).
- **Use**: predict perf given compute budget, decide model size vs data.
- **Caveat**: lower test loss ≠ better downstream. Fine-tuning/prompting/data domain influence end metric.

---

## 2.3 Long Sequence Modeling

- **Three problem types**:
	- Long context $\mathbf{x}$ (e.g. summarize long doc).
	- Long output $\mathbf{y}$ (e.g. long-story generation).
	- Both long (e.g. doc translation).
- "Long-context LLMs" = focus of this section.
- **Core issue**: self-attention is $O(m^2)$ in seq length; KV cache scales linearly per token but grows huge at inference.
- **Two research strands**:
	- Efficient training/architectures for learning attention on long seqs.
	- Adapt pretrained LLMs to handle long seqs with little/no fine-tuning.

### 2.3.1 Optimization from HPC Perspectives

- **Low-precision**: INT8/INT16 for arithmetic — more memory throughput.
- **Hardware-aware kernels**: IO-aware attention (FlashAttention, Dao 2022; PagedAttention via vLLM, Kwon 2023).
- **Sequence parallelism** (Li 2023b, Korthikanti 2023):
	- Partition $\mathbf{K}$, $\mathbf{V}$ by rows → $\{\mathbf{K}^{[1]},\ldots,\mathbf{K}^{[n_u]}\}$ across nodes.
	- Attention weight $\alpha_{i,j} = \exp(\beta_{i,j}) / \sum_{j'}\exp(\beta_{i,j'})$, $\beta_{i,j} = \mathbf{q}_i\!\cdot\!\mathbf{k}_j/\sqrt{d} + \text{Mask}$.
	- Numerator local; denominator = sum of partial sums across nodes → **all-reduce**.
	- Weighted-sum step similarly splittable + reduced.

### 2.3.2 Efficient Architectures

- **Sparse attention**:
	- Most attention weights ≈ 0 in dense attention → keep only set $G\subseteq\{0,\ldots,i\}$:
$$\text{Att}_{\text{sparse}}(\mathbf{q}_i, \mathbf{K}_{\leq i}, \mathbf{V}_{\leq i}) = \sum_{j\in G}\alpha'_{i,j}\mathbf{v}_j$$
	- Patterns: sliding window (local), strided, global tokens, etc.
	- **Still must keep full KV cache** (only saves compute, not memory).
- **Linear attention** (Katharopoulos 2020):
	- Kernel feature map $\phi$: $\mathbf{q}'_i = \phi(\mathbf{q}_i)$, $\mathbf{k}'_i = \phi(\mathbf{k}_i)$. Remove softmax:
$$\text{Att}_{\text{linear}}(\mathbf{q}'_i, \mathbf{K}'_{\leq i}, \mathbf{V}_{\leq i}) = \frac{\mathbf{q}'_i \mu_i}{\mathbf{q}'_i \nu_i}$$
	- Recurrent updates: $\mu_i = \mu_{i-1} + \mathbf{k}'^\top_i \mathbf{v}_i$, $\nu_i = \nu_{i-1} + \mathbf{k}'^\top_i$.
	- **Constant per-step cost** — no KV cache needed.
- **Recurrent models**: hidden state $\mathbf{h}_i = f(\mathbf{h}_{i-1}, \text{input}_i)$, fixed memory footprint. Resurgent (Mamba — Gu & Dao 2023).

### 2.3.3 Cache and Memory

> Standard Transformer = global model storing all left-context. KV cache grows unbounded. Two families: **internal memory** (modify KV cache) and **external memory** (separate datastore).

#### 2.3.3.1 Fixed-size KV Cache (Internal)

Replace full $\mathbf{K}_{\leq i}, \mathbf{V}_{\leq i}$ with bounded $\text{Mem}$:

- **Window** (local attention): $\text{Mem} = (\mathbf{K}_{[i-n_c+1,i]}, \mathbf{V}_{[i-n_c+1,i]})$.
- **Moving-average summary**:
	- Unweighted: $\text{Mem} = \left(\tfrac{1}{n_c}\sum \mathbf{k}_j, \tfrac{1}{n_c}\sum \mathbf{v}_j\right)$.
	- Weighted: $\beta$ coefficients (learned or set), e.g. increasing → emphasize recent.
	- Cumulative recursive: $\text{Mem}_i = \frac{(\mathbf{k}_i,\mathbf{v}_i) + i\cdot\text{Mem}_{i-1}}{i+1}$.
- **Neural memory**: $\text{Mem} = \text{Update}(S_{\text{kv}}, \text{Mem}_{\text{pre}})$.
	- RNN-like: $\text{Mem} = f((\mathbf{k}_i,\mathbf{v}_i), \text{Mem}_{\text{pre}})$.
	- **Segment-level recurrence**: process segments of $n_s$ tokens, FIFO memory of segments (Transformer-XL, Block-Recurrent Transformer, RMT).
- **Compressive Transformer** (Rae 2019):
	- Two memories: $\text{Mem}$ (local, full-resolution) + $\text{CMem}$ (compressed long-range).
	- Local FIFO pops old → compressed (via compression net) → appended to $\text{CMem}$.
	- Attention over $[\text{Mem}, \text{CMem}]$.
- **Global tokens** (e.g. first few tokens always attended) — stabilizes softmax distribution at very long contexts (Xiao 2024 / "attention sink").

#### 2.3.3.2 Memory-based Models (External)

- **$k$-NN over context datastore**:
	- Store $\{(\mathbf{k}_j, \mathbf{v}_j)\}$ in vector DB. For query $\mathbf{q}_i$, retrieve top-$k$ → $\text{Mem}_{k\text{nn}}$.
	- Combine with local memory:
$$\text{Att}(\mathbf{q}_i, \text{Mem}, \text{Mem}_{k\text{nn}}) = \mathbf{g}\odot\text{Att}_{\text{local}} + (1-\mathbf{g})\odot\text{Att}_{k\text{nn}}$$ (Wu 2021), $\mathbf{g}$ from learned gate.
- **$k$-NN LM** (Khandelwal 2020):
	- Datastore = (hidden state, next-token) pairs from training.
	- At inference: retrieve $k$ nearest hidden states → distribution over their next-tokens:
$$\Pr_{k\text{nn}}(\cdot|\mathbf{h}_i) = \text{Softmax}([-d_0,\ldots,-d_{|V|}])$$
	- Interpolate: $\Pr(\cdot|\mathbf{h}_i) = \lambda\Pr_{k\text{nn}} + (1-\lambda)\Pr_{\text{lm}}$.
- **RAG** (retrieval-augmented generation):
	- Retrieve $k$ docs via IR, splice into prompt: $\mathbf{x}' = g(\mathbf{c}_1,\ldots,\mathbf{c}_k, \mathbf{x})$, then $\Pr(\mathbf{y}|\mathbf{x}')$.
	- No model arch changes — purely input augmentation.

#### 2.3.3.3 Memory Capacity

- KV cache size or vector-DB size ≈ memory capacity, distinct from parameter count.
- Trade-off perf vs memory footprint. Streaming (continuously growing context) favors fixed-size / recurrent.

### 2.3.4 Sharing across Heads and Layers

- KV cache size: $O(L\cdot\tau\cdot d_h \cdot m)$ ($L$ layers, $\tau$ heads, $d_h$ head dim, $m$ tokens).
- **Multi-head attention (MHA)**: each head has its own $\mathbf{Q},\mathbf{K},\mathbf{V}$.
- **Multi-query attention (MQA)** (Shazeer 2019):
	- Single $(\mathbf{K},\mathbf{V})$ shared by all $\tau$ heads; each head has its own $\mathbf{q}^{[j]}$.
	- Cache: $O(L\cdot d_h\cdot m)$ (drops $\tau$ factor).
- **Grouped-query attention (GQA)** (Ainslie 2023):
	- $n_g$ groups, each shares one KV set. $g(j)$ = group of head $j$, $\text{head}_j = \text{Att}_{\text{qkv}}(\mathbf{q}^{[j]}, \mathbf{K}^{[g(j)]}, \mathbf{V}^{[g(j)]})$.
	- $n_g=\tau$ → MHA; $n_g=1$ → MQA.
	- Cache: $O(L\cdot n_g\cdot d_h\cdot m)$. Tune $n_g$ for compute/expressiveness trade.
	- LLaMA3, Gemma2, Qwen2.5, DeepSeek-V3 all use GQA variants.
- **Cross-layer sharing**: share KV/attn weights across layers (Dehghani 2018, Lan 2020 / ALBERT; Xiao 2019; Brandon 2024). Layer-$l$ query attends to KV cache from layer-$l-1$.

### 2.3.5 Position Extrapolation and Interpolation

- Transformers are order-invariant → need positional info.
- **Additive position embeddings**: $\mathbf{e}_i = \mathbf{x}_i + \text{PE}(i)$.
	- **Learned**: vector per position. Cannot handle positions beyond training range.
	- **Sinusoidal**: extends to any length but degrades when length >> train length.
- Two generalization strategies:
	- **Extrapolation**: learned model still meaningful outside trained range (e.g. fit function, evaluate outside domain).
	- **Interpolation**: rescale new range $[1,20]$ into trained $[1,10]$.

#### 2.3.5.1 Attention with Learnable Biases (Relative PE)

- **Relative PE** (Shaw 2018): bias on QK product
$$\alpha(i,j) = \text{Softmax}\!\left(\tfrac{\mathbf{q}_i\mathbf{k}_j^\top + \text{PE}(i,j)}{\sqrt{d}} + \text{Mask}(i,j)\right)$$
- **T5 bias** (Raffel 2020):
	- Bucketize offset $d(i,j) = i-j$ into $n_b+1$ buckets, each gets learnable $u_{b(i-j)}$.
	- First half: 1 bucket per offset. Second half: **logarithmically increasing** bucket sizes. Last bucket = catch-all → handles arbitrary length.
$$b(i-j) = \begin{cases} i-j & 0\leq i-j < (n_b{+}1)/2 \\ \min\!\Big(n_b, \tfrac{n_b{+}1}{2}+\lfloor \tfrac{\log(i-j)-\log((n_b{+}1)/2)}{\log(\text{dist}_\max)-\log((n_b{+}1)/2)}\cdot\tfrac{n_b{+}1}{2}\rfloor\Big) & i-j\geq (n_b{+}1)/2 \end{cases}$$
	- Shared parameter for similar offsets → generalizes; controls overfitting.

#### 2.3.5.2 Attention with Non-learned Biases

- **ALiBi** (Press 2022): linear bias
$$\text{PE}(i,j) = -\beta\cdot(i-j) = \beta(j-i)$$
$$\alpha(i,j) = \text{Softmax}\!\left(\tfrac{\mathbf{q}_i\mathbf{k}_j^\top + \beta(j-i)}{\sqrt{d}} + \text{Mask}(i,j)\right)$$
	- Penalty grows linearly with offset.
	- Per-head $\beta_k = 1/2^{8/k}$ (geometric).
	- No training of PE → directly handles any length.
- **Other variants** (Table 2.4):
	- **Kerple** (Chi 2022): power $-\beta_1(i-j)^{\beta_2}$ or log $-\beta_1\log(1+(i-j))$.
	- **Sandwich** (Chi 2023): $\sum_k \cos((i-j)/10000^{2k/\bar{d}})$.
	- **FIRE** (Li 2024b): $f(\psi(i-j)/\psi(\max(m_{\text{len}}, i)))$, $\psi$ monotonic.

#### 2.3.5.3 Rotary Positional Embedding (RoPE)

- **Idea**: encode position as **rotation in complex/2D plane** rather than addition. $\mathbf{e}_i = \mathbf{x}_i R(i)$.
- **2D case**: rotation matrix $R_\theta = \begin{bmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{bmatrix}$. $\text{Ro}(\mathbf{x},\theta) = \mathbf{x}R_\theta$.
- Position $t$ = rotating by $t\theta$ → $\text{Ro}(\mathbf{x}, t\theta) = \mathbf{x} R_{t\theta}$. Magnitude preserved (token "meaning" intact).
- **Complex view**: $\mathbf{x}\to x' = x_1 + \mathrm{i}x_2$, rotation = multiply by $e^{\mathrm{i}t\theta}$.
	- Inner product of pos-$t$ and pos-$s$ representations: $\langle C(\mathbf{x},t\theta), C(\mathbf{y},s\theta)\rangle = (\mathbf{x}'\overline{\mathbf{y}'})e^{\mathrm{i}(t-s)\theta}$ → depends on **offset $t-s$** only ⇒ self-attention naturally encodes relative position.
- **General $d$-dim**: split into $d/2$ pairs, each pair has its own rotation angle $\theta_k$:
$$\text{Ro}(\mathbf{x}, t\theta) = \mathbf{x}\,\text{diag}(R_{t\theta_1}, R_{t\theta_2}, \ldots, R_{t\theta_{d/2}})$$
	- Typically $\theta_k = 10000^{-2(k-1)/d}$ (mirrors sinusoidal).
- **Efficient elementwise impl**: rewritten as Hadamard products with $[\cos t\theta_k]$ and $[\sin t\theta_k]$ vectors (Eq. 2.93).
- Used in LLaMA, PaLM, Mistral, most modern LLMs.

#### 2.3.5.4 Position Interpolation

- Trained on $[0, m_l]$; want to use on $[0, m]$ with $m \gg m_l$.
- Period of RoPE dim $k$: $T_k = 2\pi b^{2(k-1)/d}$.
- **Linear PI** (Chen 2023c): scale period by $m/m_l$:
$$\text{Ro}'(\mathbf{x}_i, i\theta) = \text{Ro}(\mathbf{x}_i, \tfrac{m_l}{m}i\theta)$$
	- Equivalent to compressing $[0,m]$ into $[0,m_l]$ before applying RoPE.
- **Base scaling** (NTK-aware scaled RoPE — first proposed on Reddit/LocalLLaMA):
	- Scale base $b \to \lambda b$ such that last-dim period matches linear PI target:
$$\lambda = (m/m_l)^{d/(d-2)}$$
	- **Non-uniform** across dims: high-freq dims barely scaled, low-freq dims scaled most → preserves local resolution while extending range.
	- Improvements: **YaRN** (Peng 2024), **LongRoPE** (Ding 2024).

### 2.3.6 Remarks

#### Need for Long Context
- "Infinite context" = continuously reading words. Approaches:
	- Recurrent/fixed-memory (time-series view).
	- Continuous-space attention (Martins 2022).
- **Question**: can we compress infinite context into small model? Are all context tokens useful?
	- **LLMs as in-context compressors** (Deletang 2024).
	- Studies of whether earlier features already suffice for prediction (Pal 2023, Wu 2024).
- Task-dependence: summarization needs distillation; retrieval needs memorization.

#### Pre-training or Adapting?
- Common recipe: pretrain on general data, fine-tune for long-context.
- RoPE / relative PE → can be pretrained directly + extrapolate, but fine-tuning on long seqs is often more effective.
- **External memory** (RAG, k-NN): pretrained LM **frozen**, fine-tune for retrieval use.
- **Architecture swap**: pretrain with full attention, swap to sparse attention at fine-tune.

#### Evaluating Long-context LLMs
- **Perplexity**: reflects local context > global → poor.
- **Synthetic**:
	- **Needle-in-a-haystack**: hide info in long doc → recall it.
	- **Passkey retrieval** (Mohtashami & Jaggi 2024, Chen 2023c).
	- **Copy memory tasks** (originally for RNNs, Hochreiter & Schmidhuber 1997).
- **Real**: long-doc summarization, long-doc QA, code completion.
- **Caveats**:
	- Tests model's ability to *pick out* info, not necessarily "understand" full context.
	- Sensitive to prompt; benchmarks small-scale → mismatch with real use.
	- Open issues: context length limits, latency.

---

## 2.4 Summary

- Chapter introduced LLMs: arch (decoder-only Transformer), training (MLE on next-token), fine-tuning (SFT + RLHF) and prompting (ICL, COT).
- **Scaling at training**:
	- Data: quality, diversity, bias, privacy.
	- Architecture mods for stability: pre-norm, RMSNorm, SwiGLU/GeGLU, no biases.
	- Distributed: data/model/tensor/pipeline parallelism + mixed precision.
	- Scaling laws (Kaplan, Chinchilla) — power-law + irreducible error; guide compute allocation.
- **Long sequence modeling**:
	- HPC: low precision, FlashAttention, sequence parallelism.
	- Architectures: sparse attention, linear attention, recurrent revivals.
	- Cache: window/MA/recurrent/compressive memory; external memory (k-NN, k-NN LM, RAG).
	- KV-cache reductions: MQA, GQA, cross-layer sharing.
	- Position: relative PE (T5), non-learned biases (ALiBi, Kerple, Sandwich, FIRE), RoPE + linear/base-scaling interpolation.
	- Eval still open: synthetic (needle-in-haystack) vs real long tasks.
- **Big picture**: scaling token prediction yields general capabilities (LLMs as foundation models); scaling laws drive the field. Inference-time compute scaling (o1-style) also boosts reasoning. Simply scaling parameters not sufficient for AGI on its own.
