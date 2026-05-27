# MemoryLLM: Plug-n-Play Interpretable Feed-Forward Memory for Transformers

**Authors**: Ajay Jaiswal, Lauren Hannah, Han-Byul Kim, Duc Hoang, Arnav Kundu, Mehrdad Farajtabar, Minsik Cho (Apple)
**Venue**: arXiv 2602.00398, Jan/Feb 2026
**TL;DR**: Redesign the transformer block so that **FFNs are decoupled from self-attention and the residual stream**, taking the *raw token embedding* $X_0$ as input instead of the layer-$L$ residual $\tilde X_L$. This makes each FFN a **context-free token-indexed key-value memory** over the vocabulary, which (a) is mechanistically interpretable via a TKV (token-key-value) framework, and (b) can be **precomputed once per vocab token** and stored as **Token-wise Lookups (ToLs)** that get offloaded VRAM->disk and prefetched on demand. **Flex-MemoryLLM** trades a little interpretability for performance by splitting FFN params into a context-aware compute branch (FFN-C) and the context-free memory branch (FFN-M).

---

## 1. Motivation

- **FFNs hold ~2/3 of LLM params** but are way less studied than self-attention.
	- Hard to study because in a standard block the FFN input $\tilde X_L = X_L + \text{Attn}(X_L)$ is a **non-interpretable additive mixture** of self-attention output + residual stream.
	- Geva et al. (2021/2022), Dar et al. (2023), Nichani et al. (2024) interpret FFNs as **neural key-value memories** but suffer from:
		1. FFN keys live in a *contextualized latent space* that drifts across layers as the residual stream evolves.
		2. Requires laborious **reverse engineering** (calibration corpus + n-gram / template mining + manual annotation) to label what keys mean.
		3. Unclear influence of this FFN memory on **downstream tasks**.
- **Core question**: *Can we disentangle FFNs from self-attention so they encode deterministic memory mapped to a finite human-interpretable vocabulary (i.e. the tokens themselves)?*
- **Approach**: Don't probe pretrained models, **design + pretrain a new architecture** that enforces this separation.
- **Adjacent work that inspired it**:
	- **MoLE (Mixture of Lookup Experts, Jie 2025)**: majority of MoE experts can be trained directly on token embeddings; but MoLE still uses a context-aware router. MemoryLLM goes further: kill the context dependence entirely.
	- **Persistent memory in attention** (Sukhbaatar et al. 2019), **product-key memory layers** (Lample et al. 2019), **PEER** (He 2024) - all memory-augmented FFN-ish ideas.

> Disambiguation: this is **NOT** the 2024 wangyu-ustc/MemoryLLM paper (which is about a self-updating long-term-memory pool inside an LLM). Same name, totally different idea.

---

## 2. Architecture

### 2.1 Conventional transformer block (baseline notation)

- Input embeddings $X_0 \in \mathbb{R}^{M\times d}$ from embedding matrix $E\in\mathbb{R}^{|V|\times d}$.
- At layer $L$:
	$$\text{Attn}(X_L) = \text{softmax}\!\left(\frac{X_L W_Q^\top (X_L W_K^\top)^\top}{\sqrt{d_k}}\right) X_L W_V^\top$$
	$$\tilde X_L = X_L + \text{Attn}(X_L)$$
	$$\text{FFN}(\tilde X_L) = W_{\text{Down}}\!\left(\left(\tilde X_L W_{\text{Up}}^\top\right)\odot \text{SiLU}(\tilde X_L W_{\text{Gate}}^\top)\right)$$
	$$X_{L+1} = \tilde X_L + \text{FFN}(\tilde X_L)$$
- **SwiGLU** FFN has three matrices: up-projection $W_{Up}$, gate $W_{Gate}$, down-projection $W_{Down}$ ($K$ = intermediate dim).

### 2.2 MemoryLLM block

- **Key change**: every FFN in every layer eats the *same* input - the layer-normed token embedding - regardless of layer.
	- $\hat X_0 = \text{LayerNorm}_L(X_0)$ (independent LN per layer; needed for convergence).
	- $\text{FFN}(\hat X_0) = W_{\text{Down}}\!\left(\left(\hat X_0 W_{\text{Up}}^\top\right)\odot \text{SiLU}(\hat X_0 W_{\text{Gate}}^\top)\right)$
	- **Residual update**:
		$$\boxed{X_{L+1} = X_L + \text{Attn}(X_L) + \text{FFN}(X_0)}$$
- **Self-attention is unchanged** - still trained conventionally on the contextual residual stream.
- **FFN and Attn run in parallel** off different inputs (a la PaLM/parallel blocks, but the FFN side ignores residual entirely).
- **Why this is interesting**:
	- FFN input depends only on the **token id** -> a vocab of $|V|$ tokens means each FFN sees only $|V|$ possible inputs ever.
	- Removes context dependence -> output can be **pre-tabulated**.
	- FFNs become a **static per-token additive contribution** to the residual stream.

### 2.3 TKV (Token-Key-Value) interpretation

- Reinterpret SwiGLU FFN as a *retrieval memory* with $K$ key-value cells.
	- $W_{Up}\in\mathbb{R}^{K\times d}$: columns are **keys** $k_i$.
	- $W_{Down}\in\mathbb{R}^{K\times d}$: rows are **values** $v_i$.
	- $W_{Gate}$: per-key **reweighting** $g_q$ that amplifies/suppresses keys.
- **Two-step retrieval** for query $q = x_i$ (a context-free embedding of token $t$):
	- **Step I** (importance scores per key):
		$$\tilde c_{k_i} = \left(q^{1\times d}\cdot W_{Up_{[:,k_i]}}^\top\right) \times g_{k_i}$$
	- **Step II** (linear combination of values):
		$$\text{FFN}(X_0^q) = o_q = \sum_{i=1}^K \tilde c_{k_i}\cdot v_{k_i}$$
- Whereas in a standard FFN the "query" lives in some opaque mid-layer latent space, here the **query space = vocab embeddings**, finite and labeled.

### 2.4 ToLs - FFNs as precomputed token-wise lookups

- Once trained, for each vocab token $t_i$ with embedding $x_{t_i}$, dump the FFN output across all $N$ layers:
	$$\text{ToL}_{x_{t_i}}^{1\times (N\cdot d)} = \text{Concat}_{k=0}^{N-1}\{\text{FFN}_{L_k}(x_{t_i}),\ \text{dim}=1\}$$
- **Inference**: for tokens $\{t_1,\dots,t_M\}$ at layer $L$:
	$$X_{L+1} = X_L + \text{Attn}(X_L) + \text{ToL}_{\{t_1,\dots,t_M\},L,:}$$
- **Active params** in VRAM = embedding + attention only (~1/3 of dense params).
- **Storage**: $|V|\times N\times d\times \text{bits\_per\_param}$. For 1B MemoryLLM (24 layers, $d=2048$, $|V|=128k$, fp16) = **~12.6 GB on disk**.

#### On-demand plug-n-play
- **Zipf's law**: token frequencies are heavy-tailed -> cache ToLs for frequent tokens in VRAM, stream rare ones from disk.
- **Non-uniform layer importance**: Figure 4 shows dropping FFN layer $L$ in MemoryLLM hurts only the first few layers; middle/late FFN ToLs can be cropped/offloaded with marginal perplexity hit. (Conventional base shows a **U-shape** because dropping FFN there disrupts the residual stream too.)

---

## 3. Flex-MemoryLLM (bridging perf gap)

- Pure MemoryLLM under-performs same-total-param dense baselines because **all FFN params are spent on context-free memory** -> low computational expressivity over the residual stream.
- **Fix**: split each FFN's params into two pieces with the **same total budget**:
	- **FFN-C (compute)**: small **context-aware** dense module that operates on the residual stream like a normal FFN. Adds `LN(x_0)+Attn(x_0)` style compute.
	- **FFN-M (memory)**: context-free, fed by $X_0$, identical to MemoryLLM's FFN; can still be precomputed -> ToLs.
- **Sizing**: LLaMa uses FFN $\approx 8h^2$ params (intermediate ratio 2.67, with 3 matrices). Define Flex-MemoryLLM-$\beta h^2$: move $\beta h^2$ params from FFN-M into FFN-C.
- Tried $\beta\in\{1,2,3\}$; **$\beta=3$ closely matches dense baseline** while still letting **$5h^2$ params worth of FFN-M get offloaded** from VRAM during inference.

| Architecture | FFN-C | FFN-M |
|---|---|---|
| Base | 33.554M | 0 |
| MemoryLLM | 0 | 33.554M |
| Flex-MemoryLLM-$h^2$ | 4.194M | 29.360M |
| Flex-MemoryLLM-$2h^2$ | 8.388M | 25.165M |
| Flex-MemoryLLM-$3h^2$ | 12.582M | 20.971M |

---

## 4. Training setup

- **Train from scratch** at **250M / 750M / 1B** total params (24 layers each, hidden 960/1600/2048).
- **Tokens**: 25-150B from **C4**.
- **Tokenizer/vocab**: LLaMa-3.1-8B, $|V|=128{,}256$.
- **Seq len**: 2048.
- **Optim**: Adam, $\beta_1{=}0.9$, $\beta_2{=}0.95$, weight decay 0.1, cosine LR (max 1e-4, min 1e-5), 5000-step warmup, cross-entropy + Z-loss 1e-6.
- All variants (Base/MemoryLLM/Flex) trained with the **same recipe** for fairness.

### Active vs total params (1B scale, illustrative)

| | Base-1208M | Flex-1h2 | Flex-2h2 | Flex-3h2 | MemoryLLM |
|---|---|---|---|---|---|
| Active params (VRAM) | 1208M | 704M | 604M | 503M | 402M |
| Inference mem (GB) | 9.541 | 7.025 | 7.409 | 7.825 | 6.041 |
| Decode (ms/tok) | 21.50 | 18.75 | 20.28 | 21.47 | **14.42** |

> MemoryLLM is the fastest because the FFN side reduces to a memory lookup; Flex variants pay for the FFN-C compute.

---

## 5. Empirical findings

### 5.1 Spatial structure of memory (interpretability)

- K-means on $c_k$ vectors (per-key importance scores) across all 128k vocab tokens, for $L_0$ of MemoryLLM-1B.
	- **Well-formed clusters** corresponding to **punctuation, personal names, geographical locations, linguistic properties**, etc.
	- => **Semantically similar tokens activate similar FFN keys** -> they live in nearby "rooms" of the memory.
- **Clustering coefficient (CC)** stays high across all 24 layers (~0.9+), with a slight dip in the middle.
- **Outlier count** of $c_k$ per token: **terminal (late) layers have higher outlier counts** -> a few dominant keys carry most of the contribution -> tighter token-level info convergence near the output.
- **Implication**: opens up clean **knowledge editing / injection / toxicity suppression** by surgically altering specific keys in interpretable fashion.

### 5.2 FFN contribution probe: $\alpha\cdot\text{FFN}(X_0)$

- Scale the FFN contribution by $\alpha\in[0,1]$ at inference and measure task delta.
- Categories:
	- **Recall/retrieval-dominated**: Wikitext-2, LAMBDA, SiQA, ARC-Easy.
	- **Logical/reasoning-dominated**: HellaSwag, Winogrande, BoolQ, PIQA.
- Findings (MemoryLLM-1B):
	- **Decreasing $\alpha$ hurts recall tasks far more** than reasoning tasks.
		- e.g. LAMBDA: 0.36 -> 0.10 at $\alpha=0.5$ (-69.9%); BoolQ: 0.62 -> 0.61 (-1.18%).
		- Wikitext PPL: 24.3 -> 53.7 at $\alpha=0.5$.
	- => FFNs **act as token-level parametric knowledge reservoirs** dominant for retrieval/recall.
- Compared to base LLM: dropping $\alpha$ also degrades base, but **MemoryLLM degrades more gracefully on reasoning** because residual flow is not disrupted (FFN only adds; doesn't transform the stream).

### 5.3 Perplexity vs baselines

| (50B tokens) | base-750 | MemoryLLM (1208M total / 245M active) | MemoryLLM (737M total) | base-250 |
|---|---|---|---|---|
| C4 PPL | 19.73 | 20.93 | 22.08 | 23.19 |
| Wikitext-2 | 25.49 | 27.26 | 29.98 | 32.22 |

- **By total params**: MemoryLLM underperforms same-total dense (expected).
- **By active params**: MemoryLLM **outperforms** dense of equivalent active size - the memory you can offload basically pays back compute-for-storage.
- **Flex-MemoryLLM-$3h^2$@1B**: 704M active params **matches dense 737M base**, and even **outperforms it** in some configs - so it can be a recipe for "train a bigger model, run a smaller one".

### 5.4 vs pruning baselines (Magnitude, SparseGPT, Wanda)

- At 1B and 750M scales, compared the C4 PPL at matched active-param budgets.
- **MemoryLLM and Flex-MemoryLLM dominate** pruning methods across active-param sweep, especially at heavy compression.
- Framing: **architecture-level alternative to pruning**, no retraining/calibration needed.

### 5.5 ToL compression studies (Appendix D)

- **Quantization** of ToLs from fp16 -> int4: 12.6 GB -> 3.15 GB with **<1% accuracy hit** on Wikitext/LAMBDA/SiQA/ARC-Easy/HellaSwag/Winogrande/BoolQ/PIQA.
- **Low-rank SVD** on ToLs per layer (uniform rank reduction): 50% rank cut -> 6.4 GB (49% storage saved) at C4 PPL 18.92 -> 19.59 (+3.5%). Many layers exhibit heavy-tail singular value spectra; terminal layers compress especially well.
- **Layer-wise drop**: dropping middle-layer ToLs is largely free; dropping first-few-layer ToLs is the only thing that really hurts. -> Middle ToLs are highly redundant.

---

## 6. Why this is interesting (connections + spicy takes)

- **Same direction as MoE-style decoupling** (MoLE, PEER, product-key memory): "compute" vs "memory" separation in a transformer is becoming an architectural primitive.
	- MemoryLLM is the *extreme* form: not just mixture-of-experts routed by token, but **all FFN params indexed by token**, no router.
- **Sister to RAG**: instead of retrieving from a text corpus, retrieve from a parametric per-token table that *is* the FFN. Both reduce active-VRAM compute by leveraging external storage.
- **Mechanistic interpretability story**: lets you study Geva et al.'s key-value memory hypothesis cleanly because the query space is no longer some drifting residual latent - it's the literal token embedding.
- **Pruning vs architecture**: paper makes the case that *designing* a context-free FFN > *pruning* a dense one for active-param reduction.
- **Limitations / open questions**:
	- Reasoning/logical tasks still suffer relative to dense - residual flow gets a static per-token update, not a context-aware FFN.
	- Storage scales as $O(|V|\cdot N\cdot d)$ - blows up if you want huge vocab or many layers; mitigated by quant+SVD+layer-dropping.
	- Trained from scratch only - no story for **converting an existing pretrained dense LLM into MemoryLLM** (would be the killer follow-up).
	- ToL prefetch/cache latency depends on workload's token entropy; Zipf helps in practice but worst-case tokens add storage I/O.
	- Only tested up to 1B / 150B tokens - scaling curves at 7B+ unknown.

---

## 7. Implementation knobs that matter

- **Independent LayerNorm per FFN** ($\text{LN}_L$) is critical for convergence even though all FFNs eat the same $X_0$ - otherwise they collapse onto the same function.
- Architecture details (Table 5):
	- 250M: 24 layers, $d{=}960$, intermediate 2560, 16 heads.
	- 750M: 24 layers, $d{=}1600$, intermediate 4272, 16 heads.
	- 1B: 24 layers, $d{=}2048$, intermediate 5464, 32 heads.

```text
# Per-layer forward (pseudo) -- MemoryLLM
x_attn = Attn_L(X_L)                 # context-aware
x_mem  = FFN_L(LN_L(X_0))            # context-free, can be replaced by ToL[token_ids, L]
X_{L+1} = X_L + x_attn + x_mem
```

```text
# Per-layer forward -- Flex-MemoryLLM
x_attn = Attn_L(X_L)
x_compute = FFN_C_L(LN(X_L + x_attn))   # small dense, context-aware
x_mem     = FFN_M_L(LN_L(X_0))          # large context-free, offloadable
X_{L+1} = X_L + x_attn + x_compute + x_mem
```

---

**No public code release.** Do not confuse with the 2024 wangyu-ustc/MemoryLLM repo, which is an unrelated paper.
