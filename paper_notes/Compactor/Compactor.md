# Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores

**Chari & Van Durme, JHU, 2025** — [arXiv 2507.08143](https://arxiv.org/abs/2507.08143) · code: `compactor-vllm`

> TL;DR: training-free, **query-agnostic** KV-cache **token eviction**. Score each cached key by `leverage score (geometry) + non-causal attention (task relevance)`, keep top-`r·N`. Plus *context-calibrated compression*: fit a curve to predict the max compression a given context tolerates, set `r` per-context automatically. ~Matches full cache on LongBench while keeping ~1/3 of KV (≈68% memory cut); ~20% fewer tokens than baselines at equal quality. Ships a custom paged-KV vLLM-style engine + Triton kernels for the resulting sparse, ragged, per-head KV.

---

## 1. Motivation & Problem Setup

- **KV cache is the deployment bottleneck.** Autoregressive decode caches all past `(K,V)` → memory grows **linearly** in context length `N`.
  - **Concrete:** `Qwen2.5-32B` at 100K prompt → **26 GB** KV per request. Caps throughput / batch size on inference servers.
- **Two orthogonal directions** (taxonomy from Li et al. 2025, 4 stages: KV *generation / retrieval / loading / compression*):
  - **(a) Efficiency of management** — PagedAttention (vLLM), prompt caching across requests.
  - **(b) Reduce the footprint** — quantization (orthogonal, per-element) vs **token dropping / eviction** (this work, along the *sequence* dimension). Compactor composes with quantization.
- **Query-aware vs query-agnostic** (the central axis):
  - **Query-aware** (H2O, SnapKV, PyramidKV, TOVA): use the *question's* attention to pick which context tokens to keep. Works great *if* the query is known at compression time.
    - **Two fatal problems for serving:**
      1. **Degrades badly when query is unknown** ("query-agnostic regime", Li 2025 / Chari 2025) — e.g. SnapKV needs only 10% cache when query-aware, collapses otherwise.
      2. **Incompatible with prefix caching** — each new question evicts *different* prefix tokens → compressed prefix can't be reused across requests. This is the killer for multi-query serving.
  - **Query-agnostic** (Compactor): compress the *context* once, reuse for any question. Required for shared-prefix / RAG serving.
- **Second gap: how much to compress?** Given a token/memory budget, `r` is fixed. With **no budget**, you want the *smallest `r`* that doesn't hurt quality — but different contexts tolerate wildly different compression (a string of UUIDs is ~incompressible; a few-shot prompt is highly compressible). **Context-calibrated compression** infers per-context `r`.

---

## 2. Method: Compactor

**Setup.** Sequence of `N` tokens → keys/values `\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}`. Retain fraction `r\in(0,1]`, i.e. keep top-`k`, `k=\lceil rN\rceil` tokens by importance score `\vec{s}\in\mathbb{R}^N`. Two complementary score families blended:

- **`\vec{o}` = outlier / leverage scores** → capture *geometry* of `\mathbf{K}` (which keys carry unique directions).
- **`\vec{a}` = non-causal attention scores** → capture *task-driven* relevance (which keys get attended to).
- Scoring applied **independently per head and per layer**.

### 2.1 Outlier scores via statistical leverage

- **Why:** keeping only most-attended keys fails when a key is rarely queried *but encodes unique info* — an **outlier in key space**. Evicting outliers = irreversible info loss. Formalize "outlierness" via **statistical leverage**.

> **Definition (Leverage score).** For `\mathbf{K}=\mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top` (SVD), leverage of row `i`:
> `\ell_i = \lVert U_i\rVert_2^2`, with `\sum_i \ell_i = \mathrm{rank}(\mathbf{K})`.

- **Intuition:** rows whose left-singular-vector has large norm align with directions of greatest variance in `\mathbf{K}`; dropping them discards significant spectral information.
- **Theory (Thm 1, Spectral Preservation):** sampling `k\ge Cd\log(d/\delta)\epsilon^{-2}` rows w.p. `\propto \ell_i/\mathrm{rank}(\mathbf{K})`, rescaled, gives `(1-\epsilon)\mathbf{K}^\top\mathbf{K}\preceq\hat{\mathbf{K}}_k^\top\hat{\mathbf{K}}_k\preceq(1+\epsilon)\mathbf{K}^\top\mathbf{K}`. Connects to spectral sparsification / column-subset selection literature (Drineas, Mahoney). In practice **deterministic top-`k` leverage** works as a good sparsifier.
- **Defn:** `\vec{o}=\mathrm{diag}(\mathbf{U}^\top\mathbf{U})=[\ell_1,\dots,\ell_n]`.
- **Speed trick (exact):** SVD of `N\times d` is slow. Use `\mathrm{SVD}(\mathbf{K}^\top\mathbf{K})=\mathbf{V}\boldsymbol\Sigma^2\mathbf{V}^\top \Rightarrow \mathbf{U}=\mathbf{K}\mathbf{V}\boldsymbol\Sigma^{-1}` → SVD of a `d\times d` matrix + 2 GEMMs. Same asymptotics but hardware-fast when `d\ll N` (always true for KV).

### 2.2 Approximate leverage scores (the practical contribution)

- **Subspace embedding / sketch** `\boldsymbol\Phi:\mathbb{R}^d\to\mathbb{R}^k`, `(1-\epsilon)^2\lVert x\rVert^2\le\lVert\boldsymbol\Phi x\rVert^2\le(1+\epsilon)^2\lVert x\rVert^2`.
- **Key choice — RIGHT sketch `\mathbf{K}\boldsymbol\Phi`** (vs prior work's left sketch `\boldsymbol\Phi\mathbf{K}`). Reduces *column* (feature) dimension. Three reasons:
  1. Speeding up `\mathbf{U}=\mathbf{K}\mathbf{V}\boldsymbol\Sigma^{-1}` requires a right sketch.
  2. Left sketch needs materializing `\boldsymbol\Phi\mathbf{K}` → huge memory when `N` large.
  3. Empirically no degradation for KV-cache use.

> **Algorithm 1 — Approximate Leverage Scores.** Input `\mathbf{K}\in\mathbb{R}^{N\times d}`, target dim `k`.
> 1. `\boldsymbol\Phi\in\mathbb{R}^{d\times k}`, `\Phi_{ij}\sim\mathcal N(0,1/k)`
> 2. `\hat{\mathbf{K}}\leftarrow\mathbf{K}\boldsymbol\Phi`
> 3. `\mathbf{G}\leftarrow\hat{\mathbf{K}}^\top\hat{\mathbf{K}}`
> 4. `\mathrm{SVD}(\mathbf{G})=\tilde{\mathbf{V}}\boldsymbol\Sigma^2\tilde{\mathbf{V}}^\top`  *(only a `k\times k` SVD!)*
> 5. `\tilde{\mathbf{U}}\leftarrow\hat{\mathbf{K}}\tilde{\mathbf{V}}\boldsymbol\Sigma^{-1}`
> 6. return `\vec{o}\leftarrow\mathrm{diag}(\tilde{\mathbf{U}}\tilde{\mathbf{U}}^\top)`

- **Thm 2 (quality bound):** for Gaussian right-sketch with `k\ge 12\epsilon^{-2}(\mathrm{rank}(\mathbf{K})\log(42\epsilon^{-1})+\log(2\delta^{-1}))`, w.p. `1-\delta`:
  `\kappa(\mathbf{K})^{-1}\frac{1-\epsilon}{1+\epsilon}\ell_i \le \tilde\ell_i \le \kappa(\mathbf{K})\frac{1+\epsilon}{1-\epsilon}\ell_i` (`\kappa` = condition number). Proof via sketched-SVD singular-value preservation (Gilbert 2012). They use `k=48` which *violates* the bound but works (bounds known to be loose on real data).
- **Robustness (appendices):** SVD ≈ QR ≈ Gram-eigendecomposition for the `\hat{\mathbf{K}}` step (eig slightly worse, fixable by clamping small `\lambda`). **SRHT** (Subsampled Randomized Hadamard Transform) sketch ≈ Gaussian, and can be applied *in-place* on KV → lower peak memory.

### 2.3 Attention-based scoring (`\vec{a}`)

- **Prior (H2O):** accumulated **causal** attention `\vec{s}=\mathbf{1}^\top\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top+\mathbf{M})`, `\mathbf{M}` = causal `-\infty` mask. **SnapKV:** only last `W` queries `\vec{s}=\mathbf{1}^\top\mathrm{softmax}(\mathbf{Q}'\mathbf{K}^\top+\mathbf{M}')`.
- **Problem in query-agnostic regime:** no reason the last `W` tokens are more informative than any other window — the actual query isn't there.
- **Compactor's fix — NON-CAUSAL attention.** Drop the causal mask, use the full `\mathbf{A}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)`. Entry `A_{ij}` = model's tendency to route info from `j`→`i` *regardless of generation order* → a query-agnostic estimate of "how likely is token `j` to be attended by any future token".
  - `\vec{a}=\mathbf{1}^\top\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)` (column-wise sum).
  - **Empirical structure (Figs 1, 7):** non-causal maps show **columnar "anchor" stripes** (separator tokens `\n`, `,`, `.`; conjunctions `and/but`; system prefixes attract disproportionate attention from many positions) + strong **diagonal/locality bands**. These are *suppressed in the causal view*.
- **Chunking (for cost):** full `N\times N` attention is prohibitive. Chunk context into blocks of size `C`, score within-chunk:
  `\vec{a}=\mathrm{concat}_{i=1}^{\lceil N/C\rceil}\big(\mathbf{1}^\top\mathrm{softmax}(\mathbf{Q}[i]\mathbf{K}[i]^\top)\big)`, then mean-pool (per Li 2024).

### 2.4 Score blending & selection

- **Blend (z-scored, hyperparam `\lambda`):**
  `\vec{s}=\dfrac{\vec a-\bar a}{\mathrm{std}(\vec a)} + \lambda\cdot\dfrac{\vec o-\bar o}{\mathrm{std}(\vec o)}`
- **Procedure:** (1) compute `\vec{s}`; (2) keep top-`\lceil rN\rceil` (either across all heads, or within a single head = head-adaptive). Scoring per-head, per-layer.

### 2.5 Context-calibrated compression (auto-`r`)

- **Goal:** predict the max compression a context `c` tolerates with minimal overhead, *without* the query.
- **Quality proxy = NLL.** Token-averaged negative log-likelihood `\mathrm{NLL}(t;c)=-\frac1L\sum_i\log \mathrm{LM}(t_i;t_{<i},c)`. Higher NLL = model finds text more surprising.
- **Target = NLL ratio** between original and compressed (`\tilde c_r`) context:
  `g(r,q,c)=\dfrac{\mathrm{NLL}(t;q,c)}{\mathrm{NLL}(t;q,\tilde c_r)}`.
- **Query-agnostic assumption (crucial):** `g(r,q,c)\approx f(r,c)` — the NLL gain is explainable by `(c,r)` *alone*, drop `q`. Empirically `g` decays smoothly & monotonically in `r`.
- **Two-param curve fit:** `b=\alpha\cdot\mathrm{NLL}(c)+\beta`, and
  `f_{\alpha,\beta}(r,c)=\dfrac{\exp(rb-b)-\exp(-b)}{1-\exp(-b)}` — satisfies `f(1,c)=1` (no compression) and `f(0,c)=0` (empty cache). Fit `(\alpha,\beta)` by least squares on `(r,c,y)` training triples (`y` = NLL ratio of the true answer); use **asymmetric MSE** to penalize *under-estimating* `r^\star` more.
- **Inference:** user gives quality budget `\tau\in(0,1]` (e.g. `\tau=0.95` ⇒ allow ≈5% NLL increase). Pick `r^\star(c,\tau)=\arg\max_{r} f_{\alpha,\beta}(r,c)` s.t. `f\ge\tau` (closed form). Retain top-`\lceil r^\star N\rceil` per head.
- **Applicable to any eviction policy** (not just Compactor) — used in experiments to compare methods at matched quality.

---

## 3. Experiments

- **Models:** `Llama-3.1-8B-Instruct`, `Qwen2.5-14B-Instruct-1M`. **HW:** H100 80GB.
- **Benchmarks:**
  - **RULER** (synthetic, "how much of the stated context can the model use?"): 13 tasks / 4 categories — NIAH retrieval, multi-hop tracing, aggregation, distractor stress tests. Use **RULER-4k**. Report performance **vs compression rate**.
  - **LongBench** (real-world, 21 datasets / English subset): single-doc QA, multi-doc QA, summarization, few-shot, synthetic, code.
- **Baselines (all run query-agnostic, head-adaptive):** TOVA, KeyDiff, SnapKV, DuoAttention, PyramidKV, random eviction. **H2O excluded** (too slow).
- **Hyperparams:** `\lambda=0.3`, sketch `k=48`, chunk `C=256`, `\tau=0.95` (Llama) / `0.90` (Qwen).

### 3.1 Speed (selection overhead)

- Compactor selection ≈ **as fast as SnapKV** for contexts > 16k. Short-context overhead is the *constant* SVD cost (fixed regardless of length).
- **Compactor(`\vec{o}`)** (leverage only) has near-zero added overhead; the `\vec{a}` step's GEMM is the main extra cost. → **drop-in replacement** for other query-agnostic evictors at long context.

### 3.2 RULER (degrades most gracefully)

| KV retention | Compactor (Llama) |
|---|---|
| 50% | **99%** of baseline |
| 25% | **87.1%** |
| 10% | **68.0%** |

- Similar curve on Qwen. **SnapKV/TOVA/PyramidKV/DuoAttention degrade poorly** in the query-agnostic regime (confirms the regime is hard for query-aware methods). All methods poor at *severe* compression (intrinsic task difficulty).
- On **NIAH-UUID** (incompressible), Compactor gets ~full performance at 50% retention where SnapKV/TOVA collapse (Fig 3).

### 3.3 LongBench (matches full cache at ~1/3)

- **50% retention:** matches / slightly exceeds full-cache (gains in few-shot & code — less contextual diversity).
- **25%:** retains ~**95%** of full-cache total.
- **10% / 5%:** advantage *grows*, **outperforms competitors by >15%**. Strong **across all task categories** (vs KeyDiff failing on code, TOVA on single-doc QA) — task-agnostic robustness, key for deployment.

### 3.4 Context-calibrated compression results

- **Per-method curve fit on RULER, applied to LongBench** at `\tau=0.95`. For **all methods**, calibrated compression stays within **.5% of uncompressed** quality → validates NLL as proxy and the calibration curve.
- **Compactor achieves this with fewer retained tokens** (Fig 6): Zero-shot **Compactor 32% KV** vs SnapKV 51%, KeyDiff 48%, TOVA 53%, Random 90%; all within 0.1 of full quality.
- **Finetuned setting** (LoRA r=128 on Q/K/V, 3 epochs, on the doc corpus, no queries): all methods get higher compression; Compactor **19% KV**. Notably **Compactor zero-shot (32%) beats SnapKV/TOVA even after finetuning**.
- **Abstract claim:** ~**68% average KV memory reduction** at full LongBench quality; **~20% fewer tokens** than baselines at equal quality.

### 3.5 Ablations (Table 2, RULER Llama)

| Variant | 75% | 50% | 25% | 10% |
|---|---|---|---|---|
| **Full** | 95.6 | 94.7 | 83.1 | 64.8 |
| − Exact `\ell_i` | 95.5 | 94.9 | 82.8 | 64.7 |
| − `k=64` | 95.3 | 94.2 | 83.2 | 64.8 |
| − `\lambda=0.4/0.2/0.15` | ~94–95 | ~94 | ~83 | ~62–63 |
| **− Only `\vec a`** (attn) | 93.5 | 79.0 | 61.3 | 35.7 |
| **− Only `\vec o`** (leverage) | 96.1 | 94.1 | 73.5 | 44.7 |

- **Both components needed.** `\vec a` helps a lot at aggressive compression; `\vec o` carries high-retention regimes. `\lambda` barely matters → robust. **Approximate ≈ exact** leverage (big speedup, no quality loss). `k=48` vs `64` negligible.

### Limitations

- `k=48` violates Thm 2's bound (works empirically, but no formal guarantee at used settings).
- Calibration assumes `g(r,q,c)\approx f(r,c)` — query-independence of NLL gain (necessary but approximate).
- Constant SVD overhead makes it relatively slower at short contexts.
- Severe compression (5–10%) still degrades on genuinely incompressible content (UUIDs) — calibration *recognizes* this (keeps tokens) rather than fixing it.

---

## 4. Implementation (`compactor-vllm`)

A from-scratch vLLM-style engine (Llama3 / Qwen3 / Qwen3-MoE models, paged KV, CUDA graphs, TP) + Triton kernels purpose-built for the **sparse, non-contiguous, per-head-ragged** KV that token-dropping produces. Key insight: after compaction each **KV head keeps a *different* number of tokens** (`bh_seq_lens` is `[batch, H_kv]`), so standard contiguous attention kernels don't apply.

### 4.1 Where compression hooks in

- `models/llama3.py :: LlamaAttention.forward` (mirrored in qwen3): after QKV proj, **before & after RoPE**:
  ```python
  scores = None
  if context.is_prefill and context.do_compression:
      scores = apply_prerope_compression(q, k, v, context)   # leverage on PRE-rope K
  q, k = self.rotary_emb(positions, q, k)
  if context.is_prefill and context.do_compression:
      scores = apply_postrope_compression(q, k, v, scores, context)  # non-causal attn on POST-rope Q,K
  o = self.attn(q, k, v, scores)
  ```
  - **Leverage uses pre-RoPE keys** (geometry of the learned key space, position-free); **attention uses post-RoPE** Q/K (real attention logits). Matches `BaseCompressionMethod`'s two-phase `pre_rope_scoring` / `post_rope_scoring` interface (`compression/common.py`).
- Dispatch: `compression/__init__.py` maps `CompressionMethod.{COMPACTOR,SNAPKV} → {CompactorCompression,SnapKVCompression}`.

### 4.2 Leverage scores — `compression/compactor.py :: approximate_leverage_scores`

Direct impl of Algorithm 1, batched over heads & **chunked along sequence** for memory:
```python
X = key_states.transpose(0,1) @ PHI          # [H, N, k]  (right sketch KΦ)
# per chunk (size 512 default): center, Gram, batched k×k SVD
chunk -= chunk.mean(-2, keepdim=True)
g = chunk.transpose(-1,-2) @ chunk           # [H, NC, k, k]  Gram
V,S,Vt = torch.linalg.svd(G, driver="gesvda")# many tiny k×k SVDs at once
SV = V * S.rsqrt()                           # V Σ^{-1}
U  = chunk @ SV
scores = (U*U).sum(-1)                        # leverage = ||U_i||²
```
- **`PHI`** built once per batch (`utils/arguments.py`): `torch.randn(head_dim, sketch_dim) / sqrt(sketch_dim)`, `sketch_dim = leverage_sketch_size = 48` (config). i.e. `\Phi_{ij}\sim\mathcal N(0,1/k)`.
- **Numerics:** adds `regularizer=5e-3` to Gram diagonal; on `gesvda` failure retries with `10×` reg, then falls back to a **batched QR** path (`_approximate_leverage_scores_qr_fallback`, ~50× slower — `\ell_i=\lVert Q_{i}\rVert^2`).
- **Chunking helper** `split_into_chunks` decomposes ragged varlen sequences into full `chunk_size` blocks + epilogue tails; full blocks processed as one batched `[H, NC, chunk_size, k]` tensor.
- **z-score normalization** done in a Triton kernel `_zscore_per_batch_epilogue_no_window` (per-sequence mean/var over all tokens×heads, two-pass).
- `chunk_size` is `compression.chunk_size = 512` (the *leverage* chunk; distinct from the attention chunk = `128`).

### 4.3 Non-causal attention scores — `_non_causal_attn_kernel` (Triton)

Flash-attention-style **2-pass online softmax**, but:
- **No causal mask** — every query attends every key in its chunk (the whole point).
- Computed **per chunk of `CHUNK_SIZE=128`**, grid = `(HKV, B, ceil(maxlen/chunk))`. GQA handled by flattening the `QUERY_GROUP_SIZE` query heads per KV head into the M-tile.
- Pass 1: per-row max + softmax denom (log2-domain `exp2`). Pass 2: recompute probs, **column-sum over queries** = `\vec a` per key, `atomic`-accumulated into `accum_scores[key, head]`.
- Short-chunk handling: invalid query rows contribute `1/CHUNK_SIZE` to preserve attention mass.
- **Blending happens here**: `post_rope_scoring` passes the leverage scores in as `accum_scores` with `accum_blending=0.5`, so output = `non_causal_a + 0.5·leverage` (after each is z-scored) — the `\vec s` blend, with `\lambda` folded into the blend constant.
- **Protected tokens:** `protected_first_tokens=16`, `protected_last_tokens=64` (`SequenceCompressionParams`) get score `+inf` → always retained (attention sinks + recency).

### 4.4 Top-k selection & sparse store — `compression/common.py`, `kv_cache/store_kv_cache.py`

- `scores_to_retain_indices`: pad each sequence to `max_k_len`, flatten `[B, max_k_len*H]`, per-batch `torch.topk(top_k)` → **global flattened indices `token*H + head`** (selection is per (token,head) pair, so different heads keep different tokens).
- `max_tokens_to_retain` / `batch_tokens_to_retain` computed in `arguments.py`:
  `retain = round(compression_ratio * (L - protected_first - protected_last) * num_kv_heads)`.
- **`prefill_store_topk_kv` (Triton):** scatters selected K,V into the paged cache. For each selected `(tok,head)` it does `pos = atomic_add(bh_lens[b,head], 1)` — i.e. **the per-head cache length grows independently**, KV is written **scrambled / out of order**, into physical pages via the page table (`phys = page_table[(b·H+head)·NLP + logical_page]`, `dst_row = phys·PAGE_SIZE + offset`).
- **`_prefill_store_topk_pad_kernel`:** pads each `(batch,head)` up to a page-size multiple by pulling extra ranked-but-unselected tokens, so pages are dense (avoids ragged partial pages hurting the decode kernel).

### 4.5 Paged KV cache — `kv_cache/page_table.py :: PagedKVCache`

- Global backing buffer `kv_cache[2, num_layers, n_pages·page_size, head_dim]` (K vs V on dim 0).
- **Page table** `[num_layers, max_num_seqs, H_kv, max_pages_per_head]` maps `(batch, kv_head, logical_page) → physical page id`.
  - Note the **per-head** page dimension — required because Compactor's heads have different lengths (unlike vanilla vLLM where all heads share length).
- Per-`(layer,batch,head)` lengths `bh_seq_lens[L, B, H_kv]` + `bh_num_pages`. Allocator = **min-heap of free physical pages per layer** + free batch indices (one `RESERVED_BATCH`).
- `reserve_tokens` / `reclaim_pages` / `free_batch` manage page lifecycle; `reclaim_pages` shrinks to `ceil(seq/page_size)` after compaction frees tokens. `page_size = kvcache_page_size = 128` (config).

### 4.6 Attention kernels over the sparse cache — `layers/attention.py`

- **Prefill:** if compressing → `extract_and_store_top_kv` (store compressed), then attention runs on the **uncompressed** `q,k,v` (full prefill still attends everything; only the *cache* is compressed for decode). Backend either FlashAttention varlen or `causal_sparse_varlen_with_cache` (Triton).
- **Decode:** `decode_store_kv` appends the new token's KV per head, then `head_sparse_decode_attention` (`attention/sparse_decode_kernel.py`):
  - reads ragged per-head paged KV via page table (`phys_row = page_id·PAGE_SIZE + tok%PAGE_SIZE`),
  - supports GQA (`HQ` multiple of `HKV`), optional `key_split` (split-K for long sequences),
  - `batch_mapping` maps launch-batch row → true page-table batch row.
- Async: stores run on a separate `STORE_STREAM` (`maybe_execute_in_stream`) to overlap with compute.

### 4.7 Key configs / hyperparameters (`config/`)

| Param | Default | Meaning |
|---|---|---|
| `leverage_sketch_size` | **48** | sketch dim `k` (Φ is `head_dim × k`) |
| `kvcache_page_size` | **128** | tokens per physical page |
| `attention_backend` | `COMPACTOR_TRITON` | vs `FLASH_ATTENTION`; Triton faster at long ctx |
| `CompactorCompression.chunk_size` | **128** | non-causal attn chunk `C` |
| `BatchCompressionParams.chunk_size` | **512** | leverage Gram/SVD chunk |
| `accum_blending` | **0.5** | leverage weight in `\vec s` blend (`\lambda` proxy) |
| `protected_first_tokens` | **16** | always-kept prefix (sinks) |
| `protected_last_tokens` | **64** | always-kept suffix (recency) |
| `compression_ratio` | 1.0 | per-sequence `r` (1.0 = no compression) |
| `regularizer` | 5e-3 | Gram diagonal load for SVD stability |

---

## 5. Connections / takeaways

- **Leverage scores** = the standard tool in randomized NLA for **column/row subset selection & spectral sparsification** (Drineas–Mahoney, CUR, sketching). Compactor reframes KV eviction as *selecting a spectrally-faithful row subset of `\mathbf K`* — a clean theoretical lens vs the heuristic "keep heavy hitters."
- **Non-causal attention** as a query-agnostic importance signal is the conceptual novelty: it answers "would *any* token want this key?" instead of "did the recent/query tokens want it?". The visible "anchor column" structure (separators/sinks) is the query-agnostic analogue of attention-sink findings (StreamingLLM).
- **Context-calibration** decouples *which* tokens to keep (eviction policy) from *how many* (budget) — an orthogonal, policy-agnostic add-on. NLL-as-proxy is the trick that makes it query-free.
- **Systems contribution is non-trivial:** per-head ragged KV breaks the contiguous-length assumption baked into vLLM/FlashAttention, hence the bespoke per-head page table + scatter-store + sparse-decode Triton kernels. This is the practical gap most KV-eviction papers skip.
