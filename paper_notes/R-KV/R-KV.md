# R-KV: Redundancy-aware KV Cache Compression for Reasoning Models

> Cai et al., **NeurIPS 2025**. arXiv:2505.24133. UW-Madison / Microsoft / CMU / Caltech / UCSD / Surrey / Berkeley.
> Project: https://zefan-cai.github.io/R-KV.page/ · Code: https://github.com/Zefan-Cai/R-KV

**TL;DR**: Decoding-time KV cache compression for reasoning LLMs (DeepSeek-R1 distills). Combines **attention importance** + **key-vector redundancy** scoring to keep tokens that are important *and* non-redundant. **~100% of full perf at 10% KV cache**, 90% memory saving, up to 6.6x throughput. Training-free, model-agnostic.

---

## 1. Motivation

- **Reasoning models emit huge CoT** → KV cache explodes during decoding.
  - DeepSeek-R1-Distill-Llama-8B: ~32K tokens for a math problem → 15.5GB weights + 4.1GB KV cache.
  - Reasoning models generate **8-14x more tokens** than ground truth (Fig 2, MATH-500 & AIME24).
- **The output is pervasively redundant**: superfluous self-reflection, repeated re-verification, verbose self-dialogue.
  - **1-/2-gram frequency 5-7x higher** than ground truth → quantitative evidence of repetition.
- **KV compression literature targets the prefill/prompt stage** ([SnapKV], [PyramidKV], [Ada-KV]) — long *input*, short output. R-KV instead targets the **decoding stage** where the *output* dominates length. (StreamingLLM, H2O are the decoding-focused exceptions.)
- **Key failure mode — attention-only importance filtering breaks on repetition**:
  - Repetitive sections **self-attend highly** (a repeated phrase mirrors earlier repeated text → high attention score).
  - So pure attention methods (SnapKV) **over-retain duplicated self-reflections** and may evict crucial-but-scattered reasoning bits with "low" attention.
  - Fig 3: SnapKV-selected tokens are dominated by repeats of "3 students are leaving early." and "But in the initial".
- **Insight**: must explicitly model **redundancy** and retain "important AND non-repetitive context."

---

## 2. Method: R-KV

Three components: **(1) importance scoring** (attention), **(2) redundancy estimation** (key cosine similarity), **(3) joint selection** (weighted combine). Runs **on-the-fly during decoding**.

### 2.1 Decoding-time compression schedule

- Two memory regions per head:
  - **Budget** $B_{\text{budget}}$: retained KV tokens.
  - **Buffer** $B_{\text{buffer}}$: newly generated tokens since last compression. Total $B_{\text{total}} = B_{\text{budget}} + B_{\text{buffer}}$.
- After each fixed-length **text segment** is generated into the buffer → run compression.
  - Last $\alpha$ tokens = **observation tokens** (always retained; their queries are the scoring queries, à la SnapKV).
  - Candidates $n = B_{\text{budget}} + B_{\text{buffer}} - \alpha$ (existing budget concat first $B_{\text{buffer}} - \alpha$ buffer tokens).
  - Select **top $k = B_{\text{budget}} - \alpha$** candidates by joint score, plus the $\alpha$ observation tokens → fills budget back up.
- **Constant memory footprint** vs FullKV's linear growth → the source of the throughput win (bigger batches).

### 2.2 Importance scoring (attention)

- Attention from the $\alpha$ observation-token queries onto the $n$ key candidates.
- **MHA**: $\boldsymbol{A}^h = \mathrm{softmax}(\boldsymbol{Q}^h (\boldsymbol{K}^h)^\top / \sqrt{d})$.
- **GQA** (Llama-3, Qwen): compute per query head in the group $\boldsymbol{A}_{\text{group}}^{h,g}$, then **max-pool across the group dimension**, renormalize:
  $$\boldsymbol{A}_{\text{group}}^h = \mathrm{maxpool}([\boldsymbol{A}_{\text{group}}^{h,0},\dots,\boldsymbol{A}_{\text{group}}^{h,G-1}]),\quad \boldsymbol{A}^h = \mathrm{softmax}(\boldsymbol{A}_{\text{group}}^h)$$
  - **Max-pool (not mean) chosen deliberately** (App A.2): preserves the most critical tokens for *each* query head; empirically better than SnapKV's mean-pool aggregation.
- **Stabilization**: per-token scores have spiky outliers → **max-pool over a sliding window $2W$** across recent keys before averaging:
  $$\tilde{A}^h_{j,i} = \max(A^h_{j,i-W},\dots,A^h_{j,i+W-1}),\qquad I^h_i = \tfrac{1}{\alpha}\sum_{j=0}^{\alpha-1}\tilde{A}^h_{j,i}$$

### 2.3 Redundancy estimation (key cosine similarity)

- **Semantic similarity ≈ cosine similarity between key vectors** (high similarity → redundant).
- L2-normalize keys, build similarity matrix, zero the diagonal (no self-redundancy):
  $$\overline{\boldsymbol{K}}_i^h = \frac{\boldsymbol{K}_i^h}{\|\boldsymbol{K}_i^h\|_2 + \epsilon},\quad \boldsymbol{S}^h = \overline{\boldsymbol{K}}^h(\overline{\boldsymbol{K}}^h)^\top,\quad S^h_{i,i}\leftarrow 0$$
  ($\epsilon \approx 10^{-8}$.)
- **Enforce retention of recent tokens**: redundant tokens can still matter; later tokens support decoding better than earlier ones.
  - For each token $i$, find highly-similar set $\mathcal{T}_i^h = \{j \mid S^h_{j,i} > T\}$ (threshold $T$).
  - Take its $\beta$ **most-recent** similar tokens $\mathcal{T}^h_{i,\beta}$ and **zero their similarity links** ($S^h_{j,i}\leftarrow 0,\ \forall j\in\mathcal{T}^h_{i,\beta}$) → nullifies the direct similarity to recent duplicates so the *recent* copy survives, older copies counted as redundant.
- **Redundancy score** = average similarity, softmax-normalized:
  $$\bar{S}_i^h = \tfrac{1}{n}\sum_{j} S^h_{j,i},\qquad R_i^h = \mathrm{softmax}([\bar S_0^h,\dots,\bar S_{n-1}^h])_i$$
  - High $R_i^h$ = token's content is shared with many others = redundant.

### 2.4 Joint selection

$$\boxed{Z_i^h = \lambda I_i^h - (1-\lambda) R_i^h}$$

- Maximize importance, **subtract** redundancy. Retain top-$B_{\text{budget}}$ per head.
- **$\lambda$ trade-off** (sensitivity in §5.1):
  - $\lambda = 0$ → pure redundancy (ignores attention; **initial-token sink not preserved** → can break LLM, see StreamingLLM attention-sink argument). $\lambda = 1$ → pure attention (= SnapKV-like, fails on repetition).
  - Both extremes worst. **Optimal $0.01 \le \lambda \le 0.1$**; paper uses $\lambda = 0.1$. (Must be $\ge 0.01$ to keep attention-sink first tokens.)
  - Note: $I^h$ is **sparse/spiky**, $R^h$ is **dense** → even small $\lambda$ keeps attention influential.

---

## 3. Experiments

- **Models**: DeepSeek-R1-Distill-Llama-8B (R1-Llama-8B), DeepSeek-R1-Distill-Qwen-14B (R1-Qwen-14B). Both GQA.
- **Datasets**: MATH-500, AIME 2024 (math reasoning).
- **Eval**: pass@1 over **64 sampled responses/question**, temp 0.6, top-p 0.95. Max gen 16,384 (MATH) / 32,768 (AIME). Greedy was too noisy. Hardware: A100 80G.
- **Hyperparams (paper)**: $B_{\text{buffer}}=128$, $\alpha=8$, $\lambda=0.1$. (Note: code defaults differ — `mix_lambda=0.07`, `retain_ratio=0.2`; see Implementation.)
- **Baselines**: SnapKV (adapted to decoding w/ same compression interval), FullKV (gold standard). Head-level (Ada-KV) / layer-level (PyramidKV) budget allocation deemed orthogonal, excluded.

### Results

| Metric | Result |
|---|---|
| **Lossless @ ratio** (R1-Llama-8B) | **34%** KV on MATH-500, **10%** on AIME24 |
| **R1-Llama-8B @ 16%** AIME24 | **105%** of FullKV (beats full cache) |
| **Lossless @ ratio** (R1-Qwen-14B) | 54% MATH-500, 25% AIME24; **105%** FullKV at 33% |
| **vs SnapKV** | up to **+40% acc** |
| **Memory** | **90% saving** (10% ratio); footprint constant, doesn't grow with seq len |
| **Throughput** | up to **9x larger batch, 6.6x throughput** @ 10% ratio, 16K gen; up to **13.4x batch / 9.19x throughput** @ fixed budget 1024, 16K |
| **Batch=1 speedup** | only slight (reduced attention compute ≈ scoring overhead); real win = bigger batches |

- **Ratio budget more meaningful than fixed budget**: longer outputs (e.g., 2,979→15,535 tokens) need a *smaller* ratio for lossless → lossless ratio is dominated by generation length.
- **Complexity** (App C.1): importance $O(\alpha B_{\text{budget}})$, redundancy $O(B_{\text{budget}}^2)$. Overhead outweighed by reduced attention over a smaller cache; advantage grows with sequence length.

### Ablations / Analysis

- **§5.1 $\lambda$**: U-shaped, both extremes bad, sweet spot $0.01$–$0.1$ (see 2.4).
- **§5.2 / Fig 7**: R-KV selects a **more diverse, broadly distributed** token set; SnapKV clusters near query and re-picks redundant "3 students leaving early." / "But in the initial" segments. R-KV completes the example correctly where SnapKV fails.
- **App A.2**: max-pool > mean-pool for GQA group aggregation.

### Related-work positioning

- **Eviction** (SnapKV, PyramidKV, Ada-KV, HeadKV — prefill; StreamingLLM, H2O — decoding): all attention-only → struggle with reasoning redundancy and risk evicting critical CoT steps.
- Other axes: **quantization** (KVQuant, KIVI), **merging** (CaM, DMC), **low-rank** (ShadowKV, Eigen Attention, LoRC).
- **Efficient reasoning** (RL length penalty / variable-length SFT / test-time prompting): all need training or hurt perf. R-KV is **training-free**, complementary, usable in **RL rollout** and serving.

### Limitations (App D)

- Not compatible with **paged attention** (vLLM-style) out of the box — non-trivial to integrate.
- Serving frameworks without native KV-compression interfaces → cost of reallocating/deallocating cache can offset speedups.

---

## 4. Implementation

Repo has 3 backends: **`HuggingFace/`** (reference, full integration via monkeypatch — analyzed here), **`vLLM/`**, **`SGLang/`**. Core class is **`R1KV`** in `rkv/compression/r1_kv.py`.

> ⚠️ **Naming/sign mismatch vs paper**: code uses `mix_lambda` where `final = attn*mix_lambda - sim*(1-mix_lambda)`, i.e. `mix_lambda` ≡ paper's $\lambda$. **Default `mix_lambda=0.07`** in code (paper text says $0.1$). Redundancy here is the raw softmaxed similarity (no separate $R$ symbol).

### 4.1 `R1KV.update_kv(key_states, query_states, value_states)` — `rkv/compression/r1_kv.py`

The whole scoring + eviction per layer. Shapes: `key_states` = `(bsz, num_kv_heads, kv_len, head_dim)`; `query_states` = the **cached window of observation queries** `(bsz, num_q_heads, window_size, head_dim)`.

1. **Early exit**: if `kv_cache_len < budget` → return uncompressed (nothing to evict).
2. **Importance** via `compute_attention_scores(query, key, pooling="max")` (in `utils.py`):
   - GQA: reshape queries to `(bsz, kv_heads, group, q_len, head_dim)`, matmul with keys, **`.max(dim=group)`** (max-pool over query group). MHA path = plain `QK^T/sqrt(d)`.
   - Then in `update_kv`: softmax over the `window_size` observation rows (excluding the recent window cols `[:, :, -window:, :-window]`), **mean over the window queries** → per-key importance.
   - **`F.max_pool1d(..., kernel_size=kernel_size=7, padding=3, stride=1)`** = the $2W$ sliding-window stabilization (`attn_cache`).
3. **Redundancy** via `cal_similarity(key_states, threshold=0.5, retain_ratio, retain_direction="last")` (in `utils.py`):
   - L2-normalize keys (`+1e-8`), `sim = k_norm @ k_normᵀ`, **`fill_diagonal_(0.0)`** per head.
   - `similarity_mask = sim > threshold` (0.5).
   - **Recent-token retention**: for each row, find the **last** (largest-index) similar token via `torch.where(mask, arange, 0).max(dim=-1)` and **zero that entry** in `sim` → suppresses link to the most-recent duplicate (so it isn't penalized; `retain_direction` chooses last/first/percent variants — `retain_ratio` only used by the `*_percent` modes).
   - Return **`sim.mean(dim=heads).softmax(dim=-1)`** = redundancy score (note: averages over heads here, then sliced to drop recent window).
4. **Combine** (line 75):
   ```python
   final_score = attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)
   ```
   ↔ paper's $Z = \lambda I - (1-\lambda) R$.
5. **Eviction** (line 82): `indices = final_score.topk(budget - window_size, dim=-1).indices`.
   - **Always keep last `window_size` tokens** (recent window): scoring is done on `[:-window_size]`; the recent window is concatenated back unconditionally.
   - `gather` selected past K/V, `cat` with the recent window K/V → compressed cache of size `budget`.
6. `record_kept_token_indices` branch = analysis-only bookkeeping (maps compressed indices back to original positions across steps for the Fig 7 visualizations in `utils.py`).

### 4.2 Integration — monkeypatch (`rkv/monkeypatch.py`, `rkv/modeling.py`)

- `replace_llama / replace_qwen2 / replace_qwen3(compression_config)` overwrite HF classes:
  - `*Attention.__init__` → `*Attention_init`: builds `self.kv_cluster = KV_COMPRESSION_MAP[method](**method_config)` (`KV_COMPRESSION_MAP` maps `"rkv"→R1KV`, plus snapkv/streamingllm/h2o/analysiskv), and stashes config.
  - `*Attention.forward` → `*Attention_forward`.
  - `*ForCausalLM.forward` → `CausalLM_forward`.
- **Query cache** (the observation queries): `past_key_value.query_cache[layer_idx]` holds only the **last `window_size` queries**, rolled forward each step. These (`cached_queries`) — *not* the current single decode query — are passed as the scoring queries to `update_kv`. This is what gives R-KV the SnapKV-style observation window during pure decoding.
- **Compression gating** in attention forward keyed on `self.config.compression`:
  - `None` → compress every step (compute compressed K/V, write back if `update_kv=True`).
  - `True` → first `past_key_value.update(...)` (append), then compress and overwrite `key_cache`/`value_cache` in place.
  - else → no compression, normal append.
- Attention itself still runs through HF's `ALL_ATTENTION_FUNCTIONS` (`flash_attention_2` by default) over the (possibly compressed) cache.

### 4.3 Compression schedule — `CausalLM_forward` (`modeling.py`)

- Tracks running output `self.length`; predicts next token (`logits[:, -1].argmax`).
- **When to compress** (`divide_method`):
  - `"step_length"` (default in `run_math.py`): `is_newline = (self.length % divide_length == 0)`, **`divide_length=128`** ⇒ compress every 128 decoded tokens (= $B_{\text{buffer}}$).
  - `"newline"`: trigger on newline token ids (segment = a reasoning step).
- **`compression_content`**:
  - `"all"` (default): compress whole output.
  - `"think"`: only compress inside the `<think>` block — once an after-think token appears, `is_newline` forced `False` (stop compressing the final answer).
- Sets `layer.self_attn.config.compression = is_newline` for **all layers at once**, so the next attention forward pass performs the eviction synchronously across the model.

### 4.4 Key hyperparameters (`run_math.py` defaults)

| Arg | Default | Meaning |
|---|---|---|
| `kv_budget` | (required) | $B_{\text{budget}}$ retained tokens/head |
| `window_size` | **8** | observation/recent window $\alpha$ (queries cached + always-kept recent keys) |
| `kernel_size` | **7** | max-pool stabilization window (code default in `R1KV`) |
| `mix_lambda` | **0.07** | $\lambda$ importance↔redundancy weight (paper text: 0.1) |
| `retain_ratio` | **0.2** | only used by `*_percent` retain modes |
| `retain_direction` | `"last"` | keep most-recent duplicate |
| `divide_method` | `"step_length"` | compress trigger |
| `divide_length` | **128** | $B_{\text{buffer}}$, compress interval |
| `compression_content` | `"all"` | compress all output vs only `<think>` |
| `update_kv` | `True` | write compressed cache back |
| `first_tokens` | 4 | (attention-sink-ish; threaded into config) |
| `attn_implementation` | `flash_attention_2` | |
| dtype | `bfloat16` | |

### 4.5 Notes / discrepancies worth flagging

- Redundancy in code is averaged **across heads** then softmaxed (`sim.mean(dim=1).softmax`), so the final redundancy term is effectively head-shared, whereas importance (`attn_cache`) is per-kv-head — selection top-k is per-kv-head.
- The "enforce recent retention" in code only zeroes the **single last** similar token per row (`retain_direction="last"`), a simpler version of the paper's "$\beta$ most-recent" set.
- `threshold=0.5` for the similarity mask is hard-coded in `cal_similarity`'s signature (paper's $T$).
