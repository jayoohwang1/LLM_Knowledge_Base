# Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution

> Devoto, Jeblick, Jégou (NVIDIA + Sapienza), 2025. arXiv:2510.00636. Ships with **KVPress** library (NVIDIA/kvpress, >20 compression methods).

**TL;DR**: Training-free KV cache pruning. Score each KV pair by the **expected attention that *future* queries will pay to it**, computed in **closed form** by exploiting the fact that LLM hidden states (hence queries) are approximately **Gaussian**. Works in both prefill (one-shot) and decode (streaming) phases. Prunes up to 60% cache with negligible quality loss; beats SnapKV / TOVA / KeyDiff / KNorm / StreamingLLM.

---

## 1. Motivation

- **KV cache is the long-context bottleneck.** Grows linearly with sequence length: a 70B model needs ~320 GB for a 1M-token cache. Hardware saturates well before advertised context limits.
- **Why not just use attention scores to rank KV pairs?** Two practical blockers:
	1. **Future scores unavailable.** At compression time (after prefill or during decode), you only have past queries. But a token's *true* importance depends on how **all future queries** (tokens $t+1, t+2, \dots$) will attend to it. Those queries don't exist yet.
	2. **Flash Attention doesn't materialize the attention matrix.** Modern kernels (FlashAttention-1/2) compute $\mathrm{softmax}(QK^\top/\sqrt d)V$ on-the-fly without ever forming the $L\times L$ score matrix → even *past* scores are inaccessible. Methods like H2O/SnapKV that read attention weights are incompatible with real deployment.
- **Alternatives are heuristics**: discard oldest (StreamingLLM), lowest key-norm (KNorm), key-embedding distance (KeyDiff). These ignore the global, output-level effect of a token.
- **The right objective** (their framing): a KV pair's significance = its **effect on the residual stream / model output**, not local attention metrics.

---

## 2. KV Cache & The Importance Objective

**Setup** (decoder-only, single head; extends to MHA/GQA). For hidden state $h_i \in \mathbb{R}^h$:
$$q_i = R_i W_Q h_i, \quad k_i = R_i W_K h_i, \quad v_i = W_V h_i$$
- $R_i \in \mathbb{R}^{d\times d}$ = **RoPE** rotation at position $i$, $d$ = head dim, $W_Q,W_K \in \mathbb{R}^{h\times d}$.

**Attention at step $t$**: $a_{ti} = z_{ti}/\sum_j z_{tj}$ where $z_{ti} = \exp(q_t^\top k_i / \sqrt d)$.

**Residual decomposition** (the key idea — à la Elhage transformer-circuits): the attention output is added to the residual stream,
$$h_t^{\text{out}} = h_t + \sum_{i=1}^{t} a_{ti} W_o v_i = h_t + \sum_i \Delta h_{ti}$$
So each KV pair contributes a residual update $\Delta h_{ti} = a_{ti} W_o v_i$. Its **importance = norm of its contribution**:
$$\|\Delta h_{ti}\| = a_{ti}\,\|W_o v_i\| \quad\quad (\star)$$
- Captures **both** how much it's attended ($a_{ti}$) **and** the magnitude of what it writes ($\|W_o v_i\|$).
- This is the *optimal* per-pair impact measure — prune lowest $\|\Delta h_{ti}\|$ → minimal output perturbation.
- **Problem**: $a_{ti}$ needs future queries (and Flash Attention hides it). → solved by **Expected Attention**.

---

## 3. Method: Expected Attention

### 3.1 Distributional property (the foundation)
- **Empirical finding** (Fig 1, App B; cites Liu et al. 2025, massive-activations Sun et al., Xiao et al.): **hidden states before attention/MLP are ~Gaussian**, zero-mean unimodal (intermediate layers more Laplacian). Validated across Llama3.1-8B, Qwen3-8B, Gemma3-12B (Figs 7–9).
- Since $q_t = R_t W_Q h_t$ is a **linear** map of $h\sim\mathcal N(\mu,\Sigma)$, queries inherit a Gaussian:
$$q_t \sim \mathcal N(\mu_{q_t},\Sigma_{q_t}),\quad \mu_{q_t}=R_t W_Q\mu,\quad \Sigma_{q_t}=R_t W_Q \Sigma W_Q^\top R_t^\top$$
- Holds even under **QK-norm** (Qwen3/Gemma3) where the query map isn't strictly linear — queries still look Gaussian empirically.

### 3.2 Position-averaged RoPE
- Future queries sit at unknown positions → can't pick one $R_t$. **Average RoPE over the next $T$ positions** to get a single tractable query dist:
$$\bar q \sim \mathcal N(\bar\mu_q, \bar\Sigma_q),\quad \bar\mu_q=\bar R W_Q\mu,\quad \bar\Sigma_q=\bar R W_Q\Sigma W_Q^\top \bar R^\top,\quad \bar R=\tfrac1T\sum_{j=1}^T R_{t+j}$$
- $T$ = `n_future_positions` = **512** in experiments.

### 3.3 Closed-form expected score
- The unnormalized score $z_i = \exp(q^\top k_i/\sqrt d)$ is a function of Gaussian $q$. Its expectation is the **moment-generating function of a Gaussian** — closed form:
$$\hat z_i = \mathbb E_{\bar q\sim\mathcal N(\bar\mu_q,\bar\Sigma_q)}\!\left[\exp\!\Big(\tfrac{\bar q^\top k_i}{\sqrt d}\Big)\right] = \exp\!\left(\frac{\bar\mu_q^\top k_i}{\sqrt d} + \frac{k_i^\top \bar\Sigma_q k_i}{2d}\right)$$
- **Mean term** $\bar\mu_q^\top k_i/\sqrt d$: expected dot product. **Covariance term** $k_i^\top\bar\Sigma_q k_i/2d$: variance correction (queries that vary a lot can occasionally align strongly with $k_i$).
- **Expected attention weight**: softmax over keys, $\hat a_i = \hat z_i / \sum_j \hat z_j$.
- **Expected contribution** (plug into $\star$):
$$\widehat{\|\Delta h_i\|} = (\hat a_i + \epsilon)\,\|W_o v_i\|$$
- In practice use $\|v_i\|$ instead of $\|W_o v_i\|$ — minor accuracy loss, big memory save (don't need $W_o$ projection).
- **Validated**: $\hat a_i$ strongly correlates with observed attention across layers/heads (Fig 10), and Expected Attention gives the **lowest residual reconstruction error** $\|h - h_{\text{compr}}\|$ vs all baselines (Fig 6).

### 3.4 Algorithm
1. Estimate query mean $\mu$, cov $\Sigma$ from observed hidden states.
2. Compute avg RoPE $\bar R$ over next $T$ positions; rotate $\mu,\Sigma$.
3. Closed-form $\hat z_i \to$ softmax $\to \hat a_i$.
4. Rescale by $\|v_i\|$.
5. Evict the lowest-scoring $r\%$ KV pairs.

```python
def compress(queries, keys, values, compression_ratio):
    mean_query, cov_query = compute_statistics(queries)        # Gaussian fit
    scores = matmul(mean_query, keys.T) / math.sqrt(d)
    scores += einsum("i,ij,j->", keys, cov_query, keys) / (2 * d)  # cov term
    scores = softmax(scores, dim=-1) * values.norm(dim=-1)     # + value norm
    n_kept = int(keys.size(0) * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    return keys[indices], values[indices]
```

### 3.5 Extras
- **Head-adaptive compression** (AdaKV / Feng et al.): per-layer adaptive budgets — more important heads keep more KV pairs (heads serve different roles).

---

## 4. Prefill vs Decode

Two inference phases with different compute profiles; a good method must handle **both** (many baselines target only one):
- **Prefill** (compute-bound, parallel over prompt): SnapKV, TOVA target this.
- **Decode** (memory-bound, autoregressive, appends KV every step): StreamingLLM, KNorm target this. Reasoning models (chain-of-thought) generate huge intermediate caches → decode compression matters.

**Expected Attention handles both:**

| | Prefill | Decode |
|---|---|---|
| **Query stats from** | the prompt's hidden states (one-shot before generation) | a **rolling buffer of 128 hidden states** from recent decode steps |
| **When compress** | once, after prefill | **every 512 generation steps** |
| **Assumption** | no question assumed about context (realistic; avoids favoring SnapKV which peeks at the query) | KV cache grows to a predetermined size, then eviction begins |

---

## 5. Experiments

**Models**: Llama3.1-8B (128k), Qwen3-8B (32k), Gemma3-12B (128k) for prefill. Reasoning models Qwen-15B-R1, Qwen-7B-R1, OpenMath-Nemotron-14B for decode. 8×H100, bf16, batch 1.
**Benchmarks**: LongBench, Ruler (4k/16k), Needle-in-a-Haystack (NIAH, up to 125k) for prefill; Aime25, MATH-500 for decode.
**Baselines**: SnapKV, TOVA (attention-based), KeyDiff (key-embedding distance), DuoAttention (trainable) for prefill; KNorm, StreamingLLM, KeyDiff for decode.
**Hyperparams**: $\epsilon=0.02$ (0 for NIAH), $T=512$.

### 5.1 Results
- **LongBench** (Fig 2, Table 3): best compression-vs-score tradeoff across most of the 10 datasets / 6 categories. At 90% compression Qwen3-8B: EA 39.71 vs SnapKV 34.57, KeyDiff 20.69.
- **Ruler** (Table 1): strongest at high compression ratios. Qwen3-8B 16k @ 90%: EA **62.7** vs TOVA 52.4, SnapKV 26.8, KeyDiff 53.1. KeyDiff does well on Llama but **collapses on Qwen3/Gemma3** (likely due to QK-norm). EA robust across all.
- **NIAH** (Fig 3): comparable to DuoAttention (trainable!), far more stable than other heuristics regardless of needle depth / context length. At 50% compression on Qwen3-8B: **no accuracy loss** with 2× smaller cache (Fig 5b).
- **Decode — Aime25 / MATH-500** (Fig 4, Table 2): matches or beats baselines at all sizes, especially strong at **4× and 12×** compression. Confirms reasoning traces are highly redundant. MATH-500 Qwen-R1-7B @ 12×: EA **0.49** vs KeyDiff 0.35, KNorm 0.12.
- **Memory** (Fig 5a): savings grow with context — up to **15 GB less** at 120k tokens (Llama3.1-8B, 90% compression).

### 5.2 Ablations / design choices
- **Covariance term** (`use_covariance`): more accurate but more expensive; can drop to mean-only.
- **Value norm rescaling** (`use_vnorm`): accounts for magnitude of written info.
- **$W_o V$ vs $V$**: using raw $V$ instead of $W_o V$ trades a tiny accuracy drop for much lower memory.
- **Quantization is orthogonal** — composable with EA (KIVI, KVQuant, NQKV).

---

## 6. Limitations
- **Training-free ceiling**: doesn't match *trainable* methods (DMC, learned compression). Intentional — no retraining needed. Future: combine theory with light fine-tuning.
- **Manual compression ratio** — no automatic per-scenario budget selection.
- **PyTorch impl not optimized** — custom CUDA kernels would speed it up.

---

## 7. KVPress (the released library)
- PyTorch + HuggingFace `transformers`. Implements compression via **forward hooks** on each attention layer — no architecture changes, no custom cache engine.
- After each attention layer's forward pass, the hook computes scores and evicts low-score KV pairs in-place, before the next layer.
- >20 methods + public **leaderboard** for standardized benchmarking. Explicitly **not** a deployment engine — prioritizes readability/fair comparison.

---

## 8. Implementation (kvpress)

### 8.1 Class hierarchy
- `BasePress` → defines `forward_hook` + `__call__` context manager (registers/removes hooks).
- `ScorerPress(BasePress)` → implements generic `compress` = score then top-k keep; subclasses only implement `score()`.
- `ExpectedAttentionPress(ScorerPress)` → implements `score()`.
- `DecodingPress(BasePress)` → wraps a `ScorerPress` to run during **decode** instead of prefill.

### 8.2 How the hook works (`base_press.py`)
- `__call__(model)` is a context manager: for every attention layer it sets `layer.self_attn.rotary_emb = language_model.rotary_emb` (so the press can recompute RoPE) and registers `forward_hook` with `with_kwargs=True`.
	- **Gemma3**: skips sliding-window layers, only compresses full-attention layers.
- `forward_hook(module, input, kwargs, output)`:
	- Pulls `hidden_states`, `past_key_values`, `cache_position` from kwargs.
	- **Prefill-only guard**: `if cache_position[-1] > q_len: return output` → only fires during prefill (default `BasePress`).
	- `extract_keys_and_values(cache, layer_idx)` → calls `compress` → writes compressed `cache_layer.keys/.values` back **in place** (handles `QuantizedCache` by re-quantizing).

### 8.3 Generic scorer compress (`scorer_press.py`)
```python
scores = self.score(module, hidden_states, keys, values, attentions, kwargs)  # (b, n_kv_heads, k_len)
n_kept = int(k_len * (1 - self.compression_ratio))
indices = scores.topk(n_kept, dim=-1).indices.unsqueeze(-1).expand(-1,-1,-1, head_dim)
keys   = keys.gather(2, indices).contiguous()    # keep highest-score positions
values = values.gather(2, indices).contiguous()
```
- `compression_ratio == 0` → no-op early return.

### 8.4 `ExpectedAttentionPress.score()` — the core
**Config params** (dataclass fields):

| param | default | meaning |
|---|---|---|
| `compression_ratio` | 0.0 | fraction of KV pairs removed |
| `n_future_positions` | 512 | $T$, future positions to average RoPE over |
| `n_sink` | 4 | initial sink tokens never pruned + excluded from stats (attention-sink phenomenon) |
| `use_covariance` | True | include the $k^\top\bar\Sigma_q k$ term vs mean-only |
| `use_vnorm` | True | rescale by $\|v_i\|$ |
| `epsilon` | 0.0 | additive constant before vnorm rescale |

**Step 1 — query statistics** (`get_query_statistics`):
```python
h = hidden_states[:, self.n_sink:]                  # drop sink tokens (outliers)
query_states = get_prerope_query_states(module, h)  # (b, n_heads, s, d), PRE-RoPE
mu = query_states.mean(dim=2, keepdim=True)         # mean over the seq dim
# covariance per head:  cov[b,n,i,j] = E[(q-mu)_i (q-mu)_j]
centered = query_states - mu
cov = einsum("bnsi,bnsj->bnij", centered, centered) / h.shape[1]   # (b, n_heads, d, d)
mu, cov = self.apply_avg_rope(module, mu.squeeze(2), cov, q_len)
```
- `get_prerope_query_states` (utils): runs `q_proj` (or slices Phi3 fused `qkv_proj`), reshapes to `(b, n_heads, s, d)`. **Applies `q_norm` for Qwen3/Gemma3 (QK-norm)** — so stats reflect the actually-used query distribution. Statistics are computed **before RoPE** because RoPE is then applied analytically.
- In **prefill** `hidden_states` = prompt; in **decode** it's the 128-token rolling buffer (passed in by `DecodingPress`).

**Step 2 — average RoPE** (`apply_avg_rope`): builds the rotation as a closed-form matrix and averages it.
```python
position_ids = arange(q_len, q_len + n_future_positions)   # the FUTURE positions
cos, sin = module.rotary_emb(mu, position_ids)             # (T, d)
Id = eye(head_dim)
# P implements the "rotate-half" pairing of RoPE as a linear operator:
P[d//2:, :d//2] =  eye(d//2);  P[:d//2, d//2:] = -eye(d//2)
R = cos[:,None]*Id + sin[:,None]*P     # (T, d, d) rotation matrices
R = R.mean(dim=0)                      # \bar R : average over T future positions
mu  = mu @ R.T                         # \bar\mu_q = \bar R W_Q \mu
cov = R @ cov @ R.T                    # \bar\Sigma_q = \bar R \Sigma_q \bar R^T
```
- Reconstructs RoPE as an explicit $d\times d$ matrix $R=\cos\cdot I + \sin\cdot P$ (vs the usual rotate-half trick) precisely so it can be **averaged across positions** and **applied to a covariance matrix** ($R\Sigma R^\top$). $P$ encodes the half-swap with sign flip.

**Step 3 — closed-form scores** (`score`):
```python
keys = keys[:, :, self.n_sink:]; values = values[:, :, self.n_sink:]   # drop sinks
mean_query, cov_query = self.get_query_statistics(module, hidden_states)
num_kv_groups = config.num_attention_heads // num_key_value_heads      # GQA
keys = repeat_kv(keys, num_kv_groups).transpose(2, 3)                  # expand KV heads to Q heads
# mean term:  \bar\mu_q^T k_i / sqrt(d)
scores = matmul(mean_query.unsqueeze(2), keys).squeeze(2) / sqrt(d)
if use_covariance:                                                     # + k_i^T \bar\Sigma_q k_i / 2d
    scores += einsum("bhin,bhij,bhjn->bhn", keys, cov_query, keys) / d / 2
scores = softmax(scores, dim=-1)                                       # \hat a_i
scores = scores.view(b, n_kv_heads, num_kv_groups, q_len).mean(dim=2)  # avg over GQA group → per-KV-head
if use_vnorm:
    scores = (scores + epsilon) * values.norm(dim=-1)                  # (\hat a_i + eps) * ||v_i||
# re-insert sinks with score = max+1 so top-k always keeps them
scores = F.pad(scores, (n_sink, 0), value=scores.max().item() + 1)
```
- **GQA handling**: keys repeated to match query heads, scored per query head, then **averaged back down** to KV-head granularity (since pruning is per KV head).
- **Sink protection**: padded back with `max+1` → guaranteed kept by the `topk` in `ScorerPress.compress`.
- Note: the exponential in $\hat z_i$ is implicitly handled by computing the log-score (mean+cov terms) then applying `softmax` directly — numerically equivalent to softmax over $\exp(\cdot)$.

### 8.5 Decode path (`DecodingPress`)
- Wraps a `ScorerPress` (`base_press`), overrides `forward_hook` to fire **only during decode** (`cache_position[-1] > q_len`).
- **Per-layer rolling buffer** of hidden states (`hidden_states_buffer`, default cap `hidden_states_buffer_size=256`; paper uses 128) + per-layer step counter.
- Every `compression_interval=512` steps (or when `q_len >= target_size`): concatenate buffered hidden states, call `base_press.compress(...)` with a `target_size`-derived compression ratio (found via binary search in `_find_target_compression_ratio` to hit an exact token budget, default `target_size=2048`), write compressed cache back in place, reset counter + clear buffer.
- So Expected Attention plugs in unchanged — `DecodingPress` just feeds it the buffered hidden states and re-derives the ratio.

---

## 9. Connections / commentary
- **Residual-stream framing** ($\star$) ties this to interpretability (Elhage transformer-circuits): pruning = minimizing perturbation to what each head *writes* to the residual stream. Cleaner objective than raw attention mass (H2O/SnapKV).
- **Gaussian-MGF closed form** is the elegant trick: replaces an intractable expectation over future queries with a single $\exp(\text{mean}+\tfrac12\text{var})$. Analogous to how Gaussian assumptions linearize hard expectations elsewhere (e.g. probit/Gaussian-process approximations, reparameterization).
- **RoPE-as-matrix-averaging** lets positional encoding be marginalized over an unknown future window — neat way to make a position-dependent op position-agnostic.
- **vs KeyDiff/KNorm**: those are pure geometry on keys; EA additionally models the *query side* distribution and the *value* magnitude → why it survives QK-norm models where key-geometry methods collapse.
- **Orthogonal to quantization** and to architectural cache reducers (GQA, MLA, sliding window) — stackable.
