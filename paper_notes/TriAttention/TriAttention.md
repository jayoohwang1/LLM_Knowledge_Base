# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

> Mao, Lin, Huang et al. (MIT / NVIDIA / ZJU), 2026. arXiv:2604.04921. Code: `github.com/WeianMao/triattention`.

**TL;DR**: Long CoT reasoning blows up the KV cache. Existing KV-compression methods score key importance from *recent post-RoPE queries* — but RoPE rotates queries with position, so only a tiny window of queries is "representative" → unstable, misses long-dormant keys. TriAttention instead works in **pre-RoPE space**, where it observes **Q/K concentration**: pre-RoPE Q and K vectors cluster tightly around fixed non-zero centers, *stable across position and content*. Substituting these centers into the RoPE attention formula collapses the attention logit into a **trigonometric series in Q-K distance** $\Delta$, i.e. each head has a *predictable distance-preference curve*. TriAttention scores cached keys with this curve (Q center as proxy for all future queries) + a norm/magnitude term, keeps top-$B$. Matches Full Attention accuracy at **2.5× throughput / 10.7× KV memory reduction** on AIME25 32K-token generation; baselines get ~half the accuracy at the same budget.

---

## 1. Motivation: the KV bottleneck in long reasoning

- **Long reasoning = tens of thousands of CoT tokens** (DeepSeek-R1, o1-style). KV cache grows linearly with sequence → severe memory bottleneck during generation.
- **KV cache compression**: keep only a budget $B$ of "important" KV pairs. Core problem = **importance estimation** = which past tokens will be attended to by *future* queries.
- **Existing methods score from post-RoPE recent queries** (SnapKV, H2O, Scissorhands, R-KV, LazyEviction, Quest...). Two failure modes flagged:
  - **Attention-based methods**: post-RoPE queries **rotate with position** (Fig 2B — vectors smear into arc patterns). Only the most recent queries are "up to date," giving a tiny observation window. A key getting low attention *during this window* can be permanently evicted even if it becomes critical later → bad for retrieval heads / reasoning where tokens stay dormant then reactivate. Empirically, widening the observation window **doesn't help** — performance peaks at ~25 queries then declines (Zhang et al. 2025).
  - **Norm-based methods** (VATP): use only vector *magnitudes*, ignore direction. But in post-RoPE space, direction is **entangled with positional rotation** (changes continuously with position) → can't cleanly exploit directional info.
- **Key escape hatch**: pre-RoPE vectors are *unaffected by positional rotation*, and they feed *directly* into attention. So measure importance there.

---

## 2. The key observation: Q/K Concentration

> **Observation 3.1 (Prevalent Q/K Concentration)**: In **pre-RoPE** space, across a large fraction of attention heads, Q and K vectors are **highly concentrated around fixed non-zero centers**, and this concentration is **stable across token positions and input contexts**.

- **Why pre-RoPE?** RoPE rotates by a *position-dependent* angle. Before rotation, the vectors are positionally invariant — so the centers are an **intrinsic** property of the head, not an artifact of where you look.
- **Quantified via Mean Resultant Length (MRL)** $R$ (standard in directional statistics, Mardia & Jupp 1999):
  $$ R = \frac{\|\mathbb{E}[q]\|}{\mathbb{E}[\|q\|]} \in [0,1], \qquad R\to1 \text{ perfect concentration},\; R\to0 \text{ uniform dispersion} $$
  - Computed **per frequency band** $f$: $R_f = \|\mathbb{E}[q_f]\| / \mathbb{E}[\|q_f\|]$.
  - On Qwen3-8B the vast majority of heads have $R\to1$ (Fig 2C). ~**90% of heads have $R>0.95$** regardless of domain.
  - **Model-intrinsic**: MRL nearly identical across Math/Coding/Chat domains (0.977–0.980). Robust to calibration data.
- **Generalizes across architectures**: GQA (Qwen3-8B) and **MLA** (GLM-4.7-Flash) — MLA actually shows *stronger* concentration (96.6% of heads $R>0.95$ vs 84.7% for GQA; Appendix I).

---

## 3. From concentration → predictable distance preferences (the trig series)

The whole method rests on a chain: **concentration ⇒ attention logit is a function of Q-K distance alone ⇒ that function is a trigonometric series ⇒ its peaks are predictable from the centers**.

### 3.1 RoPE attention as a cosine sum
- RoPE in complex form (per frequency band $f$, position $p$): $\tilde q_f(p) = q_f e^{i\omega_f p}$, $\tilde k_f(p)=k_f e^{i\omega_f p}$, with $\omega_f = \theta^{-2f/d}$ ($\theta=10000$).
- Dot product (real part) summed over bands gives the **pre-softmax logit**:
  $$ \langle q,k\rangle_\Delta = \sum_f \|q_f\|\|k_f\|\cos(\omega_f \Delta + \phi_f), \qquad \Delta = p_q - p_k,\; \phi_f = \arg(q_f)-\arg(k_f) $$
- **Crucially the coefficients $\|q_f\|\|k_f\|$ and phases $\phi_f$ depend on the specific Q/K vectors**, which normally vary token-to-token → not a clean function of $\Delta$.

### 3.2 The collapse
- **When Q/K are concentrated**, approximate $q_f \approx \bar q_f$, $k_f \approx \bar k_f$ (their centers). Coefficients become **constants** → logit is now a pure function of distance $\Delta$:
  $$ \langle q,k\rangle_\Delta \approx \sum_f \big[a_f\cos(\omega_f\Delta) + b_f\sin(\omega_f\Delta)\big],\quad a_f=\|q_f\|\|k_f\|\cos\phi_f,\; b_f=-\|q_f\|\|k_f\|\sin\phi_f $$
- This is a **trigonometric series in $\Delta$** = an **attention-vs-distance curve**.
  - **Analogy**: Fourier synthesis. The RoPE frequencies $\omega_f$ are fixed (geometric progression, *not* harmonic $n\omega_0$); the learned **Q/K centers set the coefficients** → the head can "synthesize" arbitrary distance-dependent attention patterns.
  - Curve typically peaks at specific $\Delta$: some heads peak at small $\Delta$ (**local/nearest-key attention**), others at large $\Delta$ (**attention sinks**). **Different centers → different preferred distances.**
- **Validation — Reconstruction Correlation $\bar r$**: substitute calibration-estimated centers $\mathbb{E}[q_f],\mathbb{E}[k_f]$ into the series, predict logits at log-spaced $\Delta\in\{1,2,4,8,\dots\}$, correlate with actual logits (Pearson), avg over queries.
  $$ \hat s(\Delta) = \sum_f \|\mathbb{E}[q_f]\|\,\|\mathbb{E}[k_f]\|\cos(\omega_f\Delta+\phi_f), \qquad \bar r = \tfrac1N\sum_i \rho(\mathbf a_i, \hat{\mathbf s}) $$
  - First head: $\bar r=0.72$ (Fig 2D). Across Qwen3/Qwen2.5/Llama3 heads, $\bar r$ peaks 0.6–0.9, **mean > 0.5** (Fig 3, right-skewed).
  - **Log-spaced $\Delta$ sampling** is deliberate: prevents the (numerous) near distances from dominating; ensures long-range patterns are covered.

---

## 4. TriAttention scoring rule

Keys are *already cached* (positions known). Score each key $k$ at distance $\Delta=p_q-p_k$ from a hypothetical future query, using the **Q center as a proxy for all future queries** (justified by concentration). Two additive components:

### 4.1 Trigonometric Series Score $S_\text{trig}$ (distance preference)
$$ S_\text{trig}(k,\Delta) = \sum_f \|\mathbb{E}[q_f]\| \cdot \|k_f\| \cdot \cos(\omega_f\Delta + \phi_f), \qquad \phi_f = \arg(\mathbb{E}[q_f]) - \arg(k_f) $$
- $\mathbb{E}[q_f]$ = offline-calibrated Q center; $k_f$ = the actual cached key (its real pre-RoPE value, since we have it).
- Captures: *given this head's distance-preference curve, how much attention does a key at this distance get?*

### 4.2 Norm-Based Score $S_\text{norm}$ (complementary salience)
- The trig series assumes Q/K sit *exactly* at centers; in practice there's spread. Norm term accounts for it.
- Base: $S_\text{norm}^{(0)}(k) = \sum_f \mathbb{E}[\|q_f\|]\cdot\|k_f\|$. Unlike pure norm methods (only $\|k\|$), this weights each band by expected query contribution.
- **Why needed (Fig 4)**: $S_\text{trig}$ alone assigns high score to nearby keys but can over/under-rank; some far keys with low norm get little real attention — the norm term correctly flags **low-norm early tokens as unimportant** despite their large $\Delta$. The two components are complementary.

### 4.3 Adaptive weighting via concentration ($1-R_f$)
- Trust $S_\text{trig}$ more when concentration is high; lean on $S_\text{norm}$ when low. Scale each band by $(1-R_f)$:
  $$ S_\text{norm}(k) = \sum_f (1-R_f)\,\mathbb{E}[\|q_f\|]\,\|k_f\| = \sum_f \big(\mathbb{E}[\|q_f\|] - \|\mathbb{E}[q_f]\|\big)\,\|k_f\| $$
  - The rewrite uses $R_f = \|\mathbb{E}[q_f]\|/\mathbb{E}[\|q_f\|]$, so $(1-R_f)\mathbb{E}[\|q_f\|] = \mathbb{E}[\|q_f\|] - \|\mathbb{E}[q_f]\|$ = **the gap between mean-of-norms and norm-of-mean** = exactly the dispersion not captured by the center. **High $R_f$ (strong concentration) → norm term ~0, $S_\text{trig}$ dominates.**
- **Combined**: $\boxed{S(k,\Delta) = S_\text{trig}(k,\Delta) + S_\text{norm}(k)}$

### 4.4 Averaging over future offsets
- A key may be queried from *any* future position, so importance averages over many distances:
  $$ \tilde S(k) = \frac{1}{|\mathcal D|}\sum_{\delta\in\mathcal D} S(k, \Delta+\delta), \qquad \mathcal D = \{1,2,4,\dots,2^{16}\}\ \text{(geometric)} $$
  - **Geometric spacing is essential**: ablation 45.8% (geometric) vs **28.7% (linear)**, −17.1%. Near distances need dense sampling.

### 4.5 GQA handling: normalize-then-aggregate
- In GQA one KV head is shared by $G$ query heads, each with its own $\mathbb{E}[q_f]$ → scores live on different scales, not directly comparable. Fix:
  - **z-score normalize within each query head**: $\hat S^{(g)}(k) = (\tilde S^{(g)}(k)-\mu_g)/\sigma_g$.
  - **aggregate by max over the group**: $S_\text{final}(k) = \max_g \hat S^{(g)}(k)$ — a key is kept if *any* sharing head deems it important. Retain top-$B$ by $S_\text{final}$.

### 4.6 Window-based pruning (amortize cost)
- Scoring every decode step is expensive. Trigger **once per window of $\beta=128$ generated tokens**: when the 128th token of an interval is produced and cache > budget $B$, score all keys, keep top-$B$, evict rest (follows R-KV).

---

## 5. Experiments

**Setup**: Models = Qwen3-8B, DeepSeek-R1-Distill-{Llama-8B, Qwen-7B}, GPT-OSS-20B. Math reasoning: AIME24/25 (30 problems each, pass@avg over 8 samples), MATH-500. Gen: 32K tokens, temp 0.6, top-p 0.95, bf16, FlashAttention-2 (FA-3 for GPT-OSS on H100). Default budget 2048 (512 for DS-Llama & MATH-500). Baselines: Full Attention (upper bound), SnapKV, R-KV.

### 5.1 Reasoning accuracy (Table 1, budget 2048; DS-Llama 512)
| Method | AIME24 Qwen3-8B | DS-Llama | DS-Qwen | GPT-OSS | AIME25 Qwen3-8B | DS-Llama | DS-Qwen | GPT-OSS |
|---|---|---|---|---|---|---|---|---|
| Full Attn | 57.1 | 50.4 | 43.8 | 69.2 | 40.8 | 31.4 | 34.2 | 60.0 |
| SnapKV | 34.6 | 5.0 | 34.6 | 48.3 | 20.0 | 6.7 | 25.0 | 36.7 |
| R-KV | 25.4 | 25.8 | 34.6 | 49.6 | 17.5 | 11.2 | 23.3 | 39.2 |
| **TriAttention** | **42.1** | **33.8** | **42.5** | **59.2** | **32.9** | **19.6** | **30.0** | **49.2** |
- **Best compression method across the board**, closely approaches Full Attention; nearly **doubles** R-KV on Qwen3-8B AIME25 (32.9 vs 17.5).
- **MATH-500 (budget 512)**: TriAttn 56.0 vs R-KV 46.4 / SnapKV 49.2 (Qwen3-8B); on DS-Llama 80.6 vs Full 82.4.

### 5.2 Throughput / efficiency (Table 4, Qwen3-8B; 16K decode, single A100-80GB)
| Benchmark | Budget | Full Acc | TriAttn Acc | Full Tput | TriAttn Tput | Speedup |
|---|---|---|---|---|---|---|
| MATH-500 | 1024 | 69.6 | 68.4 | 222.8 | 1405.2 | **6.3×** |
| AIME24 | 4096 | 57.1 | 54.6 | 222.8 | 413.9 | **1.9×** |
| AIME25 | 3072 | 40.8 | 40.8 | 222.8 | 563.5 | **2.5×** |
- **Headline (Fig 1)**: at *equal accuracy* to Full Attn on AIME25, **2.5× throughput** and **10.7× KV memory reduction**. R-KV at the same efficiency gets ~half the accuracy.
- vs R-KV (Table 5): at comparable accuracy needs only **half the budget** (1024 vs 2048) + 85% higher throughput; at comparable memory, higher accuracy (+8% MATH-500, +15% AIME24). **Strictly better trade-off.**

### 5.3 Budget sweep (Fig 5A–C) & Memory retention (5D)
- Across budgets 512→4096, TriAttn consistently beats R-KV, biggest gap at low-mid budgets. Matches/exceeds Full Attn at higher budgets (AIME25 43.3 @ 4096 > Full 40.8).
- **Recursive State Query (DFS) benchmark** (Appendix C): simulates depth-first search; deeper recursion = more intermediate states to retain = higher memory pressure. *Stack exact-match* metric (strictest). TriAttn ≈ Full Attn up to depth 16 (even beats it at 8,12); lags only beyond depth 18. **R-KV catastrophically degrades from depth 16** (~61%@14 → 31%@16) — its pruning loses intermediate states.

### 5.4 Ablations (Table 3, budget 2048)
- **Remove $S_\text{trig}$** (norm only): AIME24 42.1→18.8, AIME25 32.9→21.2. Distance preferences are essential.
- **Remove $S_\text{norm}$** (trig only): AIME24 45.8→40.4 (−5.4). Norm gives complementary salience.
- **Remove $R$ weighting** (use unweighted $S_\text{norm}^{(0)}$): AIME24 42.1→41.3, AIME25 32.9→28.7. Concentration-adaptive weighting helps.
- **Cross-domain calibration** (calibrate on coding, eval on reasoning): AIME25 29.2 (coding) vs 32.9 (reasoning) — stats generalize, Q/K statistics are model-intrinsic not domain-specific.
- **Future offset design** (Table E): max distance 128→4096 raises acc 41.7→48.8; geometric ≫ linear spacing (45.8 vs 28.7).
- **Calibration sensitivity** (Table F): stable across 50k–960k calibration tokens (45.4–45.8); robust to data quality (HTML 46.2, Code 43.3, Chat 46.7).

### 5.5 General tasks (Appendix E/F)
- **LongBench** (16 subtasks, Qwen3-8B 50% KV): TriAttn avg **48.1**, wins 11/16, +2.5 over Ada-KV+SnapKV. **RULER** avg **66.1** (vs SnapKV 55.6, StreamingLLM 61.1). Beats H2O 10/12 where H2O fits in 48GB.
- **Real-world (Appendix J)**: Qwen3-32B INT4 + OpenClaw agent on a single RTX 4090 (24GB). Full Attn OOMs mid-session; TriAttn keeps KV within budget → completes the multi-turn document-report task.

### 5.6 Limitations (Appendix A)
- Current implementation gets memory/throughput gains but **lacks a dedicated hardware-aware kernel** for the trig-series scoring + pruning → latency could improve further.
- **Requires offline calibration** to estimate Q/K centers (though robust/cheap).
- Evaluation focused on math reasoning + LongBench/RULER; broader domains (coding, agentic), head-specific budgets = future work.
- Trig approximation degrades for the minority of **low-concentration heads** (mitigated by the $S_\text{norm}$ + $(1-R_f)$ weighting).

---

## 6. Implementation

Two code paths: a **HF reference** (`triattention/methods/`, used for the paper experiments via a `model.forward` monkeypatch) and a production **vLLM plugin** (`triattention/vllm/`, Triton kernels). The math is identical; below maps paper formulas → code.

### 6.1 Calibration — estimating Q/K centers (`scripts/calibrate.py`)
- Single forward pass on plain text; **forward-pre-hook on each `self_attn`** recomputes $Q$ from `q_proj`, applies RoPE, then **inverts RoPE back to pre-RoPE** (`_invert_rope`) per head.
- Per head, convert pre-RoPE Q to complex pairs (`_to_complex_pairs`) and store **two statistics**:
  ```python
  q_mean_complex = q_complex.mean(dim=0)    # E[q_f] = the Q CENTER (complex), per band
  q_abs_mean     = q_complex.abs().mean(dim=0)  # E[||q_f||] = mean of per-band norms
  ```
  - These are exactly $\mathbb{E}[q_f]$ (center, drives $S_\text{trig}$) and $\mathbb{E}[\|q_f\|]$ (drives $S_\text{norm}$). Note $R_f = \|\text{q\_mean\_complex}\| / \text{q\_abs\_mean}$ is *implicit* — the code never stores $R_f$ explicitly; the $(1-R_f)$ weighting falls out algebraically (see §6.4 `extra`).
- Saved as `.pt` with `metadata` (head_dim, rope_style, rope_type, `sampled_heads`) + per-head `{q_mean_real, q_mean_imag, q_abs_mean}`. Loaded by `load_head_frequency_stats` → `HeadFrequencyStats(q_mean_complex, q_abs_mean)`. Precomputed stats shipped in `triattention/vllm/stats/` (qwen3_32b_int4, gpt_oss_120b).
- **No per-key calibration**: only Q centers are calibrated; K values are read live from the cache.

### 6.2 Core class `TriAttention` (`methods/triattention.py`)
- `TriAttentionConfig`: `budget`, `divide_length=128`, `offset_max_length=65536`, `score_aggregation="mean"`, plus toggles: `normalize_scores`, `per_head_pruning`, `per_layer_perhead_pruning`, `disable_mlr`, `disable_trig`, `use_slack_trigger`, `allow_prefill_compression`.
- Constructor builds: `self.rotary` (HF RotaryEmbedding, RoPE-aligned via `verify_rotary_alignment` against the live model's `inv_freq`), `self.omega = inv_freq[:freq_count]` ($\omega_f$), `self.offsets = build_geometric_offsets(...)` (= $\{1,2,4,\dots\}$, matches $\mathcal D$), `self.freq_scale_sq` (RoPE attention-scaling² per band, e.g. YaRN). GQA: `num_key_value_groups = num_attention_heads // num_key_value_heads`.
- Tracks `cache_positions` (and per-head / per-layer-per-head variants when independent pruning is on) and `absolute_position` (= `round_start`).

### 6.3 Hooking into attention/generation (`apply_triattention_patch`)
- Replaces `model.forward` with `triattention_forward` (via `types.MethodType`). Per step it:
  1. Detects new generation (empty cache) → `reset_compression_state()`.
  2. Overrides `position_ids` / `cache_position` from its own `absolute_position` (so positions stay correct after eviction shrinks the cache).
  3. Calls `orig_forward`, then converts the returned `Cache` to a legacy tuple, extends `cache_positions`.
  4. **Trigger**: default (R-KV style) compress when `effective_size >= budget AND absolute_position % divide_length == 0`; `use_slack_trigger` variant fires at `budget + divide_length`.
  5. `compute_keep_indices(pkv_tuple)` → `index_select`/`gather` the kept K,V per layer (1D global / 2D per-head / 3D per-layer-per-head), rebuild `DynamicCache`, update `cache_positions`.
- vLLM equivalent: `install_vllm_integration_monkeypatches(patch_scheduler=True, patch_worker=True)` — auto-activates as a vLLM plugin (`[TriAttention] Runtime (V2) plugin activated`). HF-transformers path patched per-arch in `integration/monkeypatch.py` (`replace_qwen3/qwen2/llama/gpt_oss`).

### 6.4 The scoring math in code (`pruning_utils.py`)
Two functions = the heart of the method.

**`compute_frequency_statistics_from_means(q_mean_complex, q_abs_mean, k_unrot, ...)`** — builds per-key coefficients. K is un-RoPE'd first (`invert_rope`) then `to_complex_pairs`:
```python
q_mean_abs = |q_mean_complex|                       # ||E[q_f]|| = norm of center
k_abs      = |k_complex|                             # ||k_f||
relative   = q_mean_complex.unsqueeze(0) * conj(k_complex)
phi  = atan2(relative.imag, relative.real)           # phi_f = arg(E[q_f]) - arg(k_f)
amp  = q_mean_abs.unsqueeze(0) * k_abs               # ||E[q_f]|| * ||k_f||   -> S_trig amplitude
extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs   # (E[||q_f||] - ||E[q_f]||) * ||k_f||  -> S_norm
# disable_mlr branch: extra = q_abs_mean * k_abs  (drops the (1-R_f) gap, uses raw norm product)
```
- **`extra` is exactly $S_\text{norm}$**: $(\mathbb{E}[\|q_f\|]-\|\mathbb{E}[q_f]\|)\,\|k_f\|$ = the §4.3 rewrite of $(1-R_f)\,\mathbb{E}[\|q_f\|]\,\|k_f\|$. The $(1-R_f)$ weighting is realized as this subtraction, never as an explicit ratio.

**`score_keys_for_round(key_indices, round_start, amp, phi, omega, extra, offsets, freq_scale_sq, ...)`**:
```python
base_delta = round_start - key_indices                 # Δ = p_q - p_k
delta_grid = base_delta[:,None] + offsets[None,:]       # Δ + δ for each future offset δ
phase      = delta_grid[...,None]*omega + phi[:,None]   # ω_f(Δ+δ) + φ_f
cos_phase  = cos(phase)
base_scores = (amp[:,None] * freq_scale_sq * cos_phase).sum(dim=2)   # S_trig summed over bands
additive    = (extra * freq_scale_sq).sum(dim=1, keepdim=True)       # S_norm (offset-independent)
combined    = base_scores + additive                                 # S = S_trig + S_norm
return combined.mean(dim=1)  # average over offsets (max if aggregation="max")  -> tilde S(k)
```
- Maps 1:1 to $S(k,\Delta+\delta)=\sum_f \text{amp}_f\cos(\omega_f(\Delta+\delta)+\phi_f) + \text{extra}$, then $\tilde S(k)=\text{mean}_\delta$.
- `disable_trig` → drops `base_scores`, keeps only `additive` (the norm-only ablation). `freq_scale_sq` applies RoPE's per-band attention-scaling² (handles YaRN/long-rope).

### 6.5 Top-$B$ key retention (`compute_keep_indices`)
- Scores **decode tokens only**; **prefill tokens always preserved** (unless `allow_prefill_compression`). `_compute_layer_head_scores` runs per sampled head: gather K from cache, map GQA query-head→kv-head, invert RoPE with that head's positions (`positions_per_kv_head` after independent pruning), score.
- Optional z-score `normalize_scores` per head + tiny noise for tie-breaking.
- **Default = union-based global selection** (`_select_union_based`): each head picks its own top-$k$, take the **union**, then keep top-$B$ from the union by `combined = head_matrix.max(dim=0)` (max over heads); fill from residual if union < B. Prefill indices prepended, sorted.
- **`per_head_pruning`** (`_select_per_head_independent`): each KV head keeps its own top-$B$ independently → 2D keep_indices, gathered per-head. Aggregation = **mean over layers of per-layer max over the ~G heads** (deliberate, preserves per-head variance; noted as a bug-fix over naive global max). This is the GQA normalize-then-aggregate of §4.5.
- **`per_layer_perhead_pruning`**: maximum independence, each (layer, kv-head) keeps its own tokens → 3D keep_indices.

### 6.6 vLLM Triton path (`vllm/core/scoring.py`)
- `compute_scores` dispatches to `compute_scores_triton` (default) or `compute_scores_pytorch` (reference). **Key difference vs HF path**: vLLM stores keys **post-RoPE** ($K_\text{rot}=k\,e^{i\omega_f p}$), so position $p$ is already baked in → instead of inverting RoPE, it forms $Q_\text{mean}\cdot\overline{K_\text{rot}}$ directly and uses only the **query-side phase** $t\cdot\omega_f$ where $t=\text{round\_start}+\delta$:
  ```python
  prod_real = q_real*k_real + q_imag*k_imag      # Re(Q conj(K_rot))
  prod_imag = q_imag*k_real - q_real*k_imag      # Im(Q conj(K_rot))
  position_term = freq_scale_sq * (prod_real*cos(t*ω) - prod_imag*sin(t*ω))   # = S_trig
  extra_term    = k_abs * (q_abs_mean - q_mean_abs) * freq_scale_sq           # = S_norm
  scores = mean_or_max_over_offsets(position_term) + extra_term
  ```
  - Mathematically equivalent to §6.4 (avoids the lossy amp/phi reconstruction that's only valid for un-rotated K). A `trig_cache` reuses cos/sin tables across rounds via angle-addition (`cos(a+b)` identities) to skip recompute.

### 6.7 Key hyperparameters / env vars (vLLM runtime)
| Var | Default | Meaning |
|---|---|---|
| `TRIATTN_RUNTIME_KV_BUDGET` | 2048 | top-$B$ tokens kept (12000 recommended for multi-turn chat) |
| `TRIATTN_RUNTIME_DIVIDE_LENGTH` | 128 | compression trigger interval ($\beta$ window) |
| `TRIATTN_RUNTIME_WINDOW_SIZE` | 128 | recent tokens always preserved |
| `TRIATTN_RUNTIME_PRUNING_MODE` | per_head | `per_head` or `per_layer_per_head` |
| `TRIATTN_RUNTIME_SPARSE_STATS_PATH` | — | path to calibrated Q/K stats `.pt` |
| `TRIATTN_RUNTIME_PROTECT_PREFILL` | false | pin initial prompt tokens |
| `ENABLE_TRIATTENTION` | true | master on/off |
- Code-level knobs (`TriAttentionConfig`): `offset_max_length=65536` (geometric offsets $\{1..2^{16}\}$ = $\mathcal D$), `score_aggregation` (mean/max over offsets), `disable_trig` / `disable_mlr` (ablation switches), `use_slack_trigger`.
- vLLM serve needs `--enforce-eager --enable-prefix-caching false` (prefix caching incompatible with compressed entries) and a capped `--max-num-batched-tokens` so a prefill chunk can't overshoot budget before compression triggers.

---

## 7. Connections / takeaways

- **Pre-RoPE vs post-RoPE framing** is the conceptual crux — same idea behind why RoPE direction info is "entangled" and why observation-window methods plateau. Compare DuoAttention (retrieval/streaming heads) and EliteKV (RoPE freq selection).
- **Trig series = Fourier-synthesis lens on RoPE attention**: the head learns Q/K centers → sets coefficients → "designs" a distance-preference curve. Ties to Hong et al. 2024 (higher RoPE dims model token distance) and Barbero et al. 2025 (what makes RoPE useful).
- **Concentration as a model-intrinsic prior**: calibration is cheap, domain-robust, and even stronger in MLA — suggests the phenomenon is fundamental to trained attention, not a quirk of one arch.
- **Practical lever**: the $(1-R_f)$ trick elegantly folds "how much to trust the center" into a single subtraction (`E[||q||] - ||E[q]||`), no extra stored quantity.

---

## Related Notes

- **[KVTC](../KVTC/KVTC.md)** — Complementary axis: TriAttention **drops** low-scoring keys; KVTC **lossily encodes** all retained keys. Stackable (evict → transform-code).
- **[SAGE-KV](../SAGE-KV/SAGE-KV.md)** / **[Expected Attention](../Expected_Attention/Expected_Attention.md)** / **[Compactor](../Compactor/Compactor.md)** / **[R-KV](../R-KV/R-KV.md)** — Sibling **score-then-evict** methods, differing in the *importance signal*: SAGE-KV uses observed self-attention, Expected Attention estimates *future* attention, Compactor uses leverage scores, R-KV adds a redundancy penalty. TriAttention's signal is **distance preference** derived from Q/K concentration + a trigonometric series — a query-agnostic geometric prior rather than an attention readout.
- **[Physics of KV Compression](../Physics_of_KV_Compression/Physics_of_KV_Compression.md)** — Explains TriAttention's operating regime: its high KV-reduction ratios (10.7×) sit below the ~90% Global-Eviction-Ratio cliff that paper identifies as the hallucination phase transition.
