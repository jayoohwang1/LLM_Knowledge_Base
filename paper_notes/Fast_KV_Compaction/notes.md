# Fast KV Compaction via Attention Matching

**Authors**: Adam Zweiger, Xinghong Fu, Han Guo, Yoon Kim (MIT) - arXiv Feb 2026
**Code**: https://github.com/adamzweiger/compaction

---

## TL;DR

- **One-shot, training-free KV cache compaction** in latent space at compaction ratios up to 50x with little quality loss, in **seconds-to-minutes** instead of GPU-hours.
- Replaces end-to-end gradient optimization (Cartridges) with a closed-form **Attention Matching (AM)** objective: directly fit compact $(C_k, \beta, C_v)$ so attention outputs + attention mass match those of the full cache, per query, per head, per layer.
- Decomposes into subproblems each admitting closed-form (least squares / NNLS / OMP) solutions $\Rightarrow$ orders of magnitude speedup vs gradient descent.
- Bridges the gap between cheap heuristic eviction (H2O, SnapKV, KVzip) and expensive optimization (Cartridges), forming the new Pareto frontier of compaction-cost vs downstream quality.

---

## 1. Problem Setup & Motivation

- **KV cache** dominates memory in long-context LM serving (multi-turn dialogue, agentic coding, reasoning).
- Existing KV reduction:
    - **Token-space**: summarization / drop / RAG. Lossy, hurts retrieval-style downstream.
    - **Latent-space**: eviction (H2O), pruning (SnapKV, PyramidKV), merging (CaM), head sparsification (DuoAttention), reconstruction (KVzip).
    - **Optimization**: Cartridges (Eyuboglu 2025) trains a tiny latent KV via prefix-tuning + self-study. Excellent quality at 50x but **hours of GPU time per context**.
- **"Compaction"** (vs compression): the one-shot operation of shrinking a fixed KV prefix while preserving downstream behavior. Term borrowed from systems / agentic coding tools.
- **Goal**: Cartridges-level quality, heuristic-level speed.

---

## 2. Attention Matching Objective

> Replace original $(K, V) \in \mathbb{R}^{T \times d}$ with compact $(C_k, C_v) \in \mathbb{R}^{t \times d}$, $t \ll T$, plus per-token scalar bias $\beta \in \mathbb{R}^t$, such that attention behaves the same for queries of interest.

### Compatibility w/ Future Tokens (Concatenation)

- Need compaction to be valid even when arbitrary $K_{\text{fixed}}, V_{\text{fixed}}$ are appended later (user query, model continuation).
- **Key identity**: attention over $[K; K_{\text{fixed}}]$ decomposes into a **mass-weighted mixture** of attention on each block:
$$\text{Attn}(q;[K;K_{\text{fixed}}],[V;V_{\text{fixed}}]) = \frac{\text{Mass}(q;K)}{\text{Mass}(q;K)+\text{Mass}(q;K_{\text{fixed}})}\text{Attn}(q;K,V) + \dots$$
    - Same trick used by **FlashAttention / Cascade Inference**.
- $\Rightarrow$ Sufficient to match **both** (i) local attention output and (ii) **attention mass** of the compacted block on reference queries; future tokens then come out correctly via the mixture.

### Per-Query Objectives

For each reference query $q$:
$$\frac{\exp(qK^\top)V}{\sum_j \exp(qK_j^\top)} \approx \frac{\exp(qC_k^\top + \beta)\, C_v}{\sum_j \exp(q (C_k)_j^\top + \beta_j)} \quad (1)$$
$$\sum_j \exp(qK_j^\top) \approx \sum_j \exp(q(C_k)_j^\top + \beta_j) \quad (2)$$

- $\text{Mass}(q;K) := \sum_j \exp(qK_j^\top)$ - unnormalized softmax denominator.
- Eq. (1) preserves *local* attention output; Eq. (2) preserves *attention mass* (i.e. block weight under concatenation).

### Why $\beta$ (per-token attention bias)

- If $C_k \subset K$ (subset eviction) **without** $\beta$: $\text{Mass}(q;C_k) \le \text{Mass}(q;K)$ always $\Rightarrow$ block gets systematically underweighted in mixture.
- $\exp(\beta_j) > 0$ acts as a **multiplicative reweight**: one retained key can represent the mass of many removed keys.
- Also needed at $q=0$: forces matching $T$ vs $t$ (impossible without bias).
- Cost: extra $\frac{2d+1}{2d}$ memory; supported by SDPA / FlexAttention naturally.
- Related: T5 / ALiBi use per-token biases as **positional** encoding; here it's a **mass-correction** bias - different purpose.

### Logical vs Physical Length

- Compacted cache stores $t$ entries but **logical length stays $T$**: new tokens get the same RoPE phase as if uncompacted. Disentangles what cache "has seen" from its physical size.

---

## 3. Method: Construction of $(C_k, \beta, C_v)$

Jointly optimizing is hard; solve **sequentially in closed form**:

1. **Pick $C_k$** (key selection - subset of $K$).
2. **Fit $\beta$** in closed form (NNLS on mass-matching).
3. **Fit $C_v$** in closed form (least squares on attention-output matching).

### 3.1 Reference Queries $Q_{\text{ref}}$

- **Repeat-prefill** (from KVzip): prompt `"{C} Repeat the previous context. {C}"`, prefill, extract query activations from the reconstruction segment.
- **Context-prefill** (H2O-style): just prefill $\mathcal{C}$ alone - cheaper but slightly worse.
- **Self-study** (from Cartridges): generate synthetic conversations conditioned on $\mathcal{C}$ using 4 fixed prompts (`summarize`, `structure_json`, `aggregate`, `3-question`) + model-generated starters. Run two LM "instances": A makes starters $a_i$, B answers $b_i \sim \text{LM}(\cdot \mid \mathcal{C}, a_i)$. Extract queries during B's prefill + decoding.
- **Random vectors**: $q \sim \mathcal{N}(0, I_d)$ with q-norm scaling. Works but slightly worse.
- **Best**: self-study; **fastest decent**: repeat-prefill / context-prefill (within 1-2 pts of self-study).
- Cap at ~50k queries per KV-head per chunk; **GQA** is favorable (many query heads share each KV head $\Rightarrow$ many reference vectors).

### 3.2 On-Policy Queries

- Layerwise compaction shifts later residual streams.
- **Fix**: compact layers sequentially. For layer $\ell$, extract $Q_{\text{ref}}^\ell$ by running model with layers $< \ell$ already compacted.
- Small but consistent gain.

### 3.3 Fitting $\beta$ (mass matching, NNLS)

Let $m_i = \sum_k \exp(q_i K_k^\top)$ (target mass). Set $w_j = \exp(\beta_j) > 0$:
$$m_i \approx \sum_{j=1}^t w_j \exp(q_i (C_k)_j^\top)$$

$$\min_{w \ge 0} \|Aw - m\|_2^2, \qquad A_{ij} = \exp(q_i (C_k)_j^\top)$$

- Solved with **nonnegative least squares (NNLS)** via projected gradient.
- $\beta_j = \log w_j$, clamp at small positive if needed.
- Intuition: $w_j$ = how many original keys' worth of mass $(C_k)_j$ accounts for.

### 3.4 Fitting $C_v$ (output matching, OLS)

After freezing $(C_k, \beta)$: for each reference $q_i$ compute
$$y_i = \frac{\exp(q_i K^\top) V}{\sum_j \exp(q_i K_j^\top)}, \quad x_i = \frac{\exp(q_i C_k^\top + \beta)}{\sum_j \exp(q_i (C_k)_j^\top + \beta_j)}$$

Then $C_v^\star = \arg\min_{C_v} \|X C_v - Y\|_F^2 = (X^\top X)^{-1} X^\top Y$.

- Standard OLS; `torch.linalg.lstsq` (gels/QR) used in practice.
- Numerics: solve in FP32, cast to BF16 for storage.

### 3.5 Selecting $C_k$ (two flavors)

Both restrict $C_k = K_{S,:}$ for some index set $S$, $|S|=t$.

**(a) Highest Attention Keys (cheap, H2O-like)**
- For each $q_i$: $a_i = \text{softmax}(q_i K^\top)$.
- Aggregate across queries via **RMS**: $s_j = \sqrt{\frac{1}{n}\sum_i a_{i,j}^2}$ (RMS more robust than mean/max - though no clear winner across methods).
- Top-$t$ scores $\Rightarrow$ $S$. Fit $\beta$ via NNLS, $C_v$ via OLS.

**(b) OMP Keys (better, Orthogonal Matching Pursuit)**
- Directly maximize mass-matching: greedily pick $S$ to fit Eq. (2).
- Mass feature matrix $\Phi_{ij} = \exp(q_i K_j^\top / \sqrt{d})$, target $m_i = \sum_j \Phi_{ij}$.
- Greedy step: add column maximizing reduction in $\|\Phi_{:,S} w - m\|^2$, then refit $w$ via NNLS.
- Yields $\beta = \log w$ directly; then $C_v$ via OLS.

```text
Algorithm 1: OMP Key Selection
1. Phi_ij = exp(q_i K_j^T / sqrt(d))
2. m_i = sum_j Phi_ij                       # target mass
3. r = m;  S = {}
4. for k = 1..t:
5.   j* = argmax_{j not in S} (r^T Phi_:,j) # greedy correlation
6.   S = S U {j*}
7.   w = argmin_{w >= 0} ||Phi_:,S w - m||^2  # NNLS refit
8.   r = m - Phi_:,S w                       # update residual
9. return S, w
```

- **OMP-fast**: hyperparameters $k=4$ keys/step, refit interval $\tau=2$ $\Rightarrow$ 4-8x speedup with little loss (Algorithm 2).

### 3.6 Nonuniform Per-Head Compaction (important ablation winner)

- Different heads have different sensitivity to capacity (retrieval vs streaming heads, etc.).
- **Head sensitivity curves**: hold all but one head at baseline ratio $0.05$, sweep that head's ratio, measure $\Delta$log-perplexity. Stable across articles & target ratios.
- Use stability to **precompute per-model head budget once** via greedy swap (Algorithm 4):
    - Start uniform; iteratively swap budget units between heads to minimize total predicted loss.
    - Assumes near-separability across heads.
- Tends to give **more budget to later layers** (opposite of PyramidKV's pyramid heuristic).
- Implementation note: variable-length packing (varlen, FlashAttention 2024) avoids padding waste from per-head sizing.

### 3.7 Chunked Compaction (for long inputs)

- Split context into $N$ contiguous chunks; compact each with own $(C_k, \beta, C_v)$; concat.
- **KV-based chunking** (default): prefill full context once, slice KV per chunk, compact, concat $\Rightarrow$ preserves cross-chunk attention via accurate query vectors.
- **Text-based chunking**: prefill each chunk in isolation, apply RoPE phase shift $\Delta = p_{\text{global}} - p_{\text{local}}$ on compact keys to merge. Cheaper but approximate.

---

## 4. Method Variants (Pareto front)

| Variant | $Q_{\text{ref}}$ | $C_k$ selection | $\beta$ fit | $C_v$ fit |
|---|---|---|---|---|
| **AM-OMP** | self-study + repeat-prefill (on-policy) | OMP (greedy, NNLS refit each step) | $\log w$ from OMP | OLS |
| **AM-OMP-fast** | same | OMP, $k=4$, refit every 2 | $\log w$ | OLS |
| **AM-HighestAttnKeys** | self-study + repeat-prefill | top-RMS attention | NNLS | OLS |
| **AM-HighestAttnKeys-fast** | repeat-prefill only | top-RMS attention | NNLS | OLS |

Speed (Gemma-3-12B, 60k LongHealth, single H100):

| Stage | Method | Time (s) |
|---|---|---|
| Query generation | context-prefill | 7 |
| Query generation | repeat-prefill | 8 |
| Query generation | self-study | 139 |
| Key selection | Highest attention | 3 |
| Key selection | OMP | 565 |
| Key selection | OMP-fast ($k=4, \tau=2$) | 104 |
| $\beta$ fit | NNLS | 2.2 |
| $C_v$ fit | Least squares | 1.8 |

- **AM-OMP** entire pipeline: minutes; Cartridges: **hours**.
- Query generation dominates; can be amortized / further optimized.

---

## 5. Experiments

### Models
- Qwen3-4B (Qwen3-4B-Instruct-2507 for long-context)
- Llama-3.1-8B-Instruct
- Gemma-3-12B-it (5:1 sliding-window:global ratio - only compact global layers)

### Datasets
- **QuALITY**: 5-7k token articles, 15-20 MCQ per context, first 50 articles (894 q's).
- **LongHealth**: ~60k tokens (5 patient records), 100 MCQ per context, 4 contexts. Chunked compaction (5 chunks).

### Baselines
- **Cartridges** (full end-to-end gradient training, hours per context).
- **Eviction / pruning**: H2O+ (one-shot variant), SnapKV (mock-question agnostic), PyramidKV (linearly decreasing), KVzip (global top-k, gives natural nonuniform allocation).
- **DuoAttention** (Llama only): retrieval+streaming head split. Performed poorly at high ratios, omitted from main fig.
- **Summarization**: 5 prompt variants (`summarize`, `summarize_indepth`, `summarize_keypoints_questions`, `summarize_concise`, `summarize_very_concise`).

### Main Results

- Figure 1 (Qwen3-4B, 50x compaction): **AM-OMP matches Cartridges quality (~0.65 acc)** in **<1 minute** (Cartridges: ~hours). AM-HighestAttnKeys-fast trades slight quality for ~10s speed.
- Figure 3: across 3 models x 2 datasets x sweep of ratios ($0.01-0.20$ compacted size), **AM consistently dominates** heuristic baselines, especially in **high-compaction (20x-100x)** regime.
- LongHealth (info-dense) is harder: summarization collapses to no-context baseline; KVzip occasionally matches AM (its global top-k gives implicit nonuniform allocation).

### Sliding-Window Models
- Gemma3-12B (sliding:global = 5:1) compacted only on **global** layers. Achieves similar compaction with slightly more conservative ratios.
- Supports hypothesis: many heads naturally specialize as local/streaming; a few globally-attending heads carry long-range info $\Rightarrow$ nonuniform compaction is exactly the right primitive.

### Summarization + AM Combo (cute extension)

| Method | Cache Size (%) | Acc (%) |
|---|---|---|
| Full Context | 100 | 71.5 |
| Summarize | 4.80 | 55.2 |
| Summarize x (0.2x AM-OMP) | 0.92 | 55.7 |
| Summarize x (0.1x AM-OMP) | 0.46 | 55.0 |
| Summarize x (0.05x AM-OMP) | 0.21 | 49.2 |

- Applying AM **on top of a summary** $\Rightarrow$ ~200x total compaction with quality comparable to summary alone (6340 $\to$ 31 effective tokens).

### Ablations (leave-one-out on AM-OMP)

Most $\to$ least important (by drop in log-perplexity):
1. **Nonuniform head budgets** - biggest hit when removed (uniform).
2. Self-study queries (vs repeat-prefill only).
3. On-policy query extraction.
4. Learned $C_v$ values (refit via OLS vs direct copy).
5. $\beta$ biases (zeroing $\beta$ after OMP key selection still works decently - keys + values do most of the work).

**Aggregation function** for highest-attn-keys: no consistent winner across mean / RMS / max; RMS is the safest balance.

**Reference query** ranking (by log-perplexity):
- Best: self-study; ~tied: repeat-prefill, context-prefill, ss-plus-repeat.
- Worse: random-vectors (with or without q-norm scaling).

### Reconstruction Loss vs Downstream Accuracy

- Compaction reconstruction loss ~ correlates with QA accuracy within a method family.
- **Outliers**: Cartridges has lower loss for same accuracy (end-to-end target); summarization has *higher* loss for similar accuracy (discards detail, preserves answer-relevant info).
- $\Rightarrow$ reconstruction is a useful within-family proxy, not a universal predictor.

### Online Compaction (proof of concept on AIME 2025)

| Phys Len | Eff Len | AIME (/30) | Notes |
|---|---|---|---|
| 2048 | 2048 | 1.25 | baseline (truncated) |
| 4096 | 4096 | 7.75 | |
| 8192 | 8192 | 13 | |
| 2048 | 4096 | 8 | $\le$2 compactions |
| 2048 | 8192 | 13 | $\le$6 compactions |

- Compacting **mid-trajectory** during reasoning (every time cache hits Phys Len, compact all but last 20 tokens) lets a 2048-physical-cache model produce as well as 8192-full-cache after $\le$6 compactions.
- Decouples reasoning depth from physical context limit. Strong promise for agentic / long-CoT settings.

---

## 6. Why It Works (Conceptual Highlights)

- **Mathematical primitive**: the mixture identity for attention over concatenated blocks. By preserving both *output* and *mass* of a block, you preserve its contribution to attention under *arbitrary* future content. This is what makes "compatibility with future tokens" achievable without knowing them.
- **Decomposition** of joint nonconvex optimization into:
    - Discrete subset selection $\to$ greedy (OMP).
    - Positive linear reweighting $\to$ NNLS.
    - Linear reconstruction $\to$ OLS.
  Each is well-studied and closed-form.
- **Per-token attention bias $\beta$** as an "unlock": removes the structural underweighting issue of pure subset-eviction.
- **Stable head sensitivity** $\Rightarrow$ amortizable, per-model precomputed budget allocation. Avoids the per-context overhead of online adaptive allocation.
- **GQA is your friend** for query generation: each token yields multiple query vectors per KV head, so reference query supply is abundant cheaply.

---

## 7. Relation to Adjacent Work

| Work | Mechanism | Relation |
|---|---|---|
| **H2O** (Zhang 2023) | Online eviction via attention scores | One-shot analogue (H2O+) used as baseline. AM is its closed-form-fit successor. |
| **SnapKV** (Li 2024) | Question-aware highest-attn keys | Adapted to question-agnostic via mock-question. |
| **PyramidKV** (Cai 2025) | Linearly decreasing per-layer budget | AM's learned budget gives *more* budget to *later* layers - opposite shape. |
| **KVzip** (Kim 2025b) | Repeat-prefill + global top-k | Donates the repeat-prefill query trick; AM goes further with mass biases + value refit. |
| **Cartridges** (Eyuboglu 2025) | End-to-end gradient training of compact latent KV via self-study | AM matches it in seconds-minutes vs hours; Cartridges still wins at extreme (100x+) on hard data. |
| **Lexico** (Kim 2025a) | Universal per-layer dictionary + OMP at inference | OMP used here too, but for $\ell_2$ reconstruction of KV vectors not attention-behavior matching. |
| **DuoAttention** (Xiao 2025) | Binary retrieval vs streaming heads | AM learns *continuous* per-head budgets via sensitivity curves. |
| **T5 / ALiBi** | Per-token bias as positional encoding | Same construct, used here for mass correction not position. |
| **StreamingLLM / sinks** | Attention sinks | AM keeps BOS / template tokens fixed (not compacted) - implicit sink handling. |
| **FlashAttention / Cascade Inference** | Block-mixture identity for parallel softmax | AM repurposes the same identity for *compatibility*: ensure compacted block correctly mixes with future blocks. |
| **Gist / soft prompt compression** (Mu 2023, Chevalier 2023) | Train model to generate compact prefix tokens | AM is post-hoc, no training; works on any pretrained model. |

---

## 8. Limitations / Open Questions

- **Cartridges still wins** at extreme ratios (100x+) on info-dense data (LongHealth). Gradient descent searches a wider space of compact representations (not restricted to subset of $K$).
- $C_k$ is constrained to be a **subset of original keys** - no closed-form for free-form $C_k$. Direct optimization of compact keys is open.
- Self-study query generation dominates runtime; further speedups could come from cheaper query proxies.
- Per-head sensitivity assumes near-separability - might break for very few global heads / hybrid architectures.
- Online / multi-turn compaction explored only as proof-of-concept (AIME). Selective recent-token compaction, principled scheduling, and integration with prefix caches (RadixAttention, paged KV, varlen) are systems-level future work.
- No quantization combo studied.

---

## 9. Novel Bits Worth Remembering

- **Attention mass** as a first-class quantity to preserve (not just attention output). Falls out of the FlashAttention block-mixture identity.
- **Per-token attention bias $\beta$** repurposed from positional encoding to *mass repair* after subset eviction. Cheap memory, no kernel changes.
- **Decomposition**: subset selection (OMP) $\to$ NNLS for $\beta$ $\to$ OLS for $C_v$. All closed-form. No gradient descent at compaction time.
- **Logical vs physical cache length**: separate `seen positions` (drives RoPE) from `stored entries` $\Rightarrow$ clean primitive.
- **Sensitivity-curve based per-model head budget**: amortize one optimization across all contexts/ratios.
- **Reconstruction loss != accuracy in general**: summarization beats its own loss; Cartridges over-achieves on loss. Cautionary tale for compaction-loss-only evaluation.
- "Compaction" terminology borrowed from systems / agentic coding - signals a primitive to be invoked repeatedly during long-horizon interactions.

---

## Code Walkthrough

Repo: `raw_data/papers/Fast_KV_Compaction/compaction/`. Roughly four layers from outside in:
- **`evaluation/`** drives experiments (loads model/data, calls compaction, runs QA).
- **`compaction/compaction_methods/`** = "full-cache" wrappers (per-layer-head loop, chunking, global key selection, summarize-then-compact).
- **`compaction/algorithms/`** = per-(layer, head) solvers (OMP, HighestAttentionKeys, etc.). All share the `CompactionAlgorithm` ABC.
- **`models/`** = vendored modeling files (Qwen3, Llama, Gemma3) lightly patched + `CompactedPrefixCache` that splices $(C_k, \beta, C_v)$ into HF attention via the `attention_mask`.

### Architecture at a glance

```
evaluation/run_qa_evaluation.py
  -> get_compaction_method(name, kwargs)              # compaction_methods/registry.py
     -> (optional) ChunkedCompaction wrapper          # chunked.py
     -> PerLayerHeadOnPolicyCompaction                # per_layer_head_on_policy.py
        (extends PerLayerHeadCompaction)              # per_layer_head.py
        -> QueryGenerator(self_study, repeat, ...)    # query_generation/*
        -> per (layer, head): OMPCompaction.compute_compacted_cache(K, V, queries, t)
           -> _select_keys_omp + _nnls_pg + _compute_C2
     -> returns (C1, beta, C2) per layer; wrapped in CompactedPrefixCache
        which the patched models read in attention forward
```

### Core abstraction: `CompactionAlgorithm`

`compaction/algorithms/base.py:14` defines the per-head ABC:

```python
def compute_compacted_cache(self, K, V, queries, t, attention_bias=None):
    """K:(T,d), V:(T,d), queries:(n,d) -> (C1:(t,d), beta:(t,), C2:(t,d), indices)"""
```

- All shapes are flat `(T, d)` — per (layer, head, batch=1). The wrapper slices outer dims.
- `attention_bias` parameter is **not** $\beta$. It's an additive bias on the *original* cache scores, used when a chunk's prefix is already compacted (the bias carries the previous layer's $\beta$ forward when extracting on-policy queries).

Two reusable primitives on the base class:
- **`_compute_C2`** (`base.py:61`): OLS for $C_v$ given fixed $(C_k, \beta)$.
- **`_nnls_pg`** (`base.py:471`): box-constrained NNLS, used both inside OMP and by HighestAttentionKeys.

---

### OMP key selection (`algorithms/omp.py`)

`SimpleOMPCompaction.select_keys` (`omp.py:25`) is the unhyperparametrized reference Algorithm 1. The production version is `OMPCompaction._select_keys_omp` (`omp.py:478`). The pre-loop setup builds the **mass feature matrix** $\Phi$ in fp32 (with stability shift):

```python
# omp.py:520-537 — Phi_ij = exp(q_i K_j^T / sqrt(d)), target m_i = sum_j Phi_ij
inv_sqrt_d = (1.0 / d) ** 0.5
scores_raw = queries @ K.T                              # (n, T) original dtype
scores32   = scores_raw.to(torch.float32) * inv_sqrt_d  # (n, T) fp32
max_scores = scores32.max(dim=1, keepdim=True)[0]       # for numerical stability
exp_scores = torch.exp(scores32 - max_scores)           # (n, T) = Phi (up to const)
target     = exp_scores.sum(dim=1)                      # (n,)
```

- **Precision discipline**: matmul in original dtype (kernel does fp32 accum), softmax/scale in fp32, downcast results back to model dtype (typically bf16) for storage. Same recipe in `_compute_C2`.
- **Stability shift** $\max$ over $T$ is fine: the constant cancels in the NNLS objective because $\beta = \log w$ absorbs any global scale.

**Greedy loop** (`omp.py:551-627`):

```python
while i < t:
    residual = target - current                          # (n,)
    corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)  # (T,)
    corr[mask_selected | mask_excluded] = -float('inf')
    # k_choice = 1 in vanilla OMP; >1 = "fast" variant (Algorithm 2)
    top_k_values, top_k_indices = torch.topk(corr_abs, k_select, largest=True)
    # ... append indices ...
    M = exp_scores[:, selected_indices_tensor[:i]]       # (n, i)
    B, solved = self._solve_nnls(M, target, prev_B, iteration,
                                 nnls_interval=current_nnls_interval, ...)
    beta32[:i] = torch.log(B)
    current = M @ B                                      # update residual implicitly
    iteration += 1
```

Key implementation details:
- **Residual is in *mass* space**, not output space. Correlation is just $r^\top \Phi_{:,j}$ — a single matmul over $T$ keys, fully vectorized.
- **`k_choice` > 1**: pick top-$k$ by correlation in one `torch.topk` instead of $k$ argmaxes. This is the "AM-OMP-fast" knob (Algorithm 2 in paper). Best config in `configs/algorithms/best.py:81-90` uses `k_choice=4, nnls_interval=2`.
- **`nnls_interval > 1`**: skip the NNLS refit on intermediate iterations and extend `prev_B` with lower-bound entries for new columns (`_solve_nnls` at `omp.py:412-476`). This is the second axis of "fast".
- **`progressive_schedule`** (`omp.py:120-124`): `DEFAULT_PROGRESSIVE_SCHEDULE = [(300, 1, 1), (1500, 2, 2), (None, 4, 2)]` — vanilla OMP for first 300 keys, then progressively coarser. The "best" config for $t \ge 300$.
- **`drop_key_beta_cutoff`** (`omp.py:629-702`): refinement phase. After reaching $t$ keys, if any $\beta_j < -7$ (default cutoff $e^{-7}\approx 9 \times 10^{-4}$), drop them and re-select. Loop bounded to 3 entries. Used in all "best" configs (`-7` cutoff in `best.py`).
- **`cached_selection_order`** (`omp.py:340-410`): once you have the full greedy order for max-$t$, evaluating any smaller $t' < t$ just takes the prefix and re-runs NNLS. Used by the head-influence sensitivity sweep to amortize cost over many ratios.
- **`zerobeta`** ablation: zero out $\beta$ after key selection (paper ablation 5).
- **`nnls_upper_bound = exp(7)`** in best config (`best.py:43,75`): caps mass weight to $\approx 1097$ to prevent a single key absorbing too much (numerical sanity).

---

### NNLS for $\beta$ (`algorithms/base.py:471`)

```python
# base.py:498-543 — solve min_B ||M B - y||^2, B >= [lb, ub]
B = torch.linalg.lstsq(M, y.unsqueeze(1), driver='gels').solution.squeeze(1)
# ... fallback to cholesky w/ small ridge if lstsq has NaNs ...
B = B.clamp_min_(min_val)
if upper_bound is not None:
    B = B.clamp_max_(upper_bound)
if iters == 0:
    return B
# else: projected gradient descent
sigma = power_iteration(M)          # ~||M||_2
eta   = 1.0 / sigma**2              # 1/L step size
for _ in range(iters):
    B = (B - eta * M.T @ (M @ B - y)).clamp_min_(lb)
    if upper_bound is not None: B = B.clamp_max_(upper_bound)
```

- **`iters=0`** (default for OMP): just unconstrained lstsq + clamp. Surprisingly works fine because the active constraint is almost always the lower bound (most $w_j$ are positive anyway after greedy selection).
- **`iters > 0`** (e.g. `nnls_iters=2` for HighestAttnKeys): proper projected gradient. Lipschitz constant from 3-step power iteration; cheap.
- **Bounds in log space**: `best.py` uses `nnls_lower_bound=exp(-3), nnls_upper_bound=exp(3)` for HighestAttnKeys $\Rightarrow$ $\beta \in [-3, 3]$.
- **`beta = log(B)`** — base mathematical step. If `B` was clamped to 1e-12, $\beta \approx -27.6$, effectively masking that key (close to $-\infty$).

---

### OLS for $C_v$ (`algorithms/base.py:61`)

After freezing $(C_k, \beta)$, form normalized softmax matrices and solve:

```python
# base.py:113-140 (simplified)
sK32 = (queries @ K.T).to(fp32) * inv_sqrt_d  # +attention_bias
attn_K = exp(sK32 - max) / sum_exp_K          # (n, T)
Y = attn_K @ V.to(fp32)                       # (n, d) fp32

sC32 = (queries @ C1.T).to(fp32) * inv_sqrt_d + beta.to(fp32)
X = exp(sC32 - max) / sum_exp_C               # (n, t)
# C2 = (X^T X + lam I)^-1 X^T Y, solved via lstsq with gels driver
C2_32 = torch.linalg.lstsq(X, Y, driver='gels').solution
return C2_32.to(dtype_param)                  # back to bf16
```

- **Ridge scaling** (`base.py:148-161`): three modes — `spectral` (default; $\lambda \cdot \sigma_{\max}^2(X)$), `frobenius` ($\lambda \cdot \|X\|_F^2 / t$), `fixed`. Default `ridge_lambda=0` is fine; fallback to $10^{-6}$ + Cholesky on NaN.
- **Underdetermined dispatch** (`base.py:184-196`, `:219-228`): if $n < t$ uses $X^\top (XX^\top + \lambda I)^{-1} Y$; else $(X^\top X + \lambda I)^{-1} X^\top Y$. Sometimes called the "dual" trick.
- **`_direct_C2`** (`base.py:348-407`): ablation that copies $V[\text{indices}]$ directly instead of solving — paper's "no $C_v$ refit" baseline (ablation 4). Used in KVzip baseline config (`global_highest_attn_keys_max_nobeta_direct`).
- **`_compute_C2_on_policy`** (`base.py:242-346`): docstring says **UNUSED**; on-policy compaction reuses `_compute_C2` with on-policy queries throughout instead of mixing original/generated. Interesting design choice — keeps the closed-form clean.

---

### HighestAttentionKeys (`algorithms/highest_attention_keys.py`)

The simpler "AM-HighestAttnKeys" variant. Key snippet (`highest_attention_keys.py:182-190`):

```python
if self.score_method == 'rms':
    key_scores = torch.sqrt((attention_weights ** 2).mean(dim=0))  # (T,)
elif self.score_method == 'max':
    key_scores = attention_weights.max(dim=0)[0]
else:  # 'mean'
    key_scores = attention_weights.mean(dim=0)
_, top_indices = torch.topk(key_scores, t, largest=True)
```

- **`attention_weights = softmax(qK^T/sqrt(d))`** — proper softmax, unlike OMP which works in unnormalized exp space.
- After selection: same `_nnls_pg` for $\beta$ then `_compute_C2` for $C_v$.
- **Pooling option** (`pooling='avgpool'|'maxpool', kernel_size=7`) — adopted from **SnapKV**'s observation that retained keys tend to cluster; smooths the score before top-$k$. Best config does **not** use pooling, but it's there.
- The H2O / KVzip baselines reuse this class with `beta_method='zero', c2_method='direct'` (config `highest_attn_keys_mean_nobeta_direct`).

---

### Reference query generation (`compaction/query_generation/`)

`QueryConfig` lists `MethodConfig`s (`self_study | random_vectors | cache_keys | context_prefill`); `QueryGenerator.generate_queries` dispatches each, concats over the query axis, and **converts query-head queries to KV-head queries** via:

```python
# query_generator.py:185-198 — GQA: concat heads_per_kv_head queries together
queries_reshaped = queries_attention_heads.view(
    num_layers, num_kv_heads, heads_per_kv_head, n_queries_per_head, head_dim
)
queries_kv = queries_reshaped.reshape(
    num_layers, num_kv_heads, heads_per_kv_head * n_queries_per_head, head_dim
)
```

$\Rightarrow$ GQA models get **$\text{heads\_per\_kv\_head} \times$** more queries per KV head for free. Qwen3-4B has 32 Q / 8 KV $=4\times$. This is *why* the paper says "GQA is your friend".

**Conversation specs** (`query_generation/conversation_specs.py`):
- 4 fixed in paper map to keys: `summarize`, `structure_json`, `aggregate`, `3_question`.
- Plus `question` (single thinking question), `repeat` (KVzip's repeat-prefill, uses `prefill_with_article=True` so vLLM isn't even called — just teacher-forces the article text), `summarize_backward`, `structure_yaml`, `rephrase`.
- "Model A" generates a starter via vLLM (`SelfStudyQueryGenerator._generate_starters`); "Model B" prefills `[context, starter]` and we grab `q_proj` activations from both the prefill segment and the generated answer.
- **`extract_after_thinking_then_split`** — gnarly regex to split multi-question outputs across `\n\n` / `---` / numbered items. Comment in source admits: `# TODO: switch to json outputs`.

**Random vectors** (`random_vectors.py`): $q \sim \mathcal{N}(0, I)$ optionally scaled element-wise by `q_norm.weight` (Qwen3 / Gemma3 apply LayerNorm to Q before attention; without `scale_by_qnorm` the random qs don't match real-q scale).

**Context-prefill** (`context_prefill.py`): just sample positions from the context-only prefill.

---

### Chunked compaction (`compaction/compaction_methods/chunked.py`)

Two variants:

**1. KV-based chunking** (default, `use_kv_based=True`, `_compact_kv_cache_kv_based` at `chunked.py:821`):
- Prefill full context **once** ($\Rightarrow$ needs to fit in memory).
- For each chunk, slice out the chunk's KV from the full cache (already has correct RoPE positions).
- Build a `CompactedPrefixCache` for **query generation** with `[prefix + chunk + suffix + sliding_layers]` so self-study queries get the right context at the right RoPE positions (`chunked.py:963-997`).
- Compact the chunk; concat outputs. **No RoPE correction needed** since slices preserved positions.

**2. Text-based chunking** (`_compact_kv_cache_text_based` at `chunked.py:372`):
- Memory-efficient: prefill **each chunk independently** with positions starting at 0.
- After compaction, apply **RoPE phase shift** to move chunk-local keys to global positions:

```python
# chunked.py:115-186 — compute cos(p' - p), sin(p' - p)
position_diff = target_positions - current_positions
dummy = torch.zeros(1, 1, rotary_emb.inv_freq.shape[0] * 2, device=device, dtype=dtype)
cos_diff, sin_diff = rotary_emb(dummy, position_diff.to(device))

# chunked.py:87-112 — apply: K_new = K * cos_diff + rotate_half(K) * sin_diff
return (cache * cos) + (rotate_half(cache) * sin)
```

The math note in the docstring: two RoPE rotations compose into one rotation by $(p' - p)$, identical to applying `apply_rotary_pos_emb` once with that delta.

**Sliding-window handling** (Gemma3 / Qwen3 hybrid): only **global layers** get compacted. Sliding layers are kept verbatim from the **last** chunk's prefill (because they only see the recent window anyway). For the last chunk, the **outer suffix** (chat-template `<|im_end|>...`) is **included** in that chunk's prefill so sliding layers see correct recent context (`chunked.py:529-535`). Subtle but important.

**Sliding-layer RoPE shift** uses `rotary_emb_local` (different `rope_theta` in Gemma3) — also subtle (`chunked.py:677-686`):
```python
cos_diff, sin_diff = compute_rope_correction(
    model, current_positions, target_positions, device, dtype,
    use_local_rope=True  # Sliding layers use local RoPE
)
```

---

### Per-layer-head loop + per-head budgets (`per_layer_head.py`)

`PerLayerHeadCompaction.compact_kv_cache` is the workhorse for "AM applied to every (layer, head)". Flow:

1. Generate queries once with `QueryGenerator` (`per_layer_head.py:239-243`).
2. (Optional) load **per-head budgets** from a JSON (e.g. `head_budgets/Qwen3-4B-Instruct-2507/optimized_agnostic.json`) and compute `head_target_size = int(proportion * actual_target_size * total_heads)` (`per_layer_head.py:313-318`).
3. Loop layers $\times$ heads, call `algorithm.compute_compacted_cache(K, V, queries_head, head_target_size, ...)` (sequential path, `per_layer_head.py:482-526`).
4. **Pad heads within each layer to the per-layer max** with $\beta = -\infty$ for padded slots (`per_layer_head.py:634-654`):

```python
beta_padding = K.new_full((num_padding,), float('-inf'))
# These padded positions get e^{-inf}=0 attention weight at inference time
```

That's the implementation of **variable per-head budgets without varlen kernels** — just pad with $\beta = -\infty$. Cheap and FlashAttention-compatible since standard mask handling already supports $-\infty$.

5. Batched paths (`_compact_per_layer_batched`, `_compact_batched`) exist for `OMPCompaction` / `OptimJointCompaction` only (no per-head budgets in those paths) and stack heads/layers along a batch dim.

**On-policy variant** (`per_layer_head_on_policy.py`): same loop, but for layer $\ell > 0$ rebuilds the queries by re-running the model with **layers $< \ell$ already compacted** wrapped in a `CompactedPrefixCache` (`per_layer_head_on_policy.py:332-355`). Only self-study queries get on-policy treatment; context-prefill / repeat / random use original queries (cheap).

---

### Head budget optimization (`head_budget_optimization/`)

Three pieces:
- **`influence.py:HeadInfluenceComputer`** — for each (layer, head), sweep ratios in $[0, \text{max}]$ (defaults to 5 eval points), compact just that head, recompute log-perplexity on a held-out continuation, record $(\text{ratio}, \Delta\log\text{ppl})$. OMP's `cached_selection_order` is reused across ratios for speed.
- **`solver.py:HeadBudgetSolver`** — given per-head curves, solve for budget allocation. Three methods (`solve_greedy`, `solve_swap`, `solve_annealing`, `solve_ratio_agnostic_swap`). The paper's Algorithm 4 corresponds to **`solve_swap`** (`solver.py:313-465`):

```python
# Initialize at uniform allocation, then swap step_size budget between heads
best_recipient = argmax_h interpolate_marginal_benefit(h, ratio_h, +step)
best_donor    = argmin_h interpolate_marginal_cost(h, ratio_h, -step)
if benefit - cost > 0: transfer step from donor -> recipient
```

  - "Start from uniform" avoids the U-shaped-curve local minimum that `solve_greedy` (start from zero) gets stuck in.
  - **`solve_ratio_agnostic_swap`** (`solver.py:1017-1159`) optimizes a **proportion vector** that works across multiple target ratios simultaneously — this is what `optimized_agnostic.json` (used in `qwen-longhealth5.sh:22`) comes from.

- **`run.py`** — driver script: build curves, aggregate across articles, solve, save proportions.

Output files in `head_budget_optimization/head_budgets/{model}/optimized_t{ratio}.json` are dicts `{"L0H0": proportion, ...}` summing to 1.0. At runtime: `PerLayerHeadCompaction(precomputed_budget_path=...)` loads and scales by `total_heads * target_size`.

---

### Splicing compacted KVs into HF attention (`models/cache.py`, `models/{qwen3,llama,gemma3}/`)

The clever bit: **don't touch the attention kernel; instead add $\beta$ to the attention mask**. `CompactedPrefixLayer` (`models/cache.py:9-103`) stores `(keys, beta, values)` per layer and acts like a `DynamicLayer` — new tokens just `torch.cat` into `self.keys / self.values`.

In the patched attention forward (`models/qwen3/modeling_qwen3.py:218-271`):

```python
# qwen3 model: after standard q/k/v projection + RoPE + KV cache update
if isinstance(past_key_values, CompactedPrefixCache):
    beta_kv = past_key_values.beta_for_layer(self.layer_idx)
    beta_heads = repeat_kv_for_beta(beta_kv, self.num_key_value_groups)  # GQA expand

    # Take last kv_length cols of the (possibly oversized) mask, then add beta:
    modified_attention_mask = attention_mask[:, :, :, -kv_length:].clone()
    modified_attention_mask[:, :, :, :prefix_length] += beta_heads.unsqueeze(2)

attn_output, _ = attention_interface(
    self, q, k, v, modified_attention_mask, ..., scaling=self.scaling, ...)
```

- **$\beta$ is just an additive bias on attention scores**, applied only to the **first `prefix_length` (compacted) positions**. New tokens past the prefix get $\beta = 0$ (no bias).
- **GQA expand** (`repeat_kv_for_beta`, `modeling_qwen3.py:130-140`): $\beta$ stored at KV-head granularity, expanded to all Q heads sharing that KV head. TODO comment notes SDPA's `enable_gqa=True` doesn't yet broadcast custom masks, otherwise this expand could be avoided.
- **FlashAttention compatibility**: $\beta$ goes through the same `attn_mask` path SDPA / FA already support $\Rightarrow$ **no kernel changes**. Quote from the source: "supported by SDPA / FlexAttention naturally" matches paper claim.
- **Variable-length per head**: the mask is **per-head** $(B, A, Q, T)$ once expanded. Padded slots have `beta = -inf` (set in `per_layer_head.py:653`) $\Rightarrow$ they contribute zero attention weight. No `varlen` kernel needed for the chunked-compaction case.

**RoPE positions** (`cache.py:120-211`):

```python
# For nonuniform caches, all layers use the same rope_base because cache_position
# is based on the maximum cache length. This ensures all layers get the correct
# RoPE embeddings for the current absolute position in the sequence.
self._rope_base = int(original_seq_len) - int(max_compacted_len)
```

I.e., `cache_position` starts at `max_compacted_len`, but RoPE adds `original_seq_len - max_compacted_len` $\Rightarrow$ new tokens get RoPE phase = `original_seq_len + offset_in_decode`. This is the implementation of the paper's "logical vs physical length" decoupling.

Per-layer `attention_mask` slicing handles the case where different layers have different compacted sizes (heads with smaller budgets get more padding) — `modeling_qwen3.py:243` slices `attention_mask[:, :, :, -kv_length:]` per layer.

**Sliding-window layers** use `DynamicSlidingWindowLayer` from HF (`cache.py:166-177`) so global vs local layers coexist cleanly in the same cache object.

---

### Fast variants in code

| Paper variant | Config name (`evaluation/configs/algorithms/best.py`) | Code knob |
|---|---|---|
| **AM-OMP** | `omp_nnls0_-inf_7_drop-7_lsq` | OMP with `k_choice=1, nnls_interval=1`, drop cutoff $-7$ |
| **AM-OMP-fast** | `omp_nnls0_-inf_7_drop-7_lsq_k4int2` | `k_choice=4, nnls_interval=2` (paper Alg 2: $k=4, \tau=2$) |
| **AM-OMP best** | `omp_nnls0_-inf_7_drop-7_lsq_progressive` | `progressive_schedule = [(300,1,1),(1500,2,2),(None,4,2)]` |
| **AM-HighestAttnKeys** | `highest_attn_keys_rms_nnls2_-3_3_lsq` | rms agg, NNLS w/ 2 PGD iters, $\beta \in [-3, 3]$ |
| **AM-HighestAttnKeys-fast** | same, but `query_config=repeat` (only repeat-prefill) | skips self-study entirely |

Plus `_on-policy` suffix for any of them: routes through `PerLayerHeadOnPolicyCompaction`.

---

### Evaluation harnesses

**QA** (`evaluation/run_qa_evaluation.py` + `qa_evaluator.py`): the main entry. Pipeline per (method × ratio × article):
1. Format article with chat template; prefill via `models/generate.py:chunked_prefill` (extracts `past_key_values`).
2. Call `method.compact_kv_cache(past_key_values, target_size, indices, query_config, model, tokenizer, formatted_context, ...)`.
3. Wrap result in `CompactedPrefixCache` and run MCQ questions through `generate_with_compacted_cache`.
4. Score correct / total; aggregate.

Datasets in `evaluation/datasets.py`: `quality` (QuALITY MCQ), `longhealth`, `longhealth5` (5-chunk), plus `qasper`, `ruler`, `longbenchv2` in `scripts/v2/`. Each dataset has dataset-specific MCQ formatting and chunking.

**Reasoning / AIME** (`evaluation/run_reasoning_evaluation.py`): the **online compaction** proof of concept. Loop:
```python
while not done and num_compactions < max_compactions:
    generate up to max_seq_len tokens
    if not done:
        compact (all - protected_tokens at end)  # uses same compaction methods
        num_compactions += 1
if not done: force "</think>Final Answer: " and decode MAX_ANSWER_TOKENS=16 more
```

This is the source of the Table-7 results. Note the disclaimer at the top: "Warning: This eval runner uses online compaction, which is experimental and not fully tested."

**LongBench / Qasper / RULER**: shell scripts in `scripts/v2/` show CLI patterns; reuse the same `run_qa_evaluation.py` with different `--dataset-name`.

---

### Script entry points and key CLI args

Main shell drivers (paper Figs 1, 3):
- `scripts/main/{qwen,llama,gemma}-{quality,longhealth5}.sh` — SLURM array jobs sweeping methods × ratios per (model, dataset).
- `scripts/qwen-reasoning.sh` — AIME online compaction.
- `scripts/v2/*.sh` — LongBench-v2, RULER, Qasper (newer).
- `scripts/head-budget-optimization.sh` — runs `head_budget_optimization/run.py` to produce JSON budgets.

Key CLI flags for `python -m evaluation.run_qa_evaluation`:

| Flag | Meaning |
|---|---|
| `--algorithm-config best\|fast\|default\|...` | which config dict to load (from `evaluation/configs/algorithms/`) |
| `--query-config self-study\|ss-plus-repeat\|repeat\|context-prefill\|random-vectors` | which queries to use |
| `--methods omp_nnls0_-inf_7_drop-7_lsq_progressive_on-policy` | name(s) from algorithm config to run |
| `--target-size 0.05` | compaction ratio (float < 1) or absolute target (int) |
| `--precomputed-budget-path .../optimized_agnostic.json` | nonuniform per-head budgets |
| `--max-ratio-per-head 0.75` | cap any single head's ratio (default 1.0) |
| `--chunking longhealth\|fixed\|lqa` | chunked compaction strategy |
| `--n-articles N --start-article K` | which articles to evaluate |
| `--compute-stats 1` | also compute train-time reconstruction metrics |

The conventional naming for methods encodes everything: e.g. `omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy` = OMP, `nnls_iters=0`, no lower bound, upper $e^7$, drop cutoff $-7$, lsq solver for $C_v$, $k=4$, $\tau=2$, on-policy.

---

### Implementation surprises / subtleties

- **Attention sinks kept verbatim**: there's no special "sink token" code, but the **partial-compaction mechanism** (`is_partial_compaction=True`) lets you compact only `indices=range(prefix_end, article_end)`, leaving the chat-template prefix (which includes BOS/`<|im_start|>system`) and suffix untouched (`per_layer_head.py:486-501`, `chunked.py` similarly). So "attention sinks" fall out of compacting only the *article* portion. The paper's claim of "natural sink handling" maps to this.
- **`beta = -inf` for padding** rather than implementing variable-length attention — works because all kernels respect $-\infty$ in the additive mask. Clever shortcut around varlen.
- **`drop_key_beta_cutoff=-7`** as default: low-weight keys (mass weight $< e^{-7}$) are essentially redundant; dropping them lets OMP fill those slots with more impactful keys. Not described in detail in the paper (just stated "refinement"). Bounded to 3 entries to avoid oscillation.
- **`nnls_upper_bound=exp(7)`** prevents pathological single-key dominance; not in paper.
- **`lstsq` then clamp** as NNLS approximation when `iters=0`: hack that works because OMP's greedy selection mostly avoids the simplex boundary. Empirically the projected-gradient `iters>0` is only needed for HighestAttnKeys.
- **`_compute_C2_on_policy` is `UNUSED`** per its docstring — on-policy path uses `_compute_C2` with on-policy queries everywhere. Cleaner mathematically (queries are consistent through key selection, $\beta$ fit, and $C_v$ fit).
- **Numerical scale**: scores are stability-shifted by `max` per query before `exp`; everything else (NNLS / OLS) handles this absorbed scale via $\beta = \log w$. Means $\beta$ values stored are off by a per-query const $\Rightarrow$ but since $\beta$ is added to the *same* per-query softmax at inference, the shift cancels.
- **GQA query expansion in `_attention_to_kv`**: reinterprets attention-head queries as KV-head queries by concatenating along the query axis. Combined with self-study giving N queries per token, effective queries per KV head explode quickly (paper caps at ~50k).
- **Cache for queries vs cache to compact**: `past_key_values_for_queries` is a separate kwarg (`per_layer_head.py:240`) used in KV-based chunking so on-policy queries see the right RoPE context even though the cache being compacted is just a slice.
- **`vllm_model` reuse**: self-study calls vLLM for generation but the **same model object** is reused for query-vector extraction via HF forward. The vLLM weights are loaded once and shared — important for memory budget.
- **Repeat-prefill is just teacher-forcing**: `ConversationSpec(conversation_starter="Repeat the previous context verbatim.", prefill_with_article=True)` — vLLM isn't even called for that spec; the article tokens are appended as if the model had emitted them, and Q activations from `q_proj` are extracted.
- **Sliding-layer rotation uses local rotary** (`rotary_emb_local`) in Gemma3 because that layer family has a different `rope_theta`. Easy to miss in chunked text-based mode.

---

### Files worth opening in this order if reading the code

1. `compaction/algorithms/base.py` — `CompactionAlgorithm`, `_compute_C2`, `_nnls_pg`, `evaluate_compaction` (metrics).
2. `compaction/algorithms/omp.py` — `SimpleOMPCompaction.select_keys` then `OMPCompaction._select_keys_omp`.
3. `compaction/algorithms/highest_attention_keys.py` — short, mirrors OMP structure.
4. `compaction/compaction_methods/per_layer_head.py` — outer loop, padding, per-head budgets.
5. `models/cache.py:CompactedPrefixCache` + `models/qwen3/modeling_qwen3.py:Qwen3Attention.forward` (lines ~218-271) — how $\beta$ enters attention.
6. `compaction/compaction_methods/chunked.py` — RoPE-shift math (`compute_rope_correction`, `apply_rotary_pos_emb_to_cache`).
7. `head_budget_optimization/solver.py:HeadBudgetSolver.solve_swap` and `solve_ratio_agnostic_swap`.
8. `compaction/query_generation/{self_study.py,conversation_specs.py,query_generator.py}`.
9. `evaluation/configs/algorithms/best.py` — see exactly what config strings expand to.

