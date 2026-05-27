# KVzap: Fast, Adaptive, and Faithful KV Cache Pruning

**Authors**: Simon Jégou, Maximilian Jeblick (NVIDIA) — arXiv 2601.07891, Feb 2026.
**Code**: NVIDIA/kvpress (KVzap submodule + `KVzapPress` press class)
**One-liner**: Train a tiny per-layer linear/MLP surrogate on hidden states to *predict* KVzip importance scores; combine with DMS-style threshold eviction + sliding window to get a phase-agnostic (prefill + decode), kernel-friendly KV cache pruner achieving 2–4x compression with negligible accuracy loss.

---

## TL;DR / Why It Matters

- **The gap**: 20+ KV cache pruning methods exist, but *none* have shipped in vLLM / SGLang / TRT-LLM. Reason: speed-vs-accuracy tradeoffs.
	- Pruning that works (KVzip) is slow; pruning that's fast (heuristics like H2O, SnapKV) degrades.
	- Most pruners also only work in *prefill* — useless for reasoning models that generate thousands of tokens.
- **KVzap = oracle approximation** of KVzip:
	- KVzip is essentially an *oracle* for "which KV pairs are useful?" but needs a 2x-prompt repeat forward pass — too expensive online.
	- KVzap distills that oracle into per-layer surrogate (linear or 2-layer MLP) over hidden states.
- **Four criteria** the authors argue any production-ready pruner must satisfy:
	1. **Fast & lightweight** — negligible overhead.
	2. **Phase-agnostic** — works in both prefill and decode.
	3. **Optimization-friendly** — compatible with FlashAttention2 / PagedAttention.
	4. **Faithful** — minimal accuracy loss across tasks.
- KVzap hits all four; achieves SOTA on the KVpress Leaderboard (Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-32B).

---

## Background: KV Cache Compression Landscape

- **Shape recap**: KV cache has shape $(2, L, H, T, D)$ — layers, KV heads, sequence, head dim.
	- E.g. Llama1-65B at $T{=}128k$ bf16 → **335 GB**. Bottlenecks both TTFT and decode throughput.
- **Axes of compression**:
	- **H-axis (heads)**: GQA (4x Llama3), MQA (12x GLM4.5), GLA, etc.
	- **D-axis (head dim)**: MLA (DeepSeek-V2) → 4H/9 compression.
	- **L-axis (layers)**: sliding window / hybrid attention (GPT-OSS-120B 2x, Gemma3 6x), state-space layers (Jamba 8x, Kimi-Linear 4x).
	- **T-axis (tokens)**: this is where pruning lives. No widely adopted architectural change here.
		- Sparse attention (DSA in DeepSeek V3.2) **retrieves** top-k KVs per step → faster but **does not reduce memory**.
		- Pruning *evicts* KVs permanently → reduces memory.
- **Pruning families**:
	- **Heuristic / training-free**: H2O (accumulated attention "heavy hitters"), StreamingLLM (sliding window + sinks), SnapKV, PyramidKV, KeyDiff, Knorm, etc. → degrade at modest ratios.
	- **Per-head budget**: AdaKV, Duo Attention.
	- **Oracle-ish**: **KVzip** — current SOTA on KVpress leaderboard, up to 4x, but requires double-prompt forward pass at prefill and *cannot run at decode*.
	- **End-to-end learned**: DMS (Dynamic Memory Sparsification) trains per-token eviction policy via Gumbel-sigmoid + KL+sparsity loss. KVzap differs by distilling KVzip scores rather than end-to-end optimization (and has no public DMS ckpts to compare).
- **Quantization** (KIVI, ZipCache) is orthogonal and stackable.

---

## KVzip (the Oracle Being Approximated)

- **KVzip** (Kim et al. 2025) defines per-token importance via a "**copy-and-paste pretext task**":
	- Build extended prompt:
		```
		user: <prompt>
		Repeat the previous context exactly.
		assistant: <prompt>
		```
	- For each head, score KV pair at position $i$ in the original `<prompt>` as **max attention** that any repeated-prompt token $j$ pays to original position $i$:
$$s_i = \max_{j \in \text{<prompt>}} a_{ji}$$
	- **Intuition**: if the model needs token $i$ to reproduce the prompt, it's useful; otherwise it's dead weight.
- **Limitations**:
	- 2x-length prefill → **prohibitively slow** for online inference (violates Criterion 1).
	- Inherently a prefill-only signal; cannot apply during decode (violates Criterion 2).

---

## KVzip+ (Normalization Tweak)

- KVzap adds a normalization inspired by **Expected Attention** (Devoto et al. 2025).
- Hidden-state update at decoding step $j$:
$$\mathbf{h}_j^{out} = \mathbf{h}_j + \sum_{i \leq j} a_{ji} W_O \mathbf{v}_i$$
	- Each token's contribution to the residual stream is $a_{ji} W_O \mathbf{v}_i$.
- **KVzip+ score**:
$$s_i^+ = \max_{j \in \text{<prompt>}} a_{ji} \cdot \frac{\|W_O \mathbf{v}_i\|}{\|\mathbf{h}_j\|}$$
	- Numerator: actual contribution magnitude of token $i$ to the output at $j$.
	- Denominator: residual stream norm at $j$ (relative importance).
- **Result**: KVzip+ consistently matches or beats raw KVzip on benchmarks (Fig. 2), validating the normalization.

---

## KVzap: The Surrogate Method

### Core Idea

- **Train a tiny per-layer model** $f_\ell: \mathbb{R}^{D_h} \to \mathbb{R}^{H}$ that maps the input hidden state $\mathbf{h}_t$ at position $t$ to **log-KVzip+ scores** for every KV head $\log(s_t^+)$.
	- Log-space chosen to match the exponential nature of softmax attention.
	- One model **per layer** (`n_modules = num_layers`).
	- Two variants:
		- **KVzap-Linear** — single linear projection $D_h \to H$. ~1.1M params (Llama-3.1-8B).
		- **KVzap-MLP** — 2-layer MLP $D_h \to D_h/8 \to H$ with GELU. ~76M params (Qwen3-8B), ~210M (Qwen3-32B).
- **Why hidden states (and not keys/values)**: ablation in Appendix A reports lower $R^2$ when training on $(\mathbf{k}, \mathbf{v})$ directly. Hidden states carry richer/earlier-stream info.

### Algorithm (Prefill + Decode)

```python
def compress(hidden_states, keys, values, kvzap_model, threshold, window=128):
    scores = kvzap_model(hidden_states)         # shape (B, H, T)
    scores[..., -window:] = float("inf")        # protect sliding window (recent tokens)
    indices = torch.where(scores >= threshold)  # threshold eviction (DMS-style)
    return keys[indices], values[indices]
```

- **Threshold-based eviction (not top-k)**:
	- Discard KV pairs with $\log(s^+) < \tau$.
	- **Adaptive**: same $\tau$ yields different compression ratios per prompt → keeps more KVs for dense prompts, fewer for redundant ones. Ablation shows up to 20% variation across prompts (Fig. 5 left).
	- Thresholding beats fixed top-k per-head and AdaKV-style per-layer top-k (Fig. 5 right).
- **Sliding window** ($w{=}128$, StreamingLLM-style):
	- Protects most recent tokens from eviction.
	- *Critical*: ablation on LongBench-LCC — without window, accuracy drops to **28.37%**; with $w{=}128$, **62.51%**; $w{=}512$ no further gain.
	- Reason: hidden states don't explicitly encode position info, so the surrogate can't tell "recent" from "old".
- **Decoding mode**: requires a **score buffer** because of the sliding window — you only know a token is safe to evict once it exits the window.

### Training Procedure

- **Dataset**: `nvidia/Nemotron-Pretraining-Dataset-sample` — 9 subsets (CommonCrawl, multilingual, code, math, synthetic-QA, SFT-code/general/math).
	- Filter to 750–1250 tokens per prompt (so attention denominators aren't length-dependent).
	- 500 prompts/subset train + 5 test → ~2.4k prompts.
	- Randomly sample 500 tokens per prompt → **1.2M (h, log(s+)) pairs per KV head**.
	- 23k held-out for validation.
- **Targets**: extract KVzip+ scores by running the model on the repeat-prompt; collect attention weights, normalize, log, store.
- **Loss**: MSE on log-scores. Trained per-layer:
	- **MLP**: AdamW, lr 5e-3, cosine LR schedule, 15 epochs, batch 512, gradient clip 1.0, 5% validation split (skorch wrapper around PyTorch).
	- **Linear**: scikit-learn `Ridge` regression per layer.
- **$R^2$ on validation** (Table 1):
	- Qwen3-8B: 0.671 (Linear) / **0.711** (MLP)
	- Llama-3.1-8B-Instruct: 0.743 / **0.772**
	- Qwen3-32B: 0.629 / **0.668**
	- MLP > Linear on $R^2$ everywhere; first transformer layer hardest to predict (token embeddings carry little info).
- **Per-head heatmaps** (Fig. 6–8): some heads much easier to predict than others; correlates with compression ratio achievable per head.

---

## Integration: Prefill, Decode, Kernels

- **Where it runs**: after each attention forward pass (post-hoc). Unlike DMS which *evicts before attention* (sparse prefill), KVzap masks/evicts *after*.
	- Trade-off: KVzap doesn't accelerate prefill compute, but is much easier to integrate.
- **Compute / memory overhead** (Appendix B, Table 3):
	- Per-layer linear projections cost in transformer: $C = 4D_h(H_QD + HD) + 6D_hD_{int}$ (attention proj + SwiGLU FFN).
	- KVzap-MLP cost: $C_{MLP} = \frac{D_h}{4}(D_h + H)$.
	- KVzap-Linear cost: $C_{Linear} = 2D_hH$.
	- **Relative overhead** (linear projections only — ignores quadratic attention which dominates at long context):

| Model | $C_{MLP}/C$ | $C_{Linear}/C$ |
|---|---|---|
| Qwen3-8B | **1.09%** | 0.02% |
| Llama-3.1-8B | 0.96% | 0.02% |
| Qwen3-32B | 0.67% | 0.01% |

- **In long-context regimes, attention is $O(T^2)$ → KVzap is effectively free**.
- **Decode is memory-bandwidth-bound** → KVzap's extra FLOPs fill otherwise-stalled GPU cycles.
- **Kernel compatibility**:
	- Works with **FlashAttention2** (post-attention pruning).
	- Requires **PagedAttention** kernels supporting variable-length blocks because **per-head non-uniform cache lengths** result from threshold eviction.
	- Authors note: turning compression into wall-clock speedup needs careful engineering — they explicitly didn't tackle this.

---

## Experiments

### Models
- Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-32B.

### Thresholds Used
- Qwen3-8B / Qwen3-32B: $\tau \in \{-6, -5, -4, -3\}$
- Llama-3.1-8B: $\tau \in \{-9, -8, -7, -6\}$ (lower KVzip+ score distribution; see Fig. 7).

### Benchmarks
- **RULER 4k** (n=6500) — long-context retrieval / multi-hop / aggregation / QA.
- **LongBench** (n=4750) — 21 subsets, single/multi-doc QA, summarization, few-shot, code.
- **AIME25** (n=30) — reasoning (long *outputs*, tests decode-phase pruning).

### Headline Results (Table 2, best config)

| Metric | Qwen3-8B (MLP, τ=-4) | Llama-3.1-8B (Linear, τ=-7) | Qwen3-32B (MLP, τ=-4) |
|---|---|---|---|
| RULER 4k | 95.32 → 95.09 (0.74) | 95.69 → 95.55 (0.68) | 95.65 → 95.95 (0.68) |
| RULER 16k | 92.99 → 92.78 (0.72) | 93.42 → 93.29 (0.70) | 95.19 → 94.96 (0.65) |
| LongBench | 46.74 → 46.49 (0.66) | 45.25 → 44.65 (0.62) | 50.56 → 50.40 (0.57) |
| AIME25 pass@4 | 0.77 → 0.77 (0.75) | – | 0.83 → 0.87 (0.60) |
| **Avg compression** | **0.72 (3.5x)** | **0.67 (3.0x)** | **0.63 (2.7x)** |

- Negligible degradation, sometimes *improves* over full cache (e.g. Qwen3-32B on RULER 4k 95.65→95.95, AIME25 0.83→0.87).

### Key Trends from Detailed Results
- **KVzip+ ≥ KVzip** on average → validates the $\|W_O v\| / \|h\|$ normalization.
- **KVzap ≈ KVzip+ up to ~3–4x**, despite being a cheap surrogate.
- **MLP > Linear on Qwen**; surprisingly **Linear > MLP on Llama-3.1-8B-Instruct** (despite lower $R^2$ — Linear even *beats* the KVzip+ oracle it approximates on RULER for Llama).
- **Expected Attention pseudo-wins on LongBench** are an artifact of one outlier subset (TREC) — excluding TREC makes KVzap the clear winner (Fig. 12).
- **AIME25 reasoning** (Fig. 4): KVzap-MLP holds pass@1 and pass@4 even when discarding >50% of KV; KVzap-Linear collapses at high compression (e.g. τ=-3 gives 96% compression → 0 correct).
- **Adaptive compression**: same $\tau$ → different compression depending on prompt complexity (RULER higher, LongBench lower).

### Ablations
- **Threshold vs top-k**: thresholding wins (Fig. 5 right).
- **Sliding window size**: $w{=}128$ essential, $w{=}512$ no extra gain.
- AIME25 detailed (Table 4): τ=-3 gives 96/93% compression on Qwen3-8B/32B → 0 correct answers (over-pruned). τ=-4 to -6 sweet spot.

---

## Limitations / Discussion

- **Not training-free**: post-hoc adds-on. Long-term, end-to-end objectives like DMS may win (analogy in paper: Multi-Token Prediction supersedes ad-hoc speculative decoding).
- **Variable-length per-head caches**: requires PagedAttention-style kernels; kernel optimization is "non-trivial" and **not done here**.
	- Pre-attention application would directly accelerate prefill — left as future work.
- **Scaling**: only validated up to 32B; needs testing on 235B, GLM 4.5, MLA architectures (DeepSeek V3.2).
- **Per-layer surrogates** = bigger model size on disk; the MLP variant scales with model hidden dim ($D_h/8$ inner dim).
- **Train-test distribution shift**: trained on ≤1250-token prompts, but evaluated on much longer (RULER 16k, etc.). Authors acknowledge this.
- **First-layer hard to predict**: token embeddings alone lack contextual signal needed for KVzip+ score.

---

## Connections to Related Work

- **Closest cousin: DMS** (Łańcucki et al. 2025) — also learns per-token eviction via thresholding + sliding window. Difference:
	- DMS = end-to-end Gumbel-sigmoid + KL loss; evicts *before* attention (sparse prefill).
	- KVzap = distill KVzip+ scores; evicts *after* attention.
- **vs KVzip**: same scoring philosophy ("can the model reproduce the context?"), but oracle vs surrogate.
- **vs Expected Attention**: borrows the residual-stream-contribution normalization $\|W_O v\| / \|h\|$.
- **vs H2O / SnapKV**: cleaner score, no reliance on past attention statistics → works in decode.
- **vs AdaKV / Duo Attention**: KVzap is also per-head-variable but uses *score thresholding* not budget allocation.
- **vs DSA (DeepSeek V3.2 sparse attention)**: DSA retrieves; KVzap evicts. Orthogonal.
- **Stackable with quantization** (KIVI, ZipCache).

---

## Implementation

Code lives in NVIDIA/kvpress repo. Two pieces:

1. **Training pipeline** in `kvzap/` subfolder.
2. **Inference press** in `kvpress/presses/kvzap_press.py` (the surrogate scorer), combined with `kvpress/presses/dms_press.py` (the threshold + sliding-window eviction logic).

### Surrogate Model: `KVzapModel` + `KVzapPress`

- **File**: `kvpress_repo/kvpress/presses/kvzap_press.py`
- **`KVzapModel`** (extends `PreTrainedModel`, lines 25–48):
	- Single `nn.ModuleList` of $L$ per-layer modules — either `nn.Linear(D_h, H)` (linear) or `nn.Sequential(Linear, GELU, Linear)` (MLP).
	- `forward(x)` stacks per-layer outputs: `torch.stack([module(x[:, i, :]) for i, module in enumerate(self.layers)], dim=1)`.
- **`KVzapPress`** (extends `ScorerPress`, lines 51–82):
	```python
	def post_init_from_model(self, model):
	    kvzap_model_name = f"nvidia/KVzap-{self.model_type}-{model.config.name_or_path.split('/')[-1]}"
	    self.kvzap_model = KVzapModel.from_pretrained(kvzap_model_name)

	def score(self, module, hidden_states, keys, values, attentions, kwargs):
	    kvzap_module = self.kvzap_model.layers[module.layer_idx]
	    kvzap_module = kvzap_module.to(hidden_states.device, dtype=hidden_states.dtype).eval()
	    scores = kvzap_module(hidden_states).transpose(1, 2)
	    return scores
	```
	- Loads pretrained surrogate by HF id `nvidia/KVzap-{linear|mlp}-{base_model}`.
	- `score()` simply applies the surrogate for the current `layer_idx` → returns `(B, H, T)` scores.

### Eviction Logic: `DMSPress`

- **File**: `kvpress_repo/kvpress/presses/dms_press.py`
- `DMSPress` **wraps any `ScorerPress`** (including `KVzapPress`), implements DMS-style thresholding + sliding window.
- **Key fields** (lines 48–53):
	```python
	press: ScorerPress
	threshold: Optional[float] = None
	sliding_window_size: int = 128
	decoding: bool = False
	scores_buffer: dict[int, torch.Tensor] = field(default_factory=dict, init=False)
	```
- **`forward_hook`** (lines 69–130) — runs after each attention layer's forward:
	- **Prefilling detection**: `prefilling = cache_len == q_len` (line 74).
	- **Buffer reset** on layer 0 of prefill (lines 80–82).
	- **Decode gating**: skip if `not prefilling and not self.decoding` (lines 85–86).
	- **Score computation**: delegates to wrapped `KVzapPress.score()` (line 90).
	- **Score accumulation**:
		```python
		if prefilling:
		    self.scores_buffer[layer_idx] = scores
		else:
		    self.scores_buffer[layer_idx] = torch.cat([self.scores_buffer[layer_idx], scores], dim=-1)
		```
	- **Sliding-window-aware eviction** (lines 99–120): only KVs whose scores have *left* the trailing window are candidates:
		```python
		if self.scores_buffer[layer_idx].shape[-1] > self.sliding_window_size:
		    n_to_evict = self.scores_buffer[layer_idx].shape[-1] - self.sliding_window_size
		    scores_to_evict = self.scores_buffer[layer_idx][..., :n_to_evict]
		    self.scores_buffer[layer_idx] = self.scores_buffer[layer_idx][..., n_to_evict:]
		    new_masked_key_indices = list(torch.where(scores_to_evict < self.threshold))
		    ...
		    module.masked_key_indices = ...   # accumulates (batch, head, seq) tuples
		```
		- Note: this implements *fake* per-head compression by writing index tuples to `module.masked_key_indices` — see attention patch below.
	- **Compression ratio tracking** (lines 123–128): `n_masked / (B * H * cache_len)` per layer.

### Per-Head Masking via Attention Patch (FA2 compatibility)

- **File**: `kvpress_repo/kvpress/attention_patch.py`
- Because non-uniform per-head cache lengths can't be expressed in standard kernels, KVzap (and AdaKV, KVzip) use a "fake key" trick.
- **`attention_patch(func)` wrapper** (lines 43–87):
	- During *decoding only* (when `query.shape[2] != key.shape[2]`), if `module.masked_key_indices` is set:
		```python
		q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
		q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
		k = search_hyperplane(q)                           # finds k s.t. <q,k> <<0 for all queries
		k = k.view(bsz, num_key_value_heads, head_dim)
		batch_indices, head_indices, seq_indices = module.masked_key_indices
		key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]
		```
	- **`search_hyperplane(X)`** (lines 8–40): iterative search for a vector $Y$ with $\langle X_i, Y\rangle \leq 0$ for all $i$; returns $-10^5 \cdot Y / \|Y\|^2$ so that $\exp(\langle q, k\rangle) \approx 0$.
	- Replaces masked keys with "annihilator" keys that produce near-zero attention weight → mathematically equivalent to removing them, **without breaking FA2's fixed-shape kernel**.
- **`patch_attention_functions()`** (lines 90–110): monkey-patches all entries in `transformers.ALL_ATTENTION_FUNCTIONS`. Called automatically when `kvpress` is imported.
- **Caveat in docstring**: "This solution is not optimal as it does not reduce peak memory and slightly increases runtime" — matches paper's "implementation challenges" discussion.

### Score Extraction for Training: `KVzapDataCollector`

- **File**: `kvpress_repo/kvzap/data.py`
- **`repeat_prompt_tokenization`** (lines 90–141): builds the KVzip extended prompt via chat template:
	```python
	messages = [
	    {"role": "user", "content": prompt + "\n\nRepeat the previous context exactly."},
	    {"role": "assistant", "content": prompt},
	]
	```
	- Returns token spans `(start_prompt, end_prompt, start_repeated_prompt, end_repeated_prompt)`.
- **`_forward_hook`** (lines 173–222) — registered on every attention layer; computes KVzip+ targets:
	```python
	scores = attn_weights                                 # (B, H, T, T)
	h_norm = torch.norm(hidden_states, dim=-1)
	scores = torch.einsum("b h t i, b t -> b h t i", scores, 1 / h_norm)  # /||h||
	# ||W_o V|| by column
	Wo = module.o_proj.weight.transpose(0, 1).view(num_heads, V.shape[-1], hidden_size)
	WoV_norm = torch.einsum("h i j, b h t i -> b h t j", Wo, V).norm(dim=-1)
	scores = torch.einsum("b h t i, b h i -> b h t i", scores, WoV_norm)
	# max over repeated-prompt tokens; max over GQA group; log
	scores = scores[:, :, start_repeated:end_repeated, start_prompt:end_prompt].amax(dim=2)
	scores = scores.view(..., num_kv_heads, num_kv_groups, T).amax(dim=2)
	scores = torch.log(scores)
	self._data.append((hidden_states[0, start_prompt:end_prompt, :].cpu(), scores[0].T.cpu()))
	```
	- Direct implementation of Eq. (3) (KVzip+ formula).
	- Stores `(hidden_states, log_scores)` pairs per layer → become $(X, y)$ for training.
- **FP8 handling** (lines 199–203): unscales `o_proj` weight if it's an `FP8Linear`.
- **`collect()`** (lines 239–289): runs forward pass with hooks, randomly samples 500 tokens per prompt, stacks into $(N \cdot 500, L, D_h)$ and $(N \cdot 500, L, H)$ tensors.

### Training Loop

- **File**: `kvpress_repo/kvzap/train.py`
- **`train_mlp`** (lines 28–84): skorch `NeuralNetRegressor` wrapping `KVzapModel`, MSE loss, AdamW, cosine LR, gradient clip 1.0, 5% ValidSplit, batch 512, 15 epochs by default.
- **`train_linear`** (lines 87–119): per-layer scikit-learn `Ridge` regression; weights copied into a `KVzapModel(hidden_dim=None)` (i.e. Linear variant) `nn.Linear`.
- **`train()` orchestrator** (lines 122–231):
	1. Load model (eager attention, optional FP8).
	2. Load Nemotron-Pretraining-Dataset-sample, filter by length, split.
	3. `KVzapDataCollector.collect()` → (X, y).
	4. `train_mlp(X, y)`, `train_linear(X, y)`.
	5. `save_pretrained(output_path / name)`.

### Usage (from README)

```python
from transformers import pipeline
from kvpress import KVzapPress, DMSPress

pipe = pipeline("kv-press-text-generation", model="Qwen/Qwen3-8B", ...)
press = DMSPress(KVzapPress(model_type="mlp"), threshold=-4)

# Prefill-only
press.decoding = False
pipe(context, question=q, press=press)

# Prefill + decode (for reasoning)
press.decoding = True
pipe(prompt, press=press, enable_thinking=True, max_new_tokens=2000)
```

### Press Hierarchy (kvpress abstraction)

- `BasePress` (kvpress/presses/base_press.py) — defines `forward_hook` that wraps attention layer; default applies `self.compress()` only during prefill (line 139 check: `if kwargs["cache_position"][-1] > q_len: return output`).
- `ScorerPress(BasePress)` — adds `score()` abstract method; `compress()` does top-k by score (fixed `compression_ratio`).
- `KVzapPress(ScorerPress)` — implements `score()` via the surrogate.
- `DMSPress(BasePress)` — overrides `forward_hook` entirely to do threshold + sliding window + decode support; wraps an inner `ScorerPress` for scoring.

---

## Bottom Line

- KVzap = "distill the slow oracle into a tiny per-layer regressor", then plug into a DMS-style threshold + window evictor + attention patch for FA2 compat.
- 2.7–3.5x average compression, near-zero accuracy delta, <1.1% compute overhead, works in *both* prefill and decode.
- Strongest practical pruner today by the criteria the authors lay out; the main remaining work is **kernel-level** wins (variable-length PagedAttention) and end-to-end objectives.
