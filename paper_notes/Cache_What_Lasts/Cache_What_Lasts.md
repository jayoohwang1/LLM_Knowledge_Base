# Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs

**Authors**: Ngoc Bui, Shubham Sharma, Simran Lamba, Saumitra Mishra, Rex Ying (Yale + JPMorganChase AI Research). ICLR 2026, arXiv 2512.03324. **Repo calls the method `TRIM-KV`** (Token RetentIon for Memory-bounded KV).

---

## TL;DR

- **Problem**: KV cache grows linearly with context length; existing eviction policies (StreamingLLM, H2O, SnapKV, R-KV, ...) are **attention-guided heuristics** that use recent attention scores as a proxy for future importance. This is **myopic**: useful for next-token prediction, bad for long-horizon decisions where a token might be evicted now but needed thousands of steps later.
- **Idea**: Learn each token's **intrinsic, query-agnostic, long-term utility** at *creation time*. A tiny **retention gate** $g$ maps token hidden state to a scalar $\beta_i \in [0,1]$ per (layer, head). The token's effective attention contribution **decays exponentially** as $\beta_i^{t-i}$. Evict the token with the *lowest* current retention score whenever the fixed budget $M$ is exceeded.
- **Training**: Backbone frozen. Only the gates train, jointly across all layers, via (i) **KL distillation + NTP** to preserve quality and (ii) a **hinge capacity loss** that penalizes $\sum_i \beta_i^{t-i} > M$. The exponential decay is the **smooth, differentiable surrogate** for the otherwise non-differentiable hard eviction mask $\alpha_{ti} \in \{0,1\}$.
- **Result**: Beats SeerAttention-R, R-KV, SnapKV, H2O, StreamingLLM, LocRet across math (AIME24/GSM8K/MATH-500), LongProc, LongMemEval, SCBench, LongBench-V2. **Sometimes beats full-cache** (selective retention as regularization). ~2x decode throughput vs full cache at 32K context. Learned scores exhibit **emergent** sink-token / sliding-window / gist patterns.

---

## Motivation & Framing

- **Two camps of KV-budgeting**:
	1. **Compression / quantization** (KIVI, KVQuant): shrink each entry but still O(T). Works in prefill, scales poorly in generation.
	2. **Offload + retrieval** (Quest, RetrievalAttention, SeerAttention-R): keep full KV in CPU, retrieve top-k blocks. Adds CPU<->GPU orchestration overhead.
	3. **Eviction** (StreamingLLM, H2O, SnapKV, R-KV): hard drop tokens, O(M). Simple and direct, but mostly *heuristic* and *attention-guided*.
- **Eviction as constrained optimization** (Sec 3.2):
	$$\mathbf{o}'_t = \sum_{i=1}^t \frac{\alpha_{ti}\exp(\mathbf{q}_t^\top \mathbf{k}_i)}{\sum_j \alpha_{tj}\exp(\mathbf{q}_t^\top \mathbf{k}_j)} \mathbf{v}_i,\quad \alpha_{ti}\in\{0,1\},\ \alpha_{ti}\ge \alpha_{t+1,i}$$
	$$\min_\alpha \mathcal{L}_{\text{base}}(\mathbf{o}'_t;\mathbf{o}_t)\ \text{ s.t. } \sum_i \alpha_{ti}\le M$$
	- The **monotonicity** constraint encodes that eviction is irreversible.
	- Combinatorial → existing methods solve heuristically. This paper *learns* $\alpha$ implicitly via the retention gate.
- **Key conceptual shift**: ask "how important is token $i$ **long-term**?" not "how important is it for the **current query**?" The contextual embedding $\mathbf{x}_i$ at creation already encodes most of what's needed to predict long-term utility.

---

## Method

### Retention-gated attention

- **Retention gate** $g$: tiny per-block module producing $\beta_t = g(\mathbf{x}_t)\in [0,1]^h$ (one score per KV head).
	- In practice: 1-hidden-layer MLP, hidden dim 512, output dim $h$ = num_kv_heads. Linear projection works too but worse.
- **Replace hard $\alpha_{ti}$ with smooth exponential decay**:
	$$\bar\alpha_{ti} = \beta_i^{t-i}\quad (\beta_i \in [0,1])$$
- **Modified attention** (Eq. 3):
	$$\mathbf{o}_t = \sum_{i=1}^t \frac{\beta_i^{t-i}\exp(\mathbf{q}_t^\top \mathbf{k}_i)}{\sum_j \beta_j^{t-j}\exp(\mathbf{q}_t^\top \mathbf{k}_j)} \mathbf{v}_i$$
	- Equivalent to **additive bias on logits**: $\exp(\mathbf{q}_t^\top\mathbf{k}_i + (t-i)\log\beta_i)$. A learnable, per-token **ALiBi-like recency bias** where slope is data-driven, not fixed.
	- When all $\beta_i = 1$ ⇒ vanilla attention.
- **Why exponential not sigmoid?** They considered $\bar\alpha_{ti}=\sigma(f(\mathbf{x}_i,t))$ but: (i) sequence length unknown during decode → domain unnormalized; (ii) sigmoid saturates → vanishing gradients.
- **Brain analogy**: Ebbinghaus forgetting curve $R = \exp(-tS)$. $\beta$ is memory strength.

### Training objective

Frozen backbone. Only gate params $\theta$ trained.

- **Quality loss**: KL between baseline pretrained LLM $p(\cdot|x)$ and the retention-gated proxy $q_\theta(\cdot|x)$, plus standard NTP.
	$$\mathcal{L}_{\text{quality}} = D_{KL}(p(\cdot|x)\ \|\ q_\theta(\cdot|x)) + \mathbb{E}_{(x,y)}[-\log q_\theta(y|x)]$$
- **Capacity loss** (hinge, **the key sparsity driver**):
	$$\mathcal{L}_{\text{cap}} = \frac{1}{T}\sum_{t=1}^T \frac{1}{t}\max\!\left\{0,\sum_{i=1}^t \beta_i^{t-i} - M\right\}$$
	- Pushes effective "soft cache size" $\le M$ at each step, **per layer, per head**.
	- $M$ acts as a **soft** hyperparameter at training time; at inference any budget can be used.
- **Combined**: $\min_\theta \mathcal{L}_{\text{quality}} + \lambda_{\text{cap}}\mathcal{L}_{\text{cap}}$, with $\lambda_{\text{cap}}=1.0$ in main experiments.
- **Initialization trick**: bias of gate's final linear init to **large positive value** (e.g. $b=18$) → $\beta \approx 1$ at start → minimal early forgetting, stable training. Ablation Fig 9 shows $b=0$ degrades sharply.
- **Hardware-aware**: implemented with **FlexAttention** + custom **Triton** kernel for $\mathcal{L}_{\text{cap}}$ to avoid materializing the $T\times T$ $\beta$-matrix. Trains up to **128K tokens on 4xH100**.

### Inference

- **Decoupled from training-time attention modulation**. At inference: gates produce $\beta_i$ on the fly; attention is **standard** (FlashAttention). The decay terms are *only* used as the eviction *score*.
- **Eviction rule** (greedy, append-then-evict):
	1. Project Q/K/V for token $t+1$, compute $\beta_{t+1} = g(\mathbf{x}_{t+1})$.
	2. Append to cache, run standard FlashAttention restricted to current set $S_t$.
	3. If $|S_t| > M$: evict $j_{\text{evic}} = \arg\min_{j\in S_t} \beta_j^{t-j}$.
	- Score is per-head; cache compressed independently per (layer, head).
- **Memory overhead**: 1 scalar $\beta_i$ per token per head → $\approx 1/d_h$ relative to KV (negligible). Does **not** store queries (unlike R-KV).
- **Throughput** (Table 6, H200, M=1024): at 32K context, batch 4 → **130 tok/s vs 68 (Full) vs 124 (SnapKV) vs 69 (SeerAttn-R)**. Decode time **31.4s vs 59.8s** full.

### Positional encoding agnosticism

- The exponential decay is **not** a positional encoding. It approximates the eviction process. With RoPE, cache **post-rotated** keys (same as R-KV/SnapKV) → eviction orthogonal to PE.

---

## Brain-inspired Interpretation

- $\bar\alpha_{ti} = \exp((t-i)\log\beta_i)$ ≡ Ebbinghaus exponential forgetting with **per-token memory strength** $\log\beta_i$.
- Some tokens encode "durable memories" (high $\beta$), others fade fast — emergent, not hardcoded.
- Contrast with prior work (Zhong et al. 2024) that applied Ebbinghaus to RAG retrieval, not to attention internals.

---

## Experiments

### Setup

- **Models**: Qwen3-{1.7B, 4B, 8B, 14B}, DeepSeek-R1-Distill-{Qwen-7B, Llama-8B}, Phi3-mini-128K.
- **Training data**: OpenR1-Math-220K (reasoning), SynthLong + BookSum + Buddhi (long-context), LongAlpaca (chunk-prefill).
- **Training**: 4xH100, lr $2e^{-4}$, weight decay 0.01, batch 1, gradient accum 4. Max seq 16K (reasoning) or 128K (long-context). Only gates trained.
- **Baselines**: SeerAttn-R (retrieval, learnable), R-KV (eviction, learnable), SnapKV/H2O/StreamingLLM/KeyDiff (heuristic eviction), LocRet (chunk-prefill learnable eviction).

### Long-generation (math reasoning)

- Fig 3 / Fig 6: Pareto frontier (accuracy vs KV budget) on AIME24, GSM8K, MATH-500 across Qwen3-1.7B/4B/8B/14B and R1-Distill variants.
- **TRIM-KV dominates at low budgets**: e.g. on AIME24 at 512 KV budget, 198.4% relative gain over R-KV/SnapKV; 58.9% pass@1 gain over SeerAttn-R at same budget.
- **Sometimes beats full cache** (e.g. Qwen3-4B AIME24): selective retention as regularization, suppresses noise from uninformative tokens (filler, punctuation).
- **Pure eviction**, no CPU↔GPU offload → strictly faster than SeerAttn-R.

### Long procedural generation (LongProc, Table 1 / Table 7)

- Tasks: HTML→TSV, Thought-of-Mind, Travel Planning, CountDown, Pseudo→Code.
- TRIM-KV (trained on **math**) still wins on these non-math procedural tasks → retention scores generalize beyond training domain.
- Even surpasses full cache on CountDown 0.5K/2K.

### Long-context understanding

- **LongMemEval-S (Table 3, 8)**: 123K-token chat-memory benchmark. TRIM-KV@32K = 44.8% vs StreamingLLM@32K = 27.6%, SnapKV@32K = 27.6%; **Full@131K = 49.4%**. Holds up at 25% of budget while baselines collapse.
- **SCBench (Table 2)**: competitive, struggles on incompressible retrieval tasks (Retr.KV) like all eviction methods.
- **LongBench-V2 (Table 4, Phi3-mini)**: +18.41% over Full-KV (!), beats LocRet (-2.64%).

### Qualitative findings (Figs 4, 5, 11–19)

- High retention: task-relevant tokens (`ometer`, `shop`, `walk`, `minutes`), initial sink token `<|im_start|>`, problem-statement tokens.
- Low retention: whitespace, punctuation (except periods in some heads → implicit gist).
- **Emergent eviction patterns per head**:
	- Sliding window (early layers)
	- Sink tokens (late layers, A-shape)
	- Multiple coexisting sliding windows of varying widths
	- Context switching (sentence-aligned blocks)
	- **Period-only retention** → implicit gist tokens, summarizing prev sentence
	- Mathematical-symbol heads, problem-description heads, reasoning-keyword heads
- Sparsity ↑ with depth: later layers sparser than early (consistent with PyramidKV).
- **No sliding window in some heads** (e.g. layer 15 head 2) — counter-evidence to handcrafted sliding-window methods (LocRet, Yuan 2025) that hardcode this.

### Ablations

- **Objective (Table 5)**: removing each loss
	- −KL → 72.1 vs 75.8 (full TRIM-KV)
	- −NTP → 72.5
	- −Capacity loss → **42.9** (sharp drop — capacity loss is critical).
- **Training $M$**: $M=128$ slightly > $M=512$ (over-optimizes sparsity); $M=\infty$ (no cap loss) collapses. Match to deployment budget.
- **Gate arch**: MLP > linear; large positive init bias ($b=18$) essential.
- **Training data**: gates trained on general long-context data generalize to math; math-trained gates degrade faster at tight budgets.

### Limitations / Future

- Backbone still frozen; jointly training Q/K/V with retention from pretraining unexplored.
- Implementation assumes uniform KV length across heads in a layer (FlashAttention constraint) → **per-head variable budgets** could squeeze more (DBTrimKV in repo addresses this).
- No multimodal / tool-use evaluations in paper (repo has DBTrimKV VLM extension).

---

## Connections & Context

- **vs ALiBi/RoPE**: TRIM-KV's $\beta_i^{t-i}$ acts as additive logit bias just like ALiBi, but slope is **per-token learnable** (intrinsic to content) rather than fixed per-head.
- **vs Forgetting Transformer (Lin 2025)**: similar in spirit (learnable forget gate on attention logits), but TRIM-KV: (i) only gates trained, backbone frozen; (ii) explicit budget constraint via hinge loss; (iii) inference does hard eviction, not soft decay.
- **vs SSMs/Linear attention (Mamba, RetNet, Titans)**: those put forgetting into a fixed-size hidden state; TRIM-KV keeps softmax attention but converts a pretrained LLM into a memory-bounded model via plug-in gates.
- **vs LocRet (Huang 2024)**: also learnable eviction with retention scores, but per-layer/head trained independently and requires handcrafted sliding window. TRIM-KV trains gates **end-to-end** so sliding-window etc. emerge as needed.
- **vs gist tokens (Mu 2023)**: TRIM-KV discovers gist behavior emergently in some heads (period tokens summarizing sentences).
- **vs SeerAttn-R / Quest**: retrieval-based, full KV in host RAM. TRIM-KV is strictly eviction → no orchestration cost.
- **DBTrimKV** (repo, post-paper update): tied projection across layers/heads + paged-attention-style block reassignment → global budget, dynamic per-head reallocation.

---

## Implementation

Repo: `trimkv_repo/`. Package: `trimkv` (Python ≥3.11, PyTorch ≥2.7, Transformers 4.57.1, FlashAttention ≥2.7.2).

### Layout

- `src/trimkv/models/{qwen3,qwen2,llama,phi3,qwen3_vl,qwen2_5_vl,llava}/` — per-model HF Transformers patches (`modeling_trimkv_*.py`, `configuration_trimkv_*.py`). Subclass the original model, replace attention block with retention-gated attention.
- `src/trimkv/cache_utils.py` — three cache classes that subclass `transformers.cache_utils.DynamicCache`: `TrimKVCache`, `DynamicBudgetTrimKVCache`, `PagedTrimKVCache`.
- `src/trimkv/attn/` — attention forward variants (`eager_attn.py`, `flex_attn.py`, `db_flash_attn.py`); dispatched by `get_attention_interface(impl_name)`.
- `src/trimkv/triton/` — custom Triton kernels for retention-sum (capacity loss) and dynamic-budget cache ops.

### Retention gate

`src/trimkv/models/qwen3/modeling_trimkv_qwen3.py:134-173` — `RetentionGate` (the "rg" variant):

```python
class RetentionGate(nn.Module):
    def __init__(self, config):
        ...
        self.linear1 = nn.Linear(hidden_size, retention_gate_intermediate_size, bias=True)  # hidden=512 by default
        self.linear2 = nn.Linear(retention_gate_intermediate_size, num_key_value_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_key_value_heads))
        self.act_fn = ACT2FN[hidden_act]
    def reset_parameters(self):
        ...
        self.bias.data.fill_(self.config.retention_gate_bias_init)  # e.g. 18.0 → β≈1 at init

    def forward(self, x):
        out = self.act_fn(self.linear1(x))
        out = self.linear2(out) + self.bias       # (B, S, num_kv_heads)
        return F.logsigmoid(out)                  # returns log β  ∈ (-∞, 0]
```

- Outputs **log β** (not β) for numerical stability — log-sigmoid avoids saturating sigmoid gradients.
- `RetentionGate10` (lines 176-214): the "rg10" variant used by **DBTrimKV** — 2-hidden-layer MLP, final projection tied across heads.

### Hook into attention

`src/trimkv/models/qwen3/modeling_trimkv_qwen3.py:262-350` — `TrimKVQwen3Attention.forward`:

```python
# project Q/K/V
query_states = self.q_norm(self.q_proj(hidden_states).view(...))
key_states   = self.k_norm(self.k_proj(hidden_states).view(...))
value_states = self.v_proj(hidden_states).view(...)
# retention gate -> log β per (B, H_kv, S)
if self.config.retention_gate in ['rg', 'rg7', 'rg10']:
    retention_weights = self.retention_gate(hidden_states).transpose(1, 2)
# RoPE
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
# write to TrimKVCache, getting back cached K, V, β, kv_positions
key_states, value_states, retention_weights, kv_positions, flash_attn_kwargs = \
    past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
# pick attention impl: training -> rg_attn_flex; inference -> flash/eager/etc.
attn_impl = "rg_attn_flex" if (self.training and not vanilla_forward) else self.config._attn_implementation
attention_interface = get_attention_interface(attn_impl)
```

- **Critical separation**: at **inference**, attention is standard FlashAttention — retention weights are *not* multiplied into logits, they only steer cache eviction. At **training**, `retention_gated_attention_forward` injects β into logits (smooth surrogate).

### Training-time retention-gated attention

`src/trimkv/attn/eager_attn.py:77-112` (reference impl):

```python
def retention_gated_attention_forward(module, q, k, v, attn_mask, retention_weights, ...):
    retention_weights = torch.exp(retention_weights.to(torch.float32))   # β (the gate returned log β)
    retention_weights = power_stair(retention_weights, q_len)            # M[..., i, j] = β_j^(i-j+1) for i≥j
    attn_weights = (q @ k.transpose(2,3)) * scaling
    attn_weights = attn_weights * retention_weights                       # multiplicative decay before softmax
    attn_weights = softmax(attn_weights + causal_mask, dim=-1)
    return attn_weights @ v, ..., summerized_retention_weights
```

- `power_stair(a, L)` builds the lower-triangular matrix $M[i,j] = a_j^{(i-j)}$ — the per-token decayed retention.
- This **materializes the $T\times T$** matrix → only used by eager / FlexAttention; not used for long-context.

`src/trimkv/attn/flex_attn.py:60-106` — **FlexAttention** training kernel; avoids materializing the matrix by injecting decay as a `score_mod`:

```python
def score_mod_wo_kv_pos(score, b, h, q_idx, kv_idx):
    return score + (retention_weights[b, h//N_Q_PER_GROUP, kv_idx] * (q_idx + offset - kv_idx))
# i.e. logit += log(β_kv) * (q_idx - kv_idx)  ⇔  multiplicative β^(q-kv) in prob space
attn_output = flex_attention_compiled(q, k, v, score_mod=score_mod, block_mask=attn_mask, ...)
```

- The `score_mod` adds $\log\beta_i \cdot (t-i)$ to the logit — exactly the additive-bias form noted in Sec 4.1.
- `torch.compile`d for speed.

### Cache & eviction (`TrimKVCache`)

`src/trimkv/cache_utils.py:55-276`. Inherits `transformers.cache_utils.DynamicCache`, stores 4 lists per layer: `key_cache`, `value_cache`, `retention_weights` (log β), `kv_positions`.

**Update** (`update`, lines 129-175): appends new K, V, log β, positions. Standard concat along seq dim. Called by the attention block.

**Compress / evict** (`compress`, lines 199-247) — called once **per forward pass** from `TrimKVQwen3Model.forward` line 718-719:

```python
if past_key_values is not None and isinstance(past_key_values, TrimKVCache):
    past_key_values.compress()
```

Eviction body:

```python
for layer_idx in range(num_layers):
    if memory_size + buffer_size <= self.get_cache_length(layer_idx):
        log_beta = retention_weights[layer_idx].to(torch.float32)
        q_idx = self.get_seq_length() + 1
        scores = log_beta * (q_idx - kv_positions)         # = log( β^(t-i) ); per (B, H_kv, S)
        if self.sliding_window_size > 0:
            scores[:, :, -self.sliding_window_size:] = float('inf')  # force keep recent
        if self.attention_mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        top_k_indices = torch.topk(scores, memory_size, dim=-1).indices
        top_k_indices, _ = torch.sort(top_k_indices, dim=-1)          # preserve temporal order
        # gather K, V, β, positions by top-k indices
        self.key_cache[layer_idx]   = key_states.gather(-2,   top_k_indices.unsqueeze(-1).expand(..., D))
        self.value_cache[layer_idx] = value_states.gather(-2, top_k_indices.unsqueeze(-1).expand(..., D))
        self.retention_weights[layer_idx] = retention_weights[layer_idx].gather(-1, top_k_indices)
        self.kv_positions[layer_idx]      = kv_positions.gather(-1, top_k_indices)
```

- **Implementation = batched top-k + gather**, not the line-by-line argmin loop in Algorithm 1 of the paper. Equivalent because the cache only grows by `buffer_size` per step → top-k restores to size $M$.
- **Eviction is per layer × per head** (the head dim is preserved by top-k along the last axis).
- Sliding-window override is optional (default 0) — for emergent learning experiments, set to 0.
- `buffer_size` (default 1) acts as a small slack so eviction triggers when cache exceeds $M + \text{buffer}$.
- `lookahead_steps` > 1 uses `compute_log_G` (lines 28-52) — a multi-step retention score $\sum_{n} \beta^n$ via `log1mexp` for log-domain stability. Not the default.

### DynamicBudgetTrimKVCache (per-head variable lengths)

`src/trimkv/cache_utils.py:279+` — flattens K/V across heads into 1D layout `(B*H*S, D)` and tracks per-head lengths in `head_lens`, `cu_seqlens_k` (FlashAttention-varlen style). Uses Triton `update_flatten_view_triton` to insert new tokens at the right head boundary. The `compress` method computes scores **globally** across all heads in all layers, runs a single top-k for `total_memory_size = num_layers * num_heads * M`, then redistributes survivors per head. Enables global budget reallocation.

### PagedTrimKVCache (DBTrimKV — repo only, post-paper)

Same `cache_utils.py`, paged-attention-style blocks (re)assigned to heads on the fly. Coupled with `RetentionGate10` (tied final projection). Activated via env vars `RETENTION_GATE=rg10`, `GLOBAL_CAPACITY=True`.

### Loss computation

`src/trimkv/models/qwen3/modeling_trimkv_qwen3.py:762-883` — `TrimKVQwen3ForCausalLM`:

- **`compute_retention_loss`** (lines 762-809): the hinge capacity loss.
	```python
	# summarized_retention_weights[b, l, h, t] = sum_i β_i^(t-i)  precomputed by Triton kernel
	memory_size = self.config.memory_size  # M
	retention_loss = torch.maximum(
	    (summarized_retention_weights - memory_size) / (position_ids - memory_size).clamp(min=1.0),
	    torch.zeros_like(summarized_retention_weights)
	)
	# averaged over positions where the hinge is active (positive)
	retention_loss = retention_loss.sum() / num_non_zero
	```
	- Normalized by $(t-M)$ → prevents over-penalizing early positions when $t < M$.
	- Global-capacity branch (lines 787-794) used by DBTrimKV: sums retention across all layers/heads, compares to `M * L * H`.
- **`compute_fwkl_loss`** (822-841): forward KL distillation between proxy logits and frozen-baseline logits `base_logits` (computed elsewhere by a vanilla forward).
- **`compute_ntp_loss`** (811-820): standard cross-entropy on labels.
- **`compute_base_loss`** (863-883): combines NTP + KL according to `config.base_loss` (e.g. `"ntp|fwkl"`).
- **Total loss** (lines 963-969): `loss = base_loss + retention_weight * retention_loss`.

### Custom Triton kernels

`src/trimkv/triton/retention_sum.py`, `retention_sum_packed.py`: compute $\sum_{i\le t} \beta_i^{t-i}$ over all (b, l, h, t) **without materializing the $T\times T$** $\beta$-matrix. Used both for the capacity loss and (optionally) as `summarized_retention_weights` returned from `retention_gated_attention_forward`. Packed variant handles document-packing (multiple sequences concatenated with `position_ids` resets).

### Training entry point

`train/llm/train.py` — HF Trainer + DeepSpeed.
- `model_args.trainable` = `"self_attn.retention_gate"` → only gate params get gradients.
- Two recipes: `train/llm/scripts/train_trimkv_math.sh` (R1-style math), `train_trimkv_long.sh` (long-context KL distill on SynthLong/BookSum/Buddhi).
- `RETENTION_GATE` / `GLOBAL_CAPACITY` env vars toggle TrimKV vs DBTrimKV.

### Inference quickstart

```python
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM
from trimkv.cache_utils import TrimKVCache
model = TrimKVQwen3ForCausalLM.from_pretrained("ngocbh/TrimKV-Qwen3-4B-Math",
    torch_dtype=torch.bfloat16, load_trimkv_weights=True, device_map="cuda")
model.config._attn_implementation = "flash_attention_2"
past_key_values = TrimKVCache(memory_size=512, buffer_size=32, device="cuda")
# model.generate(..., past_key_values=past_key_values)
```

- `from_pretrained` (lines 460+) auto-downloads the gate weights (`trimkv_weights.pth`) from HF/wandb and merges into the base-model checkpoint.

---

## Key Takeaways

- **Token retention** is a different abstraction than attention-based importance: query-agnostic, intrinsic, long-horizon. Captures what attention proxies miss.
- **Smooth exponential decay** is the trick that makes the discrete eviction objective differentiable — and it's also brain-plausible (Ebbinghaus).
- The **hinge capacity loss** is doing most of the heavy lifting — ablation shows it's load-bearing.
- **Decoupling training-time attention modulation from inference-time eviction** is what gives you full-FlashAttention speed at inference. Train with FlexAttention, ship with FA2.
- **Emergent eviction patterns** (sinks, sliding windows of multiple widths, gist via period tokens, context switching) match prior hand-crafted heuristics — and reveal new ones — without explicit design. Retention scores double as an interpretability probe for head/layer specialization.
- Selective retention can **outperform full cache**: noise from uninformative tokens hurts; eviction = implicit regularization.
- Repo extension **DBTrimKV** with `PagedTrimKVCache` does paged-attention-style dynamic head budgets — natural next step the paper flagged as future work, already shipped.
