# Artificial Hippocampus Networks for Efficient Long-Context Modeling

**ByteDance Seed, 2025** (arXiv 2510.07318). Fang, Yu, Zhong, Ye, Xiong, Wei.
Code: `github.com/ByteDance-Seed/AHN`. Models on HF: `ByteDance-Seed`.

> **TL;DR**: Keep a **sliding window** of the Transformer KV cache as *lossless short-term memory*; train a tiny **RNN-like module (AHN)** to recurrently compress everything that falls out of the window into a *fixed-size long-term memory*. Only the AHN params (~0.4%) are trained via **self-distillation**, base LLM frozen. Augmenting Qwen2.5-3B reduces FLOPs 40.5% / memory cache 74% on LV-Eval 128k while *improving* the score (4.41 → 5.88).

---

## 1. Motivation

- **Two memory paradigms, opposite trade-offs:**
  - **RNN-like (Mamba2, DeltaNet, ...)**: compress all history into fixed-size hidden state. Constant per-token compute + memory, but **lossy** → bad at exact long-range recall.
  - **Attention / KV cache**: essentially **lossless** (every token retained), high capacity, but cache grows **linearly** with $L$ and attention cost is **quadratic** $O(L^2)$.
- **Cognitive-science framing — Multi-Store Model (MSM) of memory** (Atkinson–Shiffrin):
  - Lossless **short-term / working memory** = limited capacity & duration.
  - The **hippocampus** continually **consolidates** short-term memory into compact **long-term cortical** representations.
  - Human brain volume is ~constant through adulthood yet keeps absorbing info → motivates fixed-size compressed memory living *alongside* a small lossless buffer.
- **AHN analogy**: sliding-window KV cache = short-term memory; AHN = artificial hippocampus that consolidates evicted KV into a fixed-size state.
- **Key design choice vs prior hybrids**: use a **large** sliding window (default **32k**). AHNs only fire once $L > W$, i.e. exactly where attention's quadratic cost actually bites. For short/medium context the model is **byte-identical to the base full-attention Transformer** (no efficiency loss, no perf regression, no extra effort to preserve short-context quality).

---

## 2. Method

### 2.1 Setup
- Standard self-attention: $Q=XW_Q,\ K=XW_K,\ V=XW_V$, $\text{Attn}=\text{softmax}\!\big(\tfrac{QK^T}{\sqrt{d}}\odot\mathcal M\big)V$ with causal mask $\mathcal M$.

### 2.2 AHN module
- Sliding window of size $W$. For step $t>W$, the KV pair $(k_{t-W}, v_{t-W})$ that **just exited** the window is folded into the compressed memory:
$$ h_{t-W} = \text{AHN}\big((k_{t-W}, v_{t-W}),\, h_{t-W-1}\big) $$
  - $h_{t-W}$ = fixed-size (vector/matrix) compressed summary of context up to $t-W$.
  - Recurrent form ⇒ implementable with any RNN-like architecture.
- **Integration with lossless memory** (Eq. 4): inside window use exact causal attention; once exceeded, compress the out-of-window KV into $h_{t-W}$, then **discard** that KV from cache (keep only $\{(k_i,v_i)\}_{i=t-W+1}^{t}$). Current query reads from **both** memories:
$$ y_t = f\big(h_{t-W},\, \{(k_i,v_i)\}_{i=t-W+1}^{t},\, q_t\big) $$

### 2.3 Instantiations (modern linear-recurrent models)
Three variants; per-head update + output rules. AHN **does not add QKV projections** — it reuses the attention KV cache directly as input, transforming lossless memory into fixed-size memory.

- **AHN-GDN (Gated DeltaNet)** — representative, gated delta rule:
$$ h_{t-W} = \alpha(x_{t-W})\big(\mathbf I - \beta(x_{t-W})k_{t-W}^Tk_{t-W}\big)h_{t-W-1} + \beta(x_{t-W})k_{t-W}^Tv_{t-W} $$
  - learnable per-head $W_\alpha, W_\beta \in \mathbb R^{D\times1}$ feeding $\alpha(\cdot)$ (decay gate) and $\beta(\cdot)$ (write strength).
  - Differs from vanilla GatedDeltaNet: only compresses **out-of-window** tokens, not all past tokens.
- **AHN-Mamba2** (Eq. 9): $h_{t-W} = \exp(-\Delta(x_{t-W})A)h_{t-W-1} + \Delta(x_{t-W})k_{t-W}^Tv_{t-W}$.
- **AHN-DN (DeltaNet)** (Eq. 10): $h_{t-W} = (\mathbf I - \beta(x_{t-W})k_{t-W}^Tk_{t-W})h_{t-W-1} + \beta(x_{t-W})k_{t-W}^Tv_{t-W}$ (no decay gate).

- **Output rule** (shared by all three, Eq. 6–7):
$$ y_{\text{AHN},t} = \gamma(x_t)\, q_t\, h_{t-W}\, W_o, \qquad y_t = y_{\text{AHN},t} + \text{Attn}\big(\{(k_i,v_i)\}_{i=t-W+1}^t, q_t\big) $$
  - $\gamma(x_t)=x_tW_\gamma$ scalar gate per head ($W_\gamma\in\mathbb R^{D\times1}$); output linear $W_o\in\mathbb R^{H\times H}$ grouped by head ($H$ = head dim).
  - **AHN output is simply summed with the sliding-window attention output.**

### 2.4 Complexity (Table 1, $N_q/N_{kv}$ heads, $W$ window)
| | Full causal attn | Window + AHN-GDN |
|---|---|---|
| Params | $2DH(N_q{+}N_{kv})$ | $+\,3DN_q + H^2N_q$ (tiny) |
| Memory cache | $2LHN_{kv}\sim O(L)$ | $2WHN_{kv}+H^2N_q\sim O(W)$ **constant** |
| FLOPs | $\sim O(L^2)$ | $\sim O(WL)$ **linear in $L$** |

→ AHN makes attention **linear in $L$** with **constant cache** beyond the window.

### 2.5 Training — self-distillation (Fig 2b)
- **Teacher** = original open-weight LLM (e.g. Qwen2.5-Instruct), full attention, output prob $p'$.
- **Student** = same LLM, **frozen**, but token mixer switched to **(sliding-window attention + AHN)**; output prob $p$.
- Loss = **KL divergence** $\ell = \text{KL}(p'\,\|\,p)$. **Only AHN params trained** (for AHN-GDN: $W_\alpha, W_\beta, W_\gamma, W_o$ per head) → **~0.4% of base params** with $N_q$ heads.
- No retraining of base LLM; pretraining stage of an AHN model = identical to base (AHNs inactive at short pretraining lengths).

---

## 3. Experiments

### 3.1 Setup
- **Base models**: Qwen2.5-Instruct **3B / 7B / 14B**.
- **AHN module**: Mamba2, DeltaNet, GatedDeltaNet → AHN-Mamba2 / AHN-DN / AHN-GDN.
- **Train data**: ChatQA2, **1B tokens**, 1 epoch. Max seq len **24k**. AdamW, lr **1e-4** (10% linear warmup → cosine decay), global batch **128**, **740 update steps**. Training AHNs for a **7B** model: **~10h on 32 A100s**.
- **Randomization during training** (key to generalization):
  - attention-sink size $\sim$ uniform over $\{0,32,64,128,512,2048,4096\}$ (drop candidates $>$ half seq len).
  - total lossless tokens (sinks + window) $\sim$ uniform over $\{32,64,128,256,512,1024,2048,4096,8192\}$ (drop $<$ one-eighth seq len).
- **Inference default**: 32k lossless memory = **128 attention sinks + 32640-token sliding window**.
- **Baselines**: Sliding-Window Attention + attention sinks (SWA); **Compressive Transformer (CT)** with max/avg pooling at 4× compression (matched lossless budget + compressed-memory size).
- **Benchmarks**: LV-Eval (128k subset), InfiniteBench (128k), LongBench (>8k tasks), RULER, PG19 (illustration).

### 3.2 Results
- **LV-Eval + InfiniteBench (128k, Table 2)**: AHN-augmented models **beat SWA+sinks across nearly all tasks** and often **surpass full attention**, at **~46–50% mixing FLOP ratio, ~59–65% model FLOP ratio, 26% memory cache**.
  - Qwen2.5-3B: LV-Eval avg **Full 4.41 → SWA 4.59 → AHN-GDN 5.88**; InfiniteBench **9.52 → 10.47 → 13.24**.
  - 7B AHN-GDN: LV-Eval **6.54** vs Full 3.62; InfiniteBench **16.93** vs 9.52.
- **LongBench 6 tasks (Table 3, fixed 8192 lossless = 128 sinks + 8064 window)**: AHN > both baselines. 7B AHN-GDN avg **40.59**, AHN-DN **41.04** vs SWA 38.52.
- **Efficiency (Fig 3, PG19 57k)**: AHN gives **linear FLOPs** (Fig 3a), **constant memory cache** (3b), **flat low perplexity** beyond 32k window (3c, base Qwen perplexity blows up past pretrained context), **near-constant CUDA memory** (3d).

### 3.3 Ablations
- **Training objective (Table 4)**: self-distill KL >> next-token CE. CE → marked LongBench drop (sparse signal, pushes tiny AHN to shortcuts); KL gives dense distributional guidance → 38.53(CE) vs 40.59(KL self-distill, random window).
- **Window: randomized vs fixed**: randomized training window generalizes; fixed window **overfits** (38.53 fixed vs 40.59 random).
- **Inference window size (Fig 4)**: AHN beats SWA at all window sizes. Perf climbs 1k→16k; mild dilution drop past 64k (LV-Eval) / 96k (InfiniteBench) — attention dilutes when #keys huge. **32k chosen** as balance.
- **Pure RNN baseline (Table 8)**: removing SWA and keeping only the AHN/RNN module **collapses** (LV-Eval 0.04, InfiniteBench 1.19). AHN's value is *augmenting* a large SWA, not replacing attention.
- **RULER NIAH (Table 5)**: AHN-GDN ≈ SWA, **markedly below full attention** on exact-recall → confirms the lossy-compression limitation.
- **Gradient probing (Fig 5)**: visualize $\partial/\partial x_\text{out}\ \text{KL}(f'(x_\text{win},x_\text{out})\,\|\,f(x_\text{win},h_\text{AHN}))$. Low-grad out-of-window tokens = already well captured in $h_\text{AHN}$. AHN **preferentially stores informative tokens** (math symbols, numbers) over pronouns/special tokens.

### 3.4 Limitations
- Fixed-size compressed memory ⇒ **information loss**, hurts **exact-recall** tasks (NIAH/RULER).
- Future: stronger recall mechanisms; apps in lifelong learning, streaming video, edge deployment.

---

## 4. Implementation

Code under `AHN/src/ahn/`. Built on **HF Transformers**, **LLaMA-Factory** (training harness), and **`fla` (flash-linear-attention)** for the recurrent kernels. Implements only Qwen2/Qwen3; the AHN modules are architecture-agnostic.

### 4.1 RNN modules — `src/ahn/rnn/{gated_deltanet,delta_net,mamba2}.py`
- Forked from `fla`, **modified to consume the attention KV cache directly** instead of having their own QKV projections. Each exposes:
  - `forward(hidden_states, q_states, k_states, v_states, ...)` — note `q/k/v` come **from the base attention layer**, not from internal projections.
  - `get_beta_alpha(hidden_states)` — precomputes the gate inputs ($\beta$, $\alpha$/$\Delta$) for caching at inference.
- **`hidden_states` is a 2-tuple `[hidden_states_ab, hidden_states_g]`**:
  - `_ab` → gate params (`b_proj`→$\beta$ via sigmoid; `a_proj`+`A_log`+`dt_bias`→$\alpha$ via softplus for GDN/Mamba). At **inference** `_ab` is instead a dict `{"beta","alpha"}` of precomputed/cached gate values.
  - `_g` → output gate via `g_proj` (FusedRMSNormSwishGate).
- **`GatedDeltaNet`** (`gated_deltanet.py`): `expand_v=1`, `use_gate=True`, `head_dim` from base. Params per head: `b_proj`, `a_proj` ($\to N_q$ scalars each), `A_log`, `dt_bias`, `D`, `g_proj`, and **`o_proj` = `GroupLinear`** (per-head grouped output linear, $W_o$).
  - Training uses `chunk_gated_delta_rule`; inference uses `fused_recurrent_gated_delta_rule` when `q_len<=64`, else chunk. `use_qk_l2norm_in_kernel=True`.
- **`DeltaNet`**: same shape minus the decay gate ($\alpha$); `get_beta_alpha` returns `(beta, None)`. Kernels `chunk_delta_rule` / `fused_recurrent_delta_rule`.
- **`Mamba2`**: `q/k` L2-normalized then used as SSM `C`/`B`; `dt_proj`→$\Delta$ (the "beta" slot), `A_log`, `dt_bias`, `g_proj`, `o_proj=GroupLinear`. Uses `mamba_chunk_scan_combined` (prefill/train) and `selective_state_update` (single-step decode). `D` disabled.
- **`GroupLinear`** (`utils.py`): per-head weight $(\text{num\_groups}, \text{in/g}, \text{out/g})$, einsum `"bgi,gio->bgo"`. **`reset_parameters` zeros the weight** → AHN output starts at 0 so the model initially equals the base (important for stable distillation).

### 4.2 AHN wrapper — `BaseAHN` (`utils.py`)
- Wraps an `ahn_cls` (GDN/DN/Mamba2), forces `expand_v=1`. Holds an **inference-time gate cache** `ahn_cache = {"beta","alpha"}` and `num_cached_tokens`.
  - `update_cache(hidden_states)` — runs `get_beta_alpha`, appends gate values along seq dim (so gates are computed once, reused as tokens stream out of window).
  - `query_cache(num_attn_sinks, num_cached_tokens)` / `trim_cache(...)` — slice out the gate values for the tokens being compressed this step, then evict them (skipping the sink region).
  - `reset_cache()` — per generation.
- `AHNRouter` (optional, `use_ahn_router`): learnable position-dependent gate $g=\sigma(\alpha\log r + \beta)$ over `pos_ratio` to mix `h_mem` and `h_local` instead of plain sum; optional RoPE-style dim-wise enrichment. Default config (training script) does **not** use it → plain additive combine.

### 4.3 Integration with base model — `src/ahn/transformer/qwen2_ahn/qwen2_ahn.py`
- **`Qwen2MemDecoderLayer`** replaces the standard decoder layer (selected via `config._layer_implementation="Qwen2MemDecoderLayer"`, `_ahn_implementation` ∈ {GatedDeltaNet, DeltaNet, Mamba2} → `AHN_CLS`). Holds `self.self_attn = Qwen2MemFlexAttn`, `self.ahn = BaseAHN(...)`, plus MLP/norms. `fa_layer_ids` lets specific layers stay full-attention (AHN disabled there).
- **`Qwen2MemFlexAttn`** = Qwen2 attention split into:
  - `pre_attn_forward` — projects Q/K/V, applies RoPE, updates KV cache, returns `(q,k,v)` **before `o_proj`** (o_proj applied later, after AHN fused in).
  - `flex_attn_forward` — uses **`torch.nn.attention.flex_attention`** with a block mask for the **sliding-window + sink** pattern (used in training & prefill). Pads seq to {2048,4096,...,32768} multiples of 128.
  - `_flash_attention_forward` for full-attention layers / single-token decode.
  - `forward` dispatches `forward_train` / `forward_inference`.
- **Sliding-window + sink mask** — `Qwen2Model.create_sparse_mask`:
  ```python
  def ctx_sliding_window_mask(b, h, q_idx, kv_idx):
      causal  = q_idx >= kv_idx
      sliding = (q_idx - kv_idx) < sliding_window
      sink    = kv_idx < num_attn_sinks
      return causal & (sliding | sink)
  ```
  Built into a `create_block_mask` (BLOCK_SIZE=128) in `pre_model_forward`.
- **Randomized window/sink at train time** (`pre_model_forward`): `sliding_window_type="random"` samples window from `[32..8192]` with `seq_len/8 < c < seq_len`; `ahn_position="random"` samples sinks from `[0,32,...,4096]` with `c <= 0.5*window`. Stored as `dy_sliding_window` / `dy_num_attn_sinks`.

#### Training forward (`mem_forward_train`)
1. Prenorm → `self_attn(enable_ahn=True)` returns attention output (windowed via flex mask) **and** raw `(q,k,v)`.
2. `in_ahn_seq_len = seq_len - sliding_window`. If `>0`:
   - **out-of-window KV** `mem_k, mem_v = k/v[sinks : sinks+in_ahn_seq_len]` and their hidden states `mem_h_ab` → AHN input.
   - **current queries** `mem_q = q[sliding_window:]`, `mem_h_g = hidden[sliding_window:]` → drive output gate.
   - `ahn_attn_output = self.ahn([mem_h_ab, mem_h_g], mem_q, mem_k, mem_v)`.
3. **Combine**: `ensemble_attn_output[:, sliding_window:, :] = attn_output[:, sliding_window:, :] + ahn_attn_output` (or via `AHNRouter`). Then `o_proj` → residual → MLP.
4. Note the **teacher path runs in parallel**: `Qwen2Model.forward` keeps a `detach().clone()`'d `mem_hidden_states` stream with `requires_grad=True` for the AHN student, while the frozen full-attention stream produces the teacher logits.

#### Loss (`Qwen2ForCausalLM.forward`)
- `loss_type` ∈ {`kl`,`ce`,`distill`} (config; training script uses **`kl`**). Loss computed **only on tokens beyond the window**: `valid_mask[:, sliding_window:]=True` (`& labels!=-100`).
  - **`kl`**: `KL(softmax(logits_ref) || log_softmax(logits_mem))` over valid mask — teacher = full-attn logits_ref, student = AHN logits_mem.
  - **`ce`**: next-token CE on `logits_mem`.
  - **`distill`**: per-layer MSE between teacher/student attn outputs (`all_distill_pairs`, sliced past window).
- **Freezing**: LLaMA-Factory `--finetuning_type freeze --freeze_extra_modules ahn`; only `ahn.*` trains. `save_pretrained` with `save_ahn_only=True` filters state dict to keys containing `"ahn"`.
- **Init** (`from_pretrained` w/ `do_train`): `ahn.fn.q_proj` ~N(0,0.02); `ahn.fn.o_proj` **zeros**; router alpha/beta **zeros** → AHN contributes nothing at step 0.

#### Inference forward (`mem_forward_inference`)
- KV cache already updated by attention. `in_ahn_seq_len = cur_cache_size - sliding_window - num_attn_sinks`.
- `self.ahn.update_cache(prenormed_hidden_states)` caches gate values incrementally.
- If `in_ahn_seq_len > 0`: query cached gates, build AHN input from the out-of-window KV slice, run `self.ahn(...)` updating the **recurrent state** in `mem_past_key_value` (an `fla` `Cache`), add to attention output.
- **`trim_cache`** then **physically evicts** the just-compressed KV tokens from `past_key_value.key_cache/value_cache` (concatenating sink region + post-window region), and trims the gate cache → KV cache stays bounded at `sinks + window`.
- Decode (`q_len==1`): drops the current token's own KV from the windowed-attention key set (it's handled by AHN/state), uses flash attention.

### 4.4 Training script — `examples/scripts/train_qwen2.5_3b_ahn_gdn.sh`
Key flags:
```
--model_name_or_path Qwen/Qwen2.5-3B-Instruct
--finetuning_type freeze  --freeze_extra_modules ahn
--loss_type kl  --use_normalized_l2 True
--layer_implementation Qwen2MemDecoderLayer
--ahn_implementation GatedDeltaNet
--sliding_window 256  --sliding_window_type random  --ahn_position random
--dataset chatqa2  --cutoff_len 24576
--learning_rate 1e-4  --num_train_epochs 1  --lr_scheduler_type cosine --warmup_ratio 0.1
--weight_decay 0.1  --bf16  --save_ahn_only True
--deepspeed examples/deepspeed/ds_z3_config.json   # ZeRO-3
--per_device_train_batch_size 1 --gradient_accumulation_steps 4  (×8 GPUs)
```
- `--sliding_window 256` is just the lower-bound seed; actual window is **randomized per batch** via `sliding_window_type=random`.
- `register_custom_accelerator` (utils) monkey-patches `Accelerator.backward` to `retain_graph=True` (teacher + student share graph).

### 4.5 Inference demo — `examples/scripts/inference.py`
- Registers custom Qwen, loads merged base+AHN checkpoint with `AutoModelForCausalLM`. Defaults: **`--sliding-window 32640`, `--num-attention-sink 128`** (= 32k lossless), greedy decode. Standard `model.generate` — AHN cache/eviction handled inside the decoder layer.

---

## Related Notes

- **[Recurrent Memory Transformer](../RMT/RMT.md)** — Conceptual ancestor. Both compress past context into a **fixed-size recurrent state** carried forward across the sequence. AHN differs by (1) compressing the **KV cache** via modern RNNs (Mamba2/DeltaNet/Gated DeltaNet) instead of routing through learned memory tokens, (2) keeping a **lossless sliding window + attention sinks** of recent KV alongside the compressed long-term memory, and (3) training only the AHN module (~0.4% params) on a **frozen** base via KL self-distillation rather than full BPTT.
- **[AutoCompressors](../AutoCompressors/AutoCompressors.md)** — Same compress-out-of-window-context goal; AutoCompressors uses accumulated lossy **summary vectors** as soft prompts and retains no exact recent tokens, while AHN keeps an exact sliding window and a *constant*-size compressed state (AutoCompressors' summary set grows with #segments).
- **[Compress and Attend Transformer (CAT)](../CAT/CAT.md)** — Sibling 2025 efficient-long-context-via-compression work. Key architectural contrast: AHN holds a **constant-size** recurrent long-term memory (linear FLOPs, constant cache), whereas CAT keeps a **growing-but-compressed** set of chunk reps the decoder attends to ($\approx C\times$ KV reduction). CAT also exposes a **test-time** quality/compute knob (chunk size $C$); AHN's compression ratio is fixed by the trained window.
