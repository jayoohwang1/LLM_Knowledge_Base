# End-to-End Test-Time Training for Long Context (TTT-E2E)

> arXiv 2512.23675v2 (Dec 2025). Tandon, Dalal, Li, Koceja, Rød, Buchanan, X. Wang, Leskovec, Koyejo, Hashimoto, Guestrin, McCaleb, Choi, Y. Sun.
> Astera / NVIDIA / Stanford / UC Berkeley / UCSD. Code: github.com/test-time-training/e2e (JAX).

**One-liner**: Long-context LM reframed as **continual learning, not architecture design**. Use a *standard* Transformer + **sliding-window attention (SWA)**; at test time keep training the model via **next-token prediction on the given context** (compresses context into weights). Meta-learn the test-time *initialization* at training time. The method is "End-to-End" in **both** directions: inner loop optimizes the *same* next-token loss as the test loss (E2E at test time), outer loop optimizes the *post-TTT* loss (E2E at training time).

---

## 1. Framing: continual learning vs architecture design

- **Full attention**: nearly lossless recall but cost/token grows linearly with context → $O(T^2)$ prefill, $O(T)$ decode. Attends to every detail.
- **RNNs (Mamba2, Gated DeltaNet)**: constant cost/token but degrade in long context — fixed hidden state can't hold everything.
- **Key insight**: humans don't recall every detail; they **compress** experience into the brain. Next-token-prediction training also compresses data into weights. **So: just keep training the LM at test time on the given context.**
- This is **Test-Time Training (TTT)** = continual learning where the test instance itself is the training data. Old idea = **dynamic evaluation** [Mikolov, Krause]. TTT-E2E's contribution is fixing what dynamic eval missed.
- **Memory analogy** (conclusion): test-time-updated weights = long-term memory; sliding window = short-term memory. Biological hierarchy.

---

## 2. Method

### 2.1 TTT via next-token prediction (the inner loop)

Test time has **prefill** (condition on $x_0,\dots,x_T$, $x_0$=`<BOS>`) and **decode** (predict $\hat p_{T+1}$). Test loss = $\mathrm{CE}(\hat p_{T+1}, x_{T+1})$.

- **Toy baseline**: Transformer with *all self-attention removed* → only MLPs → effectively a bigram (no memory).
- Give it memory by **training on the context**. Per-token next-token loss:
$$\ell_t(W) = \mathrm{CE}(f(x_{t-1};W), x_t)$$
- Update $W$ online, sequentially, $t=1,\dots,T$:
$$W_t = W_{t-1} - \eta\,\nabla \ell_t(W_{t-1}), \qquad W_0 = \text{init at test time}$$
Output $\hat p_{T+1} = f(x_T; W_T)$. (Fig 2 left: baseline uses only "up" arrows; TTT adds backward + horizontal arrows that carry $W$ forward.)

### 2.2 Learning to learn (the outer loop / meta-learning)

- Because $\ell_t(W_{t-1})$ **is** the test loss for predicting $x_t$ from $x_0..x_{t-1}$, the **whole-sequence test loss** is:
$$\mathcal{L}(W_0; X) = \tfrac{1}{T}\sum_{t=1}^{T}\ell_t(W_{t-1}) = \tfrac{1}{T}\sum_{t=1}^{T}\mathrm{CE}(f(x_{t-1};W_{t-1}), x_t)$$
- **E2E training**: directly minimize this *post-TTT* loss over a training set of sequences → call result **TTT-E2E**. Training loss == test loss.
- **Contrast — TTT-naive** (= dynamic evaluation): imitates a *static* model, ignores that $W$ updates at test time:
$$\mathcal{L}_{\texttt{naive}}(W_0;X) = \tfrac{1}{T}\sum_{t=1}^{T}\ell_t(W_0)$$
  - mismatch between train/test behavior → no guarantee minimizer helps real test loss. Performs only slightly better than the toy baseline.
- $\nabla\mathcal{L}(W_0)$ requires **gradients of gradients** (since update rule itself contains $\nabla$) → autodiff handles it. Outer loop = grad steps on $\mathcal{L}$; inner loop = grad steps on $\ell$.
- **Two problems with $b=1$ online inner loop**: (1) efficiency — many serial steps, no parallelism; (2) stability — each step depends on a single token → gradient explosion.

### 2.3 Mini-batch TTT + sliding window

- Use **mini-batch GD** instead of online. Split $T$ into $T/b$ batches of size $b$, one grad step per batch:
$$W_i = W_{i-1} - \eta\,\tfrac{1}{b}\sum_{t=(i-1)b+1}^{ib}\nabla \ell_t(W_{i-1}), \qquad i=1,\dots,T/b$$
  Output $\hat p_{T+1}=f(x_T;W_{T/b})$. Training loss generalizes to:
$$\mathcal{L}(W_0;X) = \tfrac{1}{T}\sum_{i=1}^{T/b}\sum_{t=(i-1)b+1}^{ib}\ell_t(W_{i-1})$$
  $b=1$ recovers Eq. 2/3.
- **New problem**: within a mini-batch the model is a **bigram again** (all preds use $W_{i-1}$, not updated per-token) → $\ell_t(W_0)$ rises across the batch as it misses more context (Fig 2 right, red $b=16$ line). Bad gradients.
- **Fix = sliding-window attention**: instead of removing attention, restrict it to window $k$. Window lets the model see context *within* the current mini-batch before TTT updates the weights. Require $k \ge b$.
- **Main settings**: $T$ up to 128K, window $k = 8\text{K}$, TTT mini-batch $b = 1\text{K}$.
- This completes the main architecture: **standard Transformer with SWA**, plus TTT on the MLPs.

#### 2.3.1 Three implementation details (each ablated)
- **TTT only the MLP layers**: freeze embeddings, norms, attention during TTT (updating them → instability in outer loop). Only MLPs are inner-updated.
- **TTT only the last 1/4 of blocks**: more layers updated = larger storage but more backprop compute. Trade-off between compute and context scaling. Choose last 1/4 (model-size dependent; longer context may want more).
- **Two MLP layers per block** (in TTT'd blocks): add a *static, second MLP* as "safe storage" of pretrained knowledge (guards against forgetting). To keep param count equal, **shrink the MLP hidden dim throughout the whole network** (incl. frozen ones).

#### 2.3.2 Decoding multiple tokens
- TTT only takes a grad step once decoded tokens have **completely filled a mini-batch** ($b$ tokens). Decode $b$ tokens normally (SWA), then TTT on that decoded batch, then continue. "Self-training" at test time.

### 2.4 Alternative derivation (connects to RNN/linear-attention lineage)

Start from **TTT-KVB** (Key-Value Binding) [Sun et al. 2024 "RNNs with expressive hidden states", Zhang et al. 2025 "TTT done right"]; reach TTT-E2E in 3 steps. (Not needed for results, but shows the RNN connection.)

- **TTT-KVB / TTT layer** (drop-in replacement for self-attention). At each layer $l$, learn at test time to predict each value from its key (KV binding):
$$\ell_t^{(l)}(W_{t-1}^{(l)}) = \big\| g(\theta_K^{(l)} x_t^{(l)}; W_{t-1}^{(l)}) - \theta_V^{(l)} x_t^{(l)} \big\|^2$$
  with $g$ usually an MLP; $\theta_K,\theta_V$ are **outer-loop** params (like K/V projections). Output rule:
$$z_t^{(l)} = g(\theta_Q^{(l)} x_t^{(l)}; W_t^{(l)})$$
  $\theta_Q$ also outer-loop. This is a **TTT layer**; each is an independent unit with its own loss + weights = generalizes MesaNet, Titans, Nested Learning; linear attention / Gated DeltaNet derivable from this view (footnote: linear model $g$ → DeltaNet; grad w.r.t. $W_0$ → linear attention; nonparametric kernel → self-attention).
- **Step 1 — simplified output rule** (Eq. 9): reuse $g$'s prediction from the loss, drop separate $\theta_Q$: $z_t^{(l)} = g(\theta_K^{(l)} x_t^{(l)}; W_{t-1}^{(l)})$. Negligible harm (Table 1). Justified because SWA already gives local context.
- **Step 2 (the key step) — E2E at test time**: replace the *layer-wise KVB reconstruction* loss with the **single next-token-prediction loss at the end of the network**. This removes the need for $\theta_K,\theta_V$. → "TTT-E2E all layers MH" (MH = multi-head). Big improvement (Table 1, 2.819→2.806).
  - Now test loss == token-level $\ell_t$ → E2E at test time. The two derivations are **dual**: primary derivation is E2E at test time then makes it E2E at training time (meta-learn); this derivation starts E2E at training time (TTT-KVB) and makes it E2E at test time.
- **Step 3 — larger state, less compute**: (a) update only last 1/4 of blocks instead of all (smaller compute), (b) revert multi-head LoRA MLPs → regular MLPs (larger effective state at cost of compute). Net: **5× larger hidden state (88M vs 18M for 760M model), half the inference latency** vs "all layers MH". Larger state ⇒ better context scaling (ablation §3.2.1).

> **4 differences between TTT-KVB (right of Fig 3) and TTT-E2E (left):** (1) per-block reconstruction loss vs single end-of-net next-token loss; (2) KVB has extra $\theta_K,\theta_V$ per block; (3) KVB updates an MLP in *every* block vs E2E only last 1/4; (4) KVB MLPs are multi-head + LoRA (much smaller capacity) vs full MLPs.

### Why TTT-E2E can be viewed as an RNN
- Its hidden state = the entire network's inner-updated weights, updated once per mini-batch in one backward pass. Unlike TTT-KVB (one RNN layer per block) it's a *single* RNN layer (whole net updated together).

---

## 3. Architecture (concrete)

- Standard Transformer, **QK-norm** (RMSNorm on q,k per head), **RoPE**, **SwiGLU MLPs**, pre+post RMSNorm, **Llama-3 tokenizer**, tied word embeddings (3B).
- **SWA** replaces full attention everywhere, window $k=8\text{K}$.
- Last 1/4 of blocks ("suffix blocks") are TTT'd; first 3/4 ("prefix blocks") are static (no state, full backprop through them stops at the suffix — gradients pass through last L/4 only).
- Each suffix block has **two MLPs**: a *prime* MLP (inner-updated at test time) and a regular MLP (static safe storage).

---

## 4. Why TTT-E2E scales like full attention while Mamba2 / GDN don't

- **Fig 1 (left)**: loss $\Delta$ vs full attention ($y$=method loss − full-attn loss; full attn = flat line at 0). At 128K, TTT-E2E turns the *worst* line (SWA, green) into the *best* (blue), staying flat/below 0 across 8K→128K. Mamba2, GDN, TTT-KVB, hybrid SWA all degrade (rise) with context.
- **Cause (Fig 9 appendix)**: for the RNN-style baselines, longer context → fewer training sequences per outer-loop batch during extension fine-tuning → higher-variance gradients; they *also* can't leverage longer context → variance harm dominates. TTT-E2E *can* leverage longer context so it keeps improving.
- **Larger effective state** (88M vs 18M) is what lets it scale; ablation: updating only 1 or 3 layers (small state) → doesn't scale; 6 or 12 layers → scales (12 ≈ 6, so use 6 = 1/4 of 24).
- **TTT-E2E even beats full attention on its own pretraining setup**: with $k=8\text{K}$ = pretraining context, SWA == full attention, yet TTT-E2E improves loss by **0.018** on top of full attention → it's an *orthogonal* improvement, not just compensating SWA's deficit.

### Loss breakdown by token index (Fig 6, §3.4.1)
- TTT-E2E is the **only** method with lower loss than full attention across the *entire* context length.
- Its aggregate advantage comes mostly from **earlier tokens** (advantage exists even before $t=1\text{K}$, i.e. before the first TTT step — same compute graph as full attn, differs only in weights).
- **Intuition**: full-attn weights must be good at *all* future tokens in the window (hard, limits capacity for any one). TTT-E2E weights only need to be good at the *present* mini-batch since TTT regenerates weights for future tokens → easier focused task. "Key intuition of TTT = focus on the present."

---

## 5. Inference latency (Fig 1 right, §3.7)

- **Constant prefill latency regardless of context** (like SWA/RNNs), because hidden state = regular MLPs (no $O(T)$ scan). **2.7× faster than full attention at 128K on H100**.
- **TTT-E2E only uses standard infra**: MLP-shaped hidden state shards across GPUs with standard tools, **no custom kernels**. Contrast: TTT-KVB must shrink state via LoRA; Mamba2/GDN need custom kernels to fit state on-chip / for memory I/O.
- **Decode latency** = SWA decode + a prefill-cost TTT step per filled mini-batch.
- **Training latency = the main limitation**: gradients-of-gradients are poorly optimized. **1.2× faster than full attn at 128K but 3.4× slower at 8K** (most pretraining is short context → real cost). FLOPs/token constant but latency grows 8K→32K due to **gradient checkpointing through time** by factor $\log(T)$. cuDNN FlashAttention can't be used (no grad-of-grad support).
  - **Future fixes**: custom flash-attention kernel supporting grad-of-grad; initialize TTT-E2E from a pretrained Transformer *without* TTT so TTT is a small fraction of total compute.

---

## 6. Training & experiments (§3)

- **5 model sizes**: 125M / 350M / 760M / 1.3B / 3B. Headline results use **3B, 164B tokens** (3× Chinchilla pretrain + 3× fine-tune).
- **Two stages**: pretrain at 8K context on **DCLM** (DCLM-Baseline, filtered Common Crawl; discard docs <8K to avoid resetting MLPs across doc boundaries when packing); extension fine-tune at final length (≤128K) on **Books**. Eval on held-out Books.
- **Basic recipe** (Table 3): GPT-3 configs + Mamba tweaks (5× peak LR, half batch for 1B). LR warmup 10% linear then cosine decay to 1e-5. Fine-tune = 5% of pretrain tokens, double batch size, restart cosine. **RoPE θ**: 500K pretrain (8K); for extension θ=10M@128K, log-linear in between (1M@16K, 2M@32K, 5M@64K).
- **6 baselines** (all SWA $k=8\text{K}$): full attention, SWA, Hybrid SWA+full (5:1, Gemma-2 style), Mamba2 (hybrid w/ SWA, Nemotron-H), Gated DeltaNet (hybrid w/ SWA, Kimi Linear), TTT-KVB (hybrid TTT-MLP+KVB+SWA; = Titans/Nested Learning family). Baselines 1-3 reimplemented in JAX; 4-6 use official code.
- Trained on **GB200s**; latency benchmarked on **H100/H200** for fair comparison (baselines lack Blackwell kernels).

### Ablations (Fig 4, §3.2)
- **Window size $k$**: larger $k$ helps all (SWA, GDN, TTT-E2E); TTT-E2E's sensitivity to $k$ similar to baselines. Pick $k=8\text{K}$ (smaller doesn't help runtime much).
- **TTT mini-batch $b$**: only comparable baseline is TTT-KVB. Larger $b$ hurts both; $b<1\text{K}$ hurts hardware utilization/stability. Pick $b=1\text{K}$.
- **Architecture without TTT** ($b=8\text{K}$ ≡ no TTT since pretrain context is 8K): TTT-E2E (2.825) ≈ TTT-KVB (2.826) ≈ full attn (2.827). ⇒ **architecture tweaks play a minor role; the gain comes from TTT itself.**
- **Number of layers updated** (§3.2.1, most important): update last 1/2, 1/4, 1/8 (=12,6,3 of 24). 1 or 3 layers don't scale with context; 6 or 12 do; 12≈6 → use 1/4.

### Scaling with training compute (Fig 5, §3.3)
- Two axes: model size (125M→3B) and #tokens (16B→80B). GDN as representative RNN baseline.
- **Small budget**: TTT-E2E's advantage over full attn shrinks. **Medium+ budget**: TTT-E2E tracks full attention's scaling (blue line flat). Regime boundary ≈ **760M model size**, ≈ **48B tokens**.
- GDN follows the **same** trend as TTT-E2E → both are fixed-state RNNs; alternatively, Transformers are known to underperform with low compute (it's a deficiency of low-compute full attention, not of RNNs).
- **Conclusion**: TTT-E2E should match full-attention scaling trend in large-budget production runs.
- **Tokenizer/data sensitivity**: Llama-3 tok (vs Llama-2) +0.01 advantage @3B; DCLM-2024 (vs SlimPajama) and FineWebEdu make token-scaling match full attn after 48B.

### Needle-in-a-Haystack (Table 2, §3.5) — the weakness
- RULER S-NIAH (pass-key, number-in-haystack, UUID), 3B @128K.
- **Full attention dramatically outperforms everyone, esp. long context.** E.g. S-NIAH-1 @128K: full attn 0.99 vs TTT-E2E 0.06; SWA 0.07.
- Confirms full attention's strength = **near-lossless recall**. TTT-E2E's mechanism is **compression** → discards "irrelevant" details like a random target string. Honest stated limitation.

### Decoding long sequences (Fig 7, §3.6)
- Qwen-3-8B-Base as evaluator. Prefill 8K Books tokens, decode 8K more, plot Qwen NLL over 16K.
- TTT-E2E achieves **lower Qwen loss than full attention** in this limited eval. Both methods: loss jumps at prefill→decode boundary then recovers (Qwen adapting to each method's generation style). Sampling: temp 1, top-p 0.95, repetition penalty 1.1.

---

## 7. Related work positioning (§4)

- **Continual learning**: usually learn from a slowly-shifting *distribution* (e.g. hourly chatbot updates) and avoid forgetting. TTT differs: (1) *personalized* — each instance/context is its own world; (2) *no train/test boundary* — like human commute is both testing and training.
- **TTT taxonomy**:
  - *TTT on nearest neighbors* (locally weighted regression 1970s, local learning, KNN-SVM, recently Hardt-Sun for LLMs, Hübotter): larger effective capacity.
  - *TTT for novel instances* (self-supervised, MAE/BERT auxiliary tasks, TTT-MAE, AlphaProof, Akyürek ARC-AGI): better OOD generalization.
  - *TTT on sequences* (video streams, robotics — no reset between predictions): longer memory; **this paper = text-on-sequences**.
- **TTT-KVB critique**: KVB objective is attention-inspired → people wrongly think long-context TTT = *memorization* and = *architecture design*. This work shows TTT needs **no** key-value memorization and is derived purely from continual learning with minimal architecture change.
- **Fast weights / Fast Weight Programmers (FWPs)** [Schmidhuber 1992, Hinton-Plaut 1987]: inner $W$ = "fast", outer $\theta$ = "slow" ⇒ TTT-E2E is a special case of FWPs. **Clark et al. 2022** is the closest prior method (MLP fast weights, meta-learned init, grad step on chunked next-token loss) but adds fast weights only at the *end* of the net (not interleaved) and lacks linear complexity → no efficiency gain. Linear attention / DeltaNet / Gated DeltaNet are FWPs (Schlag et al.). Irie et al. used MLP layer-wise fast weights for long context, preceding Sun et al.
- **Learning to learn / MAML** [Finn et al.]: MAML's outer loop also meta-learns an init via grad-of-grad, but its inner loop learns from a whole *dataset* (needs many datasets). TTT-E2E casts *language modeling itself* as learning-to-learn (any supervised problem can be cast this way).

---

## 8. Limitations (summary)
- **Recall** (NIAH): compression loses exact details; far worse than full attention for retrieval.
- **Training latency**: grad-of-grad + grad-checkpoint-through-time → 3.4× slower than full attn at 8K (where most compute is spent). No flash kernel for grad-of-grad yet.
- Small-compute regime: advantage over full attention shrinks (matters less at production scale).
- Decode/RL settings not properly evaluated (base models, no instruction-tune/RL).

---
---

## Implementation

> JAX + Equinox + Optax + Hydra. Core: `e2e/ttt/model/{transformer.py, attention.py, loop.py, loss.py}`, `e2e/ttt/{config.py, optimizers.py}`. Configs under `e2e/configs/`. README is in `e2e/README.md`.

The codebase implements the **alternative-derivation architecture**: prefix (static) blocks via `jax.lax.scan`, suffix (TTT'd) blocks holding the inner loop, with the "prime" MLP as the inner-updated weight and the regular MLP as static safe storage. The inner loop is a `scan` over mini-batch chunks taking SGD steps on the next-token loss; the outer loop differentiates through it with AdamW.

### 8.1 Outer loop — `ttt/model/loop.py: train_on_sequence`

`/home/jayoo/code/jayoo_personal/LLM_Knowledge_Base/raw_data/papers/End-to-End Test-Time Training for Long Context/e2e/ttt/model/loop.py`

- `@eqx.filter_vmap(..., axis_name="data_parallel")` ⇒ data-parallel "pmap at call site". `donate="all-except-first"` for memory.
- Loss for a batch = vmap `MetaModel.loss_for_sequence` over sequences then mean. The per-sequence loss is the **post-TTT** $\mathcal{L}(W_0;X)$.
- Gradients accumulated with **Welford's online mean** (`welfords_online_mean`) to avoid storing per-microbatch grads (deep grad-of-grad graph). `accum_steps` reshapes `(accum batch) -> accum batch`.

```python
loss_fn = lambda model, b: vmap_mean(lambda seq: MetaModel.loss_for_sequence(model, seq, state), b, axis_name="batch")
meta_grad_fn = lambda batch: eqx.filter_value_and_grad(loss_fn, has_aux=True)(meta_model, batch)
meta_grad_fn = lambda batch, fun=meta_grad_fn: welfords_online_mean(fun, batch)
(loss, metrics), grads_meta = meta_grad_fn(batch)
avg_loss, avg_metrics, avg_grads_meta = jax.lax.pmean((loss, metrics, grads_meta), axis_name="data_parallel")
avg_grads_meta = avg_grads_meta.trainable_parameters()        # frozen-param spec applied
outer_tx, _ = make_optimizer(cfg.training.optimizer_outer)    # AdamW
updates, opt_state = outer_tx.update(avg_grads_meta, opt_state, meta_model.trainable_parameters())
meta_model = filter_apply_updates(meta_model, updates)
```
- **Note**: the post-TTT state (inner-updated weights) is **discarded** at training time ("Do not return new state") — only the outer weights $W_0$ are persisted; the inner loop is re-run fresh each outer step.

### 8.2 The inner loop / meta-forward — `transformer.py: MetaModel.loss_for_sequence`

`/home/jayoo/.../e2e/ttt/model/transformer.py` (`MetaModel`, lines ~647-745).

Two modes set by `cfg.training.train_mode`:
- **`pretrain`** = full-attention / SWA baselines (no TTT). Just scans `lm_loss` over mini-batch windows carrying SWA KV-cache state. This is the *static* path.
- **`meta`** = TTT-E2E. Steps:
  1. Rebuild block collection as **`BlockCollectionSplit`** → splits `blocks` into `prefix_blocks` (first `L - suffix_len`) and `suffix_blocks` (last `suffix_len`). `suffix_len` = number of TTT'd blocks (e.g. 8 for 3B = 1/4 of 32; 6 for 760M).
  2. Embed tokens, run **prefix** through static blocks once (`prefix_call`, `eqx.filter_checkpoint`'d) → `prefix_output`. Prefix blocks have **no state** and gradients do *not* flow below the suffix (matches paper: backprop only through last L/4).
  3. Reshape seq + prefix_output into mini-batch **chunks** of `mini_batch_size` (=1024 = $b$).
  4. **`scan_remat_chunk`** over chunks: each step = `inner_loop_step`, carrying `(model, inner_opt_state, state)`.

```python
if cfg.training.train_mode == "meta":
    model = jax.tree.map(lambda p: p.astype(self.state_dtype), self)        # fp32 inner state
    inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())
    xt_embed = self.language_model.wte_call(seq.input_ids)
    prefix_output = eqx.filter_checkpoint(self.language_model.prefix_call)(
        self.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq).last_hidden_state

    def process_suffix_chunk(model__opt_state__state, inputs):
        model_inner, inner_opt_state, state_tuple = model__opt_state__state
        suffix_chunk, prefix_chunk = inputs
        # re-combine: inner params come from the *carried* (updated) model, outer params from the frozen W0
        spec_inner = get_filter_spec(model_inner, self.config.training.spec_inner, "inner parameters")
        inner_params, _ = eqx.partition(model_inner, spec_inner)
        _, outer_params = eqx.partition(model, spec_inner)
        model_inner = eqx.combine(inner_params, outer_params)
        new_model, inner_opt_state, state_tuple, metrics = MetaModel.inner_loop_step(
            model_inner, inner_opt_state, state_tuple, suffix_chunk, prefix_chunk)
        return (new_model, inner_opt_state, state_tuple), metrics

    seq = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    prefix_output = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    carry, metrics = scan_remat_chunk(eqx.filter_checkpoint(process_suffix_chunk, ...),
        (model, inner_opt_state, (state_all, state_suffix)), (seq, prefix_output),
        remat_n_loops=cfg.training.inner_remat_freq, unroll=cfg.model.unroll_inner_scan)
    loss = metrics[M.loss].mean()
```
- **`scan_remat_chunk`** = the **gradient-checkpointing-through-time** mechanism the paper flags as the latency cost (`inner_remat_freq` controls how often inner steps are rematerialized).
- Only **inner params** (the `feed_forward_prime` MLPs in suffix blocks) get carried/updated; **outer params** are spliced back from the frozen `model` ($W_0$) every chunk. This is exactly the inner/outer split.

### 8.3 Single inner step — `MetaModel.inner_loop_step` + `lm_loss`

```python
def inner_loop_step(self, opt_state, state_tuple, seq, prefix_outputs):
    state_all, suffix_state = state_tuple
    value_and_grad_fn = eqx.filter_value_and_grad(MetaModel.lm_loss, has_aux=True)
    (_loss, (md[loss], md[token_nll], new_suffix_state)), grads = value_and_grad_fn(
        self, seq, suffix_state, prefix_outputs=prefix_outputs)
    inner_grads = grads.inner_parameters()                                 # only prime MLPs
    updates, new_optimizer_state = self.inner_optimizer(state_all).update(
        inner_grads, opt_state, self.inner_parameters())
    new_model = filter_apply_updates(self, updates)                        # W_i = W_{i-1} - eta * grad
    return InnerLoopStepResult(new_model, new_optimizer_state, (state_all, new_suffix_state), md)
```
- `lm_loss` runs **only the suffix** (`suffix_call`) from the cached `prefix_outputs` → standard CE next-token loss (`cross_entropy_loss_and_accuracy`). The grad-of-`lm_loss` step is the inner SGD update; the outer `value_and_grad` over `loss_for_sequence` differentiates through *this whole loop* = the gradients-of-gradients.
- **Inner optimizer = plain SGD, no momentum** (`make_sgd_optimizer`, `optax.sgd(momentum=None)`), constant LR, optional grad clip. LR = `optimizer_inner.lr` (=1) scaled by `ilr_multiplier`.

### 8.4 Inner learning-rate warmup (meta-learned LR schedule) — not detailed in paper

`MetaModel.get_ilr_multiplier` / `inner_optimizer`:
```python
progress = min(1.0, (step+1)/ilr_warmup_steps)
ilr = ilr_init + (optimizer_inner.lr - ilr_init) * progress
ilr_multiplier = ilr / optimizer_inner.lr
```
- The **inner LR is warmed up over outer-loop training steps** (`ilr_warmup_steps`, e.g. 5200 for 3B pretrain = 10% of 52000). Starts at `ilr_init` (0.1 for pretrain, 1.0 for extension) and ramps to `optimizer_inner.lr` (=1). So early in *outer* training the inner steps are gentle. This detail is in code/config, not the paper body.
- For 3B pretrain: `optimizer_inner.lr=1`, `ilr_init=0.1`, `clip_gradient=1.0`. Extension: `ilr_warmup_steps=0`, `ilr_init=1`.

### 8.5 The two-MLP block ("prime" = inner, regular = static) — `transformer.py: Block`, `PrimeStorage`

- `Block.feed_forward_prime` = the **inner-updated** SwiGLU MLP (the TTT'd weights / the "hidden state"). `Block.feed_forward` = the **static safe-storage** MLP.
- Forward order in `Block.__call__`: SWA → residual → **prime MLP** (if present) → residual → **regular MLP** → residual. So prime sits *between* attention and the static MLP in suffix blocks.
- `PrimeStorage` holds `feed_forward_prime` + its norms, `jax.vmap`'d over `suffix_len`. Lives in the *permanent* model (`BlockCollection.prime_storage`) so there's a fixed place to store inner grads/state. `config.prime=True` enables it.
- `spec_inner = ["language_model.**.suffix_blocks.feed_forward_prime.**"]` ⇒ **only the prime MLPs of the suffix blocks are inner params**. Everything else (attention, norms, embeddings, regular MLP, prefix blocks) is frozen during TTT.
- **Matches paper §2.3.1**: TTT only MLP layers, only last 1/4 blocks, two MLPs per block.

### 8.6 Sliding-window attention — `attention.py: SWA` / `SWAFull`

`/home/jayoo/.../e2e/ttt/model/attention.py`

- Two SWA implementations:
  - **`SWAFull`** = one `dot_product_attention` call with `local_window_size=(window-1, 0)`, `is_causal=True`. Used for prefix blocks / `is_prefix=True` path (cuDNN flash). Standard SWA over the full sequence.
  - **`SWA`** = **chunked, stateful** SWA used inside the inner loop. Holds a **KV cache of size `window_size`** (`kv_cache_index` state) + a `chunk_index`. Each call: project current chunk's q,k,v (QK-norm), concat prev window-KV with new K/V, build a banded causal mask (`sw_causal_mask`), attend, then **slide the cache** keeping last `window_size` KV. RoPE applied with chunk-relative positions.

```python
class SWA(AttentionBase):
    def init_kv_cache(self):
        return (zeros((window_size, hidden_size)), zeros((window_size, hidden_size)))
    def sw_causal_mask(self, chunk_id):
        nk = window_size + mini_batch_size; nq = mini_batch_size
        qi = (arange(nq) + chunk_id*nq)[:, None]
        ki = (arange(-nk, 0) + chunk_id*nq + nq)[None, :]
        return (qi >= ki) & (qi < ki + window_size) & (ki >= 0)
    def __call__(self, hidden_states, seq, state, is_prefix=False):
        if is_prefix: return self.full_sw_attention(...)        # whole-seq flash SWA
        ... concat prev_k/v with new; new_kv_cache = last window_size ...
        attn_output = self.core_attention_op(xq, xk, xv, sw_causal_mask(chunk_id))
        state = state.set(kv_cache_index, new_kv_cache); state = state.set(chunk_index, chunk_id+1)
```
- This stateful chunked SWA is what lets the model **see local context within a mini-batch before TTT updates the prime MLPs** — the $k \ge b$ requirement (here $k=8\text{K} \ge b=1\text{K}$).
- `Attention` (full) and `SWAFull` use `jax.nn.dot_product_attention` with cuDNN flash when `force_flash`/`is_prefix`. **`force_flash` cannot be on for the TTT (`meta`) path** (configs set `force_flash: False` for e2e, `True` for full-attn baseline) — consistent with the paper's note that flash kernels don't support grad-of-grad.

### 8.7 Config / hyperparameters (`config.py` + YAMLs)

Defaults `ModelConfig`: `mini_batch_size=1024`, `sliding_window_size=1024` (overridden to 8192), `qk_norm=True`, pre+post norm, `state_dtype="fp32"`, `compute_dtype="bf16"`, `param_dtype="fp32"`, `suffix_len=0`, `prime=False`.

| | 3B e2e pretrain | 3B e2e ext-128K | 760M e2e pretrain |
|---|---|---|---|
| `seq_modeling_block` | SWA | SWA | SWA |
| `sliding_window_size` ($k$) | 8192 | 8192 | 8192 |
| `mini_batch_size` ($b$) | 1024 | 1024 | 1024 |
| `suffix_len` (TTT'd blocks) | 8 (of 32) | 8 | 6 (of 24) |
| `intermediate_size` | 5632 (shrunk from 6912) | 5632 | 3328 |
| `rope_theta` | 500000 | 500000 (changes by ctx) | 500000 |
| `prime` | True | True | True |
| inner opt | SGD lr=1, clip 1.0 | SGD lr=1, clip 1.0 | SGD lr=1, clip 1.0 |
| `ilr_init` | 0.1 | 1 | 0.1 |
| outer opt | AdamW lr=8e-4, b2=0.95, wd=0.1 | AdamW lr=4e-4 | (760m recipe) |
| `seq_length` | 8192 | 131072 | 8192 |
| `global_batch_size` | 128 | 16 | — |
| `total_steps` | 52000 | 1300 | — |
| `ilr_warmup_steps` | 5200 | 0 | — |

- **`intermediate_size` shrunk** (5632 vs base 6912 for 3B) ⇒ the paper's "reduce MLP hidden dim throughout to keep param count equal" given the extra prime MLP. Confirmed in code.
- **Full-attn baseline** (`pretrain-3b-fa.yaml`): `seq_modeling_block: self_attention`, `force_flash: True`, `mini_batch_size: 8192` (=no chunking), `train_mode: pretrain`, full-remat blocks. Same model config otherwise.

### 8.8 Training entry / data
- `pyproject.toml` exposes `train = "ttt.train:main"`. Launch via `uv run --exact train +deploy=interactive +experiment=.../pretrain-3b-e2e ...`. Hydra Submitit launcher for multi-node Slurm.
- Datasets: Llama-3-tokenized DCLM-filter-8k and Books3, in GCS buckets (`lm_dataset`, grain dataloader). Eval split `val`, no shuffle, one epoch.
- Released checkpoints: 125M/1B/3B pretrained (DCLM 8K) and fine-tuned (Books 8K/128K). Optimizer state not included ⇒ load with `training.load_part=params`.

### 8.9 Notable code details / deviations from paper
- **Inner LR warmup over outer steps** (`ilr_warmup_steps`, `ilr_init`) — a meta-LR schedule not described in the paper body.
- **Welford's online mean** for grad accumulation to keep the deep grad-of-grad graph in memory — an engineering detail behind the "training latency" limitation.
- **`scan_remat_chunk` / `inner_remat_freq`** = the gradient-checkpointing-through-time ($\log T$ factor) the paper cites as causing latency growth 8K→32K.
- **Post-TTT state discarded at train time** (only $W_0$ kept) — the inner loop is purely a meta-learning device during training; at test time the updated weights *are* the carried state.
- Inner state computed in **fp32** (`state_dtype`) even though compute is bf16 — for inner-loop stability.
- Code keeps a **`pretrain` mode** (no-TTT static scan) and a **`meta` mode** (TTT) in the same `MetaModel`, sharing the SWA chunking — so baselines and TTT-E2E run through one model class.
- `prefix_call` is `eqx.filter_checkpoint`'d and prefix blocks carry **no state**; gradients explicitly do not propagate below the suffix (`is_prefix=True` disconnects), matching "gradients pass through the last L/4 blocks but not further down" (Fig 3).
