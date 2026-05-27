# GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent

**Authors**: Kuratov, Kairov, Bulatov, Rodkin, Burtsev (Cognitive AI Systems Lab, MIRAI, MBZUAI, LIMS) — arXiv 2603.13875, March 2026.

## TL;DR

- **Compressive memory** alternative to KV-cache: read context once, store in small fixed-size memory, answer many queries from that state.
- WRITE = a few steps of **test-time gradient descent** on a small set of **prefix memory tokens** (model weights frozen), minimizing a self-supervised **context-reconstruction loss**.
- Memory init $\mathcal{M}_0$ and base model $\theta$ are **meta-learned** (outer loop) so that very few inner-loop steps ($K \leq 5$) suffice.
- Beats forward-only writers (RMT, ARMT) at the same memory size on KV-retrieval; transfers to bAbI / Short SQuAD / WikiText LM with pretrained GPT-2.

---

## 1. Problem Setup: Context Removal

- Each task instance = **(context $C$, query $Q$, target $Y$)**.
- Standard LM: $f_\theta(Y \mid [C; Q])$ — pay attention to full $C$ for every query.
- **Memory-augmented decomposition**:
    - **WRITE**: $\mathcal{M} = \mathcal{E}_\theta(C)$, where $\mathcal{M} \in \mathbb{R}^{m \times d}$ ($m$ prefix vectors, $d$ = model dim).
    - **READ**: $f_\theta(Y \mid \mathcal{M}, Q) \triangleq f_\theta(Y \mid [\mathcal{M}; Q])$.
- **Critical constraint**: at READ, $C$ is **removed**. All info needed for $Y$ must flow through $\mathcal{M}$.
- Memory size **independent of context length** (vs KV-cache size = $\text{layers} \times N \times d \times 2$).

---

## 2. GradMem Method

### Memory parameterization

- $m$ vectors of dim $d$, used as **prefix embeddings** prepended to model input.
- Shared meta-learned init $\mathcal{M}_0$; per-example WRITE produces $\mathcal{M}_K$.
- One model-level memory state (not per-layer like Mamba/ARMT).

### WRITE objective (self-supervised reconstruction)

$$\mathcal{L}_{\text{write}}(\mathcal{M}; C) = -\sum_{i=1}^{N} \log f_\theta(t_i \mid [\mathcal{M}; t_{<i}])$$

- Standard autoregressive cross-entropy over context tokens **with memory prepended**.
- Intuition: minimizing $\mathcal{L}_{\text{write}}$ forces $\mathcal{M}$ to encode info about $C$ that isn't predictable from prefix $t_{<i}$ alone (novel/high-entropy content).
- **Task-agnostic** — same objective across all downstream tasks.

### WRITE update (inner loop)

$$\mathcal{M}_{k+1} = \mathcal{M}_k - \alpha \nabla_{\mathcal{M}_k} \mathcal{L}_{\text{write}}(\mathcal{M}_k; C)$$

- $K$ steps of GD on memory only (model frozen during inner loop).
- $\hat{\mathcal{M}} \triangleq \mathcal{M}_K$; encoder = $\mathcal{E}_\theta(C) \triangleq \text{GD}_K(\mathcal{M}_0, \mathcal{L}_{\text{write}}(\cdot; C))$.
- Practical augmentations: gradient clipping, a learned linear layer applied to memory before/after updates, separate prediction heads (output embeddings) for WRITE vs READ phases.

### Meta-learning (outer loop)

- **Outer objective** = downstream task loss under context removal:
    $$\mathcal{L}_{\text{task}}(\hat{\mathcal{M}}, Q, Y) = -\log f_\theta(Y \mid \hat{\mathcal{M}}, Q)$$
- Backprop through WRITE optimization → **second-order gradients** (MAML-style, Finn et al. 2017).
- Outer-loop params: $\theta$ + $\mathcal{M}_0$ (+ memory projections / control tokens).
- **First-order approximation failed**; needed full second-order to reach strong performance.

### Comparison to TTT (Sun et al. 2025)

| | TTT layers | GradMem |
|---|---|---|
| **What is adapted** | Layer-specific $W_t$, per layer | Prefix memory tokens, single model-level state |
| **When** | Token-level online while processing | Whole context segment $C$, once per context |
| **Self-supervised loss** | Layer input/activation reconstruction | Context token reconstruction $\mathcal{L}_{\text{write}}$ |
| **Outer-loop obj** | Next-token LM | Downstream task with $C$ removed at READ |

- TTT spreads adaptation across all layers and timesteps; GradMem concentrates all test-time adaptation into **one memory state at model input**.

---

## 3. Experimental Setup

### Datasets

- **Associative KV-retrieval** (main synthetic benchmark): context = list of $N$ key-value pairs ($k_i$, $v_i$) from a 62-char vocabulary; query asks for $v_j$ given $k_j$. Controllable info density.
- **bAbI QA1–QA5**: story → question → short answer. Tests multi-sentence reasoning.
- **Short SQuAD**: extracted to the sentence containing the answer span (isolates writing mechanism from long-context distractors).
- **WikiText-103 LM**: 256-tok chunks split into 128-tok context + 128-tok target; predict continuation from memory only.

### Baselines

- **Full-attention transformer**: upper bound (uncompressed KV-cache). 4-layer Llama for AR; GPT-2 / Pythia for downstream.
- **Mamba-2 (130M)**: SSM with recurrent state.
- **RMT** (Bulatov '22): forward-only memory write via segments. 2-segment setup = equivalent to GradMem **except WRITE rule is forward pass only**.
- **ARMT** (Rodkin '24): per-layer associative matrix memory with DeltaNet-style update; forward-only.

### Hyperparameters

- $d_{\text{mem}} = 64$ in ARMT; otherwise = model hidden size.
- $m$: 8 for AR & bAbI; 32 for SQuAD & LM.
- Inner LR $\alpha = 0.4$ default; tuned per $K$ on KV-retrieval.
- AR: curriculum from 32 KV-pairs, models trained from scratch (4-layer Llama, 128 hidden).
- bAbI/SQuAD/LM: fine-tuned from pretrained checkpoints.

---

## 4. Key Results

### 4.1 KV-retrieval (Table 1)

- **GradMem ($m=8$, $K=5$)**: 99.9% @ 32 pairs, 99.1% @ 64, **88.4% @ 96**.
- **RMT ($m=8$, forward-only)**: 45.5% @ 16, 12.9% @ 96 — collapses.
- **Repeating forward writes** (RMT x2–x5) gives inconsistent gains (e.g. drops from 45→18 then back up); **gradient steps consistently improve**.
- **Without meta-learning** (first-order MAML): collapses to 3% @ 32 — second-order backprop through WRITE is essential.
- **Per-layer recurrent state** (Mamba, ARMT) is a different memory interface, included as a reference point — not directly comparable to single model-level state.

### 4.2 Inference-time compute scaling (Fig 4)

- Training with $K_{\text{train}} \leq 2$ but **evaluating with $K_{\text{eval}} \gg K_{\text{train}}$** extrapolates: e.g. $K_{\text{train}}=2$ on 96-pair KV-retrieval → 31.8% @ $K_{\text{eval}}=2$, **95.8% @ $K_{\text{eval}}=30$**.
- Avoids costly long-trajectory meta-training; shifts cost to inference where it can be amortized.
- **Inner-loss ↔ Exact-Match correlation**: $r \approx -0.83$. Lower reconstruction loss → better task accuracy.

### 4.3 Selective compression (Appendix E, Fig 8)

- On KV-retrieval, decomposing inner loss into key-token vs value-token contributions:
    - **Key loss** stays roughly flat across WRITE iterations.
    - **Value loss** drops sharply with $K_{\text{eval}}$.
- → Memory learns to be **selective**: keys are retrieval cues (just need to be discriminable), values must be reproduced verbatim at READ.
- Despite the WRITE objective treating both equally, the meta-learned encoder allocates capacity toward answer-relevant content.

### 4.4 Pretrained LM transfer (Table 2)

- **GradMem (GPT-2, K=1)**: bAbI QA1=100, QA2=94.2, QA3=80.0, QA4=100, QA5=99.2; SQuAD short EM=38.1; WikiText LM CE=2.92.
- Matches or beats forward-only RMT on all bAbI tasks except QA3 (highest info density).
- Falls behind Mamba on QA2/QA3 (Mamba's recurrent state was pretrained heavily); falls behind full GPT-2 upper bound on SQuAD (64.2) — there's headroom.
- **Larger $K$ helps QA3** (compression-hard task).

---

## 5. Compute & Implementation

### Compute trade-off (Appendix D)

- $T_{\text{full}} \approx c^2 + cqN$ (process context once + cross-attend per query).
- $T_{\text{GradMem}} \approx R(c+m)^2 K + m^2 + mqN$ where $R$ = ratio of update-step cost to forward cost.
- **Break-even**: $N \gtrsim c(RK-1)/q$ — favorable when context $c \gg m$ and many queries $N$ reuse the same context.
- E.g. for $c=4096, m=32, q=128$: GradMem wins after ~50 queries per context.

### Double-backward implementation (Appendix C)

- **Main cost**: backprop through inner-loop WRITE optimization = backward-over-backward through attention.
- FlashAttention / SDPA support only first-order backward → custom kernels needed.
- Best approach: **Flash HVP** (fused fwd/bwd kernels + analytical double backward); for $L=1024$, drops backward latency from ~1000ms to ~600ms, peak GPU mem from ~60GB → ~30GB.

---

## 6. Contributions & Takeaways

1. **Gradient-based context memorization**: explicit model-level WRITE objective, no specialized per-layer update rule.
2. **Few-step writing**: meta-learning makes $K \leq 5$ steps sufficient — practical at inference.
3. **Gradient WRITE > forward-only WRITE** at same memory size; iteratively self-corrects.
4. **Capacity scales with $K$ and transfers to natural language** with the same task-agnostic reconstruction objective.

### Strengths

- Clean separation: weights frozen, only memory adapts → portable compressed representation.
- Loss-driven write naturally prioritizes high-entropy/novel content (where reconstruction fails).
- Inference-time $K$ scaling = compute-accuracy knob without retraining.

### Limitations / open directions

- Double-backward through attention is expensive; constrains scale.
- First-order approximations (iMAML, Reptile) might help training efficiency.
- Plain reconstruction is unlikely optimal — task-aware self-supervised WRITE objectives could improve transfer.
- Comparison to per-layer recurrent state methods (Mamba, ARMT) is apples-to-oranges; would be nice to see GradMem applied to per-layer states.

---

## 7. Connections

- **Test-time training** (Sun '25): GradMem = TTT but at model input, once per context, with reconstruction of context tokens (not layer activations).
- **Cartridges** (Eyuboglu '25): also compresses into prefix tokens via self-study, but uses many iterations of standard optimization (not meta-learned few-step).
- **Kuratov '25 (cramming 1568 tokens)**: lossless compression via embedding optimization needs hundreds–thousands of steps; GradMem trades exact reconstruction for few-step compression by meta-learning the encoder.
- **RMT / ARMT / Mamba**: forward-only update rules — GradMem's framing argues these are equivalent to a fixed forward computation, while gradient writes are strictly more expressive (can correct earlier errors, allocate compute to specific context).
- **iMAML / Reptile**: alternative first-order/implicit meta-learning that could replace MAML for the outer loop.
- **Prefix tuning** (Li & Liang '21): static prefix learned per task; GradMem learns prefix init that is then adapted **per example** at test time.

---

## 8. Implementation Details (from `gradmem/` codebase)

Repo layout: `grad_memgpt.py` (core model + inner loop), `rmt.py` (forward-only baseline), `run_gradmemgpt_on_*.py` (per-task entry points using HF Trainer), `attn_double_bwd/` (custom attention kernels for second-order backprop), `kv_dataset_utils.py` / `squad_utils.py` / `prepare_pg19_chunks.py` (data).

### 8.1 Model class: `GradMemGPT`

`GradMemGPT` (`grad_memgpt.py:134`) is a `PreTrainedModel` wrapping any HF `AutoModelForCausalLM`. Backbone is fetched generically via `get_backbone()` so the same code handles GPT-2 / Llama / Pythia.

All meta-learned parameters live as `nn.Parameter` on the wrapper, not inside `self.model` (`grad_memgpt.py:223–256`):

```python
# memory parameters (shape = n_mem_tokens × d)
n_embd = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
# self.mem are inner loop per-sample params, intial states of mem (self.mem) are meta-learned
self.mem = nn.Parameter(torch.randn(self.n_mem_tokens, n_embd) * 0.02)

# optional mem projection linear layer
if self.mem_proj_mode != "none":
    self.mem_proj = nn.Linear(n_embd, n_embd, bias=True)
    with torch.no_grad():                       # initialize as identity
        nn.init.eye_(self.mem_proj.weight)
        self.mem_proj.bias.zero_()

# optional read/write control parameters (shape = n_ctrl_tokens × d)
if self.n_ctrl_tokens > 0:
    self.write_st  = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
    self.write_end = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
    self.read_st   = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)
    self.read_end  = nn.Parameter(torch.randn(self.n_ctrl_tokens, n_embd) * 0.02)

if self.use_write_head:
    self.write_head = nn.Linear(n_embd, self.model.config.vocab_size, bias=False)
    # init from output embeddings (then trained independently)
    head_params = self.model.get_output_embeddings().weight
    with torch.no_grad():
        self.write_head.weight.copy_(head_params.detach())
```

- `self.mem` = the meta-learned memory init $\mathcal{M}_0$; per-sample $\mathcal{M}_K$ is computed on the fly and never stored.
- `self.mem_proj` = the "learned linear layer applied to memory" augmentation (Appendix B). Identity init means it starts as a no-op.
- `write_st/end`, `read_st/end` = phase-specific control tokens that let the model tell WRITE vs READ apart.
- `self.write_head` = separate LM head used only during WRITE (the "separate prediction heads" augmentation).

WRITE-only LoRA (`use_write_lora`, `:320`): adapters are enabled inside the inner loop and **disabled at READ** via a context manager:

```python
# grad_memgpt.py:348
@contextmanager
def _disable_write_lora(self):
    if not self.use_write_lora:
        yield
        return
    self.model.disable_adapter_layers()
    try:
        yield
    finally:
        self.model.enable_adapter_layers()
```

### 8.2 Input layout (matches paper Fig 2c)

`forward()` (`:441`) consumes a dict `input_ids = {'context_input_ids', 'query_input_ids'}` built by the per-task `collate_fn`. The two phases assemble different prefix sequences:

```
WRITE: [write_st] [mem] [write_end] [context tokens]      # length n_ctrl + M + n_ctrl + S
READ : [read_st]  [mem] [read_end]  [query tokens] [target]
```

Where `mem_offset = n_mem_tokens + 2 * n_ctrl_tokens` (`:458`). The query and target are pre-concatenated by the data collator (see `run_gradmemgpt_on_kv_retrieval.py:38`) with `labels` set to `-100` everywhere except the target tokens.

### 8.3 Inner loop (WRITE phase) — `grad_memgpt.py:495–593`

The paper's equation $\mathcal{M}_{k+1} = \mathcal{M}_k - \alpha \nabla_{\mathcal{M}_k} \mathcal{L}_{\text{write}}$ is implemented as a **manual differentiable optimization loop** (the model is **never updated in-place**):

1. **Per-sample memory copy** (`:468`):
    ```python
    mem_batch = self.mem.unsqueeze(0).expand(B, -1, -1).clone()  # [B, M, d]
    mem_batch = mem_batch.requires_grad_(True)
    ```
    Each example in the batch gets its own copy of $\mathcal{M}_0$. This is what makes the WRITE update *per-sample* — exactly what the paper means by inner-loop variables.

2. **Loop over K steps** (`:515–593`). Core body:
    ```python
    for k in range(self.K):
        # optional projection: identity-init Linear, or per-sample fast weights via _apply_linear (baddbmm)
        if   self.mem_proj_mode == 'none':       mem_inp = mem_batch
        elif self.mem_proj_mode == 'proj':       mem_inp = self.mem_proj(mem_batch)
        else:                                    mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

        # WRITE-phase prefix: [write_st][mem][write_end][context]
        if self.n_ctrl_tokens > 0:
            x_ctx = torch.cat([write_st_batch, mem_inp, write_end_batch, ctx_emb], dim=1)
        else:
            x_ctx = torch.cat([mem_inp, ctx_emb], dim=1)               # [B, M+S, d]

        outs   = self.model(inputs_embeds=x_ctx, return_dict=True)
        logits = outs.logits[:, mem_offset-1:, :]                       # [B, S, V]
        # ^ last memory/ctrl token predicts the first context token → autoregressive
        #   reconstruction  log f_theta(t_i | [M; t_<i])  from Eq. 4.
    ```

3. **Per-sample reconstruction loss** (`:542–551`) — the loss equivalent of Eq. 4:
    ```python
    inner_loss = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        lm_labels.reshape(-1),
        ignore_index=-100,
        reduction='none',
    ).view(B, -1)
    # per_sample losses, make per-sample inner loss invariant to batch size B:
    # g_i = d inner_loss / d mem_i = d inner_loss_i / d mem_i
    inner_loss = (inner_loss * mask).sum(1) / seq_len                  # per-sample mean
    inner_loss = inner_loss.sum()                                       # sum over batch
    ```
    The `.sum()` of per-sample-mean losses is the trick that makes $\nabla_{\mathcal{M}_i} \mathcal{L}_{\text{inner}}$ depend only on sample $i$'s own context — `mem_batch[i]` is independent of `mem_batch[j]`, so the batch-sum and per-sample gradient agree.

4. **Differentiable gradient step** (`:554–584`):
    ```python
    is_second_order_step = (self.grad_mode == "second") and (k >= (self.K - self.last_K_second_order))
    create_graph = is_second_order_step
    retain_graph = create_graph or (self.add_inner_loss_to_outer and (k == self.K - 1))

    g_mem = torch.autograd.grad(inner_loss, mem_batch,
                                create_graph=create_graph, retain_graph=retain_graph)[0]

    mem_batch = self._sgd_step(mem_batch, g_mem,
                               clip_value=self.inner_clip_value,
                               clip_norm=self.inner_clip_norm)
    ```
    `_sgd_step()` returns `p - lr * g` as a **new tensor** (functional, never in-place), with optional element-wise and per-sample 2-norm clipping (`grad_memgpt.py:401`):
    ```python
    if clip_norm is not None:
        reduce_dims = tuple(range(1, g.ndim))             # all non-batch dims
        g_norm = g.norm(dim=reduce_dims, keepdim=True)    # (B,1,1)
        scale  = clip_norm / (g_norm + 1e-6)
        g = torch.where(g_norm > clip_norm, g * scale, g)
    return p - lr * g
    ```
    There's also a functional Adam (`:359`) but it explicitly cuts the meta-graph (`torch.no_grad()` around moment math) and is marked untested.

5. **Gradient mode switch** (`:586–592`):
    ```python
    if self.grad_mode in ['none']:
        mem_batch = mem_batch.detach().requires_grad_(True)            # cut meta-graph
    elif self.grad_mode in ['first', 'second']:
        pass                                                            # let gradients flow
    ```
    - `grad_mode="none"` → meta-gradient cut, ablation baseline.
    - `grad_mode="first"` → `create_graph=False` throughout: first-order MAML / Reptile-style approximation.
    - `grad_mode="second"` → `create_graph = (k >= K - last_K_second_order)` (see the inline in step 4): full MAML on the last `last_K_second_order` steps only. This is the "second-order is what makes GradMem learn" finding from the README, matching the paper's "first-order approximations are insufficient" claim (Sec 3.3). The `last_K_second_order` knob lets you train with e.g. `K=5` but only retain the meta-graph through the final 1–2 steps to cap memory.

### 8.4 Outer loop (READ phase) — `grad_memgpt.py:614–663`

```python
qry_emb = self.model.get_input_embeddings()(query_input_ids)           # [B, Q, d]

# project memory (same modes as WRITE), then build READ prefix
if   self.mem_proj_mode == "none":       mem_inp = mem_batch
elif self.mem_proj_mode == "proj":       mem_inp = self.mem_proj(mem_batch)
else:                                    mem_inp = self._apply_linear(mem_batch, W_batch, b_batch)

if self.n_ctrl_tokens > 0:
    x_qry = torch.cat([read_st_batch, mem_inp, read_end_batch, qry_emb], dim=1)
else:
    x_qry = torch.cat([mem_inp, qry_emb], dim=1)                       # [B, M+Q, d]

with self._disable_write_lora():                                        # READ uses base model only
    logits_q = self.model(inputs_embeds=x_qry).logits                  # [B, M+Q, V]
logits_q = logits_q[:, mem_offset-1:mem_offset+qry_emb.size(1), :]     # [B, Q+1, V]

target_loss = nn.functional.cross_entropy(
    logits_q[:, :-1].reshape(-1, logits_q.size(-1)),
    labels.reshape(-1),
    ignore_index=-100,
)
combined_loss = target_loss + self.inner_loss_weight * (inner_loss / B) \
                if self.add_inner_loss_to_outer else target_loss
output['loss'] = combined_loss
```

- Uses the **final** post-inner-loop `mem_batch`.
- Logits sliced so the last memory/ctrl token predicts the first query token, mirroring the WRITE slicing.
- WRITE-only LoRA disabled here so the READ path uses only the base + meta-learned read-time parameters.
- Labels were masked to `-100` everywhere except the target span in the data collator (enforces context removal).
- The HF `Trainer.step()` calls `loss.backward()` on `combined_loss`. Because the inner-loop autograd graph is retained when `grad_mode="second"`, this single backward call produces the **second-order MAML gradient** wrt $\theta$, $\mathcal{M}_0$, `mem_proj`, control tokens, write head, and write-LoRA — no explicit Hessian computation needed.

### 8.5 Custom attention kernels — `attn_double_bwd/`

The bottleneck in MAML training is **backward-over-backward through self-attention**: standard FlashAttention/SDPA only expose first-order gradients. The repo registers four custom HF `AttentionInterface` implementations (`attn_double_bwd/__init__.py`):

```python
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

AttentionInterface.register("jvp_flash",      jvp_flash);       AttentionMaskInterface.register("jvp_flash",      sdpa_mask)
AttentionInterface.register("hvp_manual",     hvp_manual);      AttentionMaskInterface.register("hvp_manual",     sdpa_mask)
AttentionInterface.register("hvp_semi_manual", hvp_semi_manual); AttentionMaskInterface.register("hvp_semi_manual", sdpa_mask)
```

- `jvp_flash` — fast forward + JVP-based backward (Appendix C's "Fast forward → autograd": rebuilds attention forward inside backward so autograd can differentiate again).
- `hvp_manual` — fully analytical fwd / bwd / double-bwd in pure PyTorch (Appendix C's "Manual HVP"). `SDPA_FullManualBwd` (`hvp_manual.py:8`) implements the analytical second-order with closed-form `bar_dS`, `bar_dP`, `bar_scores` terms. E.g. the second-order backward computes (`hvp_manual.py:70–80`):
    ```python
    bar_dS  = (grad_q @ k.transpose(-2, -1) + q @ grad_k.transpose(-2, -1)) * scale
    beta    = (bar_dS * P).sum(dim=-1, keepdim=True)
    bar_dP  = P * (bar_dS - beta)
    bar_P   = grad_out @ grad_v.transpose(-2, -1) + bar_dS * (dP - Ssm) - dP * beta
    tmp     = (bar_P * P).sum(dim=-1, keepdim=True)
    bar_scores = (bar_P - tmp) * P
    dgrad_grad_out = P @ grad_v + bar_dP @ v
    dgrad_q        = (dS @ grad_k + bar_scores @ k) * scale
    dgrad_k        = (dS.transpose(-2, -1) @ grad_q + bar_scores.transpose(-2, -1) @ q) * scale
    ```
    Closed-form chain rule through softmax-attention twice, avoiding materializing the full second-order graph in autograd.
- `hvp_semi_manual` — fused fwd/bwd + analytical double-bwd (Appendix C's "Flash HVP", "the best balanced approach").
- `hvp_triton` — Triton-kernel variant.

Selection is via `--attn_implementation` (default `"eager"`). The forward pads context length to a multiple of 32 for these kernels (`grad_memgpt.py:509`):
```python
if self.attn_implementation in ('jvp_flash', 'hvp_semi_manual'):
    pad_list = [0, -(ctx_emb.size(1) + mem_offset) % 32, 0, 0]
    mask      = F.pad(mask,      pad_list, "constant", 0)
    lm_labels = F.pad(lm_labels, pad_list, "constant", -100)
    ctx_emb   = F.pad(ctx_emb,   [0, 0] + pad_list, "constant", 0)
```

### 8.6 Training entry point — `run_gradmemgpt_on_kv_retrieval.py`

Uses HF `Trainer` with a `CustomTrainer` subclass:

- **Model creation** (`:283`): either loads a pretrained HF checkpoint (`--pretrained_model gpt2`/`pythia`/etc.) or builds a small from-scratch config (`--base_model llama --n_layer 4 --n_head 4 --n_embd 128` matches the paper's KV-retrieval setup).
- **Data collator** (`:36`) produces `context_input_ids` and `query_input_ids` separately. Crucially, it builds `labels_mask` by walking the `offset_mapping` **from the end backward** to find the target span — this enforces the context-removal constraint by zeroing out all non-target positions in `labels`:
    ```python
    # run_gradmemgpt_on_kv_retrieval.py:50–67
    labels_mask = torch.zeros_like(query_input_ids)
    for i, item in enumerate(batch):
        query_seq_len  = len(item['query'])
        target_seq_len = len(item['target'])
        target_st, target_end = query_seq_len, query_seq_len + target_seq_len
        in_target = False
        for j in range(len(offsets_mapping[i]) - 1, -1, -1):
            st, end = offsets_mapping[i][j]
            if st < target_end and end > target_st:
                labels_mask[i, j] = 1
                in_target = True
            elif in_target:
                break
    labels = query_input_ids * labels_mask + (1 - labels_mask) * -100
    return {
        'input_ids': {
            'context_input_ids': context_input_ids,
            'query_input_ids':   query_input_ids,
        },
        'labels': labels,
    }
    ```
- **TrainingArguments**: `eval_on_start=True`, constant-with-warmup LR, `EarlyStoppingCallback`, and a custom `StopOnMetricValue` that stops training as soon as `eval_exact_match == 1.0`:
    ```python
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
               StopOnMetricValue(metric_name='exact_match', value=1.0, higher_is_better=True)]
    ```
- **Metrics** (`compute_metrics_fn`, `:83`): per-sample exact match on target tokens, ignoring the syntactic tokens `!` and `|`:
    ```python
    mask = (labels != -100)
    for t_id in ignore_token_ids:                          # ['!', '|']
        mask &= (labels != t_id)
    exact_match = np.mean([np.all(pred[mask[i]] == lab[mask[i]])
                           for i, (pred, lab) in enumerate(zip(preds, labels))
                           if np.any(mask[i])])
    ```
- The `preprocess_logits_for_metrics` hook forwards both predictions and `inner_loop_stats` (mem norm, delta-mem norm, inner grad norm, inner loss) so all the diagnostic plots from the paper can be reproduced from trainer logs.

### 8.7 RMT baseline — `rmt.py`

Same wrapper shape, but the WRITE update is a **forward pass**, not a gradient step. The whole inner loop reduces to (`rmt.py:220–278`):

```python
for k in range(num_loops):
    mem_inp_write = self.mem_proj(mem_batch) if self.mem_proj_mode in ["proj","proj_rw"] else mem_batch

    # [write_st][mem][write_end][context][write_st][mem]
    #                                            ^^^^^ trailing mem slot = new mem state
    if self.n_ctrl_tokens > 0:
        x_ctx = torch.cat([write_st_batch, mem_inp_write, write_end_batch,
                           ctx_emb,
                           write_st_batch, mem_inp_write], dim=1)
        attn_mask = torch.cat([ctrl_mask, mem_mask, ctrl_mask, ctx_mask, ctrl_mask, mem_mask], dim=1)
    else:
        x_ctx = torch.cat([mem_inp_write, ctx_emb, mem_inp_write], dim=1)  # [B, M+S+M, d]

    outs = get_backbone(self.model)(inputs_embeds=x_ctx, ...)
    h    = outs.hidden_states[-1]
    mem_out = h[:, -self.n_mem_tokens:]                # take trailing M tokens as new memory
    mem_batch = mem_out                                # carry over to next iteration
```

- The **trailing memory slot** is what the model "writes into": its hidden state after the forward pass becomes the next memory.
- Repeating this for `K` steps is the "RMT x2/x3/x5" experiments in Table 1 — the code confirms it's just **re-reading the same context** with the same model, not optimization.
- Optional `use_reconstruction_loss` adds a final segment with the **same** context-reconstruction CE used by GradMem, but applied to the read-out memory rather than driving an update rule.
- This makes the RMT-vs-GradMem comparison genuinely apples-to-apples: identical architecture, identical objective for the recon variant, only the WRITE *update rule* differs.

### 8.8 What the code reveals beyond the paper

- **Per-sample loss summation** trick: `inner_loss` uses `reduction='none'` then `.sum()` of per-sample means — necessary for correct per-sample MAML gradients; the paper waves this off as "per-sample optimization."
- **`last_K_second_order`** is an explicit knob (not documented prominently in the paper) — you can train with `K=5` inner steps but only retain the meta-graph through the last 1–2, the practical compute-memory escape hatch making large-$K$ training feasible:
    ```python
    # grad_memgpt.py:554
    is_second_order_step = (self.grad_mode == "second") and (k >= (self.K - self.last_K_second_order))
    create_graph = is_second_order_step
    ```
- **`mem_proj_mode="per_sample"`** (`grad_memgpt.py:472–486`) treats $W, b$ as per-sample fast weights themselves updated in the inner loop, with meta-learned initial values:
    ```python
    if self.mem_proj_mode == "per_sample":
        W_batch = self.mem_proj.weight.unsqueeze(0).expand(B, -1, -1).clone()  # [B, d, d]
        b_batch = self.mem_proj.bias.unsqueeze(0).expand(B, -1).clone()         # [B, d]
        W_batch = W_batch.requires_grad_(True)
        b_batch = b_batch.requires_grad_(True)
    # ... and in the inner step:
    g_mem, g_W, g_b = torch.autograd.grad(inner_loss, [mem_batch, W_batch, b_batch],
                                          create_graph=create_graph, retain_graph=retain_graph)
    ```
    The docstring (`:153`) lays out the interpretation: when $W$ is meta-learned but fixed across the inner loop, the resulting update `mem* = W(mem - α W^T grad) + b = (Wmem + b) - α(WW^T)grad` means $WW^\top$ is a **learned gradient preconditioner** — diagonal $W$ = per-dim LR, dense $W$ = full mixing. Connects to Meta-SGD / WarpGrad, though the paper doesn't make this comparison.
- **Always-on grad in WRITE** (`:497`): the inner loop runs under `torch.enable_grad()` regardless of outer context:
    ```python
    if self.K and context_input_ids.ne(pad_id).any():
        with torch.enable_grad():                  # re-enable autograd even if outer is no_grad
            ctx_emb = self.model.get_input_embeddings()(context_input_ids)
            ...
    ```
    So WRITE always builds a graph even during evaluation (when HF Trainer wraps the forward in `torch.no_grad()`). This is what enables the $K_{\text{eval}} \gg K_{\text{train}}$ inference-time scaling experiments in Sec 3.4 — you can crank `K_eval` at test time with zero training changes.

