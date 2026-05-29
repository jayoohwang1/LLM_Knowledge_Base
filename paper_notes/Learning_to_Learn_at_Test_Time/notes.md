# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

> arXiv 2407.04620 (Sun, Li, Dalal, et al. — Stanford / UCSD / UC Berkeley / Meta). Introduces **TTT layers**, instantiated as **TTT-Linear** & **TTT-MLP**.

---

## TL;DR / Core Thesis

- **RNN bottleneck**: RNNs are $O(1)$ per token but compress all context into a **fixed-size hidden state** $s_t$ → expressive power capped. Self-attention is expressive (KV cache = lossless, growing hidden state) but $O(t)$ per token, $O(T^2)$ total.
- **Key idea**: make the hidden state **itself a model** $f$ with weights $W_t$; make the **update rule a step of self-supervised learning (SGD)** on $W$. Output rule is a forward pass of $f$.
  - $\Rightarrow$ updating hidden state on a test sequence = **training $f$ at test time** → **Test-Time Training (TTT) layer**.
- Compression heuristic insight: **self-supervised learning already compresses huge datasets into model weights** (LLMs compress the internet via next-token pred). Use that as the RNN's "what to remember/forget" heuristic. $W$ remembers inputs with **large gradients** (those that make $W$ learn a lot).
- Linear complexity like RNNs, but expressive hidden state. Scales 125M–1.3B; matches Mamba @2k, **beats Mamba @8k+**, keeps reducing perplexity with more context (Mamba plateaus after 16k).

---

## 1. Unified view of sequence layers (Fig 1, 3)

All seq layers = **initial state + update rule + output rule**:

| Layer | Initial state | Update rule | Output rule | Cost/token |
|---|---|---|---|---|
| **Naive RNN** | $s_0 = \text{vector}()$ | $s_t = \sigma(\theta_{ss}s_{t-1} + \theta_{sx}x_t)$ | $z_t = \theta_{zs}s_t + \theta_{zx}x_t$ | $O(1)$ |
| **Self-attention** | $s_0 = \text{list}()$ | $s_t = s_{t-1}.\text{append}(k_t,v_t)$ | $z_t = V_t\,\text{softmax}(K_t^\top q_t)$ | $O(t)$ (grows) |
| **Naive TTT** | $W_0 = f.\text{params}()$ | $W_t = W_{t-1} - \eta\nabla\ell(W_{t-1};x_t)$ | $z_t = f(x_t; W_t)$ | $O(1)$ |

- RNN & TTT compress growing context into **fixed-size** state → constant cost. Attention's state grows.

---

## 2. Method

### 2.1 TTT as updating a hidden state
- Hidden state $s_t \equiv W_t$ = weights of model $f$ (linear model, MLP, anything).
- **Output rule**: $z_t = f(x_t; W_t)$ — output token is $f$'s prediction on $x_t$ with current weights. **(Eq 1)**
- **Update rule** = one GD step on self-supervised loss $\ell$:
  $$W_t = W_{t-1} - \eta\,\nabla\ell(W_{t-1}; x_t) \quad\text{(Eq 2)}$$
- **Naive self-supervised task** = reconstruct a corrupted input $\tilde x_t$:
  $$\ell(W; x_t) = \lVert f(\tilde x_t; W) - x_t\rVert^2 \quad\text{(Eq 3)}$$
  - Like a **denoising autoencoder** [Vincent]: $f$ must learn correlations among dims of $x_t$ to reconstruct from partial info.
  - Empirically GD reduces $\ell$ but not to zero (Fig 4).
- Even at "test time" we train a fresh weight sequence $W_1,\dots,W_T$ per input sequence → hence the name.
- Considered & rejected: adding a decoder $g$ (reconstruct via $g\circ f$) — slightly better but unstable + costly. Stick to **encoder-only**.

### 2.2 Training the network (outer loop)
- TTT layer's forward pass is all **differentiable operators** + the gradient operator $\nabla$ (which itself maps $\ell\mapsto\nabla\ell$, also differentiable).
- Backprop through it = **gradients of gradients** (standard meta-learning trick).
- **Two nested loops**:
  - **Inner loop** = TTT itself; trains $W$ (params of $f$); grad w.r.t. $W$. "One level below" regular training.
  - **Outer loop** = train rest of network $\theta_{\text{rest}}$ (+ TTT's outer params) via next-token prediction; grad w.r.t. $\theta$.
- TTT layer has same interface as RNN/attention → drop into any architecture, optimize end-to-end.

### 2.3 Learning the self-supervised task (multi-view reconstruction)
- The SSL task **determines what features $W$ learns** → don't handcraft, **learn it in the outer loop**.
- Corruption = **low-rank projections** (learnable matrices, outer-loop params), terminology from multi-view (MoCo-style):
  - **Training view** $\theta_K x_t$ (the corrupted input, $K$ hints at "Key").
  - **Label view** $\theta_V x_t$ (reconstruction target; maybe not all of $x_t$ worth remembering).
  - **Test view** $\theta_Q x_t$ (input at output time; $Q$ = "Query").
- **Learnable inner loss**:
  $$\ell(W; x_t) = \lVert f(\theta_K x_t; W) - \theta_V x_t\rVert^2 \quad\text{(Eq 4)}$$
- **New output rule** (test view, since training view is lower-dim):
  $$z_t = f(\theta_Q x_t; W_t) \quad\text{(Eq 5)}$$
- $\theta_K,\theta_V,\theta_Q$ are **outer-loop "hyper-parameters" of $\ell$**; $W$ is the inner-loop variable. Direct analogy: $\theta_K,\theta_V,\theta_Q$ ↔ self-attention's K/V/Q projections.
- Family of multi-view tasks; outer loop **selects** the task. All views = linear projections here (future: richer).

### 2.7 Implementation details
- **Instantiations of $f$** (differ only in $f$):
  - **TTT-Linear**: $f_{\text{lin}}(x) = Wx$, $W$ square.
  - **TTT-MLP**: $f_{\text{MLP}}$ = 2-layer MLP, hidden dim $4\times$ input, GELU between (like Transformer FFN).
  - **Both** wrap with LN + residual for stability: $f(x) = x + \text{LN}(f_{\text{res}}(x))$.
- **Learnable $W_0$**: instead of $W_0=0$, learn the **shared init** $\theta_{\text{init}}=W_0$ as outer-loop param (negligible param count; improves training stability a lot).
- **Learnable $\eta$**: inner LR is the key hyperparam → make it a learnable, **token-dependent** gate:
  $$\eta(x) = \eta_{\text{base}}\,\sigma(\theta_{\text{lr}}\cdot x)$$
  - $\theta_{\text{lr}}$ outer-loop vector, $\sigma$ sigmoid; $\eta_{\text{base}}=1$ (TTT-Linear), $0.1$ (TTT-MLP). Interpretable as a **gate on $\nabla\ell$**.
- **Backbone**: cleanest is to replace attention in a Transformer. But Mamba/Griffin use a different backbone with **temporal convs before** the seq layer (collect local info). Adopting the **Mamba backbone** improves TTT perplexity → used by default.

---

## 3. Efficiency: making TTT fast on GPUs/TPUs

> Naive TTT is already FLOP-efficient ($O(1)$/token) but wall-clock-slow: sequential dependency + few matmuls. Two techniques fix it.

### 2.4 Mini-batch TTT (parallelization)
- Naive update $W_t = W_{t-1} - \eta\nabla\ell(W_{t-1};x_t)$ is **online GD** ($G_t = \nabla\ell(W_{t-1};x_t)$) — strictly sequential.
- General GD: $W_t = W_0 - \eta\sum_{s=1}^t G_s$ → once all $G_s$ known, get all $W_t$ via a **cumsum**.
- Choice of descent direction $G_t$ defines the variant:
  - **Online GD**: $G_t = \nabla\ell(W_{t-1};x_t)$ (fully sequential).
  - **Batch GD**: $G_t = \nabla\ell(W_0;x_t)$ — all w.r.t. $W_0$ → parallel, but $W_t$ is only **one step from $W_0$** → smaller search space → worse LM perplexity.
  - **Mini-batch GD (proposed)**: $G_t = \nabla\ell(W_{t'};x_t)$, $t' = t - \text{mod}(t,b)$ = last step of **previous** mini-batch. Parallelize $b$ grad computations at once.
- **TTT mini-batch size $b$** trades speed vs quality (Fig 7). **Chose $b=16$** for all experiments.
  - Smaller $b$ → more GD steps → better perplexity but slower.
- Two info channels $W_s\to W_t$ ($s<t$): **cumsum** (always active) + **gradient channel** (active only across mini-batches). The descent step always starts from $W_{t-1}$ (autoregressive), orthogonal to choice of $G_t$.

### 2.5 Dual form (matmul-ization)
- Mini-batch alone still has **few matmuls** → underuses TensorCores (matmul-only HW units).
- Problem: gradients $G_t = 2(W_0 x_t - x_t)x_t^\top$ are **outer products** ($d\times d$), computed one-by-one, heavy memory/IO.
- **Observation**: don't need to materialize $G_1,\dots,G_b$ or intermediate $W_t$ — only need $W_b$ (end of mini-batch) + outputs $z_1,\dots,z_b$. Express both as **matmuls**.
- For simplified TTT-Linear ($\theta_*=I$, first mini-batch), $X=[x_1,\dots,x_b]$:
  $$W_b = W_0 - 2\eta(W_0 X - X)X^\top$$
  $$z_t = W_t x_t = \Big(W_0 - 2\eta\sum_{s=1}^t (W_0x_s - x_s)x_s^\top\Big)x_t$$
  $$\Delta = (W_0 X - X)\,\text{mask}(X^\top X), \qquad Z = W_0 X - 2\eta\Delta \quad\text{(Eq 7,8)}$$
  - `mask` = upper-triangular zero mask (like attention's causal mask but zeros not $-\infty$).
- **Primal vs dual**: dual is **equivalent in output**, trades theoretical complexity for HW utilization.
  - Primal mini-batch: $O(b\times d^2)$. Dual: $O(b\times d^2)$ for $W_b$ + extra $O(b^2\times d)$ for $z$'s.
  - In practice $d$ ~ few hundred, $b=16$ → dual wall-clock cheaper. **>5× faster** in JAX impl.
- **App A**: dual form extends to **arbitrary-depth MLPs with nonlinearities** (more complex notation; uses `mask`, sums, $\sigma$, $\sigma'$; nonlinear $\sigma$ / its vjp can't be matmul-accelerated).
- **Forward/backward** (training) and **prefill** (inference) are parallel → dual form. **Decode** (generate) is sequential → primal form.

---

## 2.6 Theoretical equivalences

> TTT layers are a strict generalization. RNN layers ↔ TTT layers w/ **parametric** learners; attention ↔ TTT w/ **nonparametric** learner (Fig 8, 9). A **learner** (model + optimizer) uniquely induces a TTT layer.

- **Theorem 1**: TTT w/ $f(x)=Wx$, **batch GD**, $\eta=1/2$, $W_0=0$, output rule Eq 5 ⟹ output = **linear attention** [Katharopoulos].
  - Linear attention = self-attention sans softmax: $z_t = V_t(K_t^\top q_t) = \sum_s v_s k_s^\top q_t$; $\sum_s v_s k_s^\top$ is the hidden state (cumsum → linear complexity).
  - DeltaNet ≈ TTT-Linear w/ inner mini-batch size 1 (no LN/residual). Mamba-2 / Gated DeltaNet are related matrix-state RNNs.
- **Theorem 2**: TTT w/ **Nadaraya-Watson** nonparametric estimator + kernel $\kappa(x,x';\theta_K,\theta_Q)\propto e^{(\theta_Kx)^\top\theta_Qx'}$, label $y_s=\theta_V x_s$ ⟹ output = **self-attention** (App B derives N-W estimator as conditional expectation via kernel density).
- Unified **learner** abstraction: `train`/`predict` methods (sklearn-style). Hidden state = learner's internal storage (can include **optimizer state** → enables Adam-style inner optimizers in future).

---

## 4. Experiments & scaling

- **Codebase**: JAX (EasyLM); Mamba in PyTorch/Triton/CUDA. Run on TPUs.
- **Baselines**: Transformer (Transformer++ = Llama arch: RoPE, SwiGLU, RMSNorm) & **Mamba**.
- **Sizes**: 125M / 350M / 760M / 1.3B (Mamba: 130M/370M/790M/1.4B). **Chinchilla recipe**, Llama tokenizer, 0.5M tokens/batch.
- **Datasets**: **Pile** (2k, 8k context); **Books3** subset for long context (1k→32k, ×2 increments).
- TTT-Linear/MLP use **Mamba backbone** by default; **(T)** = Transformer backbone, **(M)** = Mamba backbone.

### Results
- **Pile 2k (Fig 10 left)**: TTT-Linear (M), Mamba, Transformer ~ comparable. TTT-MLP (M) slightly worse at large FLOPs (better ppl/size but extra FLOPs offset it).
- **Pile 8k (Fig 10 right)**: TTT-Linear (M) & TTT-MLP (M) **significantly beat Mamba**. Even TTT-MLP (T) beats Mamba ~1.3B. **Robust trend: advantage over Mamba widens as context grows.** Transformer has good ppl but bad FLOP-efficiency.
- **Books 2k (Fig 11)**: like Pile 2k but Mamba now slightly > TTT-Linear.
- **Books 32k (Fig 11)**: TTT-Linear (M) & TTT-MLP (M) **beat Mamba**; even (T) variants beat Mamba. TTT-MLP (T) only slightly worse than (M) at 1.3B → Transformer backbone may suit **larger models + longer context**.
- **Context-length use (Fig 2 right)**: TTT-Linear, TTT-MLP, Transformer keep reducing per-token perplexity through 32k; **Mamba plateaus after 16k**.
- **Backbone effect**: TTT layers prefer Mamba backbone overall. With Mamba backbone, TTT-MLP ≈ TTT-Linear; with Transformer backbone TTT-MLP clearly > TTT-Linear. Hypothesis: temporal convs help more when the seq layer's hidden state is **less expressive** (linear benefits more than MLP).
- **No clean linear scaling-law fit** (unlike Chinchilla) — attributed to dataset/context/tokenizer/arch differences; points connected, not regressed.
- **Fig 16**: for models trained from scratch, perplexity worsens once context too large; **best context length grows with model size**. TF finetune avoids this.

### Wall-clock (Fig 12, A100)
- v5e-256 TPU training: Transformer 0.30s/iter, **TTT-Linear 0.27s/iter** (10% faster, no systems opt).
- **Forward (prefill, bs 16)**: Transformer latency grows **linearly** with context; TTT-Linear/MLP & Mamba ~**constant**. TTT-Linear ≈ cheapest; TTT-MLP higher constant.
- **Generate (decode, bs 512)**: same — Transformer linear, others flat. TTT-MLP constant but higher than TTT-Linear/Mamba (heavier memory I/O).
- Inference kernels written (vLLM for TF baseline); training kernel not (TPU-only).

### Ablations (Table 1, 125M, building up from linear attention)
| Configuration | Ppl | Δ |
|---|---|---|
| Linear attention | 15.91 | – |
| Linear attn. improved | 15.23 | −0.68 |
| TTT equivalence | 15.23 | 0 |
| + learnable $W_0$ | 15.27 | **+0.04** (hurts slightly) |
| + LN and residual in $f$ | 14.05 | −1.22 |
| + **mini-batch TTT** ($b{=}2048\to16$) | 12.35 | **−1.70** (biggest) |
| + learnable $\eta$ | 11.99 | −0.36 |
| + Mamba backbone | **11.09** | −0.90 |

- **Biggest wins**: mini-batch TTT (−1.70) and LN+residual in $f$ (−1.22) — both arise naturally only from the TTT conceptual framework. Learnable $W_0$ hurts on its own but enables the rows below to train stably.

---

## 5. Related work

- **Learning at test time**: local learning [Bottou-Vapnik], transductive learning [Vapnik], dynamic evaluation (NLP finetune on test prompt), TTT-with-reconstruction [Sun 2020] (helps outliers), TTT on video streams [Wang] (most relevant — $f_t$ trained on past frames, autoregressive).
  - Key diff vs prior TTT: SSL task is **learned in the outer loop**, not handcrafted.
- **Fast weights / Fast Weight Programmers (FWPs)** [Schmidhuber 1992]: update a "fast" model on relevant data. Inner $W$ = fast weights, outer $\theta$ = slow. TTT = special case of FWPs with an **explicit learning problem + explicit optimization update**. Irie et al. made fast weights an MLP (precursor to TTT-MLP).
- **Modern RNNs**: Mamba, RWKV, xLSTM, GLA, DeltaNet (= TTT-Linear w/ mini-batch 1, no LN/residual), Mamba-2, Gated DeltaNet.
- **Learning to learn / meta-learning**: prior work's inner loop learns from a **dataset** (outer loop = across datasets, hard to scale). TTT's inner loop = **one sequence**, outer loop is **regular supervised learning** ("same level") → easy to scale (Table 2).

### Limitations & future
- TTT instantiations still need **substantial wall-clock** even after dual form. **TTT-MLP** especially: good FLOP-efficiency & long-context potential but heavy **memory I/O**, much worse wall-clock relative to FLOPs.
- Future: richer outer-loop task families; systems opt (pipeline parallelism over time for million-token seqs); larger models + longer context; convolutional $f$ for video/embodied; **multi-level** learning-to-learn (if $f$ is itself attention → nested inner loops).
- **Why TTT?** humans don't learn from i.i.d. shuffled data — they learn from a long temporal stream where any datum can be train or test. TTT's inner loop matches this.

---

# Implementation

> Source: official **PyTorch** reference impl `ttt-lm-pytorch/ttt.py` (~1650 lines). README: "naive implementation for tutorial purposes; **do not train** with this (pure PyTorch, no systems opt) — use the JAX codebase." HuggingFace-`transformers`-style (`PreTrainedModel`, `PretrainedConfig`). File: `/home/jayoo/code/jayoo_personal/LLM_Knowledge_Base/raw_data/papers/Learning to Learn at Test Time - RNNs with Expressive Hidden States/ttt-lm-pytorch/ttt.py`

## Architecture wiring (`TTTConfig`, `Block`, `TTTModel`, `TTTForCausalLM`)
- **Config defaults = TTT-1B** (`hidden_size=2048, intermediate_size=5504, layers=24, heads=32`); presets `TTT_STANDARD_CONFIGS` for 125m/350m/760m/1b.
- Key TTT knobs: `ttt_layer_type ∈ {"linear","mlp"}`, `ttt_base_lr=1.0`, `mini_batch_size=16`, `use_gate` (Mamba backbone), `share_qk` (Mamba backbone shares Q/K proj + separate convs), `pre_conv` + `conv_kernel=4` (temporal conv), `scan_checkpoint_group_size` (grad checkpoint through time).
- **`Block`** (residual block, lines 1276–1330): optional pre-`Conv` (RMSNorm + depthwise causal `Conv1d`) → `seq_norm` (RMSNorm) → **TTT layer** (`seq_modeling_block`) → residual → `ffn_norm` → **SwiGluMLP** → residual. Standard Transformer-style block with the seq-mixer swapped for TTT.
- **`TTTModel.forward`** builds `position_ids` from `seqlen_offset`; creates `TTTCache` when `use_cache`. `TTTForCausalLM` adds LM head (tied embeddings by default).

## Hidden-state-as-model abstraction
- The inner model $f$'s weights are **`nn.Parameter`s on the TTT layer = the learnable initialization $W_0$** (paper's $\theta_{\text{init}}$). Per-head:
  - **TTT-Linear** (lines 918–923): `W1 [nh, f, f]`, `b1 [nh, 1, f]` (`f`=head_dim).
  - **TTT-MLP** (lines 1072–1079): `W1 [nh, f, 4f]`, `b1 [nh,1,4f]`, `W2 [nh,4f,f]`, `b2 [nh,1,f]`.
- At runtime these are **tiled across batch** → become the per-sequence hidden state $W_t$ that evolves; the `nn.Parameter` is just the shared init (trained in outer loop).
- **Multi-view projections** = `q_proj/k_proj/v_proj` (`TTTBase._init_qkvo_proj`, lines 639–664). Output proj `o_proj`. These are $\theta_Q,\theta_K,\theta_V$. (`share_qk` merges Q/K to one proj + separate depthwise convs, for Mamba backbone.)
- **Reconstruction LN params** (`_init_ttt_ln`, 694–699): per-head learnable `ttt_norm_weight/bias` — the LN inside $f$ that the inner loss is taken through.

## Inner learning rate $\eta$ (learnable, token-dependent) — `_init_ttt_lr_gate` + `get_eta`
```python
# _init_ttt_lr_gate (674-692): per-head linear gate weight [nh, width, 1], bias 0
# get_eta (776-801):
ttt_lr = einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + bias   # [B,nh,nmb,K,1]
ttt_lr = F.sigmoid(ttt_lr)
ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim                 # η(x)=η_base·σ(θ_lr·x)/f
# token_idx scale = 1/arange(1..K) + learnable_token_idx (clamped ≥0)
token_eta = broadcast(token_idx[offset:offset+K])
eta = token_eta * ttt_lr_eta
```
- **Deviation/detail vs paper**: $\eta$ is split into two factors — `ttt_lr_eta` = the paper's learnable gate $\eta_{\text{base}}\sigma(\theta_{\text{lr}}x)/d$, and **`token_eta`** = a per-position scale $1/t$ (`token_idx = 1/arange(1,K+1)`) **plus a learnable per-position offset** (`learnable_token_idx`). This `token_idx` implements the $\tfrac1t$ averaging of the summation in Eq 4 within a mini-batch. Decoupled so decoding can apply them separately.

## TTT forward (`TTTBase.forward`, lines 850–915)
1. `get_qkv_projections` → `XQ,XK,XV` (+ depthwise convs if `share_qk`).
2. Reshape to heads `[B, nh, L, f]`; **RoPE applied with `position_ids % mini_batch_size`** (positions reset every mini-batch). `permute_qk`/`undo_permute_qk` align PyTorch RoPE convention with the JAX pretraining convention.
3. Split sequence into `num_mini_batch` full mini-batches + a **remainder** (`reminder_len = L % mini_batch_size`); each calls `self.ttt(...)`. Remainder reuses `last_mini_batch_params_dict` to continue.
4. `get_ttt_inputs` (810–839): reshape to `[B, nh, num_mini_batch, K, f]`, compute `eta`.
5. After TTT: `post_norm` (LayerNorm) → optional `apply_gate` (Mamba backbone: `gelu(g_proj(x)) * ttt_out`, tanh-approx GELU) → `o_proj`.

## The inner update — TTT-Linear `ttt` (lines 925–1069)
Core `compute_mini_batch` (run per mini-batch via `scan`):
```python
X1 = XK_mini_batch
Z1 = X1 @ W1_init + b1_init                       # f forward on training view (key)
reconstruction_target = XV_mini_batch - XK_mini_batch   # label = V - K  (residual recon target!)
grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)  # ∇ of LN∘L2 loss
```
- **Key deviation from paper Eq 4**: target is **`XV - XK`** not `XV` — because $f$ has a **residual** ($f(x)=x+\text{LN}(\cdot)$), so the residual branch must reconstruct $\theta_V x - \theta_K x$. The LN+residual lives implicitly: `ln_fused_l2_bwd` computes $\nabla_{Z}\,\tfrac12\lVert\text{LN}(Z)-\text{target}\rVert^2$ analytically (fused LN backward, lines 478–505), and `XQW = XQ + Z1_bar` (line 1023) adds the residual.

**Dual form** (`use_dual_form = cache_params is None or mini_batch_size % mini_batch_size == 0`, i.e. always during prefill/training):
```python
Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2,-1))          # masked QKᵀ  ([B,nh,K,K])
b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar  # = output tokens
# end-of-mini-batch state (the single W carried forward):
last_eta = eta_mini_batch[:, :, -1, :, None]
W1_last = W1_init - (last_eta * X1).transpose(-1,-2) @ grad_l_wrt_Z1
b1_last = b1_init - torch.sum(last_eta * grad_l_wrt_Z1, dim=-2, keepdim=True)
```
- This is **Eq 7/8 realized as matmuls** + a causal `tril` mask (the paper's `mask`). `Attn1 = tril(XQ X1ᵀ)` is exactly the attention-like term. Output `Z1_bar` never materializes intermediate $W_t$ or per-token gradients.
- `grad_W1_last`/`grad_b1_last` set to **zero** in dual form (not needed — only used to carry partial-mini-batch grads during decode).

**Primal form** (`else` branch, used during decode when a mini-batch isn't full): explicitly builds per-token gradients via `einsum`, accumulates with **stored `W1_grad`/`b1_grad`** from cache, applies `tril(ttt_lr_eta)` cumsum:
```python
grad_W1 = einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)              # outer products G_t
grad_W1 = einsum("bhnk,bhkij->bhnij", tril(ttt_lr_eta), grad_W1)    # cumsum w/ lr weighting
grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)            # accumulate across calls
W1_bar  = W1_init.unsqueeze(2) - grad_W1 * token_eta.unsqueeze(-1)  # per-token W_t (mini-batch GD)
Z1_bar  = (XQ.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
```

## TTT-MLP `ttt` (lines 1081–1268)
- Same structure, two layers: `Z1 = X1@W1+b1 → X2 = gelu(Z1, tanh) → Z2 = X2@W2+b2`. Recon target again `XV - XK`.
- **Inner backprop chain** (manual, not autograd):
  ```python
  grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, recon_target, ln_w, ln_b)
  grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.T * gelu_bwd(Z1)   # gelu_bwd = analytic tanh-GELU deriv (509-512)
  ```
- Dual form uses **two attention-like terms** `Attn1` (from XQ·X1ᵀ) and `Attn2` (from X2_bar·X2ᵀ), updates both layers. Matches App A's depth-$K$ dual form (mask + matmuls + $\sigma,\sigma'$).

## Mini-batch sequencing — `scan` (lines 431–458)
- Pure-Python emulation of `jax.lax.scan`: carries `params_dict` (the evolving $W$) across mini-batches, writes outputs into a preallocated tensor. **Sequential over mini-batches, parallel within** (dual form).
- `scan_checkpoint_group_size` → `torch.utils.checkpoint` **through time** (gradient checkpointing over mini-batch groups) to bound activation memory during training (App C).
- Inputs permuted `[B,nh,nmb,K,f] → [nmb,B,nh,K,f]` so scan iterates over mini-batch axis.

## Inference cache — `TTTCache` (lines 515–597)
- Stores, per layer, the **hidden-state weights** (`W1_states,b1_states,...` = current $W_t$) **and partial gradients** (`*_grad`) + conv states.
- `update(seq_len)`:
  - If `seq_len % mini_batch_size == 0`: **commit** new states, **zero gradients** (mini-batch boundary → start fresh accumulation).
  - If `seq_len < mini_batch_size` (decode): if it completes a mini-batch, commit states + zero grads; else **save partial `*_grad`** so the next token continues mini-batch GD (primal form).
- This is why decode needs the primal form: per-token grads must persist & accumulate (`W1_grad`) until the mini-batch fills.

## Helper math (analytic, matmul-friendly)
- `ln_fwd` (461–475): standard LN forward.
- `ln_fused_l2_bwd` (478–505): closed-form $\nabla_x\,\tfrac12\lVert\text{LN}(x)\cdot\gamma+\beta - \text{target}\rVert^2$ — fuses the LN backward + L2 grad so the inner update is one analytic expression (no autograd inside the inner loop).
- `gelu_bwd` (509–512): tanh-approx GELU derivative (from Megatron), for TTT-MLP layer-1 inner grad.

## Notable details / deviations from the paper
- **Recon target is `V−K`, not `V`** (residual branch of $f$), and LN+residual are **baked into the inner forward/backward** via fused analytic ops rather than relying on autograd.
- **$\eta$ factored into `token_eta` (1/t averaging + learnable offset) × `ttt_lr_eta` (gate)** — the $1/t$ within-mini-batch scaling is an impl detail not spelled out in the main text.
- **RoPE uses `position_ids % mini_batch_size`** — positions are local to each mini-batch.
- **Dual vs primal switch is automatic**: dual whenever no cache or full mini-batch (train/prefill); primal for partial mini-batches (decode) — because primal can store/accumulate partial gradients that dual discards.
- Manual inner-loop backprop (hand-derived grads), no `torch.autograd` for the inner step — required for the matmul dual form & speed.
- `share_qk`, `use_gate`, `pre_conv` toggles implement the **Mamba backbone** (Fig 13): pre temporal conv, Q/K conv, GELU gate.
