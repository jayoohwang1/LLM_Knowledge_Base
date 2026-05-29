# Test-Time Training with KV Binding Is Secretly Linear Attention (TTTLA)

> **Liu, Elflein, Litany, Gojcic, Li** — NVIDIA SIL, **ICML 2026**. arXiv:2602.21204.
> Project: research.nvidia.com/labs/sil/projects/tttla. Code = LaCT (`tttla` branch) + ViTTT (`tttla` branch).

**One-liner**: TTT-with-KV-binding is *not* test-time memorization of a K→V map. A broad class of TTT layers (incl. multi-layer MLP inner loops + momentum) is **exactly a learned linear attention operator**. This explains weird empirical behaviors, justifies aggressive simplification, and unlocks a **fully parallel** form (up to $4\times$ throughput).

---

## 1. Setup: what is "TTT with KV binding" (TTT-KVB)

- **TTT** = update *some* model params (**fast weights** $f_\theta$, usually a small MLP) during *both* training and inference, per sequence, via an inner-loop self-supervised objective. As a sequence-modeling layer it competes w/ softmax attention but with linear-time compute + constant-size state (RNN-like).
- **Two TTT flavors** (paper studies #1 only):
  1. **TTT-KVB**: inner loop = **key–value binding loss** (dot-product or MSE), no backprop from final task loss. (= "TTT-KVB" in Tandon et al. 2025). Variants: TTT, LaCT, ViTTT, Titans, Atlas, etc.
  2. **TTT-E2E**: backprop through inner loop from final (e.g. CE) loss.
- **KV binding mechanism** (precise):
  - Project tokens to keys $K$, values $V$, queries $Q$.
  - Inner loop: online **gradient descent on $\theta$** of fast-weight net $f_\theta$ using **key as input, value as target**: $\mathcal{L} = \lVert f_\theta(k) - v\rVert^2$ (MSE) or dot-product loss $\mathcal{L} = -\langle f_\theta(k), v\rangle$ (LaCT/ViTTT use the Frobenius/dot-product form).
  - After updating $\theta$, **query** $q$ is passed through updated $f_\theta$ → output.

**Standard interpretation (which paper rejects)**: *online meta-learning / storage-and-retrieval*. Inner loop "memorizes" KV associations into $f_\theta$; querying = "retrieving". Motivates complexity: fancy optimizers (Muon), weight normalization, multi-layer MLP inner nets, momentum, per-token LR — all framed as "improving memorization fidelity."

---

## 2. The Memorization Paradox — empirical contradictions (§4)

Four anomalies that a true storage-retrieval system should NOT exhibit. Evaluated on **LaCT-LLM** (perplexity), **LaCT-NVS** (PSNR), **ViTTT** (top-1 acc).

- **More inner-loop steps → worse downstream perf** (Fig 1).
  - Lower inner loss (better "memorization") but degraded PSNR / higher perplexity, monotonically. Holds on LLM + NVS.
  - Inner-loop fitting quality is **anti-correlated** with task performance — opposite of memorization prediction.
  - Mechanistic reason (later): extra steps at test time induce an attention operator *different from the one used at training* → train/test mismatch, not better memory.
- **Gradient ascent ≈ gradient descent** (Table 1, "Gradient Ascent Anomaly", most striking).
  - Flip sign of *all* fast-weight gradients (retrain from scratch w/ inverted inner loop). Perf is **comparable, sometimes better**.
  - For dot-product (Frobenius) loss, flipping grad = negating the loss itself → "memorization" actively sabotaged, yet model fine.
  - LaCT-LLM ppl 16.43→16.19, NVS PSNR 25.94→25.85, ViTTT 79.34→79.61.
- **Distributional asymmetry between $Q$ and $K$** (Fig 2, §4.3).
  - Retrieval needs $Q$ distribution ≈ $K$ distribution (else $f_\theta$ evaluated OOD). t-SNE shows **persistent, layer-consistent $Q$/$K$ and $V$/$O$ mismatch** in pretrained LaCT-NVS.
  - Inner loop is fit on $K$ but always queried OOD on $Q$ → outputs can't be reliable retrieval.
- **Replacing $Q$ with $K$ → no degradation** (Table 1, §4.4).
  - In real attention this is catastrophic (self-similarity dominates). Here perf nearly unchanged (LaCT-LLM 16.18, NVS 25.95, ViTTT 79.18).
  - ⇒ $q$ is **not** a semantic query into a memory.

> **Conclusion**: downstream behavior is insensitive to (a) inner-loop optimization quality, (b) its *direction*, (c) presence of a meaningful query. TTT is not test-time memorization.

---

## 3. Central claim: TTT-KVB *is* learned linear attention (§5)

**Recall linear attention** (Katharopoulos 2020): replace softmax w/ kernel feature map $\phi$; recurrent state $S_t = S_{t-1} + \phi(k_t)^\top v_t$, output $o_t = \phi(q_t) S_t$. Constant-size state, linear time. DeltaNet, Gated DeltaNet, Mamba, RWKV, GLA, etc. add data-dependent decay / delta rule.

**Key prior fact** (Sun et al. 2025): single linear inner-loop layer + zero init = linear attention. **This paper generalizes to nonlinear multi-layer MLP inner nets + momentum.**

### 3.1 Theorem 5.1 — Linearization of a single inner-loop GD step
Inner-loop net has a **linear bias-free final layer**: $f(x) = \phi(x;\Theta)\,W$, $\phi(x;\Theta)\in\mathbb{R}^{D_h}$, $W\in\mathbb{R}^{D_h\times D_{out}}$.
One GD step on loss $\mathcal{L}$ w/ LR $\eta$ updates **all** params $(W,\Theta)$ using key $k$:
$$ (W_{t+1},\Theta_{t+1}) = (W_t,\Theta_t) - \eta\nabla_{(W_t,\Theta_t)}\mathcal{L}(f_t(k)), \quad \phi_t(\cdot)\triangleq\phi(\cdot;\Theta_t),\ f_t(\cdot)\triangleq\phi_t(\cdot)W_t.$$
The **final-layer** update is rank-structured (chain rule): $W_{t+1} = W_t + \phi_t(k)^\top g_t(k)$, with $g_t(k)\triangleq -\eta\,\partial\mathcal{L}/\partial f_t(k)$.
Evaluating on query $q$:
$$ o = \phi_{t+1}(q)\big(W_t + \phi_t(k)^\top g_t(k)\big) = \hat q\,(S_0 + \hat k^\top\hat v), $$
$$ \boxed{\hat q = \phi_{t+1}(q),\quad \hat k = \phi_t(k),\quad \hat v = g_t(k),\quad S_0 = W_t.}$$
**This is exactly a linear-attention operator.** (Proof App. B.)

### 3.2 Theorem 5.2 — Unroll over the sequence (one GD step / token)
By induction (App. C):
$$ o_t = \phi_{t+1}(q_t)\Big(W_0 + \sum_{i=0}^{t}\phi_i(k_i)^\top g_i(k_i)\Big) = \hat q_t\Big(S_0 + \sum_{i=0}^{t}\hat k_i^\top\hat v_i\Big). $$
⇒ standard causal/extended linear attention; state $S_t=W_0+\sum \hat k_i^\top \hat v_i$.

### 3.3 Theorem 5.3 — GD with momentum
Momentum accumulator $\Delta W_t = \nabla\mathcal{L}(f_t(k_t)) + \alpha_t\Delta W_{t-1}$, update $W_{t+1}=W_t-\eta\Delta W_t$. Cumulative momentum coeff $\beta_i^j = \prod_{s=i+1}^j \alpha_s$ (1 if $i=j$). Unrolling (App. D):
$$ o_t = \hat q_t\Big(W_0 + \sum_{i=0}^t \phi_i(k_i)^\top m_i(k_i)\Big),\quad \boxed{\hat v_i = m_i(k_i)\triangleq g_i(k_i)\cdot\sum_{j=i}^t\beta_i^j.}$$
**Momentum = effective value is a momentum-weighted sum of past gradients.** Still linear attention.

> **KV binding precisely** = the rank-structured outer-product update $\phi_t(k)^\top g_t(k)$ to the linear-attention state $S$. The "binding" is just the key-feature ⊗ value-gradient outer product, identical to linear-attention's $\phi(k)^\top v$. There is no associative memory — just a learned feature mixer.

### 3.4 The anomalies, explained by the linear-attention lens (§5.2)
- **More inner steps** → changes the *learned attention operator* itself (kernel $\phi$, effective $\hat q,\hat k,\hat v$ depend on # steps); test-time # steps ≠ train-time → train/test mismatch ⇒ degradation. Not "more memory."
- **Gradient ascent** → only flips sign of effective value $g_t$; sign is **absorbed** into the learned operator (the surrounding params adapt during training). Harmless.
- **$Q$/$K$ asymmetry** → $q$ and $k$ feed *different* parts of the operator: $q\!\to\!\hat q=\phi_{t+1}(q)$ (effective query), $k\!\to\!\hat k=\phi_t(k)$ (key) AND $\hat v=g_t(k)$ (value). They're intermediate features, not symmetric similarity operands ⇒ distributional mismatch is expected, not pathological.
- **Replace $q$ with $k$** → effective query $\phi_{t+1}(k)$ vs key $\phi_t(k)$ remain *distinct* ($\phi$ evaluated at different param states $t+1$ vs $t$) ⇒ operator preserved.

---

## 4. Which TTT variants are unified

### 4.1 LaCT (Zhang et al. 2025) — §5.3, App. E
- Inner net = **bias-free SwiGLU MLP**: $f(x) = \big(\mathrm{silu}(xW_0)\odot(xW_2)\big)W_1$, $W_0,W_2\in\mathbb{R}^{D_h\times D_k}$, $W_1\in\mathbb{R}^{D_h\times D_v}$.
- Loss = **Frobenius inner product** $\mathcal{L}(f(k),v)=-\langle f(k),v\rangle$ ⇒ upstream grad $\partial\mathcal{L}/\partial f = -v$.
- Per-token LR $\eta_t$, momentum $\alpha_t$, **Muon-style grad orthogonalization** $\mathcal{M}(\cdot)$ (Newton-Schulz), **weight normalization** after each step.
- Kernel $\phi_t(x)=\mathrm{silu}(xW_{0,t})\odot(xW_{2,t})$. Linear-attention form (Eq. 1):
  $$ o_t = \phi_{t+1}(q_t)\Big(W_{1,0} + \sum_{i=0}^t \mathcal{M}(\phi_i(k_i)^\top m_i)\Big),\quad m_i(k_i)=v_i\cdot\sum_{j=i}^t \eta_j\beta_i^j. $$
  Muon applied **element-wise to each KV outer product before accumulation**. Weight-norm omitted in main text, handled in App. E.4.

### 4.2 ViTTT (Han et al. 2025, "ViT³") — §5.4, App. F, G
Two **independent** fast-weight components, each a linear-attention instance ⇒ whole layer is linear attention:
- **GLU component**: $f(x)=\mathrm{silu}(xW_0)\odot(xW_1)$, Frobenius loss. Both $W_0,W_1$ square, updated by GD. After 1 step:
  $$ o_t = \phi_{t+1}(q_t)\odot\Big(q_t\big(W_1 + \eta k_t^\top(v_t\odot\phi_t(k_t))\big)\Big),\ \ \phi_t(k)=\mathrm{silu}(kW_0). $$
  $\langle q_t,k_t\rangle$ = scalar attention weight modulating $v_t$; $\phi_t(k_t)$ gates values; $\phi_{t+1}(q_t)$ gates output.
- **$3\times3$ depthwise conv component** (App. G): conv = sliding-window linear layer ⇒ **spatially-local (sliding-window) linear attention**. Attention weight between positions = overlap of their $3\times3$ neighborhoods; first term = initial state $S_0=W_t$.

> Other KV-binding variants (**Titans, Atlas**) also satisfy the theorem assumptions; empirical validation left to future work. **Limitation**: theory requires inner-loop **final layer linear & bias-free**.

### 4.3 Relation to fast-weight / linear-attention literature
- **Fast-weight programming** (Schlag 2021: "linear transformers are secretly fast weight programmers") — this is the converse direction: fast-weight TTT is secretly linear attention.
- **DeltaNet** (Schlag/Yang): already known equivalent to TTT w/ single linear layer + MSE. This paper extends to nonlinear/multi-layer/momentum.
- **Gated DeltaNet / Mamba / GLA / RWKV**: data-dependent decay = the per-token LR $\eta_t$ + momentum decay $\alpha_t$ here.

---

## 5. Practical implications (§6)

### 5.1 Ablation: reduce TTT → standard linear attention (Table 2, §6.1)
Applied **sequentially** (Variant N = Variant N-1 + next step). Baseline = full LaCT / ViTTT.

| Step | Change | Justification (linear-attn view) |
|---|---|---|
| **1** | Update **only last layer** $W_1$; freeze $\Theta=\{W_0,W_2\}$ | $\phi(\cdot)=\phi(\cdot;\Theta)$ becomes a **static learnable kernel**; $\phi_t$ no longer history-dependent |
| **2** | **Remove weight normalization** | On $\Theta$ (fixed) it's a no-op; on $W_1$ = normalizing state $S_t$, uncommon in linear attn. **After this, fully parallelizable.** |
| **3** | **Multi-layer MLP → single linear layer** | Deeper MLP just = more complex kernel $\phi$; with capable $q,k$ unhelpful. Exposes true $\hat q=q,\hat k=k$. |
| **4** | **Remove per-token LR** $\eta_t$ | Frobenius loss absorbs $\eta_t$ into learnable $v_t$ ⇒ redundant. Constant LR=1.0 suffices. |
| **5** | **Remove momentum in SGD** | Momentum only remixes value vector $\hat v$; w/ learnable K,V it's redundant. $g_t(k)=-v$, momentum recovers $\hat v=v$. |
| **6** | **Remove gradient orthogonalization** (Muon $\mathcal{M}$) | Apply $\mathcal{M}(\hat k^\top v)$ → drop it. **Reduces exactly to standard linear attention** $o=q(W+\sum_i k_i^\top v_i)$. |

**Results** (32k seq for LLM):
| Variant | Desc | LaCT-LLM ppl↓ | NVS PSNR↑ | ViTTT acc↑ | Recurrent tok/s | Parallel tok/s |
|---|---|---|---|---|---|---|
| Baseline | full LaCT/ViTTT | 16.43 | 25.94 | 79.34 | 4.30M | N/A |
| V1 | +update only $W_1$ | **15.93** | **25.97** | 79.63 | 10.60M | N/A |
| V2 | +remove weight-norm | 16.31 | 25.93 | 79.63† | 11.02M | 30.18M |
| V3 | +multi-layer→single linear | 16.23 | 25.71 | 79.39 | 12.95M | 49.69M |
| V4 | +remove per-token LR | 16.12 | 25.70† | 79.39 | 13.31M | 53.99M |
| V5 | +remove momentum | 15.97 | 25.70† | 79.39 | 14.40M | 57.28M |
| V6 | +remove grad orthog. (=std linear attn) | 16.80 | 25.73 | 79.54* | **89.67M** | **124.6M** |

- † = step N/A for that task (matches preceding variant). * = ViTTT removes grad-norm instead of orthog.
- **V1 best overall on all 3 tasks** — restricting to last-layer update *improves* perf, contradicting "deeper memory better."
- **Most components contribute marginally / are detrimental.** Exceptions: deeper MLPs help NVS; grad orthog. helps LLM.
- **Full reduction to basic linear attention (V6) = only minor degradation**: +0.4 ppl on LLM, −0.2 dB NVS. Standard linear attn already captures almost everything TTT adds. (Abstract: full TTT only beats linear-attn variant by +0.87 ppl / +0.24 dB.)
- Per-token learnable LR & weight-norm = "valuable" assumptions shown unnecessary.

### 5.2 Parallel form of TTT (§6.2, App. H)
- **Key insight**: when (a) only $W_1$ dynamic, $W_0,W_2$ static AND (b) weight-norm removed ⇒ kernel $\phi_t(\cdot)=\phi(\cdot;\Theta)$ static & **state update is associative** ⇒ recurrence (incl. momentum) computable via **parallel prefix scan** instead of token-by-token.
- **Chunked parallel form** (App. H.1): $N$ chunks of size $L$, $\Phi(X)=\mathrm{silu}(XW_0)\odot(XW_2)$, accumulation mask $\mathcal{C}_{ti}=\sum_{j=i}^t \eta_j\prod_{s=i+1}^j\alpha_s$ folds per-chunk LR into cumulative momentum decay:
  $$ \mathbb{O} = \Phi(\mathbb{Q})W_{1,0} + \big((\Phi(\mathbb{Q})\Phi(\mathbb{K})^\top)\odot\mathcal{C}^{\dagger L}\big)\mathbb{V}, $$
  $(\cdot)^{\dagger L}$ = Kronecker w/ $\mathbf{1}_{L\times L}$ expanding $N\times N$ mask to $(NL)\times(NL)$. Proof of equivalence to seq. recurrence in App. H.2.
- **Speedups**: parallel TTT layer up to **$4.0\times$** inference throughput (tok/s, single batch) vs recurrent (124.6M vs 30.18M for V2; see Table 2). Combined w/ Steps 1+2 ⇒ **$1.19\times$ end-to-end training** speedup, comparable convergence (Fig 4).
- **Non-reducible cases break parallelism** (App. I): (1) updating kernel params $\Theta=\{W_0,W_2\}$ → nested silu/silu′ dependency chain (history-dependent kernel); (2) **weight normalization** → $\mathrm{Norm}(A+B)\neq\mathrm{Norm}(A)+\mathrm{Norm}(B)$, non-associative ⇒ forces sequential. Both keep linear-attn *interpretation* but kill the *parallel computation*.

---

## 6. Experiments setup (App. A)
- **LLM**: 760M LaCT-LLM, 100B tokens FineWeb-Edu, 8×A100, 20k iters (~56h), eval ppl on 2.5B Book-3 tokens, seq 32k. Built on **Flame** codebase.
- **NVS**: 12-layer 768-dim LaCT-NVS (114M), RealEstate10K, 4×A100, 20k iters (~38h), 128×128, PSNR. MSE loss only.
- **Image cls**: ViTTT-B (90M), ImageNet-1K, 2×H100, 60 epochs (~16h), top-1.

## 7. Limitations
- Only **KV-binding** TTT (not TTT-E2E).
- Inner-loop **final layer must be linear & bias-free**.
- Nonlinear final layers + deeper connections to modern linear-attn mechanisms = future work.
- Titans/Atlas reductions theoretically covered but not empirically validated.

---
---

## Implementation

Local repo (`raw_data/.../tttla/`) contains **only README**; actual code in two external repos (`tttla` branches not yet public, but the **`main` branch of LaCT already contains the full TTTLA ablation machinery** under `_ttt_operation_impl/` and `_lact_ttt/`, NVIDIA Apache-2.0 headers). Analysis below from cloned `main` branches: `JunchenLiu77/LaCT` and `JunchenLiu77/ViTTT`.

**Repo layout**:
- `lact_llm/lact_model/` — LLM (HF-style) model. `ttt_operation.py` (helpers + baseline recurrent op), `_ttt_operation_impl/` (ablation variants + dispatcher), `layer_lact_swiglu.py` (layer), `configuration_lact_swiglu.py` (config).
- `lact_nvs/_lact_ttt/` — NVS variants + dispatcher (`__init__.py`).
- `minimal_implementations/` — standalone `causal_lact_with_sliding_window_attn.py`, `bidirectional_lact_layer.py`.
- ViTTT: `ttt_block.py` (root + `vittt/models/`).

### Core recurrent op (LaCT baseline) — `ttt_operation.py` / `_ttt_operation_impl/original.py`
Block-causal chunked recurrence: **"apply then update"** (shifted block-causal). For each chunk: (1) **apply** previous fast weights to query → output; (2) **update** fast weights from this chunk's $k,v$. SwiGLU fast weight $f(x)=W_1(\mathrm{silu}(W_0 x)\odot(W_2 x))$ (note transposed BMM layout, weights `[b, d_out, d_in]`).

Forward + hand-written backward (chain rule, no autograd) computing the three weight deltas (`original.py` L86-108):
```python
gate_before_act = torch.bmm(w0, ki.transpose(1, 2))      # W0 k
hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))    # W2 k
hidden = F.silu(gate_before_act) * hidden_before_mul     # phi(k) = kernel feature
dhidden = torch.bmm(w1.transpose(1, 2), vi)              # backprop from v (upstream grad = -v)
dhidden_before_mul = dhidden * F.silu(gate_before_act)
dgate = dhidden * hidden_before_mul
dgate_before_act = silu_backprop(dgate, gate_before_act)
dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))   # v ⊗ (phi(k)·lr1) = KV outer product
dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(...))
dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(...))
```
- `dw1 = vi @ (hidden·lr1)` is **literally the linear-attention KV outer product** $\hat k^\top \hat v$ from Thm 5.1, with $\hat k=\phi(k)$, $\hat v = -v\cdot$lr (dot-product/Frobenius loss ⇒ upstream grad $=-v$). Confirms paper §5.3.
- **Momentum** (L110-119): `dw_i = dw_i + dw_momentum * m_i; dw_momentum = dw_i` — recursive momentum accumulator $\Delta W_t = g + \alpha\Delta W_{t-1}$ (Thm 5.3). `m_i` = chunk-mean of per-token momentum.
- **Muon orthogonalization** (`use_muon`): `dw = zeropower_via_newtonschulz5(dw)` — 5-step Newton-Schulz on each `[b,d,d]` delta. Matches $\mathcal{M}(\cdot)$ in App. E.
- **Weight normalization** (L137-140): `w = w/(w.norm(dim=2)+1e-5) * w_norm` — channel-wise $\ell_2$, "post-norm" (state normalization). This is the **non-associative** op that blocks parallelism (App. I.2).
- **Last chunk**: apply only, no update (causal shift).
- `silu_backprop(dy,x) = dy·σ·(1 + x(1−σ))` (analytic SiLU′). `zeropower_via_newtonschulz5` = batched `[b,d,d']` Muon (coeffs `(4.0848,-6.8946,2.9270)...`, bf16, spectral-norm-1 prescale).
- A `prenorm_block_causal_lact_swiglu` exists but `raise NotImplementedError` — paper only ran the post-norm version.

### Dispatcher — `_ttt_operation_impl/__init__.py`
Single entry `block_causal_lact_swiglu(..., loss_type=...)` routes to variant files via **magic-string parsing** of `loss_type`:
```python
if "no_wn" in loss_type:  kwargs["weight_norm"] = False; loss_type = loss_type.replace("_no_wn", "")
if "no_lr1" in loss_type: kwargs["use_lr1"] = False;     loss_type = loss_type.replace("_no_lr1", "")
# dot_product→original | no_query_dot_product→no_query | only_w1 | ga_dot_product→ga
# only_w1_straight_qk | only_w1_parallel→only_w1_no_wn_parallel
```
Wired from `layer_lact_swiglu.py` L405-412 (`loss_type=self.ttt_loss_type`), config field `ttt_loss_type` (default `"dot_product"`).

### Variant files = the paper's empirical contradictions + ablation steps
**Each maps 1:1 to a paper experiment**:
- `ga.py` (**§4.2 gradient ascent**): identical to `original.py` except `w = w - dw` instead of `w + dw` (signs flipped on all three weights).
- `no_query.py` (**§4.4 replace Q with K**): `del q; q = k.transpose(1,2)` — redefines query as key, then identical.
- `only_w1.py` (**Step 1**): only `dw1` computed/updated; `del lr0, lr2`; `w0,w2` frozen kernel. Optional `weight_norm` flag (Step 2).
- `only_w1_straight_qk.py` (**Step 3**): drops the SwiGLU kernel entirely — `hidden = ki` (straight key), output `= w1 @ qi` (straight query) ⇒ $\hat q=q,\hat k=k$, i.e. **standard linear attention** core. `use_lr1` flag toggles per-token LR (Step 4). Trick to keep frozen weights in autograd graph: `return ... + w0.sum()*0.0 + w2.sum()*0.0`.
- `only_w1_no_wn_parallel.py` (**§6.2 parallel form**) — see below.
- `mse.py` (146 lines): MSE-loss variant (vs dot-product). Not central to TTTLA claims.

### Parallel formulation — `_ttt_operation_impl/only_w1_no_wn_parallel.py`
The **fully parallel prefix-scan** form (App. H), `@torch.compile`, asserts `not weight_norm`. Implements only-$W_1$-dynamic + no-weight-norm + static kernel:
```python
w02 = torch.cat([w0, w2], dim=1)                 # fuse the two static-kernel projections
q_flat = q_pad.view(b*n_blocks, chunk_size, dk)  # flatten (B, N_chunks)
z_all = torch.bmm(w02_exp, k_flat.transpose(1,2)); gate_k, hidden_k = z_all.chunk(2, dim=1)
gated_k = F.silu(gate_k) * hidden_k              # phi(k), static kernel (no history dep)
dw1 = torch.bmm(v_flat.transpose(1,2), (gated_k.transpose(1,2) * lr1).type_as(v_flat))  # per-chunk KV outer product
dw1 = dw1.view(b, n_blocks, dv, dh)
# Momentum via parallel scan (cumprod/cumsum) instead of sequential loop:
m_prod = m_blk.cumprod(dim=1).unsqueeze(-1)
dw1 = m_prod * torch.cumsum(dw1 / (m_prod + 1e-8), dim=1)
if use_muon: dw1 = zeropower_via_newtonschulz5(dw1.reshape(-1,dv,dh)).reshape_as(dw1)
prefix_exclusive = torch.cumsum(dw1, dim=1) - dw1     # exclusive prefix sum = block-causal state S
w1_before = prefix_exclusive + w1.unsqueeze(1)        # state at each chunk
# apply: out = w1_state @ phi(q)
gated_q = F.silu(gate_q) * hidden_q
out = torch.bmm(w1_before.view(b*n_blocks, dv, dh).type_as(gated_q), gated_q)
```
- **Key mechanism**: sequential weight recurrence → `cumsum` over chunk-wise KV outer products (associative prefix scan = App. H "prefix scan"). Momentum decay folded in via `cumprod`/`cumsum` identity (the $\mathcal{C}$ mask in chunked form). This is the source of the up-to-$4\times$ throughput.
- **Deviation/detail**: paper's App. H matmul form $(\Phi(Q)\Phi(K)^\top \odot \mathcal{C})V$ is the *attention-matrix* view; the code uses the **mathematically-equivalent state/cumsum view** (materialize per-chunk state $W_1$, then apply) — block-causal chunked, not full $NL\times NL$ attention matrix (more memory-efficient).
- Muon applied per-chunk-block after the scan (consistent with App. E "element-wise to each KV outer product").

### NVS variants — `lact_nvs/_lact_ttt/`
Same scheme, **finer ablation granularity** directly matching Table 2 steps. Dispatcher (`__init__.py`) routes `ttt_loss_type` ∈ {`dot_product`, `ga_dot_product`, `only_w1`, `only_w1_no_wn`, `only_w1_straight_qk`, `..._no_wn`, `..._no_lr1`, `..._no_lr1_no_wn`, `..._no_lr1_no_muon`, `..._no_lr1_no_wn_muon`} — the exact Variant-1→6 trajectory. NVS ops take an explicit `ttt_ua_order` (list of `(start,end,update,apply)` tuples) for flexible update/apply scheduling (bidirectional/multi-pass), e.g. `only_w1_straight_qk_no_lr1_no_wn_muon.py`:
```python
for start, end, update, apply in ttt_ua_order:
    if update:
        lr1i = lr1[:, start:end, :] * 0.0 + 1.0       # no_lr1 ⇒ constant 1.0
        hidden = ki                                    # straight_qk
        w1_grad = (hidden * lr1i).transpose(-1,-2) @ vi   # k^T v outer product
        w1 = w1 + w1_grad                              # no_wn ⇒ no normalization
    if apply:
        oi = qi @ w1_now                               # straight query
```
⇒ literally `o = q (W + Σ k_i^T v_i)` = **standard linear attention** (the no_wn_muon name = Variant 6 sans grad-norm). Confirms paper's reduction endpoint exactly.

### ViTTT — `ttt_block.py`
`class TTT` has the **two independent fast-weight components** (paper §5.4):
- `inner_train_simplified_swiglu(k,v,w1,w2)`: GLU $f(x)=(xw_1)\odot\mathrm{silu}(xw_2)$, **hand-derived closed-form gradient** (no autograd — comment notes vector-valued per-head loss can't use `autograd.backward`):
  ```python
  e = -v / v.shape[2] * self.scale                 # dL/dv_hat (dot-product loss)
  g1 = k.transpose(-2,-1) @ (e * a)                # outer products = KV bindings
  g2 = k.transpose(-2,-1) @ (e * z1*(sig*(1+z2*(1-sig))))  # silu' backprop
  g1 = g1/(g1.norm(dim=-2)+1.0); g2 = g2/(g2.norm(dim=-2)+1.0)  # grad clipping (≈ normalization, ablated as "no_wn_muon")
  w1, w2 = w1 - lr*g1, w2 - lr*g2                  # GD step
  ```
- `inner_train_3x3dwc(k,v,w)`: $3\times3$ depthwise conv fast weight, gradient via cross-correlation; `'prod'` impl does explicit 9-offset shifted dot products `(k[shifted]*e).sum()` ⇒ **the spatially-local linear-attention form of App. G** (each output attends to KV via $3\times3$ neighborhood overlap).
- `forward`: split `qkv` into two q/k/v sets (one per branch), inner-train both, **apply updated weights to queries** (`(q1@w1)*silu(q1@w2)`, `conv2d(q2, w3)`), concat, project. `scale = 9**-0.5` (equivalent head_dim of $3\times3$ conv).

### Notable implementation details / deviations from paper
- **Hand-coded backward everywhere** (LaCT `silu_backprop`, ViTTT closed-form `g1,g2,g`) — no autograd through inner loop, exploiting that KV-binding gradient = simple outer products. This is *why* the linear-attention equivalence is computationally exact and cheap.
- **Block-causal "apply-then-update" chunking** (chunk_size default 2048 LLM): output at chunk uses *pre-update* weights ⇒ the causal-shift; last chunk apply-only. Not spelled out as such in paper math (which is per-token) but is the standard LaCT chunked realization.
- **Per-token LR** parameterized Mamba-style: `lr = softplus(lr_proj(h) + base_lr_inv)`, split into `lr0,lr1,lr2` (one per fast-weight matrix). `no_lr1` ablation sets it to constant 1.0 *while keeping it in the autograd graph* (`*0.0 + 1.0`).
- **Frozen-weight autograd hack**: `+ w0.sum()*0.0 + w2.sum()*0.0` keeps untrained kernel params connected to the graph in only_w1 variants (avoids "param did not receive grad" errors).
- **Parallel form uses state/cumsum view**, not the attention-matrix matmul of App. H (equivalent, more memory-friendly); requires static kernel + no weight-norm (matches App. I non-reducibility conditions).
- ViTTT grad clipping `g/(g.norm+1.0)` ≈ a normalization that the paper's ablation removes ("ViTTT does not use gradient orthogonalization, so we ablate gradient normalization instead", Table 2 footnote *).
