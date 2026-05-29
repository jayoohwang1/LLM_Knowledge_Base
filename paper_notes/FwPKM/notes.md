# Fast-weight Product Key Memory (FwPKM)

> **Source**: arXiv 2601.00671v2 (2026-02-22), Sakana AI — Tianyu Zhao & Llion Jones.
> **Code**: https://github.com/SakanaAI/fast-weight-product-key-memory
> **TL;DR**: Turn the *static* Product Key Memory (PKM) of Lample et al. 2019 into a *dynamic fast-weight episodic memory*. Keep PKM's sparse product-key **retrieval** unchanged; redesign the **memorization** to update K/V matrices online via chunk-level gradient descent (TTT-style). Activates only a tiny slot subset per token → vast storage at fixed low per-token compute. Trained on 4K seqs, generalizes to 128K NIAH; iterative rereading boosts NIAH retrieval from <10% to >70%.

---

## 1. Motivation: the capacity ↔ compute tension

- **Sequence layers = associative memory**: encode past tokens into storage (*memorization*), query at predict time (*retrieval*).
- **Landscape trade-off**:
  - **Softmax attention** — stores K/V per token, retrieves by query-vs-all-keys weighted sum. High-fidelity but $O(T^2)$ compute, $O(T)$ KV cache.
  - **Linear/RNN/SSM** (Linear Attention, Mamba2, DeltaNet, Gated DeltaNet) — compress history into *fixed-size* state w/ learned update (Hebbian / delta-rule). Sub-quadratic, but **fixed state caps retainable info**.
  - **TTT / Titans** — replace fixed state matrix w/ a **fast-weight MLP** updated by gradient descent at inference (minimize reconstruction loss). But **dense MLP compute scales w/ param count** → to store lots, MLP must be big → updating/querying a big dense MLP frequently is infeasible. **Scaling bottleneck = dense structure.**
- **FwPKM idea**: use **sparse** associative memory (PKM) as the fast-weight substrate. PKM holds millions of slots but activates a tiny subset/token → can be huge w/o per-token compute blowing up. Make it *dynamic* by gradient-updating its matrices online.
- **Conceptual memory split**:
  - **Slow weights** (frozen at inference) = **Semantic Memory** (dataset-wide facts).
  - **FwPKM fast weights** (updated at inference) = **Episodic Memory** (context-specific bindings).

**Contributions**: (1) architecture — "Sparse TTT" gradient updates over sparse memory, w/ modifications to make sparse fast weights stable; (2) performance — robust episodic memory in hybrid backbones, NIAH to 128K from 4K training, PPL drops on LC64/Longbench; (3) interpretability — slots traceable to input tokens; cost analysis.

---

## 2. Preliminary: sparse KV memory & PKM (Lample et al. 2019)

### 2.1 Sparse Top-$k$ KV memory
- Key matrix $K\in\mathbb{R}^{N\times d_k}$, value matrix $V\in\mathbb{R}^{N\times d_v}$, $N$ = #slots.
- Score: $s_i=\mathbf{q}^\top K_i$. Select $\mathcal{I}=\mathrm{Top}\text{-}k(\{s_i\})$. Normalize $\{s'_i\}=\mathrm{softmax}(\{s_j\}_{j\in\mathcal{I}})$. Read $\hat{\mathbf{v}}=\sum_{i\in\mathcal{I}}s'_i V_i$.
- **Problem**: Top-$k$ cuts *accessed* rows but still **scores all $N$ keys** → $O(N)$, prohibits $N\approx10^6$.

### 2.2 Product Key Memory (PKM)
- Factorize keys into **two sub-codebooks** to cut scoring cost.
- Split query $\mathbf{q}=[\mathbf{q}^{(1)};\mathbf{q}^{(2)}]$, $\mathbf{q}^{(m)}\in\mathbb{R}^{d_k/2}$; sub-key matrices $K^{(1)},K^{(2)}\in\mathbb{R}^{\sqrt N\times d_k/2}$.
- Per-codebook score: $s^{(1)}_i=\mathbf{q}^{(1)\top}K^{(1)}_i$, $s^{(2)}_j=\mathbf{q}^{(2)\top}K^{(2)}_j$.
- Per-codebook Top-$k$: $\mathcal{I}^{(1)},\mathcal{I}^{(2)}=\mathrm{Top}\text{-}k$ over $\sqrt N$ each.
- Search **restricted Cartesian product** $\mathcal{I}^{(1)}\times\mathcal{I}^{(2)}$ w/ additive scores $s_{i,j}=s^{(1)}_i+s^{(2)}_j$.
- Final Top-$k$ pairs → softmax → read from $V$; pair $(i,j)$ maps to row $((i-1)\sqrt N + j)$ (1-indexed).
- **Win**: index $N=10^6$ slots scoring only $2\sqrt N=2\times10^3$ candidates instead of $10^6$. Ideal high-capacity associative memory.

---

## 3. Method: Fast-weight Product Key Memory

**Core**: PKM is a slow-weight sparse channel mixer (frozen at inference → semantic only). FwPKM keeps **retrieval unchanged**, redesigns **memorization** by updating memory matrices online via a **local reconstruction objective over chunks** (TTT spirit).

### 3.1 Overview / architecture (Fig. 1)
- Process hidden states $\mathbf{h}_{1:T}$ in **chunks of size $C$**.
- An FwPKM **layer is placed *before* a token mixer** (self-attn or Gated DeltaNet layer), inside the residual stream (like an extra "MLP-position" module before the mixer).
- Within each chunk, FwPKM:
  1. **construct** query/value/scalar-gate from slow-weight projections,
  2. **retrieve** sparse value prediction via PKM using *current* fast weights,
  3. **write**: update fast weights by optimizing local objectives over the chunk.
- **Causality preserved**: fast-weight updates applied **after** processing the chunk → within-chunk predictions depend only on fast weights from *previous* chunks.
- **Parameter roles (color-coded in Fig.1)**:
  - **Slow-weight params** (blue): projection linears + RMSNorms ($\phi$). Updated *only at training time* via global LM loss.
  - **Fast-weight value param** $V$ (red): updated at train **and** inference by **memorization loss** $\mathcal{L}_{\text{mem}}$.
  - **Fast-weight key params** $K^1,K^2$ (green): updated at train **and** inference by **addressing loss** $\mathcal{L}_{\text{addr}}$.

### 3.2 Forward pass: inputs & retrieval
- Slow-weight ($\phi$) projections per token $t$:
$$\mathbf{q}_t=\mathrm{Linear}^q_\phi(\mathrm{RMSNorm}^q_\phi(\mathbf{h}_t)),\quad
\mathbf{v}_t=\mathrm{Linear}^v_\phi(\mathrm{RMSNorm}^v_\phi(\mathbf{h}_t)),\quad
g_t=\sigma(\mathrm{Linear}^g_\phi(\mathrm{RMSNorm}^g_\phi(\mathbf{h}_t)))$$
  - $\mathbf{q}_t\in\mathbb{R}^{d_k}$, $\mathbf{v}_t\in\mathbb{R}^{d_v}$, $g_t\in(0,1)$ scalar gate.
- **Sparse retrieval** = same PKM as §2.2 w/ current fast weights $\theta=\{V,K^1,K^2\}$:
$$\hat{\mathbf{v}}_t=\mathrm{PKM}(\mathbf{q}_t;\theta)=\sum_{i\in\mathcal{I}_t}s'_{t,i}V_i$$
- **Gated residual output** (episodic not always needed → interpolate retrieved value w/ residual value path):
$$\mathbf{o}_t=g_t\cdot\hat{\mathbf{v}}_t+(1-g_t)\cdot\mathbf{v}_t,\qquad
\mathbf{o}'_t=\mathrm{Linear}^o_\phi(\mathrm{RMSNorm}^o_\phi(\mathbf{o}_t))$$
  - $g_t\to1$ ⇒ rely on fast-weight episodic memory; $g_t\to0$ ⇒ rely on slow weights / residual.

### 3.3 Backward pass A — Memorization (learn to reconstruct), updates $V$
- **Chunk-level local objective**: make retrieval reconstruct target values from the same stream. Gated MSE over chunk of size $C$:
$$\mathcal{L}_{\text{mem}}=\sum_{t=1}^{C}\tfrac12\,g_t\,\lVert\mathbf{v}_t-\hat{\mathbf{v}}_t\rVert_2^2$$
- **Why MSE + ½**: gradient w.r.t. prediction $\nabla_{\hat{\mathbf v}}\tfrac12\lVert\mathbf v-\hat{\mathbf v}\rVert^2=-(\mathbf v-\hat{\mathbf v})$ → a unit step would *rewrite* the prediction to the target ($\hat{\mathbf v}\leftarrow\hat{\mathbf v}-\nabla=\mathbf v$). FwPKM doesn't update $\hat{\mathbf v}$ directly; it updates the underlying value rows $V_i$, but the residual $(\mathbf v-\hat{\mathbf v})$ is the **explicit "rewrite" signal** → fast-weight update behaves like memory rewriting.
- **Gradient aggregation (consensus over competing writes)**: many chunk tokens may write to the same row. Let $N^{\text{read}}_i$ = #times row $V_i$ accessed in the chunk. Normalize per-row grad:
$$\nabla^{\text{agg}}_{V_i}=\frac{1}{N^{\text{read}}_i}\nabla_{V_i}\mathcal{L}_{\text{mem}}$$
- **Value update** (gradient *descent*, LR folded into magnitude):
$$V_i\leftarrow V_i-\nabla^{\text{agg}}_{V_i}$$

### 3.4 Backward pass B — Addressing (anti-collapse), updates $K^1,K^2$
- **Slot collapse**: sparse memory tends to use only a small slot fraction. Fix via auxiliary objective encouraging **balanced slot usage on average across a chunk** (not per-query uniformity).
- Let $\mathbf{s}'^1_t,\mathbf{s}'^2_t\in\mathbb{R}^{\sqrt N}$ = normalized Top-$k$ score vectors over the two sub-key sets (unselected = 0).
- **Marginal slot-usage distributions** (chunk-averaged):
$$\bar{\mathbf p}^1=\frac1C\sum_{t=1}^C\mathbf s'^1_t,\qquad \bar{\mathbf p}^2=\frac1C\sum_{t=1}^C\mathbf s'^2_t$$
- **Addressing loss = negative entropy** (maximize entropy of marginal usage → spread load):
$$\mathcal{L}_{\text{addr}}=-H(\bar{\mathbf p}^1)-H(\bar{\mathbf p}^2)=-\sum_i\bar p^1_i\log\bar p^1_i-\sum_i\bar p^2_i\log\bar p^2_i$$
- **Key updates**: $K^1\leftarrow K^1-\nabla_{K^1}\mathcal{L}_{\text{addr}}$, $K^2\leftarrow K^2-\nabla_{K^2}\mathcal{L}_{\text{addr}}$.

### 3.5 Practical choices (key for stability/perf; ablated in App. D)
- **Lookahead value targets** (App. A.1): pair query $\mathbf q_t$ with the *next* token's value $\mathbf v_{t+1}$ during memorization → store info about the immediate *future* of similar queries → aligns local reconstruction w/ global next-token objective. Retrieval def $\hat{\mathbf v}_t=\mathrm{PKM}(\mathbf q_t)$ unchanged; only shift the target:
$$\mathcal{L}^{\text{LA}}_{\text{mem}}=\sum_{t=1}^{C-1}\tfrac12 g_t\lVert\mathbf v_{t+1}-\hat{\mathbf v}_t\rVert_2^2$$
  - Update applied **after** whole chunk (lookahead doesn't leak within chunk). At **train time** drop last predicted value per chunk for efficiency (drops 8 tokens for a 4096 seq). At **inference** keeps a value cache accumulating $(\mathbf v_{t+1},\hat{\mathbf v}_t)$ pairs; when predefined update chunk size reached (e.g. 512 in PPL eval, haystack length in NIAH), consumes stored pairs to update fast weights.
- **Inverse-Distance Weighting (IDW) scoring** (App. A.2): dot-product score lets a key inflate score by growing magnitude w/o being *close*. Replace **only the query–key scoring inside Top-$k$ selection** (per sub-key set) by
$$s^{\text{IDW}}_i=-\log\!\big(\epsilon+\lVert\mathbf q-K_i\rVert_2^2\big),\quad \epsilon=10^{-3}$$
  - Euclidean-distance grads push keys to become **prototypes / clustering centroids** of query clusters. All other PKM retrieval steps unchanged. **Dot-product score → memory collapse + loss divergence**, so IDW is essentially required (excluded from ablations for this reason).
- **Target normalization** (App. A.3): z-score normalize $\mathbf v_t$ along feature dim for stability. **No gradient clipping** for fast-weight updates (unlike slow-weight training) → lets FwPKM match scale of unbounded targets.
- **Loss reduction / step size** (App. A.4): **sum** (not mean) over tokens & features in $\mathcal L_{\text{mem}}$ so update magnitude doesn't shrink w/ chunk size $C$ or $d_v$. Mean reduction would scale element grad by $1/(Cd_v)$ → effectively changes fast-weight step size.
- **Gating-weighted memorization** (App. A.5): the same gate $g_t$ used in output interpolation (Eq.12) also weights MSE in $\mathcal L_{\text{mem}}$ — $g_t$ = how much model relies on memory here → natural importance weight for which tokens write more strongly.

### Pseudocode (App. B, Algorithm 1) — per chunk
```
for t = 1 to T step C:
  # Phase 1: Forward (inference)
  h_chunk = h[t:t+C]                                   # (C, d)
  q = Linear_q(RMSNorm_q(h_chunk))                     # (C, d_k)
  v = Linear_v(RMSNorm_v(h_chunk))                     # (C, d_v)
  g = sigmoid(Linear_g(RMSNorm_g(h_chunk)))            # (C, 1)
  q1, q2 = split(q)                                    # (C, d_k/2) each
  for m in {1,2}:                                      # IDW scoring
    s_m = -log(1e-3 + ||q_m - K_m||^2)                 # (C, sqrt(N))
    I_m = topk(s_m)                                    # (C, k)
  S_target = { s1_i + s2_j | (i,j) in I1 x I2 }        # k^2 candidates/token
  I = topk(S_target)                                   # final k indices/token
  v_hat = sum_{(i,j) in I} softmax(S_target)_(i,j) * V[flat-idx(i,j)]   # (C, d_v)
  o_chunk = g * v_hat + (1-g) * v                      # gated residual
  # Phase 2: Memorization backward
  v_target = shift(v, +1)                              # lookahead v_{t+1}
  L_mem = sum_{step=1..C-1} 0.5 * g_step * ||v_target - v_hat_step||^2
  grad_agg_Vi = (1/N_read) * grad_Vi L_mem             # aggregate by slot usage
  V = V - grad_agg_V
  # Phase 3: Addressing backward (anti-collapse)
  for m in {1,2}:
    p_bar_m = AvgSlotUsage(I_m, chunk)                 # (sqrt(N),)
    L_addr_m = -H(p_bar_m)                             # maximize entropy
    K_m = K_m - grad_Km L_addr_m
return o[1:T]
```

---

## 4. Experiments

### 4.1 Setup
- **Models**: 12-layer LMs following **QwenNext** design, scaled to ~112M params. Interleave **Gated DeltaNet (GDN)** + gated softmax attention.
  - **Baseline layouts**: `GDN` (all GDN), `GDN+SWA` (GDN + Sliding Window Attn w=512 @ 3:1), `GDN+FA` (GDN + Full Attn @ 3:1), `FA` (full attn all layers).
- **Memory module insertion**: PKM / FwPKM inserted at **layers 2, 6, 10** (`@2,6,10`), *before* the token mixer. Notation e.g. `PKM@6 + FwPKM@2,10` = PKM at layer 6, FwPKM at 2 & 10.
- **Memory size**: both PKM & FwPKM have $512^2=262{,}144$ slots.
  - **PKM**: effectively Top-128 (4 heads × 32 slots/head).
  - **FwPKM**: Top-8 (1 head × 8 slots/head) — chosen for efficiency/perf balance.
- **Extra baselines**:
  - **FwMLP** (`FwMLP@2,6,10`): replace PKM in FwPKM w/ a **SwiGLU MLP** holding 3 fast-weight matrices (up/gate/down) + biases. Dense fast-weight ablation. Uses mean reduction, no grad aggregation, no addressing loss.
  - **LaCT** (Zhang et al. 2025b): improved TTT baseline — SWA + fast-weight SwiGLU MLP + slow-weight SwiGLU MLP in every layer; fast weights updated w/ SGD+momentum + L2 weight norm.
- **Data**: 5B tokens **LongContext64 (LC64)** (docs >64K tokens, from RedPajama V2) + 5B **Fineweb-Edu**. Train **seq len 4K**.
- **HParams** (Table 2): hidden 768, 12 layers, vocab 32000. FwPKM: key dim 512, value dim 512, 1 head, Top-8, $512^2$ slots, chunk 512. Attn: head dim 64, 12 q-heads, 4 kv-heads. GDN: conv 4, head 64, 8 heads. PKM: key/val 512, 4 heads, Top-32, $512^2$ slots. FwMLP: in 512, hidden 2304, out 512, LR $\eta=0.1$. LaCT: SWA 2048, TTT chunk 512. Training: maxLR 1e-3, minLR 1e-4, global batch 128, micro 8, 100 warmup, 20000 steps, weight decay 0.1. **FwPKM runs Float32**; others BF16 via FlashAttention + FLA kernels. 4×H100.
  - **Key impl note**: FwPKM/FwMLP use a **shared set of fast weights across all sequences in a minibatch**, whereas LaCT/TTT keep **per-sequence** fast weights.

### 4.2 PPL evaluation (Fineweb-Edu / LC64 / LAMBADA, 8M tokens each)
- **Protocol**: eval 4K seqs in original order, **batch 1, no fast-weight reset between adjacent segments** → fast weights capture cross-segment deps.
- **Finding 1 — FwPKM & PKM are complementary**:
  - FwPKM ↓ PPL strongly on **long-context** (LC64, LAMBADA) → **episodic memory**.
  - PKM ↓ PPL most on **Fineweb-Edu** (knowledge-intensive short ctx) → **semantic memory**.
  - **PKM + FwPKM combo = best across all metrics** (orthogonal limitations).
  - Sample numbers (`GDN`): vanilla 19.73/10.12/32.91 → `PKM` 18.27/9.22/29.97 → `FwPKM` 18.09/8.89/26.73 → `PKM+FwPKM` 17.39/8.54/25.91.
- **Finding 2 — FwPKM competes w/ Full Attention**: on `FA`/`GDN+FA` (unrestricted full attn) PPL gains from FwPKM are **marginal** — gating analysis shows models learn to **ignore FwPKM** (gating clusters near 0, Fig.3).
- **Finding 3 — restricting full attn helps FwPKM use (pSWA)**: at *training time*, impose SWA window 512 on full-attn layers w/ prob 0.9 (suffix **pSWA**). Forces less long-range perception → models learn to use FwPKM more (higher gating mass). Minimal PPL impact but big difference in downstream (NIAH).

### 4.3 Needle-in-a-Haystack (NIAH)
- Update fast weights **after reading full haystack** (and after each reread); **no updates during answer generation**.
- Basic: 500 samples from LAMBADA; 4K haystack, 5 needles (unique 4-char key + 6-digit value), question asks for one specific key's value. Also 8K/32K/128K test sets.
- **Iterative Memorization ($n$-iter NIAH)**: forward the same haystack $n$ times before answering; process chunk $C$ set to haystack length → updates memory once per full read. Basic = 1-iter.
- **Finding 4 — iterative reading boosts retrieval**: GDN / GDN+SWA: 1-iter often insufficient; **2-iter jumps <10% → >50%**; more passes → >70%. Confirms FwPKM exploits TTT to consolidate episodic memory, **surpassing softmax attention**. Effective iterative memorization correlates w/ high gating (Fig.3).
- **Finding 5 — generalizes to 128K** despite 4K training. **FA baselines degrade rapidly** on unseen lengths; FwPKM stays robust.
  - e.g. `GDN | FwPKM@2,6,10` at 4K reaches 88.6% (top iter), 128K reaches 75.8%; `FA` collapses to single-digit % at long ctx.

### 4.4 Longbench (5 tasks: 2WikiMultihopQA, HotpotQA, MultiFieldQA-en, MuSiQue, NarrativeQA)
- 950 examples, avg len 17421 / max 81616 tokens. Pretrained-only models → use **PPL on answer tokens** (not F1). 1-iter protocol (no updates during generation). LaCT omitted (PPL >100).
- **Finding 6 — FwPKM ↓ answer-token PPL** on GDN / GDN+SWA / GDN+FA where FwPKM actively used.
  - e.g. 2WikiMultiQA `GDN`: 19.50 → `FwPKM` 13.38; HotpotQA `GDN`: 36.25 → 21.38.

### 4.5 Continual learning (adapt-and-test)
- 6 Pile domains (DM Mathematics, FreeLaw, PhilPapers, PubMed Central, Ubuntu IRC, USPTO Backgrounds). Per domain: 4M adapt + 4M eval (no overlap). Run 6 sequential adapt stages (fast weights update, **slow weights frozen**); report ΔNLL vs initial NLL.
- **Finding 7 — adapts fast but poor retention**: after adapting to a domain, that domain's NLL ↓ (fast weights store domain knowledge). But gains **don't persist across stages** → previously stored fast-weight knowledge gets **flushed/overwritten** by newer domains. Motivates memory **retention/consolidation** mechanisms.

---

## 5. Interpretability (Sec. 5)
- **Inherent interpretability**: memory slots explicitly written/read → trace retrieval back to specific input tokens.
- **5.1 NIAH probing** (`GDN+SWA | PKM@6+FwPKM@2,10`, 4-iter, 4K): trace Top-8 retrieved slots at each gen step; decode stored **query token / target value token / surrounding context**; mark `[HIT]` if slot stores ground-truth next token (Fig.7).
  - **High-precision retrieval**: majority of retrieved slots hold correct target tokens; model surfaces specific needle (ID 8320 @ depth 89.66%) producing correct value "964876".
  - **Robust aggregation**: despite individual slot errors, **consensus across 16 slots (2 layers × 8)** generates correct value → robust distributed storage.
- **5.2 Selective gating** (Wikipedia "Sakana AI" article, unseen in pretrain; chunk 512→32 for fine adaptivity; Fig.8):
  - **Layer specialization**: **lower-layer FwPKM** keeps high gating across nearly all tokens → general-purpose recent-history buffer. **Higher-layer FwPKM** highly selective.
  - **Novelty detection**: higher-layer (L10) gating **spikes for rare named entities / proper nouns** ("Sakana AI", "David Ha", "Llion Jones", "Ren Ito") → model distinguishes general linguistic patterns (slow weights) from novel context-specific entities (fast weights).

---

## 6. Cost analyses (Table 1)
- Metrics: **Active params** (used per forward), **FLOPs** (theoretical), **FLOPS** (empirical T/sec), **Samples/sec**.
- **Sparsity → low compute**: total params ~5× baselines (e.g. `GDN` 112M total / `FwPKM` 519M total) but **active params stay comparable** (GDN 112M vs FwPKM 116M). Theoretical FLOPs remarkably efficient; sparse PKM/FwPKM sometimes **fewer FLOPs than dense MLP layers**.
- **Implementation gap**: wall-clock (Samples/sec) & hardware util (FLOPS) reveal FwPKM **runs slower than dense counterparts** (e.g. GDN 54 samples/s vs FwPKM ~19). Cause = **kernel maturity**: softmax attn / GDN have FlashAttention / FlashLinearAttention; sparse fast-weight ops rely on unoptimized impls. **Dedicated hardware-aware sparse kernels = critical future work.**

---

## 7. Related work (positioning)
- **Softmax attn & linear variants**: attn = powerful associative memory but quadratic; Linear Attention / Mamba(2) / DeltaNet / Gated DeltaNet / RWKV7 reduce to linear by reordering associations & compressing to fixed state. Memory Mosaics / hierarchical mem improve attn.
- **Fast weights & TTT**: rooted in Schmidhuber 1992 & Ba et al. 2016 ("fast weight programmers"). Revitalized by **TTT (Sun et al. 2025)** & **Titans (Behrouz)** — update params at inference via grad descent. Unifying frameworks **MIRAS**, **Test-Time Regression** name design axes (memory architecture, memorization rule, retention rule, optimizer). **FwPKM = a novel memory architecture + memorization rule tailored to structural sparsity.**
- **Hybrid architectures**: combine quadratic attn fidelity + linear efficiency. Samba, KimiLinear, QwenNext (interleaved design FwPKM builds on), Artificial Hippocampus Networks (AHN). FwPKM combines GDN (linear) + softmax attn + PKM (slow sparse mem) + FwPKM (fast sparse mem) = versatile memory system.
- **Memory models**: Memory Networks; even MLPs as memory (Csordás 2023). Sparse access for scaling: **PKM (Lample 2019; Berges 2025)**, **PEER (He 2024)**, **Ultra Sparse Memory** (extends PKM). FwPKM builds on PKM but turns **static "slow" mem → dynamic "fast" mem**.
- **Continual & episodic learning**: Lin et al. 2025 — parameter-efficient PKM finetuning mitigates forgetting (updates slow weights via self-supervision = semantic dimension). **FwPKM = different dimension: update episodic memory by learning fast weights online.** Nested Learning & TNT stack multiple fast-weight layers w/ updates at varying frequencies → future: hybrid varying-size FwPKMs + Titans w/ nested optimization.

---

## 8. Conclusion & limitations
- **FwPKM = "Sparse TTT"**: unify PKM's large sparse storage + TTT's rapid adaptation → static retrieval module becomes dynamic context-responsive component. Memorize vast transient info, retrieve thousands of tokens later.
- **Strengths**: 4K→128K (32× extrapolation) NIAH w/o finetune; iterative reading <10%→>70%; true episodic memory selectively activating for novel entities, complementing slow-weight semantics.
- **Limitations**:
  - **Compute overhead** from online updates, bottlenecked by **lack of optimized sparse-update kernels**.
  - **Poor long-term retention** across domain shifts (catastrophic forgetting of fast weights) → need retention/consolidation mechanisms.

---

## Ablations (App. D, on `GDN+SWA | PKM@6+FwPKM@2,10`)
- **Variants**: `w/ 1 head × Top-32`, `w/ 4 heads × Top-8` (different Top-$k$/multi-head), `w/o value norm`, `w/o addr loss` (use MSE to update both K & V instead of entropy), `w/o gating`, `w/o loss weight` (use $g_t$ for output but not to weight MSE), `w/o lookahead`. (Dot-product instead of IDW → collapse/divergence, excluded.)
- **D.1 findings**:
  - **Removing lookahead = most harmful** to performance.
  - Many techniques give slight PPL gains but **degrade memory utility → worse NIAH**.
  - **Addressing loss + gating-weighted MSE important for NIAH retrieval accuracy.**
  - Larger effective Top-$k$ slightly helps some tasks but **default 1 head × 8 slots = sweet spot** (perf vs compute).
  - Ablation NIAH (Fig.11): `w/o lookahead` collapses NIAH (4K: 8.2% vs full 80.8%); `w/o loss weight` & `w/o gating` also hurt.
- **D.2 Freezing fast weights** (Fig.14): freezing FwPKM's $V,K^1,K^2$ at eval **↑ PPL strongly on LC64/LAMBADA** (long ctx), Fineweb-Edu less affected → confirms fast-weight updates drive long-context gains.
- **D.3 Addressing metrics** (Table 4): **collision rate** $\sum_{N^{\text{read}}_i>1}N^{\text{read}}_i/N^{\text{total}}$, **coverage rate** $\sum_{N^{\text{read}}_i>0}1/N$, **KLD** vs uniform.
  - Slot use more balanced on long-ctx (LC64/LAMBADA) but still far from uniform.
  - ↑ heads or Top-$k$ → better evenness but ↑ collision.
  - **`w/o addr loss` → concentrated (collapsed) slot access** (high collision, low coverage, high KLD) → paired w/ degraded NIAH.
- **E.1 cross-sequence PPL** (Fig.15): FwPKM PPL-reduction curves jagged (exploit info from *previous* sequences, impossible for vanilla/PKM-only) vs PKM-only smooth curves (constant per-doc advantage).

---

## Relations / mental model
- **vs PKM**: identical retrieval (product-key sparse Top-$k$); FwPKM adds online K/V gradient updates → static semantic → dynamic episodic.
- **vs Fast-Weight Programmers / Schmidhuber 1992**: FwPKM = fast weights, but **sparse** (PKM) instead of dense outer-product/MLP. Memorization = grad descent on reconstruction (TTT) vs Hebbian.
- **vs Linear attention / DeltaNet**: those compress to *fixed-size dense* state; FwPKM = *huge sparse* state, sparsely addressed → decouples capacity from per-token compute.
- **vs TTT / LaCT / Titans**: same "update params at test time via grad descent" spirit, but substrate is sparse PKM not dense MLP → "Sparse TTT". Also: FwPKM shares fast weights across batch sequences; TTT/LaCT per-sequence.
- **vs episodic memory (neuro analogy)**: fast weights = hippocampal episodic store (context-specific bindings); slow weights = neocortical semantic store. Continual-learning experiments show the classic episodic forgetting problem.

---

## Implementation

> **Repo**: `fast-weight-product-key-memory/` (Sakana AI). Models built on HF `transformers` + a custom `Qwen3NextMem` model. Deps: `flash-attention`, `flash-linear-attention` (FLA), `causal-conv1d`, `deepspeed`. Tested Python 3.12.11.

### Layout / where things live
- `src/models/fwpkm/fwpkm.py` — **`FastWeightProductKeyMemory`** core module (sparse addressing + fast-weight update). **Central file.**
- `src/models/fwpkm/fwmlp.py` — **`FastWeightMLP`** (dense SwiGLU fast-weight baseline = `FwMLP`).
- `src/models/fwpkm/xformer_embeddingbag_grad_wrapper.py` — **double-backward-capable embedding-bag** (sparse gather + weighted sum), enables 2nd-order grads needed for train-time backprop through the fast-weight update.
- `src/models/fwpkm/pkm_legacy/xformer_embeddingbag.py` — fused backward kernel `embedding_bag_bw_rev_indices`.
- `src/models/qwen3_next_mem.py` — **`Qwen3NextMem*`** full model: config, decoder layer, integration of PKM/FwPKM, GDN, attention. (1732 lines.)
- `src/models/pkm/` — static PKM (`HashingMemory` in `memory.py`).
- `src/models/lact_model/` — LaCT baseline (Triton kernels).
- `cfgs/model_config/{main,baseline,ablation}/*.json` — HF configs. `src/pretrain.py` train entry; `scripts/` shell drivers.

### Core module: `FastWeightProductKeyMemory` (`fwpkm.py`)
- **Fast-weight params (the only nn.Parameters)**:
  ```python
  self.keys   = nn.Parameter(torch.zeros(2 * heads * subsize, k_dim // 2))  # K^1,K^2 packed
  self.values = nn.Parameter(torch.zeros(subsize**2, v_dim))               # V, N = subsize^2
  ```
  - `subsize=512` ⇒ `size = 512^2 = 262144` value slots. `keys` holds **both** sub-key codebooks packed into one tensor (`heads, 2, subsize, half`).
  - Init: keys uniform $\pm1/\sqrt{d_k}$, values normal std $d_v^{-1/2}$.
- **`get_fw_named_params()`**: snapshots params as fp32 (`fp32_fw=True`) into an `OrderedDict` — fast weights are tracked as a *functional* dict that gets repeatedly replaced (out-of-place) each chunk, not in-place `.data` mutation, so autograd can build a graph through updates at train time.

#### Sparse addressing (`get_indices`, `compute_qk_scores`)
- Matches paper §2.2 exactly:
  ```python
  keys = keys.view(heads, 2, -1, half); keys1 = keys[:,0]; keys2 = keys[:,1]
  q1 = query[:,:,:half]; q2 = query[:,:,half:]
  scores1 = compute_qk_scores(q1, keys1)   # (BT, heads, subsize)
  scores2 = compute_qk_scores(q2, keys2)
  topk_scores1, topk_indices1 = scores1.topk(topk, dim=2)   # per sub-key Top-k
  topk_scores2, topk_indices2 = scores2.topk(topk, dim=2)
  all_scores  = topk_scores1[...,None] + topk_scores2[...,None,:]   # k x k additive Cartesian
  all_indices = topk_indices1[...,None]*subsize + topk_indices2[...,None,:]  # flat row = i*subsize + j
  scores, best = all_scores.view(BT,heads,topk**2).topk(topk, dim=2)  # final Top-k pairs
  indices = all_indices.gather(2, best)
  ```
  - 0-indexed flat row `i*subsize + j` (paper uses 1-indexed `(i-1)√N + j`).
- **IDW scoring** (`compute_qk_scores`, default `qk_score_type="idw"`):
  ```python
  dists = torch.cdist(q_perm, k, p=2)          # Euclidean
  scores = -torch.log(1e-3 + dists.pow(2))     # eps = 1e-3 hardcoded (matches paper A.2)
  ```
  - `"dot_product"` available (`einsum("bhd,hnd->bhn")`) but causes collapse → unused.
- **Retrieval (`retrieve_values`)**: Top-$k$ scores `softmax`-normalized over the flattened `heads*topk` dim (`score_transform`, also supports silu/relu), then `xformers_embedding_bag(indices, values, per_sample_weights=normed_scores)` does the sparse weighted sum $\hat{\mathbf v}=\sum s'_i V_i$.

#### Memorization update (`compute_mem_loss`, `update_fw`)
- **MSE loss**, per-token, gated + masked:
  ```python
  mem_losses = F.mse_loss(hyp, ref, reduction="none").mean(dim=-1)  # mean over features
  if loss_mask:    mem_losses *= loss_mask
  if loss_weights: mem_losses *= loss_weights   # loss_weights = gate g_t (gating-weighted memorization)
  ```
- **`update_fw`** — the heart of "Sparse TTT":
  - `mem_loss *= 0.5` (the ½ factor, Eq.13).
  - If `mem_grad_to_values_only=True` (default): **mem grad flows only into `values`**, **addr grad only into `keys`** — computed as two separate `torch.autograd.grad` calls. (`create_graph=self.training` ⇒ train-time double-backprop, inference-time no graph.)
  - **Grad rescale for values** undoes the mean reductions to emulate paper's *sum* reduction + per-row aggregation (paper §3.3, A.4):
    ```python
    scale = num_samples * v_dim * topk * heads      # undo mean over samples & features, scale by k*heads
    g = g * scale
    idx_binc = torch.bincount(indices.view(-1), minlength=size)   # N_i^read per row
    binc_scale[binc_scale==0] = 1.0
    g = g / binc_scale.unsqueeze(-1)                # divide by slot access count -> consensus aggregation
    ```
  - **SGD update** (`update_fw_param`): `updated_p = p - lr*(g + wd*p)`. `learning_rate=1.0`, `weight_decay=0.0` defaults (LR folded into the rescale). `grad_clip=False` for fast weights (paper A.3 — no clipping).

#### Addressing update (`compute_addr_loss`, `compute_marginal_entropy`)
- **Negative marginal entropy** of chunk-averaged Top-$k$ slot-usage, per sub-key set:
  ```python
  topk_values,_ = M.topk(topk, dim=1); threshold = topk_values[:,-1:]   # mask to selected slots
  M = where(M >= threshold, M, -inf)
  P = softmax(M, dim=1)            # per-token slot dist
  P_bar = P.mean(dim=0)            # chunk-marginal usage  (paper p̄^m)
  marginal_entropy = -sum(P_bar * log(P_bar + 1e-8))
  addr_loss = addr_loss_weight * (-H(p̄^1) - H(p̄^2))   # minimize neg-entropy = maximize entropy
  ```
  - Keys updated via `update_fw` (SGD, no rescale on key grads — only `values` gets the bincount/scale treatment).
- **`fwpkm_addr_loss="me"`** (marginal entropy) default; ablation `no_addr_loss` sets it `None` **and** flips `mem_grad_to_values_only=False` so MSE updates *both* K and V (matches paper's "w/o addr loss" description).

#### Two forward modes (`forward` dispatch)
- **`forward_wo_past`** (training / full-sequence PPL eval, `past_key_values=None`):
  - Splits seq into chunks of `chunk_size`. Per chunk: retrieve w/ current fast weights → compute losses on `chunk[:-lookahead]` w/ **lookahead target** `v_{t+1}` → `update_fw` → push new fast-weight dict to `fw_named_params_list`.
  - **Causality**: chunk output uses fast weights from *before* the chunk's update; update applied after. `torch.enable_grad()` wraps the loop, `create_graph=training` keeps the chain differentiable end-to-end (slow weights learn to produce good q/v/g).
  - **Lookahead target construction**: drops last `lookahead` tokens per chunk (`update_end_idx=-lookahead`); shifts `ref_v` by `lookahead` to align $\hat{\mathbf v}_t \leftrightarrow \mathbf v_{t+1}$.
  - After loop: copies final fast-weight dict back into `self.parameters().data` (persists across calls).
- **`forward_w_past`** (incremental inference / streaming / NIAH, `past_key_values` set):
  - Maintains **queues** `(q, hyp_v, ref_v, gates, mask, ids)` accumulating $(\mathbf v_{t+1},\hat{\mathbf v}_t)$-type pairs across forward calls (paper A.1 "value cache").
  - **Only updates fast weights once the queue reaches `chunk_size`** (`if hyp_v_queue.size(1) < chunk_size: continue`). Then re-runs retrieval over accumulated `update_q` and applies one `update_fw`. After update, pops all but the last `lookahead` items from queues; carries remainder in `past_key_values.fwpkm_cache[layer]`.
  - This is how NIAH "update after reading full haystack" + "no update during generation" works: set `fwpkm_update_chunk_size` = haystack length so the single big read triggers exactly one consolidation, and 1-token generation steps never reach the threshold.
- **`@torch.compiler.disable(recursive=True)`** on `forward` — the custom autograd / functional-param dance isn't `torch.compile`-friendly.

#### Interpretability hooks (`write_ob_log`, `ob_mode`)
- When `ob_mode` set, logs per retrieved slot: `(q_token_id, v_token_id, prev_ctx_ids, next_ctx_ids, score)` into `ob_idx2log` — exactly the Fig.7 NIAH slot-trace visualization (query token / target value token / surrounding context / HIT).

### Sparse embedding-bag w/ double backward (`xformer_embeddingbag_grad_wrapper.py`)
- **Why it exists**: train-time loss backprops *through* the inference-time fast-weight update (which itself contains a `grad` of the embedding-bag read) → needs the embedding-bag's `grad_weight` to be itself differentiable (2nd order).
- `_XEmbeddingBagFn.forward`: gather `weight[indices]` → weighted sum w/ `per_sample_weights`.
- `.backward`: calls fast fused kernel `embedding_bag_bw_rev_indices` for first-order `(weight_grad, psw_grad)`, then **wraps `weight_grad` in `EmbeddingBagBW.apply`** which defines the VJP for the double-backward path:
  - `d_grad_output = Σ_j a_bj * ḡ[idx_bj]`, `d_a = <ḡ[idx_bj], grad_output_b>`.

### Integration into the model (`qwen3_next_mem.py`)
- **Backbone**: `Qwen3NextMemDecoderLayer` — QwenNext-style. Token mixer = `Qwen3NextMemGatedDeltaNet` (linear) or `Qwen3NextMemAttention` (full/sliding, gated, partial RoPE). MLP layers, except at `pkm_layers` the MLP is replaced by static `HashingMemory` (PKM).
- **FwPKM placement** (config `fwpkm_layers`, default `fwpkm_before_attn=True`): inserted in the residual stream **before** the token mixer:
  ```python
  if self.fwpkm is not None and self.config.fwpkm_before_attn:
      hidden_states, ... = self.fwpkm_forward(hidden_states, ...)   # residual add inside
  # then: input_layernorm -> token mixer -> residual add
  ```
- **`fwpkm_forward`** wires the paper's Eq.8-12:
  - `q = fw_q_proj(fw_q_norm(hidden))`; `v = fw_v_proj(fw_v_norm(hidden))` (both from `hidden`, `fwpkm_query_src=fwpkm_value_src="hidden"`).
  - **value compression**: `fwpkm_compress_value="zero_mean"` ⇒ z-score normalize value along feature dim = paper's target normalization (A.3). Applied via `compress_fwpkm_inputs` (`(x-mean)/std`).
  - **gate** `g_t` (Eq.10): `compute_gates(hidden)` = `sigmoid(mean(fw_out_gate(fw_out_gate_norm(hidden)), -1))`. **Note deviation**: gate linear outputs **8 dims then mean-reduces** to a scalar — comment says "out size cannot be 1,2,4 due to compiling error" (kernel workaround, not in paper).
  - Calls `self.fwpkm(q, ref_v, gates=g (if weight_loss_with_gates), chunk_size, loss_mask=attn_mask, past_key_values=fwpkm_cache[layer], token_ids=input_ids)`.
  - **Gated residual** (Eq.12): `out = out*g + v*(1-g)`; then `fw_out_proj(fw_out_norm(out))`; final `hidden = residual + out`.
- **Cache**: `Qwen3NextMemDynamicCache.fwpkm_cache[layer] = (q, hyp_v, ref_v, gates, mask, ids)` per layer; `get_fwpkm_cache_length()` reports the queue length of the first FwPKM layer.
- **Addr stats** (`compute_addr_stats`, `@torch.no_grad`): logs collision ratio / coverage ratio / KLD-vs-uniform from `bincount` of indices — the Table 4 metrics, computed live.

### Config / hyperparameters (`cfgs/model_config/main/l12-gdn-pkm6-fwpkm2_10.json`)
| Field | Value | Note |
|---|---|---|
| `hidden_size` / `num_hidden_layers` | 768 / 12 | ~112M base |
| `intermediate_size` | 2304 | MLP / FwMLP hidden |
| `head_dim`, `num_attention_heads`, `num_key_value_heads` | 64, 12, 4 | GQA |
| `linear_*` (GDN) | head 64, 8 k-heads, 8 v-heads, conv 4 | |
| `layer_types` | all `linear_attention` (GDN) here | SWA/FA variants set `sliding_attention`/`full_attention` at 4,8,12 |
| `pkm_layers` / `pkm_heads` / `pkm_topk` / `pkm_n_subkeys` | `[6]` / 4 / 32 / 512 | static PKM, 262144 slots, Top-128 effective |
| `fwpkm_layers` | `[2, 10]` | |
| `fwpkm_heads` / `fwpkm_topk` / `fwpkm_n_subkeys` | 1 / 8 / 512 | 262144 slots, Top-8 |
| `fwpkm_update_chunk_size` | 512 | $C$ |
| `fwpkm_qk_score_type` | `idw` | |
| `fwpkm_addr_loss` / `..._weight` | `me` / 1.0 | marginal entropy |
| `fwpkm_compress_value` | `zero_mean` | target norm |
| `fwpkm_target_lookahead` | 1 | $\mathbf v_{t+1}$ |
| `fwpkm_weight_loss_with_gates` | true | gating-weighted MSE |
| `fwpkm_out_fuse_gate` / `fwpkm_before_attn` | true / true | |
| `fwpkm_grad_clip` / `fwpkm_optimizer_type` / `lr` / `wd` | false / sgd / 1.0 / 0.0 | LR folded into grad rescale |
| `fwpkm_fp32_fw` | true | fast weights in fp32 even when model is bf16 |

- **Naming convention** in cfgs/scripts: `l12-<backbone>-pkm<L>-fwpkm<L>` e.g. `l12-gdn-pkm6-fwpkm2_10`. `swa_interleave`/`fa_interleave` set the sliding/full attention positions. `pSWA` (training-time SWA-on-FA prob 0.9) is an eval/train flag, not in these base configs.

### FwMLP baseline (`fwmlp.py`)
- `FastWeightMLP`: 3 fast-weight matrices + biases (`w_up,b_up,w_gate,b_gate,w_down,b_down`) = SwiGLU. Same chunked TTT update loop but **dense**: default `learning_rate=0.1`, `grad_clip=True`, mean reduction, **no addr loss / no per-row aggregation / no IDW** (paper §C: dense → averages over sample & feature dims). `lookahead_strategy` adds `"mean"` option (avg over a window) absent from PKM variant.

### Notable details / deviations vs paper
- **Gate dim-8 trick**: gate Linear outputs 8 then averages to scalar — a `torch.compile` workaround, not described in paper.
- **Two distinct update paths** (`forward_wo_past` chunked, `forward_w_past` queue-threshold) implement the train-vs-stream behaviors; paper presents one unified per-chunk algorithm.
- **`mem_grad_to_values_only`** flag explicitly routes mem-grad→V and addr-grad→K via separate `autograd.grad` calls (paper states this as two separate objectives; here it's literally two backward passes).
- **Grad rescale** (`num_samples*v_dim*topk*heads`, then `/bincount`) is how the code converts framework `mean` reduction into the paper's `sum`-reduction + $N_i^{\text{read}}$ aggregation — a single fused rescale rather than literally summing.
- **Shared fast weights across batch** (paper §C): `self.values`/`self.keys` are single tensors used for all batch sequences; confirmed — there is no per-sequence fast-weight dimension. (LaCT keeps per-sequence.)
- **`forward` is `torch.compile`-disabled**; the GDN/attention paths are compiled/kernel-accelerated — consistent with the paper's "kernel maturity gap" cost finding.
- **eps**: IDW `1e-3` (matches paper), marginal-entropy `1e-8`, l2norm/zero_mean `1e-6`.
