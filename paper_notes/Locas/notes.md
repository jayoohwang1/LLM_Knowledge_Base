# Locas: Your Models are Principled Initializers of Locally-Supported Parametric Memories

**Authors**: Sidi Lu, Zhenwen Liang, Dongyang Ma, Yan Wang, Haitao Mi, Dong Yu — Tencent AI Lab. arXiv 2602.05085v1 (Feb 2026).

**One-liner**: A test-time-training (TTT) parametric memory that is a **sideway low-rank FFN block** running in parallel to the backbone FFN, initialized *principledly* from the backbone's own activations/gradients/weights (not random), giving fast convergence + tiny param overhead + near-zero catastrophic forgetting.

---

## 1. Problem & Motivation

**Test-time adaptation** of LMs splits into two paradigms:
- **Non-parametric** = in-context learning (ICL): condition on prompt (demos, tool outputs, RAG). Stable, no weight updates, but:
  - **Bounded by context length**; degrades with formatting/ordering/distractors ("lost in the middle").
  - Controllability failures: **prompt injection / jailbreaks**.
- **Parametric** = test-time training (TTT): online weight updates via self-supervised signal at inference.
  - Can *internalize* info beyond prompt, but: extra optimization cost (multiple grad steps/token), needs careful objective to avoid catastrophic distribution shift.

**Core question**:
> Can we significantly improve the parameter & compute efficiency of TTT through **principled initialization** of the memory module?

**Thesis**: proper init dramatically accelerates convergence and slashes params needed for memorization. Prior TTT (e.g. TempLoRA) applies grad descent to **randomly-initialized** added params; Locas instead uses the **backbone's own behavior** to guide init of the new memory.

---

## 2. Key Reinterpretation: FFNs are Soft Look-up Table Memories

**Setup**: classical 2-layer FFN at layer $i$, token $t$, input activation $\mathcal{A}_t^i \in \mathbb{R}^d$ (post-attention, post-LN), intermediate dim $m$, nonlinearity $\phi$:

$$\text{FFN}(\mathcal{A}_t^i) = V^\top \phi(K^\top \mathcal{A}_t^i)$$

- $K \in \mathbb{R}^{d\times m}$ = **"key matrix"** (first layer), $V \in \mathbb{R}^{m\times d}$ = **"value matrix"** (second layer). Biases omitted.
- Let $k_j$ = $j$-th column of $K$, $v_j$ = $j$-th row of $V$. Then:

$$\text{FFN}(\mathcal{A}_t^i) = \sum_{j=1}^m \phi(\langle \mathcal{A}_t^i, k_j\rangle)\, v_j$$

**Interpretation** (builds on Geva et al. 2021, FFN = key-value memory):
- Each $(k_j, v_j)$ = a **memory slot**: $k_j$ decides *when* the slot fires (inner product w/ input), $v_j$ decides *what* content is retrieved, $\phi$ acts as activation gate.
- FFN = **unnormalized linear attention** (Katharopoulos 2020) with a *fixed parametric* set of $m$ K-V pairs:
  $$\text{FFN}(\mathcal{A}_t^i) = \sum_{j=1}^m \alpha_j(\mathcal{A}_t^i) v_j, \quad \alpha_j \doteq \phi(\langle \mathcal{A}_t^i, k_j\rangle)$$
- Unlike self-attention, keys/values are **stored in params**, not derived from the sequence → FFN = persistent **content-addressable memory**, capacity $\propto m$.

**Critical observation**: increasing $m$ = adding memory slots; editing $(k_j,v_j)$ = writing new entries. FFNs dominate param count in big transformers → they are the primary locus of memorization capacity. **Locas exploits this to write new K-V pairs into FFNs at test time with principled init.**

---

## 3. Architecture: Sideway FFN Memory (Fig 2)

- Locas memory is a **parallel ("sideway") FFN module** sitting alongside the backbone FFN inside each transformer layer.
- Both consume the same attention-block output; their outputs are **added** on the main pathway, with the Locas output scaled by factor $\tau$:
  $$\mathcal{H}_{\text{out}} = \text{FFN}(\mathcal{A}) + \tau \cdot \text{Locas}(\mathcal{A})$$
- **Why sideway (vs LoRA-style weight modification)**: genuine **model capacity expansion** at test time while leaving backbone params *entirely untouched* → strictly additive contribution → backbone behavior is a recoverable "safe baseline" (zero out the memory to revert). This is the structural source of forgetting resistance.

---

## 4. Two Variants

### 4.1 Locas-MLP (conventional 2-layer ReLU MLP — cleaner theory)

$$\text{Locas-MLP}(\mathcal{A}) = V^\top \cdot \text{ReLU}(K^\top \mathcal{A})$$

- $K \in \mathbb{R}^{d\times r}$, $V \in \mathbb{R}^{r\times d}$, $r$ = latent dim of the memory.
- **Pros**: both K and V admit **step-wise optimal closed-form** init.
- **Cons**: incompatible with models *not* trained with 2-layer MLP FFNs — piecewise-linear separability at MLP input is poor for SOTA (GLU) models.

### 4.2 Locas-GLU (matches SOTA LLM FFNs — practical)

Shares the GLU-FFN structure of LLaMA/Qwen/Mistral:

$$\text{Locas-GLU}(\mathcal{A}) = V^\top \cdot \big(\sigma(G^\top \mathcal{A}) \odot K^\top \mathcal{A}\big)$$

- $G \in \mathbb{R}^{d\times r}$ = gate, $K \in \mathbb{R}^{d\times r}$ = up-proj (key), $V \in \mathbb{R}^{r\times d}$ = down-proj (value), $\sigma$ = SiLU.
- **Per-layer params**: $3 \times L \times d \times r$ (three $d\times r$/$r\times d$ matrices over $L$ layers).
- Seamlessly attaches to existing GLU models → param- and compute-efficient continual learning.

---

## 5. Principled Initialization (the core contribution)

### 5.1 Locas-MLP: Activation + Gradient Reuse → step-wise optimal init

- Memorized token $x_t$ with context $x_{<t}$. Layer $i$ hidden output $\mathcal{H}_i$; concatenated over layers = $\mathcal{H}$.
- Next-token log-likelihood as function of param $\theta$ and hidden $\mathcal{H}$:
  $$\log p(x_t|x_{<t}) = F(\theta; \mathcal{H}; x_t)$$
- Key trick: reuse backprop **without** doing param updates. Define gradient *w.r.t. hidden output*:
  $$\mathcal{G} = \nabla_{\mathcal{H}} \log p(x_t|x_{<t})$$
- By def of gradient, $\exists \eta>0$: moving $\mathcal{H}^+ = \mathcal{H} + \eta\cdot\mathcal{G}$ increases likelihood: $F(\theta;\mathcal{H}^+;x_t) > F(\theta;\mathcal{H};x_t)$.

**Step-wise optimal init** for the new slot $(k_m^i, v_m^i)$ at layer $i$:
$$k_m^i \leftarrow \text{Normalize}(\mathcal{A}_t^i)$$
$$v_m^i \leftarrow \epsilon \cdot \text{GlobalNormalize}(\mathcal{G}_t^i)$$

- **Key = the (normalized) current input activation** → guarantees the new slot fires *exactly* on the token being memorized.
- **Value = (normalized) gradient of likelihood w.r.t. hidden** → pushes hidden state in the likelihood-increasing direction.
- `GlobalNormalize` = normalize $\mathcal{G}$ across **both** hidden-size and layer-number axes.
- Resulting hidden delta is collinear with the gradient:
  $$\Delta\mathcal{H}_t^i = \phi(\langle\mathcal{A}_t^i, k_m^i\rangle)\cdot v_m^i = C\cdot \text{GlobalNormalize}(\mathcal{G}_t^i) \propto \mathcal{G}_t^i$$
  ($C$ = constant controllable via normalization/rescaling.)
- **Claim**: step-wise optimal in both time-step and gradient-update-step under stated assumptions.

### 5.2 Locas-GLU: Activation-Guided Parameter Cloning

GLU has a gating product → can't directly inject gradients as values. Instead, **clone basis vectors from the backbone FFN**, selected by *activation patterns*.

**(1) Activation-based basis selection** — forward pass over chunk $x_{<T}$, compute backbone GLU intermediate at layer $i$:
$$\mathcal{M}_t^i = \sigma(W_G^i \mathcal{A}_t^i) \odot (W_K^i \mathcal{A}_t^i)$$
($W_G^i, W_K^i \in \mathbb{R}^{m\times d}$ = backbone gate & up-proj.)

**(2) Activation importance** per intermediate dim $j$ = mean abs activation over chunk tokens:
$$\alpha_j^i = \frac{1}{T}\sum_{t=1}^T |\mathcal{M}_{t,j}^i|$$

**(3) Top-$K$ selection = nonlinear PCA in activation space** — sort dims by $\alpha_j^i$ desc, take top-$r$ indices $\mathcal{S}_r^i$.

**(4) Clone key & gate matrices** (rows indexed by $\mathcal{S}_r^i$):
$$K^i \leftarrow \text{Normalize}([W_K^i]_{j\in\mathcal{S}_r^i}), \qquad G^i \leftarrow \text{Normalize}([W_G^i]_{j\in\mathcal{S}_r^i})$$

**(5) Zero-init value matrix** (LoRA-style, for behavioral consistency at init):
$$V^i \leftarrow \mathbf{0}$$
→ memory contributes **zero** at init; learns context-specific content via grad updates afterward.

**Interpretation**: Top-$K$ most-activated bases ≈ top principal components of the backbone's pretrained feature-decomposition space → approximates the **support manifold** of the current context. Balances *local support* (relevant to this context) vs *generalizable features* (well-learned from pretraining). Since it's a *sideway* FFN (not a weight modification like LoRA), it's genuine capacity expansion.

### 5.3 Two safeguards against forgetting / runaway influence

- **Weight Norm Clipping (implicit KL constraint)**: for each row/col vector $w$ in K, G, V matrices during inference:
  $$w \leftarrow \frac{w}{\max(\|w\|_2, 1)}$$
  - Like Weight Normalization (Salimans & Kingma) but **asymmetric**: only clips vectors with norm $>1$, leaves smaller ones alone.
  - Bounds per-step contribution within a fixed-radius ball → implicit KL constraint on behavior shift, cheaper than explicit KL reg.
- **Output scaling factor $\tau$**: memory output not added with equal weight. Set adaptively to avg row norm of backbone down-proj divided by memory width $r$:
  $$\tau = \frac{1}{r}\cdot\frac{1}{m}\sum_{j=1}^m \|W_{\text{down}}[j,:]\|_2$$
  - Calibrates memory output magnitude to backbone FFN's typical scale; division by $r$ normalizes the aggregated contribution from $r$ slots. Double safeguard w/ weight clipping.

### 5.4 Why init matters
- Random init needs many more grad steps to match performance.
- Activation-guided init = strong starting point already aligned with backbone internals → fewer dimensions needed (param efficiency) + fewer grad updates (compute efficiency) + better generalization + forgetting prevention.

---

## 6. Memory Accumulation over Context

- In practice: init principledly, then **update via standard backprop** on the LM objective as model streams context. Simple, efficient, mixed-precision-compatible.
- **Locas-MLP**: layer-wise memory grows **linearly** with # memorized tokens (one new slot per token per layer).
- To bound latent size, they propose **NL-SVD compression** (Appendix A.1, below) — but empirically BP ≥ NL-SVD on final perf at much lower cost, so **BP is recommended in practice**.

---

## 7. Non-Linear SVD (NL-SVD) — Locas-MLP compression (Appendix A.1)

Generalizes classical SVD (Eckart-Young, works only on linear projections) to the 2-layer **non-linear** ReLU FFN case.

**Theory/intuition**:
- A finite 2-layer ReLU MLP = a sparsely-activated MLP with *infinite* intermediate dim where the "virtual" extra dims (those with nonlinear keys) must have **zero value vectors** → motivates rethinking *effective* latent dim.
- Can rescale a dim's key by $s$ and its value by $1/s$ w/o changing function (rigorous for ReLU since linear/piecewise-linear against breakpoint $x=0$; approximately holds for GeLU).
- Activation pattern (*whether* + *how strongly* a dim fires) dominated by key matrix and the **product of vector norms** from K and V.

**Algorithm (Alg 1)** — `Require`: $K\in\mathbb{R}^{d\times m}$, $V\in\mathbb{R}^{m\times d}$, target rank $n$, threshold $\epsilon$:
1. Normalize each col of $K$, extract row norms $\{\alpha_i\}$.
2. Normalize each col of $V$, extract col norms $\{\beta_i\}$.
3. Composed scalars $s_i = \alpha_i\beta_i$ (captures direction + effective magnitude).
4. Weighted key matrix $\bar K_i = s_i K_i$.
5. EVD on $\bar K \bar K^\top = U\Sigma U^\top$.
6. Take top-$n$ left singular vectors $U_n$ (minimizes rank-$n$ reconstruction error of $\bar K$ → preserves dominant activation subspace).
7. Reduced key matrix $\tilde K$ from rows of $U_n$ (row-normalized → $n$ orthogonal unit **probe vectors** $\{p_j\}$).
8. Drop rows with norm $<\epsilon$.
9–12. **Value reconstruction**: feed each probe $p_j$ through the *original* FFN, record output contribution $v_j = f(p_j)\in\mathbb{R}^d$ → use as $j$-th col of reduced $\tilde V$.

**Functional equivalence**: for any input on the span of $\{p_j\}$, reduced FFN output = original (linearity of 2nd layer + values queried from original net). Off-subspace approx error governed by discarded singular values of $\bar K$.

**Expansion–Compression cycle (A.1.6)**: expansion adds 1 slot/token/layer (monotonically grows $m$); after $N_{capacity}$ tokens, NL-SVD compresses to fixed rank $n$ (typically $N_{capacity}/2$). Can compress at flexible granularity — every token (fixed-size refreshed memory) or over longer spans.

**Practical limitations (A.1.7)** — why BP is preferred:
- Needs ≥ **float32** for SVD stability → breaks bfloat16 mixed-precision pipelines.
- SVD not well-optimized on modern GPUs → 12.7× relative time vs 1.7× for BP.
- Theory **doesn't transfer cleanly to Locas-GLU** (gating mechanism) → limited applicability.
- No final-perf advantage (17.89 vs 17.90 PPL @200K).

---

## 8. Experiments

### 8.1 Setup
- **Datasets**: PG-19 (book-level LM, long narratives exceeding pretraining context) for online whole-book LM; **LoCoMo** (Long-Context Conversational Memory, Maharana et al.) — synthetic multi-turn dialogues tens-of-thousands of tokens, single-hop / multi-hop / open-domain / temporal / adversarial QA.
- **Models**: Qwen3-0.6B-Base, Qwen3-1.7B-Base, Qwen3-1.7B(-Instruct), Qwen3-4B-Base (GLU → Locas-GLU). For Locas-MLP they **pretrain their own 0.6B decoder** with 2-layer MLP FFNs on ~150B SlimPajama tokens (since SOTA uses GLU, incompatible w/ MLP variant).
- **Baselines**:
  - **Context Truncation** (2K / 4K): discard past context beyond window. Lower bound.
  - **Long Context Attention** (16K): native attention, pure ICL, no TTT.
  - **TempLoRA** (Wang et al.): SOTA TTT — LoRA on *all* linear projections (attn Q,K,V,O + FFN gate/up/down), updated via grad descent at inference. Big param + compute overhead.
  - All TTT baselines truncate context to 2K.

### 8.2 PG-19 Whole-Book Online LM (Table 1) — PPL @ context len 50K/100K/150K/200K

Key numbers (lower PPL better):

| Model | Method | PPL@200K | Extra #Params | Rel. Time |
|---|---|---|---|---|
| 0.6B MLP-FFN (theirs) | Ctx Trunc 2K | 18.72 | 0 | 1× |
| | TempLoRA | 17.93 | 41.9M | 4.7× |
| | **Locas-MLP + NL-SVD** | **17.89** | 4.2M | 12.7× |
| | **Locas-MLP + BP** | 17.90 | 4.2M | 1.7× |
| Qwen3-0.6B | Long Ctx Attn 16K | 24.57 | 0 | 13.9× |
| | TempLoRA | 25.22 | 36.7M | 4.3× |
| | **Locas-GLU** | 25.00 | **5.5M** | 1.4× |
| Qwen3-1.7B-Base | TempLoRA | 19.13 | 73.4M | 4.7× |
| | **Locas-GLU** | **19.04** | **11.0M** | 1.8× |
| Qwen3-1.7B-Instruct | TempLoRA | 21.65 | 73.4M | 4.7× |
| | **Locas-GLU** | 21.65 | **11.0M** | 1.8× |

- **Headline**: Locas-GLU matches/beats TempLoRA using **~15-25% of the extra params** and **~38% of the compute** (1.8× vs 4.7× rel. time).
- vs ctx truncation → substantial gains (successful accumulation/reuse of domain info).
- vs full attention → comparable/lower PPL at far lower compute, esp. on long docs where attention is expensive.
- **NL-SVD vs BP**: NL-SVD has no perf edge (17.89 vs 17.90) and costs much more (12.7× vs 1.7×) → use BP.

### 8.3 Ablation — Initialization Strategy (Table 2, Qwen3-1.7B + Locas-GLU, K=64, LR=4e-3)

| Init | PPL@50K | @100K | @200K |
|---|---|---|---|
| No Memory | 20.60 | 20.38 | 20.50 |
| Gaussian Random | 20.44 | 20.09 | 19.90 |
| Norm. Activation (MLP-style, only converges @LR=1e-6) | 20.58 | 20.17 | 19.93 |
| Random Selection Cloning | 20.02 | 19.51 | 19.06 |
| Bottom-$K$ (least activated) | 20.04 | 19.53 | 19.08 |
| **Top-$K$ (most activated)** | **20.00** | **19.49** | **19.04** |

- Top-$K$ consistently best. Improvement over random is substantial → activation patterns are a strong inductive bias.
- Interesting: **random selection cloning** also does well → pretrained representations carry useful info regardless of which dims chosen; Top-$K$ gives consistent edge by focusing on task-relevant dims.

### 8.4 Ablation — Memory Width (Table 3, Qwen3-1.7B-Base)

Param comparison: Locas-GLU @ rank $r$ = $3Ldr$ params/layer; LoRA (all projections) ≈ $8Ldr + 12Ldr = 20Ldr$ (since $m\approx 3d$ in GLU) → **LoRA ~6.7× more params at same rank**.

| Method | $r$ | #Params | PPL@200K |
|---|---|---|---|
| Ctx Trunc 2K | - | 0 | 20.50 |
| TempLoRA | 16 | 18.4M | 19.39 |
| TempLoRA | 64 | 73.4M | 19.13 |
| TempLoRA | 128 | 146.9M | 19.10 |
| **Locas-GLU** | 16 | **2.8M** | 19.14 |
| **Locas-GLU** | 32 | 5.5M | 19.10 |
| **Locas-GLU** | 64 | 11.0M | 19.04 |
| **Locas-GLU** | 128 | 22.0M | **19.02** |

- **Strong perf at very low rank**: Locas-GLU @ $r=16$ (2.8M) ≈ TempLoRA @ $r=64$ (73.4M) → **26× param reduction**.
- **Saturation earlier than TempLoRA**: Locas $r=64\to128$ only 0.02 PPL gain; TempLoRA keeps benefiting → consistent with PCA interpretation (first few selected bases capture most task-relevant info).
- At equal $r$, Locas beats TempLoRA despite 6.7× fewer params (e.g. $r=32$: Locas 19.10 @5.5M vs TempLoRA 19.24 @36.7M). → Concentrating adaptation in the **FFN pathway w/ principled init** > distributing low-rank updates across all projections.

### 8.5 Ablation — General Capability / Catastrophic Forgetting (Table 4, MMLU after memorizing a full PG-19 book, Qwen3-1.7B-Base)

| Method | MMLU % | Δ from baseline | Extra Params | Rel. Time |
|---|---|---|---|---|
| No Memorization (baseline) | 60.4 | 0 | 0 | 1× |
| TempLoRA | 59.8 | **-0.6** | 73.4M | 4.7× |
| TempLoRA (r=512) | 59.2 | **-1.2** | 587M | 11.2× |
| **Locas-GLU** | 60.2 | **-0.2** | 11.0M | 1.8× |
| **Locas-GLU (r=512)** | 60.3 | **-0.1** | 88.1M | 3.7× |

- Locas-GLU: only **0.1–0.2% MMLU degradation**; TempLoRA: 0.6–1.2%.
- **Positive correlation between param count and forgetting for TempLoRA** (0.6%→1.2% as $r:64\to512$); **nearly absent for Locas** (0.2%→0.1%). → TempLoRA modifies existing weights → more memorization conflicts more w/ pretrained knowledge; Locas's parallel additive pathway accommodates extra capacity w/o interference.

### 8.6 LoCoMo Dialogue QA (Table 5) — F1 % across 5 categories (adversarial = neg F1 of trap answer)

Qwen3-1.7B-Base (Full Attn context):

| Method | Single-Hop | Multi-Hop | Open-Dom | Temporal | Adv |
|---|---|---|---|---|---|
| Full Attention | 37.3 | 23.8 | 14.0 | 33.5 | -33.0 |
| Full Attn + TempLoRA | 37.7 | 23.1 | 13.3 | 29.1 | -31.8 |
| **Full Attn + Locas-GLU** | **41.6** | **25.2** | 14.1 | **34.1** | **-28.7** |

Qwen3-4B-Base, Locas-GLU: single-hop **47.6** (+15.5% over full attn 41.2), multi-hop **28.1**.

- **Outperforms both baselines on most question types**: single-hop +11.5% rel over full attn / +10.3% over TempLoRA; similar multi-hop gains → memorizes facts AND supports compositional reasoning.
- **Temporal reasoning benefits most** (34.1 vs 29.1 TempLoRA on 1.7B; 18.1 vs 17.2 on 4B) → sideway architecture preserves sequential event structure better.
- **Adversarial robustness improves** (4B: -19.8 vs -25.4 full attn / -24.3 TempLoRA) → parametric memory anchors to factual info over misleading context.
- **No-context eval (No Cxt rows) reveals retention**: w/o any dialogue context Locas beats TempLoRA (multi-hop 8.4 vs 4.5 on 1.7B; temporal 4.3 vs 1.3) → Locas internalizes facts into *persistent* parametric memory, recallable even w/o original context.

---

## 9. Related Work (positioning)

- **TTT/adaptation**: Sun et al. (self-sup auxiliary tasks; TTT layers replacing RNN hidden state w/ learnable model). **TempLoRA** (temporary LoRA progressively trained on generated chunks). VDS-TTT (verifier-selected pseudo-labels). TLM (perplexity-minimization adaptation w/ LoRA). SPINE (token-selective TTT RL). → Locas differs by (1) **principled init** not random+grad, (2) **sideway FFN = capacity expansion** not weight modification → stronger forgetting guarantees.
- **PEFT**: LoRA, QLoRA, DoRA, GraLoRA, DiffoRA, CTR-LoRA. Locas shares the param-efficiency goal but uses a **parallel memory pathway** preserving backbone reps; activation-guided init = a "warm start" doing nonlinear PCA in activation space.
- **FFN-as-KV-memory / knowledge editing**: Geva et al. (FFN neurons = pattern-detector keys → value contributions — the foundation). **WISE** (dual memory: main pretrained + side edited, w/ router) — *strikingly similar to Locas-GLU's sideway design*, but WISE targets discrete edits while Locas does **online streaming-context memorization** w/ principled slot selection. LoKI (low-damage knowledge implanting).
- **Long-context / SSM**: Longformer, sparse attention, linear attention, ALiBi/RoPE extrapolation, Mamba. Locas is **complementary** (parametric memory alongside existing attn/recurrence) → composable.
- **Memory-augmented NNs**: NTM, Memory Networks, DNC (external memory via attention). LLM-era: RAG (external doc store), **Mem0 / Mem0g** (scalable extract-consolidate-retrieve memory, graph variant for multi-hop; 26% rel improvement on LOCOMO, -91% latency, -90% tokens). Locas = parametric memory aug **within the model architecture** (integrates into forward pass, no retrieval latency/integration challenges).

---

## 10. Limitations
- **Locas-MLP** incompatible w/ GLU-based SOTA LLMs (poor piecewise-linear separability at MLP input) → required pretraining a custom 0.6B MLP-FFN model.
- **NL-SVD**: needs float32 (breaks bf16 mixed-precision), SVD slow on GPUs (12.7× time), theory doesn't transfer to GLU, no perf advantage over BP → essentially deprecated in favor of BP.
- Memory grows linearly w/ tokens (MLP variant); needs compression/bounded latent for very long contexts.
- Evaluated on Qwen3 family + custom model only; LoCoMo is synthetic.

---

## 11. Why It Works — Mental Model
- FFN = parametric content-addressable memory; adding slots = adding memory. Locas writes new slots into a **parallel** FFN.
- **Best key** = the activation of the token to memorize (fires exactly there); **best value** = likelihood gradient direction (MLP) or a backbone basis cloned from the top-activated (principal) directions (GLU).
- Sideway + zero-value-init + clipping + $\tau$-scaling → starts as a no-op, only ever adds, and stays bounded → backbone is a recoverable safe baseline → minimal forgetting.
- Principled init = init already in the right subspace → few slots + few grad steps needed → param & compute efficiency.
- Conceptual neighbors: **fast weights** (transient task-specific memory), **WISE side memory**, **LoRA warm-start**, **nonlinear PCA** of activations.
