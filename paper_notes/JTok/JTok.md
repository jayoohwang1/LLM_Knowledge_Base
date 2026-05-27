# JTok: On Token Embedding as another Axis of Scaling Law via Joint Token Self-modulation

**Yang et al., SJTU + Xiaohongshu Hi Lab, arXiv 2602.00800 (Jan 2026)**

---

## TL;DR

- **Core claim**: introduce **token-indexed parameters** as a *third* scaling axis (orthogonal to dense params + MoE expert sparsity), decoupling **model capacity** from **FLOPs**.
- **Mechanism**: per-layer learnable embedding table $\mathbf{E}^\ell \in \mathbb{R}^{V\times d}$ retrieves a token-specific **modulation vector** $\mathbf{p}_x^\ell$ that **multiplicatively gates the FFN residual** via Hadamard product. No matmuls in the modulation path — pure lookups + element-wise ops.
- **Two variants**:
	- **JTok**: static per-token gate on MLP increment.
	- **JTok-M**: top-$K$ mixture of modulators with router conditioned on hidden state (MoE-style but for embeddings, not experts).
- **Scaling law**: token-indexed params $N_n$ enter via *effective parameter count* $N_\text{eff} = N_c(1 + \eta\gamma(\rho))$ which **rescales the constant in the model-size term** of the Kaplan loss, producing a **scale-invariant multiplicative shift** of the compute-optimal frontier — same loss with **35% less compute**.
- **Results**: +4.1 MMLU, +8.3 ARC, +8.9 CEval on a **17B-A2B MoE** with 44B JTok-M embedding params; training overhead $\le 6.78\%$, inference overhead $\le 7.3\%$.

---

## Motivation: A Third Scaling Axis

- **Existing axes**:
	- **Dense scaling** (Kaplan/Chinchilla): grow $N_c$, FLOPs grow $\sim 6N_cD$ — linearly coupled, diminishing returns + data wall.
	- **MoE sparsity**: decouples *capacity* from *active* FLOPs, but log-linear gains saturate fast, sample inefficiency, routing/latency/HBM headaches.
	- **Vocabulary scaling** (SuperBPE, BLT, Over-tokenized Transformer): expand $V$, but constrained by combinatorial limits of surface forms — captures *local* patterns, no deep semantics.
- **JTok's pitch**: scale along the **feature dimension $d$** through a high-dim, context-interactive space where tokens acquire richer semantics via attention-mediated interactions during training.
	- Different from vocab scaling: not adding new tokens, but giving each existing token *layer-wise, dimension-wise* modulation vectors that are themselves trained.
	- Different from hypernet conditioning: no matmul generating params — direct embedding table lookup.
	- Different from memory layers (PEEK, Memory Layers at Scale, Product Key Memory): those *replace* an FFN with a sparse KV memory; JTok is a **plug-in bypass** that *gates* the existing FFN residual.

> **Analogy**: think of MoE as growing "which subnetwork to use" sparsely along the **expert axis**; JTok grows "what multiplicative bias to apply" along the **token axis**. Tokens get private, learnable, per-layer scaling vectors — like a per-token AdaLN/FiLM gate but with the conditioning signal coming from a *learned table* not a hypernet.

---

## Method

### Backbone: Pre-Norm Transformer

- Standard pre-norm block, per token $x$ at layer $\ell$:

$$\Delta\mathbf{a}_x^\ell = \text{Attn}^\ell(\text{RMSNorm}(\mathbf{h}_x^\ell))$$
$$\tilde{\mathbf{h}}_x^\ell = \mathbf{h}_x^\ell + \Delta\mathbf{a}_x^\ell$$
$$\Delta\mathbf{m}_x^\ell = \text{FFN}^\ell(\text{RMSNorm}(\tilde{\mathbf{h}}_x^\ell))$$
$$\mathbf{h}_x^{\ell+1} = \tilde{\mathbf{h}}_x^\ell + \Delta\mathbf{m}_x^\ell$$

- FFN is dense MLP or sparsely-activated experts for MoE.

### JTok (Static)

- Each layer $\ell$ owns a learnable table $\mathbf{E}^\ell \in \mathbb{R}^{V\times d}$.
- For token id $x$, retrieve $\mathbf{E}^\ell[x]$, normalize to hypersphere, scale per-dim, form a **multiplicative gate**:

$$\mathbf{p}_x^\ell = \mathbf{1} + \mathbf{s}^\ell \odot \text{Norm}_\varepsilon(\mathbf{E}^\ell[x]) \qquad (7)$$

- $\mathbf{s}^\ell \in \mathbb{R}^d$ is a **learnable per-dim scaler**.
- $\text{Norm}_\varepsilon(\mathbf{u}) = \mathbf{u}/(\|\mathbf{u}\|_2 + \varepsilon)$ — L2 norm onto unit hypersphere.
- The **"+1"** keeps the gate centered at identity (residual-friendly init).
- Gate is applied to the MLP increment, not the hidden state directly:

$$\Delta\hat{\mathbf{m}}_x^\ell = \Delta\mathbf{m}_x^\ell \odot \mathbf{p}_x^\ell \qquad (8)$$
$$\mathbf{h}_x^{\ell+1} = \tilde{\mathbf{h}}_x^\ell + \Delta\hat{\mathbf{m}}_x^\ell \qquad (9)$$

- **Why gate the residual instead of the hidden state?**
	- Modulating only the *update* keeps the skip path clean — gradient flows unchanged through residual stream.
	- Acts as a token-specific learnable mask on what each FFN write-back contributes.

### JTok-M (Mixture of JTok)

- Generalizes from static lookup → **dynamic, context-aware sparse routing**.
- Each layer keeps a **pool of $n_e$ modulators**: $\{\mathbf{E}_i^\ell\}_{i=1}^{n_e}$, each in $\mathbb{R}^{V\times d}$.
- **Router** is a linear layer $\mathbf{R}^\ell \in \mathbb{R}^{d \times n_e}$ over the pre-attention hidden state:

$$\mathbf{g}_x^\ell = (\text{RMSNorm}(\mathbf{h}_x^\ell))^\top \mathbf{R}^\ell \in \mathbb{R}^{n_e} \qquad (10)$$

- TopK selection $G_x^\ell = \text{TopK}(\mathbf{g}_x^\ell, K)$; **sigmoid** (not softmax) normalization over the selected logits — paper cites a sigmoid-gating MoE paper showing better sample efficiency.

$$w_i^\ell = \sigma(g_i^\ell) / \sum_{j \in G_x^\ell}\sigma(g_j^\ell)$$

- **Mixed token-indexed vector**:

$$\mathbf{e}_x^\ell = \sum_{i \in G_x^\ell} w_i^\ell \mathbf{E}_i^\ell[x] \in \mathbb{R}^d \qquad (11)$$

- **Residual injection** (additive, not multiplicative — note the asymmetry vs. JTok):

$$\Delta\mathbf{r}_x^\ell = \frac{1}{\sqrt{2N_l}} \cdot \mathbf{s}_{\text{JTok-M}}^\ell \odot \text{Norm}_\varepsilon(\mathbf{e}_x^\ell) \qquad (12)$$

- Fused with MLP output into write-back:

$$\mathbf{h}_x^{\ell+1} = \tilde{\mathbf{h}}_x^\ell + \Delta\mathbf{m}_x^\ell + \Delta\mathbf{r}_x^\ell \qquad (13)$$

- **The $1/\sqrt{2N_l}$ factor** ($N_l$ = #backbone layers) keeps hidden-state variance bounded — crucial; without it std blows up to 5.5+ in first 50B tokens (ablation Fig 7).
- Auxiliary **load-balancing loss** (Appendix B) on per-modulator routing probabilities $p_i$ × actual load fractions $f_i$, à la Switch Transformer:

$$\mathcal{L}_\text{aux} = \lambda \cdot n_e \cdot \sum_i p_i f_i = \lambda \cdot \frac{n_e}{T^2 K}\sum_i P_i n_i$$

- $\lambda = 10^{-4}$ in practice.

> **Key design contrast with vanilla MoE**: experts in MoE are *whole FFNs* selected per token to do compute; JTok-M "experts" are *embedding tables* that emit a vector — selection is over **lookup sources**, not over **computational subnetworks**. Compute stays ~constant.

---

## Computational Complexity

| Module | Per-token-per-layer cost |
|---|---|
| JTok | $\mathcal{O}(d)$ — norm + Hadamard + add |
| JTok-M router | $\mathcal{O}(d n_e)$ |
| JTok-M mixture | $\mathcal{O}(Kd)$ |
| Backbone attn/FFN | $\Theta(d^2)$ |

- $n_e, K$ are small constants (paper uses $n_e = 5$, $K = 2$) → JTok-M overhead **linear in $d$**, dwarfed by $d^2$ backbone.
- Roofline-bounded: element-wise ops are **memory-bandwidth-bound**, not compute-bound. Kernel fusion helps.

---

## System Efficiency Tricks

- **Bottleneck shifts**: from GEMM compute → memory access + HBM footprint, since lookup tables are huge ($V \times d \times L \times n_e$, e.g. 44B params for the 17B MoE setting).

### 1. Token Deduplication (Zipfian exploitation)
- Token freqs follow Zipf → high-freq tokens repeatedly accessed within a micro-batch.
- Gather unique token ids once → scatter back to all occurrences. Avoids redundant DRAM reads.

### 2. Asynchronous Overlap
- JTok modules are **bypass branches** decoupled from backbone compute graph.
- Issue embedding gathers for layer $\ell$ asynchronously while backbone runs layer $\ell$'s attn/FFN.
- Retrieved vectors fused only at write-back (Eq 9/13) — most memory latency hides under compute.

### 3. Embedding Model Parallelism in Training
- Shard $\mathbf{E}_i^\ell$ tables across GPUs.
- Naive impl would communicate $Kd$ elements per token; **owner-rank premixing** — the device owning the selected modulator does the weighted sum locally → only $d$-sized mixed vector communicated.

### 4. CPU Offloading at Inference
- Tables sparse-accessed → host→device transfer scales with $\mathcal{O}(d)$ or $\mathcal{O}(Kd)$ per token, **independent of $V$ and total table capacity**.
- Tables live in CPU RAM; only the small set needed by current batch streamed to GPU + overlapped with backbone.
- → **zero extra HBM footprint** at inference.

### Measured overheads (3.2B-A0.5B MoE, $\eta=50$, $\rho=0.25$)

| Throughput | Baseline | JTok-M naive | + EmbP | + EmbP + Dedup |
|---|---|---|---|---|
| Train (tok/s) | 4838K | 2749K | 4024K | **4510K (-6.78%)** |

| Inference (8×H800, SGLang, bs=16, seq=4K) | Prefill | Decode |
|---|---|---|
| Baseline | 363.7K | 4494 |
| JTok (CPU-offload) | 360.8K (-0.8%) | 4290 (-4.5%) |
| JTok-M (CPU-offload) | 355.2K (-2.3%) | 4166 (**-7.3%**) |

---

## Scaling Law Analysis

### Core Construction: Effective Parameters $N_\text{eff}$

- Let $N_c$ = backbone compute-intensive params (active), $N_n$ = token-indexed params.
- **Expansion ratio**: $\eta \triangleq N_n / N_c$.
- **Routing sparsity discount**: $\gamma(\rho)$ with $\rho = K/n_e$ (JTok-M activation sparsity).

$$N_\text{eff} \triangleq N_c + \gamma(\rho) N_n = N_c(1 + \eta\gamma(\rho)) \qquad (14)$$

- Plug into Kaplan loss (Eq 5: $\mathcal{L}(N_c, D) = [(A/N_c)^{\alpha/\beta} + B/D]^\beta$):

$$\mathcal{L}_\text{JTok-M}(N_c, D; \eta, \rho) = \left[\left(\frac{A/(1+\eta\gamma(\rho))}{N_c}\right)^{\alpha/\beta} + \frac{B}{D}\right]^\beta \qquad (15)$$

- → JTok-M is **equivalent to rescaling the constant $A$** in the model-size term: $A \mapsto A_\text{JTok-M} = A/(1+\eta\gamma(\rho))$.
- Data term $B/D$ and dominant FLOPs $6N_cD$ **unchanged**.

### Compute-Optimal Frontier Shift

- Under compute-optimal training (minimize over $N_c$ at fixed $C$):

$$\mathcal{L}^*(C) \propto A^{\alpha/(\alpha+\beta)} C^{-\alpha\beta/(\alpha+\beta)}$$

- Only the intercept ($A$-dependent) changes:

$$\mathcal{L}^*_\text{JTok-M}(C; \eta, \rho) = (1 + \eta\gamma(\rho))^{-\alpha\beta/(\alpha+\beta)} \cdot \mathcal{L}^*(C) \qquad (16)$$

### Isoperformance Compute Saving (Scale-Invariant)

- For any target loss $\mathcal{L}^\star$:

$$C^*_\text{JTok-M}(\mathcal{L}^\star) = \frac{1}{1 + \eta\gamma(\rho)} C^*_\text{base}(\mathcal{L}^\star) \qquad (17)$$

- **Key insight**: compute-saving ratio **independent of backbone scale** $C$ — depends only on JTok-M hyperparams $\eta, \rho, \gamma(\cdot)$.

### Empirical Validation

- **IsoFLOPs sweep**: 5 budgets $C \in \{3e18, 1e19, 3e19, 1e20, 3e20\}$ × grid of $N_c$ values for MoE backbone (top-8/145 experts, 1 shared) — U-shaped curves, quadratic fits give compute-optimal points.
- **Linear fits** in $\log_2 \mathcal{L}$ vs $\log_{10} C$:
	- Vanilla MoE: $\log_2 \mathcal{L} = -0.2016 \log_{10}(C) + 5.1334$, $R^2 = 0.9994$
	- JTok-M: $\log_2 \mathcal{L} = -0.2009 \log_{10}(C) + 5.0881$, $R^2 = 0.9991$
- **Same slope, lower intercept** → multiplicative shift of frontier as predicted by Eq 16.
- Compute saving ratio: $10^{\Delta\beta/\alpha} = 10^{0.038/-0.2016} \approx 0.648$ → **35.2% compute saving**.
- JTok-M achieves **2.2% loss reduction** at iso-compute across budgets.

### Scaling JTok-M Itself (Standalone Capacity Axis)

- Fix backbone (3.9B-A0.6B MoE), 100B tokens, sweep $\eta \in \{30, \ldots, 130\}$ at fixed $\rho = 0.25$.
- **Monotonic loss reduction**, log-linear in $\eta$: $R^2 = 0.9959$, doubling JTok-M params → **0.0118 loss drop**.
- Confirms JTok-M params form a **predictable power-law scaling axis** with no observed saturation.

### Stability across Backbone Sizes

- Fixed $(\eta, \rho) = (50, 0.25)$, three MoE backbones: 2.3B-A0.5B, 3.9B-A0.6B, 9.8B-A1.4B, 100B tokens.
- Final-loss reductions: **0.059, 0.064, 0.051** → relative gains **2.7%, 3.0%, 2.55%** — flat across scales, matching scale-invariant prediction.

---

## Experiments

### Setup

| Family | Backbone | Total params | Tokens | Dataset |
|---|---|---|---|---|
| Dense | S/M/L (190M/0.5B/1B) | + JTok | 100B | Fineweb-edu |
| Dense | XL (1.5B) | +JTok 6.5B | 300B | curated multi-source |
| MoE | 1.5B-A250M | +JTok 0.93B, +JTok-M 4.7B | 300B | curated |
| MoE | 3.2B-A0.5B | +JTok 2.1B, +JTok-M 10.5B | 500B | curated |
| MoE | 17B-A2B | +JTok-M **43.6B** | 570B | curated |

- Trained with **Megatron-LM**, inference with **SGLang**.
- 17B-A2B: 28 layers, 65 experts/layer (1 shared + 64 routed), top-6, 1.9B active. JTok-M adds 43.6B embedding params → **61B total**, 17B+44B split.
- JTok-M: $n_e = 5$, $K = 2$ → $\rho = 0.4$ (Sec 4.2 uses $\rho = 0.25$ with different config).

### Downstream (Table 1, OpenCompass)

| Backbone | Overall Avg (B.O. → +JTok / +JTok-M) |
|---|---|
| 1.5B Dense | 22.22 → 26.54 (+4.32) |
| 1.5B-A250M MoE | 18.87 → 20.51 (+JTok) → **22.78 (+3.91 JTok-M)** |
| 3.2B-A0.5B MoE | 26.75 → 29.24 (+JTok) → **32.34 (+5.59 JTok-M)** |

- **17B-A2B + JTok-M (570B tokens)**: +4.10 MMLU, +8.27 ARC, +3.93 HellaSwag, +9.34 CMMLU, +8.31 CEval, +3.22 Xiezhi.
- Gains amplified on **knowledge-intensive & reasoning** benchmarks (MMLU, CEval, CMMLU, ARC).
- Math/Code also strong: GSM8K +6.31, ARC-C +7.25 on 3.2B MoE; MATH +3.26 on 3.2B.

### Training Dynamics

- Accuracy trajectories on 17B-A2B show JTok-M **leads from the start** (early-stage gap) and the gap **widens** through 570B tokens with no saturation.
- Unlike tail-end convergence tricks, JTok-M provides **consistent capacity expansion** — better data efficiency from token 1.

---

## Ablations

### Norm in JTok / JTok-M (Sec D.2)

- $\text{Norm}_\varepsilon$ (L2 → hypersphere) on the retrieved vector matters.
- **Two benefits**:
	1. **Decouples direction from magnitude** — gate is well-conditioned. Without norm, training has to jointly tune direction + scale of $\mathbf{E}^\ell[x]$ to reach effective modulation regime. With Adam-style optimizers, effective step magnitude $\sim m/\sqrt{v} = \mathcal{O}(1)$, so embeddings init small + scale grows slowly → bad early dynamics.
	2. **Stable cross-token modulation** — Zipfian access freq → heterogeneous embedding norms otherwise. Norm ensures comparable modulation strength across tokens.
- A normalized $d$-vector retains $d-1$ DoF → expressive power essentially preserved.
- Empirical: norm gives consistently higher MMLU/ARC/CMMLU/CEval trajectories, gap widens or persists through 1.3T tokens of overtraining.

### Scaling Factor $1/\sqrt{2N_l}$ (Sec D.1)

- Without the $1/\sqrt{2N_l}$ scaling in Eq 12, hidden-state std **explodes** to >5.5 within 50B tokens and JTok-M *underperforms* baseline.
- With it, std stays near baseline → consistently lower loss.
- Critical for variance control of additive residual injection in JTok-M.

---

## How JTok Differs From Related Ideas

| Method | What's added | Compute cost | Conditioning |
|---|---|---|---|
| **Vocab scaling** (SuperBPE, BLT, Over-tokenized) | More tokens in $V$ | unchanged inference, more tokens | discrete |
| **MoE / DeepSeek-MoE** | FFN expert pool, top-K routing | active FFN compute | hidden-state router |
| **Memory Layers / PEEK / Product Key Mem** | Replace FFN with sparse KV memory | sparse compute | learned keys |
| **AdaLN / FiLM** (hypernet) | Per-step modulation from conditioning net | hypernet matmul | external condition |
| **JTok (this)** | Per-layer per-token embedding table → Hadamard gate on FFN residual | $\mathcal{O}(d)$ elem-wise | **token id** (static) |
| **JTok-M** | Pool of tables + top-$K$ router | $\mathcal{O}(Kd + dn_e)$ | **token id + hidden state** |

- Key novelty: modulation is **driven by token identity itself** through a learned lookup, not by computing from hidden state via matmul. This makes the lookup overhead near-zero compared to MoE expert compute.
- JTok-M conceptually = "MoE on the embedding axis instead of the FFN axis."

---

## Hyperparameters (Appendix E)

- **Optimizer**: AdamW, weight decay 0.1, grad clip 1.0.
- **LR**: cosine schedule for dense S/M/L; **WSD** (warmup-stable-decay) for XL & all MoE — warmup 5%, decay 0%.
- **Vocab**: 50304 (small) → 152064 (XL/MoE) — note XL/MoE use a much bigger vocab too.
- **JTok extra params** roughly scale with $L \times V \times d$ — e.g. 6.5B for XL (28 layers × 152K × 1536).
- **JTok-M**: $n_e = 5$, $K = 2$ in all main experiments (different from the $\rho = 0.25$ used in Sec 4.2 standalone scaling).

---

## Critical Take / What's Interesting

- **Strongest aspect**: **clean scaling-law derivation** — by showing token-indexed params just rescale $A$ in the Kaplan form, the multiplicative compute-saving shift falls out analytically, and the empirical iso-FLOPs sweep matches the prediction (parallel log-log lines, $R^2 > 0.999$).
- **The "free lunch" story has caveats**:
	- Total parameter count balloons (44B embedding params on 17B backbone) — fine for training compute, but you need the system tricks (CPU offload, EmbP, Dedup) to make it deployable.
	- Inference HBM footprint stays small *only because* CPU offloading is practical with $\mathcal{O}(d)$ per-token transfers — this assumes you have host RAM + PCIe bandwidth headroom.
- **Why the gate is multiplicative on FFN residual (JTok) but additive (JTok-M)** isn't really justified — likely empirical choice; mixture of multiplicative gates may be harder to stabilize.
- **Connection to FiLM/AdaLN**: JTok is essentially a token-ID-conditioned FiLM layer where the conditioning network is a lookup table — interesting that learned per-token tables outperform learning from hidden state.
- **Connection to hyper-connections** (cited [60], same authorship cluster — Xiaohongshu): both relax the strict residual stream; this is part of an ongoing line at Xiaohongshu/SJTU exploring richer residual updates.
- **Comparison fairness**: experiments compare JTok-M-augmented MoE to vanilla MoE *with the same active params* — fair on FLOPs, but the JTok-M variant has dramatically more total params (e.g. 61B vs 17B). The compute-saving framing is the right lens (iso-FLOPs).
- **What's missing**:
	- No comparison to running a *bigger MoE* with equivalent total params instead of token-indexed tables.
	- No study of whether per-layer tables share information across layers (could be huge param savings).
	- No analysis of which tokens benefit most — does it mostly help rare tokens (where capacity is wasted) or frequent ones (where compute is wasted)?
	- Auxiliary load-balancing loss formulation copies MoE exactly — unclear if needed at the embedding granularity.

---

## Summary Equation Cheatsheet

- **JTok gate**: $\mathbf{p}_x^\ell = \mathbf{1} + \mathbf{s}^\ell \odot \text{Norm}_\varepsilon(\mathbf{E}^\ell[x])$, applied as $\Delta\hat{\mathbf{m}} = \Delta\mathbf{m} \odot \mathbf{p}$.
- **JTok-M router**: top-$K$ over $\mathbf{g}_x^\ell = \text{RMSNorm}(\mathbf{h}_x^\ell)^\top \mathbf{R}^\ell$, sigmoid-normalized weights, mixed vec $\mathbf{e}_x^\ell = \sum_i w_i \mathbf{E}_i^\ell[x]$.
- **JTok-M residual injection**: $\Delta\mathbf{r}_x^\ell = \frac{1}{\sqrt{2N_l}} \mathbf{s}^\ell \odot \text{Norm}_\varepsilon(\mathbf{e}_x^\ell)$, added to MLP output.
- **Effective params**: $N_\text{eff} = N_c(1 + \eta\gamma(\rho))$, $\eta = N_n/N_c$, $\rho = K/n_e$.
- **Compute saving**: $C^*_\text{JTok-M}/C^*_\text{base} = 1/(1+\eta\gamma(\rho))$ — scale-invariant.
- **Empirical**: 35% compute saving, +4–9 pts on downstream knowledge/reasoning tasks at 17B MoE scale.

---

**No public code release at time of writing.**
