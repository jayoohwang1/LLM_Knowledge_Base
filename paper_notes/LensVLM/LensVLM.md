# LensVLM: Selective Context Expansion for Compressed Visual Representation of Text

**Xie et al., 2026 (Apple + Duke)** · arXiv:2605.07019 · Backbone: **Qwen3.5-9B-Base** · No public code (under legal review)

> **TL;DR**: Render long text as **compressed images** (cheap tokens), let the VLM **scan** the compressed view, then **selectively `EXPAND`** only the relevant images/regions back to uncompressed form (source text or hi-res image) via a learned tool. Trained with **SFT → RL (DAPO)**. Matches full-text accuracy at **4.3× effective compression**, beats baselines up to **10.1×**.

---

## 1. Motivation

- **Text-as-pixels paradigm**: feed text to a VLM as a *rendered image* instead of subword tokens.
	- Vision encoders map a fixed-size image → **fixed visual-token count**; varying **render resolution** = a continuous, retrain-free **compression knob**.
	- Compare to text-space methods: tokenizer-based compression (longer subword units) requires **vocab expansion + costly retraining** and bakes the rate in permanently; token-pruning (LLMLingua) is **discrete, lossy, no recovery path**.
	- Appealing for **long context**: self-attention is quadratic, so fewer tokens → big compute/memory wins.
	- Lineage: **PIXEL** (LM over rendered pixels), **Glyph** (long-context reasoning over rendered layouts), **DeepSeek-OCR** (optical context compression). LensVLM is the natural next step on this line.
- **The problem — fidelity collapse**:
	- As compression ↑, rendered images shrink, characters fall **below the vision encoder's effective resolution** → illegible → accuracy **collapses**.
	- Naive compressed-image reading: **31.3% @ 5×** vs **72.4% text upper bound**.
	- Past a threshold "the pixels simply do not contain the bits" — no amount of better *reading* recovers them.
- **Key reframe**: compression **should not** be a one-way uniform lossy transform; it **should** be a capability the model can *internalize to undo on demand*.
	- Name "Lens" = lens-like ability to focus on compressed context and **magnify** selected regions.

---

## 2. Core Idea: Selection-then-Expansion

> **Hypothesis**: compressed images preserve enough **coarse evidence** to *locate* the answer-containing region, even when they don't preserve the **token-level content** needed to *extract* it.
>
> → Use the compressed view for **selection**, then `EXPAND` to read the selected region uncompressed.

- **`EXPAND(k)` tool**: recovers compressed image $k$ into uncompressed form $\tau_k$.
	- $\tau_k$ can be: **source text**, **OCR-extracted text**, or a **high-resolution image** (variants compared in §5.3).
	- Introduces **no new information** beyond source $\mathcal{D}$ — $\tau_k$ is a deterministic function of $\mathcal{D}$; purely **representational** (re-presents a region in a more legible form).
- **vs. retrieval**: LensVLM acts as an end-to-end **retriever (selector) + reader**, but
	- retrieval **preprocesses & commits a-priori** to what's kept, needs an **offline index** over the full context (GPU embedding pass).
	- LensVLM keeps **global coverage** (all images rendered), selects **dynamically at inference** conditioned on evolving multi-turn context, needs only **CPU rendering**.
- **vs. prior visual-tool work** (DeepEye, ARM-Thinker, VTool-R1): those use tools to resolve **visual ambiguity within a given image**; here images are a **deliberate compression of text** the model chooses to decompress.

---

## 3. Method

### 3.1 Compression: Rendering Text as Images

- Deterministic renderer $f$: text $\mathcal{D}=(t_1,\dots,t_N)$ → $K$ images $\mathcal{V}=f(\mathcal{D})$.
	- Rendering config (font, geometry, line spacing) fixes **text capacity per image**.
	- Vision encoder tokenizes image $v_k$ → $n_k$ visual tokens; total $\sum_k n_k$ fed to LM.
- **Input Compression Rate (ICR)**:
$$ C_{\text{in}} = \frac{N}{\sum_{k=1}^K n_k} $$
	- $C_{\text{in}}=1$: one visual token per text token. $C_{\text{in}}\gg 1$: many text tokens packed per visual token.
- **Rendering presets** (Table 7, DejaVuSans, black-on-white via PIL, widths multiples of 32 for grid-aligned ViT tokenization):

	| Preset | Width | Font | Tok/img | Height | **VT/img** | ~Chars/img |
	|---|---|---|---|---|---|---|
	| 5× | 256px | 8pt | 405 | 284 | **72** | 2000 |
	| 10× | 192px | 6pt | 540 | 252 | **48** | 2700 |
	| 15× | 128px | 5pt | 378 | 190 | **24** | 1900 |

	- Clean **3:2:1 VT ratio** across presets. **5× = legible to human eye; 10× = small but distinguishable; 15× = mostly illegible.**
	- ViT token formula (patch stride 16, 2×2 merge): $n = \lceil H/16\rceil \times \lceil W/16\rceil / 4$.
	- Compression = product of two factors: **more chars per image** AND **fewer VTs per image** (smaller image → fewer patches).

### 3.2 Effective Compression Rate (ECR) — the fair-comparison metric

> Different methods turn the same source text into different numbers of **reader-visible tokens** (different encoders / retrieval budgets / tool responses). ECR normalizes this.

$$ C_{\text{eff}} = \frac{N}{T_{\text{reader}}} $$
- $N$ = source text token count; $T_{\text{reader}}$ = **total** reader input tokens.
- For LensVLM, $T_{\text{reader}}$ = initial compressed VTs **+ all `EXPAND`-appended tokens** ⇒ $C_{\text{eff}} \le C_{\text{in}}$.
- For visual reading w/o tools, $C_{\text{eff}} = C_{\text{in}}$.
- **Per-method ICR↔ECR (Table 11)**: e.g. at 15× ICR, LensVLM ECR = **10.1×** (expansions cost tokens); Glyph's GLM-4.1V encoder yields more VTs/image so its real ECR is lower (4.5/8.6/10.8× at the 5/10/15× presets).
- Retrieval ECR counts only **per-query reader input** (index pass is paid offline).

### 3.3 Inference: Multi-Turn Trajectory

- Policy $\pi_\theta$ sees all $K$ images + question $q$, samples:
$$ y = (r_1, k_1, \tau_{k_1}, \dots, r_M, k_M, \tau_{k_M}, a) \sim \pi_\theta(\cdot \mid \mathcal{V}, q) $$
	- interleaves reasoning $r_t$, `EXPAND($k_t$)` calls, tool responses $\tau_{k_t}$, terminating with answer $a$ after $M \le T$ calls.
	- **One image expanded per turn**; multi-hop → expand sequentially, each selection conditioned on prior expansions.
- Served via **vLLM**, up to **6 turns**, greedy decode, 2048 max tokens/turn.

### 3.4 Data Construction (synthetic tool-use traces)

- Render text from text-QA datasets at fixed ICR; track **char-level answer spans through pagination** → ground-truth evidence image set $E^\star \subseteq \{1,\dots,K\}$.
- **Easy/hard split** (à la Zheng et al. 2025): run base VLM single-turn on each sample; **correct → easy** (no tool needed), **wrong → hard** (tool needed).
- **Trajectory synthesis** with a larger model (**Qwen3.5-397B**): given $q$, the **uncompressed evidence**, and **gold answer** $a^\star$ → generate a coherent multi-turn CoT with $|E^\star|$ `EXPAND` calls:
$$ y^\star \sim \pi_{\text{synth}}(\cdot \mid q, \mathcal{D}_{E^\star}, a^\star) $$
	- **Why synthesis not distillation**: direct frontier distillation gives **hallucinated traces** (reasoning over illegible images) and `EXPAND` behavior isn't native to those models. Synthesis hands the model the evidence + answer, reducing the task to writing a coherent trace rather than solving under degraded vision.
- **Stats (Table 9/10)**: 4 datasets — **NQ** (1-hop), **HotpotQA** (2-hop), **MuSiQue** (2–4 hop, hardest), **HELMET** (long-doc QA). 143,145 generated → **105,202 hard** (73.5% keep rate). Docs padded to 3K–32K tokens, ~25 images & ~10K tokens/sample. **80% of hard → SFT; remaining 20% + easy → RL.**

### 3.5 Post-Training Recipe

> Three-stage pipeline: **(1) hard-sample filtering → (2) SFT on synthetic traces → (3) on-policy RL.** Vision encoder + projector **frozen**; only LM updated (adapting the LM is sufficient; training the encoder gives no benefit and saves compute).

**Supervised Fine-Tuning (SFT)**
- Decompose synthetic $y^\star$ into $L$ tokens; minimize autoregressive loss **only over policy-produced tokens** $\mathcal{A}$ (tool responses $\tau_{k_l}$ kept as context but **masked from loss**):
$$ \mathcal{L}_{\text{SFT}}(\theta) = -\sum_{i \in \mathcal{A}} \log \pi_\theta(y_i^\star \mid \mathcal{V}, q, y_{<i}^\star) $$
- Train on **hard subset only** (easy samples dilute the tool-use signal).
- LLaMA-Factory + DeepSpeed ZeRO-3, lr 5e-6, eff. batch 64, 2 epochs, ShareGPT multi-turn format, 8× B200.
- ⇒ $\theta_{\text{SFT}}$.

**Reinforcement Learning (RL)** — addresses SFT/teacher-policy mismatch (let the model learn under its **own** policy)
- Init $\theta \gets \theta_{\text{SFT}}$, optimize over $\mathcal{P}$ of $(\mathcal{V}, q, a^\star)$:
$$ \mathcal{J}(\theta) = \mathbb{E}_{(\mathcal{V},q,a^\star)\sim\mathcal{P},\, y\sim\pi_\theta} [R(y, a^\star)] $$
- **Reward** (no GT evidence images observed during RL — final-answer + answer-gated tool bonus):
$$ R(y, a^\star) = 0.7\cdot c + 0.3\cdot c\cdot u $$
	- $c = \text{acc}(y, a^\star)\in\{0,1\}$ via **LLM judge**; $u = \mathbb{1}[\texttt{EXPAND} \in y]$ tool-use indicator.
	- Tool bonus **gated by correctness** → incentivizes useful tool use; small bias toward tool use accepted (cost of an extra image ≪ cost of a wrong answer from not reading).
	- Episode capped at **T=6 turns** to bound cost & prevent tool spam. Empirically avg only **1.3 `EXPAND` calls/trajectory** (no degenerate spam).
- Algorithm: **DAPO** (via VERL), lr 1e-7, batch 32, 16 rollouts/prompt, temp 0.8, ≤6 calls, no entropy reg, 15K subsampled (hard + easy).

---

## 4. Experiments

### Setup
- **Train**: NQ, HotpotQA, MuSiQue, HELMET. **One single policy trained across all compression rates.**
- **OOD eval**: **Qasper**, **LongBench v1** (12 subtasks), **RULER** (QA subsets, 128K→~32K). All eval ≤32K text tokens. **Judge = Qwen3.5-397B** (cross-checked vs GPT-oss 120B: agree 91.7%, κ=0.81, Δ 0.3pp → conclusions judge-independent).
- **Metrics**: QA accuracy (macro-avg), **Selection Accuracy (Sel. Acc.)** = fraction of trajectories where ≥1 `EXPAND` hits a GT evidence image, ECR. Eval T=6.
- **Backbone choice**: Qwen3.5-9B-Base — native interleaved image-text, variable-resolution ViT (no resizing), and 9B is the **minimum scale where tool-use emerges** (§5.4).

### Baselines (all share Qwen3.5-9B-Base except Glyph)
- **Text** (upper bound, full doc as text), **Comp. Image** (naive compressed-image read), **Base+Expand** (untrained backbone + tool), **LLMLingua-2** (text token-pruning), **ColPali** / **Jina-v4-MM** (visual retrieval), **BM25 / BGE-M3 / Jina-v4 / Qwen3-Emb** (text retrieval), **Glyph** (GLM-4.1V-9B, SOTA visual-text compression).

### 4.2 Main Results — Text QA (Fig. 3, Table 11)

| Method | ECR | Overall Avg Acc |
|---|---|---|
| **Text (upper bound)** | 1.0× | **72.4** |
| Comp. Image | 5× | 31.3 |
| Base + Expand | 4.5× | 39.2 |
| Glyph | 4.5× | 47.2 |
| LLMLingua-2 | 5.5× | 47.2 |
| ColPali | 4.7× | 56.1 |
| **LensVLM** | **4.3×** | **68.9** |

- **LensVLM @ 5× ICR = 68.9% ≈ 72.4% text upper bound, at 4.3× ECR** → best accuracy-compression trade-off.
- **Beats every baseline at every compression rate** (Fig. 3, all 7 benchmarks).
- **Gap to baselines GROWS with compression** (10×: 62.1 vs ~36–57; 15×: 52.1 vs ~26–52) — the central selling point: as visual fidelity drops, reliance on **expanded content** rises and naive readers collapse while LensVLM degrades gracefully.
- Selection accuracy: **76.8 / 71.0 / 52.1%** @ 5/10/15×.

### Code Understanding (zero-shot transfer, Table 1/15)
- No code-specific training. RepoQA (6 langs) + CodeQueries.
- **LensVLM 38.1 / 26.1 / 20.0% @ 5/10/15×** (ECR 3.2/6.3/8.2×) — improves over Comp. Image (26.7/6.1/3.2), beats Glyph at every rate.
- RepoQA: 63.4% @ 5× vs 92.4% text; advantage over Comp. Image/Glyph **widens at higher compression** (+20pp / +17pp over Glyph @ 10×/15×).
- Persistent gap to text bound = domain mismatch (code is dense, adjacent blocks visually near-identical when compressed). Lower ECRs here because contexts are short (fixed tool-call cost amortizes worse).

### Document Understanding (native multimodal, Table 2)
- **MMLongBench-Doc** (PDFs up to 200 images). Images natively rendered (not text-rendered), downscaled by $N\times$. Two tool variants trained:
	- **`LensVLM + OCR`** (`read_image` → PaddleOCR-VL text) and **`LensVLM + Zoom`** (`read_image` → full-res image).
- @ 5×: compressed images still legible, base model wins. **Tool advantage emerges as compression ↑**: @ 15× **Zoom 41.9% vs base 31.1%**.
- **Zoom > OCR** on native docs because OCR discards **layout & visual cues**; but Zoom costs more tokens (lower ECR: 4.2/7.7/10.3× vs OCR 5.0/9.4/13.5×).
- **Practical tool-choice guidance**: **text expansion for rendered text** (text tokens = higher fidelity than visual); **image zoom for native documents** whose layout carries task-relevant info.

---

## 5. Analysis & Ablations

### 5.1 Training Erases Rendering Sensitivity (Fig. 4, Table 3)
- **Base model is wildly sensitive to rendering config**: accuracy spans **32 points** across 20 configs, incl. an **18-point gap within a single font group** — optimal config requires expensive search.
- Matched ~3× compression, 10pt font, 3 widths (700/295/145px): Base drops monotonically **56.2→38.2%** (Δ18.0); after training all **converge to ~70%** (Δ**0.5**).
- ⇒ training makes compression **robust to rendering choices** (two causes: model specializes to its own config + tool text is rendering-invariant). Config becomes a **flexible design choice, not a hyperparameter to tune.**

### 5.2 Attention Analysis (Fig. 5, Table 16)
- 8 standard self-attn layers analyzed (Qwen3.5-9B is hybrid: 8 SDPA + 24 GDN linear-attn layers, GDN has no attn matrix). Position-debiased GT-alignment used (base has 30% argmax-on-first-image bias).
- **Base model already localizes evidence above chance** (retrieval-head phenomenon) but can't improve alignment @ 15× (signal depleted).
- **Training redirects attention**:
	- → **tool-response text +14.6/+15.1/+15.8pp** (5/10/15×) — model learns to *read expanded content*.
	- ← **distractor (non-GT) images −4.3/−4.4/−6.4pp**, own tool-call turn −8pp, question −1pp.
	- Total vision attention **drops** (13→8.1%, 11.1→5.7%, 12.5→4.4%); shift is strongest in mid-late layers (L15–L27).
- **Evidence-focused shift grows with compression** → explains why LensVLM's advantage increases as fidelity drops.

### 5.3 Tool Modality: Text vs Image (Fig. 6)
- 5 variants differing in `EXPAND` response: (1) no tool/direct, (2) no tool/reasoning, (3) OCR text, (4) original text, (5) hi-res image.
- **Three findings**:
	1. **Bottleneck = information access, not reasoning** — CoT without tool access gives **no benefit** over direct answer (reasoning over illegible images adds noise).
	2. **Original source text is best response at every compression** on rendered text (text tokens > visual tokens for fidelity); **OCR ≈ original** when source text unavailable (good practical substitute).
	3. On **native multimodal docs the ranking reverses**: **zoom (image) > text** because text discards layout/visual structure.

### 5.4 Effect of Model Scale (Table 4)
| Size | Base QA | LensVLM QA | Sel. Acc. |
|---|---|---|---|
| 2B | 13.2 | 45.5 | 49.4 |
| 4B | 25.7 | 56.3 | 59.1 |
| **9B** | 31.3 | **68.9** | **76.8** |
- **2B** learns tool-call syntax but QA stays low (weak visual understanding). **4B** decent selection (59.1) but QA plateaus (reading comprehension is the bottleneck). **9B** = first scale where both visual grounding + text comprehension reach effective levels → **9B is the minimum for the full agentic flow.**

### 5.5 Contamination-Free Eval (Table 5)
- Built guaranteed-unseen testbed: **PubMed Central** articles published **after** Qwen3.5's pretraining cutoff (Feb 16, 2026); 26,164 train / 533 test, fresh model trained on it.
- **LensVLM 86.8 / 60.0 / 53.7% @ 5/10/15×** (Text 92.1, Comp. Image 39.5/25.5/16.3) — **+39.7pp avg over Comp. Image**, gains hold (largest at high compression). Confirms results aren't pretraining contamination.

### 5.6 vs Retrieval (Table 6)
- Same reader. LensVLM **matches or exceeds every retriever at every compression level using only CPU rendering**.
- Embedding/visual retrievers (BM25 62.5 @5×, ColPali, Jina-v4-MM) require a **GPU embedding pass over the full context at index time**; LensVLM needs only CPU.
- **Complementary**: a cheap retriever could prune grossly-irrelevant chunks *before* rendering.

### 15.1 SFT vs RL (Table 13)
| Stage | QA @5/10/15× | Sel. Acc @5/10/15× |
|---|---|---|
| Base | 31.3 / 21.0 / 18.1 | 25.8 / 18.0 / 18.3 |
| +SFT | 63.7 / 56.9 / 45.7 | **75.4** / 64.7 / 45.9 |
| +SFT+RL | **68.9 / 62.1 / 52.1** | **76.8 / 71.0 / 52.1** |
- **SFT teaches the format** (+32pp @5×: selection, extraction, tool syntax). **RL refines *where* to apply it**, not new capability.
- RL gains **largest at high compression** (Sel. +6.3pp @10×, +6.2pp @15×) — learns subtler cues from degraded images; +5–6pp QA across all rates.
- RL raises `EXPAND` rate (ECR 4.6/8.1/10.7→4.3/7.4/10.1×) but extra calls **co-occur with QA gains** → productive, not degenerate.

### 15.2 Oracle Selection (Table 14, in-domain)
- **Oracle Selection (text) 80.0%** *exceeds* in-domain Text upper bound (78.1%) — concise GT image text < full noisy doc.
- **LensVLM 76.4% @5× approaches the 80.0% oracle ceiling**; **Base+Expand only 42.5%** → post-training is essential for effective selection (info-theoretic possibility ≠ free).

### 15.3 Multi-Turn Budget (Fig. 9)
- **Single `EXPAND` call captures most gain (60.8%)**; 2nd call +6.1pp (66.9%, multi-hop), then diminishing (<0.5pp after 4).
- Model **self-regulates to ~1.3 calls** even with generous budget. MuSiQue (multi-hop) benefits most (+22pp from 1→4 calls).

---

## 6. Information-Theoretic Framing (App. 8)

- Two lossy stages: render $\mathcal{D}\to\mathcal{V}$, encode $\mathcal{V}\to\mathcal{T}$. Data-processing inequality:
$$ I(a^\star; \mathcal{T}\mid q) \le I(a^\star; \mathcal{V}\mid q) \le I(a^\star; \mathcal{D}\mid q) $$
	- `EXPAND` adds no info ($\tau_k$ is deterministic in $\mathcal{D}$) — purely representational, recovers what encoding lost.
- **Selection ≠ extraction**: selection predicts a small evidence set $E^\star$; extraction predicts a string over a huge space → compressed VTs may keep **coarse cues for selection** while losing char-level reading. (Property of the task distribution, not a theorem — fails for factoids whose only clue *is* the answer phrase.)
- Benefit of selective expansion is explicit:
$$ D_{\text{no}}(\rho) - \bar{D}(\pi) = p_\pi(D_{\text{no}}(\rho)-D_{\text{hit}}) - (1-p_\pi)(D_{\text{miss}}-D_{\text{no}}(\rho)) $$
	- Tool helps when **direct reading degrades faster than selection accuracy** — exactly LensVLM's empirical regime, advantage growing with compression.
- Empirically: base model's direct QA falls 31.3→18.1% (5→15×, −42% rel.); base selection only 25.8→18.0% → info-theoretic possibility of selection isn't enough, **post-training is required** to exploit it.

### Model-Agnostic (App. 9, Fig. 8)
- Framework is **backbone-independent** — applied via **prompting alone** to untrained 9B (+7.9 to +10.2pp) and **frontier Sonnet 4.6** (+4.8 to +9.0pp) with **no fine-tuning**. The information-loss bottleneck affects all scales; select-then-expand is a general inference strategy.

---

## 7. Efficiency & Limitations (App. 10)

- **Token / KV savings**: @15×, 20 images: Text 10,686 tokens (334 MiB KV) → LensVLM 2,288 (71 MiB KV), **78.6% reduction**. @100 images: **84.2% reduction**.
- **Cost = latency**: one `EXPAND` → 2 sequential turns (~17s vs ~8s single-turn Text, **~2× overhead**). Two sources:
	1. **Multi-turn sequential decoding** — each `EXPAND` forces a generate-then-prefill round-trip (can't parallelize).
	2. **ViT encoding** — every rendered image passes through encoder+projector (text-only baselines skip this).
	- Both inherent to **VLM tool-use architecture**, not specific to LensVLM. Future: parallel tool dispatch / cached vision encodings.
- **Limitations**:
	- 9B is the floor — smaller models lack capacity for the combined visual-understanding + multi-step-planning + reading-comprehension demands.
	- Latency overhead from sequential tool turns + repeated ViT passes.
	- Code transfer still far below text upper bound (dense syntax, visually similar blocks).
	- Storage: tool needs source text / original images kept available (negligible vs VRAM savings).

---

## Relation to DeepSeek-OCR-style Optical Compression

- **Shared premise**: text → image is a **lossy compression channel**; rendering resolution = the rate. DeepSeek-OCR ("contexts optical compression") and Glyph both push how *much* text a VLM can read from a compressed render.
- **Key difference**:
	- DeepSeek-OCR / Glyph / PIXEL invest in **reading better at a fixed compressed view** → bounded by what remains *legible* at that rate; hard ceiling once pixels lack the bits.
	- LensVLM **keeps the cheap compressed view for global scanning but adds a recovery path** (`EXPAND`) — it doesn't try to read illegible pixels, it **selectively decompresses** the regions that matter back to lossless source.
	- Reframes compression from a **one-way encoder property** into a **two-way, on-demand model capability** — pushing *past* the visual-reading ceiling those methods hit.
- Also distinct from **token-pruning optical-compression hybrids**: those drop tokens permanently; LensVLM's compression is **reversible per-region at inference**.

---

## Related Notes

- **[DeepSeek-OCR](../DeepSeek-OCR/DeepSeek-OCR.md)** — The method LensVLM builds on and explicitly improves. DeepSeek-OCR is **one-way** optical text compression whose accuracy collapses at high ratios (sub-resolution characters); LensVLM reframes compression as a **two-way, on-demand capability** — scan compressed, then `EXPAND` only relevant regions back to lossless source — pushing past DeepSeek-OCR's legibility ceiling.
- **[OCR-Memory](../OCR-Memory/OCR-Memory.md)** — Sibling DeepSeek-OCR descendant (agent memory vs text-QA). Both add a recovery/transcribe step over compressed optical text and share the reversible-vs-destructive-compression stance; OCR-Memory fetches verbatim from logs, LensVLM decompresses via a learned tool + RL.
- **[Knowing When to Stop](../Knowing_When_to_Stop/Knowing_When_to_Stop.md)** — **Shares authors** (Roy Xie, Bhuwan Dhingra). Both attack the long-context token budget: LensVLM compresses the representation, Knowing When to Stop truncates reading once sufficient — orthogonal and composable.
