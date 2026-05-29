# OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory (2026)

> arXiv:2604.26622v1. Li, Zhang, Yang, Qu, Xu, Yang, Ding, Ngai. (HKU / UNT / Tsukuba / Yonsei). **No public code.**

**TL;DR**: store an agent's full interaction history as *images* (rendered, marked screenshots) instead of text. Retrieve relevant memory by having a fine-tuned **DeepSeek-OCR** model *locate* relevant regions via Set-of-Mark anchors, output only segment **indices**, then *deterministically transcribe* the verbatim text from a log. Optical encoding gives >10x token compression, and index-only output kills hallucination at the retrieval stage.

---

## 1. Motivation

- **Setting**: autonomous LLM agents in long-horizon, interactive deployments (web automation, mobile/app control). Success depends on **reusing accumulated experience** across episodes, not just within-task reasoning.
- **Core tension**: richness of experience vs. finite LLM context window.
  - Agents emit huge histories: reasoning traces, tool calls, env feedback, exact error messages, intermediate states.
  - **Storing/replaying raw trajectories** → prohibitively token-expensive (can't fit in context).
  - **Summarization / experience abstraction** → cheap but **lossy**: discards structural / temporal / procedural detail critical for debugging, error analysis, multi-step planning.
  - **Text-centric context compression** (latent mem, token pruning, streaming) → trades compression ratio against fidelity; *amplified* in multimodal/GUI settings where layout & structural cues matter and are easily lost in textual summaries.
- **Key insight** (from DeepSeek-OCR, Wei et al. 2025): dense text content can be encoded into **far fewer visual tokens than raw text tokens**, while **preserving full fidelity** of the original info → vision as a **high-density, loss-free medium for long-term memory**.

---

## 2. Related Work — three agent-memory paradigms

| Paradigm | Idea | Weakness |
|---|---|---|
| **Retrieval-based memory** (MemGPT/Packer, RAPTOR, MemoryBank/Zhong, Self-RAG, Memory-R1/Yan) | store interactions externally, fetch relevant fragments at inference via semantic similarity | brittle similarity matching: retrieved snippets topically related but **logically irrelevant**, esp. for causality / long-range deps |
| **Experience abstraction** (Voyager, AWM, ExpeL, Reflexion) | compress trajectories into reusable skills / workflows / rules | discards crucial low-level detail (exact errors, intermediate states, nuanced turns) needed for faithful retrospection |
| **Context compression** (LLMLingua, attention sinks/streaming, ACON, latent mem) | compress context itself via latent reps / pruning | text-centric → compression-ratio vs fidelity tradeoff; loses visual layout/structure in multimodal settings |

OCR-Memory sits across all three: it's retrieval (fetches fragments) but stores in **image modality** for compression **without** lossy abstraction.

---

## 3. Preliminaries / Formalism

- **Trajectory** for episode $e$: $\tau^{(e)} = \{x_1^{(e)}, \dots, x_{T_e}^{(e)}\}$, where each $x_t$ = user turn / reasoning trace / tool call / tool result.
- **External memory store** $\mathcal{M}$ accumulates past trajectories; after each episode write $\tau^{(e)} \to \mathcal{M}$.
- On new query $q$, **retrieval fn** $g_\theta$ reads $\mathcal{M}$, returns evidence $E$ injected into prompt:
  $$E = g_\theta(q, \mathcal{M})$$
  Agent then reasons & acts conditioned on $E$ and $q$. **Note**: $g_\theta$ is *not* required to answer $q$ — only to surface useful evidence.
- **Visual encoding (DeepSeek-OCR)**: treats *context optical compression* as image-encoding. For image $\mathbf{I} \in \mathbb{R}^{H\times W\times 3}$, encoder produces $\mathbf{Z} = f_{\text{enc}}(\mathbf{I}) \in \mathbb{R}^{n(r)\times d_{\text{latent}}}$.
  - $r$ = resolution mode controlling **compressed-token budget** $n(r) \in \{64, 100, 256, 400\}$ for inputs $512^2, 640^2, 1024^2, 1280^2$ respectively. → tunable compression ratio via input size.

---

## 4. Method — OCR-Memory

Shift memory storage + retrieval from **text domain to image domain**, so a memory module with a small text context can still consult arbitrarily long histories. A **specialized optical model** does retrieval only (decoupled from the primary reasoning agent).

### Memory bank structure
$$\mathcal{M} = \{m_i\}_{i=1}^N, \quad m_i = (\mathbf{I}_i, \{s_{i,k}\}_{k=1}^{K_i}, \pi_i)$$
- $\mathbf{I}_i$ = rendered **marked** image of a stored trajectory chunk.
- $\{s_{i,k}\}$ = corresponding **verbatim text segments** stored in a deterministic log.
- $\pi_i$ = metadata (timestamp, episode id, ...).

Retrieval: $\hat{S}(q) = g_\theta(q, \mathcal{M})$ (index set), $E = \text{Fetch}(\hat{S}(q), \mathcal{M})$ (maps indices → exact stored texts).

### Locate-and-Transcribe (the central trick)
- **Problem**: direct visual retrieval (read text off images) **hallucinates**, esp. at low resolution / blur.
- **Fix**: transform retrieval from free-form generation → **precise pointer selection**.
  - **Set-of-Mark (SoM) prompting**: each text segment $s_{i,k}$ in image $i$ is highlighted with a **red bounding box** + a unique **numeric ID** $k \in [1, K_i]$.
  - Model acts as a **listwise relevance extractor**: for image $i$ with $K_i$ segments outputs a **binary relevance vector**
    $$\hat{\mathbf{y}}_i(q) = (\hat{y}_{i,1}, \dots, \hat{y}_{i,K_i}) \in \{0,1\}^{K_i}$$
    label tokens strictly `"0"`/`"1"`; $\hat{y}_{i,k}=1$ ⇒ segment $k$ selected.
  - Global index set $\hat{S}(q) = \{(i,k) \mid \hat{y}_{i,k}=1\}$.
  - **Transcribe**: deterministically fetch exact stored texts:
    $$E = \text{Fetch}(\hat{S}(q), \mathcal{M}) = \bigoplus_{(i,k)\in\hat{S}(q)} s_{i,k}$$
    ($\oplus$ = concatenation under fixed template.)
- **Why it matters**: decouples *context understanding* (visual, in the OCR model) from *evidence generation* (deterministic fetch). Retrieved context is verbatim → **hallucination-free**. "Index-only" output is also much faster than decoding full text.

### Inference Scoring — calibrated, recall-oriented
- Greedy decode of strict binary tokens risks **false negatives** (missed evidence). Instead derive **calibrated relevance probs** from decoder logits at the label position ($z_{i,k}(1)$, $z_{i,k}(0)$):
  $$p_{i,k}(q) = \frac{\exp(z_{i,k}(1))}{\exp(z_{i,k}(1)) + \exp(z_{i,k}(0))}$$
- Selection rule — **"better to retrieve more than to miss"**: per image, threshold $\tau$ with Top-$K$ fallback:
  $$\hat{S}_i(q) = \{(i,k) \mid p_{i,k}(q) \geq \tau\} \cup \text{TopK}(\{p_{i,k}(q)\}_{k=1}^{K_i}, K)$$
  - **Minimum-guarantee policy**: prioritizes high-confidence segments ($\geq\tau$), but always returns ≥ $K$ per image even when uncertain (nothing exceeds $\tau$). Avoids per-instance threshold tuning.
  - Global: $\hat{S}(q) = \bigcup_{i=1}^N \hat{S}_i(q)$.
  - Defaults (Appendix A): $\tau = 0.4$, $K = 5$ fallback, **cap 20 segments / query** (token budget) keeping highest $p_{i,k}$.

### Multi-Resolution Trajectories ("vivid-to-fuzzy" / optical forgetting)
- Render chunks into hi-fi marked images $\mathbf{I}_i^{\text{hi}}$ (preserves spatial layout = semantic meaning, costly in text).
- **Memory age** $\Delta t_i$ → aging tier $\ell_i$ → lower image resolution:
  $$\ell_i = \rho(\Delta t_i), \qquad \mathbf{I}_i = \phi_{\ell_i}(\mathbf{I}_i^{\text{hi}})$$
  $\rho$ monotonic, $\phi_\ell$ downsamples by tier. Older memory → fuzzier thumbnails → fewer visual tokens. **Optical forgetting** cuts long-term token cost; low-res thumbnails keep **semantic gist** (enough for retrieval).
- **Active Recall Upscaling** (reversible "memory refresh"): if a retrieved segment lives in a compressed item with $\ell_i > \ell_{\min}$,
  $$\exists (i,k) \in \hat{S}(q) \text{ s.t. } \ell_i > \ell_{\min} \Rightarrow \mathbf{I}_i \leftarrow \mathbf{I}_i^{\text{hi}}$$
  restore to full fidelity for subsequent steps. (Appendix A: this is a **persistent** state change — restored item is exempted from $\rho$ for the rest of the episode → valid memories never revert to low-fi.)
- **Storage**: do NOT keep duplicate hi/low copies. Persist raw text logs + a **single current image**; re-render hi-res on demand from logs into an active visual cache. Analogy to human memory: old details blurred, gist remains; recall sharpens it.

---

## 5. Training Strategy

Backbone = **DeepSeek-OCR (3B)**. Pretrained for literal transcription → weak at **instruction-following / relevance judging**. Need it to *judge* which passages support a query, not just *read*. Fine-tune for **discriminative retrieval**.

### Repurposing HotpotQA (Yang et al. 2018)
- HotpotQA instance: question $q$, context of $K$ paragraphs ($K\approx 10$), annotated supporting facts $\mathcal{F}_{\text{supp}}$.
- **Discard the textual answer**; supervise only on supporting facts.
  $$y_k = \mathbb{I}[k \in \mathcal{F}_{\text{supp}}], \quad \mathbf{y} = (y_1, \dots, y_K) \in \{0,1\}^K$$
- Render $K$ paragraphs into SoM-marked image: $\mathbf{I} = \text{Render}(\{(k, p_k)\}_{k=1}^K)$.
- Objective: produce correct binary vector $\mathbf{y}$ → **next-token prediction for evidence retrieval** instead of for generation.

### Optimization
- Dataset $\mathcal{D} = \{(\mathbf{I}^{(n)}, q^{(n)}, \mathbf{y}^{(n)})\}_{n=1}^{|\mathcal{D}|}$.
- **Weighted BCE** (positives sparse vs distractors → emphasize recall):
  $$\mathcal{L}_{\text{BCE}}(\theta) = -\sum_{n}\sum_{k=1}^{K}\big[w_+ y_k^{(n)} \log p_k^{(n)} + w_- (1-y_k^{(n)})\log(1-p_k^{(n)})\big]$$
  with $w_+ > w_-$ (penalize **false negatives / missed evidence** harder). Default $(w_+, w_-) = (2.0, 1.0)$.
- **Partial freezing**: $\theta = \theta_{\text{vis}} \cup \theta_{\text{dec}}$. **Freeze vision encoder** $\theta_{\text{vis}}$ (preserve optical recognition), update **decoder via LoRA**:
  $$\theta_{\text{vis}} \leftarrow \text{fixed}, \quad \theta_{\text{dec}} \leftarrow \theta_{\text{dec}} - \eta\nabla_{\theta_{\text{dec}}}\mathcal{L}_{\text{BCE}}$$
  → learns text-query → SoM-anchor mapping without destabilizing OCR.
  - LoRA (App. A): rank $r=16$, $\alpha=32$, dropout 0.05, on `q/k/v/o_proj`.
- **Resolution Curriculum**: HotpotQA renders are hi-fi → domain gap vs degraded deployment memory. Per instance, sample tier and downsample:
  $$\ell \sim \text{Categorical}(\pi), \quad \tilde{\mathbf{I}} = \phi_\ell(\mathbf{I})$$
  forces reliance on coarse visual cues when fine detail unavailable → robust under optical compression. Default $L=2$ tiers, $\pi = [0.3, 0.7]$ over $\{1024^2, 512^2\}$ ($n(r) \in \{256, 64\}$).
- **Hyperparams** (App. A): AdamW $\beta=(0.9, 0.95)$, wd 0.1, grad clip 1.0, cosine LR peak $1e{-}5$, 10% warmup, batch 128, 3 epochs.

---

## 6. Experiments

### Setup
- **Benchmarks**: **Mind2Web** (Cross-Task split, web navigation) + **AppWorld** (API interactions). Mind2Web metrics: Element Acc, Action F1, Step SR, Task SR. AppWorld: SR by difficulty (Easy/Med/Hard) + Avg; emphasize **Hard** (extensive history backtracking).
- **Memory context window strictly = 4096 tokens** unless stated.
- **Primary reasoning agent**: GPT-4 (temp 0). Backbone OCR = DeepSeek-OCR 3B, encoder frozen.
- **SoM rendering**: red bbox (RGB 255,0,0, 3px), bold **36pt** sans-serif (Arial) white-on-red indices, bicubic resize. Recent **5** steps @ hi-res $1024^2$; all prior history @ $512^2$ with active upscale on hits.
- **Baselines**: Zero-Shot (lower bound), Dense Retrieval (text), MemoryBank (Zhong 2024), AWM (Wang 2024), ACON (Kang 2025).

### Main Results (Table 1)
| Method | M2W Ele Acc | F1 | Step SR | Task SR | AppWorld Easy | Med | Hard | Avg |
|---|---|---|---|---|---|---|---|---|
| Zero-Shot | 40.1 | 46.2 | 37.9 | 2.2 | 68.7 | 36.2 | 20.9 | 41.9 |
| Retrieval | 41.3 | 48.2 | 38.9 | 2.7 | 72.5 | 44.8 | 21.4 | 46.2 |
| MemoryBank | 43.8 | 49.5 | 39.2 | 3.3 | 81.3 | 50.1 | 24.9 | 52.1 |
| AWM | 49.1 | 55.7 | 42.6 | 4.3 | 84.1 | 53.6 | 27.2 | 55.0 |
| ACON | 48.2 | 54.1 | 41.4 | 4.1 | 84.8 | 55.1 | 28.7 | 56.2 |
| **OCR-Memory** | **53.8** | **59.2** | **46.1** | **4.8** | **86.2** | **57.4** | **30.8** | **58.1** |

- Beats strong abstraction baseline **AWM** on all metrics (Ele Acc 49.1→53.8, Step SR →46.1, Task SR 4.8 SOTA).
- Biggest gain on AppWorld **Hard** (30.8 vs Retrieval 21.4, AWM 27.2) — where long-history backtracking + precise structural grounding matter most.
- Takeaway: vision as compact carrier for lengthy textual history → precise evidence recovery at far lower token cost.

### Ablations

**SoM prompting** (Table 2, Mind2Web):
| Variant | Ele Acc | Step SR | Latency (s) |
|---|---|---|---|
| OCR-Memory (Full) | 53.8 | 46.1 | 1.7 |
| w/o SoM (Text Gen) | 46.5 | 39.2 | 5.3 |
| w/o SoM (BBox) | 49.2 | 44.5 | 2.1 |
- **Text-Gen variant** (generate the relevant text instead of index): higher hallucination + **~3x latency** (long-sequence decoding). **BBox variant** (predict boxes): fast but insufficiently precise. SoM = best accuracy/efficiency balance.

**Multi-Resolution Active Recall** (Table 3, Mind2Web):
| Strategy | Step SR | Task SR | Avg Tokens/frame |
|---|---|---|---|
| Static Low-Res ($512^2$) | 39.7 | 2.9 | 65 |
| Static High-Res ($1024^2$) | 46.5 | 4.9 | 256 |
| **Dynamic (Ours)** | 46.1 | 4.8 | **82** |
- Static low-res: cheapest but **−6.4% Step SR** (loses word meaning). Static hi-res: matches dynamic but **prohibitive token cost**. Dynamic = hi-res fidelity at near-low-res cost.

**Token Constraints** (Fig 2): vs Text-RAG across 1024–8192 token budgets. OCR-Memory beats Text-RAG on all 4 M2W metrics; **gap widens as budget shrinks**. Even at extreme **1024-token** limit, OCR-Memory stays functional (strong Ele Acc & Step SR) while Text-RAG degrades from info loss.

**Retrieval fidelity on long-context — NIAH / RULER** (Table 4): adapted Needle-in-a-Haystack by rendering docs to images, measure Recall@1.
| Context | Compression Ratio | Accuracy |
|---|---|---|
| 4k | 10.3x | 98.5 |
| 8k | 10.2x | 97.2 |
| 16k | 10.7x | 95.8 |
| 32k | 10.6x | 94.1 |
- Robust as context grows (94.1% @ 32k), with consistent **>10x** visual-token compression.

**Backbone generalization** (Table 5, Mind2Web): swap GPT-4 → Qwen3-32B.
| Method | Backbone | Ele Acc | Step SR | Task SR |
|---|---|---|---|---|
| Text-RAG | GPT-4 | 41.3 | 38.9 | 2.7 |
| OCR-Memory | GPT-4 | 53.8 | 46.1 | 4.8 |
| Text-RAG | Qwen3-32B | 35.2 | 31.5 | 1.8 |
| OCR-Memory | Qwen3-32B | 48.6 | 42.3 | 3.9 |
- Relative gains preserved across backbones → advantage comes from the **memory mechanism**, not a specific reasoner.

**Retrieval-level eval + faithfulness** (Table 6, Experience Retrieval Eval Subset from M2W; candidates from same website domain, pseudo-gold from uncompressed logs):
| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| Dense Text-RAG | 52.7 | 74.3 | 82.1 | 0.61 |
| OCR-Memory | **78.6** | **93.4** | **96.2** | **0.84** |
- **Content-level faithfulness**: free-form generative retrieval = **84.3%**; OCR-Memory = **100.0%** because it predicts indices then deterministically fetches verbatim text. (Faithfulness = recovered text exactly matches stored text; not a claim that every selected segment is relevant.)

**System efficiency** (Table 7, Mind2Web, continuous logging):
| Method | Disk/Episode | Text Tokens/Step | Retrieval Latency/Step |
|---|---|---|---|
| Text-RAG | 18 KB | 3,980 | 0.3 s |
| OCR-Memory | 1.47 MB | 596 | 1.7 s |
- **6.7x fewer** reasoning-context tokens (the scarce resource), at cost of more disk + higher latency. Deliberately **shifts cost from scarce reasoning tokens → cheap storage/preprocessing**. Not universally cheaper.

---

## 7. Limitations

- Requires **fine-tuning a specialized optical retrieval model** → extra training overhead (vs training-free retrieval baselines).
- **Rendering** logs to images is more compute-expensive than direct text storage; visual histories use **more disk** than raw text.
- Deployment **memory footprint**: vision encoder params must sit in memory alongside the primary LLM.

---

## Connections / commentary

- **DeepSeek-OCR lineage**: directly reuses "contexts optical compression" — the bet that an image of text is a denser-than-text token carrier. OCR-Memory's novelty is repurposing that compressor from *transcription* to *discriminative retrieval* (relevance vector over SoM anchors), plus the deterministic fetch that converts a lossy generative read into a lossless lookup.
- **Set-of-Mark** here mirrors its GUI-agent grounding use (CogAgent/AppAgent), but applied to **memory text segments** rather than UI elements — turning retrieval into pointer-selection over numbered anchors.
- **"Optical forgetting"** = a vision-domain analogue of resolution/precision-based memory decay; the reversible Active Recall upscaling is essentially an LRU-style refresh, but the "compression" is image downsampling rather than summarization → degradation is reversible from logs (vs summarization, which is destructive).
- The token-vs-disk tradeoff (Table 7) is the honest core: OCR-Memory is a bet that **context tokens are the binding constraint** in long-horizon agents, worth paying disk + latency + a 3B encoder to relieve.

---

## Related Notes

- **[DeepSeek-OCR](../DeepSeek-OCR/DeepSeek-OCR.md)** — The foundation. OCR-Memory uses a **fine-tuned DeepSeek-OCR 3B** as its optical encoder and inherits the core thesis (render text→image, encode with few vision tokens, >10× compression). OCR-Memory's contribution is applying it to **agent memory** + the locate-and-transcribe retrieval (output relevant segment *indices*, fetch verbatim text from logs → no retrieval hallucination).
- **[LensVLM](../LensVLM/LensVLM.md)** — Sibling DeepSeek-OCR descendant on a different task (text-QA vs agent memory). Both add a *recovery* step over compressed optical text: LensVLM's learned `EXPAND` tool ≈ OCR-Memory's "transcribe" / reversible Active-Recall upscaling. Both reject one-way lossy optical compression.
- **[Knowing When to Stop](../Knowing_When_to_Stop/Knowing_When_to_Stop.md)** — Complementary token-budget lever: OCR-Memory compresses *what's stored*, Knowing When to Stop limits *how much is read*. Both bet context tokens are the binding constraint in long-horizon settings.
