# MM-SeR: Multimodal Self-Refinement for Lightweight Image Captioning

> Also titled *"One More Glance with Sharp Eyes: Rethinking Lightweight Captioning as a Practical Visual Specialist"*. Song et al. (KAIST / UNIST / CMU), arXiv:2508.21451v4, Dec 2025. No public code repo (only Google Sites project page).

**One-liner:** Build a captioning *specialist* on a **125M-param LM** (~56× smaller than LLaMA-7B), show it matches large MLLMs on factual captioning, then fix its residual **visual blindness** with a human-inspired **glance → sharpen** self-refinement loop (**MM-SeR**).

---

## 1. Motivation & Scope

### Why lightweight captioning at all
- **Captioning = foundational glue** for many vision-language apps, not just a standalone task:
  - **Video chatbots:** frame-wise caption per timestep for temporal understanding.
  - **Navigation/exploration robots:** captions encode observed scenes into graph-structured representations.
  - **Assistive devices** for the visually impaired; disaster-environment exploration.
- **Streaming use case:** these apps caption *many frames repeatedly* → per-call cost compounds.
- **Current approaches don't fit on-device:**
  - Open-source MLLMs need huge memory (see GPU table below).
  - Cloud APIs (OpenAI etc.) need stable network — unavailable in disaster/edge settings.

> **Edge memory vs MLLM GPU needs (Table 1):**
> | Edge device | available RAM | | MLLM | FP16 GPU usage |
> |---|---|---|---|---|
> | iPhone 16 | 8 GB | | LLaVA-1.5 7B / mPLUG-Owl3 8B | 16 GB+ |
> | Galaxy S25 | 12 GB | | LLaVA-NeXT 34B / InternVL 40B | 68 GB+ |
> | | | | LLaVA-OA 72B / Qwen2-VL 72B | 140 GB+ |
>
> Mismatch is the whole point: even a "small" 7B MLLM blows past phone RAM.

### Core research question
- In MLLMs like LLaVA-7B, **~96% of compute is the LLaMA backbone**. Detection/segmentation specialists in industry routinely run <500M params.
- **Q:** Is captioning *really* so hard it needs an LLM? Or can a tiny specialist do it?
- **Generalist vs specialist** (their definition, following prior work):
  - **Generalist (G):** MLLM trained on diverse data for broad zero-shot objectives (InstructBLIP, Qwen-VL, LLaVA-1.5, Cambrian).
  - **Specialist (S):** compact captioner trained only on captioning data (SmallCap, Tag2Text, LoCCa, this work).

---

## 2. Part 1 — Lightweight Captioner Construction & "Captioning ≠ Reasoning" Insight

### Model construction
- Take **LLaVA-1.5 codebase**, swap **LLaMA-7B → OPT-125M** (56× smaller LM).
- Keep everything else identical to LLaVA: ViT encoder + connector + LM, same batch size / LR.
- **Vision encoder:** CLIP ViT-L/14-336 (frozen).
- **Total ≈ 450M params** (most of it is the CLIP ViT, not the LM).
- **Training:** pretrain connector on **LLaVA ReCap-558K (LCS-558K)**, then fine-tune on task data (MS COCO / DCI / ShareGPT4V / GLaMM).
  - 4-layer MLP adapter w/ GELU. Pretrain adapter only (1 epoch); fine-tune adapter+LM.

### Metrics used
- **BLEU@4, METEOR, CIDEr** (n-gram); **BERTScore**, **CAPTURE (CAPT)** (detailed-caption metric).
- **LLM-as-judge:** **CLAIR** + **MLLM-as-a-Judge (GPT, "GPT" column)** w/ GPT-4o-mini — randomly sample 100 images × 10 seeds, report mean ± std (CLAIR shows higher variance; GPT-judge more stable).

### Results — single-sentence (MS COCO, Table 2)
- 450M specialist hits **CIDEr 129.6, B@4 39.4, METEOR 30.3, BERT 69.4**.
- **Beats all <1B small captioners** (e.g. CIDEr +6.9 over SmallCap, which *also* uses OPT-125M).
- **Comparable to 7B+ generalists** (LLaVA-1.5 7B = 133.7 CIDEr) despite ~16× fewer params.

### Results — detailed captioning (ShareGPT4V & DCI, GLaMM; Table 2)
- Authors *expected* OPT-125M to choke on long detailed captions. It didn't.
- Specialist (450M): **CIDEr 40.5, CAPT 45.9** on ShareGPT4V&DCI → **outperforms small specialists** (SmallCap, Tag2Text) and approaches generalists.
- On GLaMM: 450M model (CIDEr 29.1, CAPT 42.0) **beats LLaVA-1.5 7B** (CIDEr 23.4, CAPT 40.0).

> **Figure 2 anecdote:** even raw OPT-125M (no vision) shows surprising caption fluency w/ light fine-tuning ("a garden of a garden of a garden" — degenerate, but on-topic).

### KEY INSIGHT (the conceptual contribution of Part 1)
- **Factual image captioning does NOT require LLM-scale reasoning.**
  - LLMs matter for VQA / instruction-following (reasoning-heavy), but captioning is mostly **perceptual grounding**, i.e. *enumerating factual visual details*.
  - A 125M LM suffices → **compact specialist is the practical route** for captioning apps.
- Caveat that motivates Part 2: the small model is **less reliable** — semantic errors trace to **visual blindness**.

---

## 3. Visual Blindness (the failure mode MM-SeR fixes)

- **Definition:** MLLMs (and especially this tiny one) get **ambiguous / coarse visual features** from the encoder → produce wrong entities, attributes, relations even when language is fluent.
- **Two attributed causes** (from prior work, Eyes Wide Shut / Cambrian):
  1. **Language decoder** hallucinates details not in the image.
  2. **Vision encoder** supplies ambiguous info — the *critical bottleneck* they target here.
- **Evidence (App. F, Fig 12):** freeze CLIP, train an **MAE decoder** to reconstruct images from CLIP embeddings.
  - Reconstructions from **last-layer (LF)** features are blurry/coarse → final CLIP features lose fine detail.
  - **Multi-level (ML)** features (layers 13/18/23) reconstruct sharper → multi-layer info is richer.
- **Attention evidence (Fig 7, Fig 12):** single-pass captioner spreads attention *diffusely* across the whole image ("describe everything at once") → poor localization of fine regions.

> Human analogy: you glance to get the gist, then **look again, sharply**, at specific regions to nail details. Single-pass models only glance.

---

## 4. Part 2 — MM-SeR: Multimodal Self-Refinement Framework

### Core idea
Emulate the **human glance-then-sharpen** process as a **multi-stage** generation:
1. **Glance:** produce an initial (coarse) caption $o_k^{initial}$ — global gist.
2. **Sharpen:** feed that caption back in to *guide attention toward salient regions*, extract **richer multi-layer visual features**, and emit a refined caption $o_k^{refined}$.

- **Same LM** used for both passes; only the *connector* differs.
- **First multimodal self-refinement in MLLMs** (prior self-refine work = pure LLM/text; here visual evidence is injected into the refinement, guided jointly by language + vision).

### Pipeline (Fig 4, left)
```
ViT → Connector → LM  ──(1)──▶  o_k^initial   (glance caption)
                                     │ buffer
ViT (multi-layer) ─┐                 ▼
                   SeR-Connector ──(3)──▶ LM ──(4)──▶ o_k^refined  (sharpened caption)
o_k^initial ───────┘
```
- **(1)** standard connector → initial caption.
- **(2)/(3)** initial caption + multi-layer ViT features go into the **SeR-Connector**.
- **(4)** LM consumes refined visual embeddings + initial caption (as extra textual prompt) → refined caption.

### SeR-Connector (Fig 5, App. D)
- **NOT** the standard 2-layer MLP connector — a set of **Transformer blocks** designed to fuse refinement inputs.
- **Two inputs:**
  1. **Previously generated caption** $o_k^{initial}$, encoded as token embeddings → MLP → positional embeddings.
  2. **Multi-layer visual features:** collect $N$ visual tokens (dim $d$) from $m$ selected ViT layers, **concatenate along channel dim** → representation of size $N \times (m\,d)$ (preserves hierarchical info) → MLP.
- Fuses the two via **self-attention**, outputs $N \times (m d)$ visual representation forwarded to LM.
- **~50M extra params** for the SeR-Connector; **one extra inference step** for the LM.
- **Best layer triple:** ViT layers **{13, 18, 23}** (ablation, Table 12-III).

### Why expand layers, not vision encoders
- Prior visual-blindness fixes *add* encoders (e.g. Interleaved-MoF adds DINOv2 ≈ +300M params, +66.7% over this 450M model).
- MM-SeR instead **maximizes the existing CLIP encoder** by reading **intermediate layers** → cheaper, stays lightweight.

---

## 5. Training Strategy (two-stage fine-tune)

> Three total stages: pretrain (LLaVA) → **Stage 1** (initial caption) → **Stage 2** (refinement). 10 epochs stage 1, 2 epochs stage 2; batch 256; LR $2\times10^{-5}$; 2× A6000 GPUs.

### Stage 1 — learn the glance
- Standard LLaVA SFT: produce reliable **initial caption** from image, supervised on GT $c_k$.

### Stage 2 — learn to refine (the clever part)
- Training set $X = \{x_k\}$, $x_k = (i_k, c_k)$ = image + GT caption.
- **Naive approach (fails):** use model's own first-pass caption as refinement input, GT as target.
  - Problem: if first-pass caption is *too different* from GT (e.g. "a table in front of a window" vs GT "a cat sitting on a table"), there's **no meaningful basis for progressive refinement** → model learns to **ignore** the input caption and just regenerate from scratch.
- **Their fix — pseudo-initial captions $\hat{c}_k$:**
  - Prompt **GPT-4o-mini** to perturb $c_k$ at **0–3 elements** across 4 categories (inspired by DSG):
    - **Entity** (chair→cat), **Attribute** (color/material/count/texture/shape/size), **Relation** (A in front of B), **Action** (eating/blowing).
  - Result: $\hat{c}_k$ that is *close to* GT but with a few token-level errors.
  - **Generate 3 pseudo-initials per GT caption** (datasets ≈ 1.5–1.6M pairs in stage 2; see Table 15).
  - Train: SeR-Connector gets multi-layer features + $\hat{c}_k$ → LM takes refined embeddings + $\hat{c}_k$ as prompt → predict GT $c_k$.

### Rationale — why this works (targeted optimization)
- Let $\pi_\theta$ = LM. Pseudo-initial $\hat{c}_k$ differs from $c_k$ only at error positions $E_k = \{t \mid \hat{c}_{k,t} \neq c_{k,t}\}$.
- Sequence-level objective:
  $$\mathcal{L}(\theta) = -\mathbb{E}\Big[\sum_j \log \pi_\theta\big(c_{k,j} \mid i_k, \hat{c}_k, c_{k,<j}\big)\Big]$$
- Since $\hat{c}_k$ already matches $c_k$ except at $E_k$, **gradients concentrate on the erroneous tokens** → *targeted optimization*: keep correct parts, fix only the wrong ones.
- Writing $\Delta_k(\theta) = \log \pi_\theta(c_k\mid i_k,\hat{c}_k) - \log \pi_\theta(\hat{c}_k \mid i_k,\hat{c}_k)$, then $\mathcal{L} \propto -\mathbb{E}[\Delta_k]$ → minimizing $\mathcal{L}$ ≈ **maximizing the margin** of refined caption over its flawed precursor.
- **Resembles DPO:** treat $c_k$ = preferred, $\hat{c}_k$ = less-preferred response. Difference from DPO: DPO treats both responses symmetrically; here they get **distinct roles** (one is input, one is target).

> **Pseudo-initial robustness (App E.5, Table 13):** ablate 4 data conditions —
> Data1 (model's own stage-1 caption), Data2 (very different from GT → model ignores it, regenerates), Data3 (minor perturbations = default), Data4 (two pseudo-initials: one Data3-style + one identical to GT). Data3 best (CIDEr +3.1); Data4 ≈ Data1 → framework is **robust to moderate variation** in pseudo-initial quality.

---

## 6. Results

### MM-SeR gains (Table 3) — refinement helps everywhere
Two ablated inputs: **① initial caption**, **② multi-layer features**. Full = ①+②.

| Dataset | model | params | CIDEr | gain | CLAIR | gain | GPT | gain |
|---|---|---|---|---|---|---|---|---|
| **MS COCO** | base | 450M | 129.6 | — | 76.3 | — | 2.74 | — |
| | +MM-SeR (①+②) | 500M | 133.5 | **+3.9** | 77.6 | +1.3 | 2.82 | +0.08 |
| | +② only | 500M | 132.3 | +2.7 | | | | |
| | +① only | 500M | 131.9 | +2.3 | | | | |
| | single-pass +② | 450M | 130.6 | +1.0 | | | | |
| **ShareGPT4V&DCI** | base | 450M | 40.5 | — | 54.6 | — | 2.74 | — |
| | +MM-SeR (①+②) | 500M | 43.6 | **+3.1** (CAPT +2.5) | 57.7 | +3.1 | 3.02 | +0.28 |
| **GLaMM** | base | 450M | 29.1 | — | 51.8 | — | 2.64 | — |
| | +MM-SeR (①+②) | 500M | 30.4 | +1.3 (CAPT +0.8) | 53.4 | +1.6 | 2.88 | +0.24 |

- **Both inputs contribute**; ①+② > either alone > single-pass-with-②. → the **two-pass step itself** matters, not just adding multi-layer features.
- **Qualitative (Fig 6):** refinement fixes entity/attribute errors — "red and white train car"→"black and grey car", "feeding a light brown sheep"→"petting a cream-colored sheep", "white lab coat / metal fence"→"blue coat / red rope barrier".
- **Attention (Fig 7):** refinement pass attention becomes **concentrated on relevant regions** (vs diffuse first pass) → model "looks at what matters."

### Long-range Video QA (Table 4) — the practical payoff
- Setup: LLoVi pipeline on 10-min+ videos; per-frame captions aggregated, fed to **Qwen2.5-14B** for QA. Swap only the captioner.
- All small specialists trained on ShareGPT4V for fairness.

| Captioner | params | accuracy | time |
|---|---|---|---|
| BLIP-2 | 7.4B | 50.6 | 29m44s |
| LLaVA-1.5 (G) | 7.3B | **51.1** | 29m20s |
| SmallCap | 450M | 41.8 | 4m45s |
| Tag2Text | 900M | 47.1 | 7m14s |
| **Ours specialist** | 450M | 49.3 | 4m53s |
| **+ MM-SeR** | 500M | **50.8** | 5m10s |

- MM-SeR (500M) reaches **50.8 acc** — **approaches LLaVA-1.5 7B (51.1) with 14× fewer params**, and ~**6× faster** (5m vs 29m).

### Efficiency (Table 5)
- **93% fewer params**, **~82% shorter inference** vs MLLMs (single-pass spec is 97.97% faster than LLaVA-1.5; +MM-SeR still 97.28% faster). Refinement overhead = minimal.

### Edge device (App B.2, Table 8)
- Runs on **Jetson Nano (4GB)** and RTX 3090 — LLaVA-1.5-7B = **out of memory** on Jetson.
- Jetson: 2.6–2.7 GB mem, 13 W power, 20–21 s for 100 streaming images. Performance holds across devices.

---

## 7. Ablations & Extensions

### SeR-Connector design (App D.1, Table 12)
- **(I) Connector for initial gen:** plain **MLP (LLaVA-style) stays best** (CIDEr 129.6). Cross-Attn / Q-Former / Transformer variants only marginally better or worse → keep MLP for efficiency. (Cross-attention injection, used by SmallCap/Tag2Text, scores low 125.9 — confirms cross-attn underperforms for small LMs.)
- **(II) SeR-Connector type:** combining base config + structure (A) that uses both inputs = best (CIDEr 133.5). MLP/Qformer alternatives worse.
- **(III) ViT layer selection:** **{13,18,23}** (diverse trio) best vs single {23} or other combos.

### Larger LMs (Table 6) — does MM-SeR generalize up?
- Apply to **OPT-1.3B** and **LLaMA-2-7B** (LoRA), trained on ShareGPT+DCI.
- OPT-1.3B: CAPT 49.0→50.2 (+1.2), CIDEr 50.2→53.1 (+2.9).
- LLaMA-2-7B: CAPT 52.7→53.6 (+0.9), CIDEr 57.3→61.7 (+4.4).
- → **Refinement is not specific to tiny models**, but bigger LMs (10×, 56× larger) hurt deployability.

### Iterative refinement (Table 7) — capacity-dependent
- **OPT-125M:** more refinement passes give **no measurable gain** (CAPT: 45.9→48.4 at ×1, then 47.8/48.1 — plateaus/dips). Too small to use multi-step refinement signal.
- **OPT-1.3B:** **2–3 iterations help** (CAPT 49.0→50.2→50.5; GPT keeps rising 3.04→3.20).
- Mirrors LLM self-refine findings: **benefit grows with model capacity**. Future: dynamic iteration count.

### Learning strategy (App B.3, Table 9)
- **origin** (target datasets only) vs **more data** (COCO+ShareGPT+DCI+GLaMM) vs **distillation** (train on LLaVA-1.6-34B captions).
- **More data hurts** (CIDEr 40.5→36.8) — heterogeneous mixing weakens task alignment.
- **Distillation helps** (40.5→42.6) and **stacks with MM-SeR** (→43.6) → distillation + refinement are complementary.

### Other vision encoders (App B.4, Table 10)
- Multi-level feature strategy generalizes beyond CLIP: **SigLIPv2** and **DINOv2** both benefit.
- **CLIP + multi-level** > CLIP+DINOv2 combo on several metrics *and* cheaper. **DINOv2 alone** is worst (weak language alignment). SigLIPv2 multi-layer {13,18,23} best overall (CIDEr 45.5, CLAIR 58.2).

### GPT + self-refinement (App B.1, Fig 8)
- Even **GPT-4-turbo** benefits from a "take one more glance, fix any incorrect info" prompt — corrects subtle errors (stake bed sides, looking-off-camera, paved-path bridge). Parallels OpenAI o3's "thinking with images." Shows refinement is broadly useful, underexplored.

---

## 8. Datasets (App G)

| Dataset | #images | GT caps/img | caption style | stage-1 pairs | stage-2 (pseudo×3) pairs |
|---|---|---|---|---|---|
| **MS COCO** | 113K | 5 | single sentence (~10 words) | 565K | 1.6M |
| **ShareGPT4V** | 100K | (summarized to 5) | long, GPT-4o + human-verified (~200w → summarized to ~50w) | 500K | 1.5M |
| **DCI** | 7.4K | 10 | fully human-written, ~55 words | 74K | 0.2M |
| **GLaMM** | 550K | 1 | auto (detectors + scene parsers + LLM, ~45w, often noisy) | 550K | 1.6M |

- ShareGPT4V long caps **summarized to ~50w** (5 per image) to suit n-gram metrics + avoid hallucination; combined w/ DCI → 102.4K train / 5K test unified detailed benchmark.
- GLaMM captions noisy (1–2 wrong words every 2–3 caps) → models trained on it underperform ShareGPT/DCI.
- **Stages 1 and 2 use the same images** (just different targets/pseudo-inputs).

---

## 9. Limitations

### Residual visual blindness (App E.2/E.3, Figs 10–11)
- **Not fully solved.** Even large MLLMs (LLaVA-Next, LLaVA-OneVision, w/ grid-splitting + multi-encoders) still produce wrong captions in complex multi-object scenes.
- Their own specialist's remaining failure modes:
  - **Repetitive phrasing** (e.g. "black and white" repeated).
  - **Reduced fluency** (n-gram metrics don't capture this; MLLM-judge does).
  - **Limited OCR** (misreads map labels, brand text like "Dr. Martens").
  - **Lack of world knowledge** (small param count limits reasoning).
- Roots: (i) tiny param budget, (ii) limited-scale (~500K) machine-generated training captions.

### Evaluation limitations (App E.4)
- Metrics disagree / don't correlate well w/ human judgment; CLAIR high variance; small specialists' fluency loss invisible to n-gram metrics. **MLLM-as-judge most stable** but still imperfect.

### Future directions (App E.6)
- Minimal viable model size for assistive use? RL/DPO-style training (datasets already DPO-like)? Unified Transformer multimodal arch vs LLaVA? Better encoders (SigLIP/MAE) for fine detail? External tools (zoom/crop, à la GPT-o3)? Domain adaptation w/ user feedback? Captioning biases (gender/occupation/age)?

---

## 10. TL;DR Takeaways

- **Conceptual:** *factual captioning ≠ reasoning* → a 125M LM specialist matches 7B MLLMs on captioning at a fraction of the cost. Challenges the "everything needs an LLM" trend.
- **Method:** **MM-SeR** = human glance→sharpen; initial caption guides extraction of **multi-layer CLIP features** via a Transformer **SeR-Connector**, LM does a second pass. **First multimodal self-refinement in MLLMs.**
- **Training trick:** pseudo-initial captions (GPT-4o-mini perturbs 0–3 elements of GT) enable **targeted, DPO-like** refinement learning that fixes only erroneous tokens.
- **Practical:** runs on **Jetson Nano**, ~97% faster than MLLMs, drives long-range VideoQA to **50.8 acc (vs 51.1 for 7B LLaVA)**.
- **Honest:** visual blindness, OCR, world knowledge, fluency still partly unsolved; iterative refinement only pays off as LM grows.

---

## Related Notes

- **[DeepSeek-OCR](../DeepSeek-OCR/DeepSeek-OCR.md)** — Shared theme of squeezing capability out of **small vision-language models**, but a different goal: DeepSeek-OCR uses vision to *compress text context*; MM-SeR uses a tiny VLM (~450M, OPT-125M LM) for *image captioning* with a glance→sharpen refinement loop. Both argue much can be done below LLM-7B scale.
- **[LensVLM](../LensVLM/LensVLM.md)** / **[OCR-Memory](../OCR-Memory/OCR-Memory.md)** — Same broad VLM-efficiency neighborhood; those exploit the vision modality for *text-context compression*, whereas MM-SeR's lever is multi-layer CLIP feature extraction for perceptual grounding. Adjacent, not lineage.
