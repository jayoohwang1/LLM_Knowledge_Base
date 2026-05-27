# Papers Overview: Context Compression, Memory, and Latent Communication

A landscape report on ~38 papers covering **context/KV compression**, **scaling memory as a new axis**, **latent multi-agent communication**, and **optical text compression**. Grouped by theme; cross-cutting synthesis at the end.

---

## 1. Latent / KV-Cache Multi-Agent Systems

> Theme: agents stop talking in text — they share hidden states, KV caches, or learned memory blobs. Motivation: text is lossy + slow.

### Latent Collaboration in Multi-Agent Systems (2025) — `arXiv:2511.20639`
- **Core**: **LatentMAS** — training-free framework where each agent generates **auto-regressive latent thoughts** via last-layer hidden embeddings; a **shared latent working memory** transfers reps across agents losslessly.
- **Method**: no text mediation between agents; each agent does "latent CoT" then writes its hidden states into the shared buffer.
- **Results**: across 9 benchmarks (math/sci/commonsense/code), **+14.6% accuracy** over text MAS, **70.8–83.7% token reduction**, **4–4.3× faster** inference.
- **Key terms**: latent CoT, shared latent memory, training-free MAS.

### LatentMem: Customizing Latent Memory for Multi-Agent Systems (2026) — `arXiv:2602.03036`
- **Core**: learnable per-agent memory framework — fights *memory homogenization* (all agents see same blob) and *information overload* (memory entries too fine-grained).
- **Method**: **experience bank** stores raw trajectories cheaply; a **memory composer** synthesizes compact latent memory conditioned on retrieved experience + agent role. Trained with **Latent Memory Policy Optimization (LMPO)** — task-level RL signal propagates back through latent memory to the composer.
- **Results**: up to **+19.36%** over vanilla MAS; beats existing memory architectures.
- **Connection**: extends [[latent-collaboration-mas]] by making memory role-aware and learnable.

### Latent Briefing: Efficient Memory Sharing for MAS via KV Cache Compaction (2026) — Ramp Labs (no arXiv yet)
- **Core**: orchestrator + worker pattern; worker keeps a **persistent KV cache** of the orchestrator's trajectory across calls, reusing 90%+ unchanged prefix.
- **Method**: uses **attention patterns** to detect which context positions matter, discards the rest at the KV level (representation, not text).
- **Results**: ~**49% median token savings** on 32–100k-token docs on LongBench v2, with comparable accuracy.
- **Connection**: same goal as [[polykv]] (share KV across agents) but motivated from a single-orchestrator → worker handoff.

### PolyKV: Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference (2026) — `arXiv:2604.24971`
- **Core**: N agents reading the same context → **one compressed KV cache pool** shared via HF DynamicCache, injected into each agent's context.
- **Method**: **asymmetric quantization** — K at int8 (q8_0) for softmax stability, V via **TurboQuant MSE**: Fast Walsh-Hadamard rotation + 3-bit Lloyd-Max quantization tuned to N(0,1).
- **Results**: stable **2.91× compression**, e.g. Llama-3-8B with 15 agents @ 4K ctx → KV drops **19.8 GB → 0.45 GB (97.7% reduction)**.
- **Connection**: complementary to [[latent-briefing]]: PolyKV compresses the *physical* KV; Latent Briefing compresses *which positions are kept*.

### Recursive Multi-Agent Systems (2026) — `arXiv:2604.25917`
- **Core**: **RecursiveMAS** — agents stacked into a recursive latent loop; heterogeneous LLMs talk via hidden states.
- **Method**: lightweight **RecursiveLink** module: inner link generates latent thoughts inside each agent, outer link transfers hiddens across agents. Only **0.31%** of total params trained (link params); LLM backbones frozen. Two-stage inner-outer training with shared gradient-based credit assignment.
- **Results**: **+8.3% avg accuracy**, **1.2–2.4× speedup**, **34.6–75.6% token reduction** across 4 collaboration patterns × 9 benchmarks.
- **Connection**: explicit generalization of [[latent-collaboration-mas]] to heterogeneous agents with a learned (not training-free) bridge.

**Group synthesis**:
- **Three sub-axes**: (a) *what is shared* — raw KV (PolyKV, Latent Briefing) vs synthesized latent memory (LatentMem) vs hidden states (LatentMAS, RecursiveMAS); (b) *who trains* — training-free (LatentMAS) vs link-only (RecursiveMAS) vs full memory module RL (LatentMem); (c) *topology* — flat collaboration vs orchestrator/worker vs recursive loops.
- **Common claim**: ditching text gives 2–4× speedup and 30–80% token savings without accuracy regression.
- **Open question**: PolyKV is the only one that explicitly compresses the *bits* of shared state; others rely on architectural compression (e.g. memory tokens). Combining the two is an obvious next step.

---

## 2. Optical / Visual Context Compression

> Theme: text-as-image is way denser than text-as-tokens. A 1024-token page can be encoded in ~100 vision tokens at near-lossless OCR.

### DeepSeek-OCR: Contexts Optical Compression (2025) — `arXiv:2510.18234`
- **Core**: compress long *text* contexts by rendering them as images and encoding via a vision tower.
- **Method**: **DeepEncoder** (vision tower keeping low activations + high compression) + **DeepSeek3B-MoE-A570M** decoder.
- **Results**: **<10× compression → 97% OCR**, **20× → ~60% OCR**. Beats GOT-OCR2.0 @ 256 tok/page using **100 vision tokens**; beats MinerU2.0 (6000+ tok/page) using <800.
- **Key terms**: optical 2D mapping, vision-token compression knob.

### OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory (2026) — `arXiv:2604.26622`
- **Core**: extends [[deepseek-ocr]] to **agent memory** — render long trajectories as annotated images and retrieve via visual anchors.
- **Method**: trajectories rendered with unique visual IDs; **locate-and-transcribe** paradigm picks relevant image regions at retrieval.
- **Results**: SOTA on **Mind2Web** and **AppWorld** (long-horizon agent benchmarks).
- **Connection**: direct downstream application of DeepSeek-OCR's compression idea, with retrieval added.

### LensVLM: Selective Context Expansion for Compressed Visual Representation of Text (2026) — `arXiv:2605.07019`
- **Core**: VLMs scan compressed (small-resolution) text images, then **learn a tool to expand only relevant regions** to full resolution. Solves the accuracy cliff DeepSeek-OCR hits at high compression.
- **Method**: post-training recipe on Qwen3.5-9B-Base; learned "lens" tool decides which crops to re-render at native resolution.
- **Results**: matches full-text upper bound at **4.3× compression**, beats baselines up to **10.1×** across 7 QA benchmarks; generalizes to doc/code understanding.
- **Connection**: corrects the failure mode of [[deepseek-ocr]] (chars shrink past encoder resolution).

### MM-SeR: Multimodal Self-Refinement for Lightweight Image Captioning (2025) — `arXiv:2508.21451`
- **Core**: 125M-param captioner (56× smaller than LLaMA-7B) using a **glance-then-attend** loop mimicking human perception.
- **Method**: **DeepLens** extracts detailed reps from informative regions found in the initial glance; Sharp-Eyed Refinement framework iteratively improves caption via better visual grounding.
- **Connection**: thematic cousin — *selective region expansion* like LensVLM, but for captioning, not text compression. Weakest fit in this group; doesn't directly compress text.

### Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals (2025, NeurIPS) — `arXiv:2502.01025`
- **Core**: **dynamic context cutoff** — specific attention heads encode "sufficiency signals" detectable by a small classifier; LLM self-terminates context processing.
- **Method**: lightweight probe on attention activations predicts when enough info has been processed. Smaller models need the probe; larger models do it intrinsically via prompting (**emergent self-assessment**).
- **Results**: **+3.4% accuracy**, **1.33× token reduction** across 6 QA datasets (up to 40K tokens), 1B–70B models.
- **Connection**: not visual but lives in this group as another *selective* compression strategy — stop reading instead of compress.

**Group synthesis**:
- **DeepSeek-OCR is the anchor**; OCR-Memory and LensVLM are direct descendants extending to (a) memory retrieval and (b) selective expansion.
- **Common motif**: don't process everything — either re-encode densely (visual) or selectively zoom (LensVLM, Knowing-When-to-Stop).
- **MM-SeR is loosely related**: same selective-attention spirit but applied to captioning, not compression.

---

## 3. Scaling Memory / Embedding as a New Axis

> Theme: instead of more FFN/attention params, scale a **huge lookup table** (n-gram embeddings, hash memory, token-indexed FFNs). Decouples capacity from FLOPs.

### Engram — Conditional Memory via Scalable Lookup (2026, DeepSeek) — `arXiv:2601.07372`
- **Core**: inject **massive static N-gram embedding tables** into Transformer layers as a "conditional memory" — knowledge lookup primitive missing from vanilla Transformers.
- **Method**: replaces ~20% of MoE params with hash-based lookups. **Sparsity Allocation law** quantifies trade-off.
- **Results**: Engram-27B ≥ MoE-27B at iso-params/FLOPs on knowledge, reasoning, code, math.
- **Connection**: same family as [[scone]], [[stem]], [[jtok]], [[longcat-flash-lite]] — the "embedding-as-memory" wave.

### SCONE — Scaling Embedding Layers in Language Models (2025) — `arXiv:2502.01637`
- **Core**: scale **n-gram embedding count** and the model that learns them while keeping inference FLOPs/memory fixed.
- **Results**: 1B accelerator-resident params beat a 1.9B baseline using ~half FLOPs/memory.
- **Connection**: predecessor concept that the rest of this group expanded.

### STEM: Scaling Transformers with Embedding Modules (2026) — `arXiv:2601.10639`
- **Core**: replace FFN **up-projection with a layer-local embedding lookup**, keeping gate + down-proj dense.
- **Method**: token-indexed, no runtime routing, supports CPU offload + async prefetch. Eliminates ~1/3 of FFN params per token.
- **Results**: +3–4% accuracy at 350M/1B scale; gains on ARC-C, OBQA, GSM8K, MMLU. Also enables **interpretable knowledge editing** (edit a token's embedding entry).
- **Connection**: same idea as [[memoryllm]] but with gate+down-proj preserved.

### MemoryLLM: Plug-n-Play Interpretable Feed-Forward Memory (2026, Apple) — `arXiv:2602.00398`
- **Core**: **decouple FFNs from self-attention** entirely; train FFNs in isolation on context-free token embeddings.
- **Method**: FFN becomes **token-wise lookup (ToL)** — pre-computable, on-demand VRAM↔disk transfer. **Flex-MemoryLLM** is a middle-ground variant that bridges the perf gap.
- **Connection**: more aggressive STEM — entire FFN becomes a lookup; trades performance for interpretability and offload.

### JTok: Token Embedding as another Axis of Scaling Law (2026) — `arXiv:2602.00800`
- **Core**: **token-indexed modulation vectors** retrieved from auxiliary embedding tables; modulate the backbone element-wise (no FLOP overhead).
- **Method**: JTok-M = Mixture of Joint-Tokens. Scaling experiments span 650M–61B params.
- **Results**: predictable power-law scaling for token-indexed params; **+4.1 MMLU, +8.3 ARC, +8.9 CEval**; **35% less compute** vs vanilla MoE for matched quality.
- **Connection**: most explicit statement of "embedding-as-scaling-axis" thesis. Modulation flavor distinguishes from Engram (additive memory) and STEM (replacement).

### LongCat-Flash-Lite (2026, Meituan) — n-gram embedding analyzed in `arXiv:2601.21204` ("Scaling Embeddings Outperforms Scaling Experts")
- **Core**: production-deployed **68.5B MoE with 46% of params in N-gram embedding layer**, ~3B activated per token, 256K ctx via YARN.
- **Method**: hash current token + N-1 predecessors → single embedding vector; fuse with base embedding. Specialized **N-gram Cache** kernel.
- **Connection**: real-world realization of [[engram]] / [[scone]] thesis at scale.

### Gemma 4n architecture analysis (2026) — Twitter thread (not retrievable, paywalled X)
- **Status**: source inaccessible without auth. From group context it likely discusses Gemma 4n's **per-layer embeddings / sparse memory** design, fitting this cluster.

### Lossless Prompt Compression via Dictionary-Encoding and In-Context Learning (2026) — `arXiv:2604.13066`
- **Core**: replace **frequent subsequences** with compact **meta-tokens**; LLM learns the dictionary **in-context** (no finetune).
- **Method**: multi-scale repetitive-pattern detection + token-savings optimization (dict overhead < savings).
- **Connection**: tangential — uses *symbolic* dictionary compression rather than parameter/embedding tricks. Fits here as a baseline contrast to all the learned-embedding approaches.

**Group synthesis**:
- **Unifying thesis**: token-indexed lookups are an **orthogonal scaling axis** to attention/FFN params and MoE experts — capacity decoupled from FLOPs.
- **Spectrum**: additive memory ([[engram]], [[scone]]) → modulation ([[jtok]]) → replacement ([[stem]], [[memoryllm]]).
- **Convergent evidence**: 4 independent groups (DeepSeek, Apple, Meituan, JTok authors) hit the same idea in late-2025/early-2026 → this is the next big lever after MoE.
- **Practical pattern**: lookup tables sit off-accelerator (CPU/disk), prefetched per-token; ~30–46% of total params in the embedding layer.

---

## 4. Cartridges & Test-Time Context Distillation

> Theme: instead of stuffing context into the prompt at inference, **bake it into a small artifact** (KV cache, prefix, memory tokens, or model weights) ahead of time via training-style optimization.

### Cartridges: Lightweight long-context representations via self-study (2025, Stanford) — `arXiv:2506.06266`
- **Core**: train a **small KV cache offline per corpus** ("Cartridge"); load + decode at inference.
- **Method**: naive next-token-prediction on corpus fails; **self-study** = generate synthetic conversations about the corpus, train Cartridge with **context-distillation** loss against the full-context teacher.
- **Connection**: this is the spiritual parent for [[gradmem]], [[memo]], [[learning-by-distilling-context]].

### Idea for Cartridges + RAG (2026) — Twitter (inaccessible)
- **Status**: source paywalled. Likely proposes per-document Cartridges as a swap for vector retrieval — retrieve a *KV cache* instead of text chunks.

### Learning Facts at Scale with Active Reading (2025, Meta) — `arXiv:2508.09494`
- **Core**: **Active Reading** — train a model to "study" material with self-generated learning strategies (paraphrase, quiz, etc.), then fine-tune on the generated study material.
- **Results**: 8B model **+313% relative on Wikipedia SimpleQA**, +160% on FinanceBench. Released **Meta WikiExpert-8B** trained on 1T generated tokens — outcompetes 100B+ models on factual QA.
- **Connection**: same "bake context into weights" school as [[cartridges]] / [[learning-by-distilling-context]] but at pretraining scale.

### Learning by Distilling Context (2022) — `arXiv:2209.15189`
- **Core**: **context distillation** — model conditioned on `[instructions] + [task]` predicts `[scratchpad] + [answer]`; then fine-tunes to produce `[answer]` from `[task]` alone.
- **Results**: internalizes 8-digit addition reasoning; **+9% over GD baseline** on SPIDER Text-to-SQL.
- **Connection**: foundational paper for the whole Cartridges school. [[cartridges]] uses essentially this loss.

### Context Tuning for In-Context Optimization (2025, NYU) — `arXiv:2507.04221`
- **Core**: initialize a **trainable prompt/prefix with task-specific demonstrations**, then optimize — leveraging ICL ability vs starting from random/instruction tokens.
- **Results**: outperforms prompt tuning baselines; competitive with Test-Time Training at lower train cost. ICML 2025 TTA workshop.
- **Connection**: orthogonal to Cartridges (prompt prefix vs KV cache), but same "amortize at test time" spirit.

### Fast KV Compaction via Attention Matching (2026, MIT) — `arXiv:2602.16284`
- **Core**: construct **compact (K,V) tensors that reproduce the original attention output** (per-head, mass-preserving).
- **Method**: decomposes into subproblems, some with closed-form solutions.
- **Results**: up to **50× compaction in seconds** with little quality loss; pushes Pareto frontier of compaction time × quality.
- **Connection**: lightweight, training-free analog of [[cartridges]] (which trains a KV); fast enough for online use vs offline.

### GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent (2026) — `arXiv:2603.13875`
- **Core**: write context into a small set of **prefix memory tokens** via per-sample **test-time gradient descent**, model weights frozen.
- **Method**: optimizes a **self-supervised context reconstruction loss** — loss-driven write with iterative error correction (vs forward-only writers like RMT).
- **Results**: beats forward-only writers on associative key-value retrieval; competitive on bAbI, SQuAD.
- **Connection**: similar to [[cartridges]] but **per-query** (test-time) rather than offline corpus-level. Authors include Burtsev/Kuratov (RMT lineage).

### MeMo: Memory as a Model (2026) — `arXiv:2605.15156`
- **Core**: **separate memory model** stores new knowledge; main LLM weights untouched.
- **Method**: plug-and-play; works with closed-source LLMs (no weights/logits needed); retrieval cost independent of corpus size at inference.
- **Results**: strong on BrowseComp-Plus, NarrativeQA, MuSiQue; robust to retrieval noise; no catastrophic forgetting.
- **Connection**: most modular variant of "compress corpus into params" — externalizes the memory entirely.

### Towards Infinite Context Windows: Neural KV Cache Compaction (2026) — Twitter blog (inaccessible)
- **Status**: source paywalled. Title strongly suggests it's a sibling of [[fast-kv-compaction]] and the KV-compression work.

**Group synthesis**:
- **Spectrum of artifacts**: prompt prefix ([[context-tuning]]) → memory tokens ([[gradmem]]) → KV cache ([[cartridges]], [[fast-kv-compaction]]) → external memory model ([[memo]]) → weights ([[active-reading]], [[learning-by-distilling-context]]).
- **Two regimes**: *offline* (Cartridges, Active Reading, MeMo) amortizes over many queries; *online/test-time* (GradMem, Context Tuning, Fast KV Compaction) is per-query.
- **Common loss**: context distillation (KL/cross-entropy against full-context teacher) or self-supervised reconstruction — these dominate over naive next-token prediction on the corpus.

---

## 5. Gist Tokens & Learned Compression Soft-Prompts

> Theme: train the LM to summarize a prompt into a few **learned tokens** that act as soft prefix.

### Learning to Compress Prompts with Gist Tokens (2023) — `arXiv:2304.08467`
- **Core**: modify attention mask so the LM **must** route prompt info through K gist tokens; cache + reuse them.
- **Results**: **up to 26× prompt compression**, 40% FLOPs reduction, 4.2% wallclock speedup at minimal quality loss on LLaMA-7B / FLAN-T5-XXL.
- **Connection**: foundational. [[icae]], [[autocompressor]], [[meta-tokens]] all descend from this.

### Language Modeling with Learned Meta-Tokens (2025, ICML) — `arXiv:2509.16278`
- **Core**: inject **meta-tokens during pre-training** + dedicated meta-attention. Tokens act as **trainable content-based landmarks** that implicitly cache preceding context.
- **Results**: **2× length generalization** at inference time (with YaRN); data-efficient pretraining (<100B tokens) hits strong synthetic-task performance.
- **Connection**: pretraining-time version of [[gist-tokens]]; landmarks idea echoes [[rmt]] memory tokens.

### Neologism Learning for Controllability and Self-Verbalization (2025) — `arXiv:2510.08506`
- **Core**: add a **new word embedding** and train on concept examples — no other params change. Resulting word encodes the concept; LM can **verbalize what the new word means**.
- **Finding**: discovered **"machine-only synonyms"** — words that look unrelated to humans but trigger similar model behavior.
- **Connection**: not compression per se, but a controllability cousin of [[gist-tokens]] — single learned embedding carries a behavior.

**Group synthesis**:
- Gist Tokens defined the template (learn-a-token-as-summary); Meta-Tokens scaled it to pretraining; Neologism Learning generalizes the framing to *any* concept, not just prompts.
- All three live in the "single embedding can carry a lot of behavior" thesis — overlaps philosophically with [[engram]] / [[jtok]] (embeddings as scaling axis).

---

## 6. Adapting LMs to Compress Long Contexts (Recurrent / Hybrid)

### Recurrent Memory Transformer (RMT, 2022) — `arXiv:2207.06881`, NeurIPS
- **Core**: add **[mem] tokens** to segments; pass them between segments to carry global info recurrently.
- **Connection**: ancestor of [[autocompressor]], [[meta-tokens]], [[gradmem]] (Burtsev/Kuratov lineage).

### Adapting Language Models to Compress Contexts — AutoCompressor (2023, Princeton) — `arXiv:2305.14788`
- **Core**: compress long docs into **summary vectors** = soft prompts, accumulated across segments.
- **Method**: builds on RMT with **summary accumulation** (concatenate all-segment summary vectors).
- **Results**: trained on OPT/Llama-2 up to 30,720 tokens; summary vectors are good substitutes for plaintext ICL demos.

### Artificial Hippocampus Networks (AHN, 2025, ByteDance) — `arXiv:2510.07318`
- **Core**: **Multi-Store Model** from cognitive science → sliding-window KV (lossless short-term) + **AHN module** (RNN-like, fixed-size long-term).
- **Method**: AHN instantiated as Mamba2 / DeltaNet / GatedDeltaNet; recurrently compresses out-of-window KV. Self-distillation training freezes base LLM.
- **Results**: beats sliding-window baselines on LV-Eval and InfiniteBench; matches/exceeds full attention.
- **Connection**: modern, hybrid-architecture version of RMT/AutoCompressor.

### Attention and Compression is all you need — CAT (2025, NYU) — `arXiv:2511.05313`
- **Core**: **Compress & Attend Transformer** — decode chunks by attending to **compressed chunks** of the sequence.
- **Method**: train with multiple chunk sizes simultaneously → single model with **test-time quality/compute knob**.
- **Connection**: combines hierarchical compression of [[autocompressor]] with a controllable knob like [[lensvlm]].

**Group synthesis**:
- RMT is the great-grandparent; the 2025 wave (AHN, CAT) marries recurrent compression with sliding-window attention to get the best of both worlds.
- AHN and CAT independently arrive at "sliding window + compressed long-term store" — strong convergence.

---

## 7. In-Context Autoencoders & Continuous Compression

### In-Context Autoencoder (ICAE, ICLR 2024) — `arXiv:2307.06945`
- **Core**: LLM-as-encoder (with LoRA) compresses long context into **memory slots**; frozen LLM-as-decoder consumes them.
- **Method**: pretrain via autoencoding + LM objectives; instruction fine-tune.
- **Results**: **4× compression** on Llama, ~1% extra params; latency + memory wins.
- **Connection**: direct competitor to [[gist-tokens]] and [[autocompressor]]; the "autoencoding" framing makes context compression an explicit AE problem.

### Continuous Autoregressive Language Models (CALM, 2025) — `arXiv:2510.27688`
- **Core**: instead of next-token, predict **next continuous vector** representing K tokens (compressed by a high-fidelity AE with 99.9% reconstruction).
- **Method**: requires likelihood-free toolkit — **Energy Transformer** head, **BrierLM** metric (Brier-score-based), principled black-box temperature sampling.
- **Results**: matches discrete baseline with **44% less train FLOPs, 34% less inference FLOPs**.
- **Connection**: pushes ICAE's autoencoder idea into the **generation loop itself** — model lives in continuous chunk space.

### CompLLM: Compression for Long Context Q&A (2025) — `arXiv:2509.19228`
- **Core**: compress **per-segment independently** (not holistically) → linear cost, scalability (1k-trained model handles 100k ctx), **reusable cached segments**.
- **Results**: 2× compression rate; **up to 4× TTFT speedup**, 50% KV cache savings on long ctx.
- **Connection**: practical-deployment-flavored ICAE — sacrifices a bit of compression for segment-wise reusability (matches RAG-style chunking).

**Group synthesis**:
- ICAE = encoder/decoder same LLM, holistic. CompLLM = per-segment for reusability. CALM = generation-loop-level continuous compression.
- Trade-off across the three: compression ratio (ICAE) vs reusability (CompLLM) vs end-to-end speedup (CALM).

---

## 8. Cross-LLM Communication

### Cache-to-Cache: Direct Semantic Communication Between LLMs (2025, ICLR'26) — `arXiv:2510.03215`
- **Core**: two LLMs share KV-cache directly via a learned projector + fuser; no text in between.
- **Method**: learnable **gating** picks which target layers benefit from incoming cache.
- **Results**: **+8.5–10.5% accuracy** vs single model; **+3–5% over text-comm baseline**; **2× speedup**.
- **Connection**: the cross-model analog of the multi-agent KV/latent sharing work in §1 ([[polykv]], [[latentmas]]) — but for *heterogeneous* models, not multi-agent loops of one model.

---

## Cross-Cutting Themes

> **Three big ideas dominate this corpus.**

### A. Compression as the Universal Solvent
- Every group above is some variant of: *replace dense text with denser representation*.
- The "denser representation" varies: **vision tokens** (§2), **n-gram embeddings** (§3), **memory tokens / prefixes** (§4–§6), **continuous vectors** (§7), **KV caches** (§1, §8).
- Most papers cluster at **2–10× compression**; optical pushes to 10–20×, gist tokens to 26×, continuous vectors to K-token chunks.

### B. The "New Scaling Axis" Convergence
- §3 papers (Engram, JTok, STEM, SCONE, MemoryLLM, LongCat) **independently** arrive at the same conclusion in late-2025/early-2026: **embedding/lookup capacity** is an orthogonal axis to attention/FFN compute and MoE experts.
- This is the most surprising convergence in the list — DeepSeek, Apple, Meituan, and academic labs all land on token-indexed lookups as the next lever.
- Architectural variants: additive (Engram), modulatory (JTok), replacement (STEM, MemoryLLM), production-scale (LongCat).

### C. Latent > Text for Inter-Module Communication
- Whether between agents (§1, [[latentmas]], [[recursive-mas]]), between calls of the same agent (Latent Briefing), or between different LLMs ([[cache-to-cache]]), the field is rapidly moving to **drop text** as the IPC channel.
- All such papers report **2–4× latency reduction** and **+5–15% accuracy** vs text-mediated baselines.
- Two engineering challenges they solve: (a) representation alignment across heterogeneous models (RecursiveMAS, C2C); (b) sharing+compressing the latent state efficiently (PolyKV, Latent Briefing).

### Key Lineages
- **RMT (2022) → AutoCompressor (2023) → AHN (2025)** — segment-level recurrence with memory tokens.
- **Gist Tokens (2023) → ICAE (2024) → Meta-Tokens (2025) → CALM (2025)** — learned compression artifacts, increasingly deep in the model.
- **Learning by Distilling Context (2022) → Cartridges (2025) → Active Reading (2025) → GradMem (2026)** — context distillation school.
- **DeepSeek-OCR (Oct 2025) → OCR-Memory (2026), LensVLM (2026)** — optical compression for text.
- **SCONE (Feb 2025) → Engram, JTok, STEM, MemoryLLM (Jan/Feb 2026)** — embedding-as-axis convergence.

### Surprising Differences
- **Per-query vs amortized**: GradMem, Context Tuning, Fast KV Compaction are **per-query** (test-time). Cartridges, AutoCompressor, ICAE, Active Reading are **amortized** (offline). Same loss function, different deployment regime.
- **Symmetric vs asymmetric KV compression**: PolyKV is the only paper to compress K and V asymmetrically (K-int8, V-3bit). Most KV-cache work compresses uniformly or sparsifies positions.
- **Optical vs continuous compression**: both achieve ~10× compression of "stuff the LM has to read," but optical (DeepSeek-OCR) keeps text discrete while CALM goes fully continuous. Choice depends on whether you want interpretability + retrievability (optical) or end-to-end differentiability (continuous).

### Inaccessible Sources
The following list entries were paywalled X/Twitter posts and could not be fetched directly:
- *Idea for using Cartridges for RAG* (gstsdn/status/2022704205858046095)
- *Towards infinite context windows: neural KV cache compaction* (oneill_c/status/2039389427462770815)
- *Gemma 4n architecture analysis* (norpadon/status/2039740827975500251)

They are flagged inline above and would need either a logged-in fetch or a manual paste to be summarized.
