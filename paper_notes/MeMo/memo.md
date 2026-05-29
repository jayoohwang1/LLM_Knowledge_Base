# MeMo: Memory as a Model

> **arXiv** 2605.15156v2 (May 2026) · NUS / A*STAR / MIT CSAIL / SMART / Tokyo / Liquid AI · NeurIPS submission
> **TL;DR** Encode new knowledge into a small **dedicated MEMORY model** (SFT'd on a synthesized "reflection" QA dataset), then have a **frozen EXECUTIVE model** (any LLM, incl. closed APIs) query it via a structured multi-turn protocol. Memory is decoupled from reasoning → plug-and-play, no catastrophic forgetting, retrieval cost independent of corpus size.

---

## 1. Motivation / Problem

- **Core problem**: LLMs frozen after pretraining → knowledge goes stale; real apps need timely/domain-specific info without full retraining.
- **Knowledge Integration Problem (Def. 1)**: given frozen $\mathcal{M}_\theta$ + target corpus $\mathcal{D}$, find mechanism $(\Phi, f)$ maximizing $\mathbb{E}_{q\sim\mathcal{Q}}[\,\mathbb{P}\{f(\mathcal{M}_\theta,\Phi(\mathcal{D}),q)=a^\star(q)\}\,]$ **without modifying $\theta$**.
  - $\Phi$ maps corpus → representation $\mathcal{K}\doteq\Phi(\mathcal{D})$; $f$ combines $\mathcal{K}$ with $\mathcal{M}_\theta$ at inference.
  - $\mathcal{S}(q)\subseteq\mathcal{D}$ = supporting docs for query $q$ (theoretical construct for query complexity; $|\mathcal{S}(q)|>1$ ⟹ cross-doc).
  - Note: doc "effectively absent" from $\mathcal{M}_\theta$ if it can't answer questions grounded in it (never seen OR not retained).

### Three existing paradigms (and their flaws)
| # | Paradigm | $(\Phi,f)$ | Flaw |
|---|----------|-----------|------|
| ① | **Non-parametric** (RAG/ICL): BM25, dense, graph retrievers | $\mathcal{K}=\mathcal{D}$, $f$ retrieves subset $\hat{\mathcal{S}}$, prepends to prompt | limited context window; sensitive to retrieval noise; weak cross-doc synthesis |
| ② | **Parametric** (CPT / SFT on corpus) | $\mathcal{K}=\emptyset$, $f=\mathcal{M}_{\theta'}$ | catastrophic forgetting; expensive; memorizes train dist; infeasible for closed models |
| ③ | **Latent memory** (AutoCompressor, Gist tokens, ICAE, Memory Decoder) | soft tokens / model-specific reps | **representation coupling**: memory bound to producing model → not transferable across LLMs (needs shared tokenizer etc.) |

- **MeMo = hybrid** combining complementary strengths: uses off-the-shelf frontier model unchanged (like ①), internalizes knowledge in params (like ②), ships a compact queryable memory artifact (like ③) — while avoiding each one's failure mode.

### Desirable-properties table (Tab. 1) — MeMo hits all
> Frozen base LLM · **No** retrieval index · Black-box compatible · No catastrophic forgetting · **Constant-size** memory · Cross-LLM transferable
- Non-param: fails no-retrieval-index, constant-size. Param: fails black-box, no-forgetting, transferable. Latent: fails most. **MeMo: ✓ all.**

---

## 2. Architecture overview

Two models, two phases.

- **MEMORY model** $\mathcal{M}_\varphi$ — small LLM ($\varphi\ll\theta$, e.g. 1.5B/14B). SFT'd to internalize $\mathcal{D}$ in its weights. Queried, never sees source docs at inference.
- **EXECUTIVE model** $\mathcal{M}_\theta$ — frozen, any LLM (open or closed API, e.g. Qwen2.5-32B, Gemini-3-Flash). Does reasoning, decomposes user queries, treats MEMORY as a black-box oracle.

**Design principle — "reflections"**: corpus-derived structures that *require no knowledge of future queries* yet form the precise interface through which any query can access the underlying corpus without ever observing it directly.

**Two key challenges → two contributions**:
1. ① Training MEMORY to answer diverse/unseen/cross-doc queries → **5-step data synthesis pipeline** (Sec 4.1) producing a reflection QA dataset.
2. ② Querying MEMORY for multi-step/compositional queries → **structured 3-stage multi-turn protocol** (Sec 4.4).

---

## 3. Data Synthesis Pipeline (training-side, Alg. 1)

- Driven by a **GENERATOR model** $\mathcal{M}_{\text{gen}}$ (an LLM, may be = or smaller than EXECUTIVE; here Qwen2.5-32B-Instruct).
- Goal: distill corpus $\mathcal{D}$ into reflection QA dataset $\mathcal{Q}_{\text{final}}$ capturing single-doc facts **and** cross-doc relationships.
- **No doc identifiers/watermarks** embedded at any step → prevents MEMORY exploiting shortcut signals at eval.

**Per-document loop** (each $d$ chunked into $C$):
- **Step 1 — Fact extraction** (per chunk $c$): two *parallel* extractions:
  - **direct** → $\mathcal{Q}_{\text{dir}}$ (explicitly stated facts).
  - **indirect** → $\mathcal{Q}_{\text{indir}}$ (inferred / synthesized info beyond surface). Ensures both recall + inference in training signal.
  - merge: $\mathcal{Q}_{\text{raw}}=\mathcal{Q}_{\text{dir}}\cup\mathcal{Q}_{\text{indir}}$.
- **Step 2 — Consolidation**: identify QA pairs sharing context (entity / time / relationship type), combine into multi-fact pairs $\mathcal{Q}_{\text{mrg}}$. Goes beyond single-fact QA. $\mathcal{Q}_{\text{con}}=\mathcal{Q}_{\text{dir}}\cup\mathcal{Q}_{\text{indir}}\cup\mathcal{Q}_{\text{mrg}}$.
- **Step 3 — Verification & rewriting**: check each pair for **self-containment** (answerable in isolation). Failure modes: unresolved pronouns ("What did *they* propose?"), implicit refs ("As in the above table…"). Rewrite using source chunk as context; discard if still ambiguous → verified set $\mathcal{Q}_{\text{ver}}^d$.
- **Step 4 — Entity surfacing**: for each named entity in $\mathcal{Q}_{\text{ver}}$, generate pairs where *question encodes entity attributes/relationships, answer reveals identity*. Facts aggregated across all pairs in chunk first. Varying complexity (single→multi-fact). → $\mathcal{Q}_{\text{ent}}^d$.
  - **Why**: mitigates the **reversal curse** (trained "A is B" ⟹ can't infer "B is A"); supports the *entity-identification* inference turn.
  - $\mathcal{Q}_{\text{final}}\mathrel{+}=\mathcal{Q}_{\text{ver}}^d\cup\mathcal{Q}_{\text{ent}}^d$.

**Cross-document loop** (over topically-related doc groups $G_i$):
- **Step 5 — Cross-document synthesis**: $\mathcal{M}_{\text{gen}}$ given verified + entity pairs of all docs in group, finds two connection types:
  - **Converging clues**: multiple docs give complementary facts about same entity → jointly identify it.
  - **Parallel properties**: different entities share attribute/role → comparative/analogical reasoning.
  - → $\mathcal{Q}_{\text{cross}}$ (these have $|\mathcal{S}(q)|>1$, the cross-doc objective). $\mathcal{Q}_{\text{final}}\mathrel{+}=\mathcal{Q}_{\text{cross}}$.
- Groups arise naturally (long doc split into one group) or from human labels.

### Training the MEMORY model (Sec 4.2)
- **Full SFT**, next-token loss **on answer tokens only**, conditioned on question + preceding answer tokens, **never on source docs**:
$$\mathcal{L}(\varphi)=-\!\!\sum_{(q_i,a_i)\in\mathcal{Q}_{\text{final}}}\sum_{t=1}^{|a_i|}\log\mathcal{M}_\varphi\big(a_i^{(t)}\mid q_i, a_i^{(1:t-1)}\big).$$
  - Conditioning only on Q (not docs) **forces parametric internalization** vs RAG-style copying from retrieved context — key distinction.
- Init from small pretrained LM (1.5B / 14B), much smaller than EXECUTIVE.
- Full SFT chosen over CPT (CPT degrades instruction-following) and LoRA (App O comparison).

---

## 4. Inference: Structured Multi-Turn Protocol (Sec 4.4)

EXECUTIVE queries MEMORY as **external knowledge oracle** over 3 sequential stages; each stage = distinct prompt, sampling temp, independent interaction **budget**.

- **Stage 1 — Grounding**: EXECUTIVE decomposes query $q$ into atomic clue-probing sub-questions $\{q_1',\dots,q_J'\}$, each targeting a single identifying constraint ($J$ adaptive). MEMORY answers each *independently, no shared context* → grounding responses $\{m_1,\dots,m_J\}$. Provides contextual grounding for later stages.
- **Stage 2 — Entity identification**: using grounding as context, EXECUTIVE iteratively narrows candidate entities via targeted follow-ups across multiple turns until it converges on single entity $e^\star$ or budget exhausted. If none found → skip Stage 3, synthesize from grounding alone. Leverages $\mathcal{Q}_{\text{ent}}$ training.
- **Stage 3 — Answer seeking & synthesis**: conditioned on $e^\star$, EXECUTIVE queries MEMORY for supporting facts, then synthesizes:
$$\hat{a}=\mathcal{M}_\theta\big(q,\{m_j\}_{j=1}^J, e^\star, m_{\text{seek}}\big).$$

- **Why constant-time**: MEMORY responses $m_j, m_{\text{seek}}$ are compact NL snippets, length **independent of corpus size** → fixed cost per turn.
- All interactions via input/output interface only ⟹ **black-box compatible** (works with proprietary APIs, no weights/gradients/logits).

---

## 5. Continual Integration via Model Merging (Sec 4.3 / 5.5)

- **Streaming setting**: corpora $\{\mathcal{D}_1,\dots,\mathcal{D}_K\}$ (pairwise disjoint) arrive over time. Full retrain on union scales with cumulative corpus size → prohibitive.
- **Approach**: train separate MEMORY $\mathcal{M}_{\varphi_i}$ per corpus from same base $\mathcal{M}_{\varphi_0}$, define **task vectors** $\tau_i=\varphi_i-\varphi_0$, merge:
$$\varphi_{\text{merged}}=\mathrm{Merge}(\varphi_0,\{\tau_i\}_{i=1}^K;\Theta).$$
- **Compute scaling**: per-corpus cost merge $\Theta(K)$ vs full-retrain $\Theta(K^2)$ → at $K{=}10$, **5.5× saving** (240 vs 1320 GPU-h). At $K{=}2$: merge $X+Y\approx48$h vs retrain $X+(X+Y)\approx72$h = **33% cut**.
- **Merge methods swept** (App H): Linear, SLERP, Task-arithmetic, **TIES**, DARE, DARE-TIES. TIES = trim top-$\rho$ magnitude entries → elect sign by magnitude-weighted majority vote → disjoint-merge sign-agreeing entries.
- **Tradeoff**: best merge (TIES, $\rho{=}0.3$) trails full-retrain by **11.0%** (Qwen-32B exec) / **19.1%** (Gemini-3-Flash) on NarrativeQA, but still **beats every retrieval baseline**. Aggressive sparsification + sign-conflict resolution (TIES / DARE-Linear at $\rho{=}0.3$) = most reliable recipe.

---

## 6. Experiments

### Setup
- **Benchmarks**:
  - **BrowseComp-Plus**: deep-research, multi-hop multi-doc retrieval; 300 Qs, 1775 evidence + 1766 negative docs = **3541** docs.
  - **NarrativeQA**: discourse understanding over long books/scripts; 293 Qs over 10 docs (no negatives).
  - **MuSiQue**: 2–4 reasoning-step multi-hop over Wikipedia; 1000 Qs, **5296** docs (2648 evidence + 2648 neg).
- **Baselines**: BM25 (lexical), NV-Embed-V2 (dense), HippoRAG2 (graph RAG, SOTA), Cartridges (trained KV-cache, closest parametric, white-box only). **Perfect Retrieval** = empirical upper bound (exec gets gold docs in-context). Retrieval uses top-$k{=}9$ adaptive backoff.
- **Models**: GENERATOR = Qwen2.5-32B-Instruct (vLLM, YaRN RoPE → 131K ctx). MEMORY = Qwen2.5-14B-Instruct (3 epochs, fused AdamW, DeepSpeed, lr $2\times10^{-5}$). EXECUTIVE = Qwen2.5-32B-Instruct **or** Gemini-3-Flash (both have minimal prior knowledge of eval sets). Eval = binary accuracy judged by Gemini-2.5-Flash-Lite via DeepEval, mean±std over 3 runs (Qwen) / 1 run (Gemini-3-Flash).

### Main results (Tab. 2, Accuracy %)
| Method | BCP Qwen | BCP Gemini | NarrQA Qwen | NarrQA Gemini | MuSiQue Qwen | MuSiQue Gemini |
|---|---|---|---|---|---|---|
| Perfect Retrieval* | 79.67 | 88.33 | 51.42 | 60.41 | 62.83 | 73.00 |
| BM25 | 1.11 | 27.00 | 10.24 | 14.33 | 20.00 | 23.20 |
| NV-Embed-V2 | 50.67 | 57.00 | 20.59 | 26.62 | 37.47 | 46.60 |
| HippoRAG2 | **56.11** | 66.33 | 21.39 | 23.21 | 42.17 | 57.00 |
| Cartridges | 0.00 | — | 3.75 | — | 8.57 | — |
| **MeMo** | 54.22 | **66.67** | **26.85** | **53.58** | **48.30** | **60.20** |

- **Wins** NarrativeQA + MuSiQue across both EXECUTIVE models (often by large margins, esp. NarrativeQA Gemini 53.58 vs ~26). Competitive on BCP (narrowly trails HippoRAG2 with Qwen, leads with Gemini).
  - **NarrativeQA**: needs reasoning over long passages w/ complex connections — retrieval constrained by ctx window; MeMo captures connections via reflections.
  - **BCP weakness explained**: answers are *absent* from EXECUTIVE's parametric knowledge → direct access to evidence docs especially valuable → favors methods passing raw docs.
- **Plug-and-play**: swapping Qwen→Gemini-3-Flash EXECUTIVE gives +12.45 / +26.73 / +11.90 (BCP/NarrQA/MuSiQue) — trains once with weak GENERATOR, pairs with *any* stronger LLM, no extra training.

### Ablations
- **5.2 Retrieval-noise robustness** (Tab. 3, $0{\times}N$ vs $1{\times}N$ distractors):
  - NV-Embed-V2 & HippoRAG2 **drop up to 6.22%** (BCP) / 5.16% (MuSiQue) when noise added.
  - **MeMo stable**: +0.55% BCP, −1.77% MuSiQue (within 1 std). MEMORY gives more precise info to sub-queries than raw doc retrieval.
- **5.3 MEMORY model size** (Tab. 4, 1.5B vs 14B): consistent positive scaling, 14B > 1.5B everywhere. Gap is *task-dependent* — widens for NarrativeQA but shrinks for BCP/MuSiQue with stronger EXECUTIVE.
- **5.4 MEMORY model family** (Tab. 5, ~1–2B: Qwen2.5-1.5B, Gemma3-1B-IT, LFM2-1.2B): performance largely **robust to architecture/pretraining lineage** at similar scale → parametric compression generalizes across families.

---

## 7. Appendix highlights

- **App D — Datasets / chunking**:
  - NarrativeQA docs huge (median 65,925 tokens) → chunked w/ **6400-word sliding window, 640-word (10%) overlap** → 75 chunks, median group size 7. MuSiQue docs tiny (99.70% < 512 tok) → 1 chunk each. BCP docs also 1 chunk each.
  - **Subset of negatives only**: Step-5 cost $O(k\cdot C^2\cdot Q^2)$ scales quadratically in chunks-per-group; including all negatives (avg 78/Q BCP) blows up group size → capped at $N_{\text{evidence}}^{\text{dataset}}$ negatives.
- **App E — Pipeline-step LOO ablation** (Tab. 9, Qwen2.5-1.5B MEMORY):
  - **Step 5 (cross-doc) is by far most critical**: removal → 6.37% (NarrQA, from 24.00) / 24.17% (MuSiQue, from 42.90), with near-total data loss ($0.002\times$, $0.195\times$ retention). It's the dominant source of training pairs.
  - **Anomaly**: removing Step 2 / Step 3 *helps* NarrativeQA. Step 2 commonalities are superficial (event/scene co-occurrence, ubiquitous characters) — low-quality. Step 3 self-containment rewriting *corrupts* long-narrative pairs (pronouns/temporal refs span paragraphs = structural not fixable) → substitutes unrelated content. For MuSiQue both steps help (genuine multi-hop relationships, shallow localized violations). ⟹ Step 3 only beneficial where self-containment violations are well-defined/resolvable.
  - Steps 1a/1b/4 removal hurts both data volume + accuracy on both → each contributes.
- **App E.2 — Excluded steps**: paraphrasing (coverage already sufficient ~600k–1.6M pairs), more Step-1 sampling trials (unreliable), targeted fill (only lengthens context, exacerbates long-input attention degradation).
- **App F/G — HParams/compute**: SFT lr $2\times10^{-5}$, 3 epochs, max seq 8096, BF16, FlashAttention2, constant-w-warmup (0.05). Data gen: 240/200/150 GPU-h (BCP/NarrQA/MuSiQue). Training (14B, single run): 180/150/90 GPU-h. All on H100/H200.
- **App H — merging**: Linear / SLERP (unit-sphere interp, preserves magnitude) / Task-arithmetic / TIES / DARE (random drop $1{-}\rho$, rescale $1/\rho$) / DARE-TIES.
- **# QA pairs generated** (Tab. 11): BCP 1.64M, NarrativeQA 1.28M, MuSiQue 0.66M.

---

## 8. Takeaways / connections

- **Decoupling memory from reasoning** is the central trick — lets you (a) reuse frontier closed models unchanged, (b) sidestep catastrophic forgetting entirely (EXECUTIVE never updated), (c) make retrieval cost corpus-size-independent.
- **Reflections** ≈ query-agnostic "interview notes" of the corpus; reframes knowledge injection as *SFT-on-synthetic-QA* rather than retrieval-index building or CPT. Echoes synthetic-QA-from-corpus lines (Alberti'19, Puri'20) but adds cross-doc synthesis + entity surfacing as first-class.
- **Entity surfacing vs reversal curse**: explicit countermeasure baked into data gen — relevant to anyone training models to recall facts bidirectionally (Berglund'23, Allen-Zhu Physics-of-LMs 3.2).
- **vs Cartridges / latent memory**: MeMo's memory is a *full small model* queried in NL, so it transfers across EXECUTIVE families (no shared tokenizer/representation needed) — the key edge over Gist/ICAE/Memory-Decoder.
- **vs RAG**: trades a fixed upfront training cost for noise-robustness + cross-doc reasoning + constant inference cost; but loses when answers are entirely outside EXECUTIVE's knowledge and raw-doc access dominates (BCP).
- **Model merging** = the continual-learning story: $\Theta(K)$ vs $\Theta(K^2)$ scaling, accepting a measurable accuracy hit that still beats retrieval.

### Limitations (App B/C)
- Upfront per-corpus training cost; performance bounded by MEMORY model's **representational capacity** (large/dense corpora may exceed what a fixed-size MEMORY can compress — not yet hit in experiments).
- Pipeline expensive (Step 5 quadratic). Future: cheaper memory construction, RL post-training, dynamic corpora, tighter exec↔memory coordination, per-architecture LoRA tuning, optimal interaction budgets.
