# Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals

> Xie, Wang, Rosu, Deng, Sun, Lin, Dhingra — Duke / Rice / JHU / UC Davis — **NeurIPS 2025** — arXiv 2502.01025
> Code: https://github.com/ruoyuxie/when-to-stop

---

## TL;DR

- **Core idea — *dynamic context cutoff***: let an LLM **self-terminate context reading** once it has processed *enough* info to answer, instead of consuming the whole input. Process chunks left→right, stop at the first prefix that's "sufficient".
- **Key empirical discovery**: specific **attention heads inherently encode "sufficiency signals"** — a lightweight probe/classifier on those heads' activations predicts when critical info has been processed.
- **Emergent scaling phenomenon**: small models (1B–8B) *need* the probe; large models (14B+) can self-assess sufficiency via simple **prompting** ("do you have enough info? YES/NO").
- **Results**: ~**+3.4% accuracy** while using **1.33× fewer tokens** on average, beating compression (LLMLingua family) and RAG baselines at matched token-reduction rates.
- Efficiency is **content-adaptive** (dynamic), not a one-size-fits-all fixed compression rate.

---

## 1. Motivation

- LLMs process **every token with equal priority** regardless of relevance → wasted compute + the **lost-in-the-middle** problem (critical info diluted in long inputs).
- Human cognition analogy: **stop reading once enough evidence is gathered**; ignore redundant tail.
- For localized-answer tasks (factoid QA, retrieval, fact verification) the answer span often appears early; reading the rest is pure overhead.
- **Question posed**: *"Can we enable LLMs to self-assess context sufficiency and terminate early without compromising accuracy?"*

### Contrast with existing efficient-context methods

| Family | Mechanism | Nature |
|---|---|---|
| **LLMLingua / LongLLMLingua / LLMLingua2** | drop low-info tokens via a small LM / distillation | **static** — predefined target compression rate |
| **RAG (BM25, SBERT)** | retrieve fixed top-$k$ docs | **static** — fixed $k$ |
| **This work (dynamic cutoff)** | stop processing when *model's own* internals signal sufficiency | **dynamic + content-adaptive** |

- Distinction: prior compression imposes a *one-size-fits-all* ratio; here **efficiency emerges from the model's own understanding** of the input. Different inputs → different cutoffs.
- Orthogonal/complementary to KV-cache eviction (H2O), quantization, speculative decoding, flash-attention — those are *computational approximations*; this is *context reduction*. Operates at **textual level, no retraining** (like LLMLingua).
- Related lineage: **latent knowledge in activations** — LLMs encode task-relevant knowledge in intermediate states, often more accurate than surface outputs (used for hallucination detection, reasoning-correctness verification, knowledge-graph construction). This work extends that to detect *context sufficiency*.

---

## 2. Methodology

### 2.1 Problem formulation

- Pretrained LM $\mathcal{M}$, input context $\mathbf{C}$ partitioned left→right into ordered, **non-overlapping** chunks $\{\mathfrak{s}_j\}_{j=1}^m$:
$$\big\|_{j=1}^{m}\mathfrak{s}_j = \mathbf{C}, \quad \mathfrak{s}_i \cap \mathfrak{s}_j = \emptyset \ \text{for}\ i\neq j$$
- **Cumulative contexts** (nested prefixes): $\mathbf{C}_i = \mathfrak{s}_1 \,\|\, \mathfrak{s}_2 \,\|\dots\|\, \mathfrak{s}_i$, giving $\mathbf{C}_1 \subset \mathbf{C}_2 \subset \dots \subset \mathbf{C}_m = \mathbf{C}$.
- **Goal**: find smallest prefix $\mathbf{C}_k$ such that $\mathcal{M}(q,\mathbf{C}_k) \approx \mathcal{M}(q,\mathbf{C})$ — the *minimal sufficient context*.
- **Sufficiency classifier** $\mathcal{S}$ with confidence $\mathcal{S}_c:\mathbb{R}^d\to[0,1]$ and threshold $\tau$:
$$\mathcal{S}(\mathbf{C}_i)=\begin{cases}1 & \text{if } \mathcal{S}_c(\mathbf{C}_i)\geq\tau\\ 0 & \text{otherwise}\end{cases}$$
- First $i$ with $\mathcal{S}(\mathbf{C}_i)=1$ → stop; remaining chunks ignored.
- **Why left-to-right (not arbitrary subset selection)?**
  - Matches how transformers are trained (L→R) — no architectural change/retraining.
  - **KV-cache reuse**: each new chunk extends the prefix, so cached activations of previous chunks are reused. Overlapping/arbitrary chunks would force recomputation, negating the efficiency gain.
  - Each token receives *exactly* the same computation as in a single full pass — chunking only decides *when to stop*, it doesn't alter attention.

### 2.2 Probing for "context sufficiency"

- Probe data: cumulative contexts $\{\mathbf{C}_i\}$, each labeled **sufficient ($y=1$)** or **insufficient ($y=0$)**.
- For each $\mathbf{C}_i$, model produces per-head activations $\{x_l^h\in\mathbb{R}^D\}$ ($h$-th head, $l$-th layer).
- Train **lightweight binary linear probe** per head:
$$p_\theta(x_l^h)=\sigma(\langle\theta, x_l^h\rangle), \quad \theta\in\mathbb{R}^D$$
- 4:1 train/val split per task. Each head scored by its probe's **validation F1**.
- **Finding (Fig 3, LLaMA3.2-1B; Fig 8, Qwen2.5-14B)**: a subset of heads — **primarily middle layers** — has significantly higher predictive F1. Consistent with interpretability findings that middle layers capture higher-level semantics. *Which* heads are informative varies by architecture (LLaMA dispersed vs Qwen concentrated).
- Head selection done **once** across all tasks; pick **top-$k$=5** heads by F1.

### 2.3 Dynamic context cutoff

**Ensemble classifier.** On the top heads, train multiple lightweight base classifiers $\{\mathcal{S}_1,\dots,\mathcal{S}_e\}$; build via **StratifiedKFold ($n=5$)**, select best by mean CV AUC, weighted soft-vote:
$$\mathcal{S}_{\text{ensemble}}(\mathbf{C}_i)=\frac{1}{e}\sum_{j=1}^{e}\mathcal{S}_j(\mathbf{C}_i)$$

**Inference with iterative forward passes.**
- Process nested prefixes $\{\mathbf{C}_i\}$ incrementally; at each step extract activations, run $\mathcal{S}_{\text{ensemble}}$.
- Cache activations to avoid recompute. $\mathbf{A}_{\text{cache}}^i$ = cached activations for $\mathbf{C}_i$; current chunk computed conditioned on cache:
$$\mathbf{A}(\mathbf{C}_i)=f_{\text{model}}(\mathbf{C}_i\setminus\mathbf{C}_{i-1},\,\mathbf{A}_{\text{cache}}^{i-1})$$
- Stop at sufficient $\mathbf{C}_k$ (or process all if never sufficient). Final answer reuses cache:
$$\mathcal{M}(\mathbf{C}_k)=\mathcal{M}(C_k\setminus C_{k-1},\,\mathbf{A}_{\text{cache}}^{k-1})$$

**Alternative — self-prompting (no classifier).**
- Larger LLMs: append a **meta-prompt** to each $\mathbf{C}_i$ asking "is this enough to answer $q$?" → model emits `[[YES]]`/`[[NO]]`. Binary response = sufficiency decision.
- Becomes increasingly reliable at **14B+** → sufficiency self-assessment is an **emergent capability of scale**.

---

## 3. Experiments

### Setup

- **Datasets — 6 QA, two reasoning types** (extended to ~40K tokens by concatenating documents; also short-form ~0.5–4K):
  - **Single-hop** (answer localized in one passage): **SQuAD**, **Natural Questions**, **Code Understanding** (GPT-4o-generated distractor code snippets + PCSD).
  - **Multi-hop** (integrate info across passages): **HotpotQA**, **MuSiQue**, **Multi-hop Key-Value Retrieval** (∞Bench synthetic).
  - **Balanced by design**: gold answer-span end position ~Uniform (mean ≈0.50, std ≈0.25–0.28) → ~50% of chunks "insufficient", ~50% "sufficient". Prevents early/late-stop bias.
  - 600 points/dataset, 80/10/10 train/val/test.
- **Ground-truth sufficiency cutoff** = normalized position of the *last* token in the gold span (minimal context for correct answer). This treats sufficiency as a *dataset property* (universal cutoff) — but §4.4 notes it's really *model-dependent*.
- **Models** (1B–70B, model-agnostic method): **LLaMA-3.2-1B**, **Mistral/Ministral-8B**, **Qwen-2.5-14B**, **LLaMA-3.3-70B**.
- **Baselines**: RAG → **BM25**, **SBERT**; compression → **LLMLingua**, **LongLLMLingua**, **LLMLingua2**; plus **Fine-Tuned Classifier (FT)** and **Self-Prompting**.

### Metrics

- **Sufficiency classification**: **F1** (balances over-/under-estimating sufficiency), **Recall@90% Precision (R@90P)** — fraction of truly-sufficient contexts caught while keeping false positives <10% (FP = premature cutoff = info loss, the dangerous error).
- **Task performance**: **Accuracy** (GPT-4o-mini as LLM judge, robust to paraphrase) before/after cutoff; **Token Reduction** = tokens-processed-with-method / full-context tokens.

### Hyperparameters

- 3 knobs: threshold $\tau$ (**key** — trades efficiency vs accuracy, swept), **#heads** ($k=5$), **#classifiers in ensemble** (top 4 of 8 by AUC).
- Default chunking = **percentage-based, 10% increments** (each chunk = 10% more of context).

### Results

**Sufficiency classification (Table 1, F1):** probing ("Ours") wins across all sizes.

| Model | FT | Self-Prompt | **Ours (probe)** |
|---|---|---|---|
| LLaMA3.2-1B | 79.5 | 52.6 | **88.3** |
| Mistral-8B | 79.5 | 69.7 | **89.8** |
| Qwen2.5-14B | 79.5 | 78.3 | **87.2** |
| LLaMA3.3-70B | 79.5 | 83.1 | **91.1** |

- **Probe is reliable at *all* scales**; **self-prompting jumps with scale** (52.6 → 83.1) → emergent self-assessment. FT flat/weak (misaligned signals across models).

**Efficiency vs performance (Fig 4, Tables 2–3):**
- Dynamic methods (Ours/FT/Self-Prompt) adaptively pick cutoffs; static methods (RAG/Lingua) fixed at 1.25× for fair comparison.
- 1B: matches RAG/LLMLingua. 8B: ~1.5× fewer tokens, minimal accuracy drop. **14B+: reduces tokens up to 1.22× AND improves accuracy.**
- **RAG degrades sharply with scale**; FT underperforms universally; smaller models bad at prompting (needs instruction-following).
- Avg token reduction: **Ours 1.33×** (short) / **1.27×** (long); FT 1.54–1.61×, Prompt 1.41–1.42× (but those lose accuracy).
- Short-form avg acc (Table 2): **Ours 49.4% single / 29.0% multi** — beats top baseline LLMLingua2 by +1.5% absolute.
- Truncating tail may also **mitigate lost-in-the-middle** (key info tends to be near the end of retained span).

---

## 4. Analysis & Ablations

- **Classification threshold $\tau$ (Fig 5)**: model's sufficiency confidence rises **monotonically** as more chunks processed → signals accumulate. Stop too early → info loss. R@90P (Fig 6) shows method reliably IDs sufficient contexts while minimizing false positives (R@95P, R@98P in appendix).
- **Chunking strategy (Table 4, Qwen-14B)**:
  - **Sentence-level**: highest F1 (96.8) but **impractical** — frequent sufficiency checks → high latency.
  - **Percentage 10%**: best **accuracy/efficiency balance** (chosen default). 1%/5%/20% comparable F1 but worse trade-offs (smaller chunks → more sequential checks → more latency).
- **Inference time (Fig 7)**: for short (1K tokens) full-context is faster; **beyond 2K tokens** method is faster when **<6 chunks (<60% of context)** processed. Scales better with length.
- **Wall-clock vs accuracy (Table 5)**: Ours 8.13s @ 36.5% acc (vs Full 9.02s @ 35.0%). LLMLingua2 fastest (6.97s) @ 35.8%. Balanced latency/quality without external heuristics.
- **Universal vs model-specific cutoffs (§4.4)**: a single "gold" location is dataset-defined, but sufficiency is really *model-dependent* (e.g. ICL: smaller models need more demos to reach confidence). **ICL case study (Appendix E, TREC)**: 8B needs ~all examples to peak confidence; 14B converges with fewer → argues for **adaptive, model-specific thresholds**.
- **#heads / #classifiers (Table 9)**: top-5 heads best (more heads = flat/worse → signals concentrated in specific heads, not distributed). Top-4-of-7 classifiers best.
- **Synthetic labels (Appendix A.4, Table 8)**: GPT-4o-predicted answer locations → labels nearly as good as human (F1 84.6–87.0 synthetic vs 88.3–89.8 original) → extends beyond datasets with explicit answer spans; for 14B+ self-prompting needs **no labels at all**.

---

## 5. Limitations & Future Work

- Best suited to **localized-answer tasks** (QA, retrieval, fact verification). **Holistic tasks** (summarization, passage rewriting) need full context → classifier won't trigger early stop (gracefully falls back to full-context, doesn't hurt).
- $\tau$ chosen via validation sweep → want **adaptive threshold selection** keyed to model/task.
- Applicability to creative writing / open-ended dialogue open.
- Complementary with KV-cache optimization (Appendix F.4) — combining = future work.

---

# Implementation

> Repo `when-to-stop/src/`. Entry: `run.py` → probe training+head selection → save → eval sufficiency → `early_stopping.py` (dynamic-cutoff generation) → `early_stopping_eval.py` (LLM-judge accuracy).
> Run: `python src/run.py --model_name meta-llama/Llama-3.2-1B-Instruct --data_dir "data/short" --output_dir "./results"`

## Data tuple structure

Each chunk datapoint is a 7-tuple (`dataset_manager.py`):
```
(query, is_sufficient, chunk_text, answers, task_name, chunk_features, gold_percentage)
```
- `is_sufficient` (label): `chunk_length >= gold_location.end_idx` — answer-containing chunk and all later chunks = sufficient (1); earlier = 0. Multi-hop: only after the *last* answer-containing chunk.
- `chunk_features`: dict `{head_key: feature_vector}` keyed `layer_{L}_head_{H}`.

## Activation extraction — `activation_manager.py` (`ActivationManager`)

- **Where activations come from**: forward hooks on the **attention output projection** (`self_attn.o_proj`) per layer. Model-type aware (`_detect_model_type`): LLaMA / `qwen2` / `phi`.
  - `_register_hooks()` walks `model.named_modules()`, registers `register_forward_hook` on attention modules, stores into `self.attention_activations[f"layer_{L}"]`.
  - LLaMA path hooks generic `attn` modules (excluding key/query/value/output submodules); Qwen2/Phi hook `o_proj` explicitly.
- **Reshape to per-head**: `_process_activation` reshapes `[B, seq, hidden]` → `[B, seq, num_heads, head_dim]` (view). Qwen2 (`_qwen2_process_activation`) handles **grouped-query attention** — `repeat_interleave` the KV heads by `head_ratio = num_heads // kv_heads` so each query head gets a vector.
  - `_validate_and_sanitize_activation`: nan/inf → safe values, clamp to fp32 range, cast fp32.
- **Per-head feature = mean over sequence positions**:
  - `get_head_features_for_eval(ctx)`: `layer_tensor.mean(dim=0)` → one `head_dim` vector per `layer_{L}_head_{H}`. This is the **feature fed to the probe at inference** (single cumulative context = one feature dict).
  - `get_chunk_features(ctx, boundaries)`: efficient **cumsum / arange** trick to get the running mean activation at *every* chunk boundary in one pass (token-level fast path), so all nested prefixes get features without re-running the model. Used during training-data construction.
- `compute_activations(query, context, gold_info)`: builds input via `prepare_sufficiency_input`, single `model(**inputs)` forward (no_grad), harvests hooked activations → `ContextActivations` dataclass (`full_context`, `model_token_ids`, `attention_activations`, `gold_location`).
- `find_gold_location`: char-level `.find()` of gold span in formatted input → token start/end indices + `percentage = char_end/len(context)`.

## Probe / head selection — `probe.py` (`SufficiencyProbe`)

- `train_and_select_heads(train, val, num_heads=5)`:
  - **Stage 1**: per head, train a **LogisticRegression head-selection probe** (`FinalClassifier.evaluate_single_head` → fit, predict, return val **F1**). Build `HeadInfo(layer, head, validation_f1)`; sort desc; keep **top-5**.
    - `head_key.split('_')[1::2]` parses `layer_{L}_head_{H}` → `(L, H)`.
    - Dumps `head_heatmap.png` (Fig 3 reproduction).
  - **Stage 2**: train final ensemble on concatenated features of selected heads.
- `evaluate_sufficiency(features)` (inference): concat selected-head vectors → `classifier.predict_proba` → `is_sufficient = P(sufficient) >= threshold`, returns `(is_sufficient, confidence)`. `confidence = pred_probs[1]` (= P(sufficient)).
- `save()`/`initialize_probe()`: pickles `selected_heads` + classifier state to `probe_state.pt`.

## Ensemble classifier — `final_classifier.py` (`FinalClassifier`)

- **8 base classifiers** configured: `xgboost`, `lightgbm`, `catboost`, `gradient_boosting`, `mlp` (128-64-32 MLP), `svm` (RBF), `logistic`. (head-selection uses a separate balanced `LogisticRegression`.)
  - Tree models tuned: `n_estimators/iterations=200`, `lr=0.05`, `depth=6`, class-balancing (`scale_pos_weight=2` / `class_weight='balanced'`), `eval_metric=AUC`.
- `train_final_classifier(...)`:
  1. Build `X_train/X_val` = `np.concatenate(selected-head features)` per example; labels from tuple index `[1]`.
  2. `preprocess_features`: nan-fill + **IQR outlier clipping**; `StandardScaler`.
  3. `cross_validate_classifiers`: **StratifiedKFold(n=5)**, score by **accuracy** per fold.
  4. Select **top `num_top_classifiers=4`** by mean CV score.
  5. **Weighted soft VotingClassifier** — weights `1.0 + i/10` descending by rank → better CV classifier gets higher weight.
  6. Fit ensemble on scaled features; report val acc + AUC, ROC plots.
- `predict_proba(features)`: reshape→preprocess→scale→`ensemble.predict_proba` → `(P(insuff), P(suff))`.

## Cutoff mechanism — `early_stopping.py` (`EarlyOutputModel`)

- `generate(example, chunk=True, threshold)`:
  - `chunk=False` → `_full_context_generate` (baseline, full context, no early stop).
  - `chunk=True` (dynamic cutoff):
    1. `chunking_manager.create_chunks(strategy="percentage", num_chunks=10)` → token boundaries.
    2. Maintain `DynamicCache()` (HF KV cache) and growing `all_input_ids`.
    3. **Loop over chunks**: tokenize current chunk (chunk 0 wrapped with `prepare_sufficiency_input`, later chunks raw), append to cache, forward pass `model(input_ids, attention_mask, past_key_values=context_cache)` — **incremental, reuses KV cache**.
    4. Extract features via `get_head_features_for_eval`, call `probe.evaluate_sufficiency`.
    5. **Stop condition**: `if is_sufficient and confidence >= threshold` → `chunk_generate(...)` produces the answer from the cached prefix + suffix prompt (`"\n\nYour answer:\n\n"`); return with `early_stopped=True`, `chunks_processed`, `pred_percentage`, `percentage_diff` vs gold.
    6. If never sufficient → process all chunks then generate (`early_stopped=False`).
- `chunk_generate`: appends suffix prompt, concatenates all input_ids, `model.generate(..., past_key_values=context_cache)` — generation conditioned on the cached sufficient prefix only.
- `process_examples`: batched loop, writes `early_output_results_{mode}_{threshold}.json`, clears activations/GPU between examples.

## Chunking — `chunking_manager.py` (`ChunkingManager`)

- `create_chunks(strategy)`:
  - `"percentage"` (default): even token boundaries, `num_chunks=10` → 10% increments; remainder spread over first chunks.
  - `"sentence"`: NLTK `sent_tokenize`; task-specific splitters — code splits on `def`/`class`, `multihop_kv` splits on `;`/newline pairs.
  - `"token"`: every token a boundary (token-level cumsum fast path in `get_chunk_features`).

## Prompts — `utils.py` / `prompts.py`

- **Sufficiency / answer-gen input** (`prepare_sufficiency_input`):
  > `Please provide a response to the query based only on the given context:\n\n[QUERY]: {query}\n\n[CONTEXT]: {context}`
- **Suffix** (`get_suffix_prompt`): `"\n\nYour answer:\n\n"`.
- **Self-sufficiency prompt** (paper D.1, for large-model self-assessment): asks for strict `[[YES]]`/`[[NO]]`.
- **Evaluation prompt** (`prompts.py`): GPT-4o-mini judge, outputs `[[YES]]`/`[[NO]]` for answer correctness (paraphrase-tolerant).

## Key configs — `config.py`

| Config key | Value | Meaning |
|---|---|---|
| `MODEL_CONFIG.model_name` | `Llama-3.2-1B-Instruct` | base LM (CLI-overridable) |
| `MODEL_CONFIG.eval_model` | `"api"` | GPT-4o-mini judge (else local model) |
| `TRAINING_CONFIG.samples_per_task` | 600 | per-dataset points |
| `TRAINING_CONFIG.num_heads` | **5** | top-$k$ heads selected |
| `TRAINING_CONFIG.dev_ratio` | 0.8 | train/val split |
| `TRAINING_CONFIG.train/eval_tasks` | `{multihop_qa, singlehop_qa, multihop_kv, code}` | task set |
| `CLASSIFIER_CONFIG.target_precision` | 0.9 | for R@90P |
| `CLASSIFIER_CONFIG.default_threshold` | 0.5 | base $\tau$ |
| `CLASSIFIER_CONFIG.num_top_classifiers` | **4** | ensemble size (of 8) |
| `DATA_CONFIG.chunking` | `percentage`, 10% | chunk strategy |
| `EARLY_OUTPUT_CONFIG.min_sufficient_confidence` | **95** (%) | inference $\tau$ → 0.95 |
| `EARLY_OUTPUT_CONFIG.eval_method` | `match` | exact match vs prob |
| `GENERATION_CONFIG` | `temp≈0`, `max_new_tokens=128` | near-greedy decode |

- **Memory footprint**: full ensemble (8 tree/linear models, pick top-4) <15MB — negligible vs the LLM. Experiments ran on 2–4× A5000/A6000 (Table 10). Classifier can be offloaded to CPU.

---

## Related Notes

- **[LensVLM](../LensVLM/LensVLM.md)** — **Shares authors** (Roy Xie, Bhuwan Dhingra) and the long-context token-budget goal. Complementary levers: LensVLM **compresses** the context representation (optical), this work **truncates reading** once a sufficiency probe fires. Composable — compress *and* stop early.
- **[DeepSeek-OCR](../DeepSeek-OCR/DeepSeek-OCR.md)** / **[OCR-Memory](../OCR-Memory/OCR-Memory.md)** — Same "context tokens are the binding constraint" motivation, opposite mechanism: those pack more text into fewer (vision) tokens; this one **reads fewer tokens** by self-terminating. Orthogonal and stackable with optical compression.
