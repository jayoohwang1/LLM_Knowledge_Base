# Learning Facts at Scale with Active Reading

**Authors**: Jessy Lin, Vincent-Pierre Berges, Xilun Chen, Wen-Tau Yih, Gargi Ghosh, Barlas Oğuz (FAIR @ Meta / UC Berkeley) — Aug 2025

---

## TL;DR

- **Problem**: LLMs absorb facts unreliably during pretraining; finetuning on new docs gives brittle memorization + hallucination. Practitioners lack tools to *reliably* inject a body of knowledge.
- **Idea**: **Active Reading** = let model self-generate *diverse study strategies* per document (paraphrase, timelines, songs, associations, active recall, analogies), apply each strategy to that document, train on the resulting augmentations.
- **Results**:
    - SimpleWikiQA: 16% → **66%** factual recall (+312% relative over vanilla FT).
    - FinanceBench: +160% relative on info-extraction subset.
    - Scaled to **1T tokens** of synthetic Wikipedia → **WikiExpert-8B**, beats Llama 3.1 405B and DeepSeekV2 236B on SimpleQA, competitive with DeepSeekV3 671B.

---

## 1. Motivation

- **Knowledge incorporation is poorly understood**.
    - Pretrained LLMs encode lots of facts, but coverage is uneven — tail facts are unreliable (Kandpal '23, Wei '24).
    - Finetuning on new docs → hallucinations (Gekhman '24) or rote memorization w/o ability to *manipulate* knowledge (Allen-Zhu/Li '24).
- **Goal**: not rote recall but **generalization** — facts internalized in a form usable for reasoning, transfer, compositional inference.
- **Inspiration**: humans don't internalize text by re-reading; they actively *study* via paraphrase, recall, mnemonics, associations, diagrams, etc. (Brown et al., *Make It Stick*, 2014).

---

## 2. Active Reading Method

> **Two-stage synthetic data pipeline.**

### Stage 1 — Generate learning strategies (per document)

- Prompt the model to propose diverse, document-specific strategies.
- Two prompt variants:
    - **Task-agnostic**: "What are some strategies specific to this document that I can use to help me learn and remember all of the information contained?"
    - **Task-specific**: "I need to study for a trivia competition. Generate a list of questions... then for each, generate a general study strategy that would help me memorize that kind of info."
        - First imagines downstream questions → strategies tailored to those question types.
        - More focused on tail facts, more diverse (each strategy targets a different aspect of doc).

### Stage 2 — Apply each strategy to the document

- For each strategy, prompt: `Here's a learning strategy: {strategy}. Apply this strategy to the following document: <doc>`.
- Generates many diverse "study notes" per source doc.
- Train the model on union of all generated augmentations.

### Example strategies (auto-generated for a Wikipedia article on the IEEE Frank Rosenblatt Award)

- **Create a timeline** of recipients and ideas.
- **Create a song or rhyme** using the names.
- **Use association** with other people/events you know.
- → produces stuff like *"2006: Lawrence J. Fogel; 2007: ..."*, a verse-form song, or *"John J. Hopfield (2009): associated with the Hopfield network..."*.

### Why it works (the authors' claim)

- Existing methods (paraphrase, synth QA, EntiGraph, concept maps) are **single fixed strategies** → ceiling on diversity.
- Active Reading **recovers all of these as special cases** + adds many more, all *context-specific* per doc.
- Produces **higher diversity** training data (measured below).

---

## 3. Expert Domain Adaptation (Sec 4.1)

### Setup

- Base model: **Llama 3.1 8B**, 20k steps, ~4B generated words per method.
- Mix in **10% pretraining data (DCLM)** to prevent degradation.
- Benchmarks:
    - **SimpleWikiQA**: 3,449-question subset of SimpleQA where reference docs are Wikipedia. Tests *tail* knowledge.
    - **FinanceBench**: 10,231 finance Q&A grounded in SEC docs; focus on "information extraction" subset (factual recall vs. reasoning).

### Results (Table 1)

| Method | SimpleWikiQA | FinanceBench (info) | FinanceBench (all) |
|---|---|---|---|
| Base Llama 3.1 8B | 7.42 | 3.93 | 6.00 |
| + repeat (raw docs) | 15.92 | 18.43 | 10.49 |
| + paraphrase | 25.74 | 43.87 | 17.64 |
| + synth QA | 47.87 | 44.23 | 17.16 |
| **+ Active Reading (task-agnostic)** | **63.33** | **66.18** | **26.83** |
| + Active Reading (task-specific) | 66.25 | 61.49 | 25.16 |
| + paraphrase + synth QA + AR | 66.66 | 64.45 | 26.12 |
| gold context (8B, in-context docs) | 65.85 | 84.71 | 44.36 |
| gold context ceiling (70B Instruct) | 90.55 | 92.49 | 57.43 |

- **AR matches gold-context 8B on SimpleWikiQA** — i.e., parametric storage as good as having the doc in context (for tail facts).
- Still a gap to gold-context ceiling on FinanceBench *all* — i.e., reasoning over docs still benefits from in-context processing.

### Scaling behavior (Fig 2)

- Paraphrase **plateaus quickly** (~0.27 acc) — limited ways to rephrase a doc.
- Synth QA plateaus around ~0.47.
- **Active Reading keeps climbing** through 4B words — higher data diversity ceiling.

---

## 4. Pretraining-Scale Fact Learning (Sec 4.2)

### The distractor problem (Fig 3)

- SimpleWikiQA docs = ~0.1% of Wikipedia. When you mix in *more* (4×, 16×) other Wikipedia docs, **performance on the target QA degrades substantially** — model can't allocate enough capacity per fact.

### Two-knob fix

1. **Bigger learning rate** (1e-5 → **3e-4**): looks more like continued pretraining than finetuning. Pushes model out of local minima, creates "plasticity" to absorb new facts.
2. **More pretraining-data mixing**: increase weight of generic pretraining data in mix.
    - Counter to expectation, *decreasing* the share of target-task augmented data (80% → 2.5%) while increasing pretraining data **recovers and improves** target task perf AND **recovers guardrail (NQ) perf** (Fig 4).
    - **Surprise**: even the target task improves with *less* of its own training data — so it's not just catastrophic-forgetting mitigation. Hint at "plasticity"-restoring effect of pretraining data.

### WikiExpert-8B (the headline model)

- Continued pretraining of Llama 3.1 8B on **1T tokens of Active Reading-augmented Wikipedia + 1T tokens pretraining = 8T total**, 4 epochs.
- Augmented 6M Wikipedia articles via Active Reading.

| Model | SimpleQA | NQ | TQA |
|---|---|---|---|
| Llama 8B | 7.3 | 29.0 | 64.3 |
| **WikiExpert-8B** | **23.5** | 31.2 | 68.5 |
| Qwen2.5 72B | 9.1 | 33.2 | 71.9 |
| DeepSeekV2 236B | 10.2 | 38.6 | 80.0 |
| Llama 405B | 17.1 | 41.5 | 82.7 |
| DeepSeekV3 671B | 24.9 | 40.0 | 82.9 |

- **+222% on SimpleQA** vs base 8B → matches DeepSeekV3 (671B) on tail-fact recall.
- Guardrail tasks (OBQA, PIQA, HellaSwag, MMLU, GSM8k, MBPP) mostly preserved; small drops on ARC-e, GSM8k, MBPP (Appendix C).

---

## 5. Analysis — Why does AR work? (Sec 5)

### Hypothesis 1: Better *coverage* of target answers — REJECTED

- Sample 100 SimpleWikiQA questions, ask Llama 3.1 70B judge: "Is the answer present in this synthetic doc chunk?"
- Fig 5: All methods converge to ~100% coverage by ~75 chunks. **AR is not winning on coverage** (synth QA actually has highest coverage).

### Hypothesis 2: Higher *diversity* — SUPPORTED

- Measure **Self-BLEU** between synthetic docs from the same source (lower = more diverse).
- Fig 6: AR task-specific is the most diverse, AR task-agnostic next. Paraphrase and synth QA are highly self-similar.
- Diversity tracks downstream performance.
- **Takeaway**: it's not what facts appear, it's *how many different ways* they appear.

### Model size scaling (Table 3)

| Trained model | Data gen model | SimpleQA |
|---|---|---|
| Llama 8B | 8B-generated | **66.25** |
| Llama 8B | 70B-generated | 62.26 |
| Llama 70B | 8B-generated | 71.52 |
| Llama 70B | 70B-generated | 77.14 |

- **Surprise**: training 8B on 70B-generated data is *worse* than 8B-on-8B.
    - Hypothesis: data closer to the model's own existing comprehension → easier to absorb (not OOD).
    - Self-generated training data may be a feature, not a workaround.

---

## 6. Practical takeaways / open questions

- **AR generalizes / subsumes prior synth-data tricks**: paraphrase (Allen-Zhu/Li '24), synth QA (Lewis '21), concept maps / EntiGraph (Yang '24b) — all emerge as special-case strategies.
- **Data scale doesn't plateau** (within tested range) — unlike fixed-template baselines.
- **Knowledge incorporation needs pretraining-like training regime**, not finetuning hyperparams. Big LR + heavy pretraining-data mix.
- **Open question**: *why* does mixing pretraining data help target-task performance (not just guardrails)? Plasticity? Reverses knowledge-entropy decay (Kim '24)?
- **vs RAG**: paper deliberately stays closed-book. Gold-context = "oracle RAG" upper bound. Argues parametric knowledge has long-term advantages: lower latency/memory, can compose across facts (RAG fails on indirect queries).
- **Future**: AR as default pretraining/mid-training augmentation, esp. as fresh-web-data runs out (Villalobos '24).

---

## 7. Prompts (from Appendix D)

### Task-agnostic strategy generation

> `Consider the following document. What are some strategies specific to this document that I can use to help me learn and remember all of the information contained? Use markdown and prefix each strategy with ##`

### Task-specific strategy generation (trivia version)

> `I need to study for a trivia competition. Generate a list of questions that covers every piece of information in this document. After generating all the questions, for each question, generate a general study strategy or prompt that would help me memorize that kind of information (without focusing too much on the particular question)...`

### Strategy application

> `Here's a learning strategy: {strategy}. Apply this strategy to the following document: <document>{chunk}</document>`

---

## 8. Connections / where this sits

- **Synthetic continued pretraining**: closest neighbor is EntiGraph (Yang et al. 2024b) — entity-relation graph traversal generates synth docs. AR is more flexible: model picks its own strategies, including graph-like ones.
- **"Physics of LLMs"** (Allen-Zhu/Li): paraphrasing is necessary for facts to be *manipulable*, not just memorized. AR is paraphrasing + many other forms of restatement.
- **Knowledge editing / lifelong learning**: AR-style continual augmentation could be a route to non-catastrophic forgetting if you can sustain the LR + mix recipe.
- **Memory layers** (Berges et al. 2024) — referenced as a candidate architecture for storing more facts/param efficiently. AR addresses training-data side; memory layers address capacity side.
- **Related concept**: distillation / synthetic data from larger model → AR's Table 3 result *contradicts* the naive "use bigger teacher" intuition.
