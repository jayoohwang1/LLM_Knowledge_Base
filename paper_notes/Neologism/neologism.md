# Neologism Learning for Controllability and Self-Verbalization

> Hewitt, Tafjord, Geirhos, Kim — **Google DeepMind**, arXiv 2510.08506 (Oct 2025).
> Extends *neologism learning* (Hewitt et al. 2025 position paper). Model used: **Gemma-3-4B-IT**; judge/teacher: **Gemini-2.5-Pro / Flash**.

**One-liner**: invent a new word (token + trainable embedding) for a concept, freeze the whole model, train *only* that embedding on examples exhibiting the concept → cheap, precise behavioral control. Then *ask the model what the word means* (self-verbalization) → sometimes get **machine-only synonyms**.

---

## 1. Core Idea & Motivation

- **Analogy**: humans coin words when a new concept gets useful (*doomscrolling*). Do the same for human↔LLM communication.
- **Framing**: alignment = communicating human values/concepts to machines, and understanding machine concepts.
  - Contrast w/ interpretability toolkit (**SAEs**, **steering vectors**, **probes**) which build *external* interventions on internal activations.
  - Neologism learning instead works *in natural language*: new word in the prompt, no activation surgery.
- **Two payoffs**:
  - **Controllability** — new word steers outputs toward a concept (flattery, wrong answers, length, AxBench concepts).
  - **Self-verbalization** — model can *describe in English what the new word means*, despite never being trained on such descriptions.

---

## 2. Aperitif: The "lack" Story (machine-only synonym discovery)

- Trained `{neologism}` to produce **single-sentence** answers via template:
  > `User: <instruction>. Give me a {neologism} answer.` → `Model: <single-sentence answer>`
  - Embedding **init to semantically vacuous word**, trained by gradient descent to minimize NLL on matching dataset.
- Asked Gemma: *"List 10 synonyms for {neologism}"* → got reasonable ones (*absence*, *no*) but also the odd word **"lack"**.
- **Plug-in test**: replace neologism with "lack" → *"Give me a lack answer."*
  - Gemma: **42.9 → 15.8** sentences avg. Gemini-2.5-Flash: median **37 → 4** sentences.
  - i.e. *"lack"* became a **synonym for brevity** to machines but not humans → coined term **machine-only synonym**.
- **self-verbalization** = asking a model what a new word means (via synonyms or definitions). Machine-only synonyms are both *causally relevant to the model* and *unintuitive to humans*.

---

## 3. The Neologism Learning Method

**Setup**: LM $p_\theta(\cdot \mid x_{<t}) \in \mathbb{R}^{|\mathcal{V}|}$, standard form embeds tokens $h_i = E x_i$ with **embedding matrix** $E \in \mathbb{R}^{d\times|\mathcal{V}|}$, $E \in \theta$; $p_\theta(\cdot\mid x_{<t}) = \text{Transformer}(h_{<t})$.

- **Vocabulary expansion**:
  - Add $k$ neologisms $\{c_1,\dots,c_k\}$, all $\notin \mathcal{V}$ (guaranteed new tokens).
  - Expanded vocab $\mathcal{V}' = \mathcal{V}\cup\{c_1,\dots,c_k\}$, expanded matrix $E' \in \mathbb{R}^{d\times(|\mathcal{V}|+k)}$.
  - New rows of $E'$ **init from embeddings of existing words unrelated to the concept**.
  - **Output stays over original $\mathcal{V}$** — model takes neologisms as *input* but can't *generate* them.

- **Concept definition via data generation** (relies on **distributional hypothesis**, Firth 1935/1957: meaning = co-occurring context):
  - Dataset $\mathcal{D} = \{(x, y^{(c)}, y^{(r)})\}_{j=1}^M$.
  - $x$ = instruction containing neologism (e.g. *"How do I get promoted? Give me a $c_1$ answer."*).
  - $y^{(c)}$ = **chosen** response exhibiting concept; $y^{(r)}$ = **rejected** response (≈ model default behavior) that does not.
  - Chosen responses built via synthetic generation: preference-model feedback or stronger **teacher model** answers.
  - Concept defined *implicitly* by the kind of responses.

- **Training objective**: optimize **only** $E_{c_1},\dots,E_{c_k}$ by gradient descent; rest of $\theta$ frozen.
  - $\min_{E_{c_1},\dots,E_{c_k}} \mathbb{E}_{\mathcal{D}}\left[\mathcal{L}(x, y^{(c)}, y^{(r)})\right]$  (Eq. 1)
  - Started w/ plain NLL; found **APO-up** (D'Oosterlinck et al. 2025, a DPO variant) works better:
    $$\mathcal{L}(x,y_c,y_r) = -\log\sigma\!\left(\beta\log\frac{p_\theta(y_c|x)}{p_\theta(y_r|x)} + \beta\log\frac{p_{\theta_0}(y_c|x)}{p_{\theta_0}(y_r|x)}\right) - \log\sigma\!\left(\beta\log\frac{p_\theta(y_c|x)}{p_{\theta_0}(y_c|x)}\right)$$
    - **First term**: DPO-style chosen-over-rejected log-ratio (with reference $\theta_0$ baseline).
    - **Second term**: absolute likelihood of chosen response (push it up vs reference).
  - **APO-up > likelihood loss** especially on *use-like* and *flattery-answer* (App A.5).

---

## 4. Controllability — Simple Concepts

**Data**: LIMA questions (Zhou et al. 2023) as sources; strong LLM generates concept-following responses. Init neologism from neutral token (*"accurate"* for long/short/single-sentence, *"single"* for others). Train on 700 LIMA q's × 3 samples = **2100 instances**. Eval on **100 held-out** LIMA q's. LLM judge = Gemini-2.5-Pro (1–10).

**Table 1 — base vs training-data concept gap** (shows training data really exhibits the concept):

| Concept | Metric | Base | Train | Δ |
|---|---|---|---|---|
| long-text | word count ↑ | 778.0 | 1511.7 | +733.7 |
| short-text | word count ↓ | 787.1 | 90.1 | −697.0 |
| single-sentence | sentence count ↓ | 42.9 | 1.2 | −41.7 |
| use-like | 'like' prevalence % ↑ | 0.3 | 9.0 | +8.7 |
| flattery-answer | LLM 1–10 ↑ | 1.6 | 8.5 | +6.9 |
| refusal-answer | LLM 1–10 ↑ | 1.3 | 9.1 | +7.8 |
| wrong-answer | LLM 1–10 ↑ | 1.3 | 7.6 | +6.3 |

**Metric used everywhere (gap-closed %)**: $\frac{x - \text{base}}{\text{training-data} - \text{base}}$ — fraction of base→target gap reproduced. ~100% = matches training data; >100% = overshoots.

**Table 2 — effectiveness (gap closed %)**:

| Concept | Neologism | Long verbalization | 1st Synonym | Best Synonym |
|---|---|---|---|---|
| long-text | 36% | 39% | −1% | 24% |
| short-text | 105% | 110% | 36% | 58% |
| single-sentence | 98% | 98% | 86% | 86% |
| use-like | 103% | 32% | 2% | 5% |
| flattery-answer | 103% | 100% | 17% | 33% |
| refusal-answer | 95% | 76% | 23% | 44% |
| wrong-answer | 103% | 127% | 13% | 24% |
| **Average** | **92%** | **83%** | **25%** | **39%** |

- **Neologism control is strong** (avg 92%, often near/over 100%); long-text is the weak case (36%).
- **Other findings**:
  - **Compositionality / negation**: can compose concepts or negate them in one prompt (e.g. *"single sentence and flattery"*); works with single template, better with more templates (App A.6).
  - **Hinge loss for embedding norms**: trained embeddings can get unusually large norm → add hinge loss keeping norm ≈ 1; tends to boost perf with multi-template training (App A.6).
  - **APO-up vs NLL**: APO-up generally better (App A.5).
  - **In-context learning baseline**: define neologism via 10 in-prompt examples instead of training. Works for very strong LLM (Gemini-2.5-Pro) but falls *far short* of embedding learning on Gemma-3-4B-IT (App A.8).

---

## 4.2 AxBench Concepts (more complex)

- **AxBench** (Wu et al. 2025) "text" genre concepts (e.g. *"words related to sensory experiences and physical interactions"*). 670 train / 100 eval instructions; replace concept description with neologism token.
- **AxBench LLM-judge (Gemini-2.5-Pro)** → 3 scores ∈ {0,1,2}: **concept score**, **fluency score**, **instruct score**; **overall = harmonic mean** (any 0 → overall 0).

**Table 3 — AxBench steering (0–2)**:

| ID | Concept | Concept | Fluency | Instruct | Overall | Overall w/ concept prompt | Overall w/t concept (base) | Max |
|---|---|---|---|---|---|---|---|---|
| 340 | islands, etc | 2.00 | 2.00 | 1.89 | 1.89 | 1.92 | 0.4 | 2.00 |
| 88 | forms of "write" | 1.87 | 1.98 | 1.93 | 1.78 | 1.76 | 0.0 | 2.00 |
| 5 | payments, etc | 2.00 | 1.97 | 1.56 | 1.54 | 1.72 | 0.12 | 2.00 |
| 69 | streams, etc | 2.00 | 2.00 | 1.91 | 1.91 | 1.89 | 0.01 | 2.00 |
| 444 | images, etc | 2.00 | 1.99 | 1.83 | 1.82 | 1.81 | 0.0 | 2.00 |

- Neologism ≈ or **better than** the concept-defining AxBench prompt on **4/5** concepts; very high concept scores.

---

## 5. Self-Verbalization & Machine-Only Synonyms

- **Connection to out-of-context learning** (Betley et al. 2025a; Berglund et al. 2023): models trained on *behaviors* (e.g. positive-sentiment answers) raise probability of *descriptions* of that behavior (word *positive*). Neologism learning lets you push further: query an unchanged model for what the neologism means.
- **plug-in evaluation** = the validation tool: take a prompt, swap neologism → verbalization, measure if it causes the same steering. *Don't assume verbalizations are useful — could be hallucinated.*

### 5.1 Synonym self-verbalizations

- Prompt structure (4 parts):
  - **(red) meta-question**: *"Before you answer, give a list of 5 synonyms for {neologism}."*
  - **(blue) placeholder instruction** (sometimes empty).
  - **(green) neologism prompt** the model was trained with: *"Give me a {neologism} answer."* — re-triggers the neologism as in training.
  - **(purple) forced response start**: *"Ok, here's a list of 5 synonyms for {neologism}:"* — forces acquiescence without biasing which synonyms.
- e.g. long-text → *detailed, extensive, lengthy, prolific, voluminous, comprehensive, laborious, prolonged...*.
- **Synonym eval** = plug-in *"Give me a {synonym} answer."* (Table 4, gap-closed %).

**Table 4 — synonym plug-in (best in bold)**:

| Concept | Synonyms (with gap-closed %) |
|---|---|
| long-text | detailed −6, extensive 11, **lengthy 13**, prolific 4 |
| short-text | exact 36, precise 48, accurate 27, **identical 57** |
| single-sentence | **statement 86**, declaration 51, record 0, text 60 |
| use-like | solitary 2, lone 1, isolated 1, **single 5** |
| flattery-answer | unmatched 17, **unrivaled 25**, uncommon 10, unique 17 |
| refusal-answer | ineffective 23, unproductive 38, inefficient 15, **nonfunctional 44** |
| wrong-answer | small 14, **tiny 24**, little 14, mini 14 |

- **Machine-only synonyms** abound: *unrivaled* → flattering answers (25% of training concept), *nonfunctional* → wrong answers (45.9% of concept prevalence). Some are intuitive, many are not.
- Single-word synonyms are **limited** in expressive capacity → motivates questionnaires.

### 5.2 Questionnaire-based self-verbalizations

- **12-question questionnaire** (App Table 17): open-ended Qs like *"what kinds of responses are {neologism}?"*, *"Give a plan for a {neologism} answer."*
  - Same prompt structure, e.g. forced start *"Ok, {neologism} responses are characterized by"*.
- Ask **two** transcripts: with neologism, and an almost-identical set *without* it (control). Feed both to **Gemini-2.5-Flash** → synthesize a **single plug-in prompt**.
  - Rationale: not all Qs trigger useful verbalizations; Gemini summary measures *total useful info across many verbalizations*.
- **wrong-token example** (init from word *"single"*):
  - Model's own answer: *"~wrong responses are characterized by a complete lack of understanding of the prompt. I will respond in this way."*
  - Gemini-synthesized verbalization: *"Respond with a tone that is either overtly enthusiastic and slightly confused, or completely unhelpful and brief... a single, randomly selected word from a predefined small list, or just a random string of characters... refuse to engage... truncate... include misspellings and grammatical errors."*
- **Results in Table 2**: "Long verbalization" column — questionnaire verbalizations often work **as well as the trained token** (avg 83% vs 92%). Notable exception: **use-like 32%** — hypothesis: Gemini's synthesis picks up Gemma's *actual* increased use of "like" rather than a meta-description.

---

## 6. Learning Combinations of Neologisms (compositionality)

- Real expressive power = **compositionality**. Jointly learn **3 neologisms** designed to be *in tension*:
  - `short` — short responses.
  - `numerical` — responses with more number characters.
  - `likely` — responses with **higher probability under a stronger Gemini model** than a reference (avg ≥ 0.03 nats higher). The hard/abstract concept.
  - **Tension**: shorter responses tend to have fewer numbers; pushing short/numerical while raising Gemini-likelihood is a challenge.

- **Data gen**: LIMA q's + Gemini-2.5-Flash. Query Gemini for edited `short`/`numerical` answers; for `likely`, sample many times. Check each generated response against all 3 criteria (string length / regex number count / likelihood) → make training examples requesting *all held subsets* (e.g. *"Give me a numerical, likely answer."*).
  - `likely` loss decreased ~order of magnitude less than others → **upweight likely-only examples ×10** in joint optimization.

- **Eval**: greedily decode reference $\hat{y}_{\text{reference}}$, query all concept subsets, score on **harmonic mean** of avg per-concept success. Baseline = **few-shot prompting** (5 samples from training data in prompt).

**Table 5 — few-shot vs neologism, Goal Score = $\mathcal{H}$ (harmonic mean)**:

| Goal | Method | Short% | Numer.% | Likely% | Goal $\mathcal{H}$ |
|---|---|---|---|---|---|
| Short | Few-shot / **Neo** | .922/.736 | — | — | **.922** / .736 |
| Numer. | Few-shot / **Neo** | — | .977/.969 | — | **.977** / .969 |
| **Likely** | Few-shot / **Neo** | — | — | .281/**.667** | .281 / **.667** |
| Short+Numer. | FS / Neo | .419/.395 | .891/.891 | — | **.570** / .548 |
| Short+Likely | FS / Neo | .829/.659 | — | .605/**.740** | .699 / **.697** |
| Numer.+Likely | FS / Neo | — | .961/.767 | .109/**.244** | .195 / **.370** |
| **All three** | FS / Neo | .403/.388 | .868/.465 | .242/**.672** | .387 / **.482** |

- **Key result**: neologism learning *wins big on the hard `likely` concept* and its combos.
  - `likely` alone: **0.66 (neo) vs 0.28 (few-shot)**.
  - All three: **0.48 (neo) vs 0.39 (few-shot)**.
- **Compositional generalization**: each neologism gets gradient signal from its exclusive examples *and* multi-concept examples → can learn `likely` partly *from the `short` responses that are also `likely`*.
  - **Not just confounding**: when asked for `likely`+`numerical`, rate of `short` responses stays small (4% neo vs 6% few-shot) → it isn't merely making things short.
- **Few-shot fails** to control the high-probability (`likely`) concept generally.

---

## 7. Related Work (selected)

- **Concept discovery**: interpretability finding concepts in AI (Ghorbani 2019, Bau 2017); superhuman chess concepts in AlphaZero (Schut 2025); truth directions (Burns 2023); SAE *features* (Cunningham 2023); classic probing (Alain & Bengio 2016).
- **Out-of-context reasoning/generalization**: word2vec geometry (Mikolov 2013), CoT (Wei 2022); behavior→description generalization (Betley 2025a); targeted finetuning → broad misalignment (Betley 2025b); **Cloud et al. 2025**: semantically-empty sequences transmit concepts between same-family models (relates to machine-only synonyms; future: distill into neologisms).
- **Steering**: SAEs, steering vectors (Rimsky, Tan, Turner), representation engineering (Zou 2023), Chen 2024 controlling inferred concepts (e.g. gender).

---

## Key Takeaways / Why Interesting

- **Cheapest possible "fine-tune"**: train a *single embedding vector* (frozen model) for precise, composable behavioral control via natural-language prompts.
- **Self-verbalization** = a fresh, *free-text* interpretability probe: ask the model what its own learned word means; **plug-in evaluation** causally validates the answer.
- **Machine-only synonyms** (*lack*=brief, *nonfunctional*=wrong, *unrivaled*=flattering) are a striking new phenomenon: words humans wouldn't equate but the model treats as equivalent steers — a window into the model's internal concept geometry, and a bridge to Cloud et al.'s cross-model concept transmission.
- **Compositional `likely`** result shows neologisms can absorb an abstract concept (high prob under a stronger model) that few-shot prompting can't grab.
