# Ch 3: Prompting

**Prompting** = providing LLM with input/cue $\mathbf{x}$ to generate desired output $\mathbf{y}$ by maximizing $\Pr(\mathbf{y}|\mathbf{x})$. Prompt = condition for prediction.
- **Prompt engineering**: active research area of designing effective prompts.
- **In-context learning (ICL)**: add demonstrations of problem-solving into context, LLM "learns" from them at inference (no weight updates). Emergent ability from pre-training.

---

## 3.1 General Prompt Design

### 3.1.1 Basics

- **Prompt template**: text with placeholders/variables to fill.
  - Example: `If {*premise*}, what are your suggestions for a fun weekend.`
  - Can have multiple variables (e.g. comparing sentence similarity with `{*sentence1*}`, `{*sentence2*}`).
- **"name:content" format**: structured way to format prompts.
  - **Dialogue style**: `John: ... / David: ...` continuation.
  - **QA style**: `Q: {*question*} / A: ___`.
  - **Attribute spec**: `Task: Translation / Source language: English / Target language: Chinese / Style: Formal / Template: ...`.
  - Practical systems store such attributes in **JSON** key-value pairs.
- **Role assignment + system info**: assign LLM a role and give context/constraints.
  - Example: `You are a computer scientist with extensive knowledge... Please explain {concept} to a child...`.
  - The role text is **system information** (separate field in many APIs).

### 3.1.2 In-context Learning

- **Mechanism**: ICL doesn't update params; just **activates/reorganizes** pre-training knowledge via prompt context.
- **Three regimes**:
  - **Zero-shot**: instruction only, no demonstrations. Adapt by rewording instructions iteratively.
    - Format example with `SYSTEM` / `USER` fields.
  - **One-shot**: add a single input-output demonstration (`DEMO`).
  - **Few-shot**: multiple demos.
    - Demos define a pattern $\text{input} \to \text{output}$ for LLM to follow.
    - Patterns can be very compact (e.g. Chinese-English word pairs, math expression patterns).
    - Strong LLMs can do complex tasks (e.g. mathematical reasoning) from few-shot patterns.
- **Effectiveness depends on**:
  - **Prompt quality** (engineering effort).
  - **Underlying LLM capability**. If pre-training lacks domain (e.g. low-resource Inuktitut), better prompts won't fix — need more pre-training.
- **Theoretical interpretations of ICL**:
  - Bayesian inference [Xie et al. 2022]
  - Gradient descent [Dai et al. 2023; Von Oswald et al. 2023]
  - Linear regression [Akyürek et al. 2023]
  - Meta learning [Garg et al. 2022]

### 3.1.3 Prompt Engineering Strategies

Empirical, trial-and-error. Key strategies:

- **Describe the task clearly and specifically**.
  - "Tell me about climate change" → too vague.
  - Better: specify aspects (causes, effects, solutions), audience (10-yr-old), word count, style.
- **Guide LLMs to think**.
  - **Appending "Let's think step by step"** [Kojima et al. 2022] boosts reasoning.
  - **Provide explicit reasoning steps** in prompt:
    - e.g. `Step 1: Problem Interpretation / Step 2: Strategy Formulation / Step 3: Detailed Calculation / Step 4: Solution Review`.
  - **Multi-round interactions**: first solve, then re-prompt to evaluate/correct ("Evaluate the correctness of this solution... Then work out your own solution").
- **Provide reference information**.
  - General LLMs can leverage any extra context.
  - **RAG-style prompt**: include retrieved `Context information` + `Query`, instruct LLM to answer based on context (in own words, not copy).
  - Can restrict to context-only (e.g. context is a table, must answer using only table).
- **Pay attention to prompt formats**.
  - LLMs are **sensitive** to small changes (ordering of sentences → different outputs).
  - **Fielded format**: define fields and fill them (reduces ambiguity).
  - **Code-style prompts**: `[English] = [...] / [German] = [...]` (LLMs understand code).
  - Can use **control chars / XML tags / quotes** to delimit sections (e.g. "input is delimited by double quotes").
- See OpenAI's prompt engineering guide ("six strategies for getting better results").

### 3.1.4 More Examples

#### Text Classification (polarity / sentiment)
- Simple prompt: `Analyze the polarity... classify as positive, negative, or neutral`.
- **Problem**: LLMs output text not labels.
  - **Label mapping**: extract label from generated text via heuristics.
  - **Cloze reformulation**: ask LLM to fill the blank `The polarity of the text is ___`, constrain $\text{label} = \arg\max_{y \in Y} \Pr(y|\mathbf{x})$ where $Y = \{\text{positive, negative, neutral}\}$.
  - **Constrained output via prompt**: "Just answer: positive, negative, or neutral."
- **More categories / difficult labels**: include detailed category definitions in prompt.
- For very difficult classification with labeled data → fine-tuning (BERT + classifier) may be preferable.

#### Information Extraction
- **NER**: `Identify all person names in the provided text.`
- **Multi-type NER**: `Identify and classify... into categories such as person names, locations, dates, and organizations. List each entity with its type.`
- **Relation extraction**: feed entities + text, ask LLM to describe relationships.
- **Generic IE template**: `You will be provided with a text. Your task is to {*task-description*}` (extract keywords, key events, coreferences, etc.).

#### Text Generation
- Two classes:
  - **Text completion**: continue input text (story continuation, conversation continuation).
  - **Text transformation**: translation, summarization, style transfer.
- **Generation with requirements**: theme, structure, emotion (e.g. five-character Chinese regulated poem).
- **Code completion**: e.g. write Python function.
- **Text transformation examples**:
  - Translation: `Translate the following text from English to Spanish. Text: ...`
  - Summarization: `Summarize the following article in no more than 50 words:`
  - Style transfer: `Rewrite this text in a formal tone.`

#### Question Answering
- Simple template: `Question: {*question*} / Answer: ___`.
- Many tasks reduce to QA (MMLU = multiple choice; GSM8K = grade school math).
- **Calculation annotation**: e.g. `<< 8*2 = 16 >>` markers — at inference a calculator handles expressions in `<< ... >>`, reducing arithmetic errors. **`####`** marks final answer.

---

## 3.2 Advanced Prompting Methods

### 3.2.1 Chain of Thought (CoT)

**Idea** [Wei et al. 2022c; Chowdhery et al. 2022]: prompt LLM to generate **step-by-step reasoning** before final answer.

- **Few-shot CoT**: demos include reasoning steps, not just final answer.
  - Example: `Calculate 1² = 1, 3² = 9, 5² = 25, 7² = 49. Sum the squares, 1+9+25+49 = 84. There are 4 numbers. Divide 84/4 = 21. The answer is 21.`
- **Zero-shot CoT** [Kojima et al. 2022]: just append **"Let's think step by step"** (or variants: "Let's think logically", "Please show me your thinking steps first").
  - Sometimes needs a 2nd prompting round to extract final answer from reasoning steps.
- **Benefits of CoT**:
  - Decomposes complex problems into sequential steps (mimics human problem-solving).
  - **Transparency/interpretability** — visible reasoning.
  - **Trust** — easier to verify (medicine, education, finance).
  - **In-context approach** — works with off-the-shelf LLMs, no training needed.
  - Can inspire creative solutions by exploring alternative reasoning paths.
- **Application domains**: math, logic, commonsense, symbolic reasoning, code generation.
  - Fig 3.1 shows CoT in CSQA, StrategyQA, Dyck languages, Last Letter Concatenation.
- **CoT extensions**:
  - **Structured reasoning**: tree-based [Yao et al. 2024], graph-based [Besta et al. 2024] — explores more decision paths (analogous to **System 2 thinking** [Kahneman 2011]; System 1 = fast/automatic, System 2 = slow/deliberate).
  - **Multi-round interactions**: sub-problem decomposition, verification, refinement, ensembling.
  - CoT can be seen as a **test for LLM capability**.
- **Limitations**:
  - Need multi-step demos (hard to obtain).
  - No standard recipe for decomposing problems.
  - Errors in intermediate steps propagate.

### 3.2.2 Problem Decomposition

**Broader paradigm** than CoT — break problem into sub-problems.

- **Example: blog writing**. Instead of "write a blog about AI risks", provide an **outline** (Intro / Privacy / Misinformation / Cyberbullying / Tips / Conclusion) with section descriptions.
- **Divide-and-conquer for long docs**: split long doc into segments, score each for relevance, combine via another prompt.

**Three approaches** to multi-step reasoning:
1. LLM predicts directly (hidden reasoning).
2. LLM generates multi-step reasoning in one run (CoT).
3. **Break into sub-problems** addressed by separate LLM runs or other systems (this section's focus).

**General framework**:
- **Sub-problem Generation** $G(\cdot)$.
- **Sub-problem Solving** $S_i(\cdot)$.

**Formal model (least-to-most prompting [Zhou et al. 2023b])**:
$$\{p_1,...,p_n\} = G(p_0)$$
$$a_i = S_i(p_i, \{p_0, p_{<i}, a_{<i}\})$$
$$a_0 = S_0(p_0, \{p_{\leq n}, a_{\leq n}\})$$

- Sub-problems solved **sequentially**, prior QA pairs passed as context.
- Final answer derived from all sub-problem answers.

**Example (least-to-most)**: question "What was the duration of the environmental study?"
- Sub-problems: "When did the study start?", "When did the study end?" → solve each → combine → "5 years".

**Refinements**:
- **Dynamic sub-problem generation** [Dua et al. 2022]:
  $$p_i = G_i(p_0, \{p_{<i}, a_{<i}\})$$
  Generate next sub-problem on-the-fly based on history (reasoning path not fixed in advance).
- **Better sub-problem solvers**: $S_i$ can call IR systems, calculators, or recursively decompose [Khot et al. 2023] (hierarchical decomposition).
- **RL framing**: problem-solving as decision process; actions include $G_i, S_i$.

**Related areas**:
- **Multi-hop QA** [Mavi et al. 2024]: gather info across multiple passages (e.g. "capital of country where Einstein was born").
- **Semantic parsing / compositional generalization** (SCAN [Lake & Baroni 2018]).
- **Tool use**: a special case (covered in 3.2.5).

### 3.2.3 Self-refinement

LLMs **iteratively refine** their own outputs.

- **General 3-step framework** [Madaan et al. 2024]:
  1. **Prediction** — initial output.
  2. **Feedback Collection** — get feedback on the output.
  3. **Refinement** — refine using feedback.
- Steps 2-3 repeatable → iterative loop.
- **Feedback sources**: manual, LLM-as-critic, reward model.
- **Vague refinement vs targeted**: "Please refine it" works but is unguided. Better: "Please correct grammatical errors in the translation".
- **Self-feedback example**: same LLM is prompted to (1) answer, (2) evaluate own response, (3) refine using its own feedback.
- **Alignment connection**: self-correction can be activated via RLHF [Ganguli et al. 2023].

**Deliberate-then-Generate (DTG)** [Li et al. 2023a]:
- Single-run alternative to iterative refinement.
- Provide LLM with source + an (intentionally bad) initial translation + default error type → LLM identifies error type, then produces refined output.
- Uses **"negative evidence"** [Marcus 1993] to activate learning/contrast.
- Non-iterative but still gives refinement benefits.

**Iteration caveats**: errors in earlier steps hurt later steps; stopping criterion needs engineering.

### 3.2.4 Ensembling

Combine multiple predictions for better output.

**Three ensembling axes** (Fig 3.2):

| Type | Vary | Same |
|---|---|---|
| **(a) Model ensembling** | Multiple LLMs | Same prompt |
| **(b) Prompt ensembling** | Multiple prompts | Same LLM |
| **(c) Output ensembling** | Sample multiple outputs | Same LLM, same prompt |

- **Formal**: $\hat{\mathbf{y}} = \text{Combine}(\hat{\mathbf{y}}_1, ..., \hat{\mathbf{y}}_K)$.
- **Combination methods**:
  - **Voting** / pick most overlapping.
  - **Token-level averaging**: $\hat{y}_j = \arg\max_{y_j} \sum_k \log \Pr(y_j | \mathbf{x}_k, \hat{y}_1, ..., \hat{y}_{j-1})$.
- **Bayesian view**: treat prompt $\mathbf{x}$ as latent var:
  $$\Pr(\mathbf{y}|p) = \int \Pr(\mathbf{y}|\mathbf{x}) \Pr(\mathbf{x}|p) d\mathbf{x}$$
  - Approximated via **Monte Carlo sampling** over diverse prompts.

**Diverse prompts via**:
- Manually create different demos.
- LLM-generated demos/prompts.
- Reorder demos within a prompt.
- LLM paraphrases of one prompt.
- Translation into other languages.

**Output diversity** (without varying prompts):
- Beam search → collect all complete hypotheses.
- Modify search algorithms for richer sampling.

**Self-consistency** [Wang et al. 2022a, 2023b]:
- Prompt with CoT, sample many reasoning paths, count answer frequencies, **output most frequent answer** (not highest probability).
- Example: 3 coin-flip predictions → two give 3/8 (correct), one gives 1/3 (wrong) → consensus 3/8.
- Not strictly prompt ensembling (fixed prompt+model) — it's **output ensembling / hypothesis selection**.
- Interpretation: minimum Bayes risk search.
  $$\text{Risk}(\mathbf{y}) = \mathbb{E}_{\mathbf{y}_r \sim \Pr(\mathbf{y}_r|\mathbf{x})} R(\mathbf{y}, \mathbf{y}_r)$$

**Verifiers**:
- Score outputs / reasoning chains (binary classifiers trained supervised).
- **Outcome-based** [Cobbe et al. 2021]: score entire path.
- **Process-based** [Uesato et al. 2022; Lightman et al. 2024]: score each step.

### 3.2.5 RAG and Tool Use

#### RAG
**RAG** = Retrieval-Augmented Generation. Used when LLM pre-training lacks accuracy/recency.

**Steps**:
1. Prepare external text collection.
2. Retrieve relevant texts for query (vector DB + similarity search).
3. Input retrieved texts + query to LLM, prompt for response.

**Prompt template**:
```
Your task is to answer the following question. To help you,
relevant texts are provided. Please base your answer on these texts.
Question: {query}
Relevant Text 1: {...}
Relevant Text 2: {...}
```

**Robustness improvement**: allow LLM to refuse if context is insufficient:
> "Please note that your answers need to be... faithful to the facts. If the information provided is insufficient for an accurate response, you may simply output 'No answer!'."

- **Fine-tuning** can teach LLM when to refuse.
- RAG can be seen as **problem decomposition**: (1) retrieve, (2) generate.

#### Tool Use
- LLMs as **autonomous agents** [Franklin & Graesser 1996] — call APIs, databases, simulators.
- **Markers in output** indicate tool calls.
- **Web-search example**:
  - Output contains `{tool: web-search, query: "2028 Olympics"}` → executed, result replaces marker → LLM continues with retrieved info.
- **Calculator example**:
  - `{tool: calculator, expression: 10 * 4 * 2}` → replaced with `80` → used downstream.
- **Difference from RAG**: tool calls happen **during inference**, RAG context provided **before** prediction. Both = ways to get external context.
- **Training**: LLMs need fine-tuning to emit tool-use markers correctly [Schick et al. 2024 (Toolformer)] — annotate training data with tool markers, fine-tune.

---

## 3.3 Learning to Prompt

**Motivation for automated prompting**:
- Manual design is **labor-intensive**, requires trial-and-error.
- Relies on **human expertise**, limits diversity.
- Manual prompts can be **long/redundant** → higher compute cost.

Three questions:
1. **How to automate prompt design/optimization?**
2. **Other representations beyond strings?**
3. **How to make prompts more concise?**

### 3.3.1 Prompt Optimization

**Automatic prompt design** = instance of **AutoML**. Analogous to **Neural Architecture Search (NAS)** [Zoph & Le 2016] — search over discrete structures.

**Framework**:
- **Prompt Search Space**: all candidate prompts.
- **Performance Estimation**: input prompt to LLM, measure on validation set.
- **Search Strategy**: explore promising prompts iteratively.

**LLM-based prompt optimization** [Zhou et al. 2023c] — steps:

- **Initialization**: build candidate pool $C$.
  - Manual seeds, OR LLM-generated:
    - From task description: `You are given a task to complete using LLMs. Please write a prompt to guide the LLMs. {*task-description*}`.
    - From I/O examples: `You are provided with several input-output pairs for a task. Please write an instruction for performing this task.`
- **Evaluation**: feed each prompt to LLM, measure metric or log-likelihood of output.
- **Pruning**: keep top-% prompts.
- **Expansion**: $C' = \text{Expand}(C, f)$.
  - Prompt LLM: `Below is a prompt for an LLM. Please provide some new prompts to perform the same task.`
- Iterate evaluation/pruning/expansion.

**Expansion methods**:
- **Paraphrasing** [Jiang et al. 2020].
- **Token-level edit operations** (insertions/modifications) [Prasad et al. 2023].
- **Feedback-guided revision** [Pryzant et al. 2023] (akin to self-refinement).
- **Evolutionary computation** [Guo et al. 2024] — treat prompts as candidates evolving generation by generation.

**RL-based prompt generation** [Deng et al. 2022]:
- Integrate FFN adaptor into LLM as policy network.
- Train adaptor (other params fixed) using reward from evaluator LLM.

**Beyond instructions**: prompt optimization also covers **demonstration selection/generation** for CoT [Liu et al. 2022; Rubin et al. 2022; Zhang et al. 2023b].
- Demos easier to optimize than instructions (testing instructions is expensive).

### 3.3.2 Soft Prompts

**Hard prompts** = discrete token sequences (natural language).
**Soft prompts** = **continuous, low-dimensional, learnable** vector representations.

- Encoded form LLM understands directly (not human-readable).
- Benefit: **compact, fast** to process (vs long text), great for repeated tasks.
- Can be:
  - Derived by encoding a hard prompt $c_1...c_5$ → $\mathbf{h}_1...\mathbf{h}_5$.
  - Or learned **without any corresponding text** via continuous optimization.

#### 3.3.2.1 Adapting LLMs with Less Prompting

Fine-tuning embeds prompting knowledge into params → can use simpler prompts later.

Given prompt = (instruction $\mathbf{c}$, user input $\mathbf{z}$):
$$\mathbf{x} = (\mathbf{c}, \mathbf{z})$$

Fine-tuning objective:
$$\hat{\theta} = \arg\max_\theta \sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}} \log \Pr_\theta(\mathbf{y}|\mathbf{c}, \mathbf{z})$$

- Fine-tune to accept simplified instructions (e.g. `Translate!` instead of `Translate the following sentence from English to Chinese.`).
- **Tradeoff**: overly simplified instructions during fine-tuning hurt generalization (overfitting).

**Context distillation** [Snell et al. 2022]:
- **Teacher** (full context $\mathbf{c}$) and **Student** (simplified context $\mathbf{c}'$) models, both take user input $\mathbf{z}$.
- Train student to match teacher's predictions:
  $$\hat{\theta} = \arg\min_\theta \sum_{\mathbf{x}' \in \mathcal{D}'} \text{Loss}(\Pr^t(\cdot|\cdot), \Pr_\theta^s(\cdot|\cdot), \mathbf{x}')$$
- **Loss variants**:
  - Sequence-level: $\text{Loss} = \sum_\mathbf{y} \Pr^t(\mathbf{y}|\mathbf{c}, \mathbf{z}) \log \Pr_\theta^s(\mathbf{y}|\mathbf{c}', \mathbf{z})$ (intractable).
  - Using teacher's $\hat{\mathbf{y}} = \arg\max_\mathbf{y} \log \Pr^t(\mathbf{y}|\mathbf{c}, \mathbf{z})$: $\text{Loss} = \log \Pr_\theta^s(\hat{\mathbf{y}}|\mathbf{c}', \mathbf{z})$.
  - KL divergence: $\text{Loss} = \text{KL}(P^t \| P_\theta^s)$ [Askell et al. 2021].

#### 3.3.2.2 Learning Soft Prompts for Parameter-Efficient Fine-Tuning

**Prefix fine-tuning** [Li & Liang 2021]:
- Prepend learnable vectors $\mathbf{p}_0^l ... \mathbf{p}_n^l$ at each Transformer layer $l$.
- $\mathbf{H}^l = [\mathbf{p}_0^l \mathbf{p}_1^l ... \mathbf{p}_n^l, \overline{\mathbf{H}}^l]$ where $\overline{\mathbf{H}}^l$ comes from prior layer (the last $m+1$ outputs).
- Only prefixes trained, rest frozen.
- **Additional params**: $L \times n \times d$ (layers × prefixes × dim) — much smaller than full LLM.

**Prompt tuning** [Lester et al. 2021]:
- Like prefix-tuning but **only at embedding layer** (not all layers).
- Add pseudo embeddings $\mathbf{p}_0...\mathbf{p}_n$ at start of token embedding sequence.
- Pseudo embeddings need not correspond to real tokens → "soft prompt embeddings".
- Preserves LLM architecture, easier to deploy across tasks.

**Other variants**:
- Use a sequence model (Transformer) to encode soft prompt sequence.
- **Mixing hard + soft**: $[\mathbf{p}_0 ... \mathbf{p}_n, \mathbf{q}_0 ... \mathbf{q}_{m'}, \mathbf{e}_0 ... \mathbf{e}_m]$ (soft prefix + hard prompt embeddings + user input embeddings) [Liu et al. 2023b].
- Mu et al. 2024: prompts compressed into a few pseudo tokens appended to inputs (distill prompting from teacher model).
- **Adaptor layers**: insert trainable layer, freeze rest — can be viewed as soft prompts encoding task info.

Fig 3.7 summary:
| Variant | Where soft prompt sits |
|---|---|
| (a) Soft prompts as **prefixes** | Each layer |
| (b) Soft prompts as **input embeddings** | Embedding layer only |
| (c) **Fine-tuning parts of model** | Internal layer |
| (d) **Fine-tuning the adaptor** | External adaptor |

#### 3.3.2.3 Learning Soft Prompts with Compression

Approximate long context $\mathbf{c}$ with continuous representation $\sigma$:
$$\hat{\sigma} = \arg\min_\sigma s(\hat{\mathbf{y}}, \hat{\mathbf{y}}_\sigma)$$
where $\hat{\mathbf{y}} = \arg\max \Pr(\mathbf{y}|\mathbf{c}, \mathbf{z})$, $\hat{\mathbf{y}}_\sigma = \arg\max \Pr(\mathbf{y}|\sigma, \mathbf{z})$.

- Framing as knowledge distillation (teacher=full context, student=$\sigma$):
  $$\hat{\sigma} = \arg\max_\sigma \log \Pr(\hat{\mathbf{y}}|\sigma, \mathbf{z})$$
  $$\hat{\sigma} = \arg\min_\sigma \text{KL}(\Pr(\cdot|\mathbf{c}, \mathbf{z}) \| \Pr(\cdot|\sigma, \mathbf{z}))$$
- Compressed context = **prompt embeddings** (vectors not tokens).

**Long-context issue**: teacher must handle long inputs (use efficient transformers / long-context LLMs / KV cache tricks).

**Segment-based compression** [Chevalier et al. 2023]:
- Split context into segments $\mathbf{z}^1, ..., \mathbf{z}^K$.
- Recurrently produce $\sigma^{<i+1}$ summarizing all context up to step $i$.
- At step $i$: input = $(\sigma^{<i}, \mathbf{z}^i, \langle g_1 \rangle ... \langle g_\kappa \rangle)$ where $\langle g_j \rangle$ are **summary tokens**.
- Extract hidden states at last layer for summary tokens → new $\sigma^{<i+1}$.
- RNN-like: $\sigma$ acts as memory accumulating context.

### 3.3.3 Prompt Length Reduction

Soft prompts are dense but **not interpretable** and **inflexible** (can't easily edit without re-training).

**Alternative**: shorten the **text** prompt (classic NLP text simplification).

**Methods**:
- **Heuristic deletion**: remove tokens that contribute least [Li et al. 2023c; Jiang et al. 2023b].
  - E.g. cross out redundant phrases like "across various domains, with a particular emphasis" → keeping just "on healthcare and finance".
- **Paraphrasing**: rewrite as shorter equivalent.
- **Seq2seq simplification model** trained on simplification data.
- **LLM-based simplification**: prompt LLM to simplify under length constraints.

---

## 3.4 Summary

Two main themes covered:
- **Designing/refining basic prompts** for effective problem-solving (general prompt design, ICL, CoT, decomposition, self-refinement, ensembling, RAG/tool use).
- **Automating prompt design/representation** (prompt optimization, soft prompts, length reduction).

**Historical evolution**:
- Hand-crafted features/templates in classical NLP → "proto-prompting".
- BERT era: prompts added to elicit task behavior without fine-tuning.
- GPT-3+ era: prompting became central; **prompt engineering** as research area.

**Key tension**: prompting quality depends heavily on **LLM strength**.
- Strong (commercial) LLMs: simple prompts often suffice.
- Weaker LLMs: need careful prompt design + sometimes fine-tuning.

**Related surveys cited**: ICL [Li 2023; Dong et al. 2022], CoT [Chu et al. 2023; Yu et al. 2023; Zhang et al. 2023a], efficient prompting [Chang et al. 2024], general PE [Liu et al. 2023c; Chen et al. 2023a].
