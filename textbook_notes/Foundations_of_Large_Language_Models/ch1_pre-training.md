# Chapter 1: Pre-training

> **Big idea**: Separate common components from neural NLP systems, train them on huge unlabeled data via self-supervision, then adapt to downstream tasks via fine-tuning or prompting. Killed the "train from scratch on labeled data" paradigm.

- **Historical lineage**:
    - Early DL pre-training: RNN unsupervised init, deep feedforward, autoencoders (Schmidhuber 2015).
    - Word embedding resurgence: word2vec (Mikolov 2013), GloVe (Pennington 2014).
    - Vision: ImageNet-pretrained backbones.
    - NLP pre-training boom: BERT, GPT — same idea (predict masked words from huge text).
- **General formulation**: $\mathbf{o} = g_\theta(x_0, x_1, ..., x_m)$
    - $x_0$ = special start symbol (`<s>` or `[CLS]`).
    - Output $\mathbf{o}$ varies: distribution over vocab (LM) or real-valued vector seq (encoding).
- **Two core questions**:
    - How to **optimize $\theta$** on a pre-training task (no specific downstream task assumed → must generalize).
    - How to **apply** $g_{\hat\theta}$ to downstream tasks — fine-tune $\hat\theta$ on labeled data OR prompt.

---

## 1.1 Pre-training NLP Models

### 1.1.1 Unsupervised, Supervised, Self-supervised Pre-training

| Type | Pre-train data | Pre-train objective | Pros/Cons |
|------|---------------|---------------------|-----------|
| **Unsupervised** | Unlabeled | Reconstruction cross-entropy, etc. (no task signal) | Preliminary init step; helps find better local minima + regularizes. Doesn't replace supervised stage. |
| **Supervised** | Labeled (task A) | Standard supervised loss | Straightforward; but **needs labels** which limits scalability for big models. Transfer assumes related tasks (e.g., sentiment → subjectivity). |
| **Self-supervised** | Unlabeled | Pseudo-labels generated from the data itself (e.g., mask + predict) | **Dominant paradigm**. No initial model needed (unlike self-training). Trained from scratch on supervision derived from raw text. |

- **Self-training vs self-supervised**:
    - **Self-training**: iteratively bootstrap — seed model → pseudo-labels on unlabeled → retrain (Yarowsky 1995 word sense disambig., Blum & Mitchell 1998 doc classification).
    - **Self-supervised**: supervision purely from text structure (e.g., mask a token, predict it). No seed model.
- **Why self-supervised won**: scales — supervised needs labels, unsupervised lacks task structure. Self-supervised gives task-like signal at unlimited scale.

### 1.1.2 Adapting Pre-trained Models

Two model types in NLP pre-training:
- **Sequence Encoding Models** — produce vector representations for downstream use.
- **Sequence Generation Models** — produce token sequences given context (LM, MT).

#### 1.1.2.1 Fine-tuning

- **Setup**: $\mathbf{H} = \text{Encode}_{\hat\theta}(\mathbf{x})$, then stack a classifier on top: $\text{Pr}_{\omega,\hat\theta}(\cdot|\mathbf{x}) = \text{Classify}_\omega(\mathbf{H})$.
- Combined model $F_{\omega,\hat\theta}(\cdot) = \text{Classify}_\omega(\text{Encode}_{\hat\theta}(\cdot))$, fine-tuned on labeled downstream data:
    - Update both $\omega$ and $\theta$ → outputs $\tilde\omega, \tilde\theta$.
    - **OR freeze encoder** $\hat\theta$, only train $\omega$ (cheaper, preserves pre-trained reps).
- Fine-tune data << pre-train data → cheap adaptation.

#### 1.1.2.2 Prompting

- **For generation models** (LLMs trained on next-token prediction).
- Cast downstream tasks as **text completion**:
    - E.g., for sentiment: append `"I'm ___"` and check if completion is positive-flavored word.
- **Instruction-based prompting**: include task description in text:
    ```
    Assume the polarity of a text is a label chosen from {positive, negative, neutral}.
    Identify the polarity of the input.
    Input: I love the food here. It's amazing!
    Polarity: ___
    ```
- **Zero-shot**: no examples in prompt.
- **Few-shot / in-context learning (ICL)**: include **demonstrations** (input→output pairs) in prompt to teach format/task.
- Prompting still benefits from tuning for instruction-following + alignment with human values (see Ch.4).

---

## 1.2 Self-supervised Pre-training Tasks

Three architectures: **decoder-only**, **encoder-only**, **encoder-decoder**. Discussion restricted to Transformers.

### 1.2.1 Decoder-only Pre-training

- **Architecture**: Transformer decoder with cross-attention removed → pure causal LM.
- **Objective**: maximum likelihood, log cross-entropy:
    $$\text{Loss}_\theta(x_0,...,x_m) = \sum_{i=0}^{m-1} \text{LogCrossEntropy}(\mathbf{p}^\theta_{i+1}, \mathbf{p}^{\text{gold}}_{i+1})$$
- Equivalent MLE form:
    $$\hat\theta = \arg\max_\theta \sum_{\mathbf{x} \in \mathcal{D}} \sum_{i=0}^{i-1} \log\text{Pr}_\theta(x_{i+1}|x_0,...,x_i)$$
- Used for GPT-family, foundation LLMs.

### 1.2.2 Encoder-only Pre-training

- **Problem**: encoder outputs real-valued vectors $\mathbf{H}$ — no obvious gold target.
- **Fix**: stack a Softmax on top during pre-training only:
    $$[\mathbf{p}_1^{W,\theta}, ..., \mathbf{p}_m^{W,\theta}]^T = \text{Softmax}_W(\text{Encoder}_\theta(\mathbf{x}))$$
- After pre-training, drop $W$, keep $\hat\theta$ as the encoder.
- **Critical difference from LM**: encoder sees the **whole sequence** at once — predicting any visible token is trivial → must hide some tokens to make a real task.

#### 1.2.2.1 Masked Language Modeling (MLM)

- **Core idea**: mask tokens → predict from bidirectional context (both left + right).
- **Causal LM = special case** of MLM (mask only right-context).
- **Formal**: positions $\mathcal{A}(\mathbf{x}) = \{i_1,...,i_u\}$ replaced with `[MASK]` → corrupted $\bar{\mathbf{x}}$.
    - Example: `"The early bird catches the worm"` → `"The [MASK] bird catches the [MASK]"`.
- **Objective** (only over masked positions):
    $$(\hat{W}, \hat\theta) = \arg\max_{W,\theta} \sum_{\mathbf{x}\in\mathcal{D}} \sum_{i \in \mathcal{A}(\mathbf{x})} \log\text{Pr}_i^{W,\theta}(x_i|\bar{\mathbf{x}})$$
- **Interpretation**: autoencoding — reconstruct $\mathbf{x}$ from corrupted $\bar{\mathbf{x}}$.

#### 1.2.2.2 Permuted Language Modeling

- **Motivation** (XLNet, Yang 2019):
    - MLM uses `[MASK]` only during training → **train-test mismatch**.
    - MLM predicts masked tokens **independently** of each other → misses dependencies between masked positions.
- **Idea**: pick a permutation order; predict tokens sequentially in that order (LM-style). Original token positions unchanged — only prediction order differs.
    - Standard order: $x_0 \to x_1 \to x_2 \to x_3 \to x_4$.
    - Permuted: e.g., $x_0 \to x_4 \to x_2 \to x_1 \to x_3$.
    - For $x_3$: conditions on $\mathbf{e}_0, \mathbf{e}_4, \mathbf{e}_2, \mathbf{e}_1$ → uses both left + right context.
- **Implementation**: Transformer self-attention is order-insensitive → just modify attention masks (block attention from later positions in the chosen order).
- Bridges LM (sequential) and MLM (bidirectional): combines benefits of both.

#### 1.2.2.3 Pre-training Encoders as Classifiers

Self-supervised classification tasks built from text.

- **Next Sentence Prediction (NSP)** (BERT):
    - Input: `[CLS] Sent_A [SEP] Sent_B [SEP]`.
    - Binary task: is $\text{Sent}_B$ the actual next sentence of $\text{Sent}_A$?
    - Positive = real consecutive; negative = random sentence.
    - Use $\mathbf{h}_0$ (CLS output) → Softmax → IsNext / NotNext.
    - Captures sentence-level relationships.
- **ELECTRA** (Clark 2019): **replaced token detection**.
    - **Generator** = small MLM. Masks some tokens, predicts them → may produce different tokens.
    - **Discriminator** = Transformer encoder trained to label each output position as `original` or `replaced`.
    - Example: `[CLS] The boy spent hours working on toys .` → mask "hours" + "toys" → generator outputs `decades` + `toys` → discriminator labels: original / original / .../ replaced / .../ original.
    - **Training**: jointly optimize generator (MLE) + discriminator (classification).
    - Alternative: GAN-style (harder to scale).
    - After training: discard generator, use discriminator as pre-trained model.
    - **Why efficient**: every token gives signal (not just 15% masked).

### 1.2.3 Encoder-Decoder Pre-training

- Used for seq-to-seq (MT, QA, summarization).
- Generalizes: cast **any** NLP problem as text-to-text.

#### 1.2.3.1 Masked Encoder-Decoder Pre-training

- **T5** (Raffel 2020): unified `Source Text → Target Text` framework.
    - Translation: `[CLS] Translate from Chinese to English: 你好! → <s> Hello!`
    - QA: `[CLS] Answer: when was Albert Einstein born? → <s> He was born on March 14, 1879.`
    - Even regression: `[CLS] Score the translation ... → <s> 0.81` (output the *text* "0.81").
- Benefits: one model handles all tasks; instructions in text → potential zero-shot generalization to new instructions.

**Variants of encoder-decoder pre-training tasks**:

- **Prefix language modeling**:
    - Encoder reads prefix; decoder generates rest causally.
    - Differs from causal LM: encoder processes prefix in one pass.
    - Example: `[CLS] The puppies are frolicking → <s> outside the house .`
    - Easy to create unlimited training data.
- **BERT-style MLM (encoder-decoder)** (Fig 1.4a):
    - Corrupt input with `[MASK]`, decoder predicts only the masked tokens (loss only on masked positions).
    - Shorter target → efficient.
- **Denoising autoencoding** (Fig 1.4b):
    - Corrupt input, decoder predicts the **entire** original sequence (loss over all tokens).
- **MASS-style**: predict consecutive masked tokens.
- **Sentinel/span masking** (T5):
    - Replace consecutive masked tokens with sentinel tokens `[X]`, `[Y]`, `[Z]`.
    - E.g., `[CLS] The puppies are [X] outside [Y] . → <s> [X] frolicking [Y] the house [Z]`.
- **Multi-lingual encoder-decoder**: need shared vocab covering all languages.

#### 1.2.3.2 Denoising Training (BART)

- Generalization: $\mathbf{y} = \text{Decode}_\omega(\text{Encode}_\theta(\mathbf{x}_\text{noise}))$.
- Train to reconstruct clean $\mathbf{x}$ from noisy $\mathbf{x}_\text{noise}$.
- **BART** (Lewis 2020) corruption methods:
    - **Token Masking**: replace tokens with `[MASK]`.
    - **Token Deletion**: remove tokens entirely (model must figure out **where** + **what**).
    - **Span Masking**: replace contiguous spans with single `[MASK]`. Length-0 spans = insertions. Forces model to learn **how many** tokens per span (fertility, à la MT).
    - **Sentence Reordering**: permute sentence order. E.g., `"Hard work leads to success. Success brings happiness."` ↔ swapped.
    - **Document Rotation**: rotate document so a random token becomes first → model learns to identify true start.
- Combine multiple corruptions, randomly pick one per sample.

### 1.2.4 Comparison of Pre-training Tasks

Categorize by **training objective** (not architecture, since same objective fits multiple archs):

- **Language Modeling** — autoregressive next-token prediction.
- **Masked Language Modeling** — general mask-and-predict; uses full masked sequence as context.
- **Permuted Language Modeling** — masked-prediction in arbitrary order (XLNet).
- **Discriminative Training** — classification signals (NSP, ELECTRA).
- **Denoising Autoencoding** — encoder-decoder reconstruction from corrupted input.

**Method × architecture compatibility** (from Table 1.1):

| Method | Enc | Dec | Enc-Dec | Example I/O |
|--------|-----|-----|---------|-------------|
| Causal LM | | • | • | input → next-token continuation |
| Prefix LM | | • | • | `[C] prefix` → suffix |
| Masked LM | • | | | `[C] The [M] chasing the [M]` → fill masks |
| MASS-style | | | • | masked span → fill span |
| BERT-style (E-D) | | | • | masked → masked tokens only |
| Permuted LM | • | | | predict in permuted order |
| NSP | • | | | sentence pair → IsNext? |
| Sentence Comparison | • | | | encode pair → similarity |
| Token Classification (ELECTRA-style) | • | | | sequence → orig/replaced per token |
| Token Reordering | | | • | shuffled → original |
| Token Deletion | | | • | tokens removed → original |
| Span Masking | | | • | `[CLS] The kitten [M] is [M]` → original |
| Sentinel Masking | | | • | `[CLS] The kitten [X] the [Y]` → `[X] ... [Y] ...` |
| Sentence Reordering | | | • | shuffled sentences → ordered |
| Document Rotation | | | • | rotated doc → original |

---

## 1.3 Example: BERT

### 1.3.1 The Standard Model

- **Architecture**: Transformer **encoder**.
- **Training loss**: $\text{Loss}_\text{BERT} = \text{Loss}_\text{MLM} + \text{Loss}_\text{NSP}$.

#### 1.3.1.1 Loss Functions

- Input format: `[CLS] Sent_A [SEP] Sent_B [SEP]`.
- **MLM**: pick 15% of tokens, then for each:
    - **80% → `[MASK]`** — main signal; learn to use context.
    - **10% → random token** — learn to recover from noise.
    - **10% → unchanged** — guide model to also use easy evidence; prevents over-reliance on `[MASK]` marker.
    - Loss: $\text{Loss}_\text{MLM} = -\sum_{i \in \mathcal{A}(\mathbf{x})} \log\text{Pr}_i(x_i|\bar{\mathbf{x}})$.
- **NSP**: binary classifier on $\mathbf{h}_\text{cls}$ (= $\mathbf{h}_0$, output for `[CLS]`):
    - $\text{Loss}_\text{NSP} = -\log\text{Pr}(c_\text{gold}|\mathbf{h}_\text{cls})$.
    - Training pairs: 50% consecutive (IsNext) / 50% random (NotNext).

#### 1.3.1.2 Model Setup

- **Embedding** = token + position + **segment** embeddings:
    $$\mathbf{e} = \mathbf{x} + \mathbf{e}_\text{pos} + \mathbf{e}_\text{seg}$$
    - $\mathbf{e}_\text{seg} \in \{\mathbf{e}_A, \mathbf{e}_B\}$ — which sentence the token belongs to. **New** in BERT vs vanilla Transformer.
- **Layer**: post-norm: $\text{output} = \text{LNorm}(F(\text{input}) + \text{input})$ with $F$ = self-attn or FFN.
- **Hyperparams**:
    - **Vocab size** $|V|$, **embed size** $d_e$, **hidden size** $d$, **# heads** $n_\text{head}$ ($\geq 4$ usually), **FFN size** $d_\text{ffn}$ (typ. $4d$), **depth** $L$.
- **Two standard sizes**:
    - **BERT-base**: $d=768$, $L=12$, $n_\text{head}=12$, **110M params**.
    - **BERT-large**: $d=1024$, $L=24$, $n_\text{head}=16$, **340M params**.
- **Training trick**: train on **short sequences first** (many steps) then continue on full-length sequences (efficiency).

### 1.3.2 More Training and Larger Models

- **RoBERTa** (Liu 2019):
    - More data + more compute → big gains without architectural change.
    - **Dropped NSP** — no perf loss when training scaled up.
    - Lesson: just scale simple pre-training tasks.
- **Scaling to billions**:
    - He et al. 2021: 1.5B param BERT-like.
    - Shoeybi 2019 (Megatron-style): 3.9B BERT-like, hundreds of GPUs.
    - Challenges at scale: instability, convergence, init, parallelism.

### 1.3.3 More Efficient Models

Four threads:

1. **Knowledge distillation** — train smaller student from larger teacher (Sun 2020, Jiao 2020 / TinyBERT). Distill output + intermediate hidden states.
2. **Model compression** — pruning entire layers (Fan 2019), % of weights (Sanh 2020 / DistilBERT, Chen 2020), attention heads (Michel 2019 — many heads removable w/o quality loss); **quantization** (Shen 2020 / Q-BERT — low-precision params).
3. **Dynamic networks** — adapt compute per input:
    - **Depth-adaptive**: early exit at optimal depth (Xin 2020, Zhou 2020).
    - **Length-adaptive**: skip unimportant tokens.
4. **Parameter sharing across layers** (ALBERT, Lan 2020; Dehghani 2018 Universal Transformer): one set of weights reused at every layer → fewer params + memory savings.

### 1.3.4 Multi-lingual Models

- **mBERT**: trained on 104 languages with **shared vocab** → tokens from all languages mapped to one rep space → cross-lingual transfer.
- **Cross-lingual application**: fine-tune mBERT on English-annotated data → directly classify Chinese docs.
- **XLM** (Lample & Conneau 2019): better cross-lingual model.
    - Can use CLM or MLM.
    - **Translation Language Modeling (TLM)**: pack parallel sentence pair as one sequence, mask tokens across both:
        ```
        [CLS] [MASK] 是 [MASK] 动物 。 [SEP] Whales [MASK] [MASK] . [SEP]
        ```
    - Predicting Chinese 鲸鱼 requires looking at English "Whales" → aligns reps.
    - XLM uses input embedding = token + position + **language embedding** (each token tagged with language label).
        - **Tradeoff**: language embeddings hurt code-switching handling → mBERT-style language-independent reps better for mixed text.
- **Code-switching**: mixed-language text like `周末 我们 打算 去 做 hiking, 你 想 一起 来 吗?` — handled naturally with shared vocab.
- **Scaling concerns** (Conneau 2020 / XLM-R):
    - More languages → need larger model.
    - Larger shared vocab helps with linguistic diversity.
    - Low-resource langs benefit from transfer with related high-resource langs.
    - **Interference**: performance degrades on individual languages after extended training → may need **early stopping** in pre-training.

---

## 1.4 Applying BERT Models

- Setup: $\mathbf{y} = \text{Predict}_\omega(\text{BERT}_{\hat\theta}(\mathbf{x}))$.
- Fine-tune:
    $$(\tilde\omega, \tilde\theta) = \arg\min_{\omega, \hat\theta^+} \sum_{(\mathbf{x}, \mathbf{y}_\text{gold}) \in \mathcal{D}} \text{Loss}(\mathbf{y}_{\omega, \hat\theta^+}, \mathbf{y}_\text{gold})$$
    - Init $\theta$ with $\hat\theta$; jointly update with $\omega$.

**Task patterns**:

- **Classification (single text)**:
    - Input: `[CLS] x_1 ... x_m`.
    - Use $\mathbf{h}_\text{cls}$ → classifier → label distribution.
    - Tasks: grammaticality (Warstadt 2019), sentiment (Socher 2013).
- **Classification (pair of texts)**:
    - Input: `[CLS] x_1 ... x_m [SEP] y_1 ... y_n [SEP]`.
    - Same: $\mathbf{h}_\text{cls}$ → classifier.
    - Tasks: semantic equivalence (Dolan & Brockett), entailment / NLI (Williams 2018, RTE), commonsense inference (SWAG, Zellers 2018), QA inference.
- **Regression**:
    - Same arch, output a real value (e.g., similarity score) — typically Sigmoid head.
- **Sequence labeling**:
    - One label per token: POS tagging, NER (BIO scheme: `B-ORG`, `I-ORG`, `O`).
    - Loss: $\text{Loss} = -\frac{1}{m}\sum_{i=1}^m \log p_i(\text{tag}_i)$.
    - Decoding: dynamic programming over lattice (linear complexity; Huang 2009).
- **Span prediction** (reading comprehension):
    - Input: `[CLS] query [SEP] context [SEP]`.
    - Two heads on each context position: $p_j^\text{beg}$, $p_j^\text{end}$.
    - Loss: $-\frac{1}{n}\sum_{j=1}^n (\log p_j^\text{beg} + \log p_j^\text{end})$.
    - Inference: $(\hat{j}_1, \hat{j}_2) = \arg\max_{j_1 \leq j_2} (\log p_{j_1}^\text{beg} + \log p_{j_2}^\text{end})$.
- **Encoding for encoder-decoder models**:
    - Use BERT to **initialize encoder** of an enc-dec system (e.g., NMT).
    - Optional **adapter** between BERT output and decoder input.
    - Fine-tune end-to-end on parallel data.

**Fine-tuning challenges**:
- **Catastrophic forgetting**: fine-tuning on task B may destroy task A capability.
- **Mitigations**:
    - Mix old data into fine-tuning data.
    - **Experience replay** (Rolnick 2019).
    - **Elastic weight consolidation** (Kirkpatrick 2017) — penalize changes to params important to old tasks.
    - General topic = **continual learning** (Parisi 2019, Wang 2023).

---

## 1.5 Summary

- Self-supervised pre-training on raw text → foundation models adaptable via fine-tuning or prompting.
- Worked across encoder-only, decoder-only, encoder-decoder.
- BERT exemplifies encoder-only + MLM + NSP.
- Scaling simple objectives gives the biggest wins (RoBERTa lesson, then LLMs).
- Open problems: efficient learning from small data, complex reasoning + planning.
- Later chapters: LLMs, prompting + ICL (Ch.3), fine-tuning + alignment (Ch.4).

**Key takeaway pattern**:
- **Generative tasks** (LM, prefix LM, denoising) → train decoders / encoder-decoders → flexible (any task as text-to-text).
- **Discriminative / mask-fill tasks** (MLM, NSP, ELECTRA) → train encoders → strong reps for understanding tasks.
- These are now **complementary halves** of NLP foundation modeling.
