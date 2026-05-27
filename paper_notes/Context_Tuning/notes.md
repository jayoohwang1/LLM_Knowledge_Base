# Context Tuning for In-Context Optimization

**Authors**: Jack Lu, Ryan Teehan, Zhenbang Yang, Mengye Ren (NYU Agentic Learning Lab) — Oct 2025, arXiv 2507.04221v2.

---

## TL;DR

- **Idea**: prompt/prefix tuning, but **initialize the trainable soft prompt / KV prefix from the actual few-shot demonstration pairs** (instead of random tokens or task-instruction tokens), then fine-tune the prompt by reconstructing the demos themselves.
- Frozen model weights, gradient updates on context only.
- Two variants: **CT-Prompt** (input-layer soft prompt, Prompt-Tuning-style) and **CT-KV** (per-layer K/V prefix, Prefix-Tuning-style).
- Adds two regularizers crucial to performance: **Leave-One-Out (LOO) Masking** of the demo being predicted, and **Token Dropout** on the context.
- Frames everything (TTT, CT-Prompt, CT-KV) under one umbrella called **In-Context Optimization (ICO)**: gradient-based, single-task, few-shot adaptation that uses demo pairs as targets.
- **CT-KV** has linear time complexity in $k$ (number of demos) vs quadratic for TTT and CT-Prompt; runs $\sim 2\times$ faster than TTT with comparable accuracy. Stacking **TTT + CT-KV** yields SOTA across NLP-LR / MMLU / BBH / ARC.

---

## 1. Problem setup

- **Setting**: single-task few-shot adaptation. Frozen LM $p_\phi$ with $L$ layers, $d$ hidden dim. Demo set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{k}$. Concatenated context $\mathcal{C} = [x_1; y_1; \ldots; x_k; y_k]$. Goal: answer a new query $x_q$.
- **Baselines under comparison**:
    - **ICL**: $\hat{y}_q = \arg\max_y p_\phi(y \mid [\mathcal{C}; x_q])$ — no gradient.
    - **Prompt Tuning** (Lester 2021): trainable input soft prompt $P$, optimize $P^\* = \arg\min_P \sum_i -\log p_\phi(y_i \mid [P; x_i])$.
    - **Prefix Tuning** (Li & Liang 2021): per-layer trainable KV prefixes $\Theta = \{K_j, V_j\}_{j=1}^L$, same objective with $\Theta$.
    - **TTT** (Akyürek 2024): LoRA-tune model weights $\phi$ using leave-one-out splits of demos.

---

## 2. In-Context Optimization (ICO) framework

> **Definition**: methods that minimize $\sum_{i=1}^{k} -\log p_\phi(y_i \mid [\theta_\text{context}^{(i)}; x_i])$ where $\theta_\text{context}^{(i)}$ is *derived from the demonstration pairs*.

- Prompt Tuning / Prefix Tuning prepend random-init contexts → **not** ICO.
- ICL has no gradient → **not** ICO.
- **TTT is an instance**: sets $\theta_\text{context}^{(i)} = \mathcal{C}^{-i}$ (concat of demos except $i$), optimizes $\phi$ (not the context).
- **Context Tuning**: freezes $\phi$, directly optimizes $\theta_\text{context}$.

---

## 3. Method: Context Tuning

### 3.1 Two variants

| Variant | $\theta_\text{context}$ init | What's trained | Analog |
|---|---|---|---|
| **CT-Prompt** | $P_\text{CT}$ = model's input embeddings of $\mathcal{C}$ | input-layer prompt | Prompt Tuning |
| **CT-KV** | $\Theta_\text{CT} = \{K_j, V_j\}_{j=1}^L$ = layer-wise KV cache from one forward pass on $\mathcal{C}$ | layer-wise K/V tensors | Prefix Tuning |

- **Key conceptual move**: instead of random init, do **one forward pass on the demos to get the natural soft prompt / KV cache, then gradient-tune that**.
    - Combines ICL's strength (model already "extracts" task info from demos) with prompt-tuning's strength (gradient refinement of a compact context).

### 3.2 Design choice #1 — Leave-One-Out (LOO) Masking

- When training on pair $(x_i, y_i)$, mask out the slice of $\theta_\text{context}$ that corresponds to $(x_i, y_i)$:
$$\theta_\text{context}^{(i)} = \begin{cases} P_\text{CT}^{-i} & \text{CT-Prompt} \\ \Theta_\text{CT}^{-i} & \text{CT-KV} \end{cases}$$
- **Why needed**: otherwise the model just retrieves $y_i$ from the prefix → no learning, just memorization.
- **Vs TTT's LOO**: TTT drops the demo from context and updates *weights*. Here, weights are frozen and we drop the demo from the *context vectors themselves*.
- Implemented via attention mask (set positions to 0 in `past_key_values_attention_mask`).
- **Exception**: ARC works better *without* LOO (very few demo pairs per task; masking one removes too much signal).

### 3.3 Design choice #2 — Token Dropout

- Randomly zero out tokens in $\theta_\text{context}^{(i)}$ with prob `TokenDrop` each step.
- Needed because CT-* prepends *many more* trainable tokens than typical prompt-tuning → more overfitting risk.
- Loss is expectation over stochastic dropout masks.
- Rates used: 0.05 (NLP-LR), 0.1 (MMLU/BBH/ARC).

### 3.4 Final objective

$$\text{CT-Prompt:}\quad P_\text{CT}^\* = \arg\min_{P_\text{CT}} \sum_{i=1}^{k} -\log p_\phi\!\left(y_i \mid [\,\text{TokenDrop}(P_\text{CT}^{-i})\, ; x_i\,]\right)$$

$$\text{CT-KV:}\quad \Theta_\text{CT}^\* = \arg\min_{\Theta_\text{CT}} \sum_{i=1}^{k} -\log p_\phi\!\left(y_i \mid [\,\text{TokenDrop}(\Theta_\text{CT}^{-i})\, ; x_i\,]\right)$$

Inference: just use $P_\text{CT}^\*$ or $\Theta_\text{CT}^\*$ as the context for $x_q$.

### 3.5 TTT + CT-KV (stacked)

- Run TTT first (update $\phi$). Then run CT-KV on top of the TTT'd model.
- Idea: TTT adapts *weights*, CT-KV adapts *context* — orthogonal, so they compose.
- Best overall numbers.

---

## 4. Time complexity (Appendix A)

Per-head self-attention scaling, $n$ query tokens, $p$ trainable prefix tokens, $k$ demos of length $\ell$:

| Method | $L_Q$ | $L_K$ | Per-head cost | Cost in $k$ |
|---|---|---|---|---|
| TTT | $n+p$ | $n+p$ | $O((n+p)^2)$ | $O((k\ell)^2)$ |
| CT-Prompt | $n+p$ | $n+p$ | $O((n+p)^2)$ | $O((k\ell)^2)$ |
| **CT-KV** | $n$ | $n+p$ | $O(n(n+p))$ | $O(k\ell^2)$ |

- **Why CT-KV is cheaper**: the trainable tokens are *only* keys/values (not queries) — they don't generate Q, so $L_Q$ doesn't grow with $p$. Linear in $k$ instead of quadratic.
- Empirically: CT-KV is $\sim 2\times$ faster than TTT and $\sim 1.5\times$ faster than CT-Prompt.

---

## 5. Experiments

### 5.1 Setup

- **Benchmarks**:
    - **NLP-LR**: 26 low-resource NLP tasks from CrossFit + UnifiedQA (MetaICL split), $k = 16$, multiple choice.
    - **MMLU**: 57 subjects, $k = 16$, multiple choice.
    - **BBH** (BIG-Bench Hard): 27 tasks, $k = 10$, QA, includes task instructions.
    - **ARC** (Abstraction and Reasoning Corpus): 400 tasks, fewer than 4 demos each, grid-transformation QA.
- **Models**: GPT-2-Large (NLP-LR), Llama3-8B (BBH), Llama3.2-3B (MMLU), Llama3.2-1B fine-tuned on ARC train split (per Akyürek 2024 / Franzen 2024).
- **Scaling**: also evaluated on Mistral-NeMo-12B, DeepSeek-R1-Distill-Qwen 14B/32B, Qwen3-14B/32B for BBH.
- Multiple choice scored by lowest-loss option; QA scored by exact match.
- Means/stds over 5 seeds (different demo samples), except ARC (fixed demos).
- Hyperparams swept: LR $\in \{3e-4, 1e-3, 3e-3\}$, TokenDrop $\in \{0, 0.05, 0.1\}$.
- CT-KV typically uses 200 iters (NLP-LR/ARC), 20 (MMLU), 16 (BBH); TTT+CT-KV uses just 25/5/8/25 iters of CT-KV on top.

### 5.2 Main results (Table 1)

| Method | NLP-LR | MMLU | BBH | ARC | Time (NLP-LR) |
|---|---|---|---|---|---|
| Zero-shot | 34.9 | 35.8 | 40.9 | 1.0 | 0 |
| ICL | 35.6 | 41.2 | 50.4 | 13.3 | 0 |
| LoRA | 42.8 | 40.1 | 51.7 | 13.5 | 156 |
| DoRA | 42.9 | 40.3 | 52.6 | 13.0 | 161 |
| Prompt Tuning (m=32) | 41.4 | 39.2 | 50.8 | 12.0 | 137 |
| Prefix Tuning (m=32) | 42.0 | 39.9 | 52.7 | 9.3 | 123 |
| TTT | 44.1 | 43.6 | 57.8 | **23.8** | 342 |
| **CT-Prompt** | 43.2 | 43.6 | 56.3 | 22.5 | 228 |
| **CT-KV** | 44.2 | 43.7 | 57.9 | **23.8** | **145** |
| **TTT+CT-KV** | **47.6** | **44.1** | **58.2** | **25.8** | 372 |

- **CT-KV beats every prompt/prefix baseline and LoRA/DoRA by wide margin**, while being competitive with TTT at half the cost.
- Simply increasing $m$ in Prompt/Prefix Tuning to match CT-* parameter count doesn't help → init *content* matters more than capacity.
- **Demo-initialized prompts lower variance across seeds** (more stable than random init).
- CT-KV (44.2) > MetaICL (43.3) on NLP-LR — single-task inference-time tuning beats multi-task meta-trained ICL.

### 5.3 Scaling up (Table 2, BBH)

| Method | Mistral12B | DeepSeek14B | DeepSeek32B | Qwen14B | Qwen32B |
|---|---|---|---|---|---|
| ICL | 51.3 | 55.2 | 63.5 | 60.8 | 63.1 |
| Prefix Tuning (m=32) | 37.0 | 49.8 | 53.0 | 49.3 | 56.4 |
| **CT-KV** | **57.4** | **58.4** | **66.7** | **64.2** | **69.1** |

- Prefix Tuning *hurts* large models — bad random init. CT-KV consistently helps.

### 5.4 Robustness (Figure 4)

- **Varying $k$**: CT-KV gains scale better with $k$ than ICL or Prefix Tuning. ICL plateaus on NLP-LR.
- **Label noise**: corrupt MC answers with prob $p$. CT-KV best across all corruption rates up to 75%.

### 5.5 Ablations (Table 3, CT-KV)

| Variant | NLP-LR | MMLU | BBH | ARC |
|---|---|---|---|---|
| Neither | 41.0 | 40.2 | 51.4 | 21.0 |
| No LOO | 42.6 | 41.5 | 54.4 | **23.8** |
| No TokenDrop | 43.9 | 42.7 | 55.3 | 21.0 |
| Both | **44.2** | **43.7** | **57.9** | 22.5 |

- LOO matters more than TokenDrop on most tasks; both help.
- ARC prefers no-LOO (too few demos).

### 5.6 Why does CT-KV outperform ICL? (Sec 5.9 — most interesting analysis)

**Two-stage view of ICL**: (1) demo tokens → KV cache via forward pass; (2) attend to that cache when generating answer.

**Diagnostic — "demo pair retrieval" experiment**:
- Put $k$ demos in context, then *query with one of the demos themselves*. Correct answer is literally in context, so ICL should be ~100%.
- Result (Table 5): NLP-LR 81.9 / MMLU 84.1 / BBH 89.3 / ARC 22.6 — **far from perfect**.
- Implication: **single-forward-pass KV cache does NOT fully encode the demo info**. There's a lossy compression bottleneck.
- CT-KV directly addresses this by gradient-refining the KV cache so it actually solves each demo (with LOO so it can't cheat).

> Important takeaway: **ICL underperforms not because the demos lack info, but because the model's one-shot encoding of those demos into KV is lossy.** Gradient descent in KV space fixes this.

### 5.7 Qualitative (Figs 5–7)

- ARC confusion matrix (Table 4): CT-KV solves 51 tasks ICL fails on; CT-KV fails on 9 ICL solves → CT-KV can overfit to particular demo features (e.g., became biased to predicting 3x4 grids because 2 of 3 demos had 3x4 answers).
- ARC iteration sweep: predictions visibly improve over training iters (gradual color/shape correction).
- BBH word-sort: model fixes word ordering / restores omitted words across iterations.

### 5.8 Parameter-efficient CT-KV variants (Appendix D)

- **CT-V**: freeze key prefixes $\Theta_K$, only train value prefixes $\Theta_V$ (halves trainable params).
    - Motivated by Fisher info: $\hat F_V \gg \hat F_K$ across all datasets (e.g., NLP-LR: $8.3e\text{-}8$ vs $1.4e\text{-}8$; MMLU: $1.4e\text{-}6$ vs $2.8e\text{-}8$) → values carry more task-relevant signal.
    - Loses only ~0.2–0.4 acc.
- **CT-Prefix**: average $\Theta_\text{CT}$ across token positions, perturb with $\epsilon \sim \mathcal{N}(0, 0.02)$ to form an $m$-token Prefix-Tuning-sized prefix, condition on it. Matches Prefix Tuning param count, still beats it by ~2–14 pts.

| Method | NLP-LR | MMLU | BBH | ARC |
|---|---|---|---|---|
| Prefix Tuning (m=32) | 42.0 | 39.9 | 52.7 | 9.3 |
| CT-Prefix | 44.0 | 42.6 | 55.9 | 22.8 |
| CT-V | 44.0 | 43.5 | 57.5 | 23.5 |
| CT-KV | 44.2 | 43.7 | 57.9 | 23.8 |

### 5.9 Memory (Table 12)

- CT-KV uses *less* memory than TTT, similar to Prefix Tuning (m=# demo). Adds <20% overhead vs Prefix Tuning (m=32) but much higher accuracy.

### 5.10 Parameter counts (Table 11, thousands)

- CT-Prompt: 578 / 2160 / 3656 / 2743 (NLP-LR/MMLU/BBH/ARC)
- CT-KV: 41634 / 40327 / 58501 / 21944
- TTT (LoRA): 47186 / 89915 / 157286 / 84935
- Even CT-KV (largest CT variant) trains ~half the params of TTT.

---

## 6. Code-level details (from `train.py`)

- Backbone in released code: GPT-2-Large, frozen (`requires_grad = False`).
- Init: `past_key_values = model(demon_input_ids).past_key_values`, cast to bfloat16.
- All layers' K and V become `program_params` — full layer-wise tuning, no LoRA on top.
- Optimizer: AdamW, cosine schedule, no warmup, weight_decay=0, grad clip 1.0, lr=1e-3 default, 200 epochs default.
- LOO via `batch_past_key_values_attention_mask[batch_i, start:end] = 0` over the slice corresponding to demo $i$.
- TokenDrop: `drop_mask = (rand > token_dropout)`, applied multiplicatively to the attention mask.
- Position IDs are reconstructed to continue from `demon_input_ids.shape[1]` (so generation tokens see correct positions despite KV swap).
- Inference uses bf16, sdpa attention, picks MC option with min cross-entropy.

```python
# Heart of context_tuning() — LOO + dropout + KV-as-params
for p in program_params: p.requires_grad = True   # all K, V layers
...
for batch_i, idx in enumerate(pair_example_idx):
    start, end = demon_start_idxs[idx], demon_start_idxs[idx+1]
    batch_past_key_values_attention_mask[batch_i, start:end] = 0   # LOO
if token_dropout != 0.0:
    drop_mask = (torch.rand_like(...) > token_dropout).float()
    batch_past_key_values_attention_mask = (mask * drop_mask).long()
model_out = model(inputs_embeds=..., attention_mask=...,
                  past_key_values=batch_past_key_values, position_ids=...)
loss = model_out.loss * bs / batch_size
accelerator.backward(loss)
```

---

## 7. Relation to adjacent work

- **Prompt Tuning / Prefix Tuning / P-Tuning v2**: same machinery, but random / instruction / uniform / MLP init. CT swaps init for the natural demo embedding/KV. The paper's Table 7 shows token-based init > uniform/MLP for the baselines too, but CT-* still tops them.
- **Instruction Prompt Tuning** (Singhal 2023): prepends curated demos *as input tokens* to a learned soft prompt. CT goes further: uses demos *to initialize* the learnable prompt itself rather than concatenating.
- **TTT** (Akyürek 2024): orthogonal — modifies weights. CT-KV is the context-side analog. **Composable: TTT then CT-KV.**
- **MetaICL** (Min 2022a): multi-task meta-training to improve ICL. CT-KV beats it on NLP-LR with no meta-training — single-task inference-time.
- **In-Context Vectors** (Liu 2024b), **StreamAdapter** (Muhtar 2024): related lines of editing/steering internal representations; CT does it via straight gradient descent on KV.
- **LoRA / Rank-stabilized LoRA / DoRA**: weight-side PEFT baselines; CT-KV outperforms all at similar or lower cost.
- **Theoretical ICL-as-gradient-descent** (Dai 2023, Deutch 2024): suggests ICL is *implicit* GD; this paper essentially says *let's just do explicit GD on the same context representation* — and it works better, partly because implicit GD is constrained to one forward step.
- **ICL critiques** (Min 2022b, Jang 2024): show ICL relies on surface patterns. CT's demo-retrieval diagnostic gives a complementary failure mode: KV is lossy.

---

## 8. Limitations / future work

- **Overfitting to demo idiosyncrasies** (e.g., grid shape bias on ARC). TokenDrop helps but isn't enough.
- $\theta_\text{context}$ size scales with $k \cdot \ell$ — at very large $k$, the prefix itself becomes large. Future: KV cache compression methods (Devoto 2024, Ge 2024, Liu 2024a) on the init.
- Per-task optimization at inference time — still slower than zero-shot/ICL despite being faster than TTT.
- No exploration of mixing CT with retrieval-style or generation-side adaptation.

---

## 9. What's novel / interesting

- **Reframes prompt/prefix tuning init as the bottleneck, not the optimizer or architecture.** Tiny conceptual change (init from demos) → large empirical gain.
- **Unified "In-Context Optimization" framing** cleanly slots TTT, CT-Prompt, CT-KV together; makes the weight-vs-context axis explicit.
- **Linear-time KV prefix tuning** via the simple observation that KV prefixes don't generate queries → $L_Q$ stays at $n$. Often overlooked vs methods that prepend trainable *tokens* (which produce queries too).
- **Demo-retrieval diagnostic** is a clean, falsifiable probe of "is the KV cache really encoding the demos?" — answer: no, even for trivially-cued recall.
- **Stacking with TTT works** — context-tuning and weight-tuning compose without interference. Suggests a broader design space (TTT + CT-Prompt? CT + RAG?).
- **Fisher analysis** picking out values $\gg$ keys is a nice practical heuristic for trimming PEFT prefix params in half.
- The fact that **random-init Prefix Tuning actively *hurts* large models (Mistral12B, Qwen32B) while CT-KV monotonically helps** is a striking failure mode of vanilla prefix tuning that init-from-demos repairs.

---

## Code Walkthrough

Repo layout (`raw_data/papers/Context_Tuning/context-tuning/`):

| File | Role |
|---|---|
| `train.py` | entry point, inference-time per-task loop, `context_tuning()` optimizer, `inference_time_optimization()` eval |
| `data_utils.py` | tokenization, demo formatting, `EvalDataset` (test pairs) and `GSDataset` (gradient-step pairs over demos), collate fns |
| `hr_to_lr.json` | NLP-LR split — 61 train / 26 test / 4 unseen-domain test tasks (MetaICL split) |
| `config/<split>.json` | other splits (class→class, qa→qa, etc.); only `hr_to_lr.json` is wired by default |
| `config/tasks/<task>.json` | per-task `{"task_type": "multi-choice"|"classification", "options": [...]}` used to decide accuracy vs macro-F1 |
| `requirements.txt` | `torch==2.4.0`, `accelerate==1.1.1`, `transformers==4.47.1` |
| `README.md` | three example launch commands: zero-shot, ICL, CT-KV |

**Released variant**: only **CT-KV** is implemented in this repo (no CT-Prompt, no TTT). Backbone is hard-coded to `openai-community/gpt2-large` (NLP-LR experiments).

### 1. Top-level control flow

`train.py:main()` does:
1. Build `Accelerator` (bf16, sdpa attention) + GPT-2-Large with all dropouts forced to 0 and weights frozen (`p.requires_grad = False`).
2. Build `EvalDataset` from `hr_to_lr.json["test"]` — for each of 26 tasks, load `{task}_16_{eval_split}_train.jsonl` (16 demos) and `{task}_16_{eval_split}_test.jsonl` (queries).
3. Call `inference_time_optimization(...)` once → iterates over tasks, for each task **(a)** runs a forward pass on the concatenated demos to seed the KV cache, **(b)** optionally fine-tunes that cache via `context_tuning()` for `epochs` iters, **(c)** scores every test pair × every MC option by per-sequence cross-entropy and picks the min-loss option.

The `--epochs` flag is overloaded: `epochs > 0` → CT-KV; `epochs == 0` → plain ICL; `--zero_shot` → drops the KV entirely.

### 2. Data pipeline (`data_utils.py`)

#### 2.1 Per-pair tokenization & delimiter
- `EvalDataset.delimiter_token_id = tokenize(" ", tokenizer)` — a single **space token** as the input/output separator (not a newline, despite the comment).
- `parse_pairs()` formats each demo `(x_i, y_i)` and appends them sequentially:

```python
# data_utils.py:63
for pair_i, pair in enumerate(pairs):
    input_input_ids = tokenize(pair['input'], tokenizer)
    output_input_ids = tokenize(pair['output'], tokenizer)
    if len(input_input_ids) > 256 - len(output_input_ids):
        input_input_ids = input_input_ids[: 256 - len(output_input_ids)]
    input_input_ids = torch.cat([input_input_ids, delimiter_token_id])
    if pair_i != 0:
        input_input_ids = torch.cat([delimiter_token_id, input_input_ids])
```

- Each pair is **capped at 256 tokens** (truncating the input, never the label). A leading space delimiter is prepended for all but the first pair → final stream looks like `x1 ␣ y1 ␣ x2 ␣ y2 …`.
- **Total sequence cap = 1024** (`if len(input_ids) > 1024: return None`) — examples that overflow are dropped (with `patience=100` allowance per task before giving up).

#### 2.2 Demo span bookkeeping for LOO
- `demon_start_idxs` records the **start position of each demo pair** in the concatenated demo stream. Crucially:

```python
# data_utils.py:91
if pair_i < len(pairs) - 2:
    demon_start_idxs.append(demon_start_idxs[-1] + len(input_ids))
```

- The `- 2` is because `parse_pairs` is also called with `demonstrations + [test_pair]`; the last "pair" is the test query and is split off into `gen_input_ids`. So `demon_start_idxs` ends up indexing only the 16 demo pairs inside the KV cache.
- These indices are what `context_tuning()` later uses to zero out the appropriate KV-cache slice for LOO masking.

#### 2.3 Two datasets, two collate functions
- **`EvalDataset`** (used at scoring time): expands every test pair into one entry per MC option, each tagged with `task / test_idx / option / correct_option`. Stores `demon_input_ids` (the concatenated demo stream that seeds the KV) and `gen_input_ids` (the test `x_q ␣ y_q^{option}` to score).
- **`GSDataset`** (used inside `context_tuning()`): one entry per demo pair, with `example_idx` so the trainer knows which KV slice to LOO-mask.

```python
# data_utils.py:359  (GSDataset.format)
input_input_ids = torch.cat([input_input_ids, self.delimiter_token_id])
input_input_ids = torch.cat([self.delimiter_token_id, input_input_ids])
input_ids = torch.cat([input_input_ids, output_input_ids])
...
label_ids = torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype)
label_ids = torch.cat([label_ids, output_input_ids])
```

- **Label masking**: `-100` on the prompt (`x_i` + delimiters), the output `y_i` tokens are the only ones contributing to the loss. Standard HF causal-LM convention.
- **Length filter** here uses `1024 - past_kv_len`, i.e. accounts for KV prefix length: drops demo pairs that wouldn't fit alongside the KV cache.

### 3. KV initialization — the actual "context init from demos"

This is **section 3.1** of the paper realized in 3 lines:

```python
# train.py:131
# initialize kv
with accelerator.autocast():
    past_key_values = model(input_ids=demon_input_ids, output_hidden_states=True).past_key_values
```

- One forward pass over `demon_input_ids` (the concatenated `x_1 ␣ y_1 ␣ … x_k ␣ y_k`) using HuggingFace's standard `use_cache` machinery.
- **HF `past_key_values`** is a tuple of `L` `(K, V)` pairs; each tensor has shape `(batch=1, n_heads, seq_len, head_dim)`. For GPT-2-Large that's `L=36`, `n_heads=20`, `head_dim=64`, so per-task KV is `36 × 2 × 20 × seq_len × 64` bf16 floats.
- Then cast to bf16 and passed to `context_tuning()`:

```python
# train.py:140
past_key_values = tuple(
    (layer_k.to(torch.bfloat16), layer_v.to(torch.bfloat16))
    for layer_k, layer_v in past_key_values
)
```

**No `nn.Parameter` registration, no `nn.Embedding`** — the trainable "prefix" is literally just the list of `Tensor`s returned by the model's own forward pass, with `requires_grad = True` flipped on:

```python
# train.py:289
program_params = []
for layer_k, layer_v in past_key_values:
    program_params.append(layer_k)
    program_params.append(layer_v)
...
for p in program_params:
    p.requires_grad = True
```

- All `2L = 72` tensors enter the optimizer as a single param group. This is **full per-layer KV tuning** — not the parameter-efficient CT-V or CT-Prefix variants from Appendix D (those aren't implemented).
- Param count = $2 L \cdot H \cdot \ell \cdot d_\text{head}$ where $\ell$ = concatenated demo length. Scales with demo content length, not a fixed prefix size.

### 4. The `context_tuning()` training loop

```python
# train.py:267
@torch.enable_grad()
def context_tuning(...):
```

Notable bits:
- `@torch.enable_grad()` decorator — needed because the outer `inference_time_optimization()` is wrapped in `@torch.no_grad()`. Without this override, gradients wouldn't flow.
- `accelerator.no_sync(model)` wraps the whole call → in DDP this skips inter-rank gradient sync, because there's nothing to sync (model weights are frozen and the per-task KV is local to each rank's task).
- Initial `.detach().clone()` of every K/V tensor — defensive, makes the params leaf tensors independent of the autograd graph from the seeding forward pass.

**Optimizer / schedule** (train.py:322–328):
```python
optim = torch.optim.AdamW(optimizer_grouped_params, weight_decay=0.0)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=epochs)
```
- AdamW, `lr=1e-3`, `weight_decay=0`, **no warmup**, **cosine decay** over `epochs` (which = number of *outer epochs*, not steps).
- Grad clip 1.0.
- One `optim.step()` per *epoch* (not per batch — see below).

**Loop structure** (train.py:337–390):
```python
for _ in range(epochs):
    for batch in gs_loader:
        ...
        loss = model_out.loss * bs / batch_size
        accelerator.backward(loss)
    accelerator.clip_grad_norm_(all_params, 1.0)
    optim.step()
    scheduler.step()
    optim.zero_grad()
```

- **Gradient accumulation across the entire epoch**: the inner `for batch` loop calls `backward()` repeatedly but never zeros grads — `optim.step()` only fires once per outer epoch.
- The `loss * bs / batch_size` scaling makes this equivalent to one big SGD step on the mean over *all* demos (HF cross-entropy already averages within a micro-batch, so we re-weight by `bs / batch_size` to recover an unweighted mean across the full demo set when batches are unequal).
- Default `ct_batch_size = 16` and `num_demonstrations = 16` → typically one micro-batch per epoch, so the accumulation is mostly a no-op for the released config. But it's the right shape if you bumped demo count past available memory.

### 5. LOO masking & token dropout — the two regularizers, in code

```python
# train.py:346
batch_past_key_values_attention_mask = torch.ones(
    (bs, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64
)
batch_past_key_values = tuple(
    (layer_k.expand(bs, -1, -1, -1), layer_v.expand(bs, -1, -1, -1))
    for layer_k, layer_v in past_key_values
)

# leave one out
for batch_i, idx in enumerate(pair_example_idx):
    start = demon_start_idxs[idx]
    end = demon_start_idxs[idx + 1] if idx < len(demon_start_idxs) - 1 else demon_input_ids_len
    batch_past_key_values_attention_mask[batch_i, start:end] = 0

# token dropout
if token_dropout != 0.0:
    drop_mask = (torch.rand_like(batch_past_key_values_attention_mask, dtype=torch.float) > token_dropout).float()
    batch_past_key_values_attention_mask = (batch_past_key_values_attention_mask * drop_mask).long()
```

- **`.expand(bs, -1, -1, -1)`** broadcasts the single set of K/V tensors across the batch dimension *without copy*. Cheap, but it means LOO can't be done by modifying the K/V tensors themselves (every batch row shares the same memory) — instead it's done by **zeroing entries in the attention mask** so positions become unattendable.
- LOO span = `[demon_start_idxs[idx], demon_start_idxs[idx+1])`. Note for the last demo it falls through to `demon_input_ids_len`, but recall from §2.2 that `demon_start_idxs` only contains `k-1` entries (the last `<` was `len(pairs) - 2`), so the last demo's end is the full demo stream length. **Subtle**: this means LOO for the final demo masks from its start all the way to end-of-context, which is correct as long as nothing else lives after it.
- **Token dropout = Bernoulli mask on attention positions**, not on the KV values themselves. So a "dropped" token still occupies its position in the KV but is invisible from attention. This is exactly attention masking, applied to the KV prefix. The paper's term "Token Dropout" is implemented as **position-level attention dropout** in code — slightly different framing than e.g. `nn.Dropout` on embeddings.
- Token dropout is applied **on top of** the LOO mask via multiplication, so dropped positions inside the LOO span just stay zeroed (idempotent).

### 6. Position IDs — a non-obvious correctness detail

The HF model is asked to process the query `x_i ␣ y_i` *with the KV cache prefilled*. Naively the model would assign positions `[0, len(query))` to the query tokens, colliding with the KV positions. The code fixes this:

```python
# train.py:364  (training)
position_ids = torch.zeros((bs, pair_input_ids.shape[1]), device=device, dtype=torch.int64)
new_lens = pair_attention_mask.sum(dim=1)
for task_position_ids, new_len in zip(position_ids, new_lens):
    new_positions = torch.tensor(range(demon_input_ids_len, demon_input_ids_len + new_len), device=device, dtype=dtype)
    task_position_ids[:new_len] = new_positions
```

- Query tokens get **positions `[demon_input_ids_len, demon_input_ids_len + new_len)`**, continuing from where the KV cache "ends".
- Padding positions are set to 0 (they get masked out anyway).
- The eval path (`train.py:199`) does the same trick with `position_start = demon_input_ids.shape[1]`.
- **Why this matters**: GPT-2 uses learned absolute position embeddings. If the query thought it was at position 0 while attending to KVs at positions $[0, \ell)$, you'd get scrambled positional info and a big drop. Easy thing to forget; the paper doesn't discuss it.

### 7. Forward call & loss

```python
# train.py:372
model_kwargs = {
    "inputs_embeds": pair_inputs_embeds,
    "attention_mask": pair_attention_mask,    # [KV mask | query mask], concatenated
    "labels": pair_label_ids,
    "use_cache": True,
    "past_key_values": batch_past_key_values, # the trainable tensors
    "position_ids": position_ids,
}
model_out = model(**model_kwargs)
loss = model_out.loss * bs / batch_size
```

- Uses `inputs_embeds` (computed manually via `embed_tokens(pair_input_ids)`), not `input_ids`. Probably because the original codebase derived from a CT-Prompt-capable version where you'd want to substitute soft-prompt embeddings; here it's a no-op but harmless.
- The attention mask is the concatenation `[KV mask (with LOO+dropout) | query mask]`, length `KV_len + query_len`. HF's GPT-2 attention treats prefix positions where mask=0 as fully masked.
- `labels=pair_label_ids` triggers HF's built-in shifted cross-entropy with `-100` ignore.
- Gradients flow: loss → query tokens' logits → attention over KV → **the K and V tensors themselves**. The model weights are frozen, but the KV inputs to every attention layer are leaf tensors with `requires_grad=True`, so their grads accumulate via backprop through the attention softmax.

### 8. Eval / inference path

`inference_time_optimization()` after the optional `context_tuning()`:

```python
# train.py:182
batch_past_key_values = [
    (
        layer_k.detach().clone().expand(bs, *layer_k.shape[1:]),
        layer_v.detach().clone().expand(bs, *layer_v.shape[1:]),
    )
    for layer_k, layer_v in past_key_values
]
batch_past_key_values_attention_mask = torch.ones(
    (bs, batch_past_key_values[0][0].shape[2]), ...
)
```

- At test time, **all KV positions are unmasked** — no LOO (LOO is a training-only regularizer), no dropout.
- Per test pair × per MC option: build `gen_input_ids = x_q ␣ y_q^{option}`, label-mask the `x_q` portion with `-100`, run a single forward pass, compute mean loss over the `y_q^{option}` tokens via `get_individual_loss()`.
- **`get_individual_loss`** is a per-example unreduced cross-entropy with `-100` ignored, returning one scalar per sequence (train.py:42).
- **Picking the MC answer**:

```python
# train.py:247
lowest_loss = float('inf')
chosen_option = None
for x in task_test_outs:
    if x[0] < lowest_loss:
        lowest_loss = x[0]
        chosen_option = x[3]
```

The option with minimum mean per-token loss wins. Standard MC-LM scoring.

- **Metric**: `compute_macrof1_or_accuracy()` — macro-F1 for classification tasks, plain accuracy for multi-choice. Determined per task by reading `config/tasks/{task}.json`'s `"task_type"` field (`"classification"` vs anything else, e.g. `"multi-choice"`).

### 9. Hyperparameters & config schema

From `argparse` defaults in `train.py:404`:

| Flag | Default | Meaning |
|---|---|---|
| `--experiment_name` | (required) | accelerator project dir |
| `--config_file` | `./hr_to_lr.json` | task split |
| `--data_dir` | `./metaicl-data/data` | MetaICL HF dataset clone |
| `--num_demonstrations` | 16 | $k$ (if >16, pulls extra demos from test split) |
| `--batch_size` | 16 | eval batch size |
| `--eval_split` | `'87'` | seed used in MetaICL filename `{task}_16_{seed}_train.jsonl`; the paper averages over `{13,21,42,87,100}` |
| `--eval_ratio` | 1.0 | fraction of test pairs to score |
| `--zero_shot` | False | drop KV entirely |
| `--epochs` | 0 | CT-KV iters; 0 = pure ICL |
| `--lr` | 1e-3 | AdamW lr for KV |
| `--ct_batch_size` | 16 | inner batch in `context_tuning` (one full batch typically) |
| `--token_dropout` | 0.05 | Bernoulli rate for KV-position dropout |
| `--seed` | 0 | added to `accelerator.process_index` for `set_seed` |

The split JSON schema (`hr_to_lr.json`) is just `{"train": [...], "test": [...], "unseen_domain_test": [...]}`. Only `"test"` is read at eval time.

Per-task metadata (`config/tasks/{task}.json`):
```json
{"task_type": "multi-choice", "options": null}
{"task_type": "classification", "options": ["equivalent", "not_equivalent"]}
```
Only `task_type` is actually used downstream (to switch macro-F1 vs accuracy). The README's CT-KV command reproduces `0.4470` on `eval_split=87`, matching the paper's CT-KV NLP-LR row (44.2).

### 10. Implementation tricks & gotchas worth flagging

- **All dropouts forced to zero in the model config** (`attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0, summary_first_dropout=0.0`) — both for inference determinism *and* so the KV cache produced by the seeding forward is reproducible. Token dropout on the KV mask is the only stochasticity.
- **sdpa attention** (`_attn_implementation='sdpa'`) — uses PyTorch's `scaled_dot_product_attention`, which handles `past_key_values` + custom attention mask cleanly. No explicit FlashAttention integration, no gradient checkpointing, no hooks.
- **No `nn.Parameter`**: `program_params` is a Python list of raw tensors with `requires_grad=True`. AdamW accepts this. Slightly unusual style — most PEFT codebases register a `nn.Module` for the prefix.
- **Per-epoch (not per-batch) optimizer step** with grad accumulation across the demo batches → effectively full-batch GD over the demos in each "epoch". `epochs=200` therefore = 200 gradient updates (not 200 passes × N steps).
- **HF GPT-2 `past_key_values` shape gotcha**: it's `(batch, n_heads, seq_len, head_dim)`, indexed as `tensor.shape[2]` for seq length — this is what you see in `past_key_values[0][0].shape[2]`.
- **`gc.collect()` + `torch.cuda.empty_cache()`** called aggressively after every task and every batch — KV caches per task can be hundreds of MB; without manual cleanup they pile up.
- **No CT-Prompt or TTT here**, despite both being central to the paper. The README only documents CT-KV. To replicate CT-Prompt you'd swap the trainable param from per-layer K/V tensors to the input embeddings of `demon_input_ids` and feed via `inputs_embeds=[soft_prompt; query_embeds]`; the loop structure would otherwise be identical.
- **Multi-GPU correctness**: `set_seed(args.seed + accelerator.process_index)` differs per rank — so the *demo sample* itself can differ across ranks (different shuffling of test pairs). The KV is per-task per-rank and never synced (`accelerator.no_sync(model)`). With current code, multi-GPU only parallelizes across test pairs within a task implicitly via `Accelerator`'s data parallel (though `inference_time_optimization` doesn't seem to actually shard tasks across ranks — every rank loops over every task). Worth a closer look if you scale this up.
- **Subtle difference from paper**: the paper describes Token Dropout as "randomly zeroing tokens in $\theta_\text{context}^{(i)}$" — code zeros *attention positions*, leaving the underlying KV tensors untouched. Functionally equivalent for the loss but means the trained K/V values are not directly damped by dropout (only their effective gradient).
- **GPT-2's BOS-stripping branch** (`if not isinstance(tokenizer, GPT2TokenizerFast): input_ids = input_ids[1:]`) — GPT-2's tokenizer doesn't prepend BOS, so the strip is skipped. The branch exists to support other tokenizers if you wired one up.
- **`output_hidden_states=True`** is passed to the KV-seeding forward but never used. Dead arg, possibly leftover from earlier CT-Prompt experiments where input embeddings would be reused.

