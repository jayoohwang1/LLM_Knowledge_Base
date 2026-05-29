# Adapting Language Models to Compress Contexts (AutoCompressors)

> Chevalier, Wettig, Ajith, Chen — Princeton NLP, **EMNLP 2023**. arXiv 2305.14788.
> Teach a pretrained LM to compress long contexts into a few **summary vectors** (soft prompts), recursively, and reuse them as a cheap extended context.

---

## TL;DR

- **Core idea:** fine-tune an LM so it can emit a small set of **summary vectors** (continuous embeddings, i.e. soft prompts) that compress a text segment. Feed them as a prefix to subsequent segments → effectively extends context window very cheaply.
- **Compression rate:** e.g. 2048 tokens → **50** summary vectors ≈ **40× compression**.
- **Recurrence:** process a long doc segment-by-segment; **accumulate** (concatenate) summary vectors from *all* prior segments.
- **Training:** unsupervised next-token LM loss over the whole doc, conditioned on accumulated summaries. No distillation, no extra labels.
- **Built on RMT** (Recurrent Memory Transformer, Bulatov 2022) but adds **summary accumulation** + **randomized segmenting** + **stop-gradients**.
- **Wins:** better long-range PPL than RMT, fits 6K–30K-token training on a single A100, summary vectors beat plain-text few-shot ICL on 8/11 tasks, and pre-computed summaries give Pareto-optimal retrieval/re-ranking.

---

## 1. Motivation / Problem

- Transformer LMs limited by **finite context window** + **quadratic attention cost** over long docs.
- Two goals: **efficiency** (cheaper inference over long text) + **versatility** (process more data in new ways).
- **Pitch:** compress text into summary vectors that are
  - **one-to-two orders of magnitude shorter** than the plaintext,
  - **produced by the model itself** as a function of the input (unlike Wingate's prompt compression which re-optimizes a soft prompt per context),
  - **dual purpose:** (1) extend context window with minimal overhead, (2) **pre-compute + cache** summaries for corpora so inference over them is cheap.

### Related work positioning

| Line | Relation |
|---|---|
| **Soft prompts** (Lester 2021), **prefix tuning** (Li & Liang 2021) | Summary vectors *are* soft prompts, but **predicted from input** rather than learned per-task. |
| **Prompt compression** (Wingate 2022) | Distills context into a soft prompt $\sigma$ via KL to $p_{LM}(y\mid x)$; but **re-runs optimization per context**, no transfer. AutoCompressor learns $\sigma=f(x)$. |
| **Context distillation** (Askell 2021, Snell, Mu 2023) | Distill instructions into student / KV prefixes. AutoCompressor compresses *any* context incl. long docs into compact soft prompts. |
| **Long-range Transformers** (Transformer-XL, Sparse, Performer, RMT, CoLT5, kNN-LM) | Most need expensive **train-from-scratch** or deviate from pretrained init. AutoCompressor is a cheap fine-tune of a pretrained model. |
| **RMT** (Bulatov 2022) | Direct ancestor (segment recurrence via memory tokens). **Key diff: summary accumulation** (concat summaries from *all* segments, not just previous one). |

---

## 2. Method

Architecture overview (Fig 1): input randomly split into segments → each segment produces summary vectors → those are **prepended as soft prompts** to *all* subsequent segments, recursively.

### 2.1 Summary vectors

- Extend base vocab with $\kappa$ special **summary tokens** `<Sum>_1 ... <Sum>_κ`, with $\kappa$ newly-initialized input embeddings.
- Appending `<Sum>_1 ... <Sum>_κ` to a segment **signals the model to output summary vectors** of the preceding context at those output positions.
- **Summary vector = output hidden state** at a summary-token position (length-$\kappa$ soft prompt). Passed to next segment.
- **Init trick (OPT):** initialize summary-token embeddings with the pretrained **EOS `</s>`** embedding → better than random.
- Intuition: pretrained embedding space spans thousands of dims → high capacity to pass info; a soft prompt can **interpolate** many token embeddings → more abstract than a single discrete token.

### 2.2 Summary accumulation (the key novelty vs RMT)

- Long doc split into $S_1, \dots, S_n$, processed sequentially.
- **RMT:** prepend only previous segment's compressed memory $\sigma_{i-1}$ to $S_i$.
- **AutoCompressors:** **concatenate** summaries from *all* preceding segments:
  $$\sigma_{<i} = \text{CONCAT}(\sigma_1, \dots, \sigma_{i-1})$$
  and prepend $\sigma_{<i}$ to $S_i$.
- Length of $\sigma_{<i}$ is $(i-1)\kappa$ → **grows linearly** with document length.
- Gives a **direct information pathway** between segment $i$ and *every* preceding segment.

### 2.3 Positional embeddings

- **Absolute PE (OPT):** do **not** add positional embeddings to summary tokens `<Sum>` nor to summary vectors $\sigma$.
  - Lets summary tokens reuse the same position embeddings as context tokens → scale to arbitrary number of compression steps during training.
  - Order of summary tokens still preserved via their **separate token embeddings**.
- **Relative PE (RoPE, Llama):** apply RoPE to summary tokens/vectors **without modification**.

### 2.4 Training objective

- For segment $S_i = (x^i_1, \dots, x^i_{m_i})$, condition on accumulated summaries $\sigma_{<i}$ and predict next tokens. Minimize cross-entropy over the **entire** document:
  $$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{n}\sum_{t=1}^{m_i} \log p(x^i_t \mid x^i_1, \dots, x^i_{t-1}, \sigma_{<i})$$
  - $N$ = total tokens.
  - **Retains** base LM ability on $S_1$ (no summaries) **and** incentivizes storing useful info in summaries so later segments predict better.
- **No knowledge distillation** (unlike Wingate): the base LM has limited context so can't serve as teacher over docs longer than its window.

### 2.5 Randomized segmenting

- Randomly vary segment lengths $m_i$ during training (subject to each segment fitting the context window).
- Lets the model compress **variable-length** contexts → robust at eval with fixed-length or even much shorter segments. (Ablation Fig 2 / Table 6: helps short-segment compression.)

### 2.6 BPTT + stop-gradients

- Backprop-through-time across segments + **gradient checkpointing** per segment to shrink the compute graph.
- **Stop-gradients:** compute & **cache** summary vectors, then **detach (stop gradient) after 2 compression steps** (analogous to caching past attention in Transformer-XL).
  - Assumption: to learn to store useful info in $S_i$, it suffices to predict tokens in the adjacent $S_{i+1}$.
  - Fig 2 confirms **no PPL penalty** vs full BPTT, while **reducing GPU memory** (Table 2: e.g. 75 GB vs OOM for OPT-2.7B at 30K).

---

## 3. Experiments & Results

Base models: **OPT-1.3B / 2.7B** (fine-tuned on 2B Pile tokens), **Llama-2-7B** (15B RedPajama tokens). Single **A100 80GB**. $\kappa=50$ default.

### 3.1 Long-range language modeling

**8K-token setup** (§4.1): segments of 2048, fix final segment, compress previous $n$. Baselines:
1. **OPT-2.7B fine-tuned** (capped at 2048, pretraining limit).
2. **Extended Full Attention (FA):** extend PE to 4096 (init positions [2049..4096] from [1..2048]); can't go beyond 4096 (memory).
3. **RMT-2.7B** ($\kappa=50$).

| Finding | Detail |
|---|---|
| AutoCompressor benefits from long contexts | PPL improves up to **6144 context tokens**; consistently **beats RMT**. |
| Benefits from **shorter** sequences than seen in training | unlike RMT (which doesn't). |
| Compression is **efficient** | 6144 tokens → only $50\times3=150$ summary vectors. |
| FA wins on 4096 but loses on short / very long | trade-off; FA needs +2048 real tokens whereas AC attends to ≤150 soft prompts. |
| OOD gap | AC vs FA gap widens out-of-domain → summaries generalize slightly less than attention heads. |

**Table 1 (held-out PPL on 2048 tokens, OPT-2.7B):** in-domain AC 6.14 (128 ctx) → **5.93** (6144 ctx) vs RMT 6.42 → 6.01; FA best at 2048 ctx (5.94). Fine-tuned OPT baseline w/o context: 6.28 in-domain / 8.53 OOD.

**30K-token setup** (§4.2): train on 30720 tokens, **20 compression steps**, Books3 (Gutenberg = OOD).

**Table 2:** AC learns to use full **28K** tokens to reduce PPL; RMT does **not** improve from 14K→28K → **summary accumulation captures long-range deps**. AC also uses **less memory** (stop-grad): AC-2.7B 75 GB vs RMT-2.7B OOM; AC-1.3B 38 GB vs RMT 54 GB.

**Llama-2-7B (§4.3, Table 3):** freeze model, train only summary embeddings + **LoRA** attention weights. 6144 ctx → 150 summaries ≈ Extended-FA with 512 plaintext tokens; preserves PPL when compressing short contexts. (FA still wins with longer real contexts — room to improve summary quality.)

### 3.2 Compressing demonstrations for ICL (§5)

- Compress 1/2/3 segments of demonstrations into 50/100/150 summary vectors (Llama-2-7B AC). Up to ~90 demos. Contextual calibration + class-balanced sampling.
- **Table 4 (11 tasks):** summary vectors **beat zero-shot** everywhere; beat plain-text **150-token** ICL on **8/11**; beat even **750-token** plain ICL on **8 tasks** (AG News, SST-2, BoolQ, WiC, WSC, CB, COPA, MultiRC).
- Summary vectors = strong cheap substitute for plaintext demos (more accuracy, less inference cost).
- OPT-2.7B AC > RMT on 8/11 (Table 12) → accumulation helps ICL; RMT doesn't benefit from multiple compression steps.
- Caveat: fine-tuned Llama AC has worse zero-shot on some tasks (domain mismatch w/ Llama pretraining corpus).

### 3.3 Compressing retrieval corpora (§6)

- **Pre-compute** summary vectors for whole corpora, store cheaply, retrieve & fuse for cheap inference.
- **Retrieval-augmented LM (Fused Summaries):** concat summary vectors of top-$k$ retrieved passages (least→most relevant), fusion-in-decoder style. Compared to REPLUG (ensembling, $k$ forward passes) & Fused Passages.
  - **Table 5:** Fused Summaries (50-tok passages) > Fused Passages & REPLUG. **512-tok→50-vec Fused Summaries top-10 beats REPLUG top-2 (512-tok) with 1.7× throughput.** REPLUG top-10 still better (room to improve summary quality).
  - Storage: compressing 10B-token domains → ~5 TB/domain in fp16; vs 51 TB to store raw transformer outputs, 3276 TB for all attention states.
- **Unsupervised passage re-ranking** (§6.2, NQ test): score passages by likelihood of query conditioned on passage summary (prompt: "Passage: {p_i}. Please write a question based on this passage."). Pre-compute summaries for 21M Wikipedia passages.
  - **Fig 4:** AC lands on the **Pareto front** of Recall@20 vs re-ranking throughput — summaries retain enough info to judge relevance while being far cheaper.

---

## 4. Ablations (Fig 2)

Trained OPT-2.7B without each component:
- **Randomized segmenting:** better compression of *short* segments; still helps multi-segment.
- **Summary accumulation:** helps PPL beyond one compressed segment (expected).
- **Stop-gradients:** **no PPL impact**, but reduces GPU memory → free lunch.
- $\kappa$ sweep (20/50/70/100, Table 7): **PPL does not monotonically improve** with more summary vectors; **$\kappa=50$ best overall**. Hypothesis: more summary vectors may need bigger training budget.
- **Token-level (Fig 5):** summaries improve PPL across all 2048 positions; FA better at sequence start, AC best toward the end → summaries capture long-range deps.
- **Qualitative (App D):** summaries contain salient info (names, dates); model can reason over them.

---

## 5. Limitations

1. Only tested up to **OPT-2.7B / Llama-2-7B** — unknown at larger scale (bigger summary dim → maybe more info/vector).
2. Summaries **ignore some info** accessible via full attention; PPL doesn't always improve with more summary vectors. Training signal may be **limited** because the pretrained model is already very good at predicting from current-segment plaintext.
3. Summary accumulation is still **quadratic** in #segments (but far lower constant than full attention). Future: combine many summaries more efficiently.

---

## 6. Implementation

Repo: `AutoCompressors/`. HF `transformers==4.34.0`, PyTorch 2.1, flash-attn 2.3.5.

### 6.1 Model: `auto_compressor.py`

- **`AutoCompressorMixin`** — turns any `AutoModelForCausalLM` into an AutoCompressor. Concrete classes:
  - **`OPTAutoCompressorModel(AutoCompressorMixin, OPTForCausalLM)`** (alias `AutoCompressorModel`).
  - **`LlamaAutoCompressorModel(AutoCompressorMixin, LlamaForCausalLM)`** — uses `modeling_flash_llama.LlamaForCausalLM`.
- **`SummaryConfig`** dataclass — tracks current sequence layout for positional embedding (`softprompt_length`, `past_key_values_softprompt_length`, `summary_length`); `reset()` after each forward.
- **`setup_autocompressor(config)`** (called in subclass `__init__`):
  - requires `config.summary_length`.
  - if `summary_length > 0`: creates `self.embed_summary = nn.Embedding(summary_length, hidden_dim)`.
  - **Init:** copies the **EOS token embedding** into every summary-token embedding row:
    ```python
    self.embed_summary.weight.data[:,:] = input_embeds.weight[config.eos_token_id]
    ```
- **Config flags added at runtime** (`train.py` lines 158–160): `config.summary_length`, `config.accumulate_summary`, `config.segment_gradient_checkpointing`.

#### Summary-token embeds (per forward)

Summary token ids are just `arange(summary_length)` → embedded via `embed_summary`, expanded to batch:
```python
summary_token_ids = torch.arange(self.config.summary_length, ...).unsqueeze(0).expand(bsz, -1)
summary_token_embeds = self.embed_summary(summary_token_ids)
```

#### `forward_segment(...)` — one segment pass

- Builds the segment input by concatenating along the sequence dim:
  - **first segment / no cached softprompt:** `[softprompt, segment_embeds, summary_token_embeds]`.
  - **cached softprompt in past_key_values:** `[segment_embeds, summary_token_embeds]` (softprompt already in KV; attention mask gets `ones` for `past_key_values_softprompt_length`).
- Sets `self.summary_config` lengths, runs `self.model(inputs_embeds=..., attention_mask=..., past_key_values=...)`.
- **Slices the output hidden states** into three regions:
  ```python
  segment_last_hiddens = outputs.last_hidden_state[:, softprompt_length : total_length - summary_length]
  new_softprompt        = outputs.last_hidden_state[:, total_length - summary_length :]   # = summary vectors
  ```
  → **`new_softprompt` (the summary vectors) is literally the hidden state at the trailing summary-token positions.**
- Optional **`torch.utils.checkpoint.checkpoint(...)`** wrapping (gradient checkpointing) when `segment_gradient_checkpointing and training and not last_step`.

#### `forward(...)` — full multi-segment pass

- New args beyond standard HF: **`segment_lengths`** (int or list), **`softprompt`** (precomputed summary vectors to prepend), **`output_softprompt`** (whether the final segment also emits summaries).
- **`past_key_values`** is overloaded as a dict `{"past_key_values": ..., "softprompt": ...}` so cached summaries survive across `.generate()` steps.
- If `past_key_values is None` → **split into segments** and loop:
  ```python
  inputs_embeds_list = torch.split(inputs_embeds, segment_lengths, dim=1)
  # all segments except last get summary tokens appended; last one only if output_softprompt
  ```
- **The accumulation logic** (lines 230–233):
  ```python
  if self.config.accumulate_summary:
      softprompt = torch.cat([softprompt, new_softprompt], dim=1)   # CONCAT all summaries
  elif new_softprompt.size(1) > 0:
      softprompt = new_softprompt                                    # RMT-style: replace
  ```
  This single branch is exactly **summary accumulation vs RMT memory replacement**.
- After segment 0, `past_key_values=None` (no caching across training segments).
- Final loss = standard shifted cross-entropy over concatenated `last_hiddens` via `self.lm_head`.
- Returns **`CausalACOutputWithPast`** with extra field **`.softprompt`** (the final summary vectors).

#### Positional embedding override (OPT)

- **`OPTLearnedPositionalEmbeddingWithPadding`** replaces `model.decoder.embed_positions`.
  - Uses `summary_config` to **cut the softprompt + summary regions out** of the cumulative position count, then re-inserts `<pad>`-mapped (zero) positions for them → **summary tokens/vectors get no real position**, matching the paper.
- Llama (`LlamaAutoCompressorModel`) needs no PE override (RoPE applied as-is); just overrides `get_past_key_values_len` for the flash-attn KV layout.

#### Two ways to get summaries (README)

1. **Explicit:** `out = model(input_ids, output_softprompt=True); sv = out.softprompt`; later `model(..., softprompt=sv)`.
2. **Implicit:** `model(input_ids, segment_lengths=[...])` — auto-generates summaries after each segment and prepends to the next (for long inputs exceeding max position).

Generation: `model.generate(prompt, softprompt=summary_vectors, ...)` via `prepare_inputs_for_generation` passing `softprompt` + `segment_lengths`.

### 6.2 Training: `train.py` + `SubstepTrainer` (`substep_trainer.py`)

- `train.py`: standard HF setup; picks `LlamaAutoCompressorModel` if `"llama"` in model name, else OPT. Optionally extends PE (`max_position_embeddings`) by repeating the last `max_pos` rows. Optionally wraps with **PEFT LoRA**.
- **`SubstepTrainer(BaseTrainer)`** implements **BPTT with stop-gradients**:
  - **`training_step`** loops over `training_substeps`; each calls **`training_substep`** which:
    - runs `model(..., softprompt=softprompt, segment_lengths=..., output_softprompt=True)`,
    - `loss.backward()` (gradients accumulate across substeps),
    - **`softprompt = out.softprompt.detach()`** → **this is the stop-gradient** between substeps (summaries carry forward as values, not through the graph).
  - **`segment_input(inputs, substep)`**: slices the full sequence into `training_substeps` equal chunks via `torch.linspace`; within each substep, `random_segment_lengths(...)` draws `segments_per_substep` random lengths (via `torch.multinomial` breakpoints, min length enforced so each fits `max_position_embeddings`) → **randomized segmenting**. If `--randomize_substeps` off, uses fixed `segment_lengths`.
  - `compute_loss` (eval only): same loop, carries `softprompt` across substeps without detach-per-step; logs per-segment `nll`/`acc`.
- **`DataCollator`** pads with `bos_token_id`, masks labels with `-100`.

### 6.3 Hyperparameters (`run/train.sh`, `run/train_llama.sh`)

| Param | OPT-2.7B (`train.sh`) | Llama-2-7B (`train_llama.sh`) |
|---|---|---|
| `summary_length` ($\kappa$) | 50 | 50 |
| `accumulate_summary` | true | true |
| `randomize_substeps` | true | true |
| `segments_per_substep` | 2 | 2 |
| `training_substeps` | 2 | 2 (→ 2×2 = 4 segments, stop-grad after 2) |
| total batch | 16 | 32 (per-device 2) |
| lr | 2e-5 | 8e-4 |
| warmup | 1000 | 5000 |
| precision | bf16 | bf16 |
| LoRA | — (full fine-tune) | **r=16, α=16, dropout=0.05**, target `q/k/v/o_proj`, `modules_to_save=embed_summary` |
| train data | Pile `Books3/Github/FreeLaw/Wikipedia` 0.5B-6K-opt | RedPajama-combined-15B-6K-llama |
| epochs | 1 | 1 |

- **Note:** with `training_substeps=2`, `segments_per_substep=2` → 4 compression segments per sequence; stop-gradient kicks in after each 2-segment substep (matches "stop gradients after 2 compression steps").
- **Llama freezes base weights** (LoRA) and only trains summary embeddings + LoRA adapters; OPT does full fine-tuning.
- `fast_attention` flag (OPT only, experimental) patches `scaled_dot_product_attention` via `fast_attention.patch_opt`; flash-attn used natively for Llama.

### 6.4 Pretrained checkpoints (HF hub, `princeton-nlp/`)

`AutoCompressor-Llama-2-7b-6k`, `AutoCompressor-2.7b-6k`, `AutoCompressor-2.7b-30k`, `AutoCompressor-1.3b-30k`, plus `RMT-*` and `FullAttention-*` baselines. All AC checkpoints: $\kappa=50$, accumulation + randomized segmenting + stop-gradient ✔.

---

## Related Notes

- **[Recurrent Memory Transformer](../RMT/RMT.md)** — Direct ancestor. AutoCompressors *is* RMT with two changes: (1) the recurrent state is a set of **soft-prompt summary vectors** (output hidden states at `<Sum>` tokens) rather than RMT's read/write **memory tokens**, and (2) summaries are **accumulated** (concatenated across all prior segments) instead of overwriting a single fixed-size memory state. Both train with segment recurrence + truncated BPTT.
- **[Artificial Hippocampus Networks (AHN)](../AHN/AHN.md)** — Same goal (compress out-of-segment context into a compact memory) but AHN compresses the **KV cache** via an RNN and keeps a *lossless sliding window* of recent KV, whereas AutoCompressors discards exact token access entirely and reasons only over lossy summary vectors. AHN trains only ~0.4% of params on a frozen base via distillation; AutoCompressors-Llama similarly freezes the base (LoRA) but optimizes a next-token objective.
- **[Compress and Attend Transformer (CAT)](../CAT/CAT.md)** — Shares the "compress spans of context, then condition on the compressed reps" idea, but CAT compresses **fixed-size chunks in parallel** with a bidirectional compressor and lets the decoder **attend to all chunk reps causally** (no recurrence/accumulation of soft prompts). CAT explicitly benchmarks against recursive-compression methods in this lineage.
