# Recurrent Memory Transformer (RMT)

> Bulatov, Kuratov, Burtsev. NeurIPS 2022. arXiv:2207.06881. Code: `github.com/booydar/LM-RMT` (later merged into recurrent-memory-transformer).

**TL;DR**: add a handful of special **`[mem]` tokens** to the input/output of a *plain, unmodified* Transformer, split a long sequence into segments, and **pass the memory token outputs of one segment into the memory token inputs of the next**. This makes the Transformer segment-level **recurrent**. Trained with **BPTT** through the memory representations. No changes to the attention mechanism or model internals — memory lives entirely in the input/output sequence.

---

## 1. Motivation / Problem

- **Two limitations of vanilla Transformers**, both addressed here:
  1. **Global + local info crammed into the same element-wise representations.** Self-attention builds context-aware token reps, but global features get **"blurred"** across all positions → hard to access. Want **dedicated** storage for global info.
  2. **Quadratic self-attention cost** in sequence length $\to$ poor scaling to long inputs.
- **Goal**: a *general-purpose* memory mechanism that
  - is **architecture-agnostic** (works on any Transformer from the family, encoder/decoder/enc-dec),
  - removes the input-length limit via **recurrence over segments**,
  - **theoretically supports infinite length** (practically bounded by memory capacity / access efficiency).
- **Design philosophy**: do *not* redesign attention (unlike Longformer, BigBird, Linformer, Performer). Instead reuse the **Memory Transformer** idea (Burtsev 2020) of special memory tokens + add **segment-level recurrence** (Transformer-XL style) + **depth-recurrent** memory processing (Staircase Transformer style).

---

## 2. Memory Token Mechanism

### Special memory tokens
- Input sequence augmented with $m$ special **`[mem]`** tokens (real-valued learnable vectors, processed like ordinary tokens — no attention changes).
- **Placed at BOTH ends** of the segment token reps $H_\tau^0$:
$$\tilde H_\tau^0 = [\,H_\tau^{mem} \circ H_\tau^0 \circ H_\tau^{mem}\,]$$
  $$\tilde H_\tau^N = \mathrm{Transformer}(\tilde H_\tau^0), \qquad [\,H_\tau^{read} \circ H_\tau^N \circ H_\tau^{write}\,] := \tilde H_\tau^N$$
  - $N$ = number of Transformer layers, $\tau$ = segment index, $\circ$ = concat.
- **Why both ends?** (key design subtlety for **decoder-only / causal** models):
  - **Leading group = read memory**: lets sequence tokens attend to memory states produced by the *previous* segment.
  - **Trailing group = write memory**: with a **causal mask**, only tokens at the *end* of the sequence can attend back over the whole segment → these collect/summarize the segment. Memory at the *start* couldn't gather info from later tokens under causal masking.
  - So: start tokens **read**, end tokens **write**. (For bidirectional encoders one position suffices, but the two-ended design is universal.)
- **Causal mask applied only to the real input tokens** — within a read/write block all memory tokens can attend to everything in the block (Fig 6d setup).

### Segment-level recurrence
- Long input split into segments processed **sequentially**.
- Carry memory across segments:
$$H_{\tau+1}^{mem} := H_\tau^{write}, \qquad \tilde H_{\tau+1}^0 = [\,H_{\tau+1}^{mem} \circ H_{\tau+1}^0 \circ H_{\tau+1}^{mem}\,]$$
  - i.e. the **write** memory output of segment $\tau$ becomes the **read** memory input (and is also copied to the write slot) of segment $\tau{+}1$.
- **First segment** initialized from the **learnable** `[mem]` parameter vectors.
- **Recurrence is over global memory tokens only** — backbone Transformer untouched.

### How it differs from Transformer-XL (architectural)
| | Transformer-XL | RMT |
|---|---|---|
| What's carried | **Cached hidden states** of last $m$ positions, **per layer** ($m \times N$ vectors) | **$m$ memory vectors** (per segment) |
| How carried | Concatenated to KV as extra context, **stop-gradient** (`SG`) on cache | Memory output → memory input of next seg, **gradients flow through** |
| Recurrence depth | Per-layer same-depth reuse | Memory reprocessed by full stack each segment → effectively **$\tau \times N$ deep** |
| Effective context | Bounded by network depth (cache only reaches back $N$ layers) | **Not depth-limited** (true recurrence) |
| Model changes | Modified attention (relative pos enc) | **None** — only input/output sequence |
| Memory cost | Large ($m\times N$ states) | Small ($m$ vectors) |

Transformer-XL update for reference:
$$\tilde H_\tau^{n-1} = [\,SG(M_{-m:}^{n-1}) \circ H_\tau^{n-1}\,], \quad H_\tau^n = TL(Q_\tau^n,K_\tau^n,V_\tau^n)$$

---

## 3. Training (BPTT)

- Trained with **Backpropagation Through Time** over the memory recurrence.
- **Unlike Transformer-XL, gradients are NOT stopped between segments** — they flow through the memory token reps to previous segments (see Fig 1/2: gradient arrows go back through memory).
- **BPTT unroll depth = hyperparameter** ("number of previous segments to backpropagate through"), varied **0 → 4** in experiments.
  - **BPTT-0**: no gradient between segments (still works surprisingly well on algorithmic tasks).
  - Larger unroll = **lower perplexity / better task accuracy**, but **costly**: lots of GPU RAM, training **instabilities** and **OOM** for larger memory sizes with deep unrolls.
  - Mitigation suggested: **gradient checkpointing**.
- For algorithmic (enc-dec-style) tasks loss is computed **over target segments only** (no loss on the input copy before the start-to-generate token).

---

## 4. Experiments

### Setup
- Implemented **on top of the original Transformer-XL codebase**. Baselines = vanilla Transformer + Transformer-XL (reproduced close to original).
- **WikiText-103** (word-level LM): 16-layer, 10 heads, 410 hidden, 2100 FF; Adam, linear LR from 0.00025, 200k steps. Vocab 267,735.
- **enwik8** (char-level LM): 12-layer, 8 heads, 512 hidden, 2048 FF; 400k steps. Vocab 204.
- **Baseline** = Tr-XL with memory size 0.

### Algorithmic tasks (synthetic, char-level, decoder-only, pre-generated)
- **Copy**: reproduce input twice after a start token.
- **Reverse**: output input in reverse order.
- **Associative Retrieval**: $N$ key-value pairs given; one key selected → produce its value.
- **Quadratic equations**: given equation + solution-with-discriminant + answer; generate solution & answer (only answer scored).
- All split into segments, processed sequentially; same data across all experiments.

### Language modeling / NLP
- **WikiText-103**: input context 150 tokens (also a harder setting: **50** tokens → more segments/recurrent steps).
- **enwik8**: 512 chars (harder: 128 chars).
- **Hyperpartisan news** (long-text classification): instead of pretraining from scratch, **bolt recurrent memory (size 10) onto already-pretrained** BERT-base, RoBERTa-base, DeBERTa-base, T5-base and fine-tune. Augments 500 input tokens. (Plus SCROLLS-style tasks like Contract-NLI, QuALITY, QASPER in the code repo.)

---

## 5. Results

### Algorithmic tasks (Fig 3, 4)
- **Single segment**: Baseline, Tr-XL, RMT all solve Copy/Reverse perfectly (no recurrence needed).
- **As #segments grows**:
  - **RMT solves Copy & Reverse perfectly** across all tested segment counts; **outperforms Tr-XL** once memory size < total #tokens.
  - Tr-XL mean accuracy drops up to 0.2 at ≤6 segments and **plunges toward baseline at 9 segments**.
  - **Associative Retrieval**: RMT solves it up to 4 segments (Tr-XL close behind); at 5 segments RMT dips slightly, Tr-XL rises.
- **Fig 4 (Copy, 360 src+tgt tokens)**: Tr-XL degrades to baseline as #segments grows; **RMT stays perfect up to 9 segments**. Fixed memory (60 tokens), growing seq length to copy: Tr-XL fails quickly; **RMT degrades only after ~720 tokens**.
- **Quadratic equations (Table 1)**: Baseline (no segmentation) = 0.99 (upper bound). With 6 segments: **RMT 0.99 ± 0.002**, **Tr-XL only 0.93 ± 0.02**.
- Note: even **RMT BPTT-0** (no cross-segment gradient) solves these algorithmic tasks.

### Language modeling
- **WikiText-103, segment len 150** (Table 2, lower PPL better):
  - Tr-XL (paper) 24.0; their Tr-XL (mem 150) **24.12**.
  - **RMT BPTT-3, mem 10 → 25.04** (beats Baseline 29.95 and MemTr 29.63 by a large margin; close to Tr-XL with far less memory).
  - **RMT with mem 10 / 25 ≈ Tr-XL with mem 75**. RMT uses small memory **more effectively** (Tr-XL stores $m\times N = 10\times16$ vectors; RMT stores only $m$).
  - **Best overall = Tr-XL + RMT combined: 23.99** (Tr-XL cache = short-term memory, RMT memory = long-term).
- **WikiText-103, harder segment len 50**: more recurrent steps. **RMT mem 1 ≈ Tr-XL mem 10** (RMT compresses prior context better). Deeper BPTT unrolls in early segments **drastically improve** scores → confirms importance of BPTT.
- **enwik8**: RMT mem 5 ≈ Tr-XL mem 40 → again RMT uses smaller memory more effectively.
- **Fig 5**: (a) deeper BPTT unrolls → lower PPL for RMT (analogous to bigger cache for Tr-XL); (b) recurrence helps RMT beat Tr-XL at equal memory sizes; gains from increasing mem 5→50 with a single mem token **fade** → there's an **optimal memory size**, more doesn't help much.

### Hyperpartisan news (Table 3, F1)
- Adding **10 memory tokens + recurrence to pretrained models** lets a 512-token model encode **up to 2000 tokens**, improving F1 for most.
- **RoBERTa-base + RMT → SOTA** on the benchmark, beating BigBird/Longformer/ETC variants that use **4096** input (≥2× longer raw input).

### Interpretability (attention maps, Fig 6)
- Sequence tokens preceded by **read memory** (top-left), followed by **write memory** (bottom-right).
- **Copy**: diagonal = token→token; off-diagonal = **writing sequence tokens to memory in order**.
- **Reverse**: model learns to write the sequence to memory in **reversed order**.
- Identifiable ops: **read-from-mem**, **write-to-mem**, **rewrite mem→mem** (used to retain older info across more segments).
- **Tr-XL contrast**: reading from cache appears as diagonals; because it stores info **in the sequence reps themselves**, it must **mix** previous- and current-segment reps in the same hidden states → degrades with more segments (e.g. Reverse, 4 segments, mem 6 → only 0.8 acc, mixing multiple symbols per cached state). RMT (same small mem) compresses the whole segment and hits **1.0**.

---

## 6. Comparison to Transformer-XL (summary)
- **RMT ≥ Tr-XL** on LM for **smaller memory sizes**; **RMT >> Tr-XL** on long-context algorithmic tasks.
- RMT stores **compact summaries** in dedicated tokens vs Tr-XL storing **raw per-layer hidden states** (which forces mixing of global/local info).
- RMT recurrence is **not depth-limited**; Tr-XL effective context capped by network depth.
- **They're complementary**: Tr-XL cache (short-term) + RMT memory (long-term) = best WikiText-103 result.
- Bonus finding: **adding memory tokens to Tr-XL itself improves it**.

---

## 7. Limitations
- **BPTT is expensive**: deep unrolls need lots of GPU RAM; **training instabilities** and **OOM** for larger memory + deeper unroll.
- **Sequential segment processing** → less parallelizable than full-attention over the whole sequence (inherent to all segment-recurrent models).
- **Optimal memory size exists** — beyond it, adding memory tokens barely helps; capacity ultimately bounds the "infinite length" claim.
- Practical infinite-length limited by **memory capacity** and **read/update efficiency**.

---

## 8. Implementation

Two implementations live in the repo:
- **`base.py` (`RMTBaseModel`) + `conditional_generation.py` etc.** — the *original* Tr-XL-codebase-derived style: memory carried by **overwriting positions** in `inputs_embeds`, uses real `[mem]` token IDs added to the tokenizer, `pad_and_segment`, `bptt_depth`, optional **memory layers** (`memory_layers.py`, `horizontal_memory.py`) and **memory-layer sharing**. (Encoder-decoder path raises `NotImplementedError` for `bptt_depth != -1`.)
- **`modeling_rmt/language_modeling.py`** — the **clean modern decoder-only** implementation (`MemoryCell` + `RecurrentWrapper`); this is the canonical one to study. Detailed below.

### `MemoryCell` — wraps one segment forward pass
File: `modeling_rmt/language_modeling.py`. Wraps any HF causal LM (`base_model`).

```python
class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens):
        self.model = base_model
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        memory_dim = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        # learnable mem vectors, init scaled by embedding std
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))
        self.read_memory_position  = range(num_mem_tokens)        # leading block
        self.write_memory_position = range(-num_mem_tokens, 0)    # trailing block
```

- **Memory is a learnable embedding-space parameter** `(num_mem_tokens, dim)`, init `~N(0, embed_std)`. **No new vocab IDs** — memory is injected as `inputs_embeds`, not token IDs.
- **`set_memory(input_shape)`**: `self.memory.repeat(batch, 1, 1)` → initial state for segment 0.

**Inserting memory tokens (prepend + append)** — `process_input`:
```python
inputs_embeds = self.model.get_input_embeddings()(input_ids)
inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)  # [mem | tokens | mem]
seg_kwargs['input_ids'] = None
seg_kwargs['inputs_embeds'] = inputs_embeds
seg_kwargs['output_hidden_states'] = True
```
- Same `memory_state` tensor placed at **both ends** (read slot = leading, write slot = trailing), matching the paper's $[\,H^{mem} \circ H^0 \circ H^{mem}\,]$.
- **`pad_attention_mask`**: wraps the real mask with `1`s for the $2m$ memory positions so mem tokens are always attended.

**Forward (one segment)**:
```python
def forward(self, input_ids, memory_state=None, **kwargs):
    if memory_state is None:
        memory_state = self.set_memory(input_ids.shape)
    seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
    out = self.model(**seg_kwargs)
    out, new_memory_state = self.process_output(out, **kwargs)
    return out, new_memory_state
```

**Extracting outputs + new memory** — `process_output`:
```python
# new memory = LAST layer hidden states at the WRITE (trailing) positions
memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
# logits/hidden_states sliced back to the REAL tokens only (strip both mem blocks)
out['logits'] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]
out['hidden_states'] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in model_outputs.hidden_states]
```
- **Key line**: next memory = **final-layer hidden states of the trailing (write) block** → this is $H_\tau^{write}$ becoming $H_{\tau+1}^{mem}$.
- `num_mem_tokens in {0, None}` short-circuits → degrades to plain backbone (baseline).

### `RecurrentWrapper` — the segment loop + BPTT control
```python
class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs   # segment_size, max_n_segments, k2, segment_alignment, ...

    def forward(self, input_ids, labels=None, labels_mask=None, ...):
        memory_state = None
        segmented = self.segment(input_ids=..., attention_mask=...)
        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state,
                                                      output_hidden_states=True)
            cell_outputs.append(cell_out)
            self.manage_gradients(memory_state, seg_num)   # truncated BPTT
        return self.process_outputs(cell_outputs, labels=labels, labels_mask=labels_mask, ...)
```

- **Segmentation** (`segment` / `split_tensor`): chops every tensor along dim=1 into `segment_size` chunks. `segment_alignment` ∈ {`left`(default), `right`, `center`}.
- **Recurrent loop**: `memory_state` threaded through; output of seg → input of next seg. This *is* the recurrence — a Python for-loop over segments.
- **Truncated BPTT** — `manage_gradients`:
  ```python
  k2, max_n_segments = self.rmt_config['k2'], self.rmt_config['max_n_segments']
  if seg_num == 0 or k2 in {-1, None} or seg_num + k2 > max_n_segments:
      return True              # KEEP gradient (no detach) for last k2 segments
  memory_state = memory_state.detach()   # else cut the graph
  ```
  - `k2` = number of trailing segments to keep in the backward graph (the BPTT unroll depth from the paper). `k2=-1` → full BPTT through all segments. Detaching `memory_state` severs gradient flow into earlier segments.
- **Loss** (`process_outputs`): concat all segment logits → standard **shifted causal-LM CrossEntropy**; optional `labels_mask` selects which positions contribute (used to score target-only on algorithmic tasks). Per-segment logits/hidden/attn also stored as `logits_{seg_num}` etc.
- **`generate`**: run the loop over all-but-last segment to build up `memory_state`, then call `memory_cell.generate(...)` on the final segment (HF `.generate` over `inputs_embeds` with the memory prepended/appended).

### Wiring it up (training script)
`run_finetuning_lm_rmt.py` (and the `scripts/wikitext/*.sh`):
```python
model = get_cls_by_name(args.model_cls).from_pretrained(...)   # e.g. transformers:AutoModelForCausalLM, gpt2
if args.num_mem_tokens is not None:
    cell    = memory_cell_cls(model, args.num_mem_tokens)              # modeling_rmt.language_modeling:MemoryCell
    model   = recurrent_wrapper_cls(cell,
                                    segment_size=block_size,           # = input_size - 2*num_mem_tokens
                                    max_n_segments=args.max_n_segments,
                                    vary_n_segments=args.vary_n_segments,
                                    k2=args.k2)
```
- **`block_size -= 2 * args.num_mem_tokens`**: real-token budget per segment shrinks to leave room for the $2m$ memory tokens within the backbone's max input size.
- **Backbone is HF off-the-shelf** (`AutoModelForCausalLM`, GPT-2) — confirms "no changes to the Transformer".

### Important hyperparameters / knobs
| Arg | Meaning |
|---|---|
| `num_mem_tokens` ($m$) | memory tokens per side (e.g. 1–10 for LM; 30 for quadratic; 6–24 algorithmic) |
| `max_n_segments` | max segments per example |
| `k2` | **truncated-BPTT depth** (trailing segments kept in graph; `-1` = full) |
| `k1` | (not implemented) update every k1 segments |
| `segment_size` / `input_size` | tokens per segment; `segment_size = input_size − 2m` |
| `segment_alignment` | `left`/`right`/`center` chunking |
| `bptt_depth` | BPTT depth in the older `base.py` path (`-1` default there) |
| `vary_n_segments` | randomly pick #segments 1..max (curriculum-ish) |
| `sum_loss` | sum loss over all segments |
| `segment_ordering` | `regular`/`reversed`/`bidirectional`/`repeat_first`/`last_memory_only` |
| `memory_layers`, `share_memory_layers` | (older path) add/share dedicated memory-processing layers |
| `freeze_model_weights`, `use_lora` | fine-tune only memory/adapters on a frozen pretrained backbone |

- **Curriculum training** in scripts: train with $n{-}1$ segments, load that checkpoint (`--model_cpt`), then continue with $n$ segments (`MAX_N_SEGMENTSS=(2 3 4 5 6 7)`), often with **frozen backbone + LoRA** and a fixed small `MEMORY_SIZE` (e.g. 2), `K2=4`.

---

## Related Notes

- **[AutoCompressors](../AutoCompressors/AutoCompressors.md)** — Direct descendant. Replaces RMT's read/write **memory tokens** with **soft-prompt summary vectors**, and crucially **accumulates** summaries across segments instead of carrying a single fixed-size memory state forward. The entire RMT→AutoCompressor delta is a `torch.cat([softprompt, new_softprompt])` (accumulate) vs reassignment (RMT-style) branch.
- **[Artificial Hippocampus Networks (AHN)](../AHN/AHN.md)** — Modernized recurrent compression. Like RMT it folds past context into a **fixed-size recurrent state**, but the state is produced by an RNN (Mamba2/DeltaNet/Gated DeltaNet) compressing the **KV cache**, and it pairs this long-term memory with a **lossless sliding window** of recent KV (RMT keeps no exact recent tokens). Trains only the AHN module on a frozen base; RMT trains the whole recurrent wrapper with BPTT.
- **[Compress and Attend Transformer (CAT)](../CAT/CAT.md)** — Contrast point. RMT/AutoCompressors are *recurrent* (sequentially thread one memory across segments); CAT compresses **chunks in parallel** and attends to all chunk reps at once, avoiding the sequential BPTT dependency while still cutting attention cost.
