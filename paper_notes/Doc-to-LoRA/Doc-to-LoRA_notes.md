# Doc-to-LoRA: Learning to Instantly Internalize Contexts

> Charakorn, Cetin, Uesaka, Lange (Sakana AI). Feb 2026. arXiv:2602.15902 (preprint label).
> Code: github.com/SakanaAI/doc-to-lora

---

## TL;DR

- **D2L** = a Perceiver-style **hypernetwork** that maps a context string $c$ to a **LoRA adapter** $\Delta W_c$ for a *frozen* target LLM in a **single forward pass**.
- It is meta-trained to **emulate the context distillation (CD) process** so you don't have to run per-prompt SGD each time you internalize a doc.
- After internalization, queries are answered **without the doc in the context window** -> avoids quadratic attention cost and the KV-cache blow up.
- Beats vanilla CD under tight compute budgets, generalizes to docs **>4x** longer than anything seen in training (via context chunking), and even works **zero-shot from a VLM encoder to a text-only target LLM**.

---

## 1. Problem & Motivation

**The pain point with ICL**
- LLM users dump long docs / instructions / personas / preference statements into the context window.
	- **Quadratic attention** -> latency blows up
	- **KV-cache memory** grows with context
	- **Quality drops** with longer context (lost-in-the-middle, "context rot")
- And for repeating use (same persona, same codebase) we pay the cost every single call.

**Existing remedies are imperfect**
- **SFT** on synthetic instances of the doc -> needs a dataset, risks overfitting, expensive to redo when info changes.
- **Context distillation (CD)** -> train a small adapter so that the student LLM (no context) mimics teacher LLM (with context) on queries. Internalizes info into weights but **per-prompt SGD is slow + memory-hungry**.
	- Vanilla CD takes ~40+ s and 7+ GB per prompt for gemma-2-2b.
- **Prompt compression** (LLMLingua-2, Gisting): still occupies context tokens.

**D2L's bet**
- **Amortize the entire CD process into a hypernetwork** trained on a huge corpus of contexts.
- One forward pass -> a LoRA adapter that mimics what CD would have produced after many SGD steps.
- Sub-second internalization, low memory, **reusable** adapter (one per persistent doc/user/instruction).

> Conceptually, D2L is "**inference-time training, but compiled into a feedforward map**."
> Related to: hypernetworks for adapter prediction (HINT, HyperTuning, MEND, T2L), gist tokens, Cartridges (KV prefix-tuning via self-study), Generative Adapter (SFT-trained hypernet).

---

## 2. Preliminaries

### Context Distillation (CD)

- A flavor of **self-distillation**: same LLM is teacher (sees $c$) and student (doesn't).
- Per (context $c$, query $x$): sample $y \sim p_\theta(\cdot \mid x, c)$ from teacher; student must reproduce $y$ without $c$.
- Optimize **context-specific params** $\theta_c$ (LoRA or similar):
$$\min_{\theta_c} \mathrm{KL}\big(p_\theta(y\mid x,c) \,\|\, p_{\theta_c}(y\mid x)\big)$$
- Initialized from $\theta$. The training signal is **per-query**.

### Query-independent CD (what D2L actually emulates)

- Single $(c,x,y)$ overfits to that query.
- Robust version uses many queries: generate $\{x_i\}$ from an LLM, sample responses $\{y_i\}$ from teacher.
- Per-context dataset $D_c = \{(x_i, y_i)\}$, objective:
$$\min_{\theta_c} \mathbb{E}_{(x,y)\sim D_c}\big[\mathrm{KL}(p_\theta(y\mid x,c) \,\|\, p_{\theta_c}(y\mid x))\big]$$
- **Internalization** = the process of turning $c$ into $\theta_c$ so the model behaves as if $c$ were in-context.

---

## 3. Method: D2L

### 3.1 High-level objective

- D2L learns a **hypernetwork** $H_\phi$ such that $\Delta W_c = H_\phi(c)$ behaves like the CD-trained delta.
- Meta-training corpus: $D = \{(c_i, D_{c_i})\}_{i=1}^n$ across many contexts.
- Meta-training loss:
$$\min_\phi \mathbb{E}_{(c,D_c)\sim D}\,\mathbb{E}_{(x,y)\sim D_c}\, \mathrm{KL}\big(p_\theta(y\mid x,c) \,\|\, p_{\theta + H_\phi(c)}(y\mid x)\big)$$
- Key payoff: at test time **no SGD, no per-prompt query generation** — just $H_\phi(c)$.

### 3.2 Hypernet architecture

- **Inputs**: per-layer token activations $Z \in \mathbb{R}^{L \times N \times D}$ from the *frozen target LLM* itself reading the context.
	- $Z_l$ at layer $l$ -> the hypernet block at layer $l$ produces LoRA params $\Delta W_l$ for layer $l$.
- **Per-layer hypernet head**: maps $Z_{l-1}$ to $(A_l, B_l)$ with $A_l \in \mathbb{R}^{r \times d_l^{in}}$, $B_l \in \mathbb{R}^{d_l^{out} \times r}$.
$$W_l' = W_l + \alpha_l B_l A_l \quad ; \quad \alpha_l \text{ learnable per-layer scalar}$$
- All layers share the *same hypernetwork weights* but condition on layer-specific activations. (One hypernet, many LoRA matrices.)

**Why per-layer activations** (vs. just embeddings):
- Each layer can specialize to "what to inject into me" -> richer per-layer signal.
- The base model is already paying the cost of computing $Z$ once during the read pass, so its essentially free.

### 3.3 Perceiver-style cross-attention encoder

- Inputs are **variable-length**; LoRA shape is fixed -> need a bottleneck.
- For each transformer layer $l$, learnable **input-independent latent queries** $Q_m \in \mathbb{R}^{r \times d_q}$, $r$ = LoRA rank.
- Cross-attend to context activations:
$$U_l = \mathrm{XAttn}(Q_m, K(Z_{l-1}), V(Z_{l-1})) \in \mathbb{R}^{r \times d_q}$$
- Then two per-layer output heads turn $U_l$ into rows of $A_l$ and columns of $B_l$.
- Simple version: single cross-attention block. Extended: multiple Perceiver blocks possibly with interleaved self-attn.
- So: $H_\phi(c) = \{\Delta W_l\}_{l=1..L}$, $\Delta W_l = B_l A_l$.

> Why Perceiver: it natively handles **arbitrary length inputs -> fixed bottleneck**. The bottleneck size determines LoRA rank. Cute alignment.

### 3.4 Long-context composition via chunking

- For long docs split $c$ into $K$ contiguous chunks $\{c^{(k)}\}$, run hypernet on each independently -> $(A_l^{(k)}, B_l^{(k)})$.
- **Concatenate along the rank dimension** to combine chunks:

$$A_l = \begin{bmatrix} A_l^{(1)} \\ \vdots \\ A_l^{(K)} \end{bmatrix}, \quad B_l = \begin{bmatrix} B_l^{(1)} & \cdots & B_l^{(K)} \end{bmatrix}$$

- Result: LoRA of total rank $r \cdot K$, output shape of hypernet unchanged.
- **Crucially** this lets D2L handle docs much longer than its training length **without retraining**.
- Note: this composition is **order-agnostic** by construction (sum of per-chunk rank-1 outer products). Could be a limitation for things needing positional structure.

### 3.5 Generality of parameterization

- The arch is **not LoRA-specific**: in Appendix D they swap the head to predict **KV-cache prefix tokens** (prefix-tuning style, a la Cartridges) for Qwen3-4B. Works on NIAH up to 8K (training was 1024-token chunks).
- One subtle but important trick: when generating keys, route them through the model's **K-norm + RoPE** layers so the hypernet doesn't have to learn to emulate RoPE itself. Without that trick performance craters past training length.

---

## 4. NIAH (Needle-in-a-Haystack) Pedagogical Experiment

### Setup

- Base LLM: **gemma-2-2b-it** (8K native context).
- Training inputs: 32-256 token contexts containing a 4-digit "magic number" needle. Random chunking 1-8 chunks.
- 640K training samples, 1 epoch, lr $4\times 10^{-5}$, ~3 hrs on a single H200.
- Loss: CE on ground-truth needle (simplified vs main exp's KL).

### Eval

- Haystacks 1K -> 128K tokens.
- D2L processes haystack as chunks of 1024 tokens (4x longer chunks than training max).
- Then asks "what is the special magic number" with **no context** in the query prompt.

### Results

| Haystack length | Base + context | D2L (in-params) |
|---|---|---|
| <= 8K | ~1.0 | ~1.0 |
| 16K-40K | drops sharply | ~1.0 (perfect) |
| ~128K | ~0.1 | still > 0.5 |

- D2L matches in-context up to 8K, then **far outperforms** since the base model just can't read past 8K.
- Holds near perfect up to **40K (40 chunks, 5x more chunks than training)**, degrades gracefully beyond.
- **Memory**: base model w/ context uses >12 GB extra at 128K; D2L stays <50 MB regardless of haystack length.

> Big takeaway: chunking + rank-concatenation generalizes far past training length, and **D2L can break the native context window of the base LLM**.

---

## 5. Main Experiments (QA Benchmarks)

### Setup

- Base LLM: **gemma-2-2b-it** (also Mistral-7B-Instruct-v0.2 and Qwen3-4B in App E).
- LoRA target: **down_proj only** (MLP down projection). Rank 8.
- Hypernet: **8 Perceiver cross-attention blocks** (no self-attn), 309M trainable params.
- Outputs a rank-8 LoRA for each context chunk; chunk size 8K tokens.
- Two inference modes (mathematically equivalent, perf diffs from kernel/order):
	- **Batched**: all-layer activations in one forward, all LoRAs in one go (faster).
	- **Iterative**: layer-by-layer (lower peak memory).

### Training pipeline

- Two-stage training (App B.2):
	1. **Single-chunk pretraining** (80K steps): always 1 chunk, internalize a single doc.
	2. **Multi-chunk finetuning** (20K steps): random chunking 1-8 chunks via the prob dist `{1:0.5, 2:0.125, 3-8:0.0625 each}`. Teaches compositionality of LoRAs across chunks.
- This staging is reportedly **critical** for stability — training multi-chunk from scratch is unstable.
- Batch packing: 4K-token sequences with grad accumulation, >200K context tokens per batch across 8 GPUs.

### Data

- **Contexts** from **FineWeb-Edu** subset (~900M tokens, ~3.2M unique contexts after filtering >10K char).
- Plus passage-grounded QA tasks (PwC, SQuAD, ROPES, DROP) for response-format priors.
- Each context: **10 (q, a) pairs** generated by `gemma-3-12b-it` (a **bigger model** for quality).
	- Two-iteration prompting (Listing 4 & 5): first round produces 5 questions; second round produces 5 more with the first batch as in-context examples to push diversity / difficulty.
	- Generated answers thrown away; questions get random instruction template (Listing 6) applied.
	- Self-responses sampled from the **target LLM** (gemma-2-2b-it) using Listing 7; **top-16 token logits per generated token recorded** (these are the KL distillation targets, Eq. 4).
- ~101M (context, query, response) triplets.

### Baselines (in-parameter knowledge methods)

- **CD (oracle)**: SGD on target query directly (upper bound — peeks at the eval question).
- **CD (k generated queries)**: SGD on synthetic queries (5, 20, 25, 50, 100). The realistic CD.
- **T2L** (Charakorn et al., 2025): hypernet trained via SFT, no CD.
- **Base model w/o context**: lower bound.

### Baselines (in-context knowledge methods)

- Base model **w/ full context**: upper bound for ICL.
- **LLMLingua-2** at 10%-90% compression rates.

### Main Reading-Comprehension Results (SQuAD, DROP, ROPES)

- **D2L beats all in-parameter baselines** including high-budget CD (which it eventually exceeds with 100+ queries).
- Reaches **82.5% of full-ICL** on SQuAD.
- Comparable to LLMLingua-2 at 40% compression — but D2L **removes the context entirely** (0% context len ratio).
- Update latency: <1 s (batched ~0.2 s, iterative ~0.5 s) vs CD ~40 s (oracle) or >100 s (generated queries pipeline including query gen).
- Update memory: <2 GB for D2L vs ~40+ GB for CD with many queries.

### Long-Context QA (2WikiMultihopQA, MultiFieldQA, QASPER)

- Up to **32K-token** test docs (D2L was never trained on this length!).
- D2L generalizes via chunking, outperforms CD with limited query budgets.
- Even **slightly beats** the "+ truncated context" baseline on 2WikiMultihopQA and MultiFieldQA — i.e., reading the (truncated) doc on top of having internalized it improves further.
	- Hypothesized: D2L's internalization works *better than* shoving truncated context in due to **lost-in-the-middle / attention noise**.
- On 2WikiMultihopQA: D2L (batched) gets 0.857 vs CD oracle 0.901, with 5x less memory and 200x less latency.

### Visual Information Transfer (zero-shot!)

| Setup | SQuAD | DROP | ROPES | Imagenette |
|---|---|---|---|---|
| D2L (LLM->LLM) | 0.814 | 0.655 | 0.906 | N/A |
| D2L (VLM->LLM) | 0.705 | 0.568 | 0.772 | **75.03%** |

- Replace ctx encoder with **gemma-3-4b-it (VLM)**; target stays text-only gemma-2-2b-it.
- D2L hypernet was trained on text only.
- At eval, feed an **image** to the VLM, hypernet produces a LoRA for the text LLM, asked to **classify** the image from 10 categories. Gets **75% accuracy** purely through internalized weights — the target LLM never sees an image.
- Striking: the hypernet has learned a fairly **generic activation->LoRA** mapping that transfers across modalities.

---

## 6. Analyses & Ablations (App C)

### Zero-shot query internalization (Table 3)

- "Swapped" config: at training time D2L internalizes **documents**; at eval, force it to internalize the **query** instead, and put the doc in-context.
- D2L still works moderately (ROUGE-L 0.587 vs 0.185 no-context baseline) — recall is similar but **precision crashes** since outputs become verbose.
- Suggests D2L learned a **biased** form of CD assuming queries will be related to internalized content.

### D2L emulates many-query CD (Table 4)

- On 100-sample SQuAD: D2L 0.866 norm perf vs CD (100 gen queries) 0.650 vs CD (20 gen queries) 0.506.
- D2L's edge comes from being exposed to **millions of context-query examples** during meta-training -> learns a wider query distribution than any individual CD run could afford.

### Data ablation (Table 5)

- Removing QA-task supervision (training **only on FineWeb-Edu**) gives comparable / slightly different perf:
	- SQuAD ~same, DROP **better** (no in-domain bias?), ROPES slightly better.
- D2L is **robust to training data format**.

### Loss ablation: KL vs NTP (Table 6)

- KL (default, distill top-16 logits) clearly beats next-token-prediction.
- On the swapped config the gap widens: KL 0.385 recall vs NTP 0.235.
- KL preserves uncertainty / alternative modes from teacher; NTP just point-estimates.

### LoRA rank (Table 7)

- Rank-16 D2L outperforms rank-8 on SQuAD (0.896 vs 0.814) and DROP (0.711 vs 0.655).
- More capacity helps; paper uses rank 8 for cheaper experiments.

### Knowledge interference (Table 8)

- Replace the context with **unrelated** prompts ("You are a helpful assistant" / a random Gutenberg chapter) and ask the original SQuAD question.
- D2L scores **lower** than base model w/ replaced context — i.e., D2L's internalized info **overrides** existing knowledge even when irrelevant.
	- Caused by training bias: queries are always related to context, so model assumes that's the case.
	- Future fix: regularize with irrelevant queries during training, or layer in continual-learning ideas (constraint edits, sparse mem FT).

### Knowledge interference is a real limitation:
- **D2L doesn't know when to ignore the internalized doc** — it always uses it.

### KV-Cache variant (App D)

- Swap LoRA head for one that outputs a **20-token KV prefix**.
- Works on NIAH up to 8K tokens.
- Generating K naively breaks past 4K; routing through model's K-norm + RoPE fixes it (model has to "fake" RoPE otherwise).
- Cute proof that the architecture is general beyond LoRA.

### Other models (App E)

- Tested on **Mistral-7B-Instruct-v0.2** and **Qwen3-4B-Instruct**.
- Same overall conclusions.
- For Mistral, also compared to **Generative Adapter** (the closest related method — also trains a hypernet but via NTP/SFT, not CD).
	- GA has higher F1 but **much lower recall** -> shorter responses that happen to overlap. D2L preserves factual content better.
- Mistral observes that all internalization methods struggle on long contexts (~ no-context baseline). No explanation given.

---

## 7. Limitations (paper's own)

- **Single expensive meta-training run**: ~5 days on 8 H200s for gemma-2-2b-it. Not trivial.
- **Per-target-LLM**: hypernet retrained when changing target model. (Universal hypernet = open Q.)
- **Performance gap to ICL still exists**, especially on long documents.
- **Knowledge interference**: D2L overrides existing knowledge when internalized info is irrelevant. No preservation mechanism.
- **Order-agnostic chunking**: combining chunks loses ordering info.
- LoRA parameterization may not be optimal — could try prefix-tuning, sparse memory FT, etc.

## Notable / surprising / clever bits

- **One forward pass replacing inference-time training**: the whole D2L bet hinges on context distillation being learnable as a function rather than a search/optimization process. Empirically it is.
- **Rank-concatenation for chunk merging**: clean way to handle arbitrary doc length without retraining the hypernet. Total LoRA rank = $r \cdot K$ grows with doc.
- **Per-layer activations from the target model itself as the encoder** — almost free to compute (model was going to read the doc anyway), and gives the hypernet per-layer-specific signals to inject per-layer-specific edits.
- **Zero-shot VLM->LLM** transfer is wild. The hypernet just learned a generic activations->LoRA map.
- **KL distillation crucial over NTP** — preserves teacher's distributional mass over alternatives, especially when query/context relationship is unusual (swapped config).
- **Two-stage training (single-chunk then multi-chunk)** is needed to stabilize learning of multi-chunk compositionality.
- **Lost-in-the-middle as a free bonus**: internalization sometimes beats truncated ICL because it removes the attention noise from irrelevant tokens.
- **309M hypernet params** — small compared to base LLM, but produces edits to a 2-7B model.

---

# Implementation Notes (Codebase Analysis)

Codebase at `/raw_data/papers/Doc-to-LoRA.../doc-to-lora/`. Package name is `ctx_to_lora` (older name). Uses HuggingFace Transformers + PEFT + accelerate, with a Perceiver borrowed from **Idefics2**.

## File map

- **`train.py`** — entry point, parses args, builds model, dataset, runs `Trainer`.
- **`run_eval.py`** — eval entry point (CD, in-context, D2L baselines).
- **`src/ctx_to_lora/`**
	- `configs.py` — dataclass arg defs.
	- `model_loading.py` — LLM + tokenizer + LoRA config loading.
	- `trainer.py` — `CrossEntropyTrainer`, `DistillationTrainer` (KL).
	- `modeling/`
		- `hypernet.py` — **`HyperLoRA`** (the hypernet), **`ModulatedPretrainedModel`** (wrapper around base LLM + hypernet + ctx encoder).
		- `aggregator.py` — **`Perceiver`** wrapper (uses Idefics2Perceiver).
		- `ctx_encoder.py` — `PerLayerActivations`, `EarlyExit`, `EmbeddingOnly` context encoders.
		- `lora_layer.py` — patched LoRA forward functions (`lora_forward_packed`).
		- `lora_merger.py` — `combine_lora` rank-concatenation for chunks.
		- `idefics2.py` — vendored Idefics2 Perceiver impl.
	- `data/processing.py` — tokenization, chunking, packing pipeline.
- **`configs/`** — yaml configs (per experiment, per base model).
- **`scripts/`** — `.sh` launchers + data gen scripts.
- **`data/`** — dataset prep scripts (FineWeb-Edu download, self-gen QA, format compacting).

## Forward pass (from `ModulatedPretrainedModel.forward`)

Pseudocode (from Listing 3 + `hypernet.py`):

```python
def forward(LLM, hypernet, ctx_ids, ctx_attn_mask, input_ids, input_attn_mask):
    # 1) ctx encoder = the frozen base LLM (or a separate one)
    # Run target LLM on context, grab per-layer activations Z
    # Shape: [n_chunks, n_layers, seq_len, hidden_size]
    features = LLM.forward(ctx_ids, ctx_attn_mask).detach()

    # 2) Perceiver cross-attends Z to latent queries
    # Shape: [n_chunks, n_layers, r, d_latent]
    emb = hypernet.perceiver(features, ctx_attn_mask)

    # 3) Per-layer head maps latent -> LoRA params (A, B concatenated)
    # Shape: [n_chunks, n_layers, r, d_in + d_out]
    lora_flat = hypernet.head(emb)

    # 4) Combine across chunks by rank-concatenation
    # Shape: [n_layers, r * n_chunks, d]
    lora = combine_lora(lora_flat, n_ctx_chunks)

    # 5) Monkey-patch LLM's MLP/attn layers with lora_forward,
    #    then call LLM normally
    apply_lora_to_layers(LLM, lora)
    return LLM.forward(input_ids, input_attn_mask)
```

## Key class: `HyperLoRA` (`hypernet.py`)

- Takes a `HypernetConfig` containing aggregator config, target LoRA modules, layer indices, latent size etc.
- Submodules:
	- **`self.aggregator`** = Perceiver (defined in `aggregator.py`).
	- **`self.layers`** = stack of `ResMLPBlock` or `ResMLPBlockPerLayer` (residual MLP refinement of latent before head).
	- **`self.head`** = an `einops.EinMix` that maps `[bs, n_layers, n_modules, r, d_latent]` -> `[bs, n_layers, n_modules, r, d_lora]` where `d_lora = d_in + d_out` (max across target modules).
	- **`self.bias_A`, `self.bias_B`** = learnable data-independent LoRA biases (added via `combine_lora` if `use_bias=True`). These are essentially a learned default adapter that gets concatenated to every prediction.
	- **`self.scaler_A`, `self.scaler_B`** = per-(layer, rank) learnable scalars. Initialized to 1 (A) and 0 (B) — same as standard LoRA init so initial $\Delta W = 0$.
- `forward()`:
	1. Aggregator -> `lora_emb [bs, n_layers, n_modules, r, d_latent]`.
	2. Optional iterative mode: process one layer at a time.
	3. MLP refinement.
	4. **L2 normalize** along last dim before head (`norm_lora_emb = lora_emb / norm`).
	5. Head -> flat LoRA tensor; reshape to per-module A/B dict via `_to_lora_dict`.

### LoRA matrix layout

- `d_lora = max(d_in[m] + d_out[m] for m in target_modules)`.
- Head packs A and B together along the last dim.
- `_to_lora_dict` splits into `A: [bs, n_layers, r, d_in]` and `B: [bs, n_layers, r, d_out]`.
- Per-module scalers applied to A and B via einsum (faster than broadcast multiplication).

## Key class: `Perceiver` (`aggregator.py`)

- Uses **Idefics2Perceiver** (from HF Idefics2 model) for the cross-attention block.
- Latent queries shape: `n_latent_queries` (e.g., 208 = 26 layers x 8 rank for NIAH; 8 for main exp w/ layer-to-layer encoder).
- `Idefics2PerceiverConfig`:
	- `num_blocks` cross-attn blocks (8 in main exp).
	- `num_self_attn_per_block=0` (no self-attn — pure cross-attention stack).
	- `shared_weights=False`.
	- `attn_implementation="flash_attention_2"`.
- Two routes through it:
	- **Layer-to-layer mode** (`per_layer_activations` ctx encoder): each layer's activations go through the Perceiver independently, producing per-layer LoRA bottleneck.
		- During training: batch all layers along bs dim (`(num_layers bs)`) -> single big Perceiver call.
		- During iterative inference: loop over layers (memory savings).
	- **Pooled mode**: single Perceiver call sees all layers, output queries number = `n_layers * n_modules * r`.
- Decoder = single small Perceiver layer that maps to `n_output_queries`.

> Idefics2 was a vision-language model; reusing its Perceiver lets them inherit a tested implementation for variable-length cross-attention.

## Key class: `ModulatedPretrainedModel` (`hypernet.py`)

- Wrapper around `base_model: PeftModel` + `hypernet: HyperLoRA` + `ctx_encoder`.
- Inits with `base_model.disable_adapter_layers()` because PEFT's default LoRA forward is bypassed by the **patched** forward (custom code).
- **`patch_lora_forward()`** monkey-patches each target module's `forward` with `lora_forward_packed`/`lora_forward` so the hypernet-produced A, B can be plugged in dynamically.
- **`generate_weights(ctx_ids)`**:
	- Runs `ctx_encoder` (no grad) on context, gets activations.
	- Calls `hypernet.generate_weights(ctx_features, ...)` -> dict of `{module: {A, B}}`.
- **`forward(ctx_ids, ..., input_ids, ...)`**:
	- Generates LoRAs, combines via `combine_lora`, applies via `apply_lora_to_layers`, then runs base_model.
	- Standard model output, plus optionally returns the generated LoRAs for the loss to apply L1 regularization.
- **`internalize(ctx_str)` / `generate(ctx_ids, input_ids)`**: convenience for inference (used in `examples/python_api.py`).
- **State dict** stores hypernet weights + base model name + hypernet config + ctx encoder args. Base model and ctx encoder weights are not stored (they're frozen).
- **`torch.serialization.add_safe_globals([...])`** at the bottom — needed for the new HF safe-loading checkpoint format with custom dataclasses.

## Context Encoders (`ctx_encoder.py`)

Three options:
- **`PerLayerActivations`** (main exp) — runs base LLM (or another LLM) on context with `output_hidden_states=True`, stacks all hidden states into `[bs, n_layers, seq_len, hidden]`.
	- The lm_head is replaced with `nn.Identity()` or stripped to save memory.
	- Last few layers can be truncated via `ctx_encoder_last_layer`.
- **`EarlyExit`** (NIAH exp) — truncates to first $L/4$ layers (default), uses `last_hidden_state` only.
- **`EmbeddingOnly`** — just the embedding layer output.

## Custom LoRA forward (`lora_layer.py`)

```python
def lora_forward_packed(x, n_qs, tot_q, seq_lens, tot_len, A, B, ...):
    base_out = nn.Linear.forward(self, x)  # frozen base linear
    delta_x = dropout(x)
    # repeat per-context A,B to per-token A,B
    A = A.repeat_interleave(n_qs).repeat_interleave(seq_lens)
    B = B.repeat_interleave(n_qs).repeat_interleave(seq_lens)
    delta_x = einsum(A, delta_x, "tot_len r d_in, bs tot_len d_in -> bs tot_len r")
    delta_x = einsum(B, delta_x, "tot_len r d_out, bs tot_len r -> bs tot_len d_out")
    return base_out + delta_x * scaling
```

- Handles **sequence packing**: multiple (q, a) pairs concatenated into one batch entry, with `n_qs` queries per context and `seq_lens` token lengths.
- Used in conjunction with HF's flash attention varlen handling.
- The packed version is monkey-patched onto each target `nn.Linear` (e.g., `down_proj` in each transformer block).

## Chunk merging (`lora_merger.py`)

`combine_lora(generated_loras, n_chunks, lora_bias=None, scalers=None)`:
- Per module, per matrix key (A, B):
	- Flatten chunks along rank dim: `[tot_chunks, n_layers, r, dim] -> [1, n_layers, tot_chunks * r, dim]`.
	- Split per "group" (i.e., per context that owns some chunks).
	- Zero-pad to `max_rank_needed = (max(n_chunks) + 1) * r` (the +1 is for the bias slot).
	- After the concatenated chunk ranks, slot in the **learned bias** `B/A` matrices (data-independent LoRA component) — at rank index `combined_rank : combined_rank + base_rank`.
- Output shape: `[num_groups, n_layers, max_rank_needed, dim]` — ready to be applied via patched `Linear.forward`.

> Subtle: every adapter always carries the learned bias rows, even if chunk count varies. This bias acts like a "base persona" LoRA that's always present, with the chunk-derived rows stacked on top.

## Training loop (`trainer.py`)

Subclasses HF `Trainer`. Two flavors based on `use_kl_loss`:

### `DistillationTrainer` (used in main exp)

- For each batch:
	1. `outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)`.
	2. Pull pre-stored teacher data from the dataset: `logprobs_vals` (top-K teacher log-probs) and `logprobs_indices` (the corresponding token IDs).
		- These were precomputed during data generation (`self_generate_qa.py`) — top-16 token logits per generated label token.
	3. Get **student logits at label positions** (shifted back 1 -> next-token prediction alignment).
	4. KL loss = $-\sum_k p_k \log q_k$ where $p$ is from teacher top-K and $q$ from student renormalized over the full vocab.
		- Implementation uses `logsumexp` over student logits for normalization, only gathers selected indices for the dot product — avoids materializing full $|V|$ teacher dist.
	5. **L1 regularization** on `|A| + |B|` of every generated LoRA, coef `gen_lora_l1_reg_coef=0.1` -> encourages sparse adapters.
	6. Loss averaged either per-context (`use_per_ctx_average_loss=True`) or per-token.
- L1 reg is a clever trick: discourages large per-chunk deltas so the **bias** can absorb the average effect and per-chunk deltas only encode content-specific updates.

### `CrossEntropyTrainer` (NIAH exp)

- Plain NTP loss on labels. No teacher logits needed.
- Used because NIAH ground truth is a unique short string (the magic number), so no need for distillation richness.

### Per-context loss averaging (`per_ctx_loss_ce` / `per_ctx_loss_kl`)

- Sequence packing means a batch has multiple (context, query) groups.
- They compute mean-per-query-then-mean-per-context to ensure contexts contribute equally regardless of how many queries they have.

## Data pipeline (`data/processing.py`, `data/self_generate_qa.py`)

Heavy pipeline:
1. **Download** FineWeb-Edu subset.
2. **Generate QA**: prompt gemma-3-12b-it (two-stage prompting from Listing 4 & 5) to make 10 questions per context. Throw away its answers.
3. **Augment** instruction templates randomly per question (Listing 6 has 21 templates like "Answer the question based on the given passages...").
4. **Self-responses**: prompt the target LLM (gemma-2-2b-it) with the questions, save top-16 token logits per output token.
5. **Tokenize** + chunk + pack into 4K/6K-token sequences with grad accumulation.

Key hyperparams for chunking:
- `max_ctx_chunk_len=512`, `min_ctx_chunk_len=25` (stage 2).
- `num_chunk_probs='{"1":"0.5", "2":"0.125", "3":"0.0625", ..., "8":"0.0625"}'` — 50% of training is single chunk; rest distributed across 2-8 chunks.

## Important hyperparameters (from `scripts/main_exp/1-train.sh` + configs)

| Param | Value | Notes |
|---|---|---|
| Base model | `gemma-2-2b-it` | |
| Target modules | `down_proj` | MLP only, single module |
| `lora_r` | 8 | |
| Hypernet `latent_size` | 512 | |
| Perceiver `num_blocks` | 8 (NIAH), 9 (main) | |
| Perceiver `num_self_attn_per_block` | 0 | No self-attn |
| `n_latent_queries` | 8 (main, per layer), 208 (NIAH, all layers) | |
| `num_pre_head_layers` | 1 | ResMLP after Perceiver |
| `per_rank_gen` | True | Each rank slot is a Perceiver output |
| `per_layer_processing` | True | Per-layer hypernet weights for ResMLP |
| Stage 1 steps | 80K | Single chunk |
| Stage 2 steps | 20K | Multi-chunk |
| Stage 1 LR | 4e-5 (cosine + min_lr=1e-7) | |
| Stage 2 LR | 2e-5 | warmup 2000 steps |
| Batch | 8 GPUs x grad accum 8/16 | 200K+ ctx tokens / batch |
| `max_packed_inp_len` | 4096 | |
| `max_packed_ctx_len` | 4096 | |
| `gen_lora_l1_reg_coef` | 0.1 | |
| `use_kl_loss` | True | top-16 token distillation |
| `use_per_ctx_average_loss` | True | |
| `quantize_ctx_encoder` | True | QLoRA on the ctx encoder side (saves VRAM) |
| `neftune_noise_alpha` | 5.0 | NEFTune embedding noise reg |
| Optimizer | `adamw_torch_fused` | |
| Weight decay | 0.01 | |
| Total | ~5 days on 8x H200 | (gemma-2-2b-it) |

## Notable implementation tricks

- **Per-layer activation features** are detached (`.detach()`) so gradients don't flow into the ctx encoder.
- **L2-normalize the latent before the head** — keeps the LoRA delta magnitudes well-conditioned.
- **`bias_A`, `bias_B`** as learned data-independent LoRA rows — they get concatenated alongside the chunk-derived rows. Lets the hypernet specialize on per-context delta on top of a fixed "always-on" delta.
- **Two-stage curriculum** (single -> multi-chunk) for compositional stability.
- **`torch.compile(fullgraph=True, mode="max-autotune")`** on both hypernet and base model — relies on this for throughput.
- **Mixed precision**: TF32 + bf16 throughout; ctx encoder also Q-LoRA-quantized.
- **Sequence packing** is mandatory (`assert ctx_args.use_sequence_packing`). Without it batch sizes would be tiny.
- **Latent queries excluded from weight decay** (via custom `get_decay_parameter_names`) — same trick as for biases/layernorms.
- **NEFTune noise** on embeddings — pretty unusual for hypernet training but they kept the default.
- **`combine_lora`** writes into a pre-allocated zero tensor with explicit indexing rather than torch.cat — easier on `torch.compile`.
- **KL implementation avoids materializing full teacher dist over $|V|$** — gathers student logits at the K teacher indices and subtracts `logsumexp` of the full row. Saves memory for top-16 teacher distillation.
- **`from_state_dict` constructor** that re-instantiates the base LLM by name + reloads only the hypernet weights — checkpoint is small.
- **API**: `model.internalize(doc_string)` stores `generated_loras`, then subsequent `model.generate(...)` uses them — exactly the "load doc once, query many times" UX.

## Deviations / things not in the paper

- The package name `ctx_to_lora` suggests an earlier name.
- The codebase supports more options than the paper describes (e.g., pooling aggregator, embedding-only context encoder, light-weight LoRA, dropout) — these are config knobs not used in the main results.
- `add_negative_prompt` (in `processing.py`) generates context-shuffled negative pairs — appears to be an exploration of how to reduce knowledge interference (limitation #4) but is off by default in the main configs.
- T2L baseline implementation is included for comparison.
- Webui (`webui/`) for browsing self-gen data.
- `demo/app.py` provides a Gradio interface.

---

# Open questions / interesting directions raised

- **Universal hypernet across base LLMs?** Currently retrained per target.
- **Position-aware chunk merging** — current rank-concat is order-agnostic.
- **Preventing knowledge override** — could be tackled with continual-learning constraints (sparse mem FT, constraint edits) or with regularization over irrelevant queries.
- **Beyond LoRA**: KV-cache variant works; could also try sparse memory FT, prefix-tuning.
- **D2L as a starting point for further CD finetuning** — fast warm-start.
- **Inference-time training** (Wang et al.) and **machine unlearning** are mentioned as application-aligned directions.
- **Weight-space interpretability**: the learned LoRA biases + per-chunk deltas could be a substrate for studying what "this context means" in parameter space.
