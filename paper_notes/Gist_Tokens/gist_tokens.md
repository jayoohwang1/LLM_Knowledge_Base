# Learning to Compress Prompts with Gist Tokens

> Mu, Li, Goodman (Stanford). NeurIPS 2023. arXiv 2304.08467.
> **TL;DR**: train an LM to compress an arbitrary prompt/instruction $t$ into a few "gist" token activations $G(t)$ that can be cached + reused. Done by *just modifying the attention mask* during normal instruction finetuning — **zero extra training cost**. Up to **26x** prompt compression, ~40% FLOPs reduction, 4.2% wall-time speedup, minimal quality loss.

---

## 1. Problem & Motivation

- **Prompting** = primary way to invoke multitask LM behavior, but prompts eat context window + are re-encoded every call.
	- Self-attention is $O(n^2)$ in input length; reusing the same long prompt repeatedly is wasteful.
	- **KV caching** (cache the prompt's keys/values) avoids recompute but still costs memory/storage that grows with #cached prompts.
- **Alternatives & their drawback**:
	- **Finetune / distill** a task-specific model (incl. PEFT: prefix-/prompt-tuning, adapters, LoRA, HyperTuning) → must *retrain per prompt*. Doesn't generalize zero-shot to new prompts.
- **Gisting**: amortize *both* (1) inference cost of prompting with $t$ AND (2) train cost of learning a new model per $t$. Learn $G$ once such that $p_G(y \mid G(t), x)$ ≈ $p_{LM}(y \mid t, x)$, and $G$ **generalizes to unseen tasks with no extra training**.

---

## 2. Framework

- **Instruction finetuning setup**: dataset $\mathcal{D} = \{(t_i, x_i, y_i)\}$ — task prompt $t$ (e.g. `Translate to French`), optional input $x$ (`The cat`), output $y$ (`Le chat`). Standard IFT learns $p_{LM}(y \mid t, x)$ by concatenating $t,x$ and autoregressively predicting $y$.
- **Gisting goal**: compress $t \to G(t)$ = the KV activations on top of a small set of **gist tokens** (fewer than $|t|$), which still induce the same downstream behavior, and which can be **cached + reused**.
	- $G(t)$ = a *Transformer prefix* (à la prefix-tuning [19]) — but predicted zero-shot from $t$ instead of learned per-task by gradient descent.

### Context-distillation perspective
- **Context distillation** [Askell, Snell]: finetune $p_{CD}^t$ to mimic $p_{LM}(\cdot \mid t, \cdot)$ *without* the prompt:
$$\mathcal{L}_{CD}(p_{CD}^t, t) = \mathbb{E}_x\!\left[ D_{KL}\!\left( p_{LM}(y \mid t, x) \,\|\, p_{CD}^t(y \mid x) \right) \right]$$
	- KL approximable on synthetic samples from the LM — no external data $\mathcal{D}$ needed.
- **Gisting = meta context distillation over a distribution of tasks** $t \sim \mathcal{T}$. Instead of distilling one model per task via grad descent, **predict** the distilled "params" (gist prefix) with $G$, HyperNetwork/HyperTuning-style:
$$\mathcal{L}_G(p_G, \mathcal{T}) = \mathbb{E}_{t\sim\mathcal{T},\,x}\!\left[ D_{KL}\!\left( p_{LM}(y \mid t,x) \,\|\, p_G(y \mid G(t), x) \right) \right]$$
	- Key conceptual move: **the LM itself is its own HyperNetwork** $G$ — no separate prefix-predictor network, just an attention-mask tweak.

---

## 3. Learning Gisting by Masking (the core trick)

- **Use the LM itself as the gist predictor $G$.** Leverages pretrained knowledge + costs nothing extra over standard IFT.
- **Recipe**:
	1. Add a **single** new gist token `<GIST>` to vocab + embedding matrix (one learned embedding).
	2. Insert $k$ copies between prompt and input: $(t, g_1,\dots,g_k, x)$, e.g. `Translate French: <G1> <G2> The cat`.
		- > The gist token is the *same* token $g_1=\dots=g_k$; what differs is the *activation* learned on top of each position (like distinct prefix-tuning slots).
	3. **Modify attention mask**: tokens *after* the gist tokens **cannot attend to** any prompt token *before* the gist tokens (but can attend to the gist tokens themselves).
		- Forces all prompt info to be funneled ("bottlenecked") into the gist activations, since $x$ and $y$ have no other path to $t$. This *is* the compression pressure.
- **Decoder-only (GPT/LLaMA)**: start from causal mask, additionally **mask out the lower-left corner** of the triangle (post-gist queries → pre-gist keys).
- **Encoder-decoder (T5)**: 3 changes
	- **Encoder** (normally no masking): prevent input $x$ from attending to prompt $t$; AND prevent $t$ + gist tokens from attending to $x$ (else encoder learns $x$-dependent reps for the prompt — defeats caching).
	- **Decoder cross-attention**: prevent decoder from attending to prompt $t$ (only gist + input).
- Implementable in **~10 lines** of attention-mask code (Appendix A listing below).

### Appendix A reference impl (paper listing)
```python
def make_mask_pre_first_gist(inputs, gist_token, dtype=torch.int64):
    # mask out everything BEFORE the first gist token
    return ((inputs == gist_token).cumsum(-1) >= 1).type(dtype)

def make_mask_post_last_gist(inputs, gist_token, dtype=torch.int64):
    # mask out everything AFTER the last gist token (reverse cumsum)
    return (reverse_cumsum(inputs == gist_token) >= 1).type(dtype)

def make_gist_mask(inputs, gist_token, pad_token, dtype=torch.int64):
    pre_gist_mask  = make_mask_post_last_gist(inputs, gist_token, torch.bool)[:, None, None]
    post_gist_mask = make_mask_pre_first_gist(inputs, gist_token, torch.bool)[:, None, None]
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))
    mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)  # row-dependent select
    mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)
```
- **Mechanism**: for a query *row* that is itself pre-gist (`pre_gist_time_mask` true) use the `pre_gist_mask` (can see up to last gist); for a post-gist query row use `post_gist_mask` (can only see from first gist onward). The `where` stitches the two regimes together → the lower-left block gets zeroed.

---

## 4. Data — Alpaca+

- **Alpaca+** = Self-Instruct [36] ∪ Stanford Alpaca [31], sampled from `text-davinci-001` / `text-davinci-003`.
	- **130,321** examples, **104,664** unique tasks $t$, **48,530** unique inputs $x$, ~0.64 inputs/task.
	- ~**59%** of tasks have **no input** $x$ (e.g. `Write me a poem about frogs`) → $x$ omitted; still useful compression signal.
	- Noisy/imperfect but prior work shows comparable quality to source models.
- **Validation splits**:
	| Split | What | #ex | OOD difficulty |
	|---|---|---|---|
	| **Seen** | held-out, prompts seen in train (w/ unseen inputs) | 1000 | in-dist |
	| **Unseen** | unseen prompts (non-empty inputs) | 1000 | unseen tasks |
	| **Human** | 252 hand-written prompts from [36], 83% non-empty | 252 | strongest OOD (avg ~26 tok vs ~20 train) |

---

## 5. Models, Controls, Baselines

- **Models**: **LLaMA-7B** (decoder-only) and **FLAN-T5-XXL** (11B enc-dec). Trained with gist tokens $k \in \{1,2,5,10\}$.
- **Controls / baselines**:
	- **Positive control (upper bound)**: 1 gist token, **no mask change** = vanilla IFT (full prompt visible).
	- **Negative control (lower bound)**: no access to task $t$ at all ("random gist") — measures floor if compression captured nothing.
	- **TF-IDF discrete compression**: replace instruction with its single highest-TF-IDF keyword subword (e.g. `Write a letter ... salary` → `salary`), then IFT. A cheap "compress to one discrete token" alternative.

---

## 6. Evaluation & Results

- **Metrics**: ROUGE-L (lexical vs davinci/human refs); **ChatGPT-3.5** pairwise win-rate vs positive control (CoT reasoning, randomized A/B order, ties allowed); **Human eval** (3 Prolific annotators, 100/252 Human examples, weighted Cohen's $\kappa$).
	- A **50% win rate** = gist model indistinguishable from positive control (i.e. no quality loss from compression).

### Headline numbers (single gist token, Table 1)
| Model | Split | Gist ChatGPT win % | Notes |
|---|---|---|---|
| LLaMA-7B | Seen | **48.6** | ≈ positive control |
| LLaMA-7B | Unseen | **49.7** | competitive |
| LLaMA-7B | Human (OOD) | **45.8** | slight drop |
| FLAN-T5-XXL | Seen | **50.8** | ≈ tie |
| FLAN-T5-XXL | Unseen | **46.2** | |
| FLAN-T5-XXL | Human (OOD) | **42.5** | larger drop |

- **#gist tokens barely matters** (Fig 3): 1 token ≈ as good as 10. Too many can *hurt* (e.g. LLaMA k=10) — extra capacity → overfit train distribution. → **use single gist token** for main experiments.
- **Gist ≫ TF-IDF**: TF-IDF only marginally beats the negative control → genuine learned compression, not keyword extraction.
- **Human eval (Table 2)**: gist win rate 52.3% (LLaMA) / 40.6% (FLAN-T5) vs pos control. ChatGPT $\kappa$ ≈ 0.29 matches inter-human $\kappa$ → ChatGPT ≈ an extra human annotator (validates cheap eval).
- **Decoder-only (LLaMA) generalizes OOD better than enc-dec (T5)**.
	- Hypothesis: gist masking disrupts T5's **bidirectional encoder** flow, harder to adapt to than just truncating a decoder's history.
- **Exact-match to pos control** (Fig A.3): ~50% on Seen → ~20-25% Unseen → ~10% Human. Compression is lossy but quality stays high.

### Failure cases (§5.1)
- **Verbatim-detail loss**: instructions needing exact phrases copied into output (e.g. category list) sometimes get coarsened (`Culture` vs `Arts & Culture`).
- **Runaway generations**: gist models occasionally loop (`\subparagraph{...}` repeated for hundreds of tokens) — not seen in pos control. Likely mitigable with better sampling.

---

## 7. Efficiency (§6, App F)

- **Caching strategies compared**: (1) **None** (encode full prompt every time), (2) **Instruction caching** = standard KV cache of prompt (decoder-only only — T5 enc reps depend on $x$ so can't), (3) **Gist caching** = cache $G(t)$.
- **Single forward-pass profiling** (252 Human prompts, PyTorch 2.0, 1×A100):
	| Model | Metric | None | Instruction | Gist | Gist vs None |
	|---|---|---|---|---|---|
	| LLaMA-7B | CUDA ms | 23.4 | 22.1 | **21.8** | 6.8% faster |
	| LLaMA-7B | GFLOPs | 915 | 553 | **552** | **40% fewer** |
	| FLAN-T5-XXL | CUDA ms | 31.0 | N/A | **29.7** | 4.2% faster |
	| FLAN-T5-XXL | GFLOPs | 716 | N/A | **427** | **40% fewer** |
- **Gist vs Instruction caching** (LLaMA): FLOPs −0.11%, wall −1% — tiny, because at these short lengths most FLOPs go to processing the *new* input token + dense layers, not attention-over-KV-cache.
	- App F (Fig A.4, Chinchilla FLOPs eqs): with a 2000-len KV cache, only **~9.6%** of per-layer attention FLOPs depend on KV-cache size; KQV projections (29%) + dense block (52%) dominate. So shrinking KV cache gives limited *compute* savings.
- **The real win is memory/storage**: compressing 26 tokens → 1 gist frees context space (bounded by abs pos embeddings / VRAM) and lets you **cache 26x more prompts in the same storage**.
	- LLaMA-7B: each KV-cache token = $4\text{B} \times 2 \times 32 \text{layers} \times 32 \text{heads} \times 128 \text{dim} = 1.05$ MB. Multiplies across many users/prompts.

---

## 8. Related Work / Framing

- **Adapt LMs without backprop**: gisting = zero-shot prefix prediction. HyperTuning [25] is the few-shot analog (predicts prefix from input/output examples). Gist's twist: **LM is its own HyperNetwork** [13] — only attention-mask change needed, no separate predictor.
- **Compressive/memory transformers**: Compressive Transformer [27] compresses past activations via a learned conv op + auxiliary reconstruction loss. Gist differs: (1) compression is the LM's *own* attention (input-dependent, gated by gist token), (2) learned jointly via the LM loss (no aux loss), (3) goal is instruction caching not long-range LM.
- **Sparse attention**: gisting = an *input-dependent sparse attention pattern* tailored to caching arbitrary-length prompt spans (unlike fixed sliding-window methods).

---

## 9. Limitations & Future Work

- Compression is **lossy** → loss of nuance; practitioners must validate the compute/accuracy tradeoff before deploying, esp. on edge cases.
- **Future**:
	- Retrofit a *frozen* LM by training a *smaller* model to do the compression (when finetuning the big LM is infeasible).
	- Biggest gains for **very long prompts** (e.g. large $k$-shot exceeding a context window).
	- **"Gist pretraining"**: first learn to compress arbitrary spans of natural language (insert gist tokens into LM / T5 span-corruption objective) before prompt-compression finetuning.

---

## Training config (App B, Table A.1)

| Hyperparam | LLaMA-7B | FLAN-T5-XXL |
|---|---|---|
| num steps | 3000 (~3 epochs) | 16000 (~2 epochs) |
| batch size | 128 | 16 |
| learning rate | 2e-5 | 5e-5 |
| warmup ratio | 0.03 | 0 |
| precision | bf16 | bf16 |
| optimizer | AdamW | AdamW |
| max seq len (train/eval) | 512 (768 Human) | in=128/out=256 (384/384 Human) |
| DeepSpeed | ZeRO stage 3, 4×A100-80GB | ZeRO stage 3, 4×A100-80GB |

- Greedy decode at eval (beam=4 gave little gain). Train time ~7h (LLaMA) / ~25h (T5). ChatGPT eval: ~22.5k judgments, ~$29 total.

---
---

## Implementation (codebase)

> Repo: `gisting/` (HF Transformers + Hydra + DeepSpeed). Core idea = subclass HF LLaMA/T5, add an extra `attention_mask_gist` input, build it in the data collator. All paths below relative to `gisting/`.

### File structure
- `src/data/gist.py` — **mask generation** (the listing from App A, productionized).
- `src/data/alpaca/collator.py` — **gist token insertion + mask construction** at batch time. Two collators: `DataCollatorForAlpaca` (T5), `DataCollatorForAlpacaCLM` (LLaMA, left-padded).
- `src/gist_llama.py` — `GistLlamaForCausalLM`: LLaMA subclass that consumes the 4D gist mask.
- `src/gist_t5.py` — `GistT5ForConditionalGeneration`: T5 subclass; handles self + cross gist masks.
- `src/gist_caching.py` — `GistActivations` dataclass: extract + cache only the gist KV/hidden states.
- `src/train.py` — adds `<GIST>` to vocab, inits embedding, wires collator + Trainer.
- `src/generation_utils.py` (`GistGenerationMixin`), `src/benchmarking.py`, `scripts/eval_*.py`.

### Where gist tokens are inserted (`collator.py`)
- The `<GIST>` token is added as a literal string into the prompt during collation, **between instruction and input**.
- **T5** (`collator.py:59-65`): `maybe_gist_str = "<GIST>" * num_gist_tokens` (no spaces).
	```python
	source = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nInput: {instance['input']}"
	```
- **LLaMA** (`collator.py:185-192`): space-joined `" ".join(["<GIST>"]*num_gist_tokens)`, then `\nOutput:` appended for CLM:
	```python
	prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nInput: {instance['input']}\nOutput:"
	```
- Default `num_gist_tokens` is overridden by experiment config (paper uses 1).
- LLaMA labels: prompt positions set to `-100` (loss only on completion+EOS) — `collator.py:220-222`.

### The mask functions (`src/data/gist.py`)
- `reverse_cumsum` (`gist.py:9-20`): right-to-left cumsum trick (no native torch op).
- `make_mask_pre_first_gist` (`gist.py:23-42`): `(inputs == gist_token).cumsum(-1) >= 1` → True from first gist onward. = **what post-gist tokens are allowed to see** (gist + everything after).
- `make_mask_post_last_gist` (`gist.py:45-66`): `reverse_cumsum(...) >= 1` → True up to & including last gist. = **what pre-gist tokens see**.
- `make_gist_mask` (`gist.py:69-118`): builds 4D `(B, 1, seq, seq)` mask.
	- `gist.py:111`: `mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)` — per **query row**, pick which key-visibility rule applies (pre-gist rows vs post-gist rows). This zeros the lower-left block.
	- `gist.py:114-115`: if an example has **no gist token**, leave mask all-ones (`has_gist` guard).
	- `gist.py:116-117`: AND with `(inputs != pad_token)` to mask padding.
	- Docstring example (the canonical picture):
		```
		  a b c G d
		a 1 1 1 1 0
		b 1 1 1 1 0
		c 1 1 1 1 0
		G 1 1 1 1 0
		d 0 0 0 1 1   <- post-gist token d cannot see a,b,c (pre-gist); can see G,d
		```
- `make_neg_control_mask` (`gist.py:121-167`): like gist but post-gist `d` can't even see `G` (row `d` = `0 0 0 0 1`). Implements the negative control (no task info reaches output).
- `make_pos_control_mask` (`gist.py:170-193`): all-ones (only pad masking) = vanilla attention.
- `get_gist_index` (`gist.py:196-222`): locate the contiguous gist span (asserts contiguity) — used for caching + position offsets.

### How the mask reaches the model
- **T5** (`collator.py:108-145`): writes results directly into the standard HF keys.
	- gist condition: `model_inputs["attention_mask"] = make_gist_mask(...).squeeze(1)` (T5 wants 3D `(B, seq, seq)`), and a separate **cross-attention** mask `cross_attention_mask = make_mask_pre_first_gist(...)` so decoder can't see pre-gist prompt.
	- neg control: cross-attn mask = `1 - make_mask_post_last_gist(...)` (decoder can't see up to/including any gist).
	- pos control: leave `attention_mask` untouched, cross = standard.
	- T5 self-attn consumes the 3D mask via HF's `get_extended_attention_mask` (`gist_t5.py:547`), which broadcasts the provided mask to all heads; cross-attn via `invert_attention_mask` (`gist_t5.py:565`).
- **LLaMA** (`collator.py:269-285`): keeps the **standard causal** `attention_mask` (all-ones padding mask) and ships the gist mask in a *separate* key `attention_mask_gist` (also a `prompt_attention_mask_gist` for the cache path). The 4D shape `(B,1,seq,seq)` is kept (not squeezed).
	- **Combination happens inside the model** (`gist_llama.py:529-542`):
		```python
		attention_mask = self._prepare_decoder_attention_mask(...)   # causal + pad, additive (0 / -inf)
		attention_mask_gist_float = torch.full_like(attention_mask, finfo.min)
		attention_mask_gist_float = attention_mask_gist_float.masked_fill(attention_mask_gist.bool(), 0.0)
		attention_mask = attention_mask + attention_mask_gist        # add the two additive masks
		```
		i.e. gist mask is converted from {0,1} to additive {−inf, 0} and **summed onto the causal mask** before being added to attention logits in `GistLlamaAttention.forward` (`gist_llama.py:234`). The lower-left block thus gets −inf and is softmaxed to ~0.
	- `GistLlamaAttention` / `GistLlamaDecoderLayer` / `GistLlamaModel` (`gist_llama.py:128-618`) are near-verbatim copies of HF LLaMA, threaded with the extra `attention_mask_gist` + `gist_offset` args.

### Adding the gist token to the model (`train.py:186-205`)
```python
tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
model.resize_token_embeddings(len(tokenizer))
# init new embedding = mean of existing embeddings (Hewitt's vocab-expansion trick)
if is_t5:
    model.shared.weight[-1] = model.shared.weight[:-1].mean(0)
elif is_llama:
    model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight[:-1].mean(0)
    model.lm_head.weight[-1]            = model.lm_head.weight[:-1].mean(0)
gist_token = tokenizer.additional_special_tokens_ids[-1]
```
- New gist embedding initialized to the **mean of existing token embeddings** (not random) → more stable start.
- Whole model is finetuned (not just the gist embedding) — gisting is a flavor of full IFT.

### Gist caching at inference (`gist_caching.py`)
- `GistActivations.from_model_outputs` (`gist_caching.py:34+`): from a forward pass with `use_cache=True`, slice out **only the gist positions'** KV (`past_key_values`) + last hidden state into a small fixed-size `(B, heads, num_gist_tokens, dim)` buffer. `gist_indices` retained for position-embedding offsets.
	- `cache_all=True` option caches everything up to gist end (used for positive-control decoder).
- At decode, `GistLlamaModel.forward` (`gist_llama.py:506-513`) accepts `gist_activations` → sets `past_key_values` + `gist_offset` so RoPE positions stay correct (`GistLlamaRotaryEmbedding.forward` applies the `gist_offset`, `gist_llama.py:114-117` — noted as batch-size-1 only).
- T5 path (`gist_t5.py:717+`) prepends cached gist `last_hidden_state` to the encoder output the decoder cross-attends to.

### Training infra
- `src/conf/` Hydra configs (`model/llama-7b.yaml`, `model/flan-t5-xxl.yaml`, `experiment/`), `gist.condition ∈ {gist, pos_control, neg_control}`, `gist.num_gist_tokens`, `gist.add_gist_token`.
- `ds_configs/stage3.json` = DeepSpeed ZeRO-3; launched via `run_deepspeed.sh`.
- `check_correctness=True` (LLaMA collator, `collator.py:199-217`): asserts separately-tokenized prompt+completion == jointly-tokenized (skips batch elements where they diverge, usually empty outputs).
