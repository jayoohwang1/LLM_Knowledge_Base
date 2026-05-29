# In-context Autoencoder for Context Compression (ICAE)

**arXiv** 2307.06945 · **ICLR 2024** · Ge, Hu, Wang, Wang, Chen, Wei (Microsoft) · code+models: `github.com/getao/icae`

> **TL;DR**: Use the LLM itself as an autoencoder to squash a long context into a small set of **memory slots** (soft embeddings). LoRA-adapted LLM = encoder; frozen LLM = decoder conditioning on those slots. Pretrain w/ **autoencoding (AE) + language modeling (LM)**, then instruction-finetune. ~$4\times$ compression on Llama, +~1% params, $2\text{–}7\times$ inference speedup.

---

## Core Idea

- **Context compression** angle on long-context (vs architectural attention hacks like Longformer/Performer/RMT/Unlimiformer).
	- **Motivation**: same info representable at many "lengths" in an LLM. Fig 2 ex: 2572 chars = 512 (sub)word tokens — can we go to **128 memory slots** w/o losing response quality?
	- Orthogonal to sparse-attn / retrieval methods → composable with them.
- **Memory slots** $(\tilde{m}_1,\dots,\tilde{m}_k)$ = $k$ continuous vectors in the LLM's hidden/embedding space that **stand in for the original context** $c=(w_1,\dots,w_L)$, $k \ll L$.
	- Target LLM conditions on slots **on behalf of** the real context to answer arbitrary prompts.
- **Cognitive-science framing**: slots ≈ working memory; extensive self-supervised pretraining ≈ memory training. Claims LLM memorization pattern mirrors humans (selective, knowledge-dependent).

---

## Architecture (§2.1, Fig 3)

Standard AE (encoder→bottleneck→decoder) but **in-context** like **Gisting** (Mu 2023) and **AutoCompressor** (Chevalier 2023).

| Component | What | Trainable? |
|---|---|---|
| **Encoder** | LLM + **LoRA** adapter + learnable **memory-token embeddings** $e_m(\cdot)$ | yes (LoRA + $e_m$ only) |
| **Decoder** | the **same target LLM, untouched/frozen** | no |
| **Bottleneck** | $k$ memory slots = encoder's last-hidden-states at memory-token positions | — |

- **Encoding**: append $k$ learnable **memory tokens** $(m_1,\dots,m_k)$ to context $c$ → run LoRA-LLM → take outputs at those positions as slots $(\tilde m_1,\dots,\tilde m_k)$.
	- Encoder is **lightweight**: only adds a LoRA adapter + an embedding lookup for memory tokens on top of target LLM.
- **Decoding**: feed slots into **frozen** target LLM (compatibility guaranteed since same weights), optionally + a special task token + prompt; generate via teacher forcing.
	- Using untouched LLM as decoder ⇒ slots are "native" to the target model (unlike Gist tokens which require the gist-tuned model).
- **`[AE]` special token** appended after slots in decoder to signal the autoencoding task.

---

## Pretraining (§2.2)

Two self-supervised objectives, **no annotation** → train on massive text (**the Pile**). Combined loss:
$$\mathcal{L}_{\text{pretrain}} = \lambda\,\mathcal{L}_{\text{AE}} + (1-\lambda)\,\mathcal{L}_{\text{LM}}, \quad \lambda \in [0.4, 0.6]\ \text{best}$$

### Autoencoding (AE)
- Reconstruct original context $c$ (length $L$) from its slots.
$$\mathcal{L}_{\text{AE}} = \max_{\Theta_{\text{LoRA}},\,e_m} P(c \mid m_1\dots m_k;\ \Theta_{\text{LLM}}, \Theta_{\text{LoRA}}, e_m)$$
- Decoder input: `slots + [AE]`, target = the full context.
- Only $\Theta_{\text{LoRA}}$ and $e_m$ are optimized; $\Theta_{\text{LLM}}$ frozen.

### Text Continuation (LM)
- AE alone too simple / overfits to copying → add **next-token continuation** for generalizable reps.
- Predict continuation $o=(w_{L+1},\dots,w_{L+N})$ from slots of $c$.
$$\mathcal{L}_{\text{LM}} = \max_{\Theta_{\text{LoRA}},\,e_m} P(o \mid m_1\dots m_k;\ \Theta_{\text{LLM}}, \Theta_{\text{LoRA}}, e_m)$$
- (Fig 7) decoder input `slots [+lm tok] + continuation`.

> **Ablation (Table 5)**: AE+LM > AE-only > LM-only. Combining helps (+~1 win/lose pts each direction over single objective).

---

## Instruction Fine-tuning (§2.3)

- Pretrained slots are good at restoration/continuation but real use = **respond to prompts** grounded in context.
- Fine-tune on **PWC (Prompt-with-Context)** dataset: triples **(context, prompt, response)**, **240k train / 18k test**.
	- Built by sampling 20k Pile texts → GPT-4 generates 15 prompts/text (10 specific + 5 generic: rephrase/summarize/title/keywords/continue) w/ answers (Appendix C, Listing 1).
$$\mathcal{L}_{\text{FT}} = \max_{\Theta_{\text{LoRA}},\,e_m} P(r_1\dots r_n \mid m_1\dots m_k, p_1\dots p_m;\ \Theta_{\text{LLM}}, \Theta_{\text{LoRA}}, e_m)$$
- Decoder input: `slots + [FT] + prompt + response(teacher-forced)` (Fig 8).

---

## Experimental Setup (§3.1)

- **Data**: Pile (pretrain), PWC (FT). Default train max token len (excl. slots) = **512**.
- **Target LLM**: **Llama / Llama-2 (chat) 7B/13B** (code release uses **Mistral-7B** too).
- **Config**: LoRA on **q & v** projections, default **mem slots $k=128$**, **LoRA rank $r=128$** → ~**1% extra params**.
- **HW/HP** (Appendix A, Table 8): 8×A100-80GB, bf16, AdamW, lr 1e-4 (pretrain)/5e-5 (FT), batch 256, warmup 300, 200k updates (pretrain)/30k (FT), clip 2.0.

---

## Results

### Pretrained ICAE — Autoencoding (§3.2.1)
- Llama-7b, $k=128$: overall loss **< 0.05**; **near-100% BLEU/EM for context ≤300 tokens**.
- Degrades past **400** (128 slots run out of capacity); at **500**: median BLEU >0.98, median EM ~0.6 (≈ first 300 words perfect).
	- **EM metric**: matching-prefix-length / total. 256/512 perfect → EM 0.5.
- **Memory size effect (Fig 5)**: $k=128$ holds BLEU >95% even at len 500; $k=64,32$ collapse → **>4× compression is hard** w/ this setup.

### Continuation (Table 1)
| Compression | PPL (orig ctx) | PPL (128 slots) | Δ |
|---|---|---|---|
| 128→128 (1×) | 9.99 | 10.15 | +0.16 |
| 256→128 (2×) | 9.45 | 9.77 | +0.32 |
| 512→128 (4×) | 9.01 | 9.50 | +0.49 |

→ higher compression ⇒ bigger LM loss gap (expected).

### Memorization insight (Tables 2,3)
- Restoration errors look **human-like**: "large pretrained *language* model" → "large pretrained model"; "results prove" → "experimental evidence proves" (paraphrase, not garbage).
	- Consistent w/ Peng 2023: stronger model ⇒ less rote memorization needed.
- **Content-type matters (Table 3, 512-tok, $k=128$)**:

	| Content | Loss | BLEU |
	|---|---|---|
	| Normal text | 0.01 | 99.3 |
	| Patterned random (token_id+1) | 1.63 | 3.5 |
	| Completely random | 4.55 | 0.2 |
	→ slots compress *meaningful* text well, *random* text poorly ⇒ memorization ≈ understanding-driven (human-like).

### Fine-tuned ICAE (§3.2.2, Table 4)
- GPT-4 judge (win/lose/tie), per Mu 2023; response order randomized to debias.
- **Llama-7b ICAE ($k=128$)** on 128 slots **beats** Alpaca/StableLM-7b (which see full ~512-tok ctx): win+tie **73.1% / 81.3%**.
	- vs **GPT-4 (gold)**: only ~30% win+tie → big gap remains.
- **Llama-2-7b-chat** much better than Llama-1: $k=128$ win+tie ~54.6% vs its own full-ctx counterpart; improves as $k$↑ (256 → 77.8%).
- **Llama-2-13b-chat ($k=256$)**: 79.2% win+tie → **bigger LLM ⇒ better compression** (scalability claim).
- vs **GPT-4 128-token summary** (Table 5 last row): slots **win 34.1 / lose 17.6** (~2× win/lose) → slots **more compact & informative than natural-language summaries**.

### Ablations (Table 5, Llama-2-7b-chat)
- $k=128$ pretrained **>** $k=64$ pretrained (win 57.6) **>** $k=32$.
- **Pretraining critical**: $k=128$ pretrained vs $k=128$ no-pretrain → win **60.4** / lose 9.5. Non-pretrained hallucinates more (Table 9).
- Pretrained $k=64$ (8×) ≈ non-pretrained $k=128$ (4×) → pretraining buys ~2× effective compression.

### Scalability / Latency / Multi-span (§3.3)
- **Scalability (Table 6)**: 512→128 AE — Llama-7b/2-7b/2-13b → BLEU 99.1/99.5/99.8, loss 0.017/0.009/0.004. Bigger = better.
- **Latency (Table 7)**, fix gen len 128, Llama-7b:
	- 8×2048: **3.3×** total speedup; 32×512: **3.6×**; compute-intensive ~**3.5×**.
	- Compression is **cacheable** (textbooks/laws) → up to **>7×** speedup; 2048 slots vs 4096 ctx saves ~**20GB** GPU mem (fp16, no flash-attn).
- **Multiple spans (§3.3.3, Fig 6)**: chunk long ctx → compress each → concat slots.
	- Naive concat **fails** (never seen multi-span at train) → fix by adding a few multi-span samples in training (cf OpenAI "fill-in-middle").
	- 2048 slots ≈ 4096-tok ctx perplexity → memory represents **4× its length**.

---

## Related Work (§4)
- **Gisting (Mu 2023)**: prompt compression via gist tokens; limited to *short* prompts (task instructions), gist tokens tied to gist-tuned model. ICAE compresses long *contexts*, uses untouched LLM, LoRA-only.
- **AutoCompressor (Chevalier 2023)**: recursive summary vectors; complex training. ICAE simpler + scalable.
- **Wingate 2022** (KL soft-prompt): backprop per new prompt → expensive.
- **NUGGET (Qin & Van Durme 2023)**: agglomerative embeddings for enc-dec.
- Also: LLMLingua (NL prompt compression), "LLM = compressor" (Delétang 2023), textual-inversion (Gal 2022), style-token (Ge 2023).

## Limitations / Future
- ≤13B only (compute-limited); expects bigger gains on stronger LLMs.
- Future: multimodal slots (image/video/audio, possibly discrete) for cross-modal compact reps.

---
---

## Implementation (codebase)

Repo: `icae/code/` has **two implementations**. `icae_v1/` = original Llama-based (matches paper math, single-tensor mem). `icae_v2/` = scaled-up rewrite (Mistral default, **multi-span**, separate decoder for grad-checkpointing). Notes below mostly target **v2** (current/active) with v1 contrasts.

### File structure
- **v2** (`icae/code/icae_v2/`):
	- `modeling_icae_multi_span.py` — `ICAE` model class (encoder/decoder wiring, compress, forward).
	- `pretrain.py` — AE+LM pretraining entrypoint.
	- `instruction_finetune.py` — PWC fine-tuning entrypoint.
	- `training_utils.py` — tokenize fns (build slot/prompt/label seqs), dynamic-padding collator, `train_model` (HF `Trainer`).
	- `pretrained_inference.py` / `fine_tuned_inference.py` (+ `.sh`) — compress→generate inference.
- **v1** (`icae/code/icae_v1/`): `llama_icae_modeling.py` (`LlamaICAE`), `llama_icae_learning.py` (StableTrainer), `base/modeling_llama_icae.py` (patched Llama w/ `enable_lora` flag), bundled `peft/`.

### Token-ID layout (how mem slots are "created")
v2 `modeling_icae_multi_span.py:109-122` — extends vocab with reserved ID ranges:
```python
self.vocab_size = self.icae.config.vocab_size + 1     # +1 [PAD]
self.pad_token_id = self.vocab_size - 1
self.mem_size = training_args.fixed_mem_size          # default 128
self.vocab_size_with_mem = self.vocab_size + self.mem_size  # mem IDs ∈ [vocab_size, vocab_size+mem_size)
self.ae_token_id = self.vocab_size_with_mem + 0       # [AE]
self.lm_token_id = self.vocab_size_with_mem + 1       # [LM]
self.ft_token_id = self.vocab_size_with_mem + 2       # [FT]
self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3)
```
- **Memory tokens are virtual IDs**, not in the real tokenizer. Their embeddings live in a **separate learnable table**, not the base embed matrix:
	- `self.memory_token_embed = nn.Embedding(self.mem_size + 3, self.dim)` (`:133`) — holds the $k$ memory-token embeddings **+ the 3 special tokens** (`[AE]/[LM]/[FT]`).
- **`append_sequence`** = the fixed run of $k$ memory-token IDs appended to every encoder input (`:136`):
	```python
	self.append_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size, ...).unsqueeze(0)
	```
- `fixed_mem_size` **must be power of 2** (asserted in `pretrain.py:27`).

### Encoder + frozen decoder wiring
`modeling_icae_multi_span.py:96-154`:
- Two LLM copies loaded (flash-attn-2, bf16):
	- `self.icae` = encoder, wrapped `get_peft_model(self.icae, lora_config)` (`:129`) → **LoRA-adapted**.
	- `self.decoder` = **independent frozen copy** loaded only when training (`:106-107`), so it can use **gradient checkpointing** without touching the encoder graph (`:154`).
- `init()` (`:142-154`): `freeze_model(self.decoder)` + `.eval()`; optionally `load_state_dict` from `restore_from`; enable grad-ckpt on decoder.
- **At inference there is no separate decoder** — reuse `self.icae` with **`disable_adapter()`** so LoRA is off ⇒ the frozen base LLM acts as decoder (`:209-211`, and inference loop). Clever: one set of weights serves both roles.
- **LoRA config** (`pretrain.py:18-24`): `r = lora_r` (default 128; inference script uses **512**), `lora_alpha=32`, `dropout=0.05`, `bias="none"`, `task_type="CAUSAL_LM"`. PEFT defaults LoRA to **q & v proj** → matches paper.

> **v1 contrast** (`llama_icae_modeling.py`): single `self.icae` for both enc+dec; decoder pass done with `enable_lora=True/False` toggling via the **patched Llama** (`base/modeling_llama_icae.py`), instead of v2's `disable_adapter()`. Also optional **`MemoryHead`** (2-layer MLP w/ gelu, `:81-92`) on encoder outputs (off by default).

### Embedding splice (continuous slot injection)
The trick that turns hidden-states into "tokens" is **embedding replacement by ID range**. Encoder side, per segment (`:185-188`):
```python
segment_input_ids = torch.cat([segment_input_ids, self.append_sequence], dim=1)  # append k mem tokens
mem_flag = segment_input_ids >= self.vocab_size
segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)  # normal lookup
segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size)  # overwrite mem positions
```
- Run encoder, take **last hidden state at mem positions** as the slots (`:191-195`):
	```python
	segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True).hidden_states[-1]
	compress_outputs[seg*mem_size : mem_size*(seg+1)] = segment_compress_outputs[mem_flag]
	```
- `tokens_to_embeddings()` (`:221-225`) generalizes the same splice for arbitrary id seqs (used at inference for the `[AE]` prompt).

### Multi-span segmentation (v2-only)
`compute_num_segments` (`:157-160`):
```python
num_segments = math.ceil(total_length / (self.mem_size * self.mean_compression_rate))  # mean_compression_rate default 4
```
- Long input is split into `num_segments` equal chunks, **each compressed independently** into its own $k$ slots, then **concatenated** → `num_segments * mem_size` total slots (Fig 6 mechanism). Per-segment loop `:179-198` with `torch.cuda.empty_cache()` between segments.

### Decoder conditioning + loss (`forward`, `:163-218`)
- `prompt_answer_ids` already **contains placeholder mem IDs + special tokens + prompt + answer** (built in tokenize fns). Embedding spliced in two passes:
	```python
	prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
	decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)
	prompt_answer_embs[decoder_mem_flag] = compress_outputs           # inject the real slots (:201-203)
	special_prompt = prompt_answer_ids >= self.vocab_size_with_mem
	prompt_answer_embs[special_prompt] = self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size)  # [AE]/[LM]/[FT] embeds (:204-205)
	```
- Decoder forward then **standard shifted CE** (`:214-217`), `ignore_index=-100`:
	```python
	effective_logits = logits[:, :-1, :].reshape(-1, V)
	target_ids = labels[:, 1:].reshape(-1)
	loss = CrossEntropyLoss(ignore_index=-100)(effective_logits, target_ids)
	```

> **v1 loss** (`llama_icae_modeling.py:156-163`): slots are **prepended** via `torch.cat((memory_embedding, prompt_answer_embs), dim=1)` (paper-faithful), then logits sliced from `mem_size-1:-1`. v2 instead embeds mem IDs inline in `prompt_answer_ids` ("for easy implementation," comment `training_utils.py:105`).

### Training objectives in code (AE vs LM are data-level, not separate loss terms)
**Key**: v2 has **one CE loss**; AE vs LM is decided **per-example during tokenization** (`pretrain_tokenize_function`, `training_utils.py:84-123`):
- `text_extraction` (`:60-81`) splits a doc into `(a, b)`:
	- with prob `1 - lm_ratio` → **AE**: `a` = (random window of) the text, `b = []`.
	- else → **LM**: `a` = prefix context, `b` = continuation.
- Build decoder sequence:
	```python
	if ae:
	    prompt_ids = [mem[0]] * total_mem_length + [model.ae_token_id]   # slots + [AE]
	    answer_ids = a + [model.eos_id]
	else:  # lm
	    prompt_ids = [mem[0]] * total_mem_length (+ [lm_token_id] if add_special_token_for_lm)
	    answer_ids = b   # no eos
	labels = [-100]*len(prompt_ids) + answer_ids        # AE
	labels = [-100]*len(prompt_ids) + [-100]*leave_tokens_for_lm + answer_ids[leave_tokens_for_lm:]   # LM: skip first few continuation tokens' loss
	```
	- Note `prompt_ids` uses `mem[0]` repeated `total_mem_length` times — placeholders; the *actual distinct* slots come from the encoder splice in `forward`. Position-in-sequence + range-check is what matters, not the literal id value.
	- `min_tokens_for_lm` (default 64): only do LM if continuation long enough (else fall back to AE). `leave_tokens_for_lm` (default 8): first N continuation tokens get no loss.
	- **`lm_ratio`** controls AE:LM mix (the paper's $\lambda$); applied only to **train**, not dev (`pretrain.py:46`).
- **Instruction FT** (`instruct_ft_tokenize_function`, `:126-152`): `(input, prompt, answer)` columns. Sequence = `slots + [FT] + prompt`, answer = `answer + eos`. Wrapped in **Mistral instruct format** ids `[1,733,16289,28793] ... [733,28748,16289,28793]` (`<s>[INST] ... [/INST]`).

### Collator + trainer
- `DataCollatorForDynamicPadding` (`:155-175`): pads `input_ids`/`prompt_answer_ids` with `pad_token_id`, `labels` with `-100`; bypassed when batch size 1.
- `train_model` (`:11-57`): plain HF `Trainer`, auto-resume from last ckpt. (v1 used a custom `StableTrainer`.)

### Configs (`pretrained_inference_script.sh`)
```
BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
maxlen=5120  mem=128  r=512  mean_compression_rate=4
--fixed_mem_size 128 --lora_r 512 --bf16 --train False
```
- Released checkpoint: `huggingface.co/sggetao/icae/.../mistral_7b_pretrained_icae.safetensors`. Loaded via `safetensors.load_file` + `load_state_dict(strict=False)` → **only LoRA + `memory_token_embed` weights** are saved/restored (base LLM untouched) — that's the "~1% params."

### Inference flow (`pretrained_inference.py`)
1. Tokenize text → `model._compress(input_ids)` (`:228-261`, same per-segment encode as `forward`) → slots.
2. Prompt = `[[ae_token_id]]` → `tokens_to_embeddings` → embeds.
3. `decoder_input_embeddings = cat(slots, prompt_embs)`; greedy decode loop (`:66-79`) with `model.icae.disable_adapter()` (LoRA off = frozen decoder), KV-cache, stop on eos (id 2), argmax sampling, logits sliced to `:vocab_size-1` (exclude special ids).
