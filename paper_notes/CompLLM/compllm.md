# CompLLM: Compression for Long Context Q&A

> **arXiv** 2509.19228v1 (23 Sep 2025) · Berton, Unnikrishnan, Tran, Shah (Amazon / UCF CRCV)
> **TL;DR:** Soft context compression that compresses **per-segment, independently** (not holistically) → gains **linear scaling**, **train-short/test-long generalization**, and **cross-query reusability/caching**. At compression rate $C=2$: up to **4x TTFT** speedup, **2x smaller KV cache**, perf on par or better than uncompressed at long context.

---

## Motivation

- **Long-context Q&A** is a dominant LLM use case (codebases, web agents, doc QA, RAG).
- **Self-attention is $O(N^2)$** → long contexts unfeasibly expensive. Want to cut compute as $N$ grows.
- **Context compression** literature splits into:
	- **Hard compression**: synthesize shorter *natural-language* prompt (token/sentence pruning, paraphrasing).
		- **+** interpretable, usable through closed APIs (compress locally, send text).
		- **−** lower compression rates, bigger accuracy drops.
	- **Soft compression**: compress into *latent representations* (continuous).
		- **+** continuous, end-to-end trainable, not confined to natural-text domain → higher compression rates & quality, often on par w/ uncompressed.
		- Two forms:
			- **(A) latent embeddings**: $N_1$ vectors of dim $D$ ($N_1$ = compressed seq len). ← CompLLM is here.
			- **(B) KV-cache compression**: dim $N_2 \times L \times D \times 2$ ($L$ = layers, $\times 2$ = key+value per token).
- **Gap**: soft methods keep improving compression rate/accuracy but **real-world adoption is scarce**. CompLLM targets *deployment properties* over peak compression ratio.

---

## Core Idea: Independent Per-Segment Compression

> Existing soft methods compress context **as a whole** → every input token affects the entire compressed representation. CompLLM instead splits context into **segments** (short sentences) compressed **independently**.

- **Split** text of length $N$ into $\tfrac{N}{S}$ segments of max length $S$ tokens.
- Each segment compressed **separately** by CompLLM.
- This one design choice yields **3 critical properties**:

| Property | Mechanism | Payoff |
|---|---|---|
| **Efficiency** | Attention only *within* a segment ($O(S^2)$ each), not over whole context | Compression cost $\tfrac{N}{S}\cdot O(S^2) = O(NS) = O(N)$ → **linear** in $N$ |
| **Scalability** | Model only ever sees small chunks at a time | **Train on short seqs (≤2K tokens), test on 100k+** while matching/beating uncompressed |
| **Reusability** | Segment A's compressed rep is independent of segment B | **Cache & reuse** compressed segments across queries (RAG, static codebases, doc sets) |

- **Contrast w/ KV-cache compression**: KV embeddings depend on *all preceding tokens* → holistic by design → **non-reusable**, **non-linear $O(N^2)$**. (Li et al. push KV-cache to seq-len-1; Corallo & Papotti make it query-aware.)

---

## Concept Embeddings (CEs) vs Token Embeddings (TEs)

> Key conceptual move: replace input **TEs** with fewer **CEs** that live in the *same latent space* and produce the *same LLM output*.

- **Token Embeddings (TEs)**: the vectors in the LLM's embedding table.
	- Finite vocab: **Gemma3 ≈ 262k** TEs, **Qwen3 ≈ 151k** TEs.
- **Concept Embeddings (CEs)**: vectors in the **same feature space** as TEs but
	- **not limited in number** (continuous, not from a table),
	- **fed directly to the LLM with no tuning** of the base model,
	- **unseen at training time** by the base LLM yet still understood.
- **Analogy (Fig 2)**: sentence *"golden dogs are called"*
	- = **4 TEs** (`golden`,`dogs`,`are`,`called`) → LLM → output,
	- = **2 CEs** (`golden dogs`, `are called`) → LLM → **same output**, but more compact.
- **CompLLM's objective**: a model that *extracts CEs from TEs* to cut sequence length → lower latency & memory in the LLM forward pass.
- CEs **encode similar info per vector** → similar outputs at shorter length.

---

## Architecture

- **CompLLM = same base LLM + LoRA + one linear layer** (inspired by Ge et al. 2024b).
	- **Reuses base LLM weights** (untouched) → low extra memory; base LLM still usable in standard mode when compression not needed.
	- **LoRA** adapts the frozen LLM for the compression task; **linear layer** appended on top.
- **I/O contract**: takes **$S$ (or fewer) embeddings** in, outputs **$\tfrac{S}{C}$ embeddings** out.
	- Satisfiable by many architectures (encoder-only, decoder-only LLMs, MLPs).
- **How CEs are produced**: feed CompLLM a length-$S$ segment, **append $\tfrac{S}{C}$ EOS embeddings**; the outputs at those EOS positions are taken as the $\tfrac{S}{C}$ CEs.
- **Settings**: $S = 20$, compression rate $C = 2$ → each segment of **20 TEs → 10 CEs**.
- **Single forward pass** handles any number of segments / any TE count → proportional CE count.
- Authors stress: **benchmarking compressor architectures is out of scope** — goal is to show CompLLM is feasible, not optimal.

---

## Training: Distillation on Answer Hidden States

> Real-world framing: **context is long → compressed (optionally offline); question is short → left uncompressed (online).**

- Base LLM: instruction-tuned $p_{\text{LLM}}(y \mid c, x)$ — $c$ = context, $x$ = instruction/question, $y$ = response.
- **Compressor** $p_{\text{CompLLM}}$ maps context $c$ (made of TEs) → compressed $\hat{c} = \text{CompLLM}(c)$ (made of CEs).
- **Distillation setup (Fig 3)**: query base LLM with either $c$ (teacher) or $\hat{c}$ (student); match **hidden activations on the answer segment** (denser signal than output-distribution matching).
	- **Teacher**: hidden states from uncompressed context.
	- **Student**: hidden states when conditioning on $\hat{c}$.
	- **Only answer-token outputs contribute to loss**; outputs over context/question embeddings ignored.
	- Model **decides what each CE encodes** (e.g. CE0 may cover only first TE, or first 3 TEs; low-info TEs may barely affect CEs).

**Per-layer Smooth-$L_1$ loss**, normalized by teacher activation scale:

$$\mathcal{L}^{(\ell)}_{\text{layer}}(c,x) = \frac{1}{\sigma^{(\ell)}(c,x)} \cdot \frac{1}{|A|\,d} \sum_{t \in A}\sum_{j=1}^{d} \text{SmoothL1}_\beta\big(\widetilde{H}^{(\ell)}_{t,j},\, H^{(\ell)}_{t,j}\big)$$

- $A$ = indices of the last $|A|$ **answer tokens**; $H^{(\ell)}_A$ = teacher hidden states at layer $\ell$; $\widetilde{H}^{(\ell)}_A$ = student.
- $\sigma^{(\ell)}(c,x) = \text{Std}(H^{(\ell)}_A)$ — **per-layer normalization** compensates for large cross-layer activation-norm variability (following Shen et al. 2025).

$$\text{SmoothL1}_\beta(u,v) = \begin{cases} \tfrac{1}{2}(u-v)^2/\beta, & |u-v| < \beta \\ |u-v| - \tfrac{\beta}{2}, & \text{otherwise} \end{cases}$$

- $\beta = 1$ (PyTorch default).

**Full objective** = expectation over context–instruction pairs, summed over all $L$ layers:

$$\mathcal{L}_{\text{comp}}\big(p_{\text{CompLLM}}, \mathcal{CX}\big) = \mathbb{E}_{(c,x)\sim\mathcal{CX}}\left[\sum_{\ell=1}^{L} \mathcal{L}^{(\ell)}_{\text{layer}}(c,x)\right]$$

- **No ground-truth labels needed** — answer $y$ (hence $A$) generated by the LLM itself, online during training or precomputed offline. Self-distillation flavor.

---

## Computational Complexity

LLM inference cost = **two components**:

1. **KV-cache prefill** (≈ **TTFT**): forward pass over input tokens.
	- Standard: **$O(N^2)$** (quadratic in prompt length $N$).
	- With CompLLM: context shrinks by $C$ → **$O(N^2 / C^2)$**.
2. **Next-token prediction** (autoregressive, KV cache prefilled): each new token attends to all prior → $O(N)$ per token; $T$ tokens → $O(NT)$.
	- With CompLLM: **$O(\tfrac{N}{C}T)$** (linear drop by $C$).
	- *(Footnote: precisely $O(N+T)$ per-token; since $N \gg T$ in long-context Q&A they simplify to $O(N)$.)*

- **Extra cost = compression time = $O(NS)$** (linear, from per-segment design).
	- **Negligible for large $N$** vs quadratic prefill: $O(NS) + O(\tfrac{N^2}{C^2}) = O(\tfrac{N^2}{C^2})$.
	- Can be done **offline** in many real cases (RAG docs known beforehand).
- **Asymptotic ratios** (vs uncompressed):
	- **TTFT / prefill → up to $C^2$** speedup (quadratic term) → **4x** at $C=2$.
	- **Next-token generation → $C$** speedup → **2x** at $C=2$.

> At test time, **context is compressed but question is left uncompressed** — LLM receives CEs (context) + TEs (question). Matches deployment: contexts long & cacheable offline, questions short & online.

---

## Results

### Inference Speed (Fig 1, Fig 4) — Gemma3-4B, BFloat16, B200 GPU, PyTorch compile, $C=2$

- **Generate 1 token (= TTFT / prefill)**: latency ratio *with* vs *without* CompLLM **→ ~4x ($C^2$)** asymptotically.
	- **Compression time scales linearly** → asymptotically negligible vs the curve.
	- **TTFT ≈ cache prefill time.**
- **Generate 10k tokens**: next-token prediction dominates → ratio **→ 2x ($C$)** asymptotically.
- **Online vs offline**: green line (offline) = orange (online) − compression time; offline removes compression cost from critical path entirely.

### Accuracy (Table 1) — LOFT benchmark (128k-token long-context, 5 datasets)

> LOFT stress-tests long-context capability of frontier LLMs (Gemini 1.5 Pro, GPT-4o, Claude 3 Opus). CompLLM lets **much smaller open models** improve long-context ability.

| Model | HotpotQA | Musique | NQ | Qampari | Quest |
|---|---|---|---|---|---|
| **Gemma3-4B** | 0.02 | 0.01 | 0.02 | 0.00 | 0.00 |
| **+ CompLLM** | **0.33** | **0.13** | **0.38** | **0.14** | **0.09** |
| **Qwen3-4B** | 0.07 | 0.00 | 0.01 | 0.00 | 0.00 |
| **+ CompLLM** | 0.07 | **0.07** | **0.26** | **0.05** | **0.08** |

- **CompLLM dramatically improves** small-model accuracy at 128k context (base models near-zero on most datasets).
- **Hypothesis for gains beyond parity**: fewer tokens → **reduced attention dilution** → better use of long context.
- Competitive w/ standard pipeline at **short** contexts; **surpasses** uncompressed at **very long** contexts.

---

## Key Takeaways

- **Independence is the whole trick**: compressing segments separately (vs holistically) is what buys linear scaling, train-short/test-long generalization, and caching/reuse — the properties that matter for deployment.
- **CEs ≈ "denser tokens"** in the same embedding space; base LLM consumes them untouched (only LoRA + linear added).
- **Distill on answer hidden states**, per-layer normalized Smooth-$L_1$, no labels needed.
- **$C=2$ modest rate by design** → trades peak compression for practicality; still 4x TTFT / 2x KV / 2x gen, plus accuracy wins at 128k.
- **No public code release.**
