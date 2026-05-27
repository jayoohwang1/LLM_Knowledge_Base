# SAGE-KV: Self-Attention Guided KV Cache Eviction

**Paper**: "LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction for Efficient Long-Context Inference" — Wang et al., SambaNova Systems, ICLR 2025 (arXiv 2503.08879).

---

## TL;DR

- **Problem**: long-context LLMs (128K–1M) bottlenecked by KV cache memory + attention latency.
- **Insight**: after the prefill pass, attention is highly sparse and **the model already "knows" which past tokens matter** — the last input token's attention scores reveal which evicted-region tokens are critical, per head.
- **Method**: one-shot top-$k$ KV selection at the **head level** using last-token attention scores, performed once after prefill. Decoding then proceeds on a much smaller cache with no per-step selection.
- **Result on LongBench**: matches full attention; ~4x memory efficiency over StreamLLM (static), ~2x over Quest (dynamic block-wise), with better accuracy than both.

---

## Motivation & Positioning

- **KV cache scaling pain**: cache grows linearly with $N$ context length; attention is $O(N)$ per decode step. For 128K–1M contexts this dominates both memory and latency.
- **Two existing families** of sparsity-based mitigation:
	- **Static sparsity** (StreamLLM, H2O variants): predefined selection rule (e.g. keep sink + recent window).
		- **Pros**: cheap, no per-step selection.
		- **Cons**: discards information from the *middle* of context $\Rightarrow$ accuracy drops on tasks where the answer lives there.
	- **Dynamic sparsity** (Quest, InfLLM, RetrievalAttention, ShadowKV, InfiniGen): adaptive per-step token/block selection.
		- **Pros**: higher accuracy.
		- **Cons**:
			- need to keep **full KV cache** as a candidate pool $\Rightarrow$ memory savings limited (often offloaded to CPU $\Rightarrow$ retrieval latency).
			- per-step retrieval / ANN / chunk-pooling adds compute + hyperparameters (chunk size, ANN index, etc.).
			- block/chunk pooling weakens precise token-level retention.
- **SAGE-KV's pitch**: combine the **single-pass simplicity of static methods** with the **context-adaptive selection of dynamic methods**.
	- One-shot decision at the end of prefill $\Rightarrow$ no per-step selection, no retained full cache.
	- Selection is per-head (each KV head picks its own top-$k$) $\Rightarrow$ adaptive to learned head specialization.

---

## Core Insight: "LLMs Know What to Drop"

- After prefill, the attention map of the **last input token** (which is typically the end of the query) over the rest of the sequence is sharply peaked on a small set of relevant tokens.
- This mirrors the long-known observation (LLM2Vec, NV-Embed, etc.) that **the last token's hidden state acts as a sentence embedding** in causal decoder LLMs — its query vector is "summary-aware" of context + query.
	- **Analogy**: like using the [CLS] token in BERT as the pooled representation, but emerging implicitly in causal LMs because last-token attends to everything.
- $\Rightarrow$ Use that last-token query as a **single saliency probe** to score every past KV. No training, no aux model, no per-step recompute.
- Each **head** has its own attention distribution $\Rightarrow$ each head picks its own top-$k$ (different heads care about different tokens, consistent with retrieval-head / streaming-head literature, e.g. DuoAttention).

---

## Method

### Setup / Notation

- Input sequence $s = [t_i]_{i=1}^N$ = context concatenated with query.
- Prefill produces per-layer KV cache $\mathbf{P}^l = [(\mathbf{k}_i^l, \mathbf{v}_i^l)]_{i=1}^N$.
- $H_q$ = number of query heads, $H_{kv}$ = number of KV heads, $G = H_q / H_{kv}$ = GQA group size.
- $d_h$ = head dim; $H_q \cdot d_h = d$ hidden dim.

### Step 1 — Partition the full KV cache (per layer)

Split into 4 contiguous regions:

| Region | Notation | Length | Role |
|---|---|---|---|
| Sink / initial tokens | $\mathbf{S}^l = \mathbf{P}^l_{1:S}$ | $S$ | always kept (attention sink, per StreamLLM) |
| Eviction candidates | $\mathbf{E}^l = \mathbf{P}^l_{S+1:S+E}$ | $E$ | the middle — what we compress |
| Recent window | $\mathbf{R}^l = \mathbf{P}^l_{S+E+1:N-1}$ | $R = N{-}1{-}(S{+}E)$ | always kept (local context) |
| Last token | $\mathbf{P}^l_N$ | 1 | both kept and used as the query probe |

- **Why keep sink + recent fixed**: well-established that both regions get disproportionately high attention across all heads (StreamLLM "attention sink"); selecting them via attention scores would just rediscover them.

### Step 2 — Representative top-$k$ selection (the key step)

- Take $\mathbf{q}^l \in \mathbb{R}^{H_q \times d_h}$ = query projection of the **last token** at layer $l$.
- Compute attention scores against the eviction-candidate keys $\mathbf{E}^l$:
$$
\mathbf{a}^l_h = \mathrm{softmax}\!\left(\frac{\mathbf{q}^l_h \cdot (\mathbf{K}^l_{\mathbf{E}})^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^E \quad \text{for each head } h
$$
- For each head, pick the **top-$k$ indices** by $\mathbf{a}^l_h$ to form $\mathbf{E}^l_{\text{top}_k}$.
- **Per-head selection** $\Rightarrow$ different heads keep different tokens; with GQA the union is taken at the KV-head level.
- This is a **one-shot, training-free, hyperparameter-light** scoring rule.

### Step 3 — Construct the reduced cache

$$
\mathbf{C}^l = \mathrm{Concat}(\mathbf{S}^l,\ \mathbf{E}^l_{\text{top}_k},\ \mathbf{R}^l,\ \mathbf{P}^l_N), \qquad |\mathbf{C}^l| = S + k + R + 1
$$

- The full $\mathbf{P}^l$ is **discarded** after this — true memory savings, unlike methods that keep the full cache as a candidate pool.

### Step 4 — Decoding

- Standard causal decoding over $\mathbf{C}^l$.
- Each newly generated token appends to the **recent window** $\mathbf{R}$; oldest entry of $\mathbf{R}$ is evicted (sliding window).
- Sink + top-$k$ middle are **frozen** for the entire generation.

### Positional encoding subtlety

- SAGE-KV uses **absolute positional encoding** on the reduced cache (preserves original token positions $\Rightarrow$ semantic distance preserved).
- Contrast with HF StreamLLM, which has a known bug applying RoPE *after* cache storage, misaligning positions (esp. bad for Qwen2.5). The paper's fixed StreamLLM variants:
	- **StreamLLM$_R$**: stores pre-RoPE keys, applies relative-position RoPE within sliding window.
	- **StreamLLM$_{\text{Abs}}$**: absolute positional encoding (what SAGE-KV uses).

---

## Hyperparameter Recipe (Appendix A.2)

- Given total token budget $B$ and GQA group $G = H_q / H_{kv}$:
	- sink size $S = B/4$
	- per-head top-$k$ = $k = B / (2G)$
	- recent window $R = B - S - G \cdot k$ (with the budget split such that the multi-group selections fit)
- **Llama3.1-8B-Instruct, Llama-3-8B-ProLong-512k** (GQA group $G = 4$): $B = 8192 \Rightarrow S = 2048,\ k = 1024,\ R = 2048$.
- **Qwen2.5-7B-Instruct** ($G = 7$): $B = 8192 \Rightarrow S = 2048,\ k = 512,\ R = 8192 - 2048 - 7 \cdot 512 = 2560$.
- All baselines compared at the **same $B = 8192$** token budget.

---

## Experiments

### Setup

- **Benchmark**: LongBench — 15 tasks across QA, summarization, retrieval, code analysis (2wikimqa, gov-report, lcc, multifieldqa, mul-news, narrativeqa, passage-retrieval, qasper, repobench-p, trec, hotpotqa, musique, qmsum, samsum, triviaqa).
- **Models**:
	- Llama3.1-8B-Instruct (128K)
	- Llama-3-8B-ProLong-512k-Instruct
	- Qwen2.5-7B-Instruct (128K)
- **Baselines**: Full Attention, StreamLLM (HF + their fixed $_R$ + $_{\text{Abs}}$ variants), Quest, InfLLM.

### LongBench Accuracy (Table 1) — Headline Numbers

Average across 15 tasks:

| Method | Llama3.1-8B | Llama3-ProLong-512k | Qwen2.5-7B |
|---|---|---|---|
| **Full Attention** | 53.01 | 47.74 | 52.13 |
| StreamLLM (HF) | 50.14 | 44.86 | **35.30** (broken) |
| StreamLLM$_R$ (fixed) | 51.13 | 44.88 | 48.64 |
| StreamLLM$_{\text{Abs}}$ (fixed) | 51.18 | 44.85 | 48.70 |
| Quest | 51.27 | 46.80 | 50.77 |
| InfLLM | 50.29 | 43.08 | 47.46 |
| **SAGE-KV** | **52.49** | **47.64** | **51.19** |

- **SAGE-KV is within ~0.1–0.9 pts of full attention** across all three models at $B=8192$.
- Beats every other budgeted baseline on all three models.
- Notably **matches full attention on passage retrieval (100.0)** for both Llama models — needle-in-haystack works because the per-head top-$k$ picks up the needle.

### Memory Efficiency (Fig 2, Llama3.1-8B, 8-task subset)

Sweeping token budget $B \in \{0.5k, 1k, 2k, 4k, 8k\}$:

- **vs StreamLLM**: SAGE-KV at $B=2k$ matches StreamLLM at $B=8k$ $\Rightarrow$ **~4x memory efficiency**.
- **vs Quest**: SAGE-KV at $B=4k$ matches Quest at $B=8k$ $\Rightarrow$ **~2x memory efficiency**.
	- Quest stays competitive at large budgets but loses ground at tight budgets because chunk-pooled scores lose individual important tokens.
- All methods improve monotonically with $B$; SAGE-KV curve is the closest to full attention across the whole range.

### Why SAGE-KV beats each family

- **vs static (StreamLLM)**: static throws away the middle $\Rightarrow$ catastrophic when answer lives mid-context (e.g. passage-retrieval drops from 100 $\to$ ~70–80). SAGE-KV's top-$k$ over the middle recovers those tokens.
- **vs dynamic block-wise (Quest, InfLLM)**: block pooling dilutes individual token signal; per-step selection has runtime overhead and still keeps the full cache around. SAGE-KV does **token-level** selection (finer granularity) **once** (cheaper) and **drops the rest** (real memory savings).

---

## Why This Works — Intuition Deep Dive

- **Self-attention head specialization**: prior work (Retrieval Heads, DuoAttention) shows certain heads behave like retrievers vs streamers. Per-head top-$k$ naturally exploits this — retrieval heads pick few sharp tokens, streaming heads end up keeping more locally diffuse tokens (their top-$k$ will look like a contiguous-ish region).
- **Last-token query as oracle**: in decoder-only models trained with causal mask, the last token sees the entire sequence and aggregates it. Its attention pattern is essentially what the *next* token would attend to as well (modulo a small shift), so the selection is approximately optimal for the first few generated tokens.
- **Why one-shot works for short outputs**: LongBench tasks are short-answer (QA/retrieval/summarization). The same top-$k$ remains relevant across the few hundred decode steps. The authors flag this as a limitation for long-output generation.

---

## Ablations / Design Choices Implied by Results

- **Token-level vs block-level**: SAGE-KV (token) > Quest/InfLLM (block) on accuracy at matched budgets.
- **Head-level vs uniform selection**: per-head top-$k$ keeps a richer union than picking a single global top-$k$ shared across heads.
- **Single-pass vs per-step**: single-pass avoids decode-time overhead with no measurable accuracy cost on short-output tasks.
- **Positional encoding**: absolute positioning on the reduced cache > HF's broken post-RoPE storage; the gap is especially visible on position-sensitive Qwen2.5 (HF StreamLLM collapses from ~50 to ~35 avg).

---

## Limitations / Future Work

- **Short outputs only**: single one-shot selection may staleify during long generation; authors propose **interval-based refresh** (re-select every $T$ tokens) as future work.
- **No latency/throughput numbers** reported — paper claims "matches inference speed of static sparse methods" but no wall-clock table.
- **No training-time/finetuning ablation** — purely inference-time.
- **No analysis of head-level top-$k$ overlap** — would be informative to see how much heads agree vs diverge.
- **Sink/recent sizes fixed** at $B/4$ each; no sensitivity analysis reported on these splits.
- **Doesn't address prefill cost** — full attention is still run during prefill to get the KV cache and last-token query (only the *post-prefill decode* benefits).

---

## Relationship to Other Work

- **StreamLLM** (Xiao 2024c): sink + recent, no middle $\Rightarrow$ SAGE-KV = StreamLLM + adaptive middle top-$k$.
- **H2O** (Zhang 2023): heavy-hitter eviction using accumulated attention; SAGE-KV uses only the last token's attention (simpler, one-shot).
- **Quest** (Tang 2024): per-step block-wise top-$k$ on chunk min/max keys; SAGE-KV is token-wise + one-shot.
- **InfLLM** (Xiao 2024a): sink + recent + block-wise dynamic; SAGE-KV drops the dynamic block machinery.
- **DuoAttention** (Xiao 2024b): identifies retrieval vs streaming heads — SAGE-KV's per-head selection is consistent with this view without explicit head classification.
- **RetrievalAttention** (Liu 2024): ANN over KV; orthogonal — needs full cache pool, while SAGE-KV discards it.
- **LLM2Vec / NV-Embed**: justifies the "last-token query is a sequence embedding" assumption.

---

## Practical Notes

- **Drop-in**: works with vanilla HF Transformers / LLaMA / Qwen attention kernels — only requires:
	1. running prefill to populate $\mathbf{P}^l$,
	2. computing one extra attention score vector per head over the eviction region,
	3. gather/index op to build the reduced cache,
	4. resetting positional indices (absolute) on the reduced cache.
- Works with GQA out of the box (per-KV-head selection respects groups).
- **No public code release at time of writing. Designed to be drop-in compatible with HF Transformers / LLaMA / Qwen attention kernels.**
