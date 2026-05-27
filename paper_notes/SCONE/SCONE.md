# SCONE: Scaling Embedding Layers in Language Models

**Authors**: Da Yu, Edith Cohen, Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Daogao Liu, Chiyuan Zhang (Google)
**Venue**: NeurIPS 2025 (arXiv 2502.01637, v3 Oct 2025)
**Name**: **S**calable, **C**ontextualized, **O**ffloaded, **N**-gram **E**mbedding

---

## TL;DR

- **Idea**: Augment a transformer LM with a second input embedding table indexed by **frequent n-grams (called f-grams)** rather than just tokens. The f-gram embeddings are **produced by a separate "f-gram transformer" during training** and **precomputed + offloaded to CPU/SSD for inference**.
- **Result**: A 1B accelerator-resident model + 1B f-grams beats a 1.9B dense baseline at **~half the inference FLOPS and GPU memory**.
- **Why it matters**: Decouples model capacity from on-accelerator footprint by **moving capacity into a giant key-value lookup table living in cheap memory**. Opens **two new scaling axes** (#f-grams, f-gram-model size) that don't grow inference accelerator cost.

---

## Motivation

- **Embedding layer is special**: token -> vector is *just a memory fetch*, no compute. So it can be **offloaded off-accelerator** (CPU RAM ~$2/GB or NVMe SSD ~$0.1/GB) vs. GPU/TPU HBM which is wildly more expensive.
- **Naive fix (grow vocab) fails**:
	- **Tail token problem**: with vocab of 2M, only **7.3%** of tokens get >100 gradient updates over 100M training tokens (vs **97.6%** for 32K vocab). Sparse updates -> bad embeddings for rare tokens.
	- **Output side blow-up**: vocab also increases the softmax/logits matrix on the accelerator (tied weights), and softmax cost is roughly linear in $|V|$.
	- Empirically pretraining GPT-2 with vocabs 32K -> 2M shows **BPC degrades past ~512K vocab** and GPU memory grows linearly.
- **Goal**: get the model-capacity gains of a huge embedding table **without** increasing decoding vocab, accelerator FLOPS, or accelerator memory.

---

## Core Method

### Setup notation

- **Token vocab** $V_{\text{token}}$, **token embedding** $\mathcal{T}: V_{\text{token}} \to \mathbb{R}^d$.
- **f-gram vocab** $V_{\text{f-gram}} \subseteq \bigcup_{n=2}^{K} V_{\text{token}}^{n}$ — set of frequent n-grams over tokens, up to length $K$.
- **f-gram transformer** $\mathcal{A}_{\text{f-gram}}: (\mathbb{R}^d)^{\le K} \to \mathbb{R}^d$ — small transformer that, given the token-embedding sequence of an f-gram, returns a single contextualized embedding (output of final position).
- **f-gram lookup table** $\mathcal{F}: V_{\text{f-gram}} \to \mathbb{R}^d$ — used at **inference only**, contains precomputed outputs of $\mathcal{A}_{\text{f-gram}}$ for every f-gram.
- **Main transformer** $\mathcal{A}_{\text{main}}$ + prediction head $\mathcal{D}$ (standard).

### Per-token embedding rule (Algorithm 1)

For each position $i$ in the input token sequence $(\sigma_1, \dots, \sigma_m)$:

1. Find the **longest matching f-gram ending at position $i$**: smallest $j' < i$ such that $(\sigma_{j'}, \dots, \sigma_i) \in V_{\text{f-gram}}$. Let $j$ be that index, else $j=i$.
2. If $j = i$ (no f-gram matched, single token) -> $e_i = \mathcal{T}(\sigma_i)$ (standard token embedding).
3. Otherwise:
	- **Training**: $e_i = \mathcal{A}_{\text{f-gram}}(\mathcal{T}(\sigma_j), \dots, \mathcal{T}(\sigma_i))$ — forward through the f-gram transformer.
	- **Inference**: $e_i = \mathcal{F}(\sigma_j, \dots, \sigma_i)$ — direct table lookup.

Resulting embeddings $(e_1, \dots, e_m)$ feed $\mathcal{A}_{\text{main}}$.

- **Note**: each token gets a **contextualized** embedding that depends on the up-to-$K$ preceding tokens (only those forming an f-gram). Crucially, output vocab is still $V_{\text{token}}$ — decoder/softmax unchanged.

### f-gram discovery (Section 3.1, Algorithm 3) — BPE-style

- Construct $V_{\text{f-gram}}$ by counting frequent n-grams over the token corpus, **not characters**. Inspired by BPE but **without iterative merging** (too expensive at trillion-token scale).
- **Algorithm**: $K-1$ linear scans over the training corpus.
	- For each $n \in [2, K]$ do one scan counting n-grams.
	- Min-frequency threshold of 5 to prune.
	- Pruning trick: if a candidate $(n+1)$-gram passes threshold, its $n$-suffix or $n$-prefix must already appear at least as many times -> skip $(n+1)$-grams whose constituents didn't pass.
- After scans, rank all collected n-grams (across $n=2,\dots,K$) by frequency, keep top-$S$ -> $V_{\text{f-gram}}$.
- **Empirical**: at 1T tokens, ~1.5B unique 4-grams, ~1.3B unique 3-grams etc. (Figure 3).
- Paper sets **$K=5$** as default after ablation.

### Why a transformer instead of direct lookup table during training?

- **Sparse-update problem again**: if you had a free-parameter embedding for each of millions/billions of f-grams, each f-gram gets gradient updates only when it appears. Most f-grams appear rarely -> embeddings stay near init.
- **Memory blow-up**: instantiating an embedding table for $10^9$ f-grams on GPU during training is infeasible.
- **Solution**: parameterize embeddings as a function $\mathcal{A}_{\text{f-gram}}(\text{token embeddings of f-gram})$. Gradients flow into a **shared** small model that benefits from *all* f-gram occurrences. Token embeddings $\mathcal{T}$ are reused.
- **Cost**: at training, for each f-gram occurrence you run a tiny transformer on a sequence of length $|\omega| \le K$. Since $K$ is small, dominant cost is the FFN.

---

## Inference Path / Offloading

- After training, **precompute** $\mathcal{F}(\omega) = \mathcal{A}_{\text{f-gram}}(\mathcal{T}(\sigma_{j}), \dots, \mathcal{T}(\sigma_{i}))$ for every $\omega \in V_{\text{f-gram}}$, store in lookup table.
- Two storage options studied (Section 4.3):
	- **System memory (RAM)**: dense embedding matrix + hash dict mapping f-gram -> row index.
	- **NVMe SSD**: **LMDB** (Lightning Memory-Mapped DB) with B+ tree directly mapping f-grams -> embeddings.
- **Per-token query**: at most $K-1 = 4$ lookups (try length-$K$, then $K-1$, etc., to find longest match).
- **Latency** (Figure 7, A100, $d=2048$, 16-bit):
	- In-memory, 10M f-grams, batch 1: **1.1 ms/token**; batch 16: **0.017 ms**.
	- 1B f-grams on NVMe, batch 1: **2.3 ms**; batch 16: **0.5 ms**.
	- Both well below **~10 ms/token** typical commercial-API decoding budget.
- **Space cost** (Table 2): 
	- $10^7$ f-grams -> 41.4 GB RAM (or 77.3 GB SSD).
	- $10^9$ f-grams -> 7.6 TB SSD (RAM infeasible).

---

## Two New Scaling Axes (the headline contribution)

Traditional scaling laws (Chinchilla, Jones 2021) say: spending more train compute on a fixed model size hits diminishing returns -> you must grow params -> grow inference FLOPS/memory. SCONE breaks this:

| Axis | What grows during training | Inference accelerator cost |
|---|---|---|
| Grow #f-grams $|V_{\text{f-gram}}|$ | training compute + off-accelerator storage | **unchanged** (more cache entries, lookup still $O(1)$) |
| Grow $\mathcal{A}_{\text{f-gram}}$ model size | training compute (more forward FLOPS per f-gram during train) | **unchanged** (only outputs cached; main model untouched) |
| (Traditional) grow $\mathcal{A}_{\text{main}}$ | train compute | **grows** (FLOPS + accelerator memory) |

- **Both new axes inflate train cost only, not inference cost** — uniquely valuable when inference $\gg$ training in TCO (production deployment, billions of queries/day).
- Connects to *test-time scaling* discussions (Snell 2025, Jones 2021): when inference is the cost bottleneck, you'd happily pay more at train time.

---

## Training Details

- **Joint training**: $\mathcal{T}$, $\mathcal{A}_{\text{f-gram}}$, $\mathcal{A}_{\text{main}}$, $\mathcal{D}$ trained jointly via next-token prediction loss (standard CLM).
- **f-gram model architecture**: same architecture as main model, but **discard the token embedding layer** (token embeddings shared from $\mathcal{T}$). Has its own absolute position embedding with max length = max f-gram length.
- **Compute accounting**: f-gram model adds FLOPS during training. Authors compensate via either:
	1. Train baselines long enough to near-convergence so additional compute wouldn't help; or
	2. **Reduce training tokens** for SCONE so total train FLOPS match baselines.
- **Pre-scan optimization (App F)**: precompute longest matching f-gram for each token in training set so training-time forward doesn't bottleneck on lookup.
- **Precision**: bfloat16. Optimizer: AdamW, wd=0.1, cosine LR, peak $2 \times 10^{-4}$ (WebText).

---

## Experiments

### Setting 1 — GPT-2 / WebText (Section 4.1)

- Three baseline sizes: **76M, 340M, 510M non-embedding params** (128M, 419M, 589M total incl token embedding). Train 80B tokens.
- Plus "double-sized" baselines (204M, 759M, 1099M) so SCONE-aug models have equal *training* params.
- Vary three knobs:

#### 4.1.1 Max f-gram length $K$
- Vary $K$ from 2 -> 8, fix $|V_{\text{f-gram}}| = 20M$.
- Frequency cutoff grows with $K$ (7 at K=2, 108 at K=8).
- **Perplexity drops between $K=2$ and $K=4$, plateaus after.** Average matched length saturates around 1.6-2 tokens (most matches stay short, longer f-grams are rarer).
- Default: $K=5$.

#### 4.1.2 Number of f-grams $|V_{\text{f-gram}}|$
- Scale from **512K -> 100M** f-grams.
- Perplexity decreases monotonically on WebText, mostly on WikiText-103.
- With 100M f-grams, 419M and 589M SCONE models **match or surpass 759M and 1099M dense baselines** at half the *non-embedding* (=inference accelerator) params.

#### 4.1.3 Scale $\mathcal{A}_{\text{f-gram}}$ size
- For fixed $|V_{\text{f-gram}}| = 100M$, vary f-gram model at 0.5x, 1x, 2x, 3x main-model non-emb params.
- Bigger f-gram model -> better perplexity, with **diminishing returns** beyond ~2x.
- Example: 419M main + 170M f-gram model: 23.4 PPL on WikiText-103, beats 589M dense (24.7). Scale to 1020M f-gram (1.439B total during train) -> 22.1 PPL, close to 1099M dense (21.9).
- **Key takeaway**: scaling $\mathcal{A}_{\text{f-gram}}$ keeps inference accelerator footprint fixed (it's only used during training/precompute), but offers a worse asymptotic scaling than directly growing $\mathcal{A}_{\text{main}}$ — meaning: if you don't care about inference cost, growing main is better; if you do, this is the trick.

### Setting 2 — OLMo / large-scale (Section 4.2)

- Build on OLMo codebase; main models OLMo-1B, 1.3B, 1.9B.
- **f-gram model = 1.8B** params (OLMo-1.9B-arch minus token embedding); also a smaller 0.6B variant.
- Two f-gram set sizes: **10M** (cutoff freq 21,956) and **1B** (cutoff freq 70).
- SCONE models trained for **500B tokens**; baselines for 1T (half tokens so total train FLOPS comparable).
- **Headline result (Table 1, zero-shot avg across PIQA, HellaSwag, ARC-E/C, CSQA, MMLU)**:
	- OLMo-1B baseline: 53.7 avg.
	- OLMo-1.9B baseline: 56.8.
	- OLMo-1B + **10M f-grams**: **54.9** (matches "+3.1 budget" but with 1B accelerator params).
	- OLMo-1B + **1B f-grams**: **57.0** — *beats 1.9B baseline*.
- **Inference savings (Table 5, vs 1.9B baseline at similar quality)**:
	- SCONE-1.3B + 10M f-grams: **5.60 GB peak GPU** (vs 8.38 GB), **4.83 ms/tok** (vs 6.45 ms), +41.76 GB CPU RAM.
	- SCONE-1B + 1B f-grams: **4.45 GB peak GPU**, **4.90 ms/tok**, +7.67 TB disk.
- **Perplexity (Table 3)**: SCONE improves on 11 corpora (c4-en, books, common-crawl, pes2o, reddit, stack, wiki, ice, m2de-s2orc, pile, wikitext-103). SCONE 1B+1B f-grams gives avg 14.581 PPL, beating 1.9B baseline (14.598).
- Training curves (Figure 13): **SCONE models converge slower** — sign of higher capacity that hasn't saturated.

### Setting 3 — Post-training (Appendix E.3)

- SCONE applied to **SFT of Qwen3-4B-base** on open-r1 Mixture-of-Thoughts dataset.
- f-gram model = Qwen3-8B-base or Qwen3-14B-base. 10M f-grams.
- **Results (Table 4)**:
	- Qwen3-4B base: AIME 2024 45.3, LiveCodeBench 30.8, decoding 10.05 ms/tok.
	- SCONE-4B (8B f-gram): **48.3 / 34.5**, 10.13 ms (negligible latency hit).
	- SCONE-4B (14B f-gram): **51.6 / 36.3**, 10.13 ms.
- Demonstrates SCONE useful beyond pretraining.

---

## Comparison to Concurrent / Related Work

- **Over-Tokenized Transformer (OE-12.8M, Huang et al. 2025)**: also decouples input/decoding embeddings via extra n-gram input embedding. **Differences**:
	- They **hash n-grams into a fixed bucket count** (manage vast n-gram space via collision-prone hashing); SCONE selects frequent n-grams explicitly.
	- Their extra embedding table is **fully instantiated on accelerator** during training, hard to scale. SCONE's table is parameterized by a NN -> no full instantiation.
	- Table 1: OLMo-1B + OE-12.8M -> 54.4 avg; SCONE 10M f-grams -> 54.9 (slightly better, attributed to capacity of $\mathcal{A}_{\text{f-gram}}$).
- **Mixture of Lookup Experts (Jie et al. 2025)**: discretizes inputs to an MoE layer so inference is a lookup. Similar spirit but **fixed key count = token vocab**, no scaling axis on key count.
- **MoE (Switch, Mixtral)**: fixed inference FLOPS but **all experts on accelerator** -> high accelerator memory. SCONE both fixed-FLOPS *and* fixed-accelerator-memory.
- **Memory layers (Sukhbaatar, Lample, Berges 2024)**: large continuous embedding store queried by ANN. Embeddings still resident on accelerator and trained directly (sparse updates again).
- **N-grammer (Roy 2022)**: latent n-gram bigram embeddings — early decoupling idea SCONE generalizes.
- **Tokenization-free LMs (ByT5, MambaByte, Megabyte, BLT)**: orthogonal — could combine with SCONE's offloaded embedding idea.
- **Implicit n-gram patterns in transformers (Geva 2021/22, Voita 2024, Chen 2024)**: prior work shows attention heads / MLPs implicitly encode n-gram features; SCONE makes them explicit via input.

---

## Why This Works (intuition)

- **f-grams act as a giant external KV memory of contextualized phrase embeddings**, indexed by surface n-grams.
- During training, the f-gram transformer *generates* these embeddings from base token embeddings -> dense gradient signal flows into a *shared* network instead of getting splayed across billions of independent vectors.
- At inference, we cash in: replace the network with a precomputed table that lives in cheap memory.
- Conceptually analogous to **distillation into a lookup table** but where the table is *jointly trained* with the consumer model rather than retrofitted.
- Also resembles a **retrieval-augmented input layer**: each input token gets a richer embedding conditioned on its left-context phrase, retrieved by exact n-gram match.

---

## Limitations / Open Questions

- **Short n-grams only**: extension to longer queries (e.g., full prefixes, semantic chunks) requires designing keys with good hit rate. Raw text keys have low cache hit because semantically-similar queries differ on surface. Continuous semantic keys need discretization.
- **Scale ceiling**: max evaluated at **3B params at training time** (1B main + 1.8B f-gram). No 70B+ validation. Authors note this is hardware-limited.
- **Storage cost**: 1B f-grams = 7.6 TB SSD per deployment. Significant for edge / on-device.
- **Tokenizer-dependent**: f-grams defined over BPE tokens; behavior under tokenizer-free settings untested but conceptually applicable.
- **No statistical significance testing** at large scale (compute-prohibitive); claims rely on consistency across configs.

---

## Quick Algorithm Summary (pseudocode flavor)

```
# Build f-gram set (offline, K-1 passes over corpus)
for n in 2..K:
    count all n-grams with freq >= 5, prune using (n-1)-gram threshold
V_fgram = top-S by frequency

# Forward pass (per token i)
j = longest start such that tokens[j..i] in V_fgram, else i
if j == i:
    e_i = T(token_i)
else:
    if training:
        e_i = A_fgram(T(token_j), ..., T(token_i))   # last-pos output
    else:  # inference
        e_i = F[(token_j, ..., token_i)]             # CPU/SSD lookup
# Main model
logits = D(A_main(e_1, ..., e_m))
```

---

## Key Numbers to Remember

- **K = 5** (default max f-gram length).
- **Default frequency threshold = 5**.
- **1B accelerator-params + 1B f-grams > 1.9B dense baseline**, ~half inference FLOPS+memory.
- **41 GB CPU RAM** for 10M f-grams ($d=2048$, fp16); **7.6 TB SSD** for 1B.
- **Lookup latency** 0.5-2.3 ms/token, batch-dependent; LLM decode budget ~10 ms/token.
- **Tail-token sparsity**: vocab 2M -> only 7.3% of tokens >100 updates per 100M tokens.

---

## Code

- **No public code release by authors** as of paper v3 (Oct 2025). The NeurIPS checklist states "We will release our code after the reviewing process" but as of writing no official repo exists.
- Only an **unofficial third-party implementation** at `github.com/llmsresearch/scone`.
