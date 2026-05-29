# KVTC: KV Cache Transform Coding for Compact Storage in LLM Inference

> Staniszewski & Łańcucki, 2025 (NVIDIA / Univ. of Warsaw). ICLR 2026. arXiv:2511.01815. No public code.

**TL;DR:** Treat the KV cache as a *file to store/transmit*, not just a runtime structure. Borrow the **classical media-codec transform-coding pipeline** (JPEG-style): **PCA decorrelation → adaptive scalar quantization → DEFLATE entropy coding**. Training-free, calibrated once per model. Achieves ~**20×** compression with <1pt accuracy loss, **40×+** for some cases. Beats eviction (H2O, TOVA), quantization (KIVI, GEAR, FP8), and SVD (xKV) baselines on Llama 3, Mistral NeMo, R1-Qwen2.5.

---

## 1. Motivation: KV cache as a STORAGE problem

- **Core reframing:** prior KV-cache work optimizes *runtime* autoregressive decoding (lower memory traffic during next-token prediction). KVTC instead targets **storage + transfer** of caches *between* inference phases.
  - **Decoding operates on the decompressed cache.** Compression is applied after prefill / after decoding / when offloading. Attention math unchanged.
- **Why caches are worth storing:** chat / iterative code-editing reuse caches across conversation turns via **shared-prefix / prefix caching** (PagedAttention, prefix sharing). Inference frameworks treat local KV caches as **databases**.
  - When a new user prompt $x_i$ arrives, reuse existing cache, only forward new tokens → cuts **TTFT** (time-to-first-token). If cache deleted → quadratic $O(n^2)$ recompute of attention over whole conversation.
- **The tension** (production dilemma):
  - **Keep stale caches on-GPU** → consume scarce HBM needed for serving other users.
  - **Discard** → pay recomputation cost.
  - **Offload** to CPU DRAM / NVMe / network storage → transfer overhead.
- **KV cache is big.** 16-bit cache for 1K tokens = $4\, l\, h\, d_{\text{head}}\, t$ bytes. Examples (Table 1, per 1K tokens):
  | Model | Size/1K tok |
  |---|---|
  | Qwen 2.5 R1 1.5B | 28 MiB |
  | Llama 3.1 8B | 128 MiB |
  | Llama 3.3 70B Instruct | 320 MiB |
  | Mistral NeMo 12B / MN-Minitron 8B | 160 MiB |
- **Tiered serving architecture** (Fig 5): disaggregated **prefill node** ↔ **decode node** over RDMA (InfiniBand/RoCE). Both keep tiered caches: GPU HBM (hot), CPU DRAM (warm), NVMe/SSD (cold). A **cache-aware router** picks nodes by matching cached prefixes. KV transfers dominate cross-node traffic.
- **Two payoffs of compressing after prefill/decode:**
  1. **Extend cache DB capacity / lifetime** ∝ compression ratio → higher hit rates in hot/warm tiers → avoids prefill. E.g. 1000-line coding file w/ Llama 3.3 70B ≈ 1.6 GiB of 8-bit cache; 20× lifetime extension can decide if it stays hot.
  2. **Reduce network traffic** ∝ compression ratio → critical when bandwidth-bound. (Prefill itself is $O(n^2)$ and usually dominates transfer time.)
- **vs eviction:** eviction *throws away* tokens (irreversible, brittle, can't reconstruct). KVTC is **lossy-but-reversible compression** — keeps all tokens, just stores them in fewer bits. Complementary: can stack eviction on top.

---

## 2. KV cache redundancy: why this works

Three empirical observations justify the pipeline (all favor low-rank + quantization):

- **Cross-head subspace alignment (Fig 3).** Keys from different attention heads (and values, separately) lie in a **shared latent space up to an orthogonal transform**.
  - Solve **orthogonal Procrustes** per head pair: $R^\star = \arg\min_R \lVert K_i - K_j R\rVert_F$ s.t. $R^\top R = I$.
  - Before alignment: inter-head cosine sim < 0.2. After: rises substantially for **keys**, moderately for **values**.
  - Dissimilarity before alignment likely just stems from random init of K/V projection matrices.
  - **Key math fact motivating PCA:** for data $A \in \mathbb{R}^{n\times d}$, if $k$ directions explain all variance of $A$, then $k$ directions explain all variance of $B = [A, AR] \in \mathbb{R}^{n\times(d+d')}$ for orthogonal $R$. → concatenating aligned heads keeps rank low → PCA is the right dim-reduction tool.
  - Theory link: w/o RoPE, attention dot-products are equal up to orthogonal transform by **uniqueness of Gram realizations** (Horn & Johnson).
- **Channel sparsity / low variance (Fig 7, 8; App B.2).** Per-channel mean-abs activations and variances are concentrated in few channels for both K and V across Llama/Mistral/Qwen → compressible + quantizable.
- **Positional embeddings distort low-rank structure** → must undo RoPE before compression.

---

## 3. Method: Key-Value Transform Coder (kvtc)

Built on the **transform-coding framework** (Ahmed 1974 DCT; Goyal 2001) underlying JPEG/H.264/H.265. Three operating modes:

| Mode | When | Where | What |
|---|---|---|---|
| **Calibration** | once per (model, *and* per CR for DP) | offline, H100 | fit PCA basis $V$, mean $\mu$, DP bit allocation |
| **Compression** | between inference phases | GPU (post-prefill/decode, affects TTFT) or CPU (if in storage) | project + quantize + DEFLATE |
| **Decompression** | before generation | layer-by-layer (early-start) | reverse the steps |

> Pipeline (Fig 1): `KV cache → Linear projection (PCA, learned on calib) → Adaptive quantization (DP, dynamic) → Entropy coding (DEFLATE via nvCOMP) → Storage`

### 3.1 Feature decorrelation (PCA / SVD)

- **One global basis** computed on a calibration set $\mathcal{C}$, **reused for all requests** — unlike per-prompt SVD methods (xKV, SVDq) which recompute SVD every prompt (expensive).
- **Procedure:** forward all calib sequences, collect KV caches. Per sequence concat cache entries along time → pool of positions. Sample $n$ positions (**excluding attention sinks**). For each sampled position take K (resp. V) from all $l$ layers × $h$ heads, **undo RoPE**, concat along hidden dim $d_{\text{head}}$ → data matrix $C \in \mathbb{R}^{n\times p}$, $p = l\,h\,d_{\text{head}}$.
- **SVD of centered matrix:** with $\mu \in \mathbb{R}^p$ the per-feature mean,
  $$C - \mu = U\Sigma V^\top$$
  singular values descending = PCA of $C$. **Keys and values done separately.**
- **Forward / inverse transform** for any $X$:
  $$D = (X - \mu)V, \qquad X = DV^\top + \mu$$
  Truncate to rank $r < p$: $V \in \mathbb{R}^{p\times r}$, $X \approx DV^\top + \mu$.
- **Scalability:** use **randomized SVD** (Halko 2011) on GPU, target rank $r < p$, 8 iterations. Cuts runtime/memory.
- **Stored per model:** $V$ (key + value separately) and $\mu$. Small relative to model — only **2.4% of params for Llama 3.3 70B** (Table 21).

### 3.2 Adaptive quantization (the key trick)

- **Exploit PCA ordering by variance:** allocate **more bits to high-variance (early) components**, fewer to trailing ones — directly analogous to JPEG allocating bits to low-frequency DCT coefficients.
- Quantize $D$ coordinate-wise: $D^{q_1,\dots,q_d}$, $q_i \in \mathbb{Z}_{\geq 0}$ = bit width of $i$-th principal component.
- **Why optimize in the decorrelated domain (clean result):** right-mult by orthonormal $V^\top$ preserves Frobenius norm, so
  $$\lVert DV^\top - D^{q_1\dots q_k}V^\top \rVert_F^2 = \lVert (D - D^{q_1\dots q_k})V^\top \rVert_F^2 = \lVert D - D^{q_1\dots q_k} \rVert_F^2$$
  → **optimal bit allocation can be found directly on the PCA coefficients**, no need to map back.
- **Dynamic Programming (DP) bit allocation** (App B.17 pseudocode + optimality proof):
  - Minimize reconstruction error $\lVert DV^\top - D^{q_1\dots q_d}V^\top\rVert_F^2$ under a **global bit budget**.
  - Two DP tables: (1) min achievable error using first $i$ PCs with budget $b$ bits; (2) backpointer for optimal local decision.
  - **Microscaling-inspired grouping** (Rouhani 2023): quantize *groups* of subsequent PCs together, each group sharing a **16-bit shift + scaling factor**. Group sizes restricted to $\{1, 16, 64, 256, 1024\}$. Quant types `[None, int2, int4, fp8]` (None = drop component, 0 bits).
  - Total budget = payload bits across coords + per-group shift/scale bits.
  - **Complexity:** $O(\texttt{num\_features} \times \texttt{max\_bit\_budget} \times \texttt{batch})$.
  - **Output (Fig 6 right):** learned bit widths decrease monotonically; **DP assigns 0 bits to many trailing PCs** → motivates trimming $V$ early (faster compress/decompress) and trimming PCA rank during calibration (cheaper calib).
- **Calibration speed:** PCA on 160K tokens ~1.5 min; full DP per (model, CR) < 8 min; whole calib for a 12B model < 10 min on one H100.

### 3.3 Entropy coding

- Pack quantized values into byte array → **DEFLATE** (Huffman + LZ77) via **nvCOMP** (GPU-parallel, lossless).
- Data-dependent: adds **~1.23× on top of quantization on average** for Llama 3.1 8B (Fig 2).
- DEFLATE substitutable by faster **GDeflate** (GPU-optimized) with ≤0.1 CR difference (App B.8). RULER-VT caches are unusually compressible (repeated noise filler).

---

## 4. Compression ratio breakdown (Fig 2, Llama 3.1 8B)

Stacked contributions per stage:
| kvtc setting | PCA | + Quantization | + DEFLATE (total) |
|---|---|---|---|
| 8× | ~ | → | 9–10× |
| 16× | | | 18–22× |
| 32× | | | 34–44× |
| 64× | | | 64–88× |

CR varies because DEFLATE is data-dependent. **The `kvtc_CR×` notation = target CR for the DP, *before* DEFLATE** (e.g. `kvtc_16×` → ~20× realized). CR computed only on compressed tokens, **not** counting sliding-window tokens.

---

## 5. Sliding window & sink tokens (don't compress everything)

- **Skip compression for:**
  - **$w$ most-recent tokens** (sliding window) — disproportionate attention mass.
  - **$s$ oldest tokens** (attention sinks) — Xiao et al. streaming-LLM observation; PCA dim-reduction yields **higher reconstruction error on initial tokens** (Fig 6 middle).
- **Defaults for main experiments:** $w = 128$, $s = 4$.
- **Analogy to media codecs:** bits allocated to coefficients per perceptual distortion ≈ attention mass as a proxy for token importance.
- **Ablations (Fig 4, Table 6):** disabling sink ($s=0$) or sliding-window ($w=0$) compression at 64× **significantly lowers or entirely collapses** accuracy.
  - Table 6: Llama at 64× w/o sink skip (`kvtc▶0`) → GSM8K **1.6**, MMLU 9.9, LITM 0.0 (collapse). With sink skip (`kvtc▶4`) → GSM8K 57.2, MMLU 60.7.
- **CR-with-window formulas (App B.15):** for window $w$, sinks $s$: $\text{CR}_{\text{eff}} = \frac{\text{ctx}}{(\text{ctx}-w-s)/\text{cr} + w + s/2}$.

---

## 6. Experiments

**Models:** Llama 3 (8B, 3.3 70B Instruct), Mistral NeMo 12B, MN-Minitron 8B (pruned from NeMo, keeps cache size), R1-distilled Qwen2.5 (1.5B, 7B). All **GQA**. 1.5B–70B range.

**Tasks:** GSM8K (8-shot CoT), MMLU (4-shot CoT), Qasper (2-shot F1), Lost-in-the-Middle / LITM (0-shot KV retrieval), RULER-VT (variable tracking, 1-shot), NIAH, MATH-500, AIME 24/25, LiveCodeBench, + LongBench (2WikiMHQA, MFQA, MuSiQue, QMSum, SAMSum) and RULER (CWE/FWE, HotPotQA, SQuAD).

**Baselines:** quantization — **KIVI** (2-bit asym), **GEAR** (low-rank error correction), **FP8** (E4M3); eviction — **TOVA**, **H2O**; SVD — **xKV** (cross-layer SVD, prefill-only); reasoning — **DMS** (Łańcucki 2025, trained token eviction).
- All except xKV simulate multi-turn conversations: compress/evict every $c$ tokens. KVTC ($w=128$) compresses every $c=16$. xKV gets prefill-only advantage (recomputing SVD per decoded token too slow); only prefill CRs reported for non-Qwen.

**Calibration data:** 1:1 mix of short (1–8K) and long (8–32K) docs from **FineWeb** (generic) + **OpenR1Math** (reasoning traces). RoPE removed. 160K tokens for 8–70B; 200K + 8K dim for Qwen (fewer KV heads).

### 6.1 General base models (Table 2, 8–12B)

`kvtc` **maintains high accuracy across all tasks even at 32×/64×**. Key headline numbers (Vanilla → kvtc):

| Llama 3.1 8B | Vanilla | kvtc_8× | kvtc_16× | kvtc_32× | kvtc_64× |
|---|---|---|---|---|---|
| CR | 1 | 9–10 | 18–22 | 34–44 | 60–88 |
| GSM8K | 56.8 | 57.0 | 56.9 | 57.8 | 57.2 |
| MMLU | 60.5 | 59.8 | 60.1 | 60.6 | 60.7 |
| QASPER | 40.4 | 40.1 | 40.7 | 39.8 | 37.8 |
| LITM | 99.8 | 99.3 | 99.3 | 99.1 | 90.2 |
| RULER-VT | 99.8 | 99.1 | 99.1 | 98.9 | 95.9 |

- **`kvtc_16×` (~20× realized) consistently within <1 score point of vanilla** across all models/tasks (std errors in Table 19).
- **Sometimes beats vanilla** at high CR (compression as mild regularizer; also noted by DMS paper).
- **vs baselines:** GEAR & KIVI show degradation on GSM8K + LITM even at 5× CR. H2O/TOVA (generic eviction) **perform poorly** as compressors. xKV does well on most tasks **except Qasper**. Qasper is hardest for `kvtc` at high CR.

### 6.2 Reasoning models (Table 3, R1-Qwen2.5, temp 0.6)

| Qwen 2.5 R1 7B | CR | AIME24 | AIME25 | LCB Coding |
|---|---|---|---|---|
| Vanilla | 1 | 50.9±4.9 | 40.8±4.3 | 36.7 |
| kvtc_8× | 9–11 | 52.5±3.6 | 36.5 | 36.5 |
| kvtc_16× | 18–21 | 50.9±6.8 | 38.3±5.5 | 31.6 |
| DMS_8× | – | 50.0 | N/A | 33.4 |

- High AIME variability → 8 independent runs, score±std. Coding drops only 0.3pp (1.5B) / 0.2pp at `kvtc_8×`.
- 1.5B cache already tiny (29 KiB/tok vs 131 for Llama 3.1 8B); 9× → 3.2 KiB/tok.
- **vs DMS** (SOTA autoregressive token eviction): competitive; since DMS evicts tokens, **could be stacked with kvtc** for even lower footprint.

### 6.3 Multi-GPU (Table 4, Llama 3.3 70B, 4-GPU pipeline-parallel)

Each GPU compresses its **local cache (20 layers) separately**. MATH-500: vanilla 75.6 → **74.4 @ 10× (−1.2pp), 72.6 @ 20× (−3.0pp)**. NIAH & LITM stay 100.0 throughout. (Joint compression across chunks could improve accuracy.)

### 6.4 Latency (Table 5, Mistral NeMo 12B, H100, simple HF impl)

- **TTFT recompute vs kvtc decompress** (8K ctx, BS=8): vanilla recompute **3098 ms** → kvtc decompress **380 ms** ≈ **8× TTFT reduction**.
- Per-module (BS=8, CTX=8K): Project 153/156 ms (comp/decomp), Quantize 67/37, Deflate 66/64. Total 379 comp / 267 decomp.
- Decompression inverse-projection $V^\top$ done **layer-by-layer** (submatrices of $V^\top$) → generation starts early.

---

## 7. Ablations (Appendix B)

- **B.3 Sink skipping:** at high CR, skipping first 4 tokens is essential (collapse otherwise; see §5).
- **B.4 Separate K/V CRs (Table 7):** **values compress more than keys** for long-context retrieval (need precise attention to selected tokens → high key accuracy). Stronger value compression hurts GSM8K/MMLU.
- **B.5 Calibration size (Table 8, Fig 9):** more calib data → lower reconstruction error + better downstream, **big benefit for high CR (256×)**, diminishing for 32×/64×. 40K tokens already competitive for 64×. Trades calib cost for accuracy.
  - **Reconstruction-error ↔ perplexity ↔ accuracy strongly correlated:** Pearson Rec-Err-Key vs PPL **0.959**, PPL vs AVG-Score **−0.992**. → Frobenius reconstruction error is a usable (but task-dependent) proxy.
- **B.6 Calib domain (Table 9, 12):** OpenR1Math calib better preserves MMLU + retrieval than FineWeb (question-think-answer structure aligns w/ eval). 50/50 mix best at 256×. **Robust to domain shift** — even calibrating on StarCoder Python/C/Assembly retains in-context retrieval (LITM/RULER-VT) and at 16× matches KIVI/GEAR (except calibrating on C).
- **B.7 Sliding window size (Table 13):** bigger $w$ → better downstream; biggest jump between $w\leq16$ and $w\geq64$. ($w=128$ default.)
- **B.8 Lossless algo (Table 14):** DEFLATE/GDeflate/ANS/zStandard all improve over Identity; GDeflate ≈ DEFLATE but GPU-faster. zStandard highest on RULER-VT (46×).
- **B.9 DP quantization vs pure PCA (Table 15):** **DP quant is crucial.** `-DPQ` variant (just drop low PCs, no quant) collapses long-context at high CR — e.g. Llama 64×: LITM 80.0 → **5.1**, RULER-VT 88.0 → 55.5. Quantization is what enables scaling to high CR.
- **B.10 Cross-layer concat for PCA (Tables 16, 17):** **more concatenated layers → better** (confirms §2 cross-layer similarity hypothesis). 1 layer collapses at high CR; 32 layers recovers. **Keys benefit more from global concat; values get a big boost going 8→16 layers.**
- **B.11 Per-prompt vs one-time PCA (Table 18):** **one-time global PCA wins.** Per-prompt $V^\top$ has CR overhead (storing the matrix → CR drops to ~1×) and **generalizes poorly to the continuation** of the conversation.
- **B.14 PCA matrix sizes (Table 21):** $V$ stored 16-bit, cut to 10K PCs. Total PCA params: Llama 8B **8.7%**, NeMo 12B 7.1%, **Llama 3.3 70B only 2.4%**, Qwen 1.5B 6.8%. DP further trims by zeroing trailing PCs.
- **B.16 LMCache + vLLM (Table 22):** end-to-end multi-user, Llama 3.3 70B FP8, 2× H100, 128 GiB host mem/GPU. At **≥12 clients** vanilla host memory overflows → recompute → latency spikes (12 clients: vanilla **155.7s** vs kvtc_16× **27.9s**; TTFT 136.6s vs **5.6s**). kvtc keeps caches resident longer. (Only host-memory compression tested; HBM compression would add gains.)

---

## 8. Related work positioning

- **vs tuning-free quant (KIVI, KVQuant):** they quantize per-channel(K)/per-token(V) in raw space; KVTC quantizes in the **SVD-transformed space** with **DP-optimized precision**. Adopts KIVI's uniform scheme but applies pre-RoPE.
- **vs SVD (GEAR, LoRC, Eigen Attention, ShadowKV, SVDq, xKV):** KVTC differs in (i) models **rotational** relations between **non-adjacent** layers via cross-layer concat before decomposition; (ii) selects ranks + bitwidths via DP under a budget; (iii) adds **entropy coding** on quantized factors.
- **vs sparse/eviction (H2O, TOVA, Quest, Landmark, NSA, DMC, DMS):** orthogonal — KVTC compresses, doesn't discard; composable with all of these.
- **Transform coding lineage:** DCT/JPEG/H.264/H.265; CAT (compression-aware training), Young (CNN transform quant), LLM.265 (repurposing H.264/265 hardware for weights). KVTC = **novel combination** of decorrelation + DP quant + entropy coding exploiting cross-layer deps.

---

## 9. Limitations & future work

- **Online compression unexplored:** single global $V$ could enable operating *directly in PC space* (skip decompress) — left for future.
- **Composability:** directly compatible w/ token eviction (TOVA etc.); could compress **MLA latent state** (DeepSeek MHLA).
- **Scalability evidence:** dense decoder-only 1.5–70B only; larger / more realistic distributions untested. Multi-turn settings are *simulated*, may not reflect real interaction patterns.
- **Calibration cost:** ~200K tokens on 1 H100; scaling beyond 200K (consistently helps) deferred.
- **Reconstruction error proxy** is convenient but **not guaranteed predictive of task accuracy** (task-dependent).
- **Speed headroom:** compress/decompress could be much faster via **kernel fusion + hierarchical PCA** (per-layer then across groups). Even the naive HF impl already gives substantial benefit.

---

## 10. Key takeaways

1. **Reframe KV cache as a storage/transmission artifact** → import 30 years of media-codec engineering wholesale.
2. **PCA decorrelation + variance-ordered DP bit allocation + entropy coding** = each stage compounds; quantization (not just rank truncation) is the load-bearing piece for high CR.
3. **Cross-head/cross-layer alignment up to orthogonal transform** is the structural prior that makes global PCA work; concatenating layers is essential.
4. **Training-free, one calibration per model, ~2.4–8.7% param overhead, 8× TTFT reduction, ~20× lossless-ish / 40×+ aggressive.**
5. **Beats eviction + quant + per-prompt-SVD baselines simultaneously on CR and accuracy**, and is composable with them.

---

## Related Notes

- **[TriAttention](../TriAttention/TriAttention.md)** — Different compression axis. TriAttention **evicts** keys (keep top-scoring tokens, drop the rest); KVTC keeps **all** tokens but **lossily encodes** them (PCA + quant + entropy). Orthogonal and **composable** — evict first, then transform-code the survivors.
- **[Physics of KV Compression](../Physics_of_KV_Compression/Physics_of_KV_Compression.md)** — Explains *why* KVTC works: that paper shows moderate compression degrades internal reps with little accuracy loss (redundancy margins), which is exactly the slack KVTC's lossy codec exploits. Note KVTC's regime (decorrelate/quantize all tokens) sidesteps the eviction "hallucination cliff" the Physics paper attributes to Global-Eviction-Ratio spikes.
- **[Expected Attention](../Expected_Attention/Expected_Attention.md)** / **[Compactor](../Compactor/Compactor.md)** / **[KVzap](../KVzap/KVzap.md)** — The **eviction/scoring** family KVTC benchmarks against and is composable with. KVTC argues storage-codec compression is a distinct, stackable lever vs token dropping; it also beats quantization (KIVI/GEAR/FP8) and per-prompt SVD baselines on the CR-vs-accuracy frontier.
