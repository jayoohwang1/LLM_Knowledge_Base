# Engram: Conditional Memory via Scalable Lookup

> **Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models** — DeepSeek-AI + PKU, Jan 2026 (arXiv 2601.07372).

---

## TL;DR

- **Big idea**: add a new sparsity axis to LLMs — **conditional memory** (static $O(1)$ N-gram lookup) — orthogonal to MoE's **conditional computation**.
- **Engram module**: modernizes classic N-gram embeddings — hashed deterministic lookup into massive embedding tables, fused into transformer hidden states via context-aware gating + a depthwise conv.
- **Sparsity Allocation U-curve**: under fixed param + FLOP budget, splitting ~75–80% of inactive params to MoE experts and ~20–25% to Engram beats pure MoE.
- **Scale**: Engram-27B (iso-FLOPs, iso-params with MoE-27B baseline) beats MoE on **everything**: MMLU +3.4, BBH +5.0, ARC-C +3.7, HumanEval +3.0, MATH +2.4, Multi-Query NIAH 84.2 → 97.0.
- **Mechanism**: Engram offloads early-layer static-pattern reconstruction → frees backbone depth for reasoning AND frees attention for global context (long context).
- **Systems win**: deterministic addressing → prefetch from host DRAM → 100B-param table offload costs <3% throughput.

---

## 1. Motivation

- **Sparsity recap**: dominant paradigm in modern LLMs is MoE = *conditional computation*: route tokens to subset of expert FFNs. Scales params w/o FLOPs.
- **Linguistic duality** observation:
    - **Compositional reasoning** → needs deep dynamic computation.
    - **Knowledge retrieval / local idiomatic patterns** → static, stereotyped (e.g. multi-token entities like "Diana, Princess of Wales", phrases like "By the way").
    - Transformers have **no native retrieval primitive** → forced to *simulate retrieval through computation* (early FFN layers act as key-value memories per Geva et al. 2021).
    - This wastes sequential depth on trivial lookups.
- **Proposal**: introduce **conditional memory** as a complementary sparsity axis.
    - **Conditional computation (MoE)**: sparsely activate *parameters* to process dynamic logic.
    - **Conditional memory (Engram)**: sparsely *retrieve* static embeddings via constant-time lookups.
- **Why N-grams revisited?** Classic N-gram models (Brants 2007, FastText) showed local context regularities are computationally trivial to capture via lookup. Engram = N-gram embeddings + modern tricks.

---

## 2. Architecture

### 2.1 Pipeline overview

- For each input position $t$ with hidden state $\mathbf{h}_t^{(\ell)}$:
    1. **Retrieve**: extract suffix N-grams ending at $t$, hash to deterministic table indices, lookup embeddings.
    2. **Fuse**: gate retrieved embeddings using current hidden state as query → add short causal conv → residual connect.
- Applied only at **specific layers** (not every layer) to decouple memory from compute.
- Standard input/output embeddings stay intact.

### 2.2 Sparse retrieval via hashed N-grams

**Tokenizer Compression**
- Subword tokenizers (BPE/SentencePiece) assign disjoint IDs to semantically equivalent tokens (`Apple` vs `␣apple` vs `APPLE`).
- Pre-compute a **surjection** $\mathcal{P}: V \to V'$ via NFKC normalization + lowercase + accent strip → canonical IDs.
- For DeepSeek-V3's 128k tokenizer this gives **23% effective vocab reduction**.
- Compressed token at position $t$: $x'_t = \mathcal{P}(x_t)$.
- Suffix N-gram: $g_{t,n} = (x'_{t-n+1}, \ldots, x'_t)$.

**Multi-Head Hashing**
- Combinatorial space of N-grams is intractable to parameterize directly → use hashing (Tito Svenstrup 2017 hash embeddings).
- **K distinct hash heads** per N-gram order $n$, each mapping context to index in embedding table $\mathbf{E}_{n,k}$ of **prime size** $M_{n,k}$.
    - Prime sizes mitigate collisions / aliasing.
- Lookup:
    $$z_{t,n,k} \triangleq \varphi_{n,k}(g_{t,n}), \quad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]$$
- $\varphi_{n,k}$ = lightweight **multiplicative-XOR hash** (one $(\text{mul}, \text{xor}, \text{mod prime})$ per token in context).
- Final memory vector concatenates across all $N$ orders + $K$ heads:
    $$\mathbf{e}_t \triangleq \big\|_{n=2}^{N} \big\|_{k=1}^{K} \mathbf{e}_{t,n,k} \in \mathbb{R}^{d_{\text{mem}}}$$

### 2.3 Context-aware gating

- Retrieved $\mathbf{e}_t$ is a **static prior** — can be polysemous noise or hash collision.
- Modulate w/ attention-like gate using current hidden state $\mathbf{h}_t$ as Query, memory as KV source:
    $$\mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t$$
- Scalar gate w/ RMSNorm for stability:
    $$\alpha_t = \sigma\!\left(\frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}}\right) \in (0,1)$$
- **Semantic alignment**: if memory contradicts context, $\alpha_t \to 0$, suppressing noise.
- Gated values: $\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$.

**Short depthwise causal conv** (Mamba/RWKV-style)
- Kernel size $w=4$, **dilation $\delta$ = max N-gram order** (3 in 27B), SiLU.
- $\mathbf{Y} = \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}}))) + \tilde{\mathbf{V}}$
- Expands receptive field + adds nonlinearity.
- Residual into backbone: $\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \mathbf{Y}$, then standard Attention + MoE on top.

### 2.4 Integration with multi-branch backbone

- Default backbone uses **Manifold-Constrained Hyper-Connections (mHC)** with $M=4$ branches (Xie et al. 2025) — residual stream is expanded into 4 parallel branches.
- **Parameter sharing strategy** across branches:
    - **One shared** sparse embedding table $\mathbf{E}$ + shared value proj $\mathbf{W}_V$.
    - **$M$ distinct** key projs $\{\mathbf{W}_K^{(m)}\}_{m=1}^M$ → branch-specific gates:
        $$\alpha_t^{(m)} = \sigma\!\left(\frac{\text{RMSNorm}(\mathbf{h}_t^{(m)})^\top \text{RMSNorm}(\mathbf{W}_K^{(m)} \mathbf{e}_t)}{\sqrt{d}}\right)$$
    - Per-branch retrieved val: $\mathbf{u}_t^{(m)} = \alpha_t^{(m)} \cdot (\mathbf{W}_V \mathbf{e}_t)$.
- $\mathbf{W}_V$ + $M$ $\mathbf{W}_K^{(m)}$ fused into a **single dense FP8 matmul** → max GPU compute util.

### 2.5 System efficiency: decoupling compute and memory

- Engram's retrieval indices **depend solely on input token IDs**, not on hidden states → fully **deterministic, predictable** unlike MoE's dynamic routing.

**Training**
- Embedding tables sharded across GPUs via standard model parallel.
- **All-to-All** primitive gathers active rows in forward, dispatches grads in backward.
- Total memory capacity scales linearly w/ accelerators.

**Inference**
- Tables offloaded to **host DRAM**.
- **Prefetch-and-overlap**: since indices known before forward, asynchronously pull embeddings via PCIe while preceding layers compute.
- Engram placed at *specific* layers (not earliest) — leverages computation buffer to hide PCIe latency.
- **Multi-Level Cache Hierarchy**: N-grams follow **Zipfian distribution** → hot embeddings live in HBM/DRAM, cold tail on NVMe SSD.
- **Empirical**: offloading 100B-param table → <3% throughput hit (4B Dense: 9032 → 8858 tok/s; 8B Dense: 6316 → 6140 tok/s).

---

## 3. Scaling Laws and Sparsity Allocation

### 3.1 Allocation under finite constraints

- Define:
    - $P_{\text{tot}}$ = total trainable params (excl. vocab embedding + LM head).
    - $P_{\text{act}}$ = activated per token (= FLOPs proxy).
    - $P_{\text{sparse}} \triangleq P_{\text{tot}} - P_{\text{act}}$ = inactive "free" budget.
- **Allocation ratio** $\rho \in [0,1]$:
    $$P_{\text{MoE}}^{(\text{sparse})} = \rho \, P_{\text{sparse}}, \quad P_{\text{Engram}} = (1-\rho) P_{\text{sparse}}$$
- $\rho=1$ → pure MoE baseline. $\rho<1$ → cut routed experts, add Engram slots.

**Experimental protocol**
- Two compute regimes: $C = 2\times10^{20}$ FLOPs (5.7B/568M act) and $C=6\times10^{20}$ FLOPs (9.9B/993M act).
- Fixed sparsity ratio $P_{\text{tot}}/P_{\text{act}} \approx 10$.
- Identical training pipeline / hyperparams.

**Results — U-shaped curve** (Fig. 3 left)
- Both regimes show clear **U-shape** in val loss vs $\rho$.
- **Optimum ~$\rho \approx 80\%$** (20–25% to Engram).
    - At $C=6\times10^{20}$: val loss 1.7248 ($\rho=100\%$) → 1.7109 (optimum), $\Delta = -0.0139$.
- Engram model matches pure MoE perf even at $\rho \approx 40\%$ — substantial robustness.
- **Interpretation**:
    - $\rho \to 100\%$: no dedicated memory, backbone reconstructs static patterns inefficiently.
    - $\rho \to 0\%$: loses dynamic reasoning capacity; memory ≠ computation.

### 3.2 Infinite memory regime (Engram-only scaling)

- Fix MoE backbone ($P_{\text{tot}}\approx 3$B, $P_{\text{act}} \approx 568$M, 100B tokens).
- Sweep Engram table size $M$ from $2.58\times10^5$ to $1.0\times10^7$ slots (up to 13B params).
- **Compared to OverEncoding** (Huang et al. 2025a — N-gram embeddings averaged into vocab embedding).
- **Result** (Fig. 3 right): val loss is **log-linear** in number of slots = clear power law. Engram unlocks much larger scaling than OverEncoding under same memory budget.

---

## 4. Large-Scale Pretraining

### 4.1 Setup

- 262B tokens, DeepSeek-V3 tokenizer (128k vocab).
- Backbone: 30 layers, $d=2560$, MLA attention (32 heads), mHC ($M=4$) expansion rate 4.
- Optimizer: **Muon** for backbone, **Adam** for Engram embeddings (5× LR multiplier, no weight decay).
- Conv params zero-init → preserve identity at start.

**Models**:
- **Dense-4B** — baseline, dense FFN every block.
- **MoE-27B** — DeepSeekMoE, 72 routed + 2 shared experts, top-$k=6$, 26.7B total / 3.8B active.
- **Engram-27B** — iso to MoE-27B: reduce routed experts 72→55 ($\rho \approx 74.3\%$), reallocate to 5.7B Engram memory. Inserts at **layers 2 and 15**, max N-gram=3, 8 hash heads, $d_{\text{mem}}=1280$.
- **Engram-40B** — same backbone + compute as 27B but 18.5B Engram (39.5B total).

### 4.2 Results (Table 1)

| Domain | Benchmark | MoE-27B | **Engram-27B** | $\Delta$ |
|---|---|---|---|---|
| LM | Pile loss | 1.960 | **1.950** | -0.010 |
| Knowledge | MMLU | 57.4 | **60.4** | +3.0 |
| | CMMLU | 57.9 | **61.9** | +4.0 |
| | MMLU-Pro | 28.3 | **30.1** | +1.8 |
| Reasoning | BBH | 50.9 | **55.9** | +5.0 |
| | ARC-Challenge | 70.1 | **73.8** | +3.7 |
| | DROP | 55.7 | **59.0** | +3.3 |
| Code | HumanEval | 37.8 | **40.8** | +3.0 |
| | MBPP | 46.6 | **48.2** | +1.6 |
| Math | GSM8K | 58.4 | **60.6** | +2.2 |
| | MATH | 28.3 | **30.7** | +2.4 |

- **Key finding**: gains exceed pure knowledge tasks — biggest wins in **reasoning / code / math**. Suggests memory frees backbone capacity broadly.
- Engram-40B further improves slightly but doesn't dominate 27B uniformly → likely under-trained, gap widens late in training.

---

## 5. Long-Context Training

- **Setup**: YaRN context extension to 32768 tokens, 5000 steps, 30B tokens.
- Compare MoE-27B (50k steps) vs Engram-27B at 41k / 46k / 50k steps.
    - **Iso-loss** at 46k (matches MoE-27B's pretrain loss).
    - **Iso-FLOPs** at 50k.

**Results** (Table 2 — RULER 32k Multi-Query NIAH):
- MoE-27B: 84.2 → Engram-27B 50k: **97.0** (+12.8).
- Variable Tracking: 77.0 → **89.0** (+12.0).
- LongPPL Code: 2.49 → **2.44**.
- Even at iso-loss (46k), Engram beats baseline: MQ NIAH 97.0 vs 84.2.

**Why?** Delegating local dependencies to lookups frees attention capacity for **global** context.

---

## 6. Analysis

### 6.1 Engram ≈ increasing effective depth

**LogitLens KL-divergence analysis** (Fig. 4a)
- Project each layer's hidden state through final LM head → compare to final output distribution via KL.
- Engram variants show **systematically smaller KL** at early/mid layers vs MoE-27B.
- Steeper descent → feature composition finishes earlier → backbone reaches "prediction-ready" state faster.

**CKA (Centered Kernel Alignment)** between Engram and MoE layer reps:
$$\text{CKA}(K,L) = \frac{\text{HSIC}(K,L)}{\sqrt{\text{HSIC}(K,K)\text{HSIC}(L,L)}}$$
- Soft alignment index $a_j$ = weighted top-$k$ centroid of MoE layers most similar to Engram layer $j$.
- High-similarity diagonal **shifts upward** → Engram layer 5 ≈ MoE layer 12 → **shallow Engram layers semantically deeper**.

> **Punchline**: by replacing early-layer feature composition w/ explicit lookup, Engram effectively *deepens* the network for the same compute.

### 6.2 Structural ablations (Fig. 5)

Reference config: 3B MoE + 1.6B Engram, val loss = 1.768 (vs MoE baseline 1.808, $\Delta = -0.04$).

**Layer placement sweep** (single Engram, layer 1→12):
- **Layer 2 optimal** (1.770), degrades for deeper insertion.
- Layer 1 worse: attention hasn't aggregated enough context for gating Query.
- Late layers worse: misses early offload window.
- **Splitting** the 1.6B into 2 modules at layers 2 + 6 → **1.768** (even better).
    - Reconciles "inject early to offload local patterns" vs "inject later for rich gating Query".

**Component ablation** (each removed):
- w/o multi-branch fusion: worst regression.
- w/o tokenizer compression: large regression.
- w/o context-aware gating: large regression.
- w/o short conv: marginal.
- + 4-gram (added 4-gram order): slightly worse — at 1.6B budget it dilutes 2/3-gram capacity.

### 6.3 Functional sensitivity (Fig. 6)

- At inference, **zero out Engram output** → measure retained perf.
- **Factual knowledge collapses**: TriviaQA retains only **29%**, PopQA 44%, MGSM 44% → Engram is the **primary parametric knowledge store**.
- **Reading comprehension robust**: C3 93%, RACE-Middle 89% → context-grounded tasks rely on attention.
- Clean dichotomy → confirms separation of static memory vs dynamic processing.

### 6.4 Inference throughput (Table 4)

- nano-vLLM, 100B Engram offloaded to host DRAM, layer 2 insertion.
- 4B Dense: 9032 → 8858 tok/s (**−1.9%**).
- 8B Dense: 6316 → 6140 tok/s (**−2.8%**).
- Conservative (all retrievals via PCIe) — proper Zipfian cache would be near-free.

### 6.5 Case study: gating visualization (Fig. 7)

- Gate $\alpha_t$ heatmap shows strong activation on completion of:
    - English multi-token entities: "Alexander the Great", "Milky Way", "Princess of Wales".
    - Phrases: "By the way".
    - Chinese idioms / historical entities: "四大发明" (Four Great Inventions), "张仲景".
- Generalizes cross-lingually → captures stereotyped patterns as intended.

---

## 7. Related Work (compressed)

- **N-gram embeddings**: FastText, Hash Embeddings (Tito Svenstrup 2017), Infini-gram (Liu 2024b), Nguyen 2024.
- **Embedding scaling**: Per-Layer Embeddings (Gemma), DeepEmbed (RWKV-V8), SuperBPE, **SCONE** (Yu 2025, infer-focused, extra train FLOPs), **OverEncoding** (Huang 2025a, embedding averaging — Engram beats it under iso-compute), Byte Latent Transformer (byte N-grams).
- **MoE**: Shazeer 2017, GShard, Switch, GLaM, **DeepSeekMoE** (fine-grained + shared experts).
- **Memory networks**: PKM (Lample 2019), PEER (He 2024), UltraMem (Huang 2025c), Memory+ (Berges 2025) — parametric K-V; REALM/RETRO — non-parametric.
- **FFN-as-KV-memory**: Geva 2021, knowledge neurons (Dai 2022), ROME/MEMIT — motivates why lookup helps.

### Engram vs comparable work

| Aspect | OverEncoding / SCONE | **Engram** |
|---|---|---|
| Treats N-gram as | external augmentation | **first-class modeling primitive** |
| Fair iso-compute eval | no | **yes** (Sparsity Allocation framework) |
| Embedding placement | input layer (serializes) | **deep layer** (overlap PCIe w/ compute) |
| Gating | averaging / direct add | **context-aware sigmoid gate** |
| Hardware co-design | no | **yes** (Zipfian cache, prefetch) |

---

## 8. Conclusion / Takeaways

- **Conditional memory** = new sparsity axis complementary to conditional computation (MoE).
- Engram instantiates it via modernized hashed N-gram embeddings + gating + conv + deep layer insertion.
- **U-shaped sparsity allocation law**: hybrid 75–80% MoE / 20–25% Engram is consistently optimal.
- Gains generalize beyond pure knowledge → general reasoning, code, math, long-context retrieval.
- Mechanistic: Engram **deepens** the model (early layers freed of static reconstruction) AND **frees attention** for global context.
- Systems: deterministic addressing → host DRAM offload at near-zero cost → bypasses HBM capacity wall.

### Connections / why this is interesting

- **Vs MoE**: MoE compresses *compute* spatially (route per token); Engram compresses *memory* spatially (index per N-gram). Hardware-wise: MoE creates dynamic comm patterns hard to overlap; Engram is fully prefetchable.
- **Vs RETRO / retrieval-augmented**: those retrieve *neighbor passages*, not learned embeddings; Engram is fully parametric but with static addressing.
- **Vs FFN-as-K-V hypothesis**: Engram makes the implicit explicit — instead of FFNs learning to *be* KV stores, give the model a real KV store.
- **Vs mixture-of-million-experts (PKM, He 2024)**: PKM uses product-key routing on hidden state (dynamic, like MoE); Engram routes on **input N-gram** (static, deterministic, prefetchable).
- **Brain analogy / name**: an *engram* in neuroscience is the trace of a memory in neural tissue — fits the "static memory primitive" framing.
- **Limitation hints**: 40B doesn't fully outperform 27B at 262B tokens → larger memory likely needs more tokens to saturate.

---

## Implementation

Demo code is a single self-contained file: `/home/jayoo/code/jayoo_personal/LLM_Knowledge_Base/raw_data/papers/Engram/Engram_repo/engram_demo_v1.py` (production version w/ CUDA kernels + distributed training is omitted).

### Config (lines 38–58)

- `EngramConfig`: `max_ngram_size=3`, `n_head_per_ngram=8`, `engram_vocab_size=[129280*5, 129280*5]` (for 2-gram and 3-gram), `n_embed_per_ngram=512`, `layer_ids=[1,15]`, `kernel_size=4`.
- `BackBoneConfig`: `hidden_size=1024`, `hc_mult=4` (mHC branches), `num_layers=30`, `vocab_size=129280`.

### Tokenizer compression — `CompressedTokenizer` (lines 60–121)

- Builds a `normalizer` pipeline using `tokenizers.normalizers`:
    ```python
    self.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), SENTINEL),
        normalizers.Strip(),
        normalizers.Replace(SENTINEL, " "),
    ])
    ```
- `_build_lookup_table` (84–110): iterates all `vocab_size` tokens, decodes each, normalizes, collapses to canonical IDs → returns `lookup` (np.int64 array old→new) and reduced vocab size.
- `_compress` (112–118): vectorized indexing `lookup_table[input_ids]`.

### Hashing — `NgramHashMapping` (lines 188–303)

- **Per-layer multiplier seeding** (218–231):
    ```python
    base_seed = int(seed + PRIME_1 * int(layer_id))
    g = np.random.default_rng(base_seed)
    r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
    multipliers = r * 2 + 1   # odd
    ```
    Each layer gets distinct odd multipliers → distinct hash functions.
- **Prime table sizes** — `calculate_vocab_size_across_layers` (235–260): for each layer × N-gram order × hash head, finds the **next prime** above `vocab_size - 1` not yet used (`find_next_prime` at 181). Guarantees disjoint prime moduli across heads → reduces collision.
- **Multiplicative-XOR hash** — `_get_ngram_hashes` (262–296):
    ```python
    base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
    for n in range(2, self.max_ngram_size + 1):
        tokens = base_shifts[:n]
        mix = (tokens[0] * multipliers[0])
        for k in range(1, n):
            mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
        for j in range(num_heads_for_this_ngram):
            mod = int(head_vocab_sizes[j])
            head_hash = mix % mod
            all_hashes.append(head_hash...)
    ```
    Each N-gram is built from shifted copies of the (compressed) input, multiplied position-wise and XOR-reduced. Padding ID handles sequence start.
- `hash` (298–303): top-level entry — compresses inputs, runs hashing per Engram layer, returns dict `{layer_id: hashes [B, T, num_heads_total]}`.

### Embedding table — `MultiHeadEmbedding` (lines 305–324)

- Concatenates all hash heads × N-gram orders into a **single big `nn.Embedding`** with offsets:
    ```python
    offsets = [0]
    for n in list_of_N[:-1]:
        offsets.append(offsets[-1] + n)
    self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)
    ```
- `forward`: shifts each hash head's IDs into the global table range and does one lookup.

### Short conv — `ShortConv` (lines 123–179)

- Depthwise causal 1D conv over the mHC-expanded hidden ($D \cdot M$ channels, `groups=total_channels`), kernel 4, `dilation=max_ngram_size` (=3).
- Per-branch RMSNorm before conv, SiLU after.
- Causal trim: `y_bct = y_bct[..., :T]`.

### Main Engram module — `Engram` (lines 326–378)

Forward pass implements the math from Sections 2.3–2.4 (gating + multi-branch fusion):

```python
def forward(self, hidden_states, input_ids):
    # hidden_states: [B, L, HC_MULT, D]; input_ids: [B, L]
    hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
    embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
    gates = []
    for hc_idx in range(backbone_config.hc_mult):
        key = self.key_projs[hc_idx](embeddings)           # branch-specific W_K
        normed_key = self.norm1[hc_idx](key)
        query = hidden_states[:, :, hc_idx, :]              # branch hidden
        normed_query = self.norm2[hc_idx](query)
        gate = (normed_key * normed_query).sum(-1) / math.sqrt(hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()   # signed-sqrt squashing
        gate = gate.sigmoid().unsqueeze(-1)
        gates.append(gate)
    gates = torch.stack(gates, dim=2)
    value = gates * self.value_proj(embeddings).unsqueeze(2)    # shared W_V
    output = value + self.short_conv(value)
    return output
```

- **Notes**:
    - `value_proj` (line 351) = **shared** $\mathbf{W}_V$ across branches.
    - `key_projs` (line 352) = **$M$ distinct** $\mathbf{W}_K^{(m)}$ — matches paper Sec. 2.4.
    - The `signed-sqrt → sigmoid` squashing on the gate score isn't explicit in the paper math (paper uses plain sigmoid of scaled dot product); demo adds it for stability.
    - `dilation = max_ngram_size` in `ShortConv` instantiation (line 347) — matches paper.

### Backbone integration — `TransformerBlock` (lines 380–394)

```python
def forward(self, input_ids, hidden_states):
    if self.engram is not None:
        hidden_states = self.engram(hidden_states=hidden_states, input_ids=input_ids) + hidden_states
    hidden_states = self.attn(hidden_states) + hidden_states
    hidden_states = self.moe(hidden_states) + hidden_states
    return hidden_states
```
- Engram comes **before** Attention/MoE in the block (residual added each time).
- `attn` and `moe` are mocked as identity in demo.
- Engram only present where `layer_id in engram_cfg.layer_ids` (=[1,15]).

### End-to-end driver (lines 396–422)

- Builds list of layers, runs the sample sentence `"Only Alexander the Great could tame the horse Bucephalus."`.
- Mock hyper-connection: expand `hidden_states` to `[B, L, hc_mult, D]` after embedding, collapse back before LM head.
- Prints final shapes — pure correctness/forward-pass sanity check, no training.

### What's omitted vs production

- **All-to-All sharding** of the embedding table across GPUs (paper Sec. 2.5 / Fig. 2a).
- **CPU-offload + PCIe prefetch** for inference (Fig. 2b).
- Custom CUDA kernels for the hash + lookup.
- Real Attention (MLA), real MoE (DeepSeekMoE 55+2 experts), real mHC (Manifold-Constrained Hyper-Connections).
- The per-Engram layer-wise hash multiplier randomization is shown, but in practice would be deterministic per checkpoint.
