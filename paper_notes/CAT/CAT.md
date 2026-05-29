# CAT: Compress & Attend Transformer

> **"Attention and Compression is all you need for Controllably Efficient Language Models"** — Prakash, Puli, Ranganath (NYU, 2025). arXiv:2511.05313
> Code: `github.com/rajesh-lab/cat-transformer`

**TL;DR**: Two ingredients only — **dense attention + compression**. Decode each chunk by attending to *compressed* representations of all past chunks. Chunk size $C$ is a **test-time knob** for the quality-compute tradeoff; train across multiple $C$ at once → one model interpolates between dense transformer and efficient alternatives **without retraining**. Matches dense at LM, $1.4$–$3\times$ faster, $2$–$9\times$ lower memory; even **beats dense on in-context recall** at small $C$.

---

## 1. Motivation

- **Quadratic attention** ($O(N^2)$ compute, $O(N)$ KV-cache memory) makes transformers expensive to deploy.
- **Existing efficient methods all have a fixed apriori quality-compute tradeoff** baked into the architecture, plus other warts:
  - **Sparse / sliding-window attention** (Child, Zaheer, Longformer): heuristic masks artificially restrict which tokens are attended. Wrong mask for the task → bad performance or needs more depth. To match dense, need big windows (kills memory savings) or compose with dense layers at specific layers.
  - **Linear attention / SSMs** (Mamba2, GatedDeltaNet, Katharopoulos): fixed-size recurrent state + handcrafted/complex state-update rules → constant compute & memory but **fixed-size state struggles with in-context recall over long sequences**. To stay competitive they get **composed with sliding-window attention** via trial-and-error → cumbersome design at scale.
  - **Recursive compression** (Rae/Compressive-Transformer, Chevalier/AutoCompressor): compress past context to enable longer gen, but **sequential/recurrent training is slow + hard to optimize** (BPTT, optimization collapse — see Geiping 2025).
  - **Hierarchical / MegaByte / Block Transformer** (Nawrot, Yu, Ho): hour-glass downsample-then-upsample, or compress each chunk into a **single fixed-size** representation → **poor in-context recall even on toy tasks** + still cache all past tokens (memory bottleneck).
- **Key gap**: downstream tasks differ in compute/memory needs (short email reply vs. code autocompletion needing repo-wide recall). Fixing the tradeoff pre-training means a new model per budget; holding many models in memory + routing is prohibitive.
- **Wish list** (Table 1): unrestricted memory access, flexible memory, scalable (parallel) training, both compute+memory efficient, **adaptive**. Only CAT ticks all boxes.

---

## 2. CAT Architecture

> **Core idea**: split sequence into chunks, **compress each chunk in parallel** into a short representation, then a **decoder autoregressively decodes each chunk while attending to compressed reps of all past chunks** (and raw tokens within the current chunk).

### 2.1 Compression + decoding

- Sequence $\mathbf{x}=(x_1,\dots,x_N)$ split into $N/C$ chunks of $C$ tokens each: $\mathbf{c}_i = \mathbf{x}_{i,:} = (x_{C\cdot i+1},\dots,x_{C\cdot i + C})$.
- **Compressor** $f_\theta$: a **dense bidirectional** transformer (hidden $D_f$) + linear projection to $D_g$ → chunk rep $f_\theta(\mathbf{c}_i)\in\mathbb{R}^{D_g}$.

$$
\mathbf{x}=\{x_1\cdots x_N\} \xrightarrow{\text{chunk}} \{\mathbf{c}_1\cdots\mathbf{c}_{N_c}\} \xrightarrow{\text{compress}} \{f_\theta(\mathbf{c}_1)\cdots f_\theta(\mathbf{c}_{N_c})\}
$$

- **Decoder** $g_\theta$: a **causal** dense transformer (hidden $D_g$). Decodes token $\mathbf{x}_{i,j}$ in chunk $\mathbf{c}_i$ from **previous tokens within the chunk** $\{\mathbf{x}_{i,<j}\}$ + **past chunk reps** $\{f_\theta(\mathbf{c}_1),\dots,f_\theta(\mathbf{c}_{i-1})\}$:

$$
p_\theta(\mathbf{c}_i \mid \mathbf{c}_{i-1}\cdots\mathbf{c}_1) = \prod_{j=1}^{C} g_\theta\!\Big(\underbrace{\mathbf{x}_{i,j}}_{\text{tgt}} \;\Big|\; \underbrace{\mathbf{x}_{i,j-1},\dots,\mathbf{x}_{i,1}}_{\text{prev in chunk}},\; \underbrace{f_\theta(\mathbf{c}_{i-1})\cdots f_\theta(\mathbf{c}_1)}_{\text{past chunk reps}}\Big)
$$

- **No recurrence along the sequence**: compression of a chunk doesn't depend on earlier chunks → fully **parallel over tokens during training**, end-to-end **scalable** with vanilla next-token loss.
- **Why it works (intuition)**: decoder operates on a **reduced sequence length** ($\approx N/C$ chunk reps + $C$ local tokens instead of $N$). End-to-end training makes CAT **learn what to retain** in chunk reps rather than relying on fixed masks / handcrafted state rules. Larger $C$ → more reduction → more savings (but less retained info). Contrast w/ Block Transformer/MegaByte: they feed only a **single "global" rep** to decoder; CAT feeds **all** past local chunk reps → fixes their recall learnability problems.

### 2.2 Test-time control (the "adaptive" trick)

- **Vary $C$ → trade quality vs. efficiency.** Train **one** model over multiple chunk sizes to get a single adaptive model whose budget is set at test-time, no retraining.
- **How**: at each training iter, **uniformly sample chunk size $C$**, pass a **learnable indicator token** telling CAT which $C$ it's operating at. Compressed tokens separated from uncompressed in decoder via a **marker token shared across chunk sizes**. At test-time just swap the indicator token to change budget.
- Default training set: $C\in\{4,8,16,32\}$ → **CAT-4/8/16/32 are the same single model** at different knob settings.
- **Implementation detail (App B.4)**: the compressor's projection matrix shape depends on $C$. Independent matrix per $C$ trains poorly (updated only when that $C$ sampled). Solution (à la Beyer/FlexiViT): **one projection matrix, linearly interpolated to the needed shape under `torch.autograd`** so the interpolation is optimized.
- Spiritually like **MoE** / Matformer / FlexiViT: more params ≠ more compute; adaptive granularity.

### 2.3 Math of compute/memory savings

| Aspect | Cost | Notes |
|---|---|---|
| **Compression** | $O(\tfrac{N}{C}\cdot C^2)=O(NC)$ | each chunk self-attends over $C$ tokens, all chunks in parallel via `torch.vmap` |
| **Naive decoder training** | $O((N/C)^3)$ **cubic** | varying #past-reps per chunk → padding/loops don't scale |
| **CAT decoder training (smart)** | $O(\tfrac{N}{C}N + \tfrac{N}{C}C^2)=O(\tfrac{N^2}{C})$ | **$C\times$ better than $O(N^2)$**; exact cost $\sum_{i=1}^{N}\lfloor\tfrac{i}{C}\rfloor + (i\bmod C)+1$ |
| **Generation: tokens attended** | at most $N_c + C$ | vs. $N$ for dense |
| **Generation: KV-cache** | slashed by factor $\approx C$ | throw away raw past tokens, keep only compressed reps |

- **Memory grows *gracefully***: linear in $N$ but much slower slope than dense → enables long-context recall while staying efficient.

### 2.4 Fast & scalable training (App B.1–B.2)

- **Trick to avoid recompute**: computing logits for chunk $\mathbf{c}_i$ recomputes the **same KV vectors** for $f_\theta(\mathbf{c}_j)$, $j<i$. Reuse them.
- **Interleave reps into the sequence**: transform $\{\mathbf{c}_1,\dots\}$ → $\{\mathbf{c}_1, f_\theta(\mathbf{c}_1),\mathbf{c}_2,f_\theta(\mathbf{c}_2),\dots\}$, i.e. insert each chunk's compressed rep right after the chunk's tokens.
- Feed to decoder with a **custom attention mask** (Fig 8, block-diagonal-ish, looks like Child et al. sparse mask but **not heuristic**): a token in $\mathbf{c}_i$ attends to (a) prev tokens **within its chunk** and (b) **only past chunk reps** $f_\theta(\mathbf{c}_{i-1})\cdots f_\theta(\mathbf{c}_1)$ — never raw tokens outside its chunk. Enables KV reuse → parallel logit computation for all chunks.
- Mask implemented via **FlexAttention** (Dong 2024) — pure PyTorch, **no custom CUDA/Triton kernels**.

```python
# Training step (App B.1, Listing 1)
def forward(input_ids, targets):
    input_ids = einops.rearrange("b (k c) -> b k c", k=num_chunks, c=chunk_size)
    fx = torch.vmap(f)(input_ids)             # compress all chunks in parallel -> (b, k, D_d)
    output_logits = []
    for i in range(num_chunks):               # done in PARALLEL via the custom mask
        cur = phi(input_ids[:, i, :], fx[:, :i+1, :])  # predict chunk i from prev reps
        output_logits.append(cur)
    output_logits = torch.cat(output_logits, dim=1)
    output_logits = einops.rearrange(output_logits, "b k c v -> b (k c) v")
    return torch.nn.functional.cross_entropy(output_logits, targets)
```

### 2.5 Generation (App B.3)

- **Simpler than training**, like a dense transformer. Pure PyTorch (inspired by `gpt-fast`), on par with kernel-using efficient archs.
- Loop: compress a chunk in parallel → **prefill decoder KV cache** with its rep → autoregressively decode tokens of the **next** chunk one at a time (causal mask, prev reps already prefilled) → once chunk done, compress it & prefill its rep → repeat.

```python
# Generation (App B.3, Listing 2)
def generate_chunk_by_chunk(input_ids):       # (b, 1, chunk_size)
    fx = f(input_ids)                          # compress first chunk
    next_token = prefill(fx, input_pos=0)
    new_chunks = []
    for i in range(num_chunks - 1):
        next_chunk = generate_chunk(next_token)  # decode whole chunk token-by-token
        new_chunks.append(next_chunk.clone())
        fx = f(next_chunk)                       # compress newly generated chunk
        input_pos += 1
        next_token = prefill(fx, input_pos)      # prefill its rep
    return torch.cat(new_chunks)
```

---

## 3. Related Work — conceptual positioning (Table 1)

| Method | Unrestricted mem access | Flexible mem | Scalable train | Compute+mem efficient | Adaptive |
|---|:--:|:--:|:--:|:--:|:--:|
| Dense (Vaswani) | ✓ | ✓ | ✓ | ✗ | ✗ |
| Sparse (Child) | ✗ | ✓ | ✓ | ✓ | ✗ |
| NSA (Yuan) | ✓ | ✓ | ✓ | ✗ | ✗ |
| Sliding window (Jiang) | ✗ | ✗ | ✓ | ✓ | ✗ |
| Linear attn (Dao&Gu) | ✓ | ✗ | ✓ | ✓ | ✗ |
| Recursive compression (Chevalier) | ✓ | ✓ | ✗ | ✓ | ✗ |
| MegaByte/Block (Ho, Yu) | ✓ | ✗ | ✓ | ✓ | ✗ |
| **CAT** | **✓** | **✓** | **✓** | **✓** | **✓** |

- **vs NSA**: NSA compresses past in parallel every layer but **must retain full KV cache → no memory savings at inference**, only compute.
- **vs Block Transformer/MegaByte**: those compress history into a **single fixed-size global rep** fed to decoder; CAT feeds **all** past chunk reps directly → solves long-range recall, fixes their learnability problems (Fig 7: Block Transformer **fails simple MQAR**, memorizes train pts, val accuracy flat). CAT uses **only 2 components** (compressor + decoder) vs their 3 (embedder + global + local decoder) → smaller design space & fewer params.
- **vs Barrault (Large Concept Models)**: their encoder is an autoencoder (keeps irrelevant info); CAT compressor keeps **only info predictive of future chunks**.

---

## 4. Experiments

### Setup
- **Train data**: 15B tokens of FineWeb-Edu ($2.5\times$ Chinchilla-optimal), context length 4K, GPT2 tokenizer, AdamW, peak LR 8e-4, wd 0.1, grad clip 1.0, batch 0.5M tokens.
- **Baselines** ($L=12$, $D=1024$, $\le 300M$ params except Sparse which uses $2D=2048$ → ~800M):
  - Dense (Transformer++), **Sparse** (Child), **Mamba2** (linear), **GDN** (GatedDeltaNet, linear), **GDN-Hybrid** (alternate layers = 2K sliding window).
- **CAT config**: decoder depth $L=12$, **wider** hidden $D_g=2D=2048$; compressor $L=3$, $D_f=D=1024$. Total params **~1B**. Single adaptive model trained over $C\in\{4,8,16,32\}$.

> **"What makes CATs purr?"** Matching dense perplexity needs a **more expressive decoder** ($2\times$ hidden size). Accurate decoding *from compressed reps* needs extra compute (echoed by MegaByte/Block works). **Compressor depth has little effect** on perplexity (ablation).

### 4.1 Language modeling + common-sense reasoning (Table 2)
- Metrics: LMBADA, WikiText, FineWeb (ppl ↓); HellaSwag, PIQA, ARC-E, ARC-C, WinoGrande, OpenBookQA (acc ↑).
- **CAT-4/8/16 match or beat all baselines** on LM (ppl), match dense at scale. All CAT variants **beat efficient baselines on common-sense reasoning avg**.
- **CAT-32 best on reasoning & LM understanding** (Avg 45.1 vs dense 42.1) — because these evals have short sequences ($\le 30$ tokens avg) so the decoder consumes raw tokens with little compression.

### 4.2 Real-world in-context recall (Table 3, EVAPORATE/Arora; Fig 1)
- SWDE + FDA (long-sequence subsets), measured at 2K.
- **CAT-4 beats even the Dense transformer**: SWDE 49.1 / FDA 45.1 / avg 47.1 vs dense 43.4 / 19.7 / 32.0 — while $\ge 1.5\times$ faster, $2\times$ more memory efficient (MoE-like). Hypothesis: reduced sequence length → **fewer distractions** for attention (info over-squashing literature).
- **CAT-8 > GDN-Hybrid** at similar memory. **CAT-16 > Mamba2/GDN**, $1.15\times$ faster (~15% more memory).
- Linear models (Mamba2, GDN) lag far behind dense; GDN-Hybrid narrows but doesn't close gap. **CAT surpasses nearly all efficient baselines across budgets with ONE adaptive model.**

### 4.3 Long-context understanding (Table 4, LongBench up to 4K)
- Single-doc QA (qasper, multifieldqa), multi-doc QA (hotpotqa, 2wikimqa), few-shot (triviaqa, trec).
- **CAT-4/8/16 outperform all baselines**; expected ordering between CAT variants.

### 4.4 Synthetic needle-in-haystack + state-tracking (Table 5)
- **RULER S-NIAH-N** (recall a number) & **S-NIAH-U** (recall 32-token UUID, harder) at 1K/2K/4K; **BabiLong** (qa1 state-tracking) at 0K–4K.
- Linear models (Mamba2, GDN) struggle at long contexts; GDN-Hybrid narrows but drops at length. **CAT-4/8/16 outperform efficient baselines as context grows** (slower degradation than even dense — fewer distractions from shorter seq).
- **Large-chunk CAT underperforms at short contexts but surpasses baselines at long ones.** Reason: large chunks → ineffective compression, compressor can't always surface the *right* info; finetuning/more pretraining helps (App A.6). **Fundamental limit** on info a fixed-size chunk rep can hold.
- BabiLong (state-tracking): all decline as context grows; linear recurrent models do relatively better here.

### 4.5 Efficiency benchmarks (Fig 3)
- Fixed batch 320, varying gen length. **CAT generates $1.4$–$3.2\times$ faster** than dense, **up to $2.2$–$9.5\times$ lower total memory** as $C$ increases — *despite more params* (wider decoder + compressor).
- Why params don't hurt gen: bottlenecks are (a) **KV cache size** (not param count), (b) memory accesses per token, (c) FLOPs/token set by #attended tokens — CAT slashes all three.

### 4.6 Scaling (Fig 5)
- Dense scales $\{31,92,260\}M$ ↔ CAT equivalents $\{95,326,1000\}M$, all 15B tokens. **CAT scales like dense counterparts** while up to $3\times$ faster, $9\times$ more memory efficient.

### 4.7 Memory-matched MQAR (Fig 4; App A.4)
- Stress test on MQAR up to 1K seq len, **memory matched down to bytes**. **CAT stays near-perfect** at long contexts (flexible memory); linear models degrade.
- App A.4 (Table 8): at 2 decoder layers, **Sparse & Sliding-window FAIL MQAR** (depend on depth); **CAT solves it** with same 4096 state size.

---

## 5. Ablations & analysis (App A, C)

- **Compressor hidden size $D_f$** (C.1): bumping 768→1536 → **no change** in ppl. Compressing small chunks needs little capacity.
- **Decoder hidden size $D_g$** (C.2, C.3): $D_g=2D$ clearly beats $D_g=D$ across all $C$ (e.g. $C=4$: 17.4 vs 19.8 ppl). Table 13: CAT with $D_g=2D$ gets ppl 20.7 / recall 19.8 vs dense 21.2 / 23.8 → wider decoder is what closes the gap.
- **Compressor depth** (C.3): 3 vs 6 layers → **same ppl**. Shallow compressor OK → goes into CAT's efficiency advantage (less latency, less MLP train cost). Open Q: lower bound on depth.
- **Fixed-$C$ vs adaptive** (A.3, Table 7): **fixed-chunk CATs beat adaptive** on recall (e.g. CAT-4-fixed avg 34.5 vs adaptive 31.1) — adaptive decoder loses capacity juggling multiple $C$. More pretraining can recover. Still, adaptive beats all baselines.
- **Finetuning on S-NIAH-U** (A.6, Table 10): huge jumps (CAT-4: 46.3→97.1, CAT-8: 47.3→97.0, CAT-16: 3.8→94.2, CAT-32: 0.0→64.3). Confirms compressor *can* learn to surface the right info but **CAT-32 loss doesn't reach zero → hard limit** for big chunks.
- **CAT as a layer** (A.5, Table 9): parameterize compressor as a linear projection, reuse dense attention as decoder → CAT-as-layer **also solves MQAR** at 4096 state. Future: hybrid/adaptive nets mixing CAT layers with dense/linear attention.

---

## 6. Limitations & future work

- **Adoptability / training cost (App B.5)**: current training ~$2\times$ slower wall-clock due to **inefficient FlexAttention compilation** of the custom mask — theoretical FLOP savings **don't show up in training wall-clock** (no efficient training kernel yet; MLPs dominate FLOPs at small seq len). At 4K, CAT $\le 2.35\times$ slower to train than dense. **Custom kernel = future work.** Training is a **one-time cost**; serving cost matters more.
  - Serving argument: e.g. Qwen3-14B at batch 16 needs ~670GB KV cache vs 28GB weights — CAT cuts total mem ~$4\times$ at $C=32$ ($28\cdot(4+\tfrac14)+\tfrac{670\cdot2}{32}=160$GB). Big win for **RL post-training rollouts, synthetic data gen, large batches**.
- **Big fixed-size chunk reps fundamentally cap recall** — can't perfectly hold arbitrary info; small $C$ or task-specific finetuning needed for hard recall.
- **Not data-dependent / not auto-adaptive**: user must pick $C$ for their budget. Future: **post-train with RL to let CAT allocate budget itself** based on context/task → truly adaptive efficiency.
- **Future directions**: linear attention *as compressor* + dense decoder; CAT-as-a-layer hybrids; compression that only kicks in after e.g. 1K tokens (compress old context only); scale to larger models + longer training.

---

## Key numbers cheat-sheet

| Claim | Value |
|---|---|
| Generation speedup vs dense | $1.4$–$3.2\times$ faster |
| Total memory reduction vs dense | $2.2$–$9.5\times$ lower |
| CAT-4 (least efficient) vs dense on recall | wins, still $\ge 1.5\times$ faster, $2\times$ less mem |
| Decoder training complexity | $O(N^2/C)$ ($C\times$ better than dense) |
| KV-cache reduction | $\approx C\times$ |
| Decoder attended tokens (gen) | $\le N_c + C$ |
| Default chunk sizes (single model) | $C\in\{4,8,16,32\}$ |
| CAT params (main exp) | ~1B (decoder $L{=}12$, $D_g{=}2048$; compressor $L{=}3$, $D_f{=}1024$) |

---

## Implementation

> Repo: `github.com/rajesh-lab/cat-transformer`. **Single-file, hackable, pure-PyTorch** (no custom CUDA/Triton beyond Liger fused ops). Borrows from litgpt (transformer), Liger-Kernel (fused RMSNorm/SwiGLU/CE), gpt-fast (generation). `torch==2.5.1+cu121` + `liger-kernel`. Released HF checkpoints (`bicycleman15/cat-transformer`) are *re-trained* with the repo code at slightly different config (adds QK-norm) — not the exact paper models, same trends.

### File / module layout

| File | Role |
|---|---|
| `transformer.py` | Vanilla Transformer++ (`TransformerConfig`, `TransformerBlock`, `Attention`, `RMSNorm`, `KVCache`, `build_rope_cache`, `apply_rope_emb`, `get_mask_mod`, `_init_weights`). CAT imports these and builds on them. |
| `cat_transformer.py` | **Fixed-chunk** CAT (`CAT_Config`, `Compressor`, `CAT_Transformer`). Single $C$ per model. |
| `cat_transformer_adaptive.py` | **Adaptive** CAT — multi-$C$, the test-time-controllable model. Adds adaptive/conditioning tokens + interpolated projection. |
| `cat_layer.py` | **CAT-as-a-layer** (`CAT_Attention`, `CAT_Transformer_Block`, `CAT_Layer_Transformer`) — drop-in dense-attention replacement. |
| `train.py` | Hydra + `accelerate` (DDP) training entry point. |
| `generate.py` | Simple (slow, re-encode-every-step) sampling for the HF ckpts. |
| `benchmark.py` | Fast KV-cached chunk-by-chunk generation (gpt-fast style) for throughput/memory. |
| `utils.py`, `data.py`, `data_loader.py`, `prepare_sharded_data.py`, `eval/` | model factory, FineWeb sharded loader, harness/ppl/recall evals. |
| `fineweb.yaml`, `sample_run.sh`, `accelerate.yaml` | configs + launch commands. |

- **Two components only** (`self.f` compressor + `self.layers` decoder), matching the paper's "2 vs 3 components" claim. No separate embedder — compressor and decoder each have **their own `wte`** (untied).

### `CAT_Config` (extends `TransformerConfig`)

- Adds `chunk_size` (max $C$) and `dim_fx` (compressed-rep size, must equal decoder `dim`). `__post_init__` asserts `block_size % chunk_size == 0`, sets `num_chunks = block_size // chunk_size`.
- `head_dim = dim // n_head`, `rope_n_elem = head_dim` (full-dim RoPE), `padded_vocab_size = find_multiple(vocab, 256)`.
- **Decoder is wider**: instantiated with `dim = dim_fx = 2*D`, `n_head = 2*n_head`; **compressor** with `dim = D`, `n_layer = num_layers//4` (e.g. 3). Hard `assert f.dim_fx == decoder.dim`.

### Compressor $f_\theta$ (`Compressor.compress`)

- Bidirectional (`is_causal=False`) stack of `TransformerBlock`s over **one chunk at a time**. Prepends learnable tokens, runs layers, drops the prepended slots, flattens, projects:

```python
def compress(self, input_ids, chunk_idx, chunk_size_power):   # adaptive variant
    x = self.wte(input_ids)                                    # (b, l, D)
    pos_token = self.pos_tokens(chunk_idx)                     # which chunk position (0..num_chunks)
    adaptive_token = self.adaptive_token(chunk_size_power)     # which chunk size is active
    x = torch.cat([adaptive_token, pos_token, x], dim=1)       # (b, 2+l, D)
    for layer in self.layers: x = layer(x, cos, sin, is_causal=False)
    x = self.norm(x)[:, 2:, :]                                 # drop the 2 special slots
    x = einops.rearrange(x, 'b l d -> b (l d)')                # (b, l*D)  flatten chunk
    # interpolate the projection matrix to the active chunk size (see below)
    x = F.linear(x, new_proj_fx_weight)                        # (b, dim_fx)
    return x
```

- **Two prepended markers** (adaptive variant): a **`pos_tokens` embedding** (tells $f$ *which chunk index* it is — needed because compression is otherwise positionless) and an **`adaptive_token` embedding** of size `log2(chunk_size)+1` (tells $f$ *which $C$*). Fixed variant prepends **only** `pos_tokens` (one marker).
- A chunk compresses to a **single** $D_g$-vector (`proj_fx`: `Linear(dim*chunk_size -> dim_fx)`); the whole chunk's per-token states are concatenated then linearly mixed. So $f$ self-attends over $\le 2+C$ tokens.

### Compressing all chunks in parallel — `torch.vmap`

- `forward` reshapes `(B,L) -> (B, K, C)` then **vmaps `compress` over the chunk axis**:

```python
input_ids = input_ids.view(bsz, cur_num_chunks, cur_iter_chunk_size)        # (B,K,l)
fx = torch.vmap(self.f.compress, in_dims=(1, 0, None), out_dims=1)(
    input_ids,                                              # mapped over dim 1 (chunks)
    torch.arange(cur_num_chunks, device=...),               # chunk_idx mapped over dim 0
    torch.tensor(chunk_size_power, ...),                    # broadcast (None)
)                                                           # (B, K, D_fx)
```

- This is the literal `torch.vmap(f)` from the paper's Listing 1. **Liger fused ops are disabled in the compressor** (`use_fused_ops=False`) because Liger kernels don't support `vmap`; the wide decoder still uses fused RMSNorm + `LigerFusedLinearCrossEntropyLoss`.

### Interleaved-reps sequence + FlexAttention mask (the core training trick)

- **The custom mask** (Fig 8) is 3 lines, parameterized by an effective block stride:

```python
def get_cat_mask(block_size):
    def cat_mask(b, h, q_idx, kv_idx):
        within_block  = (q_idx // block_size) == (kv_idx // block_size)  # same chunk-block
        divides_block = (kv_idx % block_size) == 0                       # the f(c) rep slots
        causal_mask   = q_idx >= kv_idx
        return (divides_block | within_block) & causal_mask
    return cat_mask
```

- Read it as: a query attends to (a) everything **in its own block** and (b) the **first slot of every block** (the compressed reps, which sit at positions $\equiv 0 \pmod{\text{stride}}$), all under a causal mask. Compiled into a `BlockMask` via `create_block_mask = torch.compile(create_block_mask)`. **Pure FlexAttention** — confirms paper's "no kernels" claim (and explains the ~2× train slowdown: it's `torch.compile`'d FlexAttention, no bespoke kernel).
- **Stride differs between variants** because the per-chunk layout differs:
  - **Fixed** (`cat_transformer.py`): each chunk block = `[f(c_{i-1}), x_i,1..x_i,C]` → stride `1 + chunk_size`.
  - **Adaptive** (`cat_transformer_adaptive.py`): inserts a **shared `seperator` embedding** between the rep and the tokens → block = `[f(c_{i-1}), sep, x_i,1..x_i,C]` → stride `1 + 1 + chunk_size`.
- **Sequence construction** (adaptive `forward`): the first rep slot is replaced by a **conditioning vector `dummy_fx`** (an `Embedding(log2(C)+1, D)` indexed by `chunk_size_power` — this is the *decoder-side* "which $C$" token, distinct from the compressor's `adaptive_token`), then reps are shifted right by one so each chunk attends to the *previous* rep, and the last rep is appended at the tail:

```python
dummy_fx = self.dummy_fx(chunk_size_power)                       # decoder conditioning token
fx = torch.cat([dummy_fx, fx[:, :-1, :]], dim=1)                 # shift reps right by 1
fx = rearrange(fx, 'b k d -> b k 1 d')
sep = self.seperator(0)...                                       # shared marker, all C
emb_x = self.wte(input_ids)                                      # (B,K,l,D)
x = torch.cat([fx, sep, emb_x], dim=2)                           # (B,K, 2+l, D)
x = rearrange(x, 'b k l d -> b (k l) d')
x = torch.cat([x, fx_last, last_sep], dim=1)                     # tail rep so last chunk can be decoded
```

- After the decoder layers, the output is **un-interleaved** back to `(B, L, D)`: drop the rep/sep slots, re-glue `x_first` (first chunk skips its own rep) + `x_middle` + `x_last`, then slice off the 512-pad. Loss is fused linear-CE directly on hidden states (no materialized full logits) when `use_fused_ops`.
- **Padding hack**: `forward` pads `seqlen` up to the next multiple of **512** (`F.pad(..., value=0)`, sliced off at the end) specifically to **cut FlexAttention recompilations**; `torch._dynamo.config.cache_size_limit = 8`.

### Test-time chunk-size control — interpolated projection (FlexiViT trick)

- The compressor's `proj_fx` weight is `(dim_fx, dim*chunk_size)` — shape **depends on $C$**. Rather than one matrix per $C$, a **single** matrix is **bilinearly interpolated** to the size the active $C$ needs, *inside autograd* so the interpolation is learned (cites FlexiViT, arXiv:2212.08013):

```python
new_proj_fx_weight = F.interpolate(
    self.proj_fx.weight[None, None],                           # (1,1, dim_fx, dim*C_max)
    size=(dim_fx, (2 ** chunk_size_power) * dim),              # target in-features for active C
    mode="bilinear",
).squeeze(0).squeeze(0)
x = F.linear(x, new_proj_fx_weight)
```

- **RoPE resets per chunk** (see README): instead of global positions, build a length-`(2+C)` cache and **repeat it per chunk**, then trim. Aligns with the block-diagonal mask (tokens only attend within-chunk + reps, so global positions are meaningless). Adaptive variant stores a **dict of caches keyed by `chunk_size_power`** (`self.cos[c]`, `self.sin[c]`), built per valid power in `setup_cache`; a separate non-resetting `cos_gen/sin_gen` (from the **largest** $C$) is used for autoregressive generation.

### Generation / prefill loop (`benchmark.py`, gpt-fast style)

- KV cache sized `(num_chunks + chunk_size)` per layer (`setup_kv_cache`) — i.e. **reps + one chunk of local tokens**, not $N$ — this is the $\approx C\times$ KV reduction made concrete.
- `generate_chunk_by_chunk`: prefill `dummy_fx` → compress prompt chunk → loop: `generate_chunk` does `prefill_fx` (insert this chunk's rep into cache) then `decode_n_tokens` (autoregress $C-1$ local tokens, `decode_one_token` argmax) → compress the freshly generated chunk → repeat. `rope_pos` resets to 0 each chunk (resetting RoPE); `input_pos` keeps advancing (cache slot). `decode_one_token`/`prefill_fx` are `torch.compile(mode="reduce-overhead", fullgraph=True)` (CUDA graphs). `mask_mod` is offset per step via `get_mask_mod`.
- `generate.py` (HF ckpts) is the **slow** path: re-runs the full `model(cur, chunk_size_power=...)` each step (no KV cache) — fine for short demos, `--chunk_size_power` is the test-time knob (`3` → $C{=}8$).

### CAT-as-a-layer (`cat_layer.py`)

- `CAT_Attention` does the whole compress→interleave→mask→un-interleave **inside one attention module** in an expanded hidden dim `dim_fx`:
  - **compressor = a single `nn.Linear(chunk_size*dim -> dim_fx)`** (not a transformer) + `dummy_fx`; tokens `expand`ed `dim -> dim_fx`; reuses the **same `get_cat_mask`** for block-diagonal CAT attention; `final_proj` back to `dim`.
- `CAT_Transformer_Block` subclasses `TransformerBlock` swapping in `CAT_Attention`; "every layer is CAT" in the demo but the design is meant for **hybrids** (mix CAT + dense + linear layers). This is the variant the paper shows also solves MQAR at matched state size (App A.5).

### Training entry point + hyperparameters

- `train.py`: Hydra config (`fineweb.yaml`) + `accelerate` DDP (`find_unused_parameters=True`), bf16, `torch.set_float32_matmul_precision("high")`. AdamW, betas `(0.9, 0.95)`, wd `0.1`, peak lr `8e-4` → min `8e-5`, grad clip `1.0`, cosine LR w/ warmup.
- **Adaptive training = round-robin over chunk sizes** (not random per the paper text, but cyclic in code):

```python
max_power = int(math.log2(cfg.model.chunk_size)); min_power = 2   # C in {4..chunk_size}
chunk_size_powers = list(range(min_power, max_power + 1))
...
chunk_size_power = chunk_size_powers[iter_num % len(chunk_size_powers)]   # cycle each micro-step
loss = model(input_ids, targets, chunk_size_power=chunk_size_power)
```

  - Note: code's **min chunk is $C{=}4$** (`min_power=2`), matching the paper's $\{4,8,16,32\}$ default; the compressor's `adaptive_token`/RoPE caches still cover powers $0..\log_2 C$.
  - **Validation** sweeps every power and logs per-$C$ ppl (`val/perplexity` reported at the **max** $C$).
- **Paper-scale run** (`sample_run.sh`): `block_size=4096`, `chunk_size=32`, decoder `dim=2048 / n_head=32 / n_layer=12`, compressor `dim=1024 / n_head=16 / n_layer=3`, `dim_fx=2048`. `batch=32 × grad_accum=4 × 4096 = 0.5M tokens/step`, `114440` micro-steps → `28610` optimizer steps ≈ **15B tokens** (matches paper). QK-norm on.
- Checkpoints saved as `state_dict.pt` (+ periodic `intermediate_state_dict_*.pt`); `generate.py` strips `module.` DDP prefixes and `load_state_dict(strict=True)`.

---

## Related Notes

- **[Artificial Hippocampus Networks (AHN)](../AHN/AHN.md)** — Closest sibling (both 2025, efficient long-context via compression). Contrast: CAT compresses **fixed-size chunks in parallel** and the decoder **attends to all chunk reps** (growing-but-compressed cache, $\approx C\times$ reduction); AHN compresses recent KV into a **constant-size recurrent state** + a lossless sliding window. CAT's chunk size $C$ is a **test-time** quality/compute knob; AHN's ratio is fixed at train time.
- **[AutoCompressors](../AutoCompressors/AutoCompressors.md)** — Shares the "compress spans → condition on compressed reps" idea. AutoCompressors accumulates lossy **summary-vector soft prompts recurrently** across segments; CAT compresses chunks **non-recurrently in parallel** and attends causally to all reps, avoiding sequential summary threading.
- **[Recurrent Memory Transformer](../RMT/RMT.md)** — An instance of the recursive/recurrent-compression family CAT positions itself against: RMT threads a single fixed memory across segments with BPTT, while CAT's parallel chunk compression sidesteps the sequential dependency and (the paper argues) the recall bottlenecks of recursive compression.
