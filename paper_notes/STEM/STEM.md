# STEM: Scaling Transformers with Embedding Modules

**Authors**: Sadhukhan, Cao, Dong, Zhao, Purpura-Pontoniere, Tian, Liu, Chen (CMU + Meta AI / Infini-AI-Lab), arXiv:2601.10639, Jan 2026.

---

## TL;DR

- **Idea**: Replace the **FFN up-projection** $\mathbf{W}^u \in \mathbb{R}^{d_{ff} \times d}$ in a SwiGLU block with a **static, token-indexed, layer-local embedding table** $\mathbf{U}_\ell \in \mathbb{R}^{V \times d_{ff}}$. Gate and down-projection stay dense.
- **Why**: Fine-grained sparsity (per-token "micro-experts") **without** routing, load-balancing aux losses, or all-to-all comms. Static + token-indexed -> compute path is predictable -> CPU offload with async prefetch is trivial.
- **Wins**: ~3-4% downstream accuracy gain @ 350M and 1B vs dense, large gains on knowledge-heavy tasks (ARC-C, OpenBookQA, MMLU, GSM8K), better long-context (NIAH gap 8.4% -> 13%), interpretable (per-token "address vectors" -> simple knowledge editing by swapping STEM rows), no MoE-style loss spikes.

---

## 1. Motivation & Context

- **Setting**: Sparse scaling laws (Kaplan, Hoffmann) say you can grow parametric capacity without growing per-token FLOPs. MoE realizes this but is gnarly.
- **MoE pain points** (paper enumerates these as motivation):
	- **Training instability** with high expert counts (under-utilized experts, load skew).
	- **Aux losses** for load balancing interfere with main objective.
	- **All-to-all comms** dominate latency as expert count grows; small per-expert payloads -> bandwidth-starved.
	- **Kernel inefficiency** when expert sub-FFNs are tiny.
	- **Not interpretable**: hard to attribute knowledge to individual experts.
- **Desiderata** for fine-grained sparsity:
	- (a) stable optimization, (b) broad expert utilization, (c) negligible routing/comm overhead, (d) interpretability.
- **Static + token-indexed sparsity** (a la Hash Layers, PLE in Gemma 3n) is the right inductive bias:
	- No runtime routing -> CPU offload + prefetch viable, no inter-node all-to-all.
	- Each "expert" tied to a token id -> directly interpretable.
- **Related anchors**:
	- **Hash-layer MoE** (Roller et al. 2021): hash-based static routing — same idea but full FFN per "expert".
	- **MoWE** (dos Santos 2023): word-level experts, suffers Zipfian load imbalance and heavy comms.
	- **PLE / Gemma 3n** (DeepMind 2024): adds *low-dim* token-indexed embedding **alongside** a regular FFN block; STEM dispenses with the up-proj **entirely** and uses **full $d_{ff}$**-dim embeddings.
	- **PKM** (Lample 2019): product-key memory, large KV store with top-k.

---

## 2. Background — Key-Value Memory View of FFNs

- **Geva 2021 / Meng 2022 interpretation**: a 2-projection FFN $\mathbf{y} = \mathbf{W}^d \phi(\mathbf{W}^u \mathbf{x})$ is content-addressable KV memory.
	- **Keys** = rows of $\mathbf{W}^u$, scoring input $\mathbf{x}$.
	- **Values** = columns of $\mathbf{W}^d$.
	- $\mathbf{y} = \sum_{i=1}^{d_{ff}} \phi(\langle \mathbf{k}_i, \mathbf{x} \rangle) \, \mathbf{v}_i$.
- **GLU/SwiGLU enrichment**: gate factorizes addressing into **content** $\mathbf{W}^u \mathbf{x}$ and **gate** $\sigma(\mathbf{W}^g \mathbf{x})$:
	$$\mathbf{y} = \mathbf{W}^{(d)}\big( (\mathbf{W}^u \mathbf{x}) \odot \sigma(\mathbf{W}^g \mathbf{x}) \big)$$
	- Gate provides **query-dependent, per-slot amplification/suppression** -> sharper context-adaptive retrieval.
- **STEM's design choice flows from this**:
	- Up-proj is the "address generator" -> can be replaced by a **direct, token-specific address vector** (the embedding $\mathbf{U}_\ell[t]$).
	- Gate is **context-dependent modulation** -> must stay dense (replacing it kills performance — Table 3 ablation).
	- Down-proj is the value pool -> must stay dense (replacing it would break the residual/feature pathway).

---

## 3. Method

### 3.1 STEM Layer

- Per STEM layer $\ell$, a table $\mathbf{U}_\ell \in \mathbb{R}^{V \times d_{ff}}$ (one row per vocab token, full FFN intermediate width).
- Forward (replacing eq. 1):
	$$\mathbf{y}_\ell = \mathbf{W}_\ell^{(d)}\Big( \mathrm{SiLU}(\mathbf{W}_\ell^{(g)} \mathbf{x}_\ell) \odot \mathbf{U}_\ell[t] \Big)$$
	where $t$ is the **input token id** at that position.
- **Key difference vs PLE**:
	- PLE **complements** an existing FFN with a small (e.g. 256-dim) extra embedding.
	- STEM **replaces** the up-projection and uses the full $d_{ff}$ (e.g. 16384 in Gemma scale).
- **Which projection to replace** (ablation Table 3):
	- Replacing **up-proj** -> consistent gains.
	- Replacing **gate-proj** -> hurts performance (gate must be context-aware; swapping for context-agnostic embedding kills modulation).
	- Hybrid STEM$^\dagger$ (keep up-proj + add embedding additively) -> no improvement, more FLOPs.

### 3.2 Insights / Why It Works

- **Better information-storage capacity** (5.1):
	- Standard FFN up-proj outputs live in a low-intrinsic-dim address space (superposition).
	- STEM embeddings are learned directly as token-specific addresses -> **large angular spread**, low pairwise cosine (Fig 6a: layer 7 cos $\approx 0.026$, layer 13 $\approx 0.013$).
	- Reduced cross-talk between memory slots -> sharper, more disentangled knowledge attribution.
- **Knowledge specificity & interpretability** (5.2):
	- Each STEM row is tied to a token id -> "micro-expert" identity is transparent.
	- Enables **token-indexed, layer-local knowledge editing** without touching input text or auxiliary computation.
- **Context-length adaptive capacity** (3.2.5):
	- Active params per sequence: $\mathrm{Params}^{\mathrm{STEM}}_{\mathrm{act}}(L) = |\mathcal{S}| \, d_{ff} \, L_{\mathrm{uniq}}$.
	- $L_{\mathrm{uniq}}$ grows sublinearly (Heaps' law) -> longer contexts engage more distinct params at ~constant per-token FLOPs.
	- Yields **test-time capacity scaling** -> NIAH gap over dense grows from 8.4% (8k) to 13% (32k).

### 3.3 Efficiency (Table 1)

- **Training/prefill FLOPs per layer** (forward, ignoring elementwise + bias):
	- Dense: $B(3 d \, d_{ff} L + d_{ff} L)$
	- STEM: $B(2 d \, d_{ff} L + d_{ff} L)$
	- **Savings**: $\Delta F = B L d \, d_{ff}$, i.e. **~1/3 of FFN params eliminated** (the up-proj).
	- Saving fraction $= \frac{d_{ff}}{4d + 2L + 3 d_{ff}}$. For Qwen2.5: 21.7% (1.5B) up to 24.8% (32B).
- **Decoding** is memory-bound; parameter loading cost drops by $B d \, d_{ff}$ per step (same fraction as FLOPs).
- **Comms during training**: STEM needs $\mathrm{uniq}(BL) \, d_{ff}$ floats prefetched (and doubled during training to send gradients back to CPU).
- **Key vs MoE**: param traffic scales with **unique tokens**, not expert-selection diversity, so STEM stays sparse even at large batch sizes (MoE quickly activates ~all experts as batch grows).

### 3.4 System: CPU Offload + Async Prefetch

- **Problem**: STEM tables are huge — linear in $V$, $d_{ff}$, and number of STEM layers. Optimizer states + grads compound this.
- **Solution stack**:
	- **CPU offload** of STEM tables -> frees ~1/3 of FFN HBM.
	- **Replicate** tables across nodes -> no cross-node expert traffic.
	- **Async prefetch**: next batch's token ids are known beforehand at prefill -> overlap CPU->GPU embedding transfer with layer compute.
	- **Token deduplication**: a batch of $BL$ tokens has $\mathrm{uniq}(BL) \ll BL$ -> only ship unique embeddings.
	- **LFU cache** on GPU: exploit **Zipfian** token distribution; >80% cache hit rate claimed. Memory saved from removing up-proj funds the cache.
- **Decoding caveat**: autoregressive -> can't prefetch next-token row; must wait. Mitigated by the LFU cache.
- **Training**: decouples backbone parallelism (DDP/FSDP/TP) from STEM-table parallelism; tables are always **sharded across all GPUs** independently. Gradients on offloaded tables must be written back to CPU optimizer state (paper notes optimized training impl is future work).

### 3.5 Knowledge Editing with STEM

- **Setup**: source entity in prompt (e.g. "Spain"), target entity to inject (e.g. "Germany"). Replace source-token STEM rows with target-token STEM rows at every STEM layer; input text unchanged. Model outputs as if it saw the target.
- **Length-mismatch strategies** (Fig 4):
	- $n_s = n_t$ (same subword count): direct 1-to-1 swap.
	- $n_s > n_t$ (source longer): **left-padding** (default, slightly better than right) or **copying** (repeat target tokens $\lfloor n_s/n_t \rfloor$ times).
	- $n_s < n_t$ (target longer): **subset selection** (pick most semantically representative target tokens).
	- **Averaging** (length-agnostic fallback): mean of target-token STEM embeddings, used to overwrite every source position. Surprisingly robust.
- **Generalization**: works beyond countries — Country -> US-State works (United States -> California -> outputs Sacramento facts).
- **Fig 7**: top-4 next-token probabilities for "The capital of Spain is" shift from Madrid -> Berlin after $e_{\text{Spain},\ell} \leftarrow e_{\text{Germany},\ell}$ swap, matching the "The capital of Germany is" control distribution closely.

---

## 4. Experiments

### 4.1 Setup

- **Models**: MobileLLM-350M, Llama 3.2-1B (input embeddings and LM head **not** shared).
- **Pretrain corpus**: OLMo-Mix-1124 (3.9T tokens, sub-sampled to 1T).
- **Mid-training mix**: 65% OLMo-Mix, 5% Nemotron-CC-Math, 30% Nemotron-Pretraining-Code.
- **Context-extension**: ProLong-data-64k, 32,768 max seqlen, cross-doc masking.
- **STEM default**: **1/3 of FFN layers replaced** at uniform intervals.
- **Hash-MoE baseline**: top-1, expert count chosen to match STEM **total** params, activated FLOPs ~= dense.
- **Hyperparams** (Table 2): 350M peak LR 2e-3, 1B peak LR 4e-4 (cosine), batch 512, AdamW, $\beta_1{=}0.9,\beta_2{=}0.95$, wd 0.1.
- **Eval**: ARC-E/C, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande (pretrain); MMLU, GSM8K (midtrain); NIAH (long-context); BBH, MuSR, LongBench (appendix).
- **ROI metric**: $\frac{\text{Avg Acc}}{\text{Total Training FLOPs}}$.

### 4.2 Main Results (Table 3)

- **350M scale**:
	- Baseline (dense, 0.37B): avg 49.72, 0.74 GFLOPs/tok.
	- Hash-MoE (1.22B total / 0.37B active): avg 50.58 (ROI 1.02x).
	- **STEM-1/3** (1.14B / 0.35B): avg **50.90** (1.08x ROI), 0.70 GFLOPs. Big wins: ARC-E 63.01 (vs 57.66), ARC-C 32.68 (vs 30.55), OBQA 33.00.
	- **STEM gate-proj replacement**: avg 49.10 — *worse* than dense -> confirms gate must stay dense.
	- **STEM$^\dagger$** (adds up-proj back additively): avg 50.74 — no improvement, more params.
	- **STEM-1/2** (1.85B / 0.34B): avg **54.20**, ROI **1.20x**.
	- **STEM-full** (replace all FFN layers except first): avg 53.43, ROI **1.33x** (best FLOPs efficiency).
- **1B scale**:
	- Baseline: avg 55.82, 3.0 GFLOPs.
	- **STEM-1/3** (6.75B total / 1.41B active): avg **56.63**, ROI 1.08x.
	- Big gains on **knowledge-intensive tasks**: ARC-C 42.03 (vs 41.88), BoolQ 61.66, OBQA 45.90 (vs 39.84), PIQA 75.00, HellaSwag 60.37 (vs 59.65).
- **Mid-training, 1B** (Table 4): GSM8K 46.4 vs 44.2 baseline; MMLU 32.38 vs 29.92 -> reasoning + knowledge retrieval improve.
- **Training stability** (Fig 5a): Hash-MoE has loss spikes; STEM doesn't, even though both are "extremely sparse". STEM's geometric properties (high angular spread) likely smooth the loss landscape.

### 4.3 Long-Context

- **NIAH** at 8k / 16k / 32k (Fig 1b): STEM advantage **grows** with context — concrete evidence of test-time capacity scaling.
- **LongBench** (App A.2, Table 6): STEM matches or beats dense across all context-length buckets up to 12k+.
- **Contextual reasoning** (App A.1, Table 5): BBH 27.55 (vs 24.87), MuSR 36.38 (vs 35.85), LongBench Multi-hop and Code at all length ranges — STEM doesn't *impair* contextual reasoning, sometimes improves it.

### 4.4 Ablations

- **Layer count** (4.4.1, Fig 5b, Table 3): 1/3 -> 1/2 -> full STEM all improve loss-vs-FLOPs and downstream avg. Returns diminish past 1/2.
- **Placement** (4.4.2): up-proj replacement >> gate-proj replacement (gate must be input-dependent).
- **STEM$^\dagger$** (4.4.3, eq. 5): $\mathbf{y}_\ell = \mathbf{W}^d (\mathrm{SiLU}(\mathbf{W}^g \mathbf{x}) \odot (\mathbf{W}^u \mathbf{x} + \mathbf{U}_\ell[t]))$ — additive modulation on top of dense up-proj. Adds params + FLOPs, no perf gain -> the embedding **alone** is sufficient as the address vector.

---

## 5. STEM Characteristics (Sec 5)

- **Angular spread** (Fig 6):
	- (a) Pairwise cos of STEM embeddings: tight Gaussian near zero. P95 cos $\approx 0.013$-$0.030$ depending on layer -> near-orthogonal.
	- (b) Up-proj output activations have cos centered ~0.2-0.3 (much more clustered) — STEM embeddings are flatter/closer to 90°.
	- (c) After gate modulation, STEM down-proj inputs span a wider range than dense — supports richer feature interaction.
- **Interpretability**: each layer's table indexable by token -> direct knowledge attribution. Editing one row at one layer reliably steers output. Bridges the performance ⇄ interpretability tradeoff (vs SAE/causal-intervention which require extra computation).

---

## 6. Orthogonality Note (Important)

- **STEM is orthogonal to MoE**, not a replacement of the whole MoE pipeline. You can stack: have a Mixture-of-STEM-Experts where each MoE expert's FFN already uses the STEM replacement. STEM targets the **gated FFN module itself**.

---

## 7. Position vs Related Work

| Approach | Routing | Granularity | Comms | Interpretable | Replaces FFN? |
| --- | --- | --- | --- | --- | --- |
| Dense FFN | n/a | n/a | none | no | baseline |
| MoE (Switch, GShard) | learned top-k | full FFN | heavy all-to-all | no | yes |
| Hash-layer MoE | static hash | full FFN | reduced | partial | yes |
| MoWE | static word | small FFN | heavy (many tiny) | yes | yes (but tiny) |
| PKM (Lample) | learned product-key top-k | KV slots | moderate | partial | partial |
| PLE (Gemma 3n) | static token | low-dim ($d_{ple}{=}256$) extra | low | yes | **complement** |
| **STEM** | static token | full $d_{ff}$ embedding | low (CPU prefetch) | **yes** | **replace up-proj only** |

---

## 8. Limitations / Open

- **Memory footprint**: tables scale as $V \cdot d_{ff} \cdot |\mathcal{S}|$ — for Llama-style $V{=}128k$, $d_{ff}{=}14k$, 10 STEM layers -> ~17B params just for tables. Mitigated by CPU offload but requires fast PCIe / NVLink.
- **Optimizer state on CPU during training** is hand-waved as future engineering work; current public training code does **not** include the full async prefetch + LFU cache implementation.
- **Per-token forward path is context-agnostic for the up-proj direction** — STEM$^\dagger$ shows naively adding back the dense up-proj doesn't help, but more nuanced hybrids may.
- **Decode latency** still pays embedding lookup cost; LFU cache hit rate must stay high (>80%) to amortize.
- **Vocab-tied "experts"**: tokens that are rare in training (Zipfian tail) have under-trained STEM rows — same dos Santos / MoWE concern, though STEM mitigates via shared gate/down.

---

## 9. Quick Mental Model

> STEM = **"learned per-token bias vector for the SwiGLU intermediate activation, one vector per (layer, vocab-token), looked up from a CPU-resident table"**.
> Equivalent to having $V$ "experts" per layer, each consisting of a single $d_{ff}$-dim vector, selected by the **current input token id** with no router and no aux loss.

---

## Implementation

Repo: `/home/jayoo/code/jayoo_personal/LLM_Knowledge_Base/raw_data/papers/STEM/STEM_repo/` — built on **Meta Lingua**.

> Note: the public training code implements the **architecture** (FFN up-proj -> token-indexed embedding) and a **distributed sharded embedding** with all-to-all comms, but **not** the paper's CPU-offload + async-prefetch + LFU cache. README & paper explicitly defer the fully-optimized impl ("we shall provide a fully optimized training implementation in our next iteration").

### Core architecture

**`lingua/stem.py`** — STEM transformer block + replacement FFN.

- `StemTransformerArgs` (`stem.py:14-17`): adds `stem_layers: List[int]` (which layer indices get STEM) and `stem_embedding_dim: int`.
- `FeedForward` (`stem.py:20-82`): standard SwiGLU with `w1` (up), `w3` (gate), `w2` (down). For non-STEM layers.
- **`StemFeedForward`** (`stem.py:85-139`): the modified FFN — only **`w1`** (gate equivalent) and **`w2`** (down). `w3` (up-proj) is gone, replaced by the externally-supplied `y`:
	```python
	def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	    x1 = self.w1(x.view_as(x))
	    output = self.w2(F.silu(x1) * y)   # y = U_l[t], shape [B, S, hidden_dim]
	    return output
	```
	- Note: the code uses SwiGLU as $\mathrm{SiLU}(\mathbf{W}^g \mathbf{x}) \odot \mathbf{U}_\ell[t]$, matching paper eq. 4. (The variable names `w1`/`w2`/`w3` are Lingua's; `w1` here plays the role of gate.)
- `StemTransformerBlock.__init__` (`stem.py:142-171`): picks `StemFeedForward` iff `layer_idx in args.stem_layers` (line 161); asserts `stem_embedding_dim == feed_forward.hidden_dim` (line 170) — embedding row width must equal $d_{ff}$.
- `StemTransformerBlock.forward` (`stem.py:174-191`): threads the embedding `y` through to FFN: `out = h + self.feed_forward(self.ffn_norm(h), y)`.

### Top-level LM wrapper

**`apps/main/stem.py`** — decouples the FSDP-wrappable backbone from the manually-managed STEM tables.

- `LMTransformer.forward` (`apps/main/stem.py:54-102`): takes a `stem_embeddings_fn: (layer_idx, token_values) -> Tensor` callback. For each layer:
	```python
	for i, layer in enumerate(self.layers):
	    if i in self.stem_layers:
	        y = stem_embeddings_fn(i, token_values)   # per-layer embedding lookup
	        h = layer(h, freq_cis, y=y, ...)
	    else:
	        h = layer(h, freq_cis, ...)
	```
- **`StemLMTransformer`** (`apps/main/stem.py:125-243`): wraps `LMTransformer` + a `ModuleList` of `ParallelEmbedding`s — one per STEM layer (`apps/main/stem.py:142-145`).
	- `_layer_to_stem_idx` maps STEM layer index to position in the embedding list (line 148-151).
	- The closure `stem_embeddings_fn` (line 164-167) calls the right `ParallelEmbedding` for each layer.
- `build_stem_lm_fsdp_grouping_plan` (`apps/main/stem.py:246-254`): prefixes Lingua's normal FSDP plan with `lm_transformer.` so **only the backbone** is FSDP-wrapped; STEM tables stay outside FSDP and use a custom parallel scheme.

### Distributed embedding tables (the system piece in the repo)

**`lingua/stem_dist_utils.py`** — STEM uses **its own process group** (orthogonal to DP/FSDP/TP).

- `initialize_stem_process_group` (`stem_dist_utils.py:25-69`): builds `model_parallel_size`-wide STEM groups via 2D world reshape.
- **`ParallelEmbedding`** (`stem_dist_utils.py:429-523`): partitions the embedding **along the hidden dim** (`embedding_dim_per_partition = embedding_dim / stem_world_size`, line 459). Each rank owns its slice of $\mathbf{U}_\ell$.
- Forward (`stem_dist_utils.py:471-485`):
	```python
	def forward(self, input_):
	    input_parallel = gather_tokens_for_stem(input_)        # all-gather token ids across STEM ranks
	    output_parallel = F.embedding(input_parallel, self.weight, ...)
	    output = _AllToAllForStem.apply(output_parallel)        # batch <-> hidden swap
	    return output
	```
- **Custom autograd functions** (`stem_dist_utils.py:259-333`):
	- `_GatherTokensForStem` — fwd: all-gather token ids; bwd: reduce-scatter grads.
	- `_AllToAllForStem` — fwd: all-to-all scatter on batch, gather on hidden; bwd: inverse all-to-all, rescaled.
	- `_GatherEmbeddingsForStem`, `_ScatterEmbeddingsFromStem` — alternative comm primitives (commented-out alternative path on line 482).
- Low-level primitives: `_gather_along_first_dim_stem`, `_split_along_last_dim_stem`, `_reduce_scatter_along_first_dim_stem`, `_all_to_all_stem` (`stem_dist_utils.py:95-254`).

### Training loop integration

**`apps/main/stem_train.py`** — extends Lingua's `train.py` with STEM-specific handling.

- **Two optimizers** (`stem_train.py:202-233`): backbone params get a **fused AdamW** over DTensors (FSDP-sharded); STEM embeddings get a **non-fused AdamW** over regular Tensors:
	```python
	lm_optimizer = AdamW(model.lm_transformer.parameters(), ..., fused=True)
	stem_optimizer = AdamW(model.stem_embeddings.parameters(), ..., fused=False)
	optimizer = {"lm": lm_optimizer, "stem": stem_optimizer}
	```
- **STEM process group is initialized BEFORE model creation** (`stem_train.py:120`): `initialize_stem_process_group(args.distributed.stem_parallel_size)`.
- Backbone is created on meta device then wrapped with FSDP via `parallelize_model(...)` excluding STEM tables (`stem_train.py:133-141`).
- Separate **gradient clipping** (`stem_train.py:386-422`): DTensor path for backbone (`grad_norm.full_tensor()`), regular path for STEM. Also defensive checks for "all-zero grad" / "no grad" on STEM params — explicitly flagged as a debugging concern.
- Both optimizers stepped together (`stem_train.py:424-429`).
- **`StemCheckpointManager`** used (`stem_train.py:247`), defined in `lingua/stem_checkpoint.py`, which separately consolidates `ParallelEmbedding` shards (`stem_checkpoint.py:215-260` and `consolidate_stem_shards` function -> writes `consolidated_stem.pth`).

### Generation

**`apps/main/stem_generate.py`** — inference entrypoint.

- `load_consolidated_model_and_tokenizer` (`stem_generate.py:44-80`): loads backbone from `consolidated.pth`, then iterates `nn.Embedding` modules and copies STEM weights from `consolidated_stem.pth` (lines 67-73). Model is moved fully to CUDA — **no CPU offload in the public inference path** either.
- Reuses Lingua's `PackedCausalTransformerGenerator` (`apps/main/generate.py`) unchanged.

### What's missing vs the paper

- No **async CPU<->GPU prefetch** stream, no **LFU cache**, no **token deduplication** before embedding lookup. The forward path just does `F.embedding(input_parallel, self.weight, ...)` on GPU resident weights.
- No **CPU optimizer offload** for STEM tables during training (the entire `ParallelEmbedding.weight` lives on GPU; just sharded across the STEM group).
- The system parallelism implemented is essentially **TP-along-hidden** for the embedding (an `all-to-all` swap to redistribute the batch back to local DP) — useful but distinct from what the paper's Sec 3.4 describes for production.

### Example config

**`apps/main/configs/stem_debug.yaml`**:

```yaml
stem_parallel_size: 4
distributed:
    stem_parallel_size: 4
model:
    dim: 1024
    n_layers: 8
    n_heads: 8
    stem_layers: [2, 4, 6]    # 3 of 8 layers get STEM, uniformly placed
```

Matches the paper's "1/3 of FFN layers replaced at uniform intervals" default.
