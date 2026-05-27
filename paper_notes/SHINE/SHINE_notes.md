# SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass

> Liu, Wang, Mao, Gelberg, Maron, Zhang — **ICML 2026**
> Code: https://github.com/MuLabPKU/SHINE

---

## TL;DR

- **Problem**: Adapt an LLM to a specific *context* (book, document, conversation, rule set) without (a) putting context in the prompt (latency/window cost) or (b) doing per-context fine-tuning (training cost, storage).
- **Method**: **Hypernetwork** that maps a natural-language **context** -> a **LoRA adapter** for a frozen LLM in **one forward pass**. The hypernetwork *reuses* the frozen LLM itself as the encoder (in-context design), then adds a tiny Transformer (called *M2P = memory-to-parameter*) to convert memory states across layers/tokens into LoRA weights.
- **Training**: Two-stage meta-training pipeline — **pretraining** (reconstruction + completion on 6B tokens) + **instruction fine-tuning** (mqa + 1qa QA datasets).
- **Key win**: Strong context-QA results that approach In-Context with full prompt, beat SFT/TTT by a wide margin, with ~0.3s amortizable LoRA generation and zero extra inference latency.
- **Why it's interesting**: Architectural novelty — no bottleneck (memory token width = LoRA width), uses *bidirectional row+column attention* across `(layer, mem_token)` axes (mimics backprop information flow), and uses **post-layernorm** in the M2P Transformer (unusual choice, motivated below).

---

# 1. Background & Motivation

## 1.1 Three paradigms for adapting LLMs to a context

| Paradigm | Pros | Cons |
|----------|------|------|
| **Prompt engineering / ICL** | No training; uses all context info | Inflated latency, context-window pressure, FLOPs scale with prompt len |
| **Per-context fine-tuning (SFT/TTT)** | Knowledge baked into weights | Need GPU time per context, hyperparams sensitive, store one model per task |
| **Hypernetwork-generated LoRA (this work)** | Single forward pass, no per-context training, no prompt overhead at inference | Hypernetwork itself must be meta-trained; design is hard |

## 1.2 Why prior hypernetworks have struggled

Three structural challenges authors call out:
- **Semantic-to-parameter alignment**: Bridging text-space to weight-space is hard. Existing methods need an external encoder.
- **High-dimensional output**: Even LoRA weights for all layers of a 7B LLM is a *lot* of params; naive MLP heads are infeasible.
- **Efficiency**: Must be cheap or it loses to SFT/ICL.

Prior hypernetwork compromises (criticized):
- **Generative Adapter (Chen 2025)**: Only generates LoRA for *some* layers (e.g. only output projection) -> limited expressivity.
- **ICAE / Compositional Adapter (Jukic 2025)**: Tight bottleneck through small MLP / few gist tokens.
- **Text-to-LoRA (Charakorn 2025)**: Maps *task descriptions* (not arbitrary context) -> LoRA via MLP reused per layer. Easier task, limited.
- **Drag-and-Drop LLMs**: Convolutional decoder + recurrent diffusion, prompt -> LoRA, similar bottleneck issue.

> SHINE's design philosophy: **no bottleneck** (memory state width = LoRA output dim), **in-context** (reuse frozen LLM as the context encoder), **lightweight transformer** for the M2P map (not wide MLP).

## 1.3 Related: Test-Time Training (TTT)

- TTT does online weight updates at inference via gradient descent over the context (SEAL, PaST, etc.).
- SHINE achieves the *same goal* (update parameters at inference) but by **one forward pass** through the hypernetwork instead of iterative optimization.
- SHINE beats TTT methods (SEAL 58.2, PaST 58.9 vs **SHINE 63.6** on SQuAD with n=200) while being orders of magnitude faster.

---

# 2. Architecture

## 2.1 High-level pipeline

Two passes:
1. **Memory Extraction**: Context `c` is fed through the frozen LLM **augmented with a *Meta LoRA*** plus learnable **memory embeddings** appended after context tokens. Hidden states of the memory positions at every layer become **memory states** `M ∈ R^{L × M × H}`.
2. **Parameter Generation (M2P Transformer)**: `M` is processed by a tiny Transformer to produce a tensor that is **reshaped/sliced into per-layer LoRA matrices** for the target LLM.
3. **Downstream Task**: LLM + generated LoRA answers user question, *without seeing the context*.

Data flow:
```
Context  --[ LLM + Meta LoRA ]-->  Memory States
Memory States  --[ M2P Transformer ]-->  Generated LoRA
Question  --[ LLM + Generated LoRA ]-->  Answer
```

## 2.2 Memory extraction

- Input embeddings `X ∈ R^{N × H}` + learnable memory embeddings `M_0 ∈ R^{M × H}`, concat → run through LLM.
- At each layer `i`, collect the last `M` tokens' hidden state into `M_i ∈ R^{M × H}`.
- Stack across layers: `M = Stack(M_1, ..., M_L) ∈ R^{L × M × H}`.
- **Meta LoRA** is a learnable LoRA dictionary applied during memory extraction — provides the LLM extra task-specific capacity for "compression." It is trained jointly with the hypernetwork.
- **Memory length chosen so that** `M = ⌈ r·D / H ⌉` where `r` = generated LoRA rank, `D` = sum of input+output dims of all linear layers in *one* LLM layer. Guarantees memory tensor has ≥ as many scalars as the target LoRA → **no information bottleneck.**
  - In the paper: `r = 8`, Meta LoRA rank `= 128`, `M = 148`, hidden `H = 4096`, `L = 36` (Qwen3-8B), 4-layer M2P Transformer.

## 2.3 M2P Transformer (Memory-to-Parameter)

Three steps:

### Step 1 — Positional encoding
Learnable per-layer and per-token PEs:
- $\mathbf{P}^{\text{layer}} \in \mathbb{R}^{L \times 1 \times H}$
- $\mathbf{P}^{\text{token}} \in \mathbb{R}^{1 \times M \times H}$
- $\tilde{M} = M + P^{\text{layer}} + P^{\text{token}}$ (broadcast)

### Step 2 — Sparse alternating attention
Full attention over all `L · M` tokens would cost $O((LM)^2)$ — too much. Instead, **alternate row and column attention**:
- **Odd layers** = *column attention*: each column attends across layers (length `L`).
  - $Y^{(i)}_{:,j} = \text{SelfAttn}(Z^{(i)}_{:,j}), \quad j=1,\dots,M$
- **Even layers** = *row attention*: each row attends across mem tokens (length `M`).
  - $Y^{(i)}_{j} = \text{SelfAttn}(Z^{(i)}_{j}), \quad j=1,\dots,L$
- Followed by per-token 2-layer MLP.
- **Saves up to 90% FLOPs** vs full attention.

> **Why this design is clever**: It mimics *backprop* — parameter updates in shallow layers depend on deeper layers' updates. The bidirectional row/column flow allows deep-layer memory to inform shallow-layer LoRA generation, which prior unidirectional designs (e.g. Generative Adapter only uses ith-layer hidden for ith-layer LoRA) cannot.

### Step 3 — Reshape to LoRA
- Flatten the i-th slice of the final memory tensor `M̄[i, :, :] ∈ R^{M·H}` into a long vector `v`.
- Sequentially slice & reshape into LoRA `A ∈ R^{I × r}` and `B ∈ R^{r × O}` for every linear module in layer `i`.
- 4 different reshape modes (`rl`, `rr`, `lr`, `ll`) studied — `rl` and `lr` converge much faster than `rr`/`ll` (the latter are essentially outer products of two memory embeddings — harder to optimize).
- Default in code: `rl`.

## 2.4 Architectural details (Appendix A)

- **Post-layernorm in M2P Transformer** (atypical — modern LLMs use pre-LN). Justification:
  - The hypernet is shallow (4 layers).
  - It maps *hidden states* -> *weight space* with huge distribution gap between them; identity shortcut hurts.
  - Pre-LN was unstable / didn't converge as well.
- **Architecture covers all LoRA-eligible linears**: Q, K, V, O in attention + gate, up, down in SwiGLU MLP. (See codebase.)

## 2.5 Advantages vs prior

| Method | Gen all layers? | Bottleneck? | Trans-like (global attn)? |
|--------|-----------------|-------------|----------------------------|
| Generative Adapter | No (limited subset) | Implicit | No (outer-product additive) |
| ICAE / Compositional Adapter | Yes | Yes (gist tokens / small `r`) | No |
| Text-to-LoRA | Yes | MLP bottleneck | No |
| Drag-and-Drop | Yes | Conv decoder, smaller | No |
| **SHINE** | **Yes** | **None (M·H ≥ rD)** | **Yes** |

Parameter count: $P_{\text{ours}} \approx \left(\tfrac{L'}{L} + \tfrac{2r}{H}\right) P$ with `L'=4` hypernet layers, `r=128` Meta LoRA, on Qwen3-8B `P_{\text{ours}} ≈ 17% of base`.

---

# 3. Training Pipeline

Trainable parameters:
- $\Theta_T$: M2P Transformer params.
- $\Theta_M$: Meta LoRA params (used during memory extraction).
- $m$: Initial memory embeddings.

Frozen: $\Theta_{\text{LLM}}$ (Qwen3-8B base).

Generated LoRA: $\Theta_{\text{GLoRA}} = f(c; \Theta_T, \Theta_M, m)$.

## 3.1 Pretraining (self-supervised, 6B tokens)

Dataset: `TransMLA pretrain 6B`. Two co-trained objectives, mixed 50/50:

### Reconstruction
- Hypernet converts context `c` to `Θ_GLoRA`.
- LLM with `Θ_GLoRA` is prompted with `<RECON>` and must output original `c`.
- $\mathcal{J}_{\text{RECON}} = -\log P(c \mid \text{<RECON>}; \Theta_{\text{GLoRA}}, \Theta_{\text{LLM}})$
- Ensures the LoRA *fully memorizes* the context.

### Completion
- Truncate last `k ∈ [0.1N, 0.3N]` tokens of `c` → `c'`.
- Hypernet processes `c'` only; LLM with resulting LoRA must reconstruct *full* `c`.
- $\mathcal{J}_{\text{COMP}} = -\log P(c \mid \text{<COMP>}; \Theta_{\text{GLoRA}}', \Theta_{\text{LLM}})$
- Prevents pure memorization → forces generalization (LLM must "infer" missing tokens from the LoRA encoding).

Loss: $\mathcal{J}_{\text{TOTAL}} = \lambda \mathcal{J}_{\text{RECON}} + (1-\lambda)\mathcal{J}_{\text{COMP}}$, $\lambda = 0.5$.

**Short-context concatenation trick** (Appendix B.1): To efficiently fill 1150-token sequence budget, multiple short contexts `A, B, C` are concatenated with `<EOT>` separators. Each is independently assigned `<RECON>` or `<COMP>`. Random permutations of conversation order are used to force *any-position* attention from the generated LoRA.

## 3.2 Instruction Fine-Tuning (IFT)

Two stages:

### Stage A — MQA (multiple QAs per context)
- Authors construct **MS MARCO MQA**: 366K training examples, 15 QA pairs per context (10 specific + 5 general questions), Qwen-Flash–generated, fact-checked by re-prompting Qwen-Flash.
- Hypernet generates LoRA for `c`; LLM+LoRA answers `q` -> ground-truth `a`. Cross-entropy on answer tokens only.
- $\mathcal{J}_{\text{IFT}} = -\log P(a \mid q; \Theta_{\text{GLoRA}}, \Theta_{\text{LLM}})$
- Other MQA sources: QuAC, COQA, DROP, NarrativeQA, QuAIL, PwC, SQuAD, RACE, NewsQA, DuoRC.
- 2 epochs at LR 3e-5.

### Stage B — 1QA (one QA per context)
- Single-pair datasets: MS MARCO, SQuAD, RACE, DuoRC, HotpotQA, MuSiQue, 2WikiMultihopQA.
- 1 epoch at LR 1e-5.
- Smaller LR to avoid catastrophic forgetting of MQA capabilities.

---

# 4. Experiments

**Backbone**: Qwen3-8B. **Generated LoRA rank** `r = 8`, **Meta LoRA rank** `r_M = 128`, `M = 148` mem tokens, 4-layer M2P Transformer. **Hardware**: 8× A100. Optimizer AdamW + linear schedule + warmup.

## 4.1 Pretraining results (Fig 5)
- Recon PPL ~1 for context lengths > 200 tokens — perfect reconstruction.
- Comp PPL ~1.5-1.7 across lengths — strong completion.
- Minor outlier dip at ~100-token contexts (attributed to noisy short examples).

## 4.2 Multi-turn QA (MS MARCO MQA, 15 QA pairs, Fig 6)

| Method | Avg F1 |
|--------|--------|
| Naive (no context) | 23.2 |
| **In-Context (golden baseline)** | **69.4** |
| SFT (rank-8 LoRA, 10 epochs / 5 conv per context) | 33.0 |
| **SHINE** | **55.6** |

- SHINE crushes SFT (per-context LoRA) by 22 F1 points.
- ICL still wins but at 14.2 s/query vs SHINE at 11.0 s/query (plus tiny 0.3 s amortized LoRA generation).
- SHINE degrades on later conversation turns (no long-context post-training; conversation history accumulates).

## 4.3 1QA performance (Table 2, answer F1)

| Method | SQuAD | MS-MARCO v1 | v2 | HotpotQA | MuSiQue | 2Wiki | SQuAD-512 | -1k | -2k |
|--------|-------|------|------|----------|---------|------|-------|---|---|
| Naive | 22.0 | 19.6 | 16.0 | 26.9 | 11.8 | 27.8 | 22.0 | 22.0 | 22.0 |
| In-Context | 86.8 | 34.2 | 31.3 | 68.7 | 36.3 | 48.7 | 85.9 | 85.1 | 84.9 |
| Gen Adapter (baseline hypernet) | 70.3 | 35.0 | 27.9 | 40.8 | 19.4 | 32.9 | 48.8 | 43.0 | 39.9 |
| **SHINE** | 63.6 | **40.7** | **40.1** | 59.0 | 28.5 | **60.2** | 53.4 | 44.5 | 37.5 |

- **SHINE > Generative Adapter** on most.
- **SHINE > In-Context** on noisy 1QA datasets like MS-MARCO v1/v2 and 2Wiki — context absorption helps the LLM filter noise.
- Multi-hop reasoning (HotpotQA, MuSiQue, 2Wiki) is strong w/o explicit CoT — the LoRA encodes *structured* knowledge, not just rote memorization.
- Ablation w/ half-data variants (`SHINE(1/2 MQA)`, `SHINE(1/2 1QA)`) confirms both data sources matter.

## 4.4 Test-Time Training comparison (SQuAD, n=200, Table 3)

| Method | Score | Notes |
|--------|-------|-------|
| Base | 32.7 | |
| Train-on-passage | 36.0 | |
| Train-on-passage + Synthetic | 50.6 | |
| Train-on-passage + GPT-4.1 Synth | 59.4 | uses GPT-4.1 generated data |
| SEAL | 58.2 | |
| PaST 50 | 58.9 | best TTT |
| **SHINE (one forward pass)** | **63.6** | **+4.7 over best TTT** |

- And SHINE has **zero training overhead at test time**.

## 4.5 Scalability (Sec 5.4 + Table 4)

- Pretraining loss decreases steadily with no saturation across:
  - Backbone LLM scale (Qwen3-0.6B / 1.7B / 8B).
  - M2P Transformer layers (4 → 6).
  - Meta LoRA rank (4 → 8 → 32 → 128).
  - Generated LoRA rank (4 → 8).
- Best config in ablation: 4 M2P layers, Meta-LoRA 128, GLoRA 8 → **PPL 1.32** (vs 1.88 for tiny config).
- "No signs of capacity bottleneck" — claim is that scaling continues to help.

## 4.6 Long Context: SHINE-R (recurrent)

- Splits long context into chunks, processes sequentially. For each chunk, M2P generates **two LoRAs**: one for the M2P Transformer itself (used in next chunk to update encoding), one saved.
- All saved LoRAs are concatenated to form final adapter.
- Theoretical infinite context handling, linear LoRA memory growth.
- Continual pretrained + finetuned on up to 18K tokens. Results on LongBench (Table 5):
  - **SHINE-R** outperforms Naive everywhere, trails In-Context, e.g. on 2WikiMQA: Naive 26.7 / SHINE-R 33.1 / ICL 45.5.

---

# 5. Ablations (Appendix E)

## 5.1 "Bitter Lesson" — coupled cross-attention layers (E.1)
- Tried adding *coupled cross-attention* layers that *precompute* which mem tokens map to which LoRA A vs B matrix, so A tokens attend only to corresponding B tokens.
- Initially converges faster, but full bidirectional attention wins long-term.
- Conclusion: **don't hand-design structural priors** — let attention learn it.

## 5.2 M2P Transformer architecture (E.2, Fig 14)
Compared variants:
- Linear-small (2-layer linear) — fails catastrophically.
- Linear / Linear+Residual (8-layer, same params as Transformer) — much worse.
- **Only-Last-Memory** (use only last layer's memory states, replicated) — worse than aggregating from all layers.
- **Only-Horizontal** (single-axis attention, e.g. 4 token-only layers) — comparable but slower early.
- **Origin (alternating L/M attention)** — best, fastest convergence.

→ Confirms: (i) need Transformer, (ii) need multi-layer memory, (iii) need alternating row+column attention.

## 5.3 SFT alternatives (E.3)
- LoRA SFT, DoRA, IA3 all need >>1 sample to be competitive. SHINE F1=55.6 at 0.3s.
- E.g. LoRA needs `num_sample=10, num_epoch=20` and 187s to hit 58.5 F1.

## 5.4 Text quality vs F1 (E.4, Fig 15-16)
- **Higher-quality (lower PPL on base LLM) context** → better SHINE F1.
- **Shorter context** → better SHINE F1.
- Both relationships are mild but present — suggests improved encoding is mainly bottlenecked by *compressibility* of context.

## 5.5 Multitask (E.5, Table 11)
After MQA IFT, evaluate transferability:
- DailyDialog: SHINE 17.12 vs ICL 12.28 — better than ICL!
- SAMSum (summarization): SHINE 46.18 vs ICL 34.01.
- Persona-Chat: SHINE 16.07 vs ICL 9.32.
- GSM8K (reasoning): SHINE 25.27 vs ICL 44.48 — *worse*, expected since no reasoning data in IFT.

---

# 6. Efficiency Analysis (Table 1, Fig 7, App B.5-B.6)

| Method | F1 | Amortizable time (s) | Per-query gen time (s) |
|--------|-----|----|-----|
| Naive | 23.2 | 0.0 | 11.0 |
| In-Context | 69.4 | 0.0 | 14.2 |
| SFT | 33.0 | **29.3** (per-context FT) | 11.0 |
| **SHINE** | 55.6 | **0.3** | 11.0 |

- **Amortizable FLOPs**: SFT and ICL not amortizable; SHINE one-time LoRA-gen cost (memory extraction + M2P) << SFT.
- **Generation FLOPs**: Identical to Naive/SFT (no extra tokens). In-Context pays the `C` extra context-token cost (huge as `C` grows).
- **Peak memory**: SHINE has cost equivalent to Naive (`O(LNH)` activations or `O(LNH)` KV cache). In-Context's KV cache scales with `N = I + C`.

---

# 7. Limitations / Open Threads

- **Long-context still trails ICL**: Even SHINE-R doesn't fully close the gap (e.g., 2WikiMQA: 33 vs 45 F1).
- **Multi-turn degradation**: Conversation history isn't absorbed into LoRA — could fix with a richer recurrent design or by re-generating LoRA per turn.
- **Reasoning data absent**: SHINE doesn't help GSM8K. Would need a reasoning-augmented IFT stage (RL/distillation).
- **Architecture-coupled**: LoRA generation is per-LLM; need to retrain hypernet for different backbones.
- **Test-time costs**: While SHINE-R helps long context, it still doesn't match TTT's "online learning" generality (no test-time gradient at all).

---

---

# Codebase Implementation Notes

## Repo Structure

```
SHINE/
├── meta_train_parallel.py      # Main training entrypoint (DDP)
├── metanetwork_family.py       # MetanetworkTransformer + Metanetwork wrapper
├── LoraQwen.py                 # LoRA-augmented Qwen3 (LoraLinear + LoraQwen3*)
├── inference.ipynb             # Demo / interactive eval
├── configs/
│   └── Qwen3-{0.6B,1.7B,8B}.yaml  # Hydra configs
├── scripts/                    # Bash launch scripts (pretrain, IFT, test)
├── utils/
│   ├── mydataset.py            # Datasets + Collators (PretrainCollator, IFTCollator, etc.)
│   ├── myddp.py                # DDP helpers
│   ├── myloradict.py           # iter_learnable_tensors, merge_loradicts, freeze_loradict
│   ├── myoptmize.py            # init_optimize
│   ├── mysaveload.py           # save/load checkpoints + training state
│   └── ...
├── evaluation/                 # Test scripts
├── generate_data/              # MQA dataset generation (Qwen-Flash prompts)
├── test.py / test_pwc.py / test_pretrain.py
└── generate_group_idx.py       # Precomputes packing-friendly group indices for pretrain
```

## Key modules

### `LoraQwen.py` — LoRA-wrapped Qwen3

- **`LoraLinear(nn.Linear)`**: Drop-in linear that accepts an optional `lora_dict={"A","B","C"}` per forward call.
  - Forward (snippet):
    ```python
    A = lora_dict["A"]   # [Lb, in, r]
    B = lora_dict["B"]   # [Lb, r, out]
    C = lora_dict.get("C")  # [Lb, out] or None
    x = input.reshape(Lb, num_beams, -1, in_features)
    tmp     = torch.matmul(x, A[:, None, :, :])      # [Lb, beams, S, r]
    lora_out = torch.matmul(tmp, B[:, None, :, :])    # [Lb, beams, S, out]
    return base + lora_out (+ C broadcast if bias)
    ```
  - Supports `num_beams` (multiple samples sharing one LoRA — handy for batched generation).
  - `lora_params_numel(r)` = `in*r + out*r + (out if bias)`.
  - `set_generate_func(method)`: closure that selects one of `rl/rr/lr/ll` reshape modes. **`rl` is default** (used in paper).
  - `divide_idx(r, idx_start)`: returns offsets so the hypernetwork can carve a flat plain tensor into proper A/B slices.

- **`LoraQwen3MLP`**: SwiGLU MLP with all three `gate_proj`, `up_proj`, `down_proj` replaced by `LoraLinear`. Adapter dict: `{"gate","up","down"}`.

- **`LoraQwen3Attention`**: Q/K/V/O are all `LoraLinear`. Adapter dict: `{"q","k","v","o"}`.

- **`LoraQwen3DecoderLayer`**: combines attention + MLP. Adapter dict: `{"attention": {...}, "mlp": {...}}`.

- **`LoraQwen3Model`**:
  - Adds `mem_tokens` as `nn.Parameter` of shape `(num_mem_token, hidden_size)` when `config.num_mem_token != -1`.
  - In forward, **appends memory embeddings to input** (concat after context tokens):
    ```python
    inputs_embeds = torch.concat((inputs_embeds, self.mem_tokens.unsqueeze(0).repeat(B, 1, 1)), dim=-2)
    ```
  - During context encoding loop, records hidden state at memory-token positions in each layer:
    ```python
    memory_states[:, i, :, :] = hidden_states[:, -self.num_mem_token:]
    ```
  - Strips memory tokens before final `norm`.
  - `ignore_mem_token=True` disables this (used when running the *generation* pass with the produced LoRA — no memory needed).
  - Custom output dataclass: `MemoryModelOutputWithPast` carries `memory_states` (`[B, L, M, H]`).

- **`LoraQwen3ForCausalLM`**: standard CausalLM wrapper, plumbs `loradict`, `ignore_mem_token`, `use_gradient_checkpoint` through.

### `metanetwork_family.py` — Hypernetwork

- **`MetanetworkTransformer(nn.Module)`** = the M2P Transformer:
  ```python
  self.layer_pe   = nn.Parameter(zeros(L, H))
  self.token_pe   = nn.Parameter(zeros(M, H))
  self.transformer_layers = nn.ModuleList([
      nn.TransformerEncoderLayer(d_model=H, nhead=32, dim_ff=2H, ...) for _ in range(num_layers)])
  self.couple_layers = ...  # optional, paper sets couple_num_layers=0
  ```
  - **Forward** alternates token-axis and layer-axis attention:
    ```python
    for i, layer in enumerate(self.transformer_layers):
        if (i%2==0) == self.layer_transformer_first:
            # column attention: across layers, per-token
            memory_states = layer(
                memory_states.transpose(1,2).flatten(0,1)  # [B*M, L, H]
            ).unflatten(0, (B, M)).transpose(1,2)
        else:
            # row attention: across tokens, per-layer
            memory_states = layer(
                memory_states.flatten(0,1)                  # [B*L, M, H]
            ).unflatten(0, (B, L))
    ```
  - Optional **mean-pool** (`mean_pool_size`, set to 1 → no-op) and **couple-attention layers** with structural masks (ablated, not used in headline results).
  - Final reshape: `view(batch_size, -1)` → flat tensor of LoRA params.

- **`Metanetwork(nn.Module)`** = composes everything:
  ```python
  self.metamodel = metamodel                     # frozen LoRA-Qwen3
  self.metanetwork = MetanetworkTransformer(...) # the M2P
  self.lora_r = cfg.model.lora_r                 # generated LoRA rank, e.g. 8
  ```
  - **`generate_lora_dict(evidence_ids, evidence_attention_mask, metalora)`**:
    1. Forward `metamodel` on context with `loradict=metalora` to get `memory_states`.
    2. Pass `memory_states` through `MetanetworkTransformer` → `plain_output` (flat tensor).
    3. `metamodel.generate_lora_dict(r, scale, plain_tensor)` carves it into nested LoRA dict (per layer / per linear).
  - **`forward(...)`** for training:
    ```python
    loradict, plain_output = self.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora, ...)
    outputs = self.metamodel(input_ids, attention_mask, loradict=loradict,
                             labels=labels, ignore_mem_token=True, ...)
    outputs.reg_loss = adapter_reg * |plain_output|.sum()
    return outputs
    ```
    - **Two distinct LLM passes per step**:
      1. Encoding pass over `evidence_ids` (context) WITH meta-LoRA AND memory tokens (`ignore_mem_token=False`).
      2. Answering pass over `input_ids` (conversation) WITH generated LoRA, NO memory tokens (`ignore_mem_token=True`).
  - The entire `Metanetwork.forward` is wrapped in `@torch.compile` for speed.
  - `adapter_reg` = optional L1 regularization on the produced flat LoRA tensor (set to 0 in default config).

### Training loop (`meta_train_parallel.py`)

- **Hydra config** in `configs/Qwen3-8B.yaml`, overridable from bash scripts.
- **DDP** via `torchrun`, 8 GPUs by default.
- **Special tokens added**: `<RECON>`, `<COMP>`, `<NOTHING>` — appended to Qwen3 tokenizer.
- **Model loading**:
  ```python
  metamodel = LoraQwen3ForCausalLM.from_pretrained(...)
  metamodel.reset_mem_tokens()  # zero-init
  metamodel.resize_token_embeddings(...)
  metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
  freeze(metamodel)  # freezes backbone (only mem_tokens + LoRA dicts + metanet train)
  ```
- **num_mem_token computation**: 
  ```python
  config.num_mem_token = lora_params_numel(r) * mean_pool_size / (H * L)
  ```
  i.e. `M = rD/H` (matches paper Eq 4, since `D = lora_params_numel(r)/L/r`).
- **Three modes** for boot strapping checkpoints:
  - `pretrain` → train from scratch on `grouptransmla` 6B tokens.
  - `iftpwc` → load latest `pretrain` checkpoint, train on `ift-pwc` (mqa).
  - `train` → load latest `iftpwc` checkpoint, train on `ift-c1qa` (1qa).
  - For `train`, optionally **freeze the pretrained meta-LoRA** and add a second `ift_additional_metalora` for finer specialization (merged via `merge_loradicts`).
- **Optimizer groups** (3 groups):
  1. Metanet (non-LayerNorm) params with weight decay.
  2. Metanet LayerNorm/bias params without decay.
  3. The LoRA dict tensors themselves (treated as Parameters with `iter_learnable_tensors`).
- **Loss path** (per step):
  ```python
  outputs = ddp_metanet(input_ids, input_attention_mask, evidence_ids, evidence_attention_mask, labels, metalora=cur_metalora, ...)
  loss = outputs.loss + outputs.reg_loss  # CE on conversation + L1 reg on plain LoRA tensor
  loss.backward()
  ```
- **AMP**: bfloat16 autocast on CUDA.
- **Gradient accumulation**: 4 by default in scripts.
- **Gradient clipping**: per-group clip at norm 1.0.
- **Checkpointing**: every `save_steps` (1250 default for pretrain, 625 for IFT) + per-epoch. Saves metanet + metalora dict + optional ift_additional_metalora.

### Data flow specifics

- **Pretraining (`GroupPretrainCollator`)**:
  - Each batch element is a *list* of short texts grouped to a ~conversation_max_length budget.
  - For each text, randomly assign `<COMP>` or `<RECON>` task (controlled by `completion_freq`, default 0.5).
  - **Evidence text** = concatenated raw texts joined with `<EOT>` (with `<COMP>` ones truncated by `1 - random(0.1, 0.3)`). The evidence is the input to the **memory extraction pass**.
  - **Input/conversation** is built via `tokenizer.apply_chat_template` with messages alternating `user: <RECON>` or `user: <COMP>` and `assistant: <full text>`. This is the input to the **answering pass**.
  - **Random permutation** of order between evidence and conversation — ensures generated LoRA can answer in any order.
  - Labels masked so loss is only on assistant content (`mask_label` in `BaseCollator`).
- **IFT MQA (`IFTCollator`)**:
  - Evidence = passage; conversation = pre-formatted list of `{role,content}` dicts (15 QA pairs concat).
  - Same dual-pass structure: encode evidence → answer multi-turn conversation.

### Hyperparameters (Qwen3-8B headline)

| Param | Value |
|-------|-------|
| Backbone | Qwen3-8B, frozen |
| LLM hidden size H | 4096 |
| LLM num layers L | 36 |
| Generated LoRA rank | 8 |
| Meta LoRA rank | 128 |
| Num memory tokens M | 148 (= ⌈rD/H⌉) |
| M2P Transformer layers | 4 |
| M2P d_model / nhead / dim_ff | 4096 / 32 / 8192 |
| M2P activation | gelu |
| M2P layer norm | **post-LN** (`norm_first: False`) |
| Reshape method | `rl` |
| Scale (LoRA init) | 0.001 |
| Pretrain LR | 5e-5, warmup 200 steps |
| IFT MQA LR / epochs | 3e-5 / 2 |
| IFT 1QA LR / epochs | 1e-5 / 1 |
| Batch / accumulation | 1 × 4 grad accum × 8 GPUs |
| Context max length | 1150 tokens (pretrain) |
| Optimizer | AdamW, weight decay 0.01, grad clip 1.0 |
| Precision | bf16 AMP |

### Interesting impl tricks

- **`@torch.compile` on `Metanetwork.forward`**: fuses two LLM passes + M2P transform + LoRA reshape into one compiled graph.
- **`num_beams` support in `LoraLinear`**: a single LoRA can be applied to multiple sampled continuations efficiently.
- **`mask_label` in collators**: only loss on assistant tokens.
- **Special token `<NOTHING>`** added to vocab but author later commented out its zero-init; presence implies they prototyped a "no-op" task token.
- **`reg_loss = adapter_reg * |plain_output|.sum()`**: L1 reg on the produced LoRA tensor (default coefficient 0 in current config — must have been ablated and dropped).
- **Modes`rr` / `ll` (outer-product reshape) are implemented but not used** (paper says they converge poorly).
- **Coupled cross-attention** layers (`couple_layers`) are implemented with a precomputed `couple_mask` enforcing A→B / B→A only attention. Tested but Appendix E.1 shows full attention wins — hence default `couple_num_layers=0`.
- **`generate_group_idx.py`**: precomputes which short texts to pack into each training example so the conversation fits within `conversation_max_length`. Run once before pretraining.
- **`use_gradient_checkpoint`** plumbed through every layer (not enabled by default in the 8GPU config — only used for larger memory pressure).

### Test scripts

- **`test_pretrain.py`**: PPL eval on Wikitext-2 by length bucket.
- **`test_pwc.py`**: multi-turn MS MARCO MQA F1 (Fig 6 numbers).
- **`test.py`**: 1QA across SQuAD/MS MARCO/HotpotQA/MuSiQue/2Wiki/RACE/DuoRC.
- Uses `gpt-4.1-nano` as a judge (configurable) for F1 calc on free-form answers.

---

## Key takeaways for future research

- **"In-context hypernetwork"** is a great idea: leverages the pretrained LLM's own semantic encoder to bridge text→params. Saves you from training a separate encoder.
- **Memory token = parameter channel**: by sizing `M·H ≥ #LoRA params`, you avoid any compression bottleneck. Different from gist-token approaches.
- **Bidirectional row/col attention** is sufficient to model layer-wise weight dependencies — no need for hand-designed structure (the "bitter lesson" experiment in Appendix E.1 is a strong negative result).
- **Two-stage pretrain → IFT** mirrors LLM lifecycle. Reconstruction + completion is a clean self-supervised objective for "context absorption."
- **Post-layernorm in shallow hypernet** — small but useful empirical finding (pre-LN is bad when input/output distributions differ wildly).
- **Scaling looks promising**: PPL keeps dropping with model/hypernet size; no apparent saturation at 8B backbone.
- **Where it falls short**: long context, reasoning, multi-turn. Each is a natural follow-up direction.
