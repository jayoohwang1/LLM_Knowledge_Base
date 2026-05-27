# Self-Pruned Key-Value Attention: Learning When to Write by Predicting Future Utility (SP-KV)

**Authors**: Szilvasy, Faysse, Lomeli, Douze, Mazaré, Cabannes, Yih, Jégou — Meta FAIR & CentraleSupélec (MICS)
**Venue/Date**: arXiv 2605.14037, May 2026 (preprint)

---

## TL;DR

- **Core idea**: instead of evicting KVs at *read time* (H2O / SnapKV / Quest / KVZap), decide at *write time* whether each KV pair is worth keeping in the long-term cache, based on a learned **utility predictor**.
- **Mechanism**: small MLP scores each KV with $u_s^{l,k}\in(0,1)$. A **local sliding window** (default $w=128$) always attends to recent KVs; **older** KVs are reachable from global attention only if $u_s \ge \tau$.
- **Trained jointly** with the LM via standard next-token prediction loss — **no auxiliary sparsification loss**. Continual pretraining from a dense checkpoint.
- **Results**: dynamic compression $3$–$10\times$ KV cache reduction, $\sim 0\%$ NLL degradation, $2.1$–$4.6\times$ decode speedup at batch 16 / 32k context. Beats H2O / KVZap / StreamingLLM / ExpectedAttention by a wide margin in the controlled NLL-vs-density comparison.
- **Bonus**: learned per-head/per-layer densities act as an **architectural probe** — can guide where to place global vs local heads in hybrid architectures.

---

## Motivation

- **KV cache is the inference bottleneck** in modern long-context, agentic, RL-rollout settings.
	- Grows linearly with seq length; read every decode step → memory bandwidth bound.
	- GQA / MLA reduce *width* of cache; not orthogonal to *token-axis* compression.
- **Sparse-read vs sparse-write** distinction:
	- **Sparse-read** (H2O, SnapKV, Quest, DeepSeek Sparse Attention): full cache stored, only subset *read* per query. Saves bandwidth/FLOPs, not memory.
	- **Sparse-write** (StreamingLLM, KVZap, this paper): selectively *write* to the persistent cache. Saves memory AND bandwidth.
- **Train–test mismatch in post-hoc pruning**:
	- Eviction policies calibrated on small aux data; the LM weights were not adapted to expect pruning → quality drops as compression becomes aggressive or distribution shifts.
	- **SP-KV insight**: train the LM and the pruner *jointly end-to-end*, with the only loss being next-token prediction — the model adapts its internal token representations to the pruning behavior.
- **Question posed**: *Is it necessary to write every token indiscriminately into long-term memory?*
	- Empirically: no — queries/keys specialize into short-range vs long-range and a small subset of past KVs is consistently useful.

---

## Method

### Architecture: gated KV writing

- Let $H^l = [h_0^l, \dots, h_{T-1}^l]^\top \in \mathbb{R}^{T\times d_\text{model}}$ be hidden states at layer $l$.
- **Utility predictor** $f_\theta^{l,k}$ — a 2-layer MLP per layer × KV-head:
	- $u_s^{l,k} = \sigma(f_\theta^{l,k}(h_s^l)) \in (0,1)$
	- Outputs one gate **per KV head** (under GQA, broadcast across query-head group).
	- 1-layer linear predictors also tested → MLPs sharper / higher sparsity.
- **Hard threshold gate** at inference:
	- $z_s = \mathbf{1}[u_s \ge \tau]$, $\tau$ default $0.5$.
	- $z_s=1$ → KV eligible for global (long-range) attention; $z_s=0$ → dropped from persistent cache (still attendable via local window if within $w$).
- **Local sliding window** $w=128$ tokens — **always available** regardless of gate.
	- Preserves local temporal features, avoids gradient flow issues on recent tokens.
- **Combined availability mask**:
	- $\mathbf{1}_\text{win}(t,s) = \mathbf{1}[0 \le t-s < w]$
	- $g_{t,s} = 0$ if $(z_s=1) \vee \mathbf{1}_\text{win}(t,s)$, else $-\infty$
	- $B_{t,s} = M_\text{causal}(t,s) + g_{t,s}$
- **Gated attention**:
$$o_t = \sum_s \mathrm{softmax}\!\left( \frac{\langle q_t, k_s\rangle}{\sqrt{d}} + B_{t,s} \right) v_s$$

### Training procedure

**Two phases**, with no explicit sparsity loss — sparsification emerges from next-token-prediction.

**Phase 1 — Soft Gated Training**:
- Thresholding breaks gradient flow. Replace it with **log-utility additive bias** (soft gating):
	- $\widetilde{g}_s = \log u_s$
	- $u=1$ → bias $0$ (no effect); $u=0$ → bias $-\infty$ (like a closed gate).
- All utility gates **initialized fully open** (bias ≈ 5 → $\sigma(5)\approx 0.993$). Model learns to gate off keys without any sparsity objective — driven purely by NLL.
- Lasts for first 75% of cosine-decay schedule.

**Phase 2 — Thresholding-Aware Hard Gating (TAHG)**:
- Freeze utility predictor weights; binarize $z = \mathbf{1}[u \ge \tau]$.
- To avoid sharp transition shock, anneal over $\sim 500$ steps:
	- $\bar u = (1-\alpha) u + \alpha \mathbf{1}[u\ge \tau]$, $\alpha$ ramped $0\to 1$.
- Reduces train–test discrepancy from soft → hard.

**Alternative — hard gating throughout (Appendix B)**:
- $z_s \sim \mathrm{Bernoulli}(u_s)$, straight-through estimator for backward:
	- $\widetilde{g}_s = g_s.\text{detach}() + \log u_s - (\log u_s).\text{detach}()$
- Slightly worse than soft → TAHG due to STE noise.

### Continual pretraining setup

- Start from full-attention checkpoint trained at **140 TPP** (tokens-per-non-embedding-parameter; ~8× compute-optimal Chinchilla).
- Branch: continue 20 TPP either as (i) baseline full attention or (ii) SP-KV with cosine decay.
- Matched data, optimizer, compute budget.
- 8k seq lengths; 8.1B models context-extended to 32k during last 10 TPP.
- GQA throughout (already 4–6× KV reduction vs MHA — MHA gave higher sparsity since keys more redundant).

### Hyperparameter knobs (Appendix C)

- **Utility predictor**:
	- 2-layer MLP > linear (more selective).
	- Initial bias ≈ 5 → gates open at init.
	- LR multiplier ×5 on predictor → faster polarization → more sparsity.
- **Bernoulli probability clipping** $u \to \mathrm{clip}(u, p_\text{min}, 1-p_\text{min})$: prevents saturation, slows sparsification, stabilizes.
- **Aux density regularizer** (optional):
	- $\mathcal{L}_\text{aux} = -\lambda_\text{aux} \cdot \frac{1}{LKT}\sum_{l,k,s} u_s^{l,k}$ (penalizes low utility → denser).
	- Tunes operating point on Pareto frontier.
- **Local window** $w \in \{64, 128, 256, 512\}$ — larger windows → *higher* long-range sparsity (more locality already covered).

---

## Kernel / systems implementation

### Training (FlashAttention 3 fork, Hopper)

- **Gate bias** $\log u_s$ fused into FA3's pre-softmax score accumulation. Per-element, negligible arithmetic cost.
- Gate values prefetched once per KV-block column → no reloads across query rows in a tile.
- Backward: gate gradients $\partial \mathcal{L}/\partial \log u_s$ accumulated via atomic adds (multiple queries contribute).
- **Phase 2 block skipping**: binarized gates allow precomputing a per-head sparsity mask at 64-token granularity. Skip both TMA loads and MMA compute for all-zero KV blocks outside the sliding window.
- Throughputs vs full-attention baseline (1.6B–8.1B params):
	- **Phase 1 soft**: 60–62% of baseline.
	- **Phase 2 hard** ($\tau=0.7$): **82–90%** of baseline (most overhead recouped via block skipping).

### Inference (FlashInfer fork)

- Standard FlashInfer uses tightly-packed paged KV cache (`indptr` no-slack) → appending forces shifting all subsequent entries.
- **SP-KV needs per-head variable-length caches** (each head retains different tokens). Their fork pre-allocates page-index headroom per head:
	- Append page → single $O(1)$ `atomicAdd` on `used_pages[i]` + one index write.
	- Capacity dynamically grown.
- **Single fused CUDA kernel** for token-append per decode step:
	- Writes new token into next window slot; if window full → check evicted token's gate; retained → atomic page allocation for long-term; pruned → discard.
	- Capacity checks amortized via precomputed safe-step budget.
- Sliding-window region and retained long-range tokens share a single paged pool; window tracked via circular write pointer + per-slot gate metadata.

---

## Experiments

### Scaling analysis (Section 3.1)

- 48M to 7.0B non-embedding params, 11 compute regimes.
- Power-law fit: $L(C) = L_\infty + A C^{-\alpha}$, $R^2 > 0.999$ for both FA and SP-KV.
	- FA: $L_\infty=1.070$, $A=143.7$, $\alpha=0.0964$.
	- SP-KV ($\tau=0.5$): $L_\infty=1.054$, $A=139.0$, $\alpha=0.0955$.
- 8.1B-param model validates extrapolation (not used in fit).
- **Takeaway**: SP-KV achieves nearly identical loss scaling as full attention while keeping only **29.6%** of keys (at 8.1B, $\tau=0.5$). Adapting via 1/8 of full budget (continual PT) doesn't degrade scaling.

### Downstream tasks (Section 3.2, Table 1)

- 8.1B model, $\tau=0.5$, evaluated against vanilla full-attention.

| Category | Avg Δ | Avg gate density (non-local KVs kept) |
|---|---|---|
| **Standard suite (16 tasks)** | **−0.2%** | 33.7% |
| **RULER (4k–32k, 13 subtasks)** | −1.2% | 17.7% |

- Standard tasks essentially unchanged ($-0.2\%$).
- RULER: 4k +0.0% / 8k −0.9% / 16k −0.3% / 32k −3.9% — 32k is at edge of training context (OOD).
- **Per-task density varies widely**: 17–19% on NIAH (sparse retrieval), 40–50% on ARC/OBQA (broad context needed), 17–25% on generative GSM8k/HumanEval (focused chain-of-thought).
- **With RULER-style data in training mix** (Table 6): degradation collapses from $-1.2\%$ to $-0.3\%$ — model just needs exposure to long-context distribution.

### Sparsity–performance tradeoff (Section 3.3)

- $\tau$ sweep yields smooth Pareto: at $\tau=0.5$ → ~26% density with $+0.07\%$ NLL on 2.91B model; at $\tau=0.1$ → 60% density and near-zero NLL gap.
- **Pareto-dominates Hybrid 3:1** (Gemma-style interleaved local-global) at the same KV budget.
- Sparsity can also be controlled with aux density loss $\lambda$ (Figure 6) — alternative train-time knob.

### Inference efficiency (Section 3.4, Appendix E)

- At batch 16, **decode latency** vs full attention:
	- $2.1\times$–$4.6\times$ faster across 1k–32k seq lengths.
	- Gains shrink at low sparsity / short sequences (kernel overhead).
- Theoretical: at 80% sparsity, FLOPs equivalent to attending over $5\times$ shorter sequence — effective context-length expansion.
- Kaplan-style accounting: dense $C(L) = 2N + 2 n_\text{layers} d_\text{attn} L$, sparse $C_\text{sparse}(L) = 2N + 0.2 \cdot 2 n_\text{layers} d_\text{attn} L$.

### Comparison to KV compression baselines (Table 2, Appendix A.2)

- All on Llama-3.1-8B-Instruct (reference `kvpress` impls), chunked-prefill protocol, 128 sliding window + 4 sink tokens.
- Reported as **relative NLL increase** vs each method's dense baseline (controlling for different training regimes).

| Method | NLL | Δ NLL | Density |
|---|---|---|---|
| StreamingLLM | 2.141 | +11.86% | 0% |
| ExpectedAttention | 1.990 | +3.95% | 20.70% |
| H2O | 1.977 | +3.26% | 20.45% |
| KVZap (+4 sinks) | 1.938 | +1.23% | 20.15% |
| **SP-KV ($\tau=0.5$)** | **1.889** | **+0.08%** | 25.72% |
| **SP-KV ($\tau=0.7$)** | **1.896** | **+0.46%** | **11.44%** |

- SP-KV at **11.44% density** beats KVZap at 20.15% density.
- Why: SP-KV trained on 140B tokens with joint sparsification ≫ KVZap (~5 orders of magnitude less calibration budget) or DMS-7B retrofit budget.
- **Caveat**: post-hoc baselines simulated via masking (no sparse-decode kernels available for fair systems comparison).

### Ablations (Table 4, ~1.05B models)

- **Frozen LLM** (only train predictor, like prior post-hoc methods): density stays >80% — model needs to co-adapt for sparsification to emerge.
- **Linear predictor**: 33.3% density, slightly worse than MLP.
- **Bidirectional utility predictor** (1D conv kernel 65 in MLP, $\pm 32$ token lookahead within window): *degrades* validation loss — temporal aggregation needs more care, left as future work.
- **Training from scratch** with MLP predictor: 15.8% density (much sparser) but slightly worse NLL — joint with CPT is sweet spot.
- **Window size**: $w=1$ → 60.7% density with bad NLL; $w=512$ → 27.5% density with best NLL.
- **TASG (Thresholding Aware Soft Gating)** $\tau=0.5$ → 34.6% density and $-0.6\%$ NLL vs SP-KV baseline.

---

## Architectural probe: SP-KV-guided hybrid design (Section 4)

- **Insight**: closed gates = a causal *intervention* (those KVs unavailable to future tokens). Retained gates therefore identify interactions the model *needs* for next-token prediction.
- Per-head, per-layer densities are highly non-uniform — some heads almost always retain, others almost always discard.

**Setup**: select 18/63 heads to keep global (28.6% of heads) → compare allocation strategies.

| Strategy | Density | Coverage of ref SP-KV mass | Δ NLL |
|---|---|---|---|
| A. 3:1 interleaved (CWM-style, global at L0) | 28.6% | 8.28% | +0.396% |
| B. 3:1 offset (Command-A-style, global at L3) | 28.6% | 45.65% | +0.222% |
| C. 18 random heads | 28.6% | 34.34% | +0.426% |
| **D. 18 densest heads (SP-KV-guided)** | 28.6% | **68.44%** | **+0.161%** |
| Full SP-KV ($\tau=0.5$) | 28.1% | 100% | +0.074% |
| Full global attention | 100% | — | 0% |

- Picking the 18 highest-SP-KV-density heads as the global subset gives best hybrid layout among fixed selections.
- Full SP-KV still wins → learned per-token gating > static head selection, but the latter is simpler to deploy and matches throughput optimization patterns of hybrid arch.

---

## Toy demo: palindrome reversal with long instruction gap (Appendix G)

- Input: 32 two-digit ints + long ($\ge 50$ token) instruction string + reversed output.
- Trained from scratch on cross-entropy over output tokens only.
- **Full attention**: converges to near-zero loss in ~500 steps.
- **Sliding-window (w=32)**: fails (instruction string pushes input out of window).
- **SP-KV (local 32 + gates)**: converges as fast as full attention → gating retains the input tokens through the gap.
- **Demonstrates**: SP-KV is strictly more expressive than sliding-window; full attention is the $\tau=0$ limit.

---

## Related work positioning

- **Sparse-read**: Quest, DeepSeek Sparse Attention, RetrievalAttention, Landmark Attention, kNN memories — full cache stored. Orthogonal to SP-KV.
- **Sparse-write / cache compression**:
	- **StreamingLLM** — keep recent + sink only (fixed).
	- **H2O** — heavy-hitter eviction.
	- **SnapKV, AdaKV, RazorAttention, KeyFormer, Pyramid-KV** — adaptive scoring at prefill/decode time on frozen model.
	- **ExpectedAttention, KVZap** — learned eviction policies on frozen LLM.
	- **DMS** — distillation + sparsification objective retrofit.
	- **SP-KV** — only one to **train LM weights + policy jointly at full pretraining scale, with no auxiliary loss**.
- **Architectural KV compression**: GQA, MLA, Cross-Layer Attention, KV quantization (KIVI, MiniCache). **Orthogonal** — SP-KV reduces token axis.
- **Hybrid local-global**: Longformer, Gemma 2, Gated DeltaNet, Infini-attention — hard-coded structure. SP-KV is unconstrained → contains full attention as limit (all gates open).

---

## Limitations & future work

- **English-centric pretraining corpus only** — multilingual / domain transfer unverified.
- No post-training evaluation (SFT, RLHF, RL rollouts).
- Inference systems impl not fully optimized — kernel work needed to convert sparsity into proportional wall-clock gains.
- 32k context is upper bound trained → degradation at the edge.
- Bidirectional predictor variant degrades — temporal aggregation in utility predictor is open.

---

## Key takeaways / why this matters

- **Sparse-write learned jointly with the LM** sidesteps the train–test mismatch that plagues all post-hoc eviction methods, and produces sparsification ratios (~5–10×) that previously required quality sacrifices.
- **No auxiliary loss** is needed — gating sparsifies because next-token prediction *prefers* a smaller, cleaner working set. This is conceptually clean and lets you tune sparsity post-hoc via $\tau$ without retraining.
- **Layer/head density patterns are interpretable** and usable as architecture-search signal → bridges KV compression and NAS for hybrid attention.
- **Practical implication for RL / agentic training**: sparse-write is the only paradigm that gives memory wins during decoding rollouts (sparse-read keeps cache full). SP-KV being trained-in means downstream SFT/RL inherits the savings.
- **Caveat**: requires continual pretraining from a dense checkpoint (not free), and current efficiency gains require custom FA3/FlashInfer forks.

---

## Reference implementation (Appendix I, Listing 1)

```python
class UtilityPredictor(nn.Module):
    """Per-key, per-head utility predictor: u_s in (0, 1)."""
    def __init__(self, d_model, hidden, n_kv_heads):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, n_kv_heads))

    def forward(self, h):                      # h: [B, T, d_model]
        return torch.sigmoid(self.net(h))      # -> [B, T, n_kv_heads]
        # GQA: one gate per KV head; broadcast to query groups.

def sp_kv_attention(q, k, v,                   # [B, H, T, D]
                    utility,                   # [B, H, T] in (0,1)
                    hard=False, tau=0.5,
                    window_size=128):
    B, H, T, D = q.shape
    if hard:
        gate = torch.where(utility >= tau, 0.0, float("-inf"))
    else:
        gate = torch.log(utility + 1e-8)       # soft (training)

    qi = torch.arange(T).unsqueeze(1)
    ki = torch.arange(T).unsqueeze(0)
    causal    = ki <= qi
    in_window = causal & ((qi - ki) < window_size)

    mask = torch.where(in_window, 0.0,
              torch.where(causal, gate[:, :, None, :], float("-inf")))

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

---

**No public code release at time of writing (preprint).**
