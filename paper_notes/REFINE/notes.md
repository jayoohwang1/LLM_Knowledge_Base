# REFINE: Reinforced Fast Weights with Next-Sequence Prediction

> Hwang*, Wu*, Chun, Russakovsky (Princeton Visual AI). arXiv 2602.16704, Feb 2026.
> **REFINE** = **Re**inforced **F**ast we**I**ghts with **N**ext-s**E**quence prediction.
> TL;DR: fast-weight LMs are pre-trained with **next-token prediction (NTP)**, which only gives token-level supervision and ignores multi-token coherence. REFINE instead trains them on a **next-sequence prediction (NSP)** objective, formulated as an **RL problem** (GRPO) with **entropy-selected positions**, **multi-token rollouts**, and **self-supervised sequence-level rewards** (cosine sim of hidden states / exact match). Phase-agnostic: works for **mid-training, post-training, test-time training (TTT)**. Beats SFT-with-NTP on LaCT-760M & DeltaNet-1.3B across NIAH, multi-doc QA, LongBench.

---

## 1. Motivation

- **Fast weight architectures** (DeltaNet, GatedDeltaNet, LaCT, Mamba/SSMs, linear attention) replace global softmax attention with **fixed-size memory = a weight matrix $W$** that is **dynamically updated** as tokens stream in.
  - **Constant memory + linear compute** in context length vs transformer's **quadratic** attention + growing KV cache.
  - Generalized fast-weight update rule (from Zhang et al. 2025), with $k_t, v_t$ key/value of token $t$:
    $$W_{t+1} \leftarrow W_t - \eta \nabla_{W_t}\,\ell(W_t k_t, v_t)$$
    - i.e. each token does **one online gradient step** learning a key$\to$value map; readout $= W_t q_t$.
    - This **is** test-time training by construction — fast weights = inner-loop SGD over the context. Strongly connected to "fast weight programmers" and meta-learning (Clark et al. 2022).
- **Problem: NTP is a suboptimal objective for fast weights.**
  - NTP CE loss only has **immediate effect on the next token**, disregards quality of subsequent predictions that depend on the same internal state (Gloeckle et al. 2024 multi-token-prediction line).
  - Token-level feedback → parameter updates optimize only **short-term likelihood** → limits the **adaptive capacity** of fast weights over long horizons.
  - Two specific NTP limitations:
    1. **Single-token only** — ignores semantic relationships among the $k$ tokens following a prefix.
    2. **Uniform aggregation** — sums CE over all positions, ignoring which local regions are actually informative over long context.
- **Why fast weights especially suffer:** they *store contextual info in parameters*; if training only rewards 1-step continuation, the memory never learns to support coherent multi-step recall (→ unstable on NIAH retrieval).

---

## 2. From NTP to NSP

### Next-Token Prediction (NTP)
$$\mathcal{L}_{\text{NTP}} = \sum_t -\log p(x_{t+1}\mid x_{\le t})$$

### Next-Sequence Prediction (NSP)
- Optimize **multi-token sequence alignment** at **selected positions** $\mathcal{T}^*\subseteq\{1,\dots,T\}$:
  $$\mathcal{L}_{\text{NSP}} = \sum_{t\in\mathcal{T}^*}\mathcal{L}_{\text{seq}}(\hat{x}_{t+1:t+k},\,x_{t+1:t+k}),\quad k>1$$
  - $\hat{x}_{t+1:t+k}$ = generated $k$-token continuation; $x_{t+1:t+k}$ = ground-truth continuation.
- **Two challenges with naive NSP:**
  1. CE doesn't extend to multi-token continuation without **explicitly generating** the $k$ tokens (rollouts).
  2. Generating $k$ tokens at **every** prefix is **computationally prohibitive** for long contexts.
  3. (subtler) directly matching a **single reference** with CE **over-penalizes plausible paraphrases**.
     - Ex: GT "cars are fast" vs "automobiles move quickly" → high CE despite semantic equivalence.
- **Solution = cast NSP as RL.** Maximize expected self-supervised sequence reward $R$ over rollouts; only roll out at **informative (high-entropy) positions**; use a **smooth semantic reward** so paraphrases get credit.
  $$\mathcal{L}_{\text{seq}} = -\mathbb{E}_{\hat{x}_{t+1:t+k}\sim\pi_\theta(\cdot\mid x_{\le t})}\big[R(\hat{x}_{t+1:t+k},\,x_{t+1:t+k})\big]$$
  - Shorthand $R(t) \equiv R(\hat{x}_{t+1:t+k}, x_{t+1:t+k})$.
  - **Advantages:** (1) $k$-step continuations carry higher information than 1-step; (2) reward can credit **multiple plausible continuations** by semantic similarity.
  - First work to do **RL-based NSP for fast weight models** (prior NSP work — Gloeckle, Liu et al. — was for standard transformers).

---

## 3. REFINE Framework (4 steps)

Pipeline (Fig 3): **forward seq → entropy-based token selection → fill prefix & truncate → rollout generation → reward → GRPO update.**

### Step 1 — Entropy-Based Token Selection
- Forward $S=(x_1,\dots,x_T)$ through policy $\pi_\theta$; compute per-position NTP entropy
  $$H_t = H\big(\pi_\theta(\cdot\mid x_{\le t-1})\big)$$
  - High entropy = high uncertainty = informative region worth multi-token supervision.
- **Smooth** entropy with 1-D average pooling, kernel size $k$ (rollout length), to reflect that we care about the next-$k$-token uncertainty.
- **Partition** $S$ into $c$ contiguous **equal chunks** $S=(S_1,\dots,S_c)$. From each chunk $S_i$ sample **one** position via softmax-over-entropy:
  $$p_i(t) = \frac{e^{H_t/\tau}}{\sum_{t'\in\mathcal{T}_i} e^{H_{t'}/\tau}},\qquad \mathcal{T}_i=\{t'\mid x_{t'}\in S_i\}$$
  - Draw $t_i\sim p_i(t)$ → one entropy-weighted position per chunk → $\mathcal{T}^*=\{t_1,\dots,t_c\}$.
  - $\tau$ = temperature (default $1$).
  - **Per-chunk sampling** distributes training signal evenly across the whole sequence length while still focusing on locally-uncertain tokens. (App E.2: entropy is **not** index-dependent, justifying per-chunk weighting rather than global.)

### Step 2 — Rollout Generation
- For each $t_i\in\mathcal{T}^*$: build **truncated prefix** $x_{\le t_i}$ (prefixes copied from original seq up to each target token), yielding $c$ partial sequences.
- Generate a **$k$-token continuation** $\hat{x}_{t_i+1:t_i+k}$ with current policy; extract **final-layer hidden states before logits**:
  $$\mathbf{h}^{\text{pred}}_k(t_i) = \big(\mathbf{h}^{\text{pred}}(t_i+1),\dots,\mathbf{h}^{\text{pred}}(t_i+k)\big)$$
- Also extract GT-continuation hidden states from the **initial forward pass**:
  $$\mathbf{h}^{\text{gt}}_k(t_i) = \big(\mathbf{h}^{\text{gt}}(t_i+1),\dots,\mathbf{h}^{\text{gt}}(t_i+k)\big)$$

### Step 3 — Reward Assignment (self-supervised, sequence-level)
- **Cosine-similarity reward** over the $k$-token rollout, $\mathbf{h}\in\mathbb{R}^{k\times d}$:
  $$R^\varphi_k(t_i) = \frac{1}{k}\sum_{j=1}^{k}\varphi\big(\mathbf{h}^{\text{pred}}(t_i+j),\,\mathbf{h}^{\text{gt}}(t_i+j)\big),\quad \varphi=\cos$$
  - **Smooth, non-zero** reward → semantically similar tokens land **close in latent space** → improves generalizability & early-training stability vs hard CE/exact-match.
  - Qualitative (App F): GT "loved every minute of it" → "enjoyed every minute of it" scores **0.961**, "would not recommend it" scores **0.463**; captures meaning beyond lexical overlap.
- **Binary exact-match reward** (memorization-heavy regimes, e.g. TTT):
  $$R^{\text{binary}}_k(t_i) = \frac{1}{k}\sum_{j=1}^{k}\mathcal{I}[x_{t+j}=\hat{x}_{t+j}]$$
- **Hybrid reward** (post-training, train/test same dist → balance generalization + memorization):
  $$R^{\text{hybrid}}_k(t_i) = R^\varphi_k(t_i) + R^{\text{binary}}_k(t_i)$$

### Step 4 — Optimization with RL (GRPO)
- Rollouts at sampled positions $o_{\mathcal{T}^*}=\{\hat{x}_{t_1+1:t_1+k},\dots,\hat{x}_{t_c+1:t_c+k}\}$ with rewards $\mathcal{R}_{\mathcal{T}^*}=\{R^\varphi_k(t_1),\dots,R^\varphi_k(t_c)\}$.
- Rewards from the **same sequence $S$ are standardized** to compute advantages (group baseline) → **GRPO** (Shao et al. 2024).
  $$\mathcal{J}(\theta)=\mathbb{E}_{x_{\le t}\sim\mathcal{D},\,\hat{x}_{t+1:t+k}\sim\pi_{\theta_{\text{old}}}(\cdot\mid x_{\le t})}\big[R^\varphi_k(t)\big]$$
- **Final loss = weighted sum of NSP (RL) loss + standard NTP loss over full $S$** (prevents catastrophic forgetting):
  $$\mathcal{L} = \lambda_{\text{RL}}\,\mathcal{L}_{\text{NSP}} + \lambda_{\text{SFT}}\,\mathcal{L}_{\text{NTP}}$$
  - Weights adjusted per phase (Table 2).

---

## 4. Applicability Across Training Lifecycle (phase-agnostic)

| Phase | Data source | What REFINE does | Reward | $\lambda_{\text{RL}}$ |
|---|---|---|---|---|
| **Pre-training** | general web | (not applied; baseline) | — | — |
| **Mid-training** | similar to pre-train (Long-Data-Collections) | adapt pretrained init to NSP on same-domain data | $R^\varphi$ | 0.2 |
| **Post-training** | task/instruction/preference data | **nested learning**: inner loop = REFINE on prompt, then SFT on reference response | $R^{\text{hybrid}}$ | 0.2 |
| **Test-time training (TTT)** | the test prompt itself (no labels) | adapt fast weights on prompt before generating answer | $R^{\text{binary}}$ | 0.4 |

- **Mid-training**: continued pre-training (Gururangan et al.) to align init to NSP objective/reward.
- **Post-training (nested learning, Behrouz et al. 2025b)**: within each training loop, **first update model on the prompt with REFINE**, then SFT to align final response to reference. Three variants compared: SFT / Nested SFT / Nested REFINE.
- **TTT**: low-data regime (small batch), memorization > generalization → binary reward + higher $\lambda_{\text{RL}}=0.4$. Naturally fits fast weights (already do gradient updates per token); fixes NIAH instability of pure-NTP models.

---

## 5. Experimental Setup

- **Models**: **LaCT-760M** (Zhang et al. 2025 — updates fast weight params; SwiGLU-MLP fast weights) and **DeltaNet-1.3B** (Yang et al. 2024 — fixed params, parallelizable memory state). Two distinct fast-weight mechanisms.
- **Mid-train data**: Long-Data-Collections (LaCT's pre-train corpus; 68.8B-token subsample of RedPajama/Pile/UL2/NI/P3), 200M-token subset @16K ctx.
- **Benchmarks**:
  - **RULER NIAH** (single/multi-key/multi-query/multi-value) @4K/8K/16K — recall.
  - **Booksum** — NTP val accuracy / CE loss.
  - **Multi-doc QA**: RULER **SQuADQA**, **HotpotQA** (synthetic, 1600 train / 200 test).
  - **LongBench** 12 English tasks ≤16K (NarrativeQA, Qasper, MultiFieldQA, HotpotQA, 2WikiMQA, QMSum, MultiNews, SAMSum, TREC, TriviaQA, LCC, RepoBench-P); leave out MuSiQue/GovReport (<20 samples <16K).
  - **Short-context (App E)**: PIQA, HellaSwag, WinoGrande, ARC-e/c, Wikitext, LAMBADA, FDA, SWDE — check no catastrophic forgetting.
- **Train config (Table 2 / D.1)**: $c=8$, $k=5$, $n=1$ fixed across phases; vary $\lambda_{\text{RL}}$ & reward. lr $10^{-6}$, Adam (0.9, 0.999), wd 0.01, actor grad clip 0.2, $\tau=1$, KL coeff $=0$, entropy coeff $=0$, max prompt 16,384. Batch: MidTr 128 / PostTr 64 / TTT 8. PPO mini-batch: 32/16/4.
- **Compute**: 8×L40 (mid/post), 4×L40 (TTT). Mid-train 200M tokens @16K ≈ **24h**.

---

## 6. Results

### Mid-training — NIAH (Table 3)
- REFINE > pretrained > SFT-mid on most NIAH variants.
- **DeltaNet MK-NIAH avg: +23.5% over no-mid-train, +8.8% over SFT-mid.** Largest gains on multi-key retrieval.
- Avg long-context QA on RULER: LaCT +8.5% (mid) over pure NTP.

### Mid-training — multi-doc QA (Table 4) & LongBench (Table 5)
- REFINE mid (3rd rows) **consistently beats SFT mid** (2nd rows) by large margins.
  - DeltaNet RULER HotpotQA avg: **+73.1% over no-mid-train, +22.0% over SFT**.
- LongBench avg (LaCT): SFT 13.5 → REFINE 16.9. DeltaNet: 15.7 → 17.0.

### NTP accuracy / val loss (Fig 4, E.1)
- **Striking**: even though SFT *directly* optimizes NTP, REFINE's **NTP val accuracy gains exceed SFT's**; SFT-on-LaCT is stagnant (already pretrained on this data) while REFINE keeps improving.
- Val loss: SFT flat, REFINE decreases → **NSP provides a learning signal NTP alone does not.**

### Post-training (Table 4)
- **Nested REFINE > SFT > Nested SFT.** LaCT SQuADQA avg: nested SFT 21.8 → nested REFINE **25.5 (+17%)**. DeltaNet **+24.1%** (10.3 vs 8.3).

### TTT (Table 4 rows 7–8, Table 5)
- TTT with REFINE > SFT-TTT across LongBench subtasks. LaCT LongBench avg w/ REFINE-mid + REFINE-TTT: **18.0** (vs SFT 13.1 base).
- NSP gives stronger adaptation signal for compressing contextual info into fast weights at inference.

### Short-context (Table E.1)
- REFINE-mid keeps short-context perf flat (no catastrophic forgetting) — NSP **complements** NTP.

---

## 7. Ablations

- **Rollout length $k$ (Fig 5 left, Table 5)**: avg score ↑ until $k=5$, ↓ at $k=7$. Hypothesis: reward sharpness degrades over longer rollouts (App E.3: cosine reward mean & variance both ↓ as $k$↑ → signal loses sharpness).
- **Chunks per sequence $c$ (Fig 5 right)**: monotone ↑ with $c$. LaCT 16.5 ($c{=}2$) → 16.9 ($c{=}8$); DeltaNet 16.3 → 17.0. More frequent sequence-level predictions → better fast-weight init.
- **Reward function (Table 6, mid)**: $R^\varphi$ (cosine) > $R^{\text{binary}}$. LaCT +1.8%, DeltaNet +3.0%. Smooth similarity reward generalizes better.
- **Reward function for TTT (Table E.2)**: $R^{\text{binary}}$ optimal for TTT (memorization regime); $R^\varphi$ still > pure SFT.
- **Token selection (Table 7)**: entropy-weighted sampling > uniform / argmax-$H$ / argmin-$H$. LaCT: +4.3% uniform, +3.0% max-$H$, +1.8% min-$H$. → mix of uncertainty levels, not just extremes, is best.

---

## 8. Limitations & Future Work

- Cosine reward **deteriorates for overly long rollouts** → consider edit-distance / richer semantic rewards.
- **Optimal $k$ is context-dependent** → dynamic adjustment per prefix could isolate semantically meaningful regions better.
- Full NSP integration into fast-weight **training** needs **architectural changes**; efficient **fast-weight transfer across truncated prefixes** would accelerate rollout generation (currently re-forward per prefix) → scale data/compute.

---

## 9. Relation to Prior Work

- **Fast weights / linear attention / SSMs**: DeltaNet, GatedDeltaNet, LaCT, Mamba, Linear Transformer, Linformer, Performer. Unlike attention-approximators, fast weights store context **in parameters via predefined online update rules**. REFINE targets this fundamentally different class.
- **Multi-token prediction**: Gloeckle et al. ($k$ parallel heads, fixed horizon, no inter-token dependency); Liu et al. (diffusion masked multi-token). REFINE rewards the **whole rollout** capturing semantic connection, and is for **fast weights** not transformers.
- **RL for NTP / continued pretrain with RL**: Dong et al. (reasoning before next token), Hatamizadeh et al. (log-likelihood gap reward). Those assume instruction-tuned/reasoning transformers; REFINE shows RL helps **pretrained fast weights with no prior instruction tuning**, and optimizes **sequence-level (NSP)** not token-level.
- **TTT in LMs**: Akyürek (in-context examples), Zuo/Huang (pseudo-label aggregation), Bansal (gradient updates on query proj). All on transformers; **TTT-on-fast-weights for long context is novel here**.
- **GRPO / RLHF**: standard group-relative policy optimization (Shao et al.) as the optimizer; reward is **self-supervised** (no human preference / no learned RM).

---

## Implementation

Code: official repo `ReFINE/`, a **fork of [verl](https://github.com/volcengine/verl)** (Ray-based distributed RL). REFINE logic is grafted into verl's PPO trainer + actor + HF rollout, gated behind `ttt_*` / `val_ttt_*` / `task_*` config flags. (Note: code uses the prefix **`ttt_`** generically for the REFINE inner-loop regardless of phase — "TTT" here means "the fast-weight RL update loop", not specifically test-time.)

**Entry point** (all 3 phases): `python3 -m verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo`, driven by the demo scripts in `verl/examples/refine_trainer/demo/`:
- `run_midtrain_demo.sh`, `run_posttrain_demo.sh`, `run_testtimetrain_demo.sh`.

Core REFINE files (paths under `ReFINE/verl/verl/`):
- `trainer/ppo/ray_trainer.py` — orchestration: token selection, prefix build, reward, advantage, fit loop.
- `workers/actor/dp_actor.py` — forward passes (hidden states / entropy / loss), SFT+PPO TTT update functions.
- `workers/rollout/hf_rollout.py` — multi-token rollout generation via HF `.generate`.
- `workers/fsdp_workers.py` — dispatch wrappers (`update_actor_ppo_ttt`, `process_input_for_ttt`, `generate_sequences_for_ttt`, …).
- `models/lact/` — vendored LaCT model (SwiGLU-MLP fast weights, chunk-wise online update, optional Muon).

### Config → paper symbol mapping

| Config key | Paper | MidTr | PostTr | TTT |
|---|---|---|---|---|
| `actor.ttt_n_chunks` | $c$ | 8 | 8 | 8 |
| `rollout.ttt_response_length` | $k$ (rollout len) | 5 | 5 | 5 |
| `actor.ttt_n` | $n$ rollouts/position | 1 | 1 | 1 |
| `actor.ttt_reward` | reward | `cosine_similarity` ($R^\varphi$) | `hybrid` | `binary` |
| `actor.ttt_ppo_loss_coef` | $\lambda_{\text{RL}}$ | 0.2 | 0.2 | 0.4 |
| `actor.ttt_sft_loss_coef` | $\lambda_{\text{SFT}}$ | 1.0 | 1.0 | 1.0 |
| `rollout.ttt_temperature` | $\tau$ sampling | 1.0 | 1.0 | 1.0 |
| `algorithm.ttt_sampling` | selection | `entropy` | (default `entropy`) | `entropy` |
| `algorithm.ttt_sft_update`/`ttt_ppo_update` | inner-loop on/off | T/T | T/T | — |
| `algorithm.task_sft_update`/`task_ppo_update` | **outer** (nested) loop | — | T/F | — |
| `algorithm.val_ttt_sft_update`/`val_ttt_ppo_update` | TTT @ validation | — | — | T/T |
| `rollout.name` | inference backend | `hf` | `hf` | `hf` |

- **Mid-train** = inner loop only (`ttt_*_update=True`), `val_get_loss=True` (validate via NTP loss/EM on Booksum).
- **Post-train** = inner loop (REFINE on prompt) **+** outer loop (`task_sft_update=True`, `task_ppo_update=False` → SFT on reference). This is the "nested learning". Custom task reward via `custom_reward_function.path=verl/utils/reward_score/ruler.py`.
- **TTT** = `trainer.val_only=True`, `val_before_train=True`; `_validate()` runs `_inner_training_loop(validation=True)` per test batch to adapt, then generates the answer; reward `verl/utils/reward_score/longbench.py`.

### Mechanism 1 — Entropy-based position selection + prefix construction

`ray_trainer.py::prepare_ppo_batch_for_ttt`. Entropies come from the initial forward (`process_input_for_ttt` → `_forward_micro_batch_for_input_ttt` → `verl_F.entropy_from_logits(logits)`).

```python
# entropy smoothing: avg-pool with kernel ~ k (rollout length)
if ttt_sampling == "entropy":
    if ttt_k > 1:
        kernel_size = ttt_k + 1 if ttt_k % 2 == 0 else ttt_k   # force odd kernel
        entropys = entropys.unsqueeze(1)                        # [B,1,S]
        smoothed_entropys = F.avg_pool1d(entropys, kernel_size=kernel_size,
                                         stride=1, padding=kernel_size//2).squeeze(1)
    else:
        smoothed_entropys = entropys
elif ttt_sampling == "uniform":
    smoothed_entropys = torch.ones_like(entropys)               # ablation
# else "argmax"/"argmin" use raw entropy
```
```python
# split each sequence into ttt_n_chunks equal chunks, sample 1 target per chunk
seq_start_idx = seq_attention_mask.argmax(-1).item()   # first non-pad (left-padded)
seq_len       = seq_attention_mask.sum(-1).item()
seq_chunk_size = seq_len // ttt_n_chunks
for j in range(ttt_n_chunks):
    chunk_start_idx = seq_start_idx + j * seq_chunk_size
    chunk_H = seq_smoothed_entropys[chunk_start_idx : chunk_start_idx + seq_chunk_size]
    if ttt_sampling in ["entropy","uniform"]:
        probs = F.softmax(chunk_H, dim=-1)                       # Eq. 6 (softmax over chunk)
        target_idx = torch.distributions.Categorical(probs=probs).sample()
    elif ttt_sampling == "argmax": target_idx = torch.argmax(chunk_H)
    elif ttt_sampling == "argmin": target_idx = torch.argmin(chunk_H)

    response_start_idx = chunk_start_idx + target_idx + 1        # next token after selected
    response_start_idx = torch.clamp(response_start_idx, min=seq_start_idx+20, max=S-20)
    response_end_idx   = response_start_idx + ttt_k

    # build LEFT-PADDED truncated prefix x_{<=t}; GT continuation = next k tokens
    seq_new_input_ids = torch.full((S,), pad_token_id)
    seq_new_input_ids[-response_start_idx:] = seq_input_ids[:response_start_idx]
    seq_new_ground_truth_ids            = seq_input_ids[response_start_idx:response_end_idx]
    seq_new_ground_truth_hidden_states  = seq_hidden_states[response_start_idx:response_end_idx, :]
```
- **Implementation details / deviations not in paper:**
  - Selected positions are **clamped to `[seq_start+20, S-20]`** — avoids degenerate prefixes at the very start/end of the context.
  - Entropy smoothing forces an **odd** avg-pool kernel ($k{+}1$ if $k$ even). Kernel ≈ rollout length $k$.
  - **GT hidden states are cached from the original forward pass** (`hidden_states` of the full seq) and sliced — so $\mathbf{h}^{\text{gt}}$ never needs a second forward. Matches paper Eq. 8.
  - Each sampled position becomes its own training example (one prefix); when `ttt_n>1`, examples are `repeat(interleave=True)` and assigned a **shared `uid` per position** so GRPO groups the $n$ rollouts of the same position. With `ttt_n=1` (the default everywhere), the GRPO group is **all $c$ positions of a sequence** (they share the original `uid` unless `ttt_n>1` regenerates) → reward standardization across the sequence, matching "rewards from the same sequence $S$ are standardized."

### Mechanism 2 — Multi-token rollout generation

`hf_rollout.py::_generate_minibatch_for_ttt`. Forces exactly $k$ tokens (no early EOS):

```python
output = self.module.generate(
    input_ids=idx, attention_mask=attention_mask,
    min_new_tokens=response_length,   # == k
    max_new_tokens=response_length,   # == k  → always exactly k tokens
    do_sample=True, temperature=ttt_temperature, top_p=1.0, top_k=0,
    use_cache=True,
)
response = seq[:, prompt_length:]                       # (B, k)
response_mask = torch.ones_like(response)               # all k positions valid
```
- Stochastic sampling (`do_sample=True`, $\tau{=}1$) — needed for RL exploration / GRPO group variance.
- Uses HF `.generate` on the FSDP-summoned full params (`rollout.name="hf"`, not vLLM) — fast-weight models aren't supported by vLLM/sglang here.

### Mechanism 3 — Predicted hidden states + sequence-level reward

`response_hidden_states` (the $\mathbf{h}^{\text{pred}}$) are re-extracted by **re-forwarding the full generated sequence** in `dp_actor.py::_forward_micro_batch_for_ttt` (called via `compute_log_prob_for_ttt`), slicing the **last-layer** hidden states at the response positions; this call also yields `old_log_probs` for GRPO.

```python
# dp_actor.py: _forward_micro_batch_for_ttt
output = self.actor_module(input_ids, attention_mask, use_cache=False, output_hidden_states=True)
logits = output.logits.div_(temperature)[:, -response_length-1:-1, :]
log_probs = logprobs_from_logits(logits, response_ids)
hidden_states = output.hidden_states[-1][:, -response_length:, :]   # last layer, response span
```

Reward — `ray_trainer.py::compute_reward_for_ttt` (Eqs. 9/11/12):

```python
if reward == "cosine_similarity":
    sim = F.cosine_similarity(response_hidden_states, ground_truth_hidden_states, dim=-1)  # R^phi
elif reward == "binary":
    sim = (ground_truth_ids == response_ids).float()                                        # R^binary
elif reward == "hybrid":
    sim = F.cosine_similarity(resp_h, gt_h, -1) + (gt_ids == resp_ids).float()              # R^hybrid
# place scalar reward on the LAST response token (outcome reward)
scores = torch.zeros_like(response_mask).float()
scores[:, -1] = sim.mean(dim=-1)    # 1/k * sum_j  (mean over rollout)
```
- **Outcome-style reward**: the per-rollout scalar (mean over $k$) is written to the **last token** of the response, all earlier response tokens get 0 → `token_level_scores`. Then `token_level_rewards = token_level_scores` (no KL penalty — `KL coeff = 0`; there's a `# TODO: KL penalty for ttt`).

### Mechanism 4 — GRPO advantage + dual SFT/PPO update

Advantage (`compute_advantage`, `adv_estimator=GRPO`):
```python
advantages, returns = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=data.batch["token_level_rewards"],
    response_mask=grpo_calculation_mask,
    index=data.non_tensor_batch["uid"],          # group = same uid (= same sequence / position)
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,  # standardize within group
)
```

Actor update — `dp_actor.py::update_policy_ppo_ttt(sft_data, ppo_data)` does **both losses in one optimizer step** with separate grad accumulation:
```python
# --- NTP (SFT) branch: standard CE on FULL input sequence, scaled by lambda_SFT ---
_,_,_, sft_loss = self._forward_micro_batch_for_input_ttt(sft_inputs, get_loss=True)  # labels=input_ids
sft_loss = sft_loss * self.config.ttt_sft_loss_coef / self.sft_gradient_accumulation
sft_loss.backward()
# --- NSP (PPO/GRPO) branch: clipped policy-gradient on rollouts, scaled by lambda_RL ---
_, log_prob = self._forward_micro_batch_for_ttt(ppo_inputs, temperature, get_hidden_states=False)
pg_loss, *_ = compute_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                                  cliprange=clip_ratio, ...)   # standard PPO clip
policy_loss = pg_loss * self.config.ttt_ppo_loss_coef / self.ppo_gradient_accumulation
policy_loss.backward()
grad_norm = self._optimizer_step()                            # one step on combined grads
```
- Confirms $\mathcal{L}=\lambda_{\text{RL}}\mathcal{L}_{\text{NSP}}+\lambda_{\text{SFT}}\mathcal{L}_{\text{NTP}}$, with the **NTP term computed over the entire sequence $S$** (`_forward_micro_batch_for_sft`/`_forward_micro_batch_for_input_ttt` use `labels=input_ids`), exactly as the paper states.
- `update_policy_sft_ttt` = SFT-only variant (when `ppo_update=False`).
- PPO uses standard verl `compute_policy_loss` (clip ratio, dual-clip $c{=}3.0$); entropy coeff 0, KL coeff 0, `ppo_epochs` reused.

### Fit-loop structure (`ray_trainer.py::fit` / `_inner_training_loop` / `_outer_training_loop` / `_validate`)
- **Inner loop** (`_inner_training_loop`): forward seq → `process_input_for_ttt` (hidden+entropy) → `prepare_ppo_batch_for_ttt` (select+truncate) → `generate_sequences_for_ttt` (rollouts) → `compute_log_prob_for_ttt` (pred hidden + old_logp) → `compute_reward_for_ttt` → GRPO advantage → `update_actor_ppo_ttt`.
- **Outer loop** (`_outer_training_loop`): standard task SFT/PPO (the nested-learning second stage in post-training).
- **TTT path**: `_validate()` with `val_only=True` calls `_inner_training_loop(validation=True)` to adapt fast weights on the test prompt, then `generate_sequences` for the final answer and scores with the task reward fn.
- **Mid-train validation** (`val_get_loss=True`): no generation — just `process_input_for_validation` → NTP loss + roll-shifted EM on Booksum (matches Fig 4 NTP-accuracy / Fig E.1 val-loss plots).

### Fast-weight model backend (LaCT) — `models/lact/ttt_operation.py`
- LaCT fast weights = **SwiGLU-MLP**: $f(x)=w_1\,(\,\text{silu}(w_0 x)\odot(w_2 x)\,)$ with fast weights $w_0,w_1,w_2$ (fp32).
- `block_causal_lact_swiglu(...)`: **chunk-wise** (default `chunk_size=2048`) online update — within each chunk, forward+backprop produce bf16 grads, then **update fast weights** (per-token learning rates `lr0,lr1,lr2`, optional **Muon** via `zeropower_via_newtonschulz5`, optional momentum, weight-norm preservation).
- This is the literal $W_{t+1}\leftarrow W_t-\eta\nabla\ell$ mechanism REFINE is improving the **initialization** of. DeltaNet (1.3B) uses flash-linear-attention's delta-rule memory state instead (params frozen, state updated).

### Data
- `data/ruler/` — RULER SQuADQA/HotpotQA parquets per model (lact / delta_net) × ctx (4096/8192/16384) × split (1600 train / 200 test).
- `data/longbench/` — 14 LongBench task parquets @16384 (12 used; MuSiQue/GovReport excluded).
- README notes Long-Data-Collections is gone → recommends **SlimPajama-6B** filtered to ≥16K tokens for mid-train.
