# Recursive Multi-Agent Systems (RecursiveMAS)

> Yang, Zou, Pan, Qiu, Lu, Diao, Jiang, Tong, Zhang, Buehler, He, Zou. arXiv:2604.25917 (Apr 2026).
> UIUC / Stanford / NVIDIA / MIT. Project: https://recursivemas.github.io

---

## Summary

- **Core idea**: cast an entire LLM **multi-agent system (MAS)** as a single **recursive computation in latent space**, analogous to a **recursive language model (RLM)** where each agent plays the role of an "RLM layer".
- **Key module**: **RecursiveLink** $\mathcal{R}$, a tiny 2-layer residual MLP connecting agents.
  - **Inner link** $\mathcal{R}_{\text{in}}(h) = h + W_2 \sigma(W_1 h)$ – passes a model's last-layer hidden state back to its own input embedding for **latent thought** auto-regression.
  - **Outer link** $\mathcal{R}_{\text{out}}(h) = W_3 h + W_2 \sigma(W_1 h)$ – bridges hidden states between **heterogeneous** agents of different sizes/dims; $W_3$ adapts dim.
- **Chain-of-agents loop**: agents pass latents through the inner→outer→next-agent's-inner pipeline; after the last agent the final latent is **fed back to the first agent** ($\mathcal{S}^{(0)} \to \mathcal{S}^{(1)} \to \dots \to \mathcal{S}^{(n)}$). Only the final round decodes text.
- **Inner-Outer Loop training**: (i) warm-start each inner link with **cosine regression** to align latent thoughts with input-embedding semantics of ground truth, (ii) co-optimize all outer links system-wide with **CE loss** through fully unrolled $n$-round forward, gradients backprop across all recursion rounds. **Base LLMs are frozen** — only RecursiveLinks update.
- **Theory**:
  - **Runtime**: $\Theta(N(md_h^2 + (t+m)d_h^2 + (t+m)^2 d_h))$ vs text-MAS's $\Theta(N(m|V|d_h + \dots))$ — replaces per-step vocab decoding $m|V|d_h$ with cheap latent transform $md_h^2$ since $d_h \ll |V|$.
  - **Gradient stability** (Thm 4.1): text-SFT $\mathcal{R}_{\text{text}}(h) = W_{\text{in}} \text{softmax}(W_{\text{out}} h)$ has $\|\partial \mathcal{R}_{\text{text}}/\partial h\| \le O(\epsilon)$ (**vanishing** because confident tokens have $\epsilon \ll 1$); RecursiveLink keeps gradients near 1 via the identity residual.
- **Results**: across 9 benchmarks (MATH500, AIME2025/26, GPQA-D, MedQA, LiveCodeBench-v6, MBPP+, HotpotQA, Bamboogle), under 4 collaboration patterns:
  - **+8.3%** avg accuracy over strongest baseline.
  - **1.2×–2.4×** end-to-end inference speedup.
  - **34.6%–75.6%** fewer tokens.
  - Performance gains and efficiency gains **grow with recursion depth**.

---

## Motivation / Problem

- **Single LLMs** hit capacity / myopic-generation walls on complex tasks.
- **MAS** approaches scale via heterogeneous specialist agents but:
  - **Prompt-based adaptation** (TextGrad, etc.) — agents themselves don't improve.
  - **Per-agent fine-tuning** (MALT, MultiAgent-FT) — hard to train all agents; **sequential text I/O has huge latency** waiting for prior agent to finish decoding.
- **Recursive language models (RLMs)** (LoopLM, Mixture-of-Recursions, Tiny Recursive Networks) iterate a shared Transformer stack over hidden states $H^{(r)} = f_\theta(H^{(r-1)})$ to deepen reasoning.
- **Question**: *Can agent collaboration itself be scaled through recursion?* Extend the RLM scaling axis from a **single model** to a **whole MAS**.
- Why latent rather than text-mediated:
  - Avoid expensive decoding ($|V| \gg d_h$) at each agent boundary.
  - Avoid the gradient vanishing that hits text-based SFT when the token distribution becomes confident (small entropy → $S = \text{diag}(p) - pp^T$ has tiny spectral norm).

---

## Preliminaries

- **Latent-space auto-regression**: instead of $h_t \to \text{argmax}\,W_{\text{out}} h_t$ then re-embedding, directly feed $h_t$ as next input embedding: $h_{t+1} = f_\theta([E_{\le t}; h_t])$. The $h_{t+1}$ is the *latent thought*.
- **Recursive computation**: $H^{(0)} = E$, $H^{(r)} = f_\theta(H^{(r-1)})$ for $r=1..n$. Same Transformer stack reused.
- **MAS**: agents $\mathcal{A} = \{A_1,\dots,A_N\}$ each $\equiv f_{\theta_i}$ with hidden states $H_i$; system state $\mathcal{H} = \{H_1,\dots,H_N\}$.
- **Recursive MAS evolution** (Def 2.1): $\mathcal{S}^{(0)} \xrightarrow{H^{(1)}} \mathcal{S}^{(1)} \xrightarrow{H^{(2)}} \dots \xrightarrow{H^{(n)}} \mathcal{S}^{(n)}$, each $\mathcal{S}^{(r)}$ refines latent rep of the system.

---

## Method

### 3.1 RecursiveLink $\mathcal{R}$

- Two-layer residual MLP with GELU.
- **Inner link** (single agent, dense→shallow transition):
  - $\mathcal{R}_{\text{in}}(h) = h + W_2 \sigma(W_1 h)$
  - $W_1, W_2 \in \mathbb{R}^{d_h \times d_h}$
  - Plugged into the recurrence to remap last-hidden $h$ back to input embedding space of the **same** agent.
- **Outer link** (cross-model transition):
  - $\mathcal{R}_{\text{out}}(h) = W_3 h + W_2 \sigma(W_1 h)$
  - $W_3 \in \mathbb{R}^{d_{h_j} \times d_{h_i}}$ in residual branch handles **heterogeneous hidden dims** ($A_i \to A_j$).
- **Why residual**: identity branch preserves original semantics; network only learns the **distributional shift** between agents' embedding manifolds — easier and gradient-stable.
- **Empirical ablation** (Table 4) on Math500/GPQA-D/LiveCodeBench:
  - 1-Layer: 84.4 / 63.2 / 40.1
  - Res+1-Layer: 86.7 / 65.3 / 41.4
  - 2-Layer (no res): 85.6 / 64.5 / 40.5
  - **Res+2-Layer (used)**: **88.0 / 66.2 / 42.9**

### 3.2 Chain agents into a recursive loop

- **Per-agent latent thought generation**: given input embeds $E_{A_1}$, agent $A_1$ runs the Transformer to get $h_t$, applies $\mathcal{R}_{\text{in}}$ to get $e_{t+1}$, auto-regresses for $m$ latent steps yielding $H_{A_1} = [h_t, \dots, h_{t+m}]$.
- **Cross-agent transfer**: $A_2$ consumes $\mathcal{R}_{\text{out}}(H_{A_1}) \oplus E_{A_2}$ then itself produces $H_{A_2}$ via $\mathcal{R}_{\text{in}}$. Repeat through $A_N$.
- **Closing the loop**: after $A_N$, latents fed back to $A_1$ via inner-outer composition → next recursion round. Only **last round of $A_N$ decodes text**; all intermediate rounds stay in latent space.
- **Runtime advantage (Prop 3.1)**: per agent $\Theta(md_h^2 + (t+m)d_h^2 + (t+m)^2 d_h)$ vs text-MAS $\Theta(m|V|d_h + (t+m)d_h^2 + (t+m)^2 d_h)$. Vocab term replaced by latent transform.

### 4. Inner-Outer Loop training

- **Two stages**, all base LLM params **frozen**, only RecursiveLinks updated.

**(a) Inner-loop (per-agent warm-start)**:
- For each agent $A_i$ with training pair $(x, y)$:
  - $\mathcal{L}_{\text{in}} = 1 - \cos(\mathcal{R}_{\text{in}}(H), \text{Emb}_{\theta_i}(y))$.
  - $H$ = last-layer latents generated by $A_i$ on $x$.
- Encourages inner link to make latent thoughts land in semantic neighborhood of ground-truth target embeddings → skips the explicit decode/re-encode loop.
- Done **in parallel for all $N$ agents**.

**(b) Outer-loop (system-wide CE)**:
- Unroll the chain $\mathcal{S}^{(r)}$ for $r=1..n$ recursion rounds — full computation graph preserved.
- Decode final answer at round $n$, compute CE:
  - $\mathcal{L}_{\text{out}} = \text{CE}(\mathcal{S}^{(n)}(\dots\mathcal{S}^{(1)}(x)), y)$.
- Gradients flow through **all rounds**, shared credit signal across all outer links.

**Gradient stability (Thm 4.1)**:
- Text path: $J_{\text{text}} = W_{\text{in}} S W_{\text{out}}$ with $S = \text{diag}(p) - pp^T$. Confident output → $\|p\|_2 \approx 1$ → $\|S\|_2 \le 1 - \|p\|_2^2 \le \epsilon$ (using $\epsilon \ge \sum p_i(1-p_i)$).
- RecursiveLink path: $J = I + W_2 \Sigma' W_1$. Triangle inequality + Kaiming init concentration: $\|J\|_2 \ge \Omega(1 - \sqrt{(1/d_h)\log(1/\delta)}) \approx 1$ w.h.p.
- **Conclusion**: text-mediated recursion suffers gradient vanishing; latent-mediated recursion has stable near-unit-norm gradients.

---

## Collaboration Patterns (4 instantiations)

| Pattern | Roles | Latent flow |
|---|---|---|
| **Sequential** | Planner → Critic → Solver | chain; solver feeds back to planner |
| **Mixture** | {Math, Code, Science} → Summarizer | parallel specialists → summarizer; summarizer fans back out |
| **Distillation** | Expert ↔ Learner | bigger Expert guides smaller Learner; Learner decodes |
| **Deliberation** | Reflector ↔ Tool-Caller | iterative critique; Tool-Caller has Python + web search (Tavily) and produces final |

- **Model assignments** (Table 1):
  - Sequential-Light: Qwen3-1.7B Planner / Llama3.2-1B Critic / Qwen2.5-Math-1.5B Solver.
  - Sequential-Scaled: Gemma3-4B / Llama3.2-3B / Qwen3.5-4B.
  - Mixture: Qwen2.5-Coder-3B / BioMistral-7B / DeepSeek-R1-Distill-Qwen-1.5B / Qwen3.5-2B Summarizer.
  - Distillation: Expert Qwen3.5-9B, Learner Qwen3.5-4B.
  - Deliberation: 2× Qwen3.5-4B (one with tool integration).

---

## Experiments / Results

### Benchmarks
- **Math**: MATH500, AIME2025, AIME2026 (Pass@10 for AIME).
- **Sci/Med**: GPQA-Diamond, MedQA.
- **Code**: LiveCodeBench-v6, MBPP+.
- **Search QA**: HotpotQA, Bamboogle.

### Training data
- Curated: s1K (math), m1K (med/sci), OpenCodeReasoning (code), ARPO-SFT (tool-aug).
- Role-specific supervision targets built by rewriting answers per role (e.g. Qwen3.5-397B-A17B rewrites into plan / critique / solution for sequential).
- **AdamW**, **lr 5e-4**, **cosine schedule**, **batch size 4**, **max seq 4096**.

### Headline numbers

**RecursiveMAS vs Recursive-TextMAS** (same chain, text vs latent), at $r=3$ (Table 2):
- Math500: 77.8 vs 69.1 acc; 1360s vs 2952s; 519 vs 3059 tokens.
- AIME2025: 34.0 vs 18.0.
- GPQA-D: 32.6 vs 28.7.
- MedQA: 79.3 vs 77.1.
- LiveCodeBench: 31.7 vs 29.3.
- Overall improvement **+7.2%**, **×2.4** speedup, **−75.6%** tokens.

**Broader baselines at $r=3$ scaled** (Table 3):
- Single-Agent w/ Full-SFT: 83.2 / 73.3 / 76.7 / 62.8 / 38.6 / 77.0.
- TextGrad: 84.9 / 73.3 / 76.7 / 62.5 / 39.8 / 77.2.
- LoopLM: 84.6 / 66.7 / 63.3 / 48.1 / 24.9 / 56.4.
- **RecursiveMAS: 88.0 / 86.7 / 86.7 / 66.2 / 42.9 / 79.3**.
- Biggest gaps on reasoning-heavy AIME2025/26 (+18.1 / +13.0 over TextGrad).

### Scaling law (Fig 1 top)
- Joint train-/inference-recursion sweep $r \in \{1,2,3,4\}$.
- Deeper **training** rounds shift entire perf frontier up.
- Deeper **inference** continues to help models trained with fewer rounds.
- Strongest results in upper-right (both deep).
- Suggests training recursion teaches "refinement-ready" latent states that inference recursion can exploit at test time.

### Generalization across patterns (Fig 1 bottom)
- Mixture: +6.2% over best specialist on each bench.
- Distillation: Learner +8.0% with **1.5× speedup vs Expert** (effective knowledge distillation).
- Deliberation: +4.8% over tool-calling-only baseline.

### Semantic distribution analysis (Fig 7)
- PCA of solver-input-embedding-mapped final answers vs ground-truth on 500 questions.
- Round 1: noticeable shift. Round 3: distributions largely overlap. **Iterative latent refinement progressively aligns with target manifold**.

### Latent length ablation (Table 9 / Fig 8)
- Sweep $m \in \{0,16,\dots,128\}$.
- Math500: 83.3 ($m=0$) → 86.8 ($m=64$) → 86.7 ($m=128$). Saturates around $m \approx 80$.
- LiveCodeBench more sensitive to $m$. Modest latent budget suffices vs long CoT.

### Cost analysis (Table 5)
- Per-agent peak GPU mem: 15.29 GB (RecursiveMAS) vs 41.40 (Full-SFT) vs 21.67 (LoRA).
- Trainable params: 13.12M (**0.31%**) — even smaller than 15.92M LoRA.
- Estimated cost: $4.27 vs $9.67 (Full-SFT) vs $6.64 (LoRA).
- Avg acc: **74.9** vs 68.6 (Full-SFT) vs 66.9 (LoRA).

---

## Implementation

### Repo structure

```
RecursiveMAS/
├── run.py                        # unified entry; per-style CLI dispatch
├── load_from_repo.py             # STYLE_SPECS → HF repo IDs per style
├── hf_resolver.py                # downloads & resolves HF checkpoints/adapters
├── system_loader.py              # high-level API to load full MAS (LoadedMASSystem)
├── modeling.py                   # Adapter, CrossModelAdapter (inner/outer links)
├── prompts.py                    # per-role prompt builders + slot tokens
├── dataset/medqa.json
└── inference_utils/
    ├── inference_mas.py             # sequential chain inference (main impl)
    ├── inference_mas_mixture.py     # mixture (hie) inference
    ├── inference_mas_distill.py     # distillation
    ├── inference_mas_deliberation.py# deliberation + tool calls
    ├── reflector_tool_notes.py      # tool-caller system prompt
    ├── answer_utils.py              # answer extraction / normalization
    └── lcb_utils.py                 # LiveCodeBench / MBPP+ eval
```

### Key components

#### RecursiveLink modules — `modeling.py`

- **Inner link** = `Adapter` class (`modeling.py:121-136`):
  - **Architecture**: `pre_ln → proj1 → GELU → proj2 → (+x) → post_ln`.
  - Note: the released form has **LayerNorms wrapping** the 2-layer residual MLP (called `ln_res_adapter`), slightly richer than the bare paper equation $h + W_2\sigma(W_1 h)$.
  - All projections square: $d_h \to d_h \to d_h$.
  ```python
  def forward(self, x):
      h = self.pre_ln(x)
      out = self.proj2(self.act(self.proj1(h)))
      out = x + out
      return self.post_ln(out)
  ```

- **Outer link** = `CrossModelAdapter` (`modeling.py:139-159`):
  - **Architecture**: source-LN → `proj1` ($d_{\text{in}} \to 2 d_{\text{out}}$) → GELU → `proj2` ($2d_{\text{out}} \to d_{\text{out}}$) + `residual_proj` ($d_{\text{in}} \to d_{\text{out}}$, the paper's $W_3$) → target-LN.
  - **Hidden dim = $2 d_{\text{out}}$** (wider middle than inner link).
  - Adapter type tag: `outer_ln_res_adapter`.

- **Adapter type guards** (`modeling.py:15-49`): infer adapter type from state_dict keys; release code **only supports the LN-residual variant** (the original alternatives from the ablation are not in this release).

#### System loading — `system_loader.py`

- **`load_mas_system(style, dataset, device, dtype, …)`** (`system_loader.py:161-234`):
  - Resolves HF snapshots via `_materialize_repo` → `snapshot_repo`.
  - Per family (`sequential`/`mixture`/`distillation`/`deliberation`), loads:
    - **`_AGENT_LAYOUTS`**: role → repo key (lines 56-76).
    - **`_OUTER_LAYOUTS`**: outer link name → (src_role, dst_role) (lines 78-100). E.g. sequential has `outer_12 (planner→critic)`, `outer_23 (critic→solver)`, `outer_31 (solver→planner)` — the **31 closes the recursive loop**.
  - For each role: load `(model, tokenizer)` + inner adapter via `inference_mas.load_inner_adapter_module`.
  - For each outer link: probe `out_dim` from state-dict (`infer_outer_adapter_out_dim_from_file`), check it matches target agent's hidden, then load.
- **`STYLE_SPECS`** in `load_from_repo.py:5-50` enumerates the 5 released configurations and their HF repo IDs.

#### Inference pipelines — `inference_utils/inference_mas.py`

Three core primitives (`inference_mas.py:845-1419` …):

- **`autoregressive_latent_rollout`** (`inference_mas.py:845-886`):
  - The latent-space generation loop. For `latent_steps` iterations:
    1. Forward Transformer with current `inputs_embeds` + mask.
    2. Take last-token last-layer hidden state.
    3. Apply inner adapter → new embedding.
    4. Append new embedding + extend attention mask.
  - Returns the stack of `latent_steps` hidden states (the "latent thoughts" $H_{A_i}$).
  - Uses `logits_to_keep=1` to avoid full vocab logits (OOM-safe on long prompts).
  - `@torch.no_grad()` since RecursiveLinks are pre-trained.

- **Per-stage runners**: `run_planner_latent_stage`, `run_refiner_latent_stage`, `run_solver_feedback_latent_stage`, `run_planner_feedback_latent_stage`, `run_solver_latent_stage` (sequential chain).
  - **Pattern**: load model+tokenizer → load inner+outer adapter → for each batch:
    - Construct embeds (either from token IDs, or concat `prefix_embeds + previous_agent_latent + suffix_embeds` for downstream agents).
    - Run `autoregressive_latent_rollout` to get this agent's $H$.
    - Apply inner adapter again on the rollout, then outer adapter to map into next agent's embedding space.
    - Release model+adapter (each stage frees GPU before next loads — supports lightweight 1.5–4B agents on single GPU).
  - Prompts contain **slot tokens** (`PLANNER_SLOT = "<<LATENT_PLANNER_SLOT>>"`, `REFINED_SLOT`, `FEEDBACK_SLOT`) split via `split_prompt_ids_by_slots` so the latent tensor can be inserted at the right position in the input-embeds sequence.

- **Adapter loaders**:
  - `load_inner_adapter_module` (`inference_mas.py:683-712`): loads `adapter.pt` + `adapter_config.json` from each agent's repo.
  - `load_outer_adapter_module` (`inference_mas.py:757-782`): probes output dim from weights, builds `CrossModelAdapter`, loads strict.

#### Prompts / Slots — `prompts.py`

- **Slot tokens** are special markers in the user prompt that the latent embeddings replace at runtime:
  - Sequential: `PLANNER_SLOT`, `REFINED_SLOT`, `FEEDBACK_SLOT`.
  - Mixture: `HIE_{MATH,CODE,SCIENCE}_EXPERT_SLOT`, `HIE_FEEDBACK_SLOT`.
  - Distillation: `DISTILL_EXPERT_SLOT`, `DISTILL_FEEDBACK_SLOT`.
  - Deliberation: `DELIBERATION_REFLECTOR_SLOT`, `DELIBERATION_FEEDBACK_SLOT`.
- The runner tokenizes the prompt around the slot string, embeds the prefix/suffix, then `torch.cat([prefix_embeds, latent_tensor, suffix_embeds])` to form the actual input.

### Training / Inference flow

- **Inference loop** (per recursion round $r$ in a sequential chain):
  1. **Round $r=1$**: planner reads `build_math_planner_prompt(question)` (no feedback slot) → rollout latents → outer_12 → critic.
  2. **Rounds $r>1$**: planner uses `build_math_planner_prompt_with_feedback_slot`; feedback latent from previous round's solver inserted at `FEEDBACK_SLOT`.
  3. Critic / refiner consumes planner latents at `PLANNER_SLOT` → rollout → outer_23 → solver.
  4. Solver: if not final round, runs `run_solver_feedback_latent_stage` (latent-only) → outer_31 → back to planner. If final round, runs `run_solver_latent_stage` which actually **decodes text** for the final answer.
- The `run.py` dispatcher sweeps **`LATENT_STEPS_SWEEP = (16, 32, 48)`** when not given a fixed `--latent_length`, reporting best.
- Training (paper §4, not in release): inner-loop pretrains each `Adapter` with cosine regression; outer-loop unrolls full $n$-round forward and backprops CE through all outer adapters jointly — base LLMs stay frozen.

### Configs / Hyperparameters

- **Release recommended per (style, dataset)** (`inference_mas.py:92-101`):
  | Style | Dataset | seed | bs | latent_len |
  |---|---|---|---|---|
  | sequential_light | math500 | 42 | 32 | 48 |
  | sequential_light | medqa/gpqa | 42 | 16 | 32 |
  | sequential_light | mbppplus | 42 | 16 | 16 |
  | sequential_scaled | math500 | 42 | 16 | 32 |
  | sequential_scaled | medqa/gpqa | 42 | 16 | 48 |
  | sequential_scaled | mbppplus | 42 | 16 | 16 |
- **Decoding** (`run.py`):
  - Default `--temperature 0.6`, `--top_p 0.95`.
  - **MBPP+ uses temperature 0.2** (more deterministic for code).
  - `--num_recursive_rounds 3` (matches paper main results).
  - `max_new_tokens`: 1000 for sequential_light/math500, 2000 for sequential_scaled/math500, 4000 elsewhere (much smaller than the paper's training-time 16000 for AIME).
  - `enable_thinking=0` (chat template's "thinking mode" disabled).
- **Adapter types in release** (only one variant supported):
  - Inner: `ln_res_adapter` (LN-wrapped 2-layer residual).
  - Outer: `outer_ln_res_adapter`.
- **Deliberation tool config** (`run.py:315-318`): `max_tool_rounds=5`, `python_timeout=10s`, `result_max_chars=6000`, Tavily search via `TAVILY_API_KEY` env.

### Things to note in the released code

- **Only inference** is released (the README explicitly marks "Complete Inference Pipeline" and "Training Data & Implementation Details" as TODO ☑️).
- **Per-stage model swap**: each stage loads → runs → calls `release_resources` to free GPU before the next agent loads. Enables a multi-agent system to run with the memory footprint of the **largest single agent**, not the sum.
- **Heterogeneous tokenizers** are handled implicitly: each agent's prompt is built and tokenized with that agent's own tokenizer; only the **latent hidden states** crosses agent boundaries (mapped by outer link).
- **`_maybe_build_plain_model_view`** (`modeling.py:52-96`): released repos colocate adapter files with the full base model weights. Recent `transformers` may treat the dir as a PEFT adapter repo and fail; this builds a symlink view excluding adapter files so AutoModel can load cleanly.
- The released **`Adapter` is richer than the paper's eq. (3)**: includes both `pre_ln` and `post_ln` LayerNorms around the residual MLP. The paper formula $h + W_2\sigma(W_1 h)$ is the conceptual core; release adds normalization for stability.
