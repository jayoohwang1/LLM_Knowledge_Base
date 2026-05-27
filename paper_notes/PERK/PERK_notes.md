# PERK: Long-Context Reasoning as Parameter-Efficient Test-Time Learning

**Authors**: Zeming Chen, Angelika Romanou, Gail Weiss, Antoine Bosselut (EPFL) — arXiv 2507.06415, Dec 2025.

> **TL;DR**: Cast long-context reasoning as **test-time meta-learning**. At inference, a per-example **LoRA adapter** is gradient-trained on the long context with a self-supervised NLL objective (splits context into a *batch* of segments — like a parallel scratchpad). The base LLM is frozen. The whole adaptation algorithm is **meta-learned** (MAML) so the LoRA + base LLM combo learns to *quickly* encode a context such that downstream Q&A loss is minimized. Truncated gradient unrolling + LoRA make the bi-level training tractable.

---

## 1. Problem & Motivation

- **Long-context reasoning** = identify + reason over relevant info inside long, noisy contexts.
	- Even modern LLMs with 128K-1M context degrade on this.
	- Two issues exacerbate it:
		- **Distractor density**: more tokens -> more irrelevant info -> attention dilution.
		- **Multi-hop reasoning**: chained facts compound errors.
		- **Lost-in-the-middle / positional bias**: attention favors start/end.
- **Reframing**: Treat context $\mathcal{K}$ not as a prompt prefix but as something to **internalize into params** at test time.
	- Motivation: LLMs reason better with **parametric knowledge** than in-context knowledge (Roberts 2020, Bayazit 2024).
	- Compress noisy long context into params -> discard the original context -> ask questions against the parametrically-encoded knowledge.
- **Why test-time learning** specifically:
	- Process context as a **batch of shorter segments** in parallel — no need to keep the full $L$-token forward pass alive in memory.
	- Naturally side-steps positional biases because batched segments are permutation-invariant.
- **Bottleneck of prior test-time learning (Reckoning, Finn-style MAML)**: bi-level optimization through all model params is prohibitively memory-heavy → OOM at >2K tokens.

---

## 2. Background

- **CLM training objective** (token-level NLL):
	$$\mathcal{L}_{\text{NLL}}(\boldsymbol{\theta}, \mathcal{D}) = -\frac{1}{N_\mathcal{D}}\sum_{x \in \mathcal{D}}\sum_{t=1}^{\mathcal{T}} \log p_{\boldsymbol{\theta}}(x_t \mid \boldsymbol{x}_{<t})$$
- **Reasoning loss** (target answer tokens only):
	$$\mathcal{L}_{\text{reason}}(\boldsymbol{\theta}, \mathcal{D}) = -\frac{1}{N_{\mathcal{D}_y}}\sum_{r=(\mathcal{K},q,y)}\sum_t \log p_{\boldsymbol{\theta}}(y_t \mid \boldsymbol{x}(r), \boldsymbol{y}_{<t})$$
- **Reasoning as test-time learning** (Chen 2023b, Reckoning):
	- $\phi(\boldsymbol{\theta}, \mathcal{K}) = \mathcal{A}lg(\mathcal{L}_{\text{NLL}}, \mathcal{K}, \boldsymbol{\theta}, \boldsymbol{h})$ — adapt model on context first.
	- Then generate answer with $\phi$.
- **MAML** = meta-learn the *initialization* so that the inner-loop adaptation yields low outer-loop loss across tasks.
	- Standard MAML backprops through the entire inner-loop trajectory ⇒ memory $\propto N$ inner steps × full param graph.

---

## 3. PERK: Method

### 3.1 Setup

- Two parameter groups:
	- $\boldsymbol{\theta}_{\text{base}}$: the pretrained base CLM, **frozen always**.
	- $\boldsymbol{\theta}_{\text{adapter}}$: a **LoRA** adapter — this is the only thing that ever changes.
- **Inner loop** (also runs at test time):
	- Objective: self-supervised NLL of the context $\mathcal{K}$.
	- Updates only $\boldsymbol{\theta}_{\text{adapter}}$ given the frozen base.
	- The context $\mathcal{K}$ is split into a **batch of $N$ segments** (~256 tokens each), processed in parallel via gradient updates.
		- This is the trick that breaks the context-window barrier — you never run a forward pass on the full $L$-token sequence.
		- For an 8K context with 256-token segments: $N=32$ batch size.
		- For a 32K context (at inference): $N=128$.
- **Outer loop** (meta-training only):
	- After inner adaptation, feed the question $q$ to the adapted model $\phi_{\text{adapter}}((\boldsymbol{\theta}_{\text{base}}, \boldsymbol{\theta}_{\text{adapter}}), \mathcal{K})$.
	- Outer loss = $\mathcal{L}_{\text{reason}}$ on $(q, y)$. **The context $\mathcal{K}$ is NOT given to the outer step** — only the adapter encodes it.
	- $$\boldsymbol{\theta}^*_{\text{adapter}} = \arg\min_{\boldsymbol{\theta}_{\text{adapter}}} \mathbb{E}_{r \sim R}\bigl[\mathcal{L}_{\text{reason}}\bigl(\boldsymbol{\theta}_{\text{base}}, \phi_{\text{adapter}}((\boldsymbol{\theta}_{\text{base}}, \boldsymbol{\theta}_{\text{adapter}}), \mathcal{K}), \{(q,y)\}\bigr)\bigr]$$

### 3.2 Why LoRA helps the meta-learning

- Cuts trainable params drastically → smaller backward graph through inner-loop trajectory.
- $\boldsymbol{\theta}_{\text{base}}$ is never differentiated against → enormous memory savings vs. full-param Reckoning.
- LoRA acts as the **memory scratchpad** in the paper's metaphor.

### 3.3 Truncated Gradient Unrolling (TGU)

> The critical trick that makes meta-learning scale to long contexts.

- **Full unrolling (vanilla MAML)**:
	- Inner loop: $\phi^{(n+1)}_{\text{adapter}} = \phi^{(n)}_{\text{adapter}} - \alpha g^{(n)}$ for $n=0,\dots,N-1$.
	- Outer gradient: $\nabla_{\boldsymbol{\theta}_{\text{adapter}}}\mathcal{L}_{\text{reason}} = \frac{\partial \mathcal{L}_{\text{reason}}}{\partial \phi^{(N)}_{\text{adapter}}} \prod_{n=0}^{N-1} J^{(n)}$, $J^{(n)} = I - \alpha H^{(n)}$.
	- Memory ∝ $N$ — keep computation graph for **every** step.
- **TGU** (following Shaban 2019 truncated bilevel):
	- Run all $N$ inner steps for the forward pass.
	- Only **retain the computation graph for the last $T \ll N$ steps**.
	- Earlier Jacobians treated as constants (cut from product).
	- $$\nabla_{\boldsymbol{\theta}_{\text{adapter}}}\mathcal{L}_{\text{reason}} \approx \frac{\partial \mathcal{L}_{\text{reason}}}{\partial \phi^{(N)}_{\text{adapter}}} \prod_{n=N-T}^{N-1} J^{(n)}$$
	- Trade-off: slightly biased meta-gradient, massive memory win.
- **Ablations (Table 4, BabiLong-QA1 1K)**: keeping last 1-3 steps of unrolling → near-100% accuracy; truncating all (0 retention) → 92.4% (still works but drops).
	- Truncating all = essentially first-order MAML approximation — big drop, so some 2nd-order info matters.

### 3.4 Inner-Loop Algorithmic Details

- **Optimizer**: AdamW for inner loop (via differentiable `higher` library), not vanilla SGD.
- **Per-layer per-step learnable LR (LSLR)**:
	- Each LoRA param × each inner step gets its own learned LR (initialized at 5e-5).
	- Co-meta-learned alongside the LoRA params — the model learns its own optimizer schedule.
	- Borrowed from Ye & Chao 2022 ("How to train your MAML to excel").
- **Inner-loop weighted NLL** (CaMeLS, Hu 2023):
	- A small 2-layer MLP weighting head produces a per-token weight; loss = weighted CE.
	- $w$ from hidden states → softplus(beta=3) → multiply token-level CE losses → mean.
	- Weighting model params are themselves meta-learned.
	- Idea: learn *which* tokens in the context are worth memorizing harder.

---

## 4. Experiments

### 4.1 Tasks

- **NIAH (Needle-in-a-Haystack)** via **BabiLong** framework:
	- **QA1**: single-hop (one supporting fact).
	- **QA2**: 2-hop.
	- **QA3**: 3-hop.
	- Distractor text sampled from FineWeb-Edu (post-knowledge-cutoff).
	- Facts about agents moving between rooms; questions like "Where is John?".
- **Multi-Doc QA**: HotpotQA + TriviaQA, with retrieved distractor documents injected to fill context window.
- **Drops-in-the-Ocean (DIO)** — *new* benchmark proposed here:
	- **Student Records**: synthetic DB of (ID, Name, School, Major, Grade, Year).
	- Three sub-tasks:
		- **Recall**: get attribute by ID.
		- **Relation**: compare attributes of two IDs.
		- **Aggregate**: max/min/avg over all grades (many-hop).
	- Key novelty: distractors are **distributionally identical** to relevant info → much harder than synthetic NIAH where the needle looks textually distinct.
- **Lost-in-the-Middle API Retrieval** (Gorilla): used for positional-bias analysis.

### 4.2 Baselines

- **FT-ICR** = fine-tune the *same* base model on long contexts for in-context reasoning (the standard recipe).
- Trained: GPT-2-127M (1K), Qwen2.5-0.5B/7B (32K), LLaMA-3.2-1B/8B (128K), Mamba-1.4B.
- Specialized long-context LLMs: Qwen2.5-7B-Instruct-1M, ProLong-LLaMA-3.8B-512K.
- Commercial: GPT-4.1, Gemini-1.5-Pro.

### 4.3 Main Results

> All PERK / FT-ICR trained with **8K-token** training contexts; tested at 8K (in-distribution) and 32K (extrapolation).

- **NIAH (BabiLong)**:
	- PERK-Qwen-7B beats FT-ICR-Qwen-7B by **~20%** at 8K across all task complexities.
	- At 32K (extrapolation), PERK beats FT-ICR by **~23 points avg**.
	- PERK 0.5B already beats 512K-1M trained specialized LLMs (ProLong, Qwen-1M).
- **Multi-Doc HotpotQA / TriviaQA**:
	- PERK-Qwen-7B beats FT-ICR by 15% at 8K, ~14-30% extrapolating to 32K.
	- Gap to GPT-4.1 / Gemini-1.5-Pro is narrow (~3% on HotpotQA).
- **DIO (Student Records)**:
	- Performance gap PERK vs FT-ICR **widens monotonically with task hardness**: Aggregate > Relation > Recall.
		- Suggests PERK *enhances reasoning*, not just retrieval.
	- PERK-Qwen-7B matches or exceeds GPT-4.1 / Gemini-1.5-Pro on Aggregate.
- **Model family generalization (Fig 3)**:
	- BabiLong QA2 at 8K: PERK gets 91.4 / 92.4 / 96.1 / 96.8 / 99.1% for GPT-2 / Qwen-0.5B / Qwen-7B / LLaMA-1B / LLaMA-8B.
	- FT-ICR is wildly variable (18.2% → 89.8%) and **always lower**.

### 4.4 Length Extrapolation (Section 5.1, Table 1)

- Trained on **8K**, evaluated up to **128K**.
- PERK-Qwen-0.5B at 128K: **80.9%** QA1, **44.4%** QA2.
- FT-ICR-Qwen-0.5B at 128K: **0** / **0** (complete collapse).
	- Even with Yarn + Dual Chunk Attention (DCA) length-extension tricks: only 25.4 / 18.5.
- PERK beats Gemini-1.5-Pro at 128K on these tasks; competitive with GPT-4.1.
- **Why PERK extrapolates so well**: it never actually processes a long contiguous sequence at training — only batches of 256-token segments. No positional embeddings learned for positions it hasn't seen.
- Conversely, PERK trained on shorter contexts *also* generalizes to longer ones and even *increases* perf in some cases (unusual!).

### 4.5 Positional Bias Robustness (Section 5.2, Fig 5)

- Set up: relevant doc forced to **Pre / Mid / Post / Rnd** position at train/test time.
- FT-ICR: drops to ~0% when test position ≠ train position (massive overfit).
- PERK: ≤2% variation across train/test position shifts when trained with Rnd.
	- When trained with fixed position (Pre/Mid/Post), PERK still ≈consistent across test positions.
- **Mechanism**: batched segments → adapter sees the context as permutation-invariant set → absolute position info is largely washed out.
	- Caveat: PERK uses prompt prefixes `Document 1: A`, `Document 2: B`... so *implicit* ordering can still be encoded if needed.

### 4.6 Inference Efficiency (Section 5.3, Fig 6)

- Compared on H100 93GB VRAM:
	- At **8K**: FT-ICR uses ~35GB, 1.9s. PERK-Accum-1 ~35GB, 1.9s. With accumulation 16: 5.9GB, 8.5s.
	- At **64K**: FT-ICR = 55.7GB / 32.6s. PERK-Accum-16 = 19.6GB / **11.4s** (≈3× faster, much lower memory).
	- At **128K**: FT-ICR **OOMs**. PERK-Accum-8 = 35.2GB / 20.9s.
- **Key**: PERK swaps long-sequence attention cost for many short forward passes with adapter updates — and you can trade memory for time via gradient accumulation on the segment-batch.

### 4.7 Ablations (Appendix C)

- **LoRA rank** (Table 2, BabiLong 1K):
	- Rank 16: QA1/2/3 = 99.0 / 96.9 / 84.7%.
	- Rank 256: 100 / 98.1 / 89.1%.
	- Even very small adapter works — *the gradient updates themselves carry most of the info*, not the adapter's expressivity.
- **Inner-loop step count** (Table 3):
	- 2 steps: 99.2 / 94.5 / 82.4%.
	- 4 steps: 100 / 98.1 / 89.1%.
	- More steps help, but 2 is already enough to beat FT-ICR.
- **TGU truncation length** (Table 4, QA1-1K):
	- Retain 0-3 steps → ~100%. Retain 4 (no truncation) → 92.4%. Wait — actually full unrolling does *worse*? Read carefully: Table is "Truncation Length", TGU(0)=truncate 0 (retain all 4) → 100, TGU(4) = truncate all → 92.4.
	- So **truncating up to 3 steps is fine**; truncating all (≈first-order MAML) drops 8 points.
- **Training memory** (Fig 8): Reckoning OOMs at 2K; PERK fits 8K with truncation. Validates the scalability claim.

---

## 5. Important / Clever / Surprising Things

- **Reframing long-context as adaptation**: instead of "fit more tokens in the prompt", "fit context into params". Echoes the cartridges / context-tuning line of work.
- **Permutation-invariant batched context**: lossy w.r.t. global ordering but huge gains in:
	- Memory (parallelizable, never need full forward pass).
	- Positional bias (no lost-in-the-middle).
	- Length extrapolation (no long-context positional encoding needed).
	- Order is partially preserved via explicit `Document i:` prefixes.
- **Meta-learning the optimizer**: per-layer-per-step learnable LRs (LSLR) + a token-weighting MLP for the inner NLL. These two are themselves meta-parameters.
- **Truncated gradient unrolling**: lifted from bilevel-opt literature. Saves graph for last $T$ steps only — effectively a "partial second-order" MAML. Empirically you need *some* retention; full first-order is much worse.
- **The implicit role of the LoRA**: small enough to make second-order tractable, expressive enough to encode "this haystack's content". Compare to other test-time learning approaches that update full params (Reckoning) or external memory matrices (Titans / Atlas).
- **DIO benchmark** is a useful contribution: NIAH-style distractors are stylistically distinct from needles, so retrievers can cheat. DIO forces semantic-level retrieval.
- **Performance gap widens with task complexity** — argues PERK isn't *just* a better retriever but actually does compositional reasoning over the encoded knowledge.
- **Extrapolation 32× the training length** is striking — Qwen-0.5B trained on 8K still works at 128K.

---

## 6. Limitations / Open Questions

- **Training is more memory-intensive** than FT-ICR despite TGU+LoRA. Authors fit Qwen-7B on a single H100 by applying LoRA only to top-4 layers for 7B/8B models.
- **Inference latency**: each query requires running 4 inner-loop gradient steps. Fine if many queries share one context (amortized) but a one-shot QA is slower than just prompting.
	- Mitigated when context is *huge* (PERK wins for 64K+).
- **Implicit order limit**: very order-sensitive tasks (e.g., narrative comprehension with strict temporal flow across the entire doc) could suffer because each ~256-token chunk is processed independently. Mitigated only by the `Document i:` prefix trick.
- **Adapter is per-context**: cannot easily amortize one adapter across multiple contexts. Each new long doc needs a fresh inner-loop run.
- **Requires meta-training data** matching the task distribution. Generalization across task families wasn't deeply tested (the analysis shows it works across model families & scales but the inner-loop is still optimized for a particular task type — BabiLong-trained PERK probably doesn't directly handle HotpotQA without training on multi-doc QA).
- **Custom HF modeling**: requires monkey-patching Qwen to support differentiable forward (see codebase notes).
- **Hardware-bound ablation**: TGU truncation ablation only at 2K context. Hard to know how much you'd need to truncate at 64K training.

---

## 7. Key Numbers Cheat Sheet

| Setting | Model | 8K | 32K |
|---|---|---|---|
| BabiLong-QA2 (2-hop) | PERK Qwen-0.5B | 92.4 | 88.9 |
| BabiLong-QA2 | FT-ICR Qwen-0.5B | 43.8 | 34.5 |
| BabiLong-QA2 | PERK Qwen-7B | 96.1 | 84.4 |
| HotpotQA SubEM | PERK Qwen-7B | 65.2 | 44.5 |
| HotpotQA | GPT-4.1 | 72.4 | 74.3 |
| SR-Aggregate | PERK Qwen-7B | 53.3 | 49.5 |
| SR-Aggregate | Gemini-1.5-Pro | 48.7 | 50.9 |

Length extrapolation (BabiLong QA1, train@8K):

| Test Len | 32K | 64K | 128K |
|---|---|---|---|
| PERK Qwen-0.5B | 95.3 | 88.9 | 80.9 |
| FT-ICR Qwen-0.5B | 43.8 | 0 | 0 |
| FT-ICR + Yarn+DCA | 59.2 | 35.5 | 25.4 |
| Gemini-1.5-Pro | 87.7 | 82.3 | 73.1 |

---

---

# Codebase Notes

Repo entry: `run_maml.py` (Hydra-based). Core code in `perk/`. Differentiable optim via vendored `higher/`. PyTorch Lightning is the trainer.

## File Map

- `run_maml.py` — Hydra entrypoint. Sets seeds, sets output dir, calls `perk.runner.run(args)`.
- `perk/runner.py` — Builds Lightning trainer & MetaLMModule, calls `trainer.fit(model)` for both train and eval.
- `perk/modules/meta_module.py` — **The core**: training step, inner loop, outer loop, validation generation.
- `perk/modules/base_module.py` — Lightning boilerplate, dataloader hookup, LR scheduling, validation eval logging.
- `perk/model.py` — `CausalLM` wrapper around HF model + LoRA + optional `WeightingModel` MLP for weighted inner-loop NLL.
- `perk/lslr_schedular.py` — `LSLRSchedular`: per-layer-per-inner-step learnable LR (essentially `nn.ParameterDict` of LR vectors).
- `perk/dataset.py` — Dataset readers for BabiLong / Student Records / LongGorilla, collator that builds (context-batch, qa-batch).
- `perk/utils/data_utils.py` — `get_features`: splits batch into inner/outer features, optionally packs (FlashAttention varlen).
- `perk/utils/grad_utils.py` — Gradient utilities (adapter param filtering, set_grads, etc.).
- `perk/inference.py` — HF text-generation pipeline wrapper for validation.
- `perk/evaluate.py` — EM/F1 metrics, parses `<output>...</output>` blocks.
- `higher/` — Vendored Facebook `higher` library (differentiable optimizers) **with an added gradient-accumulation block** in `DifferentiableOptimizer.step()`.
- `configs/experiment/meta_train.yaml`, `meta_test.yaml` — Hydra configs (model name, LoRA rank, inner LR, etc).

## Training Loop (the meaty bit)

`MetaLMModule.training_step` → `self.step(batch)`:

```python
def step(self, batch):
    inner_opt = self.config_inner_optimizer()
    inner_batches, outer_batches, _ = get_features(
        batch, packing=self.hparams.packing, accumulation_steps=1)

    self.model.train()
    with higher.innerloop_ctx(
        self.model,
        inner_opt,
        copy_initial_weights=False,   # MAML setting — fast weights track grads through base params
        track_higher_grads=True,      # ← retain graph for outer-loop backprop
        accumulation_steps=1
    ) as (fmodel, diffopt):
        inner_loss = self.inner_funct(inner_batches, fmodel, diffopt)
        outer_loss = fmodel(outer_batches)["loss"]   # ← uses inner-adapted fast weights
    return {"loss": outer_loss, ...}
```

- `higher.innerloop_ctx` (from vendored `higher/__init__.py`) **monkey-patches** the model into a stateless `fmodel` where the params are tensors-with-grads instead of `nn.Parameter`s. Each call to `diffopt.step(loss)` returns a new set of fast weights and the patched fmodel transparently uses them on next forward.
- After the inner loop, calling `fmodel(outer_batches)` runs the LM forward through the adapted weights.
- Lightning then calls `.backward()` on `outer_loss`, which propagates through the entire retained inner-loop graph → updates `θ_adapter` (the **initial** LoRA weights) and the LSLR LRs.

### Inner Loop

Two variants in `MetaLMModule`:

**`inner_loop_step` (default)** — full unrolling:

```python
for iter in range(self.hparams.n_inner_iter):    # n_inner_iter=4
    for idx, batch in enumerate(features):
        inner_loss = fmodel(batch, is_inner=True, inner_iter=iter)["loss"]
        inner_loss /= len(features)
        if idx == len(features)-1 and self.hparams.dyna_lr:
            # update per-step learnable LRs on diffopt
            self.inner_schedular.step(diffopt, named_params, iter)
        diffopt.step(inner_loss, grad_callback=call_back)
```

**`truncated_inner_loop_step`** — TGU:

```python
unroll_start = self.hparams.unroll_start   # e.g. 2
for iter in range(self.hparams.n_inner_iter):    # 4
    for batch in features:
        inner_loss = fmodel(batch, is_inner=True, inner_iter=iter)["loss"]
        ...
        if iter < unroll_start:
            diffopt.step(inner_loss, grad_callback=grad_callback)  # ← detach grads
        else:
            diffopt.step(inner_loss)   # ← retain graph
```

- `grad_callback` (`grad_callback` fn at top of file) calls `.detach()` on every gradient produced. So the first `unroll_start` steps' updates are applied but their **gradients are detached** — the outer-loop autograd sees only the last `n_inner_iter - unroll_start` steps. Implementation of the truncation from §3.2 of the paper.

### Inner Optimizer

`config_inner_optimizer`:
- Filters params: `"lora" in n and p.requires_grad`.
- Builds `torch.optim.AdamW` param groups (with per-param `lr=self.hparams.inner_lr`, weight decay split by `["bias", "LayerNorm.weight"]`).
- `betas=(0.9, 0.95)`.
- This is wrapped by `higher.get_diff_optim` inside `innerloop_ctx` to become a `DifferentiableAdamW` whose state is part of the graph.

### LSLR (per-layer per-step LR)

```python
class LSLRSchedular(nn.Module):
    def initialization(self, params_opt):
        self.names_lr_dict = nn.ParameterDict()
        for idx, (k, param) in enumerate(params_opt):
            init_lr_group = torch.ones(self.num_inner_iter + 1) * self.init_lr
            self.names_lr_dict[k.replace(".","-")] = nn.Parameter(init_lr_group, requires_grad=True)

    def step(self, optimizer, named_parameters, step_num):
        for k, param in named_parameters:
            alpha = self.names_lr_dict[k.replace(".","-")][step_num]
            optimizer.param_groups[idx]['lr'] = alpha
```

- One `nn.Parameter` of shape `(n_inner_iter+1,)` per LoRA-tagged param tensor.
- Each inner step, it writes the current step's LR into the `diffopt`'s `param_groups`.
- These LRs are added to the outer-loop optimizer's parameter list in `configure_optimizers`:
	```python
	{"params": self.inner_schedular.parameters(), "weight_decay": 0.0}
	```
	→ they get gradient signal via the outer loss → meta-learned.

### Weighted Inner Loss

`CausalLM.weighted_loss` (in `perk/model.py`):

```python
def weighted_loss(self, outputs, features):
    weights = self.weighting_model(
        outputs["hidden_states"][-1],
        features.get("attention_mask", ...)
    )
    logits = outputs["logits"]
    loss = F.cross_entropy(logits[:,:-1,:].reshape(-1, V),
                           features["labels"][:,1:].reshape(-1),
                           reduction='none')
    loss = (loss.reshape(B, -1) * weights[:, 1:]).mean()
    return loss
```

- `WeightingModel` = 2-layer MLP: `Linear(hidden,128) -> GELU -> Linear(128,1) -> softplus(beta=3)`.
- `fc2` initialized to zero (so weights start near nonlinearity(0) = small constant → near uniform loss).
- Only applied **inside the inner loop** (`is_inner=True` flag in `CausalLM.forward`).
- The weighting model params are NOT inside the LoRA filter, so they're outer-loop parameters — they're meta-learned but **frozen during the inner loop**. So PERK learns "which tokens to weight up when memorizing" via meta-gradient.

## Data Flow

`MetaKnowledgeDataset.causal_lm_collator(qa_data)` returns one batch with:
- `input_ids / attention_mask / labels`: the **dev (outer) features** — the question + answer.
- `train_input_ids / train_attention_mask / train_labels`: the **inner features** — a stack of context segments, each formatted as `doc_{idx}: {fact}`.
- `print_out`: bookkeeping.

`get_features(batch, packing=True)`:
- Pulls out the inner/outer tensors.
- When `packing=True`: concatenates the inner batch into a single sequence using **FlashAttention varlen format** (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_*`). Saves padding tokens at the cost of needing FA2.

Each context segment gets a `doc_{idx}:` prefix → "induces ordering through explicit indexing in the prompt formatting" (paper Appx B). Effective seg length ~256 tokens; for 8K context → 32 segments → batch size 32 for the inner loop.

## Outer Optimizer Setup

`configure_optimizers` (always called by Lightning):
- Builds three param groups for the outer AdamW:
	1. Decay group: all non-bias/LayerNorm params (`requires_grad` ⇒ LoRA + weighting MLP).
	2. No-decay group: bias / LayerNorm.
	3. LSLR LRs (no decay).
- `lr=hparams.learning_rate=1e-5`, betas=(0.9, 0.95).
- Cosine LR schedule with warmup proportion 0.03 (max 8000 warmup steps).

## Validation / Inference

`MetaLMModule.validation_step`:

```python
torch.set_grad_enabled(True)    # ← inner loop needs grad even at eval
inner_opt = self.config_inner_optimizer()
with higher.innerloop_ctx(
    self.model, inner_opt,
    copy_initial_weights=False,
    track_higher_grads=False,    # ← NO second-order at test time
    accumulation_steps=inner_grad_accum   # e.g. 4 to save memory
) as (fmodel, diffopt):
    inner_loss = self.inner_loop_step(inner_batches, fmodel, diffopt)
    with torch.no_grad():
        fmodel.eval()
        outer_loss = fmodel(outer_batches)["loss"]
        self.validation_inference(records, fmodel)   # greedy decoding
```

- At eval: same inner loop as training, but `track_higher_grads=False` (no need to backprop) and gradient accumulation enabled across inner-batch segments to reduce memory.
- Generation via standard HF `pipeline("text-generation", ...)` on the adapted model.
- Output format: `<support>...</support>\n<output>ANSWER</output>` — eval regex pulls `<output>` for EM scoring.

## `higher` Library — One Important Patch

Standard `higher.optim.DifferentiableOptimizer.step` was extended for **gradient accumulation across inner-batch segments**:

```python
if accumulation_steps > 1:
    if not hasattr(self, "_accumulated_grads") or self._accumulated_grads is None:
        self._accumulated_grads = [None] * len(all_grads)
        self._accum_counter = 0
    for i, g in enumerate(all_grads):
        if g is not None:
            self._accumulated_grads[i] = g if self._accumulated_grads[i] is None \
                                           else self._accumulated_grads[i] + g
    self._accum_counter += 1
    if self._accum_counter < accumulation_steps:
        return params    # ← skip the actual update
    all_grads = self._accumulated_grads
    ...
```

- Lets you process a context-segment batch of size $N$ in $N / k$ micro-batches while still doing one "logical" inner step.
- Used at eval time (`inner_grad_accum=4` in `meta_test.yaml`) to fit larger contexts.

## Hyperparameters (`meta_train.yaml`)

| Param | Value |
|---|---|
| model_name_or_path | Qwen/Qwen2.5-0.5B (other configs: GPT-2, LLaMA) |
| lora_rank | 256 (LoRA on all q/k/v/o/gate/up/down projections; only top-4 layers for ≥7B models per paper Appx B) |
| lora_alpha | 16 (use_rslora=True ⇒ effective α scales to rank) |
| lora_dropout | 0.1 |
| outer learning_rate | 1e-5 (AdamW, betas (0.9, 0.95)) |
| inner_lr | 5e-5 (initial; LSLR learns per-layer per-step) |
| n_inner_iter | 4 |
| inner_funct | "truncated" (TGU); test config uses "default" |
| unroll_start | 2 (retain last 4-2=2 steps' graph) |
| weighted_loss | true (with softplus non-linearity, β=3) |
| packing | true (FlashAttention varlen) |
| bf16 | true; attn_implementation = flash_attention_2 |
| warmup_proportion | 0.03 |
| weight_decay | 0.01 |
| max_grad_norm | 1.0 |
| effective inner-segment len | ~256 tokens (paper Appx B) |
| 8K context inner batch size | 32 = 8192 / 256 |

## Implementation Gotchas / Tricks

- **Monkey-patch Qwen for differentiability**: Per README, the HF Qwen2 model's loop `for decoder_layer in self.layers[:n]:` needs to be rewritten as `for i in range(n): decoder_layer = self.layers[i]` because the `higher` patched model doesn't survive list iteration of `nn.ModuleList`. Subtle and important.
- **Liger kernel** (Qwen RMSNorm + RoPE + SwiGLU fused kernels) is optionally applied for speed — disabled by default (`use_liger_kernel: false`) in shipping config.
- **First-order MAML option**: `first_order: false` in config; if true, `grad_callback` detaches *all* gradients → falls back to FOMAML. Paper Table 4 shows this hurts (~8 pt drop on QA1-1K).
- **`copy_initial_weights=False`** in `innerloop_ctx`: standard MAML setting — means the *original* LoRA params are part of the graph (so meta-gradient flows back to the initial LoRA, not just the trajectory).
- **Random fact shuffling**: in `causal_lm_collator`, for `data_type=="train"`, the doc IDs are shuffled before formatting → order augmentation during training.
- **Single-GPU only**: training is single-H100 (`n_gpu: 1` default). DDP code exists but inner loop with `higher` + DDP is fragile.
- **`weighted_loss` only inside inner loop**: outer loss uses plain CE on answer tokens. Weighting is purely about *how to memorize* the context.
- **No KV-cache trick**: inference still recomputes fresh through the LoRA-adapted model each query, but the *long context* never enters the prompt — only the question does. So the prompt at generation time is just `You have memorized N documents.\n{question}\nAnswer:` plus the adapted weights.
