# Latent Collaboration in Multi-Agent Systems (LatentMAS)

> Zou, Yang, Qiu et al. (Princeton / UIUC / Stanford), arXiv:2511.20639, Dec 2025. ICML 2026 spotlight.

---

## Summary

- **TL;DR**: Multi-agent LLM system where agents reason and communicate **entirely in continuous latent space** instead of text. Each agent does **auto-regressive latent thought generation** (feeding last-layer hidden states back as next input embeddings), and agents exchange info by **prepending the producer's layer-wise KV-cache** to the consumer's KV-cache (a "latent working memory" transfer).
- **Training-free**. Works on any decoder-only transformer (Qwen3 demoed). The only non-trivial component is a small **linear realignment matrix** $W_a \in \mathbb{R}^{d_h \times d_h}$ that maps output hidden states back into the input-embedding space.
- **Two MAS topologies tested**: *sequential* (planner $\to$ critic $\to$ refiner $\to$ judger/solver) and *hierarchical* (math + science + code experts $\to$ task summarizer).
- **Headline numbers vs Text-MAS** (Qwen3 4B/8B/14B, 9 benchmarks): +13.3 % avg accuracy, 70.8–83.7 % fewer tokens, 4–4.3$\times$ faster end-to-end (even vs vLLM-accelerated TextMAS).
- **Three theoretical claims**: (T3.1) latent thoughts $\Omega(d_h m / \log|\mathcal{V}|)$ more expressive per step than text; (T3.3) KV-cache transfer is information-equivalent to feeding raw input tokens; (T3.4) lower complexity than text MAS for equal expressiveness.

---

## Motivation / Problem

- **Text as the lingua franca of MAS is wasteful**:
    - Each agent re-tokenizes the previous agent's natural-language output $\to$ lossy encode/decode, error compounding (case study on GSM8K shows planner errors propagating through critic and refiner).
    - Decoding to text is the bottleneck: every reasoning step is $O(d_h|\mathcal{V}|)$ softmax + sampling.
    - Token-by-token chain-of-thought collapses a high-dim continuous state into a discrete sample.
- **Prior latent work is partial**:
    - **CoCoNuT (Hao et al. 2024)**, RepE, soft-thinking $\to$ single-model latent CoT.
    - **Cache-to-Cache (Fu et al. 2025)**, **KVComm**, **ThoughtComm** $\to$ inter-model exchange but limited to prefilled input KV, not generated thoughts.
- **Question of paper**: *"Can multi-agent systems achieve pure latent collaboration?"* — i.e. unify intra-agent latent reasoning **and** inter-agent latent communication, training-free.

---

## Method

### Setup / notation

- Transformer $f_\theta$: input embed $W_\text{in} \in \mathbb{R}^{|\mathcal{V}| \times d_h}$, output head $W_\text{out} \in \mathbb{R}^{d_h \times |\mathcal{V}|}$, $L$ layers.
- Per-layer KV cache $(K_\text{cache}^{(l)}, V_\text{cache}^{(l)})$ acts as "working memory".
- MAS $\mathcal{S} = \{A_1, \dots, A_N\}$, each $A_i$ same $f_\theta$ but different system/role prompt.

### 1. Auto-regressive latent thoughts generation (inside each agent)

- Standard prefill on agent's question $q$ + role prompt yields input embeddings $E = [e_1, \dots, e_t]$.
- Forward pass $\to$ last-layer hidden $h_t$.
- **Instead of decoding** $h_t$ to a token: feed $h_t$ (after alignment $\to$ see below) **directly as the next input embedding** $e_{t+1}$.
- Repeat for $m$ "latent steps", yielding $H = [h_{t+1}, \dots, h_{t+m}]$ - the **latent thoughts** of agent $A_i$.

#### Input-Output distribution alignment ($W_a$)

- **Problem**: $h_t$ lives in the output residual stream; shallow layers expect inputs drawn from $W_\text{in}$'s row-space. Direct feedback $\to$ OOD activations.
- **Fix**: learn (closed-form) a small **$d_h \times d_h$ realignment matrix** $W_a$ once at init, set $e = h W_a$.
    - Objective: $\min_{W_a} \|W_\text{out} W_a - W_\text{in}\|_F^2$ — i.e. force $W_\text{out} W_a \approx W_\text{in}$ so $\hat{e} = h W_a$ behaves like a real token embedding.
    - **Closed form** (ridge-regularized): $W_a = (W_\text{out}^\top W_\text{out} + \lambda I)^{-1} W_\text{out}^\top W_\text{in}$.
    - **Theoretical justification (Thm A.1)**: $W_a$ minimizes an *upper bound* on the Wasserstein distance between the aligned-embedding distribution and the true token-embedding distribution — bound is $\|W_\text{out} W_a - W_\text{in}\|_F$.
- Computed **once per run**, reused across all latent steps & all agents.
- In code: also re-normalizes aligned vector to match mean $\|W_\text{in}\|$ norm.

#### Expressiveness theorem (Thm 3.1)

- Under the **Linear Representation Hypothesis** (hidden $h$ = $\sum_i c_i s_i$ with $c_i \in \{0, \pm 1\}$ over $d_h$ semantic bases), there are $3^{d_h}$ possible hidden states $\to$ $3^{d_h m}$ length-$m$ latent trajectories.
- To losslessly represent these in text needs $m' \geq \Omega(d_h m / \log |\mathcal{V}|)$ tokens.
- **Numerically**: Qwen3-4B/8B/14B latent thoughts are $235.7\times / 377.1\times / 471.4\times$ more efficient than text reasoning. Expressiveness scales **linearly in $d_h$** $\to$ bigger models inherently have richer latent reasoning capacity.

### 2. Latent working memory transfer (across agents)

- After $A_1$ runs $m$ latent steps, extract its full per-layer KV-cache:
$$\mathcal{M}_{A_1} = \{(K_{A_1,\text{cache}}^{(l)}, V_{A_1,\text{cache}}^{(l)}) \mid l=1,\dots,L\}$$
    - Crucially, this contains **both prefilled context AND newly-generated latent thoughts** (unlike Cache-to-Cache which only ships input KV).
- $A_2$ initializes its KV by **layer-wise prepending** $A_1$'s cache to its own:
    - For each layer $l$: $K_{A_2}^{(l)} \leftarrow [K_{A_1}^{(l)}; K_{A_2}^{(l)}]$, same for $V$.
- $A_2$ now generates its own latent thoughts **conditioned on the prepended cache**.
- Repeat chain across all $N$ agents. Only the **last agent decodes to text** (the final answer).

#### Lossless info preservation (Thm 3.3)

- Claim: feeding $A_2$ the prepended KV-cache from $A_1$ gives **identical** layer-wise hidden states as if $A_2$ had re-encoded $A_1$'s raw input tokens.
- Proof sketch: by induction on layers. For each layer the KV cache contains exactly the same keys/values the model would compute by reading the tokens; attention output is deterministic given identical $Q$/$K$/$V$. Avoids re-prefill compute — pure efficiency gain, no info loss.

### 3. End-to-end complexity (Thm 3.4)

- **LatentMAS per agent**: $O((d_h^2 m + d_h m^2 + d_h t m) L)$ where $t$ = input length, $m$ = latent steps.
- **Text MAS** for equivalent expressiveness: $O((d_h^3 m / \log^2|\mathcal{V}| + d_h^3 m / \log|\mathcal{V}| + d_h^2 t m / \log|\mathcal{V}|) L + d_h^2 |\mathcal{V}| m / \log|\mathcal{V}|)$.
- Text needs roughly $d_h / \log|\mathcal{V}|$ more steps to match latent expressiveness $\to$ cubic blowup + a $d_h^2 |\mathcal{V}|$ decoding term.

---

## Experiments / Results

### Setup

- **Models**: Qwen3-4B / 8B / 14B (all dense decoders).
- **9 benchmarks** in three buckets:
    - **Math/science**: GSM8K, AIME24, AIME25, GPQA-Diamond, MedQA.
    - **Commonsense**: ARC-Easy, ARC-Challenge.
    - **Code**: MBPP-Plus, HumanEval-Plus (executed in sandboxed Python).
- **Baselines**: Single model | Sequential TextMAS (planner/critic/refiner/solver) | Hierarchical TextMAS (math/science/code/summarizer).
- **Latent step sweep** $m \in \{0, 10, 20, 40, 80\}$. Best is $m \approx 40$–$80$ (saturates / slightly declines beyond).
- **Generation**: $T=0.6$, top-$p=0.95$, max output 2048 (ARC, GSM), 4096 (MedQA, code), 8192 (GPQA), 20000 (AIME).
- 8$\times$A100-80G; mean over 3 seeds. vLLM backend used for all baselines for fair speed comparison.

### Main numerical findings

| Setting | Avg Acc gain over single | Avg Acc gain over TextMAS | Token reduction vs TextMAS | Speedup vs TextMAS |
|---|---|---|---|---|
| Sequential | +14.6 % | +2.8 % | 70.8 % | $4.0\times$ |
| Hierarchical | +13.3 % | +4.6 % | 83.7 % | $4.3\times$ |

- **Speedup vs vLLM-TextMAS** (already accelerated): $2.6\times$ – $7\times$ depending on task.
- **Tokens vs single model**: LatentMAS uses 15–60 % fewer (because intermediate agents emit no tokens, only the final judger decodes).
- AIME24/25 (Qwen3-8B): TextMAS needs >20k output tokens to do CoT; LatentMAS hits comparable or better accuracy with $\leq 50$ latent steps and ~9k decoded tokens for the final answer.

### In-depth analyses

- **Latent thoughts are semantically meaningful** (Fig 5): t-SNE / PCA of latent thought embeddings overlaps the embedding cloud of text tokens from TextMAS on MedQA — latent thoughts cover **a superset** of the text-token region, suggesting equal-meaning + richer diversity.
- **Effect of $W_a$** (Fig 6, 7):
    - Without alignment, $h_t$ density drifts far from input $e_t$ density.
    - After $W_a$, aligned $e_{t+1}$ density realigns with $e_t$.
    - Downstream acc gain from $W_a$ alone: +2.3 % to +5.3 %.
- **Latent step depth** (Fig 8): accuracy rises then plateaus / declines past $m \approx 80$. Excessive latent steps inject noise / repetition.
- **Case study (GSM8K, Qwen3-14B sequential)**: TextMAS planner misinterprets "before that"; critic/refiner inherit the error; solver outputs 15 (wrong, gold = 75). LatentMAS solver, given continuous latent context, recovers the correct 75. Illustrates lossy text encoding being the actual failure mode, not capacity.

---

## Implementation

### Repo structure

```
LatentMAS/
├── run.py                    # CLI entry point, batches dataset, dispatches to methods
├── models.py                 # ModelWrapper: HF + vLLM, W_a realignment, latent rollout
├── prompts.py                # Per-role per-task prompt builders (sequential + hierarchical)
├── methods/
│   ├── __init__.py           # Agent dataclass + default_agents()
│   ├── baseline.py           # Single-model baseline
│   ├── text_mas.py           # Text-based MAS
│   └── latent_mas.py         # LatentMAS (HF and vLLM variants)
├── data.py                   # Dataset loaders (gsm8k, aime, gpqa, arc, mbpp+, humaneval+, medqa)
├── utils.py                  # answer extraction, sandboxed code exec, seeding
├── data/medqa.json           # bundled sample
├── example_logs/             # full agent traces for MBPP+ / HumanEval+
└── requirements.txt
```

### Key components

#### `methods/__init__.py` — agent roles

`/home/jayoo/code/jayoo_personal/LLM_Knowledge_Base/raw_data/papers/Latent Collaboration in Multi-Agent Systems/LatentMAS/methods/__init__.py:11-17`

- **Agent dataclass**: just `(name, role)`.
- **`default_agents()`** returns the fixed 4-agent chain: `Planner $\to$ Critic $\to$ Refiner $\to$ Judger`.
    - In hierarchical mode the **same 4 slots** are reused but each is reprompted as math/science/code/summarizer (see `prompts.py`).

#### `models.py` — `ModelWrapper`

- **`__init__`** (`models.py:31-85`):
    - If `use_vllm`: load `vllm.LLM` (optionally with `enable_prefix_caching=True, enable_prompt_embeds=True`), tokenizer, and **separately** a second HF model on `device2` for latent rollout (`use_second_HF_model=True`).
    - Else: just a plain HF `AutoModelForCausalLM` in bf16.
    - Calls `_ensure_latent_realign_matrix()` to compute $W_a$ once.
- **`_build_latent_realign_matrix`** (`models.py:158-185`): closed-form ridge regression
    ```python
    gram = W_out.T @ W_out + 1e-5 * I       # d_h x d_h
    rhs  = W_out.T @ W_in                    # d_h x d_h
    W_a  = torch.linalg.solve(gram, rhs)    # d_h x d_h
    target_norm = W_in.norm(dim=1).mean()
    ```
    - If `--latent_space_realign` flag is **off**, $W_a$ is **replaced by identity** (ablation control). So passing the flag is what actually enables the alignment described in §3.1 of the paper.
- **`_apply_latent_realignment`** (`models.py:204-213`): `aligned = h @ W_a`, then **renormalize to `target_norm`** (i.e. rescale aligned vector's L2 to match average input-embedding norm). Cast back to bf16.
- **`generate_latent_batch`** (`models.py:277-350`): the core latent loop.
    1. Run HF model on input ids $\to$ `past_kv`, save $h_t$ from `outputs.hidden_states[-1][:, -1, :]`.
    2. Loop $m$ steps:
        - `latent_vec = self._apply_latent_realignment(last_hidden, source_model)`
        - Feed `inputs_embeds=latent_vec.unsqueeze(1)` with extended attention mask (covers `past_len + 1`).
        - Update `past_kv` from outputs; grab new `last_hidden`.
    3. Return final `past_kv` — this is the **latent working memory** $\mathcal{M}_{A_i}$.
- **`generate_latent_batch_hidden_state`** (`models.py:353-415`): variant used in vLLM path. Same loop but **also returns the concatenated input-embedding sequence** (input emb + each $m$ aligned latent embed), used downstream to splice into the judger's vLLM prompt embeddings.
- **`generate_text_batch`** (`models.py:216-267`): standard HF `generate` that takes an existing `past_key_values` and resumes — used by the **judger agent only** to decode the final answer. Note the cache_position fixup to handle prepended past.

#### `methods/latent_mas.py` — orchestration

The single file implementing the full latent MAS pipeline.

- **`LatentMASMethod.__init__`** (`methods/latent_mas.py:18-51`): pulls hyperparams from argparse; agents = `default_agents()`; sets `latent_steps`, `latent_only` and `sequential_info_only` ablation flags.
- **`run_batch`** (HF path) (`methods/latent_mas.py:82-249`):

    ```python
    past_kv = None
    for agent in self.agents:           # 4 agents
        msgs = build_agent_message_{sequential,hierarchical}_latent_mas(...)
        prompts, ids, mask, _ = model.prepare_chat_batch(msgs, add_generation_prompt=True)
        if args.think:
            prompts = [p + "<think>" for p in prompts]   # force Qwen3 think token
        if agent.role != "judger":
            # accumulate latent thoughts + KV
            past_kv = model.generate_latent_batch(
                ids, mask, latent_steps=self.latent_steps, past_key_values=past_kv
            )
            # optional truncation ablations:
            if self.sequential_info_only or self.latent_only:
                past_kv = self._truncate_past(past_kv, tokens_to_keep)
        else:
            # final agent decodes text using the accumulated past_kv as context
            generations, _ = model.generate_text_batch(
                ids, mask, max_new_tokens=..., past_key_values=past_kv
            )
            final_texts[idx] = generations[idx]
    ```

    - **Key insight**: the **same `past_kv` object is passed unchanged** between agents. HF transformers' `generate(..., past_key_values=past_kv)` will automatically prepend it to the new attention $\to$ this is the entire "latent working memory transfer" mechanism.
    - There is **no separate "concatenate KV" call** — it is just persisting `past_kv` across agent loop iterations.
    - For non-judger agents, the *prompt itself is also prepended* (role re-prompt = "You are a Critic Agent…") so $A_2$ sees: `[prefill of A_1 + latent of A_1 + role-prompt of A_2 + latent of A_2 …]`.
- **`run_batch_vllm`** (`methods/latent_mas.py:251-438`): hybrid HF + vLLM path used when `--use_vllm`.
    - Latent steps still done with HF model on `device2` (`generate_latent_batch_hidden_state` returns both KV and the explicit hidden-embedding sequence `previous_hidden_embedding`).
    - Per non-judger agent: stash that hidden sequence in `embedding_record`.
    - For **judger** (final decode): concatenate all stashed hidden sequences, compute the judger prompt's token embeddings via `self.model.embedding_layer(judger_encoded)`, then **splice the latent embeddings into the judger prompt right after the `<|im_start|>user\n` marker**:
        ```python
        combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
        ```
    - Pass via vLLM's `prompt_embeds` interface: `vllm_engine.generate([{"prompt_embeds": embeds}, ...], sampling_params)`.
    - The vLLM source is patched (acknowledged in README) because vLLM doesn't officially support custom KV / prompt embeds for this use case.
- **Ablation flags**:
    - `--latent_only`: keep only the latest `latent_steps` worth of KV between agents — strips out the textual prefix history.
    - `--sequential_info_only`: keep only the tokens *added* by the most recent agent.
- **`_truncate_past` / `_slice_tensor`** (`methods/latent_mas.py:53-79`): trim the KV cache by slicing along the sequence axis, with handling for both legacy tuple-of-tuples format and the new HF `Cache` class.

#### `prompts.py` — role/task prompts

- **`build_agent_message_sequential_latent_mas`** (`prompts.py:2-115`):
    - Planner/Critic/Refiner prompts mention "latent KV representation format" to hint to the model that the prior thoughts arrive in the KV cache rather than as text.
    - Judger prompt: *"You are provided with latent information for reference… The latent information might contain irrelevant contents. Ignore it if not helpful."*
    - Per-task answer-format suffix (e.g. `\boxed{...}` for math/MC, `\`\`\`python … \`\`\`` for code).
- **`build_agent_message_hierarchical_latent_mas`** (`prompts.py:118-338`): math/science/code/task-summarizer prompts. Quirky: the *task summarizer* prompt is also fed via the same agent-loop machinery (mapped to the `judger` slot).
- TextMAS uses analogous functions but the previous agent's actual text output is interpolated into the next agent's prompt as `context`.

#### `run.py` — top-level entry

- Argparse (`run.py:84-119`); important flags:
    - `--method {baseline, text_mas, latent_mas}`
    - `--prompt {sequential, hierarchical}`
    - `--latent_steps` (default 0, paper sweeps to 80)
    - `--latent_space_realign` (must be set to actually use $W_a$ — else identity)
    - `--think` (appends `<think>` to push Qwen3 into thinking mode)
    - `--use_vllm`, `--enable_prefix_caching`, `--use_second_HF_model`, `--device2`
- Auto-enable: `if method == latent_mas and use_vllm: use_second_HF_model = True; enable_prefix_caching = True`.
- Standard loop: load dataset, batch by `generate_bs`, call `method.run_batch[_vllm]`, evaluate `correct/total`.

### Training / inference flow (one MAS run)

Sequential, HF backend, $N=4$ agents, $m$ latent steps each:

1. **Init**: load model; build $W_a$ from $(W_\text{in}, W_\text{out})$.
2. **Per question $q$**, set `past_kv = None`.
3. **For** agent $A_i$ in [Planner, Critic, Refiner]:
    1. Build role-prompt with $q$ $\to$ tokenize $\to$ `input_ids`.
    2. Optionally append `<think>` token.
    3. Forward through HF model with `past_key_values=past_kv` $\to$ get $h_t$ from last layer.
    4. Loop $m$ steps: `latent = h @ W_a` (renormalized) $\to$ feed as `inputs_embeds` $\to$ update past_kv $\to$ next $h$.
    5. Store updated past_kv (now contains $A_1, \dots, A_i$'s prefills + latent thoughts).
4. **For** Judger:
    1. Build solver prompt with $q$.
    2. Call HF `generate(..., past_key_values=past_kv)` to text-decode the final answer (max 2k–20k tokens depending on task).
5. **Score**: extract `\boxed{...}` or python block, compare to gold, exec code in sandbox if applicable.

### Configs / Hyperparameters (paper §C.2 + `run.py`)

| Hyperparam | Default / used |
|---|---|
| `temperature` | 0.6 |
| `top_p` | 0.95 |
| `latent_steps` $m$ | swept $\{0,10,20,40,80\}$; best $\approx 40$–$80$ |
| ridge $\lambda$ in $W_a$ | $10^{-5}$ (`models.py:173`) |
| `max_new_tokens` (judger) | 2048 ARC/GSM, 4096 MedQA/MBPP+/HumanEval+, 8192 GPQA, 20000 AIME |
| `generate_bs` | 20 |
| backbones | Qwen3-{4,8,14}B |
| dtype | bf16 (fp32 fallback if no cuda) |
| GPUs | $8\times$A100-80G |
| seeds | mean over 3 |
| code-exec timeout | 10 s sandbox |

### Notable engineering details

- **Re-norming after alignment** (`models.py:209-212`) — not described in main paper but matters in practice; clamps `aligned.norm(...).min(1e-6)` then rescales to mean input-emb norm. Without it bf16 numerical drift accumulates over $m$ latent steps.
- **`<think>` token wrapping** (`methods/latent_mas.py:112-115, 161-163`) — Qwen3's thinking mode; the latent thoughts effectively occur **inside the `<think>` block**.
- **No actual layer-wise concat in code** — paper describes "layer-wise concatenation" of KV between agents, but the implementation just **persists `past_key_values` across `model.forward` calls**; HF transformers handles the concat internally via its Cache object.
- **vLLM hack**: vLLM doesn't expose KV mutation, so the vLLM path **routes latent info through a *prompt-embeds* injection** at the judger step instead of true cache prepending. Author admits "minor numeric differences may arise compared to official HF backend".
- **No training, no checkpointing** — $W_a$ is the only "learned" parameter and it's solved analytically once per model load (no gradient updates anywhere).
