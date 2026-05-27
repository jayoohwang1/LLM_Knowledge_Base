# LatentMem: Customizing Latent Memory for Multi-Agent Systems

## Summary

- **TL;DR**: Replace handcrafted text-based MAS memory with **per-agent latent token sequences** produced by a small LLM-based "memory composer" trained with a **GRPO variant (LMPO)**. Latent tokens are appended to each agent's input embeddings as soft prompts.
- **Core pipeline**:
    - **Experience Bank** stores raw MAS trajectories (no condensation, no insight extraction). Retrieval is dense (MiniLM cosine sim, top-$K$, $K{=}1$).
    - **Memory Composer** $\sigma_\phi$: a (LoRA-fine-tuned) LLM that ingests `[retrieved trajectories + agent role profile]` and emits a fixed-length latent memory matrix $m_j \in \mathbb{R}^{L'\times D}$ ($L'{=}8$).
    - **Injection**: $m_j$ is *concatenated to the token embeddings* of the active agent's prompt; the agent backbone runs unchanged. Differentiable through the composer.
    - **LMPO**: token-level GRPO objective on full multi-agent rollout. Only $\phi$ (LoRA params of composer + a small projection) is trained; agent backbones are frozen.
- **Targets two MAS memory failures**:
    - *Memory homogenization*: one-size-fits-all memory across heterogeneous agent roles → role-conditioned composer.
    - *Information overload*: huge token cost from raw/structured memory → fixed-length latent tokens ($L'{=}8$).
- **Result**: up to **+19.36%** vs vanilla; ~**50% fewer tokens**, ~**2/3 inference time** vs SoTA MAS memory. Generalizes to held-out MAS (CAMEL, DyLAN) and OOD datasets (BigCodeBench, PDDL).

---

## Motivation / Problem

- **MAS memory** = persistent store that lets agents accumulate experience across tasks → key for continual adaptation.
- **Existing memory designs** (MetaGPT shared pool, ChatDev inside-trial, OAgents short+long-term hierarchical, G-Memory tri-graph, JoyAgent working/semantic/procedural) all share two issues:
    - **(i) Memory homogenization** — feed the same memory to every agent regardless of role. Undermines role adherence; correlated errors compound (cf. Cemri 2025, "Why do MAS LLM systems fail").
    - **(ii) Information overload** — multi-granularity memory pushes huge volumes of structured entries into context. Long contexts overwhelm LLMs and obscure signal.
- **Question**: *"Can we design a learnable memory that is both **role-aware** and **token-efficient**, without extensive manual engineering?"*

### Position vs related work
- **Latent reasoning lineage**: MemGen, SoftCoT, LatentSeek, LatentMAS, Cache-to-Cache — show latent vectors can carry reasoning/experience. LatentMem extends to **MAS memory** specifically.
- **Distinct from MARTI**: MARTI fine-tunes the shared backbone via GRPO; LatentMem *freezes* backbones, trains a small composer → cheaper, more transferable.

---

## Method

### Setup / notation
- MAS $\mathcal{X} = (\mathcal{A}, \mathcal{G}, \mathcal{M})$ — agents, exec graph, memory module.
- Each agent $a_k = (\gamma_k, \pi_{\theta_k})$: role profile $\gamma_k$, frozen policy $\pi_{\theta_k}$.
- **Objective**: $\max_{\mathcal{M}} \mathbb{E}_{q\sim\mathcal{D},\, \tau\sim \mathcal{X}(q)}[R(\tau)]$ — find memory module that maximizes task reward.
- **Trajectory**: $\tau = \{(\alpha_j, p_j, o_j)\}_{j=1}^H$ — at each step the active-agent index, its input prompt, its output.

### Pipeline overview
1. **Retrieve** top-$K$ trajectories from experience bank $\mathcal{B}$ via MiniLM cosine sim on the query.
2. **Compose** latent memory $m_j$ per agent step from `(role profile, retrieved trajectories)`.
3. **Inject** $m_j$ into agent forward by concatenating to input embeddings.
4. **Update** $\mathcal{B}$ with the new trajectory after task completion.

### 4.2 Experience Bank $\mathcal{B}$
- **Philosophy**: bitter-lesson (Sutton 2019) — scalable systems should *learn*, not rely on handcrafted summarization/insight extraction → store **raw trajectories only**.
- **Init**: populated with a diverse cross-domain, cross-MAS trajectory pool $\{\tau_i\}_{i=1}^C$.
- **Retrieve**: $\mathcal{T}_q = \text{top-}K_{\tau_i\in\mathcal{B}}(\text{sim}(\mathbf{v}(q), \mathbf{v}(\tau_i)))$, default $K{=}1$.
- **Update**: $\mathcal{B} \leftarrow \mathcal{B}\cup\{\tau_{\text{new}}\}$ — online, no retraining needed. Enables continual adaptation.
- **Why not feed raw trajectories directly to agents?** Long context overwhelms LLMs; raw trajectories are role-agnostic.

### 4.3 Memory Composer $\mathcal{C}$ / $\sigma_\phi$
- **Functional form**: $m_j = \sigma_\phi(\gamma_{\alpha_j}, \mathcal{T}_q) \in \mathbb{R}^{L'\times D}$.
    - $L'{=}8$ — fixed latent length.
    - $D$ — hidden dim of the agent backbone.
- **Implementation**: instantiate $\sigma_\phi$ as a pretrained LLM (same family as agents, e.g. Qwen3-4B) with **LoRA** + a set of *learnable latent query tokens* $\in \mathbb{R}^{L'\times D_c}$ that get appended at the end of the input embeddings.
    - Composer hidden states at the **last $L'$ positions** are taken as $m_j$.
    - Note the composer's hidden dim $D_c$ may differ from agent's $D$ → a linear projection `weaver2agent_proj` maps composer hidden states → agent embedding space.
- **Injection into agent**: $\tilde\pi_{\theta_{a_j}}(p_j, m_j) = \pi_{\theta_{a_j}}(\text{concat}(h_j, m_j))$.
    - $h_j \in \mathbb{R}^{L\times D}$ = agent's token embeddings of input prompt $p_j$.
    - **Transparent** to MAS framework: agent code sees an unchanged policy interface.
- **Composer prompt template**:
    > "# Current Task ... # Examples of previous successful tasks ... # {role} response in each successful task ... You are acting as a {role}. Using no more than {k} tokens, extract the most relevant information ..."
    - Role injected as text → composer can encode role-specific latents from same backbone.

### 4.4 Latent Memory Policy Optimization (LMPO)

- **Variant of GRPO** (Shao et al. 2024 / DeepSeekMath), adapted for MAS rollouts.

#### Parametric dependency / differentiability
- Forward factorization:
    $$\mathbb{P}(\tau \mid q, \mathcal{T}_q; \phi, \{\theta_k\}) = \prod_{j=1}^H \mathbb{P}(o_j \mid p_j, m_j; \theta_{\alpha_j})$$
- Each agent step: $\mathbb{P}(o_j \mid p_j, m_j; \theta_{\alpha_j}) = \prod_{t=1}^T \tilde\pi_{\theta_{\alpha_j}}(o_j^{(t)} \mid p_j, o_j^{(<t)}, m_j)$.
- $m_j$ is a **differentiable interface**: gradient of task reward flows back through agent fwd pass → into composer $\phi$. Backbones $\{\theta_k\}$ stay frozen.

#### Rollouts + advantages (GRPO-style)
- Sample $G$ trajectories per query: $\{\hat\tau_i\}_{i=1}^G$.
- **Group-relative advantage**: $\hat A_i = (R(\hat\tau_i) - \text{mean}(R)) / (\text{std}(R) + \epsilon)$ — normalized within the rollout group.

#### Token-level surrogate objective
- **Problem with vanilla trajectory-level GRPO**: in long-horizon MAS, naively averaging per-trajectory makes tokens in long rollouts contribute less → can't capture coordination.
- **Token-level objective** (sum-then-divide over total token count):
    $$\mathcal{J}_{\text{LMPO}}(\phi) = \mathbb{E}\Big[\frac{1}{|\{\hat\tau_i\}|}\sum_{i,j,t}\mathcal{L}_{i,j,t}(\phi)\Big]$$
    with
    $$\mathcal{L}_{i,j,t}(\phi) = \min(r_{i,j,t}(\phi)\hat A_i,\ \text{clip}(r_{i,j,t}(\phi), 1-\varepsilon, 1+\varepsilon)\hat A_i)$$
- **Token-level IS ratio**:
    $$r_{i,j,t}(\phi) = \frac{\tilde\pi_\theta(o^{(t)} | p, o^{(<t)}, \sigma_\phi(\gamma, \mathcal{T}_q))}{\tilde\pi_\theta(o^{(t)} | p, o^{(<t)}, \sigma_{\phi_{\text{old}}}(\gamma, \mathcal{T}_q))}$$
- Note ratio is over **agent likelihood**, but the only varying parameter is $\phi$ (composer).
- **PPO-style clipping** with $\varepsilon = 0.2$.

### Three claimed advantages
- **(I)** Role-aware → mitigates homogenization (composer conditions on $\gamma_k$).
- **(II)** Fixed-length latents → mitigates info overload.
- **(III)** Differentiable + LMPO → end-to-end optimizable; no manual memory engineering.

---

## Experiments / Results

### Setup
- **Benchmarks** (6, 4 domains):
    - **Knowledge QA**: TriviaQA, PopQA (held-in)
    - **Coding**: KodCode (held-in), BigCodeBench (held-out)
    - **Reasoning QA**: StrategyQA (held-in)
    - **Symbolic Planning**: PDDL (held-out)
- **MAS frameworks** (4): AutoGen, MacNet (in-dist train); CAMEL, DyLAN (held-out, unseen agent roles).
- **Backbones**: Qwen3-4B-Instruct-2507 (main), Llama-3.1-8B-Instruct (scaling).
- **Baselines**:
    - Single-agent memory: Voyager, Generative, JoyAgent.
    - MAS memory: MetaGPT, ChatDev, OAgents, G-Memory.
- **Composer**: LoRA $r{=}16$, $\alpha{=}32$, target `[q_proj, v_proj]`, dropout 0.1.
- **Hyperparams** (Table 3): lr 1e-5, AdamW, clip $\varepsilon{=}0.2$, KL coef $\beta{=}0$, $L'{=}8$, $K{=}1$, prompt limit 10240, response 4096, train temp 1.0, eval temp 0.0, train epochs 1, group size = 32 macro / 8 micro, flash attn + bf16 + vLLM + DeepSpeed.

### Main numbers
- **Qwen3-4B + AutoGen**: outperforms SoTA by avg **+7.86%** (single-agent baselines), **+6.66%** (MAS baselines). **+16.20%** on TriviaQA.
- **Llama-3.1-8B + MacNet, KodCode**: 48.50% → **65.50%** (+17.00).
- **Held-out MAS**:
    - CAMEL+KodCode: **+7.05%** vs No-Memory.
    - Others (MetaGPT, Voyager) often *degrade* on held-out MAS while LatentMem still gains.
- **Held-out datasets**:
    - AutoGen+PDDL: **+7.10%** vs No-Memory; MetaGPT *drops -4.44%*.

### vs MARTI (multi-agent backbone fine-tuning)
- Same compute budget, same data (TriviaQA, KodCode, StrategyQA, PopQA).
- **TriviaQA + AutoGen**: LatentMem **+11.73%** over MARTI.
- LatentMem better exploits *structural* MAS advantages instead of homogenizing them by fine-tuning the shared backbone.

### Cost
- LatentMem ≈ **2.16× faster inference** than OAgents on DyLAN+TriviaQA.
- Token usage on AutoGen+KodCode: even *0.01M less* than No-Memory while +8.40% accuracy. JoyAgent uses 1.87M *more* tokens for +2.50%.

### Framework analysis
- **t-SNE on latent memories**: well-separated clusters per agent role (e.g. user-proxy vs assistant in AutoGen+KodCode; 4 clusters for CAMEL critic/user-proxy/summarizer/actor on BigCodeBench). Holds even on unseen-MAS+OOD-dataset.
- **Scaling with task horizon**: cumulative accuracy keeps climbing as more experiences accrue; baselines plateau / are unstable.

### Sensitivity / Ablation
- **Sensitivity to $L'$** (latent length 2,4,6,8,16,32): monotonic improvement w/ diminishing returns, $L'{=}8$ chosen.
- **Sensitivity to $K$** (Fig. 10): LatentMem keeps improving as $K$ grows from 1→5; **G-Memory degrades when $K>3$** (info overload) — empirical proof of LatentMem's compression advantage.
- **Component ablation** (Fig. 6 right):
    - w/o role profile $\gamma$: drops 2.30% (KodCode AutoGen), 6.45% (MacNet) → role conditioning matters more for complex MAS.
    - w/o experience-bank updates: drops 3.60% on KodCode/MacNet; **7.63% on PDDL** → online accumulation crucial for harder/OOD task distributions.

### Case study (Fig. 7, PDDL ball-sort)
- Vanilla MacNet: *step repetition* (oscillates moving same ball back and forth).
- MacNet+OAgents: *Disobey Task Specification* — blindly follows retrieved trajectory steps, ignores actual goal.
- MacNet+LatentMem: role-aware latent enables **self-correction** — after invalid action, next step uses "check valid actions" loop and proceeds.

---

## Implementation

### Repo structure
```
LatentMem/
├── main.py                       # entrypoint: parse cfg, build MAS+memory+runner, dispatch mode
├── configs/
│   ├── zero2.yaml                # DeepSpeed ZeRO-2
│   └── latentmem/                # per-dataset YAML (triviaqa, kodcode, popqa, pddl)
├── scripts/
│   ├── data.sh                   # mode=data, collect raw trajectories
│   ├── sft.sh, lmpo.sh           # train modes
│   └── eval.sh, eval_hf.sh       # evaluation
├── common/
│   ├── config.py, registry.py    # registry-pattern config plumbing
│   ├── interactions/             # InteractionManager: agent<->env loop
│   │   └── multiturn_interaction.py
│   └── utils/{tensor,retrieval,math,code}_utils.py
├── data/<dataset>/{builder,env}.py   # dataset builders + envs (triviaqa, kodcode, popqa, pddl)
└── latentmem/
    ├── runner.py                 # LatentMemRunner: data / sft / lmpo / eval
    ├── trainer/
    │   ├── grpo_trainer.py       # MemMasterLMPOTrainer (subclasses trl.GRPOTrainer)
    │   └── sft_trainer.py        # MemMasterSFTTrainer (for warmup)
    ├── mas_core/
    │   ├── base_memory_mas.py    # BaseMemoryMAS (nn.Module with agents_list + centralized_memory)
    │   ├── base_centralized_memory.py  # Memory dataclass + abstract base
    │   ├── structures/           # MAS topologies: autogen / macnet / camel
    │   └── memory/
    │       ├── latentmem.py      # LatentMem composite memory (rag + composer)
    │       ├── composer.py       # Composer = LoRA LLM + learnable latent query tokens
    │       ├── prompt.py         # composer prompt templates
    │       └── backbone/         # baselines (metagpt, voyager, generative, gmemory, oagent) + experience.py
    └── utils/
        ├── agent.py              # LLMAgent (forward + invoke with latent injection)
        ├── message.py            # MessageNode / MessageGraph / Trajectory dataclasses
        ├── data.py, tools.py
```

### Key components

#### `LatentMem` central memory: `latentmem/mas_core/memory/latentmem.py`
- Class `LatentMem(BaseCentralizedMemory)` registered as `"latentmem"` (see `latentmem.py:17`).
- **Two sub-modules**:
    - `self.rag_memory` — chosen via `get_rag_cls(rag_mode)`. For LatentMem proper, `rag_mode="latentmem"` → `ExperienceBank` (`memory/backbone/experience.py`). Other modes (`metagpt`, `voyager`, `generative`, `gmemory`, `oagent`) are baselines reused inside same wrapper.
    - `self.weaver: Composer` — the memory composer (paper calls it $\sigma_\phi$; code calls it "weaver").
- **`retrieve_memory(task_description, agent)`** (`latentmem.py:77`):
    1. `self.rag_memory.retrieve(task_description, agent_role=agent.role)` → returns `RAGMemoryResponse(pos_shots, neg_shots, extra_fields)`.
    2. Build textual memory from positive/negative shot trajectories using templates in `prompt.py` (`POS_SHOTS_TEMPLATE`, `NEG_SHOTS_TEMPLATE`, `INSIGHTS_TEMPLATE`).
    3. Pull this agent's role responses from those shots via `_concat_agents_response`.
    4. Call `_construct_latent_memory(...)` → composer fwd pass → linear projection → `latent_memory: torch.FloatTensor`.
- **`_construct_latent_memory`** (`latentmem.py:143`):
    - Builds composer prompt from `prompt.EXTRACT_LATENT_PROMPT[_FULL]` (with role + agent_message if pos shots exist).
    - `latent = self.weaver.text_to_latent(prompt).squeeze(0)`.
    - `latent = self.weaver2agent_proj(latent)` → maps composer hidden_size → agent hidden_size.
- **`weaver2agent_proj`** registered in `register_agents` (`latentmem.py:65-70`):
    ```python
    agent_hidden_size = agents_list[0].model.config.hidden_size
    self.weaver2agent_proj = nn.Linear(self.config.hidden_size, agent_hidden_size)
    ```
    - Even when composer == same family as agents (Qwen3-4B), this still creates a learnable projection — extra capacity beyond LoRA params.

#### `Composer`: `latentmem/mas_core/memory/composer.py`
- `Composer(nn.Module)` (`composer.py:14`).
- **Backbone**: `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bf16, attn_implementation="flash_attention_2")` wrapped in `get_peft_model(model, peft_config)` (LoRA).
- **Learnable latent queries** (the soft-prompt suffix):
    ```python
    self.query_latents = nn.Parameter(
        torch.randn(latents_len, hidden_size), requires_grad=True
    )
    ```
    `composer.py:43-46`. These tokens are *appended* to the text embeddings before the forward pass.
- **`text_to_latent(texts)`** (`composer.py:56`):
    1. Tokenize prompt (left-padded) → `input_ids`, `attention_mask`.
    2. `inputs_embeds = embedding_layer(input_ids)`.
    3. Broadcast `query_latents` to batch, concat after `inputs_embeds`; pad attention_mask with ones.
    4. Forward with `output_hidden_states=True`.
    5. Extract **last $L'$ positions** of last hidden state → these are the latent memory tokens.
- This is essentially a *Perceiver/Q-former-style* compression but with no cross-attention — just self-attention conditioning the appended learnable tokens on the prompt.

#### Agent forward / latent injection: `latentmem/utils/agent.py`
- `LLMAgent` (dataclass with `model`, `tokenizer`, `role`, `topology_node_id`, `centralized_memory`).
- **Two paths**:
    - **`invoke()`** (`agent.py:202`) — generation/inference time.
        - Calls `self._trigger_memory(task_description)` → gets `Memory(text_memory, latent_memory)` from centralized memory.
        - **If latent_memory exists, text_memory is replaced with `EMPTY_MEMORY` placeholder** (`agent.py:351`) — latent fully supersedes textual.
        - Tokenize chat-templated `[system, user]` prompt → embed → **concat latent embeddings on the right**:
            ```python
            final_inputs_embeds = torch.cat([text_embeddings, latent_emb], dim=1)
            ```
            (`agent.py:254`).
        - Left-pad batch to `max_seq_len` (mask zeros for padding positions).
        - `self.model.generate(inputs_embeds=..., attention_mask=..., generation_config=...)`.
    - **`forward()`** (`agent.py:42`) — training time (label-supervised pass for log-prob computation in LMPO).
        - For each example reconstruct latent memory deterministically from stored `text_memory + task_description + extra_fields` via `centralized_memory.process_memory(...)` (`agent.py:363`).
        - Build `[prompt_embeds; latent_emb; response_embeds]`, with labels = -100 for prompt+latent positions, real token ids for response (+ EOS appended).
        - Truncate to `MAX_LEN=1024` from the left if too long (logits re-padded out to `seq_len` afterwards so `_get_per_token_logps` in trainer can align).
- **Why store text_memory and reconstruct latent at training time**: experience bank stores text trajectories only; recomputing the latent via composer at every training step is what makes the path differentiable w.r.t. $\phi$.

#### Experience Bank: `latentmem/mas_core/memory/backbone/experience.py`
- `ExperienceBank(BaseRAGMemory)` — backs the LatentMem rag mode.
- **Storage**: `langchain_chroma.Chroma` vector DB; `embedding_function` = `EmbeddingFunction(all-MiniLM-L6-v2)` (frozen).
- **`add(trajectory)`** (`experience.py:38`): serializes trajectory to metadata, indexes by `task_init_description`.
- **`retrieve(query, agent_role)`** (`experience.py:164`):
    - `_generative_retrieve` — fetches $2K$ candidate pos + $2K$ neg via Chroma sim search, **rescores with an LLM** (`prompt.GENERATIVE_SYSTEM_PROMPT/USER_PROMPT`, parses a 0-10 integer), keeps top-$K$.
    - `_extract_agent_context(shots, agent_role)` — pulls out only this agent's responses from each shot's MessageGraph; stored as `extra_fields[agent_role]` on the trajectory for composer use.
    - **Note**: even though paper says "lightweight, no insight extraction", an LLM is used here as a **reranker** — a softer form of selection.

#### Base MAS: `latentmem/mas_core/base_memory_mas.py`
- `BaseMemoryMAS(ABC, nn.Module)`: holds `centralized_memory`, `shared_tokenizer`, `agents_list`.
- `_agent_forward(messages)` — delegates to `agents_list[0].forward(...)` with batched system/user templates + state. All agents share the backbone (`share_llm=True`), so only one fwd pass via the first agent.
- **`from_config`**: optionally loads composer state from `.safetensors` (`load_state_dict(strict=False)` lets only composer+projection weights load while frozen backbones are skipped).

#### MAS topologies: `latentmem/mas_core/structures/`
- **AutoGen** (`autogen/autogen_main.py`): two agents `assistant_agent` (role "assistant agent") + `user_proxy_agent` (role "user proxy agent"). Static topology. `generate()` runs assistant first, then user-proxy with assistant's output appended.
- **MacNet** (`macnet/macnet_main.py`): decentralized actor/critic edges (5 agents per paper).
- **CAMEL** (`camel/camel_main.py`): role-playing 4 agents (strategy + code + test + summarizer).
- Each registers via `@registry.register_mas("name")`.

### Training / Inference flow

#### Mode selection (`main.py` + `runner.py:execute`)
- `run.mode = data | sft | lmpo | eval`.

#### Data collection (`scripts/data.sh`, `run.mode=data`, `bootstrap_data`)
- Sets `use_weaver=False` → MAS runs without latent memory.
- Iterates over `train_dataset` (batches of 16).
- For each batch, builds envs (`env.set_env(task_config)` → `(system_prompt, init_user_prompt)`), runs `interaction_controller.run_inter_loop(...)` from `MultiTurnInteractionManager`.
- Each successful trajectory (`trajectory.label == True`) is appended to the experience bank (`memory.add_memory`).
- Finally Chroma DB is exported to `rag_0/` and converted to structured JSON via `rag_to_structured_json`.

#### Multi-turn interaction (`common/interactions/multiturn_interaction.py:run_inter_loop`)
- For up to `max_turns=5` turns:
    1. Mask out inactive examples.
    2. Build chat-templated task context per example (initial user msg + history).
    3. `memory_mas.generate(domain_instructions, task_contexts, generation_config)` — each MAS structure decides how its agents fire.
    4. Post-process responses with `env.preprocess_action`, then call `env.step` → next_obs + done.
    5. Append `(assistant, response)` and `(user, observation)` to interaction history; trajectories collect each step's MessageGraph.
- After loop, compute `env.feedback()` per example → `reward`, `label = (reward == 1.0)`.

#### LMPO training (`scripts/lmpo.sh`, `mode=lmpo`)
- Uses `MemMasterLMPOTrainer` (subclasses `trl.GRPOTrainer`).
- **`gradient_accumulation_steps = len(agents_list)`** — one rollout group accumulates one mini-step per agent.
- **`_generate_and_score_completions`** (`grpo_trainer.py:203`):
    1. Build $G$ envs from inputs (paper's group rollout — `num_generations`=batch size, 8 in `lmpo.sh`).
    2. With `unwrap_model_for_generation`, set `generation_manager.memory_mas = unwrapped_model`, run multi-turn loop to produce trajectories.
    3. `rewards = [traj.label]`; compute group mean & std; advantage = (r - mean) / (std + 1e-4) when `scale_rewards`.
    4. Flatten trajectories: `_flatten_trajectory(trajectories)` produces `{agent_id: [MessageNode]}` lists, one per agent role — used as the per-agent micro-batches that hit `_compute_loss`.
- **`_compute_loss`** (`grpo_trainer.py:260`):
    - Calls `model.forward(messages, mode='agent')` → triggers composer fresh (differentiable) → agent forward → returns logits + labels.
    - `per_token_logps = selective_log_softmax(logits[:, :-1], labels[:, 1:])`.
    - `old_per_token_logps = per_token_logps.detach()` (paper uses `num_iterations=1`, so old = current).
    - PPO ratio: `coef_1 = exp(per_token_logps - old_per_token_logps)`, clipped to $[1-\varepsilon, 1+\varepsilon]$ (low/high).
    - Loss = `-min(coef_1*adv, coef_2*adv)` with optional `+ beta*KL` (default $\beta=0$).
    - **Three reductions** supported:
        - `grpo`: per-traj mean over completion mask.
        - `bnpo` (used in paper): **sum / completion_count** (token-level) — matches eq. (11) in paper.
        - `dr_grpo`: sum over batch / `(B * max_completion_length)`.
- **What gets trained**: only `LatentMem` module params with `requires_grad=True` — composer LoRA adapters, `query_latents`, `weaver2agent_proj`. Agent backbones frozen via `fix_model_parameters`.

#### Evaluation (`mode=eval`, `runner.py:evaluate`)
- Same multi-turn interaction loop on `test_dataset`.
- Each successful test trajectory is **also added back to experience bank** — incremental online accumulation, matching paper's continual-adaptation claim.

### Configs / Hyperparameters

From `configs/latentmem/triviaqa.yaml` (representative):

| Group | Param | Value |
|---|---|---|
| memory.weaver | `latents_len` ($L'$) | **8** |
| memory.weaver | use_peft | True |
| memory.weaver.peft_config | r | 16 |
| memory.weaver.peft_config | lora_alpha | 32 |
| memory.weaver.peft_config | target_modules | `[q_proj, v_proj]` |
| memory.weaver.peft_config | lora_dropout | 0.1 |
| memory.rag | mode | `latentmem` (data collection often via `metagpt` baseline) |
| memory.rag | pos_shots_num | 1 (= $K$) |
| memory.rag | neg_shots_num | 0 |
| memory.rag | insights_num | 3 |
| memory.rag | embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| run.lmpo | num_generations ($G$) | 4 (yaml default) / 8 (`lmpo.sh`) |
| run.lmpo | num_iterations | 1 |
| run.lmpo | beta (KL) | **0.0** |
| run.lmpo | loss_type | `bnpo` |
| run.lmpo | learning_rate | 1e-5 |
| run.lmpo | optim | AdamW |
| run.lmpo | warmup_ratio | 0.1 |
| run.lmpo | lr_scheduler | cosine |
| run.lmpo | max_prompt_length | 1024 |
| run.lmpo | max_completion_length | 256 |
| run.lmpo | temperature | 1.0 |
| run.lmpo | num_train_epochs | 1 |
| run.lmpo | bf16 | True |
| run.generation (eval) | do_sample | False (temperature 0.0) |
| run.generation | max_new_tokens | 512 |
| run.interaction | batch_size | 16 |
| run.interaction | max_turns | 5 |
| run.interaction | max_obs_length | 2048 |

- Paper-level (Table 3) additional: prompt limit 10240, response 4096, gradient clipping 1.0, flash attn + vLLM + DeepSpeed (`configs/zero2.yaml`).

### Notes / sharp edges in code
- **Two-stage training in README** (SFT then LMPO) — though main paper text emphasizes LMPO only; SFT acts as a warmup (`MemMasterSFTTrainer`).
- **Composer's `query_latents` may be re-used across agents** — role-awareness comes purely from the *textual role profile in the prompt* fed to the composer, not from per-role learnable queries. This is interesting design: a single composer learns to emit role-conditional latents.
- **Inference text-memory suppression**: in `_trigger_memory`, once latent_memory is produced, text_memory is overwritten with `EMPTY_MEMORY`. So at inference the agent literally never sees the textual trajectory — only the 8 latent tokens (huge token saving, matches paper's cost claims).
- **Composer LLM is loaded as full bf16 backbone**, even though only LoRA params + `query_latents` + projection are trained → still expensive memory-wise (≥4B params resident).
- **Reranker LLM in ExperienceBank** runs another LLM forward pass for retrieval scoring — non-trivial overhead, partially contradicts the "no human priors / no extraction" framing but is consistent in spirit (learned scoring vs handcrafted rules).
- **`_get_per_token_logps`** uses `selective_log_softmax` from TRL and shifts by 1 like standard CLM training, with `labels==-100` masked out (corresponds to prompt + latent positions).
