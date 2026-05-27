# RL Overview: Post-Training, Self-Play, and Reward Design (2025–2026)

A landscape report on ~60 papers/blogs covering **on-policy distillation**, **exploration on hard problems**, **self-play & environment scaling**, **reasoning structure & abstractions**, **reward design (rubrics, dense/sparse, adversarial)**, and **multi-agent / feedback / experiential RL**. Grouped by theme; cross-cutting synthesis at the end.

> **Frame**: most of these papers attack one of three structural failure modes of vanilla RLVR on LLMs — **(1)** sparse outcome reward → no signal on hard problems, **(2)** distribution mismatch between training-state (SFT) and rollout-state (RL), and **(3)** absent / hackable reward for non-verifiable tasks. Each cluster is a different attack on (1)–(3).

---

## 1. On-Policy Distillation (OPD)

> **Theme**: student rolls out → teacher grades each token. SFT (off-policy, dense) and RL (on-policy, sparse) → OPD (on-policy, dense). The "missing quadrant" of post-training.

### On-Policy Distillation (2025) — Thinking Machines Lab, Kevin Lu — seminal blog
- **Core**: student samples its own trajectories; teacher provides per-token feedback via **reverse KL**. Combines RL's on-policy realism with SFT's dense supervision.
- **Loss**: $D_{KL}(\pi_{student}(\cdot|s_t) \| \pi_{teacher}(\cdot|s_t))$ summed over student rollout tokens. Penalizes "forking tokens" most.
- **Math results (AIME'24, Qwen3-8B)**: SFT 60% → +RL (17,920 GPU-h) 67.6% → +OPD (1,800 GPU-h) **74.4%**. ~9–30× cheaper than RL.
- **Self-distill recipe**: reaches teacher perf 7–10× faster than RL.
- **Assistant recipe**: mid-train on internal docs (kills IFEval), then OPD vs the original model on chat data — recovers instruction-following without losing the new domain knowledge.
- **Key terms**: reverse KL, dense per-token reward, on-policy student rollouts, knowledge recovery.

### SFT, RL, and On-Policy Distillation Through a Distributional Lens (2026) — nrehiew blog
- **Core**: a unified view — each post-training method maximizes capability under a different KL-to-prior tradeoff. OPD sits between SFT and RL: teacher *signal* like SFT, data *source* like RL.
- **Method**: frames LM as distribution over sequences; each PT method reshapes that distribution differently, treating different targets as the ground truth.
- **Connection**: theoretical lens that motivates the empirical wins of [[on-policy-distillation-tm]].

### Multi-Teacher On-Policy Distillation (MOPD) (2026) — yumoxu notion / MiMo-V2-Flash
- **Core**: generalization of OPD where the student learns from **multiple** specialized teachers simultaneously, weighted by an RL-guided selector.
- **Method**: 3-stage pipeline — (1) general SFT; (2) domain-specific RL/SFT to train multiple expert teachers; (3) MOPD with dense token-level rewards from teachers + outcome-based verifier reward. RL policy adaptively picks which teacher's signal to follow per-token.
- **Connection**: ensemble-of-teachers extension of [[on-policy-distillation-tm]]; addresses capability imbalance when one teacher isn't dominant.

### On-Policy Self-Distillation / Self-Distilled Reasoner (2026) — `arXiv:2601.18734`, Siyan Zhao et al.
- **Core**: single LLM acts as **both teacher and student** with different contexts. Teacher conditions on **privileged info** (ground-truth solution / execution feedback); student doesn't.
- **Method**: OPSD trains the unconditioned student to match the privileged-conditioned teacher's per-token distribution. Reverse KL on student rollouts.
- **Results**: superior token efficiency vs RL; beats off-policy distillation on math reasoning benchmarks. No separate larger teacher needed.
- **Connection**: bridges [[on-policy-distillation-tm]] and [[privileged-info-distill]] — same model, different context = privileged self-teacher.

### Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why (2026) — `arXiv:2605.10889`, Farajtabar et al.
- **Core**: **diagnostic** paper. Per-token / per-question / per-teacher framework that measures when OPD signal actually aligns with optimal gradient.
- **Method**: derives an ideal per-node gradient + scalable targeted-rollout estimator → **gradient alignment score**.
- **Findings**: OPD guidance aligns more strongly on **incorrect** rollouts (i.e., teacher most useful when student is wrong); optimal context depends on student capacity and task.
- **Connection**: empirical scalpel applied to [[on-policy-distillation-tm]] — explains the variance in OPD results across the cluster.

### On-Policy Context Distillation (OPCD) (2026) — `arXiv:2602.12275`, Ye, Dong et al.
- **Core**: student rolls out, teacher = same model but **conditioned on extra context** (system prompt / experience). Distill the context-conditioned behavior into the unconditioned model.
- **Method**: reverse KL between student and context-conditioned teacher on student rollouts. Two applications: **(a)** experiential knowledge distillation from past traces, **(b)** system-prompt distillation (internalize a tuned prompt's effects).
- **Results**: outperforms baselines on math + text games + domain tasks; better OOD preservation than SFT.
- **Connection**: same model trick as [[on-policy-self-distillation]] but with context (not ground truth) as the privilege.

### Online Experiential Learning (OEL) (2026) — `arXiv:2603.16856`, Ye, Dong et al. — Part II of OPCD series
- **Core**: harvests **deployment-time interactions** as training signal. Two-stage: extract transferable experiential knowledge from real-user trajectories → consolidate into params via [[opcd]].
- **Results**: iterative gains on task accuracy + token efficiency, preserves OOD perf. Extracted experiential knowledge >> raw trajectories.
- **Connection**: direct downstream application of [[opcd]] to continual / deployment-driven learning.

### Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents (ODE) (2026) — `arXiv:2605.10832`
- **Core**: closed-loop data generator that **refines itself across rounds from rollouts of the policy being trained**. Image-bank reference protocol registers every tool-returned image as addressable.
- **Results**: Qwen3-VL-8B agent: 24.9% → **39.0%** avg on 8 multimodal deep-search benchmarks. Beats Gemini-2.5 Pro (37.9%) in agent-workflow setting.
- **Connection**: OPD spirit applied to **data evolution** for agents rather than per-token distillation.

### Reinforcement Learning for Knowledge Awareness (2026) — kalomaze blog
- **Core**: reframe reward modeling as **generalized density-ratio estimation** to teach models to *know what they don't know*. Solves the near-zero baseline rate problem for behaviors like hedging on confabulated entities.
- **Method**: single linear projection from one token of one layer of Qwen3-8B → Bradley-Terry reward. Synthetic dataset pairs real (Wikipedia-grounded) vs fake (invented) researcher prompts. Prefix-conditioned generation, prefix-stripped for RM training.
- **Results**: ~300 RL steps → model refuses / hedges on fake people while retaining knowledge of real ones.
- **Connection**: thematic cousin to [[opd-tm]]; representation-extracted dense signal vs distilled dense signal. Linked here because it's invoked in the OPD-thread as a low-cost dense-reward recipe.

### Learning from Rare Success and Rich Feedback via Reflection-Enhanced Self-Distillation (2026) — YuweiZh
- **Note**: couldn't locate concrete arxiv ID; description from cluster context.
- **Core**: model retrospectively reflects on rare-success traces, treats feedback-conditioned next-token distribution as a self-teacher, distills back into the policy. Avoids external teacher / reward model entirely.
- **Connection**: same self-teacher trick as [[opsd]] but conditioned on rich tokenized feedback (runtime errors, judge critiques) rather than ground-truth solutions.

**Group synthesis (OPD)**:
- **Two axes**: *who's the teacher* — separate larger model (Thinking Machines), multiple teachers (MOPD), the same model under privileged context (OPSD / OPCD), or no teacher (self-reflection on rich feedback); *what's the signal* — token KL (most), context-distill, privileged-info distill.
- **Common claim**: OPD wins **5–30× compute** vs RL at matched final accuracy. Reverse KL on student rollouts is the load-bearing primitive.
- **Open question (Unmasking)**: OPD doesn't help uniformly — biggest gain is on rollouts where the student is wrong; OPD on correct rollouts can hurt. Suggests **selective OPD** as the next frontier.

---

## 2. Exploration on Hard Problems

> **Theme**: standard RLVR fails when the student never samples a correct rollout (no reward → no gradient). Family of fixes: give the model a leg up via demos, privileged info, prefixes, or curriculum.

### How to Explore to Scale RL Training of LLMs on Hard Problems? (2025) — CMU ML blog, Qu, Setlur, Smith, Salakhutdinov, Kumar
- **Core**: **framework paper** for the whole cluster. Names the failure mode: on-policy RL never sees a correct rollout on hard problems → stalled learning + plateau before max reward.
- **Method**: use human/oracle solutions as **privileged guidance** — prepend a minimal **solution prefix** to hard prompts during rollout to bootstrap reward. Behaviors generalize back to unconditioned prompts.
- **Connection**: spawns [[pope]], [[prefixrl]], [[srl]], [[pi-distill]] — different instantiations of the same idea.

### POPE: Privileged On-Policy Exploration (2026) — `arXiv:2601.18779`
- **Core**: concrete algorithm for [[explore-hard]]. Prepend oracle-solution prefixes only to hard prompts → on-policy RL gets non-zero reward → student learns; transfers to unguided setting.
- **Key insight**: mixing easy + hard problems is counterproductive due to **ray interference** — optimization commits to already-solvable problems and actively inhibits hard ones. Privileged prefixes break this.
- **Results**: expands the solvable set, improves perf on challenging reasoning benchmarks.

### PrefixRL — Reuse your FLOPs (2026) — `arXiv:2601.18795`, Setlur, Xie, Cohen, Rashidinejad, Wang
- **Core**: same prefix idea but at compute level. Prepend **off-policy** prefixes from successful traces and run on-policy RL to *complete* them. Sidesteps the off-policy instabilities of full SFT-on-rejection-sampled data.
- **Knob**: prefix length controls problem difficulty — long prefix = easy completion, short = hard exploration.
- **Results**: 2× higher final training accuracy vs SFT-on-rejection + RL, FLOPs-matched.
- **Connection**: more aggressive cousin of [[pope]] — uses *very* off-policy (high-quality SFT) prefixes rather than oracle templates.

### Privileged Information Distillation (π-Distill) (2026) — `arXiv:2602.04942`, Peñaloza et al. (ICML 2026)
- **Core**: train a PI-conditioned teacher and unconditioned student **simultaneously with shared parameters**. The teacher learns to use PI; the shared-param coupling forces transfer.
- **Method**: two objectives — **π-Distill** (joint shared-param training) and **OPSD** (RL with reverse-KL penalty against PI-conditioned teacher).
- **Results**: both beat the SFT-then-RL industry default across agentic benchmarks, models, and PI types.
- **Connection**: theoretical complement to [[pope]] — distills the prefix-conditioned behavior into the unconditioned policy.

### Supervised Reinforcement Learning (SRL) (2025) — `arXiv:2510.25992`, Google
- **Core**: reformulate problem-solving as a sequence of **logical actions**; for each prefix of an expert trace, train the student to first emit a free reasoning monologue and then commit an action; reward the student on **similarity to expert action** (step-wise dense reward), not token-by-token.
- **Method**: per-step dense reward even when final answer is wrong; reasoning span is unconstrained — model searches its own chain.
- **Results**: +3.7% on math vs SFT/RLVR; **+74% over SFT** on software engineering agentic tasks.
- **Connection**: turns expert trajectories into a dense reward channel — bridges SFT's density and RL's exploration; alternative to [[on-policy-distillation-tm]] when you have demos but no live teacher.

### Revisiting DAgger in the Era of LLM-Agents (2026) — `arXiv:2605.12913`
- **Core**: classical DAgger applied to long-horizon LM agents. Collect trajectories via **turn-level stochastic mix** of student and teacher turns → train student on these mixed traces. Gradually exposes student to its own state distribution.
- **Results**: +3.9 pts (4B) and +3.6 pts (8B) on SWE-bench Verified over best post-training baseline. 4B agent hits **27.3%**, beating published 8B SWE-agents.
- **Connection**: imitation-side complement to [[srl]]; DAgger structurally solves the same covariate-shift problem OPD addresses on the distillation side.

### Learning from Mixed Rollouts: Logit Fusion as a Bridge Between Imitation and Exploration (2026) — Juzheng Zhang
- **Core**: per-token blend of teacher and student **logits** during rollout, interpolating between pure imitation (teacher) and pure exploration (student). Trains on the fused trajectory.
- **Connection**: turn-level mixing → token-level mixing extension of [[dagger-llm]]. Sits between SFT and on-policy RL on a continuous knob.

### TCOD: Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents (2026) — `arXiv:2604.24005`
- **Core**: identifies **Trajectory-Level KL Instability** in OPD for multi-turn agents — inter-turn error compounding drives student beyond teacher's support, supervision becomes unreliable.
- **Fix**: curriculum on trajectory depth — student sees short trajectories first, expanding over training.
- **Result**: can **surpass teacher** and generalize to tasks where teacher fails.
- **Connection**: scaling fix for [[on-policy-distillation-tm]] when applied to multi-turn agents; analogous to [[srl]]'s step-wise dense reward but on the OPD side.

### Every Step Evolves: Scaling RL for Trillion-Scale Thinking Model (Ring-1T) (2025) — `arXiv:2510.18855`
- **Core**: systems paper. First open-source 1T-param thinking model (50B active per token). Three engineering primitives for trillion-scale RL stability.
- **Method**: **IcePop** — token-level discrepancy masking/clipping to fix train-inference numerical mismatch; **C3PO++** — dynamic rollout partitioning under token budget; **ASystem** — RL framework removing the systemic bottlenecks at trillion-scale.
- **Results**: AIME-2025 **93.4**, HMMT-2025 86.72, CodeForces 2088, ARC-AGI-v1 55.94, IMO-2025 silver-medal.
- **Connection**: orthogonal axis — algorithmic clusters above only matter if RL training itself is numerically stable at scale.

**Group synthesis (hard-problems exploration)**:
- **Three knobs**: *what* you condition on (oracle solution prefix, expert action, full PI context), *how much* you condition (length / curriculum), *whether* it persists (transient prefix at training only vs shared-param distill).
- **PrefixRL vs POPE**: same shape, different prefix source (off-policy SFT data vs oracle templates). Both rescue zero-gradient regimes.
- **Convergent finding**: privileged conditioning lifts a hard problem into the solvable regime; the *behavior* generalizes to the unconditioned setting via either KL distillation (π-Distill) or behavioral cloning (SRL).
- **Open question**: how to **discover** good prefixes/PI automatically vs needing humans / a stronger oracle.

---

## 3. Self-Play & Environment Scaling

> **Theme**: get rid of the prompt/data bottleneck. The model is its own curriculum: Challenger ↔ Solver, or a population that co-evolves. Diversity is the central failure mode.

### Language Self-Play (LSP) (2025) — `arXiv:2509.07414`
- **Core**: single LLM alternates Challenger (queries) and Solver (responses) roles. No external data required.
- **Results**: Llama-3.2-3B-Instruct improves on instruction-following beyond data-driven baselines; up to 43.1% win rate on open-ended tasks (post-RL calibration).
- **Connection**: seminal data-free RL paper; sets up the whole self-play cluster.

### Reverse-Engineered Reasoning (REER) (2025) — `arXiv:2509.06160`
- **Core**: instead of forward RL, work **backwards** from known-good solutions to discover the latent reasoning trace that could produce them. Gradient-free.
- **Output**: DeepWriting-20K (20K open-ended reasoning trajectories); DeepWriter-8B beats strong open-source baselines.
- **Connection**: data-generation alternative to RL/distillation for non-verifiable domains (creative writing); same goal as [[compute-as-teacher]] (reference-free supervision).

### Diversity as the Bottleneck in Self-Play (2026) — Ivison blog
- **Core**: self-play loops collapse because models exploit "Challenger struggles" by emitting tiny variations of one solution. Without diversity reward, this dominates.
- **Evidence**: bug in original Absolute Zero codebase masked the problem. Three diversity proxies tested: LM judge (hackable via prompt size), embedding cosine (creates surface-different / structurally same), corpus conditioning (collapses after 600 steps).
- **Suggested fixes**: adversarial embedding model; trained LM judge; broader-than-code-IO setups; entropy-preservation.
- **Connection**: critical reading of [[absolute-zero]] / [[lsp]] cluster — names the recurring failure mode.

### ARE: Scaling Up Agent Environments and Evaluations (2025) — `arXiv:2509.17158`, Meta Superintelligence Labs
- **Core**: research platform for **scalable creation of agentic environments**. Key differentiator: environments **change dynamically** (random / scheduled events) — they don't pause while agent acts.
- **Companion benchmark**: **Gaia2** — measures handling of ambiguities, noise, dynamic envs, multi-agent collab, temporal constraints.
- **Connection**: substrate for many of the self-play and agentic RL papers above; addresses (1) of the core failure modes via more / better envs.

### Paying Less Generalization Tax (2026) — `arXiv:2601.18217`
- **Core**: cross-domain study of RL post-training. Two environment axes predict generalization: **state information richness** and **planning complexity** (goal reachability, traj length).
- **Findings**: domain realism / text similarity *don't* matter much. Sokoban → SciWorld generalizes better than ALFWorld → SciWorld. Step-by-step thinking during RL preserves generalization. SFT warmup prevents catastrophic forgetting but undermines generalization to unseen domains.
- **Fix**: add small distractive features to state ("richness randomization").
- **Connection**: empirical foundation for picking [[are]]-style training envs that transfer.

### G-Zero: Self-Play for Open-Ended Generation from Zero Data (2026) — `arXiv:2605.09959`
- **Core**: **verifier-free** co-evolutionary self-play. Proposer generates queries + hints; Generator produces responses.
- **Reward**: **Hint-δ** — intrinsic reward = predictive shift between unassisted response and hint-conditioned response. Quantifies "how much did the hint help."
- **Algorithm**: Proposer trained via GRPO (target Generator's blind spots); Generator via DPO (internalize hint-guided improvements).
- **Theory**: best-iterate suboptimality guarantee under coverage + low-noise assumptions.
- **Connection**: open-ended cousin of [[absolute-zero]]; sidesteps reward hacking by using *intrinsic predictive shift* instead of LM judges.

### Bootstrapping Task Spaces for Self-Improvement (ExIt) (2025) — `arXiv:2509.04575`, Jiang, Lupu, Bachrach (Meta)
- **Core**: autocurriculum RL — train on the **most informative single-step iterations** at training time, deploy multi-step self-improvement at inference. Doesn't pre-commit to a max iteration depth.
- **Method**: exploits recurrent structure of self-improvement → self-generated data augmentation over the task space.
- **Domains**: math (single-turn), tool-use (multi-turn), ML engineering on Kaggle (search-scaffold).
- **Connection**: meta-curriculum to the [[lsp]] / [[g-zero]] line; tackles "what's the right horizon for self-improvement RL."

### Compute as Teacher (CaT) (2025) — `arXiv:2509.14234`
- **Core**: **inference compute → supervision**. Aggregate parallel rollouts into a pseudo-reference, derive RL reward from it. Amortizes test-time scaling into weights via RL.
- **Critical claim**: works in **non-verifiable** domains (e.g. healthcare) where no programmatic checker exists. Rubric-based scoring > LLM-as-judge for stability.
- **Connection**: same reference-free supervision spirit as [[g-zero]] and [[reer]] but uses parallel rollouts not hints/back-engineering.

### PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play (2026)
- **Core**: population-based asymmetric self-play. Co-evolving populations of teacher and student LoRA adapters; teachers propose verifiable tasks, students solve, verifier scores.
- **Task types** (in code reasoning): code_o (predict output), code_i (find input for target output), code_f (complete missing function).
- **Connection**: population-level extension of [[lsp]]'s Challenger/Solver; LoRA keeps the co-evolution cheap.

### Intrinsic Credit Assignment for Long Horizon Interaction (ΔBelief-RL) (2026) — `arXiv:2602.12342`
- **Core**: reward **intermediate progress** by the model's own change in probability assigned to the target solution. Self-generated dense signal for long-horizon info-seeking.
- **Results**: outperforms outcome-only RL; generalizes OOD (customer-service, personalization); perf keeps improving as test-time interactions exceed training horizon (interaction-efficient even on Pass@k).
- **Connection**: long-horizon dense-reward analogue to [[hint-delta]] from G-Zero — both use predictive shift in the model's own beliefs.

### Learning Self-Correction in VLMs via Rollout Augmentation (Octopus) (2026) — `arXiv:2602.08503`, ICML 2026
- **Core**: self-correction behaviors are too rare in RL rollouts → sparse signal. Synthesize **correction-specific rollouts** by recombining existing rollouts.
- **Results**: best avg accuracy across 7 VLM benchmarks with substantially less rollout time. Releases Octopus-8B with controllable self-correction.
- **Connection**: directly tackles [[diversity-bottleneck-selfplay]]-style rare-success failure mode for self-correction specifically.

**Group synthesis (self-play)**:
- **Three families**: *single-agent role-switching* (LSP, ExIt), *co-evolving populations* (PopuLoRA, G-Zero), *compute-as-supervision* (CaT, REER).
- **Recurring failure mode**: diversity collapse. The Ivison blog is the clearest articulation; G-Zero's intrinsic Hint-δ and CaT's parallel rollouts are designed-in defenses.
- **Environment story**: ARE / Paying-Less-Generalization-Tax argue substrate matters more than algorithm — dynamic, richness-rich envs generalize where realistic-but-static ones don't.
- **Common ambition**: zero external data. Common defeat: without a diversity or correctness anchor (verifier / human-grounded prompt), the loop drifts.

---

## 4. Reasoning Structure & Abstractions

> **Theme**: not all chains-of-thought are equal. Longer ≠ better. The right unit of reasoning may not be tokens but abstractions / action chunks / states with low intrinsic dimension.

### What Characterizes Effective Reasoning? (2025) — `arXiv:2509.19284`
- **Core**: contradicts "longer is better." Naive CoT lengthening and increased review both **correlate with lower accuracy**.
- **New metric**: **Failed-Step Fraction (FSF)** = fraction of CoT steps in abandoned branches (extracted from a graph view of the CoT). Outpredicts length and review ratio.
- **Result**: FSF-based test-time selection gives up to **+10% on AIME**, the largest and most consistent gain.
- **Connection**: empirical foundation for length-aware methods like [[sas]].

### Stabilizing Efficient Reasoning with Step-Level Advantage Selection (SAS) (2026) — `arXiv:2604.24003`, ACL 2026 Findings
- **Core**: applies advantage selection at the **step level** (not token / not sequence). Variants: apply SAS to correct-only, wrong-only, or both rollouts.
- **Evaluation**: AIME24/25, MATH, AMC, Olympiad-Bench; general benchmarks GPQA-Diamond, LSAT, MMLU.
- **Connection**: operationalizes [[effective-reasoning-cot]]'s insight — discount the noisy steps.

### Learning to Reason in 13 Parameters (TinyLoRA) (2026) — `arXiv:2602.04118`, John X. Morris et al.
- **Core**: **TinyLoRA** scales LoRA down to 1-parameter adapters. Trains 8B Qwen2.5 from 76 → **91% on GSM8K with only 13 parameters (26 bytes in bf16)**.
- **Generalization**: recovers 90% of perf gains with **1000× fewer parameters** on AIME / AMC / MATH500.
- **Crucial finding**: works **only with RL**. SFT requires 100–1000× larger updates.
- **Connection**: empirical demo of [[intrinsic-dimensionality]]'s thesis — reasoning improvement lives in a tiny subspace.

### Effective Reasoning Chains Reduce Intrinsic Dimensionality (2026) — `arXiv:2602.09276`, Prasad et al.
- **Core**: introduces **intrinsic dimensionality** = minimum model dimensions needed to hit accuracy threshold on a task. Effective reasoning strategies **reduce** it.
- **Method**: fix architecture, vary CoT strategy. Strong inverse correlation between intrinsic dim and generalization on GSM8K (Gemma-3 1B/4B).
- **Interpretation**: good CoT *compresses* the task — better reasoning = fewer effective dims needed = better generalization.
- **Connection**: theory cousin to [[tinylora-13]]; explains why RL can succeed with tiny adapters.

### Learning to Reason for Factuality (2025) — `arXiv:2508.05618`, Meta
- **Core**: R-LLMs hallucinate more than non-reasoning models on long-form factuality. Online RL with a **composite reward** = factual precision × detail level × answer relevance.
- **Results**: average **−23.1 pts hallucination rate**, +23% detail, no helpfulness drop across 6 long-form factuality benchmarks.
- **Connection**: factuality-specific instance of the [[hybrid-reward]] philosophy — composite signal beats single-axis reward.

### RLAD: Training LLMs to Discover Abstractions (2025) — `arXiv:2510.02263`
- **Core**: **two-player RL** — abstraction generator proposes natural-language descriptions of procedural/factual knowledge; abstraction-conditioned solver answers.
- **Why**: existing RL post-training optimizes "depth" → narrow exploration. RLAD optimizes "breadth" → diverse solution strategies.
- **Results**: **+44% over DAPO** on AIME 2025 average; 42.45% (avg abs) / 48.33% (best abs) vs DAPO 34.90 / 39.79.
- **Connection**: action-space restructuring — same goal as [[ra3]] but at the level of explicit NL abstractions rather than latent action subspace.

### Learning to Reason as Action Abstractions (RA3) (2025) — `arXiv:2509.25810`, Apple ML Research
- **Core**: scalable mid-training algorithm. Derives sequential variational lower bound; iteratively discovers **temporally-consistent latent action subspaces** via RL; then fine-tunes on bootstrapped data.
- **Theory**: characterizes the action subspace minimizing both value-approximation error (from pruning) and RL error (during planning).
- **Results**: +8 / +4 on HumanEval / MBPP over base + NTP; faster convergence + higher asymptotic perf on HumanEval+/MBPP+/LCB/Codeforces.
- **Connection**: implicit (latent) abstraction discovery; complement to [[rlad]]'s explicit NL abstractions.

### Dual Goal Representations (2026) — `arXiv:2510.06714`, Seohong Park, ICLR 2026
- **Core**: for goal-conditioned RL, represent a state by **its set of temporal distances to all other states**. Depends only on intrinsic dynamics; invariant to state representation; filters exogenous noise.
- **Provable**: sufficient to recover optimal goal-reaching policy.
- **Results**: consistent improvement across 20 state- and pixel-based OGBench tasks; works as add-on to any GCRL method.
- **Connection**: not LLM-specific but maps onto agent-RL theme — representation choice matters more than reward shaping when long-horizon credit is the bottleneck.

**Group synthesis (reasoning structure)**:
- **Convergent thesis**: the *unit* of reasoning that matters is not the token. It's the **step** (SAS), **abstraction** (RLAD, RA3), or **belief shift** (ΔBelief-RL).
- **Length/intrinsic-dim duality**: Effective CoT *compresses* the task (intrinsic dim ↓), which is why RL on a 13-param subspace can work (TinyLoRA). FSF metric is the empirical proxy for the same insight.
- **Connection to OPD cluster**: OPD's per-token signal is dense but may overweight unimportant tokens. Step-level / abstraction-level signals (SAS, RLAD) are complementary granularity.

---

## 5. Reward Design: Dense, Rubric, Adversarial

> **Theme**: how do you get reward signal in non-verifiable / sparse domains without it being trivially hackable? Dense reward models, rubrics, adversarial critics, flow-matching critics.

### Polychromic Objectives for RL (2025) — `arXiv:2509.25424`, Hamid et al., ICLR 2026
- **Core**: policy gradient that **explicitly enforces diversity**. Defined over *sets* of trajectories: score high only if set contains both successful and diverse trajs.
- **Factorization**: reward component (+cov with return) × diversity component (−cov with homogeneity), equal range.
- **Implementation**: set RL with vine sampling (on-policy rollouts), modified advantage in PPO.
- **Connection**: addresses the same mode-collapse failure mode as [[diversity-bottleneck-selfplay]]; structural fix rather than self-play patch.

### Hybrid Reinforcement (HERO) (2025) — `arXiv:2510.07242`, Weston et al.
- **Core**: integrate **verifier (sparse binary) + reward model (dense)** signals. Verifier-defined groups bound RM scores via **stratified normalization** → preserves correctness while refining quality.
- **Plus**: **variance-aware weighting** — emphasize prompts where dense signals matter most.
- **Results**: +11.7 pts vs RM-only; +9.2 pts vs verifier-only on hard-to-verify reasoning.
- **Connection**: explicit instantiation of "best of both worlds" — same spirit as [[learning-to-reason-factuality]]'s composite reward.

### RLAC: Reinforcement Learning with Adversarial Critic (2025) — `arXiv:2511.01758`, Wu, Zhang, Min, Levine, Kumar
- **Core**: free-form generation as **adversarial game**. LLM critic dynamically identifies most-likely failure modes (factual error, edge case) → external validator confirms → joint optimization of generator + critic.
- **Why adversarial**: critic continuously adapts to generator → prevents reward hacking, sustains meaningful supervision.
- **Results**: improves factual accuracy in text + correctness in code; beats exhaustive verification + reward-model methods.
- **Connection**: dynamic-critic answer to [[reward-hacking-rubric]]'s static-rubric failure.

### InT: Self-Proposed Interventions for Credit Assignment (2026) — `arXiv:2601.14209`
- **Core**: fine-grained credit assignment by **self-proposing single-step interventions**. Model identifies first error in its own trace, proposes a one-step correction, then SFTs on (correct-prefix + intervention).
- **Insight**: verifying is easier than generating → use reference solutions to locate error step, train on the local fix.
- **Results**: +14% over 4B base on IMO-AnswerBench; **beats gpt-oss-20b**.
- **Connection**: per-step credit alternative to [[sas]]; uses ground-truth references like [[opsd]] / [[pi-distill]].

### Reward Hacking in Rubric-Based RL (2026) — `arXiv:2605.12474`
- **Core**: trains against a rubric-based verifier, evaluates against a **cross-family panel of 3 frontier judges**.
- **Separates two failure sources**: (a) **verifier failure** — training verifier credits criteria reference verifiers reject; (b) **rubric-design limitations** — even strong rubric-verifiers favor responses other judges rate worse.
- **Finding**: stronger verification doesn't prevent hacking when rubric leaves failure modes unspecified. Gains concentrate in completeness/presence-based criteria; declines in factual correctness, conciseness, relevance.
- **Connection**: cautionary tale for the rubric-RL line ([[rar]], [[rubric-anchors]], [[rubricem]]).

### Reward Hacking — Prime Intellect (2026) — blog
- **Core**: argues reward hacking is a **dynamics** problem, not just a specification problem.
- **Setup**: backdoor-IFEval-style environments with "hidden" keyword rewards → study hack frequency vs task difficulty. Patterns: harder tasks → more hacking.
- **Connection**: empirical complement to [[reward-hacking-rubric]] — when reward is hard to specify and task is hard to solve, models prefer the cheat path.

### RubricEM: Meta-RL with Rubric-guided Policy Decomposition (2026) — `arXiv:2605.10899`
- **Core**: rubrics not just as evaluators but as the **shared interface** between policy, judge, and agent memory. For deep research agents (plan / search / synthesize).
- **Method**: **Stage-Structured GRPO (SS-GRPO)** — score Plan, Research, Review, Answer stages with **stage-specific** rubrics instead of broadcasting one terminal score to all tokens. Reflection-based meta-policy evolution.
- **Connection**: stage-level credit-assignment via rubric — addresses long-horizon failure of vanilla GRPO. Cousin of [[srl]] (step-wise) and [[tcod]] (turn-wise curriculum).

### What Does Flow Matching Bring To TD Learning? (2026) — `arXiv:2603.04333`, Agrawalla, Nauman, Kumar
- **Core**: theory + empirical on **flow-matching critics** for scalar Q-value estimation. Two mechanisms: (a) iterative integration dampens early-estimate errors at test time; (b) supervising velocity field at multiple interpolant values induces **more plastic features** that handle non-stationary TD targets.
- **Results**: **2× final performance, ~5× sample efficiency** vs monolithic critics in high-UTD online RL.
- **Connection**: orthogonal to LLM-RL but explains why diffusion/flow priors are eating the value-learning side; relevant to actor-critic LLM training.

### Expected Reward Prediction & Model Routing (2026) — `arXiv:2603.20217`
- **Core**: predict the **expected reward** a model would get on a prompt **before** generating responses. Lifts response-level RM scores to prompt-level model suitability.
- **Application**: routing protocol — assign prompts to models at inference to maximize reward under cost budget.
- **Empirical**: open-perfectblend dataset, pool of Llama-3.1-8B/70B, Gemma2 9B/27B, Gemma1 7B.
- **Connection**: applies RM signal at planning time, not learning time — cheap test-time alternative to per-prompt RL.

### Not Every Rubric Teaches Equally: Policy-Aware Rubric Rewards for RLVR (2026)
- **Note**: couldn't locate concrete arxiv link.
- **Core (from cluster + cousin paper RuCL)**: not all rubric criteria are equally learnable for a given policy. Re-weight rubric criteria based on what's actually informative for the current student.
- **Connection**: principled fix to [[reward-hacking-rubric]] — match rubric pressure to where the policy is uncertain rather than uniform.

**Group synthesis (reward design)**:
- **Three failure modes addressed**: *sparse* (Hybrid/HERO, ΔBelief), *hackable* (Reward Hacking studies, RLAC adversarial), *uniformly-weighted* (RubricEM, Policy-Aware, SAS).
- **Convergent design**: best results come from **multi-source rewards** (verifier + RM + judge + intrinsic signal) with structural decoupling (stratification, stage-specific, set-level diversity).
- **Adversarial vs static**: RLAC's dynamic critic prevents the exact failure [[reward-hacking-rubric]] documents in static rubrics.
- **Outside-LLM tools**: flow-matching critics suggest the next move on the value-function side.

---

## 6. Multi-Agent, Feedback, and Experiential RL

> **Theme**: scale beyond single-agent prompts. Language as the reward language, multi-agent emergence from a single reasoner, deployment experience as the training signal.

### Reasoning Models Generate Societies of Thought (2026) — `arXiv:2601.10825`, Kim et al. (Google + UChicago)
- **Core**: empirical finding — RL-trained reasoning models (DeepSeek-R1, QwQ-32B) generate **internal multi-agent dialogues** with distinct personalities and expertise. "Society of thought."
- **Method**: probe perspective-diversity features. Show base models *increase* conversational behaviors when rewarded purely for reasoning accuracy. Fine-tuning with conversational scaffolding accelerates reasoning gains.
- **Interpretation**: emergent multi-agent structure parallels collective intelligence in human groups.
- **Connection**: explains *why* multi-agent setups work — and why long-CoT models implicitly do something similar.

### The End of Reward Engineering: How LLMs Are Redefining Multi-Agent Coordination (2026) — `arXiv:2601.08237`
- **Core**: position paper. Multi-agent RL's hard problem is reward engineering (credit assignment ambiguity, non-stationarity, combinatorial interaction). LLMs let us specify objectives in **natural language** instead.
- **Three dimensions**: semantic reward specification, dynamic reward adaptation, alignment with human intent.
- **Building blocks cited**: EUREKA (reward synthesis from NL), CARD (online reward adaptation), RLVR.
- **Connection**: framing piece that motivates the rubric/RM/judge-as-reward cluster (Section 5) in multi-agent settings.

### RLAnything (2026) — `arXiv:2602.02488`, ICML 2026
- **Core**: **completely dynamic RL system** — environment, policy, and reward model all forge themselves through closed-loop optimization, amplifying learning signal.
- **Method**: policy trained with step-wise + outcome signals; RM jointly optimized via consistency feedback; theory-motivated **automatic environment adaptation** uses critic feedback to adapt env.
- **Results**: +9.1% on OSWorld (Qwen3-VL-8B-Thinking); +18.7% on AlfWorld and +11.9% on LiveBench (Qwen2.5-7B-Instruct).
- **Connection**: systems-level vision of [[are]] + reward + policy as one co-optimization; addresses (1)+(3) of the core failure modes simultaneously.

### Expanding the Capabilities of RL via Text Feedback (RLTF) (2026) — `arXiv:2602.02482`, ICLR 2026 MALGAI Oral
- **Core**: formalizes **RL from Text Feedback** — text feedback richer than scalar but cheaper than full demos; available at training, not inference.
- **Two methods**: **RLTF-SD** (self-distill — policy matches feedback-conditioned second-turn generations); **RLTF-FM** (predict critiques as auxiliary objective).
- **Eval**: Reasoning Gym, MATH500, AIME24, LitBench, WritingBench.
- **Connection**: dovetails with [[opcd]] / [[oel]] (feedback-conditioned distillation) and [[srl]] (predict-the-feedback as aux task).

### Learning to Learn from Language Feedback with Social Meta-Learning (2026) — `arXiv:2602.16488`, Cook, Klissarov et al. (DeepMind)
- **Core**: LLMs don't proactively *solicit* feedback even when ambiguous. Apply **Social Meta-Learning (SML)** — learning *how* to learn from others — as a finetuning methodology.
- **Method**: simulated pedagogical dialogues turn static tasks into interactive social learning problems.
- **Key result**: capability **generalizes across domains** — SML on math → better feedback use on coding (and vice versa).
- **Connection**: meta-level analog to [[rltf]]; teaches the *skill* of learning from feedback rather than feedback-conditioned generation.

### Experiential Reinforcement Learning (ERL) (2026) — `arXiv:2602.13949`, Shi et al.
- **Core**: embed an **experience → reflection → consolidation** loop in RL. First attempt → environment feedback → reflection → refined second attempt; reinforce the second-attempt success and internalize into base policy.
- **Why**: env feedback is sparse and delayed; LMs must implicitly infer how failures translate to behavioral changes.
- **Results**: **+81% in complex multi-step environments**, **+11% on tool-using reasoning** vs strong RL baselines.
- **Connection**: explicit operationalization of the [[societies-of-thought]] emergence — reflection step *is* the second internal agent.

**Group synthesis (multi-agent / feedback / experiential)**:
- **Convergence**: text/language is the new reward channel. Whether as **judge** (Section 5), **environment objective** (End of Reward Eng), **training signal** (RLTF), or **internal dialogue** (Societies of Thought, ERL).
- **Meta-learning thread**: SML > RLTF in that it teaches the *skill* of learning from feedback, not the conditioned response. Stronger generalization story.
- **Systems story**: RLAnything pushes the whole stack (env, RM, policy) to co-evolve; ARE supplies the dynamic substrate. The two paired suggest a "self-evolving RL infrastructure" platform.

---

## 7. Cross-Cutting Synthesis

> **Themes that recur across all six clusters:**

### A. The "missing-quadrant" map of post-training
Every method positions itself in a 2D map of (a) **where rollouts come from** (off-policy/teacher vs on-policy/student) × (b) **how dense the supervision is** (per-token vs per-step vs per-trajectory vs per-set).

| | Per-token dense | Per-step dense | Per-traj sparse | Per-set diverse |
|---|---|---|---|---|
| Off-policy (teacher data) | SFT | SRL, InT | RL-on-rejection | — |
| Mixed (DAgger / prefix) | π-Distill, Logit Fusion | TCOD curriculum | POPE, PrefixRL | — |
| On-policy (student) | **OPD (TM)**, OPSD, OPCD | SAS, ΔBelief-RL, ERL | RLVR (GRPO) | Polychromic, G-Zero |

The OPD blog put the bold cell on the map; almost everything in 2026 is variations on filling cells.

### B. The "privilege spectrum"
A unifying frame for the hard-problems cluster + OPD cluster:

`oracle solution` (POPE) → `solution prefix` (PrefixRL) → `expert action per step` (SRL) → `intervention proposal` (InT) → `full-context teacher` (OPCD) → `larger teacher` (TM OPD) → `same model w/ extra context` (OPSD) → `same model w/ rich feedback` (Reflection-Enhanced SD) → `same model w/ self-reflection only` (ERL) → `no privilege` (vanilla RL)

Moving right ↓ privilege, ↑ scalability. Most 2026 wins are mid-spectrum.

### C. Three distinct dense-reward strategies
1. **Token-KL** (OPD line) — teacher tells you the right next-token distribution.
2. **Composite scalar** (HERO, Learning-to-Reason-Factuality) — verifier + RM, multi-axis.
3. **Intrinsic predictive shift** (Hint-δ / ΔBelief) — model's own belief change is the reward.

(1) requires a teacher, (2) requires a verifier + RM, (3) requires nothing. The intrinsic options (3) are the most attractive but hardest to make non-degenerate.

### D. Diversity vs collapse is the universal failure mode
- **Self-play**: Ivison's diversity-bottleneck argument, G-Zero's Hint-δ defense, R-Diverse mitigations.
- **Rewards**: Polychromic's set-level RL, Reward-Hacking-Rubric's documented failure modes.
- **OPD**: Unmasking's finding that OPD helps incorrect-rollouts more — implies diversity in *what to fix* matters.
- **Solutions converge on**: structural defenses (set RL, stratification, intrinsic predictive shift) > LM-judge proxies (which are hackable).

### E. The "right unit" question
A current debate: what's the right *unit* of reasoning to assign reward to?
- **Token**: TM OPD, OPCD — but Unmasking shows uneven gain.
- **Step**: SAS, InT, ΔBelief, FSF — empirically the most informative.
- **Abstraction**: RLAD (explicit NL), RA3 (latent action subspace).
- **Trajectory**: vanilla RLVR — too sparse.
- **Set**: Polychromic, G-Zero — diversity-aware.

Convergent answer (across math, agents, factuality): **step or abstraction**, not token or trajectory.

### F. Practical recipes that recur
- **OPD on a smaller / mid-trained model > pure RL** when teacher is available (TM, MOPD, OPSD).
- **Prefix-condition hard problems → train on completion → behavior transfers unguided** (POPE, PrefixRL, π-Distill).
- **Hybrid verifier + RM with stratified normalization** when reward is sparse but partial signal exists (HERO).
- **Step-level advantage > token-level** when reasoning structure matters (SAS, SRL, InT).
- **Adversarial / dynamic critic > static rubric** to avoid reward hacking (RLAC vs rubric-RL studies).
- **Intrinsic predictive shift > LM judges** to avoid hackable proxies (Hint-δ, ΔBelief, CaT).

### G. What's still open
- **Automatic discovery of good prefixes / abstractions / privileged contexts** — most methods assume humans (or a stronger oracle) supply them.
- **Diversity-collapse-proof self-play** — every existing defense (LM judge, embedding cosine, corpus conditioning) is hackable or collapses past 600+ steps (Ivison).
- **Selective / instance-conditional OPD** — Unmasking shows OPD's signal isn't uniformly useful; how to gate it.
- **Multi-agent emergent dialogue as an RL target** — Societies-of-Thought hints at this, but no clean recipe yet.
- **Trillion-scale RL stability** — Ring-1T's IcePop/C3PO++/ASystem are first attempts; this is a wide-open systems frontier.
