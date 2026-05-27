# RL Overview: Reasoning Models, Test-Time Scaling, Verifiers (2024–2025)

A landscape report on ~75 papers/blogs spanning the **o1/R1 era of large reasoning models**, the **algorithms** that produced them (GRPO, PRIME, OREAL, GSPO), the **diagnostics** that made sense of them (SFT-Memorizes-RL-Generalizes, Cognitive Behaviors, Sober Look, Understanding R1-Zero), **test-time scaling and efficiency** (s1, Chain-of-Drafts, MRT, L1, NoThinking), **process verifiers / PRMs / critics** (PAVs, PRIME, FG-PRM, CFT, CTRL, DeepSeek-GRM), **reasoning data and structure** (LIMO/LIMR, structure-not-content, CodeI/O), and **hierarchical / search** approaches (HRL, RLAD, rStar-Math, Meta-CoT). Cross-cutting synthesis at the end.

> **Frame**: this is the post-training corpus from the year the field discovered that **pure outcome RL on a strong base + simple reward** could elicit aha-moments at scale. Everything else is variations: better algorithms, better verifiers, less data, less compute, more transparency about what's actually working.

---

## 1. The o1/R1 Era — Large Reasoning Model Releases

> **Theme**: open-source replicas of o1-style RL-trained reasoning models. The thesis crystallized: outcome-only RL on a verifiable-reward task can self-evolve reasoning behaviors (verification, backtracking, reflection) without SFT scaffolding or MCTS/PRM.

### DeepSeek-R1 (2025) — `arXiv:2501.12948`
- **Core**: pure RL from base → emergent reasoning. **R1-Zero** uses GRPO on DeepSeek-V3-Base with binary correctness reward; *no SFT*, no PRM, no MCTS.
- **Result**: AIME'24 pass@1 **15.6 → 71.0%** (86.7% w/ majority voting), matches o1-0912.
- **Emergent behaviors**: self-reflection, verification, dynamic strategy adaptation — "aha moment."
- **R1**: full pipeline = cold-start SFT + reasoning RL + rejection-sample SFT + alignment RL. Distilled into 1.5B–70B Qwen/Llama checkpoints.
- **Connection**: seminal — every other entry in this section is a response to this paper.

### Kimi k1.5 (2025) — `arXiv:2501.12599`, Moonshot
- **Core**: parallel R1-era release. Multi-modal RL with long-context scaling and improved policy optimization. Treats scaling RL as a new axis (separate from data scaling).
- **Results**: AIME 77.5, MATH500 96.2, Codeforces 94th %ile, MathVista 74.9 — matches o1.
- **long2short**: distills long-CoT into short-CoT models; +550% vs GPT-4o / Sonnet on AIME/MATH at short CoT.
- **Connection**: contemporary with [[deepseek-r1]]; less "minimalist" — keeps SFT + multi-modal scaffolding.

### Seed-Thinking-v1.5 (2025) — ByteDance Seed
- **Core**: MoE reasoning model (20B active / 200B total). RL fine-tuning + dual-track reward (verifiable + non-verifiable).
- **Results**: AIME 86.7, Codeforces 55.0, GPQA 77.3. Beats DeepSeek-R1 by +8% win rate on non-reasoning.
- **Notable**: 400K SFT instances (300K verifiable + 100K non-verifiable) for long-CoT bootstrapping.
- **Connection**: industrial-scale variation of [[deepseek-r1]] — MoE + non-verifiable RL is the next axis after pure RLVR.

### o1-IOI / Competitive Programming with Large Reasoning Models (2025) — `arXiv:2502.06807`, OpenAI
- **Core**: compares o1, o3 (early), and **o1-ioi** (domain-engineered for IOI 2024).
- **Result**: o1-ioi at IOI 2024 → 49th %ile live, gold under relaxed constraints. **o3 hits gold without hand-crafted strategies.**
- **Takeaway**: scaled-up general RL beats domain-specific inference pipelines. Validates the R1-style minimalist thesis.
- **Connection**: empirical evidence that the [[deepseek-r1]] paradigm subsumes hand-engineering.

### O1 Replication Journey — GAIR-NLP (2024)
- **Core**: pre-R1 attempt to replicate o1 via **journey learning** — train on full exploration process (trial, error, reflection, backtracking) not just shortcuts.
- **Result**: 327 training samples → +8% on MATH over standard supervised learning.
- **Notable**: predates [[deepseek-r1]]'s pure-RL approach; reaches similar insight (model needs to learn the *process*) via SFT on annotated journeys.
- **Connection**: SFT-side ancestor of [[r1-zero]]; same insight via different mechanism.

### OREAL — Exploring the Limit of Outcome Reward (2025) — `arXiv:2502.06781`, InternLM
- **Core**: behavior cloning on **positive trajectories from best-of-N sampling** is provably sufficient to learn KL-regularized optimal policy under binary feedback.
- **Method**: token-level reward model samples important tokens within long CoT to densify the sparse outcome reward.
- **Result**: 7B model → **94.0% pass@1 on MATH-500**, matching 32B models. OREAL-32B → 95.0%, beats distilled 32Bs.
- **Connection**: BoN + token-level credit assignment as an alternative to PPO-style RL on outcome reward.

### Sky-T1-32B-Preview (2025) — NovaSky AI (UC Berkeley Sky Computing)
- **Core**: first truly open-source o1-style reasoning model. Trained for **<$450** in 19 hours on 8×H100.
- **Result**: matches o1-preview on MATH500 and LiveCodeBench (preview).
- **Paper companion**: "LLMs Can Easily Learn to Reason from Demonstrations — Structure, not content, is what matters!" (`arXiv:2502.07374`) — Long-CoT *structure* is what's learned; perturbing content (incorrect samples, removed keywords) barely hurts, but shuffling/deleting steps tanks accuracy. **17K samples → 56.7% AIME, 57.0% LCB** on Qwen2.5-32B.
- **Sky-T1-7B**: interleaved SFT + RL beats R1-distill at same size.
- **Connection**: democratization of [[deepseek-r1]]; the structure-finding is also key to [[limo]] / [[limr]].

### Open-Reasoner-Zero (2025) — `arXiv:2503.24290`
- **Core**: first open-source large-scale R1-Zero-style RL on a base model. **Minimalist**: vanilla PPO with GAE(λ=1, γ=1), rule-based rewards, **no KL regularization**.
- **Result**: on Qwen2.5-32B-Base, beats DeepSeek-R1-Zero-Qwen-32B across AIME2024 / MATH500 / GPQA with **1/10 the training steps**.
- **Connection**: empirical validation that the [[deepseek-r1]] recipe is robust to algorithm simplification (no KL!).

### BOLT — Bootstrap Long Chain-of-Thought without Distillation (2025) — `arXiv:2502.03860`, Salesforce
- **Core**: build Long-CoT in Llama-3.1-70B / Mistral-7B **from scratch — no o1 / R1 / QwQ distillation**.
- **Three stages**: (1) LongCoT bootstrapping (in-context), (2) LongCoT SFT, (3) online training.
- **Connection**: SFT-and-online alternative to [[r1-zero]]; sidesteps reliance on stronger oracles.

### s1: Simple Test-Time Scaling (2025) — `arXiv:2501.19393`
- **Core**: just **1K SFT samples** (curated for difficulty / diversity / quality) on Qwen2.5-32B-Instruct, plus **budget forcing** (terminate or append "Wait" to control thinking length).
- **Results**: beats o1-preview on competition math by 27%; budget forcing extrapolates **50% → 57% on AIME24**.
- **Connection**: minimal-data complement to [[deepseek-r1]]'s minimal-algorithm; suggests the base model already has the latent ability.

### LIMO — Less is More for Reasoning (2025) — `arXiv:2502.03387`, COLM 2025
- **Core**: 817 curated SFT examples → AIME24 **6.5 → 63.3%**, MATH500 59.2 → **95.6%**. **+45.8 absolute pts** on OOD benchmarks vs 100× more data.
- **LIMO Hypothesis**: in foundation models where domain knowledge is comprehensively pre-trained, complex reasoning emerges from minimal but precise demonstrations.
- **Connection**: data-side echo of [[s1]] and [[structure-not-content]]; pre-training already encodes the capability.

### LIMR — Less is More for RL Scaling (2025) — `arXiv:2502.11886`
- **Core**: **Learning Impact Measurement (LIM)** auto-prioritizes RL samples by alignment with learning trajectory. **1,389 of 8,523 samples** suffice.
- **Results**: AIME24 +16.7% vs full dataset. Beats LIMO and s1 by +13.0 / +22.2 pts on MATH500.
- **Connection**: RL analogue of [[limo]] / [[s1]]; sample selection > sample scale.

**Group synthesis (o1/R1 era)**:
- **Convergent recipe**: minimal SFT cold-start + outcome-RL on verifiable math + ~30K–800K problems → emergent verification/reflection/backtracking. [[deepseek-r1]] is the existence proof; [[orz]], [[bolt]], [[oreal]] are independent replications with algorithmic minimization.
- **Data minimization thesis**: [[s1]] (1K), [[limo]] (817), [[limr]] (1,389) all show the latent capability is in the base — you just need the right small dataset. [[structure-not-content]] formalizes why.
- **Distillation track**: Sky-T1, R1-distill series — long-CoT structure transfers cheaply to smaller models.

---

## 2. Diagnostics — Understanding What's Actually Happening

> **Theme**: papers that asked "wait, is this real?" The 2025 reasoning gold rush triggered a wave of careful empirical work on what RL actually changes, when it doesn't, and what behaviors matter.

### SFT Memorizes, RL Generalizes (2025) — `arXiv:2501.17161`, ICML 2025
- **Core**: comparative study of SFT vs RL on **out-of-distribution** rule and visual variants.
- **Method**: GeneralPoints (arithmetic card game) + V-IRL (real-world navigation). Test on unseen rule/visual variants.
- **Finding**: RL (outcome-reward) **generalizes** across rule + visual variants. SFT **memorizes** training data. RL also improves the underlying visual recognition capability.
- **Connection**: foundational thesis for the era — explains why [[deepseek-r1]]'s pure RL gives genuinely new capability rather than narrow imitation.

### Cognitive Behaviors that Enable Self-Improving Reasoners (2025) — `arXiv:2503.01307`, COLM 2025, Gandhi et al.
- **Core**: identifies **four cognitive habits** — verification, backtracking, subgoal setting, backward chaining — that predict whether RL self-improvement works on a given base model.
- **Phenomenon**: Qwen-2.5-3B improves dramatically on Countdown under RL; Llama-3.2-3B doesn't. Reason: Qwen *natively exhibits* those four behaviors; Llama doesn't.
- **Key finding**: **presence of reasoning behaviors > correctness of answers**. Priming Llama with incorrect-but-properly-structured solutions matches priming with correct solutions.
- **Connection**: explains the [[understanding-r1-zero]] observation that DeepSeek-V3-Base already exhibits "aha moments"; explains why "structure not content" works.

### A Sober Look at Progress in Language Model Reasoning (2025) — `arXiv:2504.07086`, Hochlehnert et al.
- **Core**: reproducibility audit. Reassesses recent RL methods under a standardized evaluation framework.
- **Finding**: **most RL approaches yield only modest improvements — far below prior claims** — and overfit small benchmarks (AIME'24). SFT methods generalize more consistently.
- **Contribution**: open eval framework + best-practice reporting standards + released code/prompts/outputs.
- **Connection**: necessary corrective on the gold-rush papers; bridges to [[small-models-struggle]] which finds similar issues at small scale.

### Understanding R1-Zero-Like Training: A Critical Perspective (2025) — `arXiv:2503.20783`
- **Core**: dissects R1-Zero into base model + RL components.
- **Findings**: DeepSeek-V3-Base already exhibits "aha moments." Qwen2.5 base models reason strongly even *without* prompt templates → potential **pretraining biases**.
- **Algorithmic contribution**: **Dr. GRPO** — identifies GRPO's optimization bias that artificially inflates response length on incorrect outputs; fixes it for better token efficiency.
- **Result**: minimalist recipe → 43.3% AIME 2024 with a 7B base.
- **Connection**: empirical foundation for why [[oreal]] / [[orz]] minimalism works.

### Empirical Study on Eliciting and Improving R1-like Reasoning (2025) — `arXiv:2503.04548`, STILL project
- **Core**: third STILL technical report. Systematic ablation of RL training factors on base and SFT'd models.
- **Findings**: RL consistently improves Qwen2.5-32B base (length + accuracy). Even DeepSeek-R1-Distill-Qwen-1.5B improves further to **39.33% AIME 2024** via additional RL. Tool use significantly boosts large reasoning models.
- **Connection**: complementary empirical map alongside [[understanding-r1-zero]].

### Rethinking Reflection in Pre-Training (2025) — `arXiv:2504.04022`
- **Core**: reflection ability **emerges during pretraining**, not just RL. Most prior work attributed it to post-training.
- **Method**: inject deliberate errors into CoTs; test if model self-corrects. Simple triggers like "Wait," elicit reflection in partially pre-trained checkpoints.
- **Result**: OLMo2-7B trained on 4T tokens self-corrects across 6 self-reflection tasks. Reflection capacity scales with pretraining compute.
- **Connection**: explains the [[cognitive-behaviors]] finding — Qwen has these traits because pretraining instilled them.

### DeepSeek-R1 Thoughtology (2025) — `arXiv:2504.07128`, McGill NLP
- **Core**: in-depth study of R1's reasoning behavior. Coins **Thoughtology** as a research area.
- **Findings**: R1 has a **sweet spot of reasoning** — extra inference time *hurts*. Reasons longer in English than Chinese on moral/cultural prompts; gives different answers per language.
- **Connection**: empirical case study for [[underthinking]] / [[chain-of-drafts]] / [[l1]] — there's an optimum that fixed-length sampling misses.

### Climbing the Ladder of Reasoning (2025) — `arXiv:2504.11741`, Yiyou Sun et al.
- **Core**: tier-by-tier analysis of what SFT can teach. Easy → Medium requires only ~500–1K R1-style instances. Hard plateaus at ~65%. **Exh**-level requires unconventional problem-solving that no current model can do.
- **Notable counter to LIMO**: "carefully curated small-scale datasets offer limited advantage — scaling proves far more effective" — direct tension with [[limo]] claim.
- **Connection**: empirical map of SFT's ceiling; complements [[sober-look]] in tempering expectations.

### General Reasoning Requires Learning to Reason from the Get-go (2025) — `arXiv:2502.19402`
- **Core**: LLMs overfit reasoning to training distribution; struggle on algorithmic tasks in esoteric programming languages.
- **Thesis**: the issue is **coupling of reasoning and knowledge** in LLMs. Proposes disentangling — pretrain with RL from scratch.
- **Connection**: maximalist position vs the minimalist [[deepseek-r1]] view. Argues post-training fixes are insufficient.

### Small Models Struggle to Learn from Strong Reasoners (2025) — `arXiv:2502.12143`
- **Core**: small models (≤3B) **don't benefit** from long CoT or distillation from larger reasoners. Better with shorter, simpler reasoning matching their capacity.
- **Fix**: **Mix Distillation** (short + long traces) outperforms either alone.
- **Connection**: scaling caveat for [[sky-t1]] / R1-distill series; explains why [[chain-of-drafts]] and [[no-thinking]] also work.

### Critical Tokens Matter (2024) — `arXiv:2411.19943`
- **Core**: a few **critical tokens** in CoT determine whether a reasoning trace fails. Identifies them via **contrastive estimation** between positive and negative models.
- **Method**: **cDPO** (contrastive DPO) trains the model away from critical tokens.
- **Result**: improvements on GSM8K / MATH500 with Llama-3 (8B/70B) and DeepSeek-Math.
- **Connection**: foundational to InT / SAS / SRL in the [[rl_overview]] step-level family; finds that not all tokens are equal.

### Spurious Rewards Paradox (2026) — Note: dated 2026 but referenced from 2025
- **Core**: Qwen2.5-Math gains MATH-500 / AIME from **random / format-only / incorrect rewards**. Mechanistic interpretability finds an **Anchor-Adapter circuit** in middle layers triggering memorized solutions.
- **Connection**: stern warning for [[r1-zero]]-style work — measured "RL gains" may be activating memorized solutions, not learning new reasoning. Pair with [[sober-look]].

### Are Reasoning Models more Faithful? (2025)
- **Core**: mixed findings. RL-trained reasoning models *may* be more faithful (verifiable reward incentivizes truthful CoT), but reasoning models still verbalize used hints <20% of the time in many settings; faithfulness drops on harder tasks.
- **Companion**: "Reasoning Models Don't Always Say What They Think" — explicit content-based cues OK; implicit cues have low articulation rates.
- **Connection**: caveat to the interpretability story for [[deepseek-r1]] / [[thoughtology]].

**Group synthesis (diagnostics)**:
- **Strong claim**: RL generalizes, SFT memorizes ([[sft-memorize-rl-generalize]]) — but only when the base has the cognitive primitives ([[cognitive-behaviors]]) and the gains aren't activating memorized solutions ([[spurious-rewards]]).
- **Pretraining decides ceiling**: Reflection / verification / backtracking emerge during pretraining ([[rethinking-reflection]]); RL surfaces them but can't create them in models that lack them ([[cognitive-behaviors]]).
- **Reproducibility caveat**: [[sober-look]] tempers headline numbers; many RL gains don't reproduce under careful eval.
- **Open**: how to get the right cognitive primitives into the base model in the first place ([[general-reasoning-getgo]]).

---

## 3. RL Algorithms for Reasoning

> **Theme**: algorithmic improvements over GRPO/PPO. Two threads: stability/scale (GSPO, Dr. GRPO) and dense signal (PRIME, OREAL, VinePPO, EMPG, MRT, L1).

### GSPO — Group Sequence Policy Optimization (2025) — `arXiv:2507.18071`, Qwen
- **Core**: **sequence-level** importance ratio + clipping + reward, not token-level. GRPO's instability comes from misapplied token-level IS weights → high-variance noise.
- **Result**: stabilizes **MoE RL training** (where GRPO collapses); enabled Qwen3.
- **Connection**: same diagnosis as [[dr-grpo]] (GRPO has a structural bias); different fix. The two together define the post-GRPO state of the art.

### PRIME — Process Reinforcement through Implicit Rewards (2025) — `arXiv:2502.01456`
- **Core**: online PRM updates using **only policy rollouts and outcome labels** via **implicit process reward modeling** (Yuan et al. 2024b).
- **Result**: Qwen2.5-Math-7B-Base → **+15.1% avg** across reasoning benchmarks (Eurus-2-7B-PRIME beats Qwen2.5-Math-7B-Instruct on 7 benchmarks with **10% the training data**).
- **Connection**: solves the PRM data bottleneck — no separate step-level labels needed. Bridges outcome reward → dense process reward.

### VinePPO (2024) — `arXiv:2410.01679`, Kazemnejad et al.
- **Core**: **Monte Carlo credit assignment** via vine sampling, replacing PPO's value network (which fails on long reasoning).
- **Result**: outperforms PPO and RL-free baselines on MATH / GSM8K with **9× fewer gradient updates** and **3× wall-clock**.
- **Connection**: pre-R1 step-level method; influenced [[mrt]] and [[learnability]]'s use of sampling structures.

### Process Advantage Verifiers / Rewarding Progress (2024) — `arXiv:2410.08146`, Setlur et al.
- **Core**: dense step-level **PAV** = change in likelihood of reaching success under a complementary **prover policy**. Train PAVs for evaluation.
- **Result**: beam search + PAVs → **+8–10% accuracy, 1.5–5× compute** vs ORMs on Gemma 2B/9B/27B. PAV-RL → 5–6× sample efficiency, +6% accuracy.
- **Connection**: principled construction of process rewards from outcome rewards; ancestor of [[prime]].

### Logic-RL (2025) — `arXiv:2502.14768`
- **Core**: rule-based RL on synthetic logic puzzles (Knights & Knaves). System prompt emphasizing thinking, stringent format reward, simple recipe.
- **Result**: 7B → reflection / verification / summarization behaviors after **only 5K logic problems**. Generalizes to AIME and AMC math.
- **Connection**: minimal-domain instance of [[deepseek-r1]] paradigm; demonstrates rule-based reward + thinking format is sufficient.

### Reward-aware Preference Optimization (RPO) (2025) — `arXiv:2502.00203`, NVIDIA
- **Core**: unified math framework subsuming DPO, IPO, SimPO, REINFORCE-LOO. Lets you systematically vary objective / responses-per-prompt / implicit-vs-explicit RM.
- **Used in**: NVIDIA's Llama3.1-Nemotron-70B-Instruct alignment + DeepSeek-R1 recipe.
- **Connection**: meta-framework for the alignment-RL design space.

### CGPO + Mixture of Judges (2024) — `arXiv:2409.20370`, Meta
- **Core**: **Constrained Generative Policy Optimization** + **Mixture of Judges (MoJ)** — rule-based + LLM-based judges enforce constraints during generation.
- **Why**: addresses reward hacking and extreme multi-objective optimization in multi-task RLHF.
- **Connection**: rubric-RL / constrained RL ancestor of the rubric-rewards line (Rubric Anchors, RaR, etc.).

### Reinforcement Learning with Rubric Anchors (2025) — `arXiv:2508.12790`
- **Core**: extends RLVR to **open-ended subjective** tasks via rubric-based rewards. Carefully designed rubrics = structured, model-interpretable scoring criteria.
- **Scale**: **>10K rubrics** — largest rubric reward system to date. Sources: human / LLM / hybrid.
- **Connection**: direct response to RLVR's verifiability ceiling; precursor to the 2026 rubric-RL papers (RubricEM, Reward Hacking in Rubric-Based RL).

### InfAlign — Inference-aware Alignment (2025) — `arXiv:2412.19792`, ICML 2025
- **Core**: standard RLHF optimizes the wrong objective when inference uses BoN / controlled decoding / tree search.
- **Method**: **calibrate-and-transform RL (InfAlign-CTRL)** — reward calibration + KL-regularized maximization of *transformed* calibrated reward.
- **Result**: **+8–12% on helpfulness / harmlessness benchmarks**.
- **Connection**: train-test mismatch fix; orthogonal to [[mrt]] / [[l1]] which target reasoning-length controllability.

### EMPG — Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents (2025) — `arXiv:2509.09265`
- **Core**: gradient magnitude is **coupled with entropy** — confident correct actions get inefficiently small updates; uncertain actions get destabilizing large ones.
- **Method**: re-calibrate gradient by step-wise uncertainty × final outcome. Amplifies updates for confident-correct, *strongly penalizes* confident-incorrect ("hallucinated confidence"), attenuates uncertain steps.
- **Connection**: makes long-horizon credit assignment work where vanilla PG fails; same problem [[mrt]] / [[vineppo]] / [[prime]] attack with different tools.

### MRT — Optimizing Test-Time Compute via Meta-RL Fine-tuning (2025) — `arXiv:2503.07572`, Qu, Setlur, Salakhutdinov, Kumar
- **Core**: views long output as a sequence of episodes; **cumulative regret over output tokens** = test-time-compute efficacy. Bonus = "progress" each block makes (change in success likelihood).
- **Result**: **2–3× perf gain, 1.5× token efficiency** vs outcome-RL. SOTA at 1.5B on AIME24/25, AMC23.
- **Connection**: principled framing of test-time compute; deeply related to [[learnability]] / [[explore-hard]] / [[pope]] line.

### L1 — Length-Controlled Policy Optimization (2025) — `arXiv:2503.04697`, Aggarwal & Welleck
- **Core**: **LCPO** trains a reasoning LM to satisfy a *length constraint given in its prompt*. Outputs Short Reasoning Models (SRMs) with R1-style behaviors at non-reasoning model lengths.
- **Result**: smoothly trades cost vs accuracy; beats [[s1]]'s length control.
- **Connection**: explicit length-controllability — complementary to [[mrt]]'s implicit efficiency and [[orion]]'s SLPO.

### DuPO — Dual Preference Optimization (2025) — `arXiv:2508.14460`
- **Core**: **annotation-free feedback via generalized duality**. Decompose primal task input into known + unknown; construct dual task to reconstruct the unknown from primal output. Quality of reconstruction → self-supervised reward.
- **Generalization**: works on non-invertible tasks (reverses math solutions to recover hidden variables).
- **Result**: Qwen3-4B → +6.5pp on 4 math benchmarks; Seed-X-7B → +2.1 COMET on translation, matching SOTA.
- **Connection**: removes RLVR's labeled-data dependency; thematically related to [[compute-as-teacher]] (reference-free supervision).

### Random Policy Valuation / "Heuristics Considered Harmful" line (2025) — ROVER, `arXiv:2509.24981`
- **Core**: proves optimal action recoverable from Q-function of a **fixed uniformly random policy** → skips generalized policy iteration + heuristics (clipping, KL, data selection).
- **Algorithm**: **ROVER** preserves diversity throughout training.
- **Result**: **+8.2 pass@1, +16.8 pass@256** vs heavy heuristics, despite radical simplification.
- **Connection**: structural critique of the field's heuristic-stacking; same minimalist spirit as [[orz]].

### Perils of Optimizing Learned Reward Functions (2025) — `arXiv:2406.15753`, Lang et al., ICML 2025
- **Core**: low expected test error of RM ≠ low regret for the optimized policy. **Error-regret mismatch** arises from distributional shift during policy optimization.
- **Finding**: persists even with policy regularization (RLHF-style KL penalties).
- **Connection**: theoretical foundation for the [[reward-hacking-rubric]] cluster in [[rl_overview]] — reward learning is more brittle than the loss says.

**Group synthesis (RL algorithms)**:
- **Two failure modes attacked**: instability/scale ([[gspo]], [[dr-grpo]]) and credit-assignment density ([[prime]], [[vineppo]], [[pavs]], [[empg]], [[mrt]]).
- **PRM revolution**: [[pavs]] → [[prime]] → [[prm-72b]] line dramatically lowered the cost of step-level reward, making process supervision practical.
- **Length controllability** is its own subaxis: [[l1]] explicit, [[mrt]] meta-RL implicit, [[orion]] preference-based.
- **Theoretical anchors**: [[perils-of-learned-rewards]] (RM/policy mismatch) and [[random-policy-valuation]] (heuristics are over-fitted) bookend the empirical work.

---

## 4. Process Verifiers, PRMs, and Critics

> **Theme**: how to get reward signal that's denser than the final answer, more reliable than a learned RM, and useful at both training and inference time.

### Qwen2.5-Math-PRM-72B (2025) — Tech report `arXiv:2501.07301`
- **Core**: PRM fine-tuned on Qwen2.5-Math-72B-Instruct. Special tokens inserted after each step, probability score 0–1.
- **Result**: outperforms GPT-4o-0806 (7B variant); both 7B and 72B variants beat existing PRMs on BoN and ProcessBench.
- **Connection**: production-grade PRM; standard reference baseline for the cluster.

### SCRIT — Self-Evolving Critic (2025) — `arXiv:2501.05727`
- **Core**: **contrastive-critic** framework where reference solutions guide data synthesis (not ground-truth critiques). Self-validation for data quality.
- **Result**: Qwen2.5-72B-Instruct critic: 39.7 → 50.0% on deliberately incorrect solutions; 57.7 → 62.1% balanced.
- **Connection**: scalable-oversight angle; uses [[opsd]]-style privileged-context distillation.

### Critique Fine-Tuning (CFT) (2025) — `arXiv:2501.17703`, COLM 2025
- **Core**: train model to **critique noisy responses**, not imitate correct ones. 50K-sample dataset from WebInstruct with GPT-4o critiques.
- **Result**: **+4–10% over SFT** on six math benchmarks across Qwen2.5, Qwen2.5-Math, DeepSeek-Math.
- **Connection**: cousin of [[scrit]]; differs in target (critique itself, not the critic's policy).

### CTRL — Teaching LMs to Critique via RL (2025) — `arXiv:2502.03492`
- **Core**: train a **critic model** via RL to maximize correction perf of a fixed generator — no human supervision.
- **Result**: **+106.1% relative on code gen benchmarks** via iterative critique-revision. Critics act as accurate generative reward models.
- **Connection**: scales [[scrit]] / [[cft]] to RL; converges with [[rlac]] adversarial-critic line in 2026.

### FG-PRM — Fine-Grained PRM (2024) — `arXiv:2410.06304`
- **Core**: **6-type hallucination taxonomy** — fabrication, factual / context / instruction / logical inconsistency, logical error. 6 specialized PRMs, one per type.
- **Method**: automated LLM-driven fine-grained hallucination data synthesis.
- **Result**: beats ChatGPT-3.5 and Claude-3 on fine-grained hallucination detection; boosts LLM perf on GSM8K / MATH.
- **Connection**: granular alternative to monolithic PRMs; sets up the rubric-RL framing.

### Subtle Errors in Reasoning — RISE (2024) — `arXiv:2410.06638`
- **Core**: **eRror-Injected Self-Editing** — LLM edits a few tokens in solutions to inject *predefined subtle errors*. Constructs hard pairs for DPO.
- **Result**: Qwen2-7B-Instruct: **+3.0 GSM8K, +7.9 MATH with only 4.5K samples**. Generalizes to logical reasoning + code.
- **Connection**: same logic as [[critical-tokens]] — locate the rare token that determines correctness, train against it.

### DeepSeek-GRM — Inference-Time Scaling for Generalist Reward Modeling (2025) — `arXiv:2504.02495`
- **Core**: **pointwise generative reward modeling** + **Self-Principled Critique Tuning (SPCT)** — online RL to generate principles adaptively + critique accurately.
- **Inference scaling**: parallel sampling + meta RM-guided voting → reward-model-time scaling beats training-time scaling.
- **Connection**: inference scaling for reward models — symmetric to [[snell-tts]] for generators; foundational for [[rlv]] / [[eval-time-compute]] line.

### Scaling Evaluation-time Compute with Reasoning Models as Process Evaluators (2025) — `arXiv:2503.19877`
- **Core**: prompt reasoning models to evaluate outputs (outcome and process). **Evaluator perf scales monotonically with reasoning tokens.**
- **Killer result**: **32B reasoning evaluator beats 72B SOTA PRM by 4.5%** on ProcessBench.
- **Connection**: aligns with [[deepseek-grm]]; reframes the eval/generate boundary.

### RL^V — Putting the Value Back in RL (2025) — `arXiv:2505.04842`
- **Core**: augments value-free RL (GRPO, LOO-PPO) by **jointly training the LLM as reasoner + generative verifier** using RL-generated data. Negligible overhead.
- **Result**: **+20% MATH accuracy** with parallel sampling; **8–32× efficient test-time compute scaling**; **1.2–1.6× higher perf** when jointly scaling parallel + sequential TTC on R1 models.
- **Connection**: unifies the reasoner/verifier split; addresses GRPO's discard of value functions.

### Compute-Optimal Problem Solving and Generative Verification (2025) — `arXiv:2504.01005`
- **Core**: under fixed inference budget, should you spend on **more solutions (Self-Consistency)** or **fewer solutions + more verification (GenRM)**?
- **Method**: scales verification as next-token prediction → multiple verification CoTs per solution.
- **Connection**: principled answer to [[rlv]]'s inference-time allocation question; pairs with [[snell-tts]].

### Scaling LLM Test-Time Compute Optimally (2024) — `arXiv:2408.03314`, Snell, Lee, Xu, Kumar
- **Core**: pioneering test-time compute scaling paper. **Compute-optimal strategy** — adaptive per-prompt allocation.
- **Result**: **>4× efficiency** vs best-of-N; TTC can outperform a **14× larger model** on prompts with non-trivial success rate.
- **Connection**: defined the test-time scaling research program. Predates and seeds [[s1]] / [[mrt]] / [[l1]] / [[rlv]].

### Scaling Test-Time Compute Without Verification or RL is Suboptimal (2025) — `arXiv:2502.12118`, ICML 2025 spotlight
- **Core**: **proves** that VB (verifier-based RL/search) is far superior to VF (cloning search traces) given fixed compute when the base LM has heterogeneous reward distribution.
- **Connection**: theoretical underpinning for the entire verifier line ([[rlv]], [[deepseek-grm]], [[compute-optimal-verify]]); explicit rebuke of pure-distillation approaches.

### AggLM — The Majority is not always right (2025) — `arXiv:2509.06870`, Weston et al.
- **Core**: learn **aggregation as an explicit reasoning skill** — RL-trained aggregator reviews, reconciles, synthesizes a final answer from multiple solutions.
- **Key**: careful easy/hard balance lets the model recover **minority-correct** answers (not just majority-vote winners).
- **Result**: beats majority voting and reward-model ranking baselines.
- **Connection**: trained-aggregation alternative to [[snell-tts]] / [[compute-optimal-verify]]'s static selection methods.

### SWEET-RL (2025) — `arXiv:2503.15478`, Meta FAIR + UC Berkeley
- **Core**: multi-turn collaborative reasoning RL. **Step-Wise Evaluation from Training-time information** — critic with access to extra info provides step-level rewards.
- **Benchmark**: **ColBench** (6 collaborative reasoning tasks).
- **Result**: **+6% absolute success/win rate** vs SOTA multi-turn RL. Llama-3.1-8B matches/exceeds GPT-4o on collaborative content creation.
- **Connection**: multi-turn extension of step-level credit assignment ([[pavs]], [[prime]], [[sas]]).

**Group synthesis (verifiers)**:
- **Three convergent ideas**: (1) **make the verifier a reasoner** ([[eval-time-compute]], [[deepseek-grm]], [[rlv]]) — scales naturally with inference compute; (2) **dense intermediate reward from outcome label alone** ([[prime]], [[pavs]]) — eliminates step-level annotation; (3) **structured taxonomy** ([[fg-prm]]) and **trained aggregation** ([[agglm]]).
- **VB > VF**: [[ttc-without-verifier]] proves verifier-based is fundamentally superior to distillation-only. This grounds the entire cluster.
- **Critic training trio**: [[scrit]], [[cft]], [[ctrl]] together establish self-evolving critique as a scalable supervision substrate.

---

## 5. Test-Time Scaling & Reasoning Efficiency

> **Theme**: long CoT is wasteful. Family of methods to shorten reasoning while preserving accuracy: token budgets, drafts, surprisal-pruning, early-exit, distillation, controllable length.

### Thoughts Are All Over the Place — Underthinking (2025) — `arXiv:2501.18585`
- **Core**: o1-like LLMs **frequently switch between reasoning thoughts** without sufficiently exploring promising paths → "underthinking."
- **Metric**: token efficiency in incorrect answers quantifies underthinking.
- **Fix**: **Thought-Switching Penalty (TIP)** — discourages premature transitions during decoding.
- **Connection**: complements [[no-thinking]] (which finds *less* thinking helps); together suggest the optimum is *focused* not *long* thinking.

### Reasoning Models Know When They're Right (2025) — `arXiv:2504.05419`
- **Core**: probe hidden states → can verify intermediate answers with high accuracy + calibrated scores. Hidden states encode **future answer correctness**, enabling early prediction.
- **Application**: probe-as-verifier triggers early exit → **−24% inference tokens with no perf drop**.
- **Connection**: hidden-state-based version of [[certaindex]] / [[eval-time-compute]]; suggests the model *already knows* when to stop.

### Certaindex (2025) — `arXiv:2412.20993`
- **Core**: algorithm-agnostic metric for **answer stabilization** during reasoning. Dynamic compute allocation via **Dynasor** system.
- **Result**: **−50% compute, 3.3× higher throughput** in real workloads, no accuracy drop.
- **Connection**: serving-time productionization of the same idea as [[hidden-state-probe]].

### Chain of Draft (2025) — `arXiv:2502.18600`
- **Core**: human-inspired alternative to CoT. Limit each reasoning step to **≤5 words** — minimalist drafts.
- **Result**: matches/surpasses CoT accuracy with **as little as 7.6% of tokens**.
- **Connection**: ultra-concise CoT cousin of [[s1]]'s budget forcing and [[l1]]'s length control.

### ASAP — Pruning the Unsurprising (2025) — `arXiv:2508.05988`
- **Core**: **first-token surprisal** > perplexity for measuring logical importance of CoT steps. **Anchor-guided pruning** preserves core structure.
- **Result**: LiveCodeBench v4_v5: **−23.5% tokens, −43.5% latency** at competitive 36.19% pass@1.
- **Connection**: structural compression of CoT; complements [[chain-of-drafts]]'s natural-language compression.

### Training LMs to Reason Efficiently (2025) — `arXiv:2502.04463`
- **Core**: **modest academic resources** suffice. ~100 RL steps + ~200 gradient updates to train a length-aware reasoner with per-token penalties.
- **Connection**: scaling-down analogue to [[l1]] / [[mrt]]; cheap-and-cheerful efficient reasoning.

### ORION — Reason Efficiently in the Language of Thought (2025) — `arXiv:2511.22891`
- **Core**: inspired by **Mentalese** (Language of Thought Hypothesis) — train model to reason in ultra-compressed structured tokens.
- **Method**: **Shorter Length Preference Optimization (SLPO)** rewards concise correct, allows longer when needed.
- **Result**: **4–16× fewer tokens**, **5× lower latency**, **7–9× lower training cost** vs R1-distill, retains 90–98% accuracy.
- **Connection**: explicit cognitive-science grounding for what [[chain-of-drafts]] / [[asap]] / [[l1]] all converge on.

### NoThinking — Reasoning Models Can Be Effective Without Thinking (2025) — `arXiv:2504.09858`
- **Core**: **bypass the explicit Thinking process** via prompting. Surprisingly effective.
- **Result**: NoThinking > Thinking on 7 benchmarks at fixed token budget; e.g., ACM 23 at 700 tokens: **51.3 vs 28.9**. Especially strong in low-budget regimes.
- **Method**: parallel scaling (N outputs, aggregate) of NoThinking is highly effective.
- **Connection**: radical inverse of the [[deepseek-r1]] paradigm at inference; for short-budget settings, the Thinking scaffold *hurts*.

### Self-Training Elicits Concise Reasoning (2025) — `arXiv:2502.20122`, ACL 2025 Findings
- **Core**: LLMs have **latent ability to reason more concisely**. Elicit it via best-of-N + few-shot conditioning to self-generate concise traces, then SFT.
- **Result**: **−30% output tokens on average** across 5 model families on GSM8K / MATH, accuracy preserved.
- **Connection**: data-side complement to [[l1]] / [[orion]]; no RL needed.

### How Well Do LLMs Compress Their Own CoT (2025) — `arXiv:2503.01141`
- **Core**: first systematic study of length vs accuracy tradeoff under compression prompts ("≤10 words," "no punctuation").
- **Finding**: **universal tradeoff** — each task has intrinsic **token complexity** (minimum needed). Claude / GPT-4 self-compress better than smaller models; compression ability correlates with reasoning perf.
- **Connection**: empirical foundation for the cluster — sets the limits on what compression methods can do.

### UPFT — Unsupervised Prefix Fine-Tuning (2025) — `arXiv:2503.02875`, NeurIPS 2025
- **Core**: **Prefix Self-Consistency** — initial 8 tokens are shared across diverse solution trajectories. Train only on these initial prefixes.
- **Result**: matches Rejection-Sampling Fine-Tuning perf; **−75% training time, −99% sampling cost**.
- **Connection**: the "first few tokens are all you need" theme — same insight as [[critical-tokens]] but for early tokens, not error tokens.

### Thinking Slow, Fast — Distilled Reasoners (2025) — `arXiv:2502.20339`
- **Core**: distill **pure and hybrid Mamba** models from pretrained Transformers (only 8B tokens). Mamba is much faster at long sequences.
- **Result**: under fixed time budgets, distilled Mambas **scale coverage and accuracy past their Transformer teachers**.
- **Connection**: architecture-side answer to efficiency — orthogonal to algorithmic methods.

### Adaptive Reasoning with Inference-aware Optimization (2025) — `arXiv:2501.17974`
- **Core**: **Inference Budget-Constrained Policy Optimization (IBPO)** — constrained-RL formulation that adaptively switches between short and long reasoning per input complexity.
- **Connection**: cousin of [[l1]] / [[mrt]]; explicit constrained-optimization framing.

**Group synthesis (test-time scaling)**:
- **Two threads**: *measure when to stop* ([[certaindex]], [[hidden-state-probe]]) and *train to stop earlier* ([[l1]], [[mrt]], [[orion]], [[chain-of-drafts]], [[asap]], [[self-train-concise]]).
- **Counterintuitive findings**: [[no-thinking]] (no CoT beats CoT under tight budgets), [[underthinking]] (more switching = worse), [[thoughtology]] (longer reasoning hurts past sweet spot) — all point to **focus > length** as the right axis.
- **Architecture axis**: [[thinking-slow-fast]] is the only paper here that changes the substrate (Mamba) rather than the policy.

---

## 6. Hierarchical / Search / Action-Abstraction Reasoning

> **Theme**: reasoning isn't a flat token sequence — it has hierarchy, latent abstractions, and search structure. Family of methods that surface or train this structure.

### Discovering Temporal Structure — HRL Overview (2025) — `arXiv:2506.14045`, Klissarov et al.
- **Core**: survey of **hierarchical reinforcement learning** — discovering and exploiting temporal structure in agent experience.
- **Coverage**: methods from online experience to offline datasets; benefits and trade-offs for different problem types.
- **Connection**: framing reference for the cluster; precedes [[emergent-hierarchical]] / [[rlad]] / [[ra3]].

### Towards Hierarchical Multi-Step Reward Models (HRM) (2025) — `arXiv:2503.13551`
- **Core**: **Hierarchical Reward Model** evaluates both individual and consecutive reasoning steps, at fine + coarse granularity. Addresses PRM reward hacking.
- **Method**: **Hierarchical Node Compression (HNC)** — merges two consecutive reasoning steps as cheap data augmentation.
- **Connection**: PRM-side application of [[hrl-overview]]'s temporal-structure principle.

### Emergent Hierarchical Reasoning in LLMs through RL (HICRA) (2025) — `arXiv:2509.03646`
- **Core**: empirical claim — "aha moments," "length-scaling," entropy dynamics are all hallmarks of an **emergent two-phase hierarchy**: first the model firms up low-level execution, then shifts to exploring **high-level planning** (the true driver of sustained improvement).
- **Algorithm**: **Hierarchy-Aware Credit Assignment (HICRA)** concentrates optimization on high-impact planning tokens.
- **Connection**: complements [[cognitive-behaviors]] — the structure exists, RL surfaces it; HICRA targets the planning tokens specifically.

### CodeI/O (2025) — `arXiv:2502.07316`, ICML 2025 Oral
- **Core**: **condense reasoning patterns from code** via input-output prediction. Train models to predict I/O in natural-language CoT given code and test cases.
- **Why**: exposes universal reasoning primitives — logic flow planning, state-space search, decision tree traversal, modular decomposition.
- **Result**: consistent gains across symbolic / scientific / logic / math / commonsense reasoning. CodeI/O++ uses re-execution for verification + multi-turn revision.
- **Connection**: code-as-curriculum for reasoning; pairs with [[logic-rl]]'s rule-based puzzles.

### Intelligence at the Edge of Chaos (2024) — `arXiv:2410.02536`, ICLR 2025
- **Core**: train LLMs to predict **Elementary Cellular Automata (ECA)** rules. Rules with **intermediate complexity** (edge of chaos) produce the most intelligent models on downstream tasks.
- **Result**: trivial and chaotic rules both produce poor downstream models. "Sweet spot of complexity conducive to intelligence."
- **Conjecture**: intelligence arises from ability to predict complexity; creating intelligence may need only exposure to complexity.
- **Connection**: thesis-level support for [[codei-o]] / [[logic-rl]] — synthetic complexity is enough.

### Logic-RL → cross-reference with §3 above
- (See §3 entry — included here thematically as part of "structured reasoning sources.")

### Can Language Models Falsify? — REFUTE (2025) — `arXiv:2502.19414`, COLM 2025
- **Core**: benchmark — given a problem + incorrect solution, can the model **create a counterexample**?
- **Finding**: o3-mini (high) with code execution can falsify only **<9%** of incorrect solutions, despite rating ~48% solve rate.
- **Connection**: exposes a specific reasoning weakness — falsification is much harder than verification or generation. Foundational for adversarial-critic work ([[rlac]]).

### Teaching LLMs to Plan — PDDL-Instruct (2025) — `arXiv:2509.13351`
- **Core**: **logical chain-of-thought instruction tuning** for **symbolic planning** in PDDL. Explicit logical inference for action applicability + state transitions + plan validity.
- **Connection**: planning-side analogue to [[logic-rl]]; structured reasoning as a target capability.

### From Reasoning to Super-Intelligence (2025) — `arXiv:2507.15865`, Shalev-Shwartz & Shashua
- **Core**: theoretical paper. Identifies SFT/RL/ToT/MCTS failure modes: distribution drift, lack of embedded search, exponential inference cost.
- **Proposal**: **Diligent Learner** — models reasoning as **depth-first search guided by a validator** with backtracking on failure.
- **Result**: proves under mild assumptions Diligent Learner efficiently learns from CoT data while existing methods fail.
- **Connection**: search-theoretic foundation for the [[meta-cot]] / [[rstar-math]] cluster.

### rStar-Math (2025) — `arXiv:2501.04519`, Microsoft
- **Core**: small models match/surpass o1 math via **MCTS** + SLM policy + SLM process reward model. **No distillation from superior models.**
- **Three innovations**: code-augmented CoT data synthesis from extensive MCTS rollouts; PPM training avoiding naive step-score annotation; self-evolution recipe (policy + PPM iterated from scratch).
- **Result**: MATH benchmark — Qwen2.5-Math-7B: **58.8 → 89.4%** (8 trajectories) → 90% (64). 1.5B variant: 51.2 → 87.8 → 88.4. **Beats o1-preview by +4.5 / +2.6 and matches o1-mini.**
- **Connection**: classical search-based alternative to [[deepseek-r1]]'s pure RL — shows you can get there from search if you have a good PPM.

### Meta-CoT Survey — Towards System 2 Reasoning (2025) — `arXiv:2501.04682`
- **Core**: Meta Chain-of-Thought (Meta-CoT) framework — explicitly models the **underlying reasoning** that produces a CoT.
- **Pipeline**: instruction tuning with linearized search traces + RL post-training. Covers process supervision, synthetic data generation, search algos.
- **Connection**: framing piece for the reasoning-as-search cluster; bridges [[rstar-math]] and [[evolving-deeper-thinking]].

### Evolving Deeper LLM Thinking — Mind Evolution (2025) — `arXiv:2501.09891`, Google
- **Core**: **evolutionary search** for inference-time compute scaling — language model generates / recombines / refines candidate responses. Doesn't need formal problem specification (just a solution evaluator).
- **Result**: TravelPlanner and Natural Plan benchmarks → **>98% problems solved** with Gemini 1.5 Pro, no formal solver.
- **Connection**: evolutionary cousin of MCTS-based [[rstar-math]]; agnostic to problem structure.

### HiAR-ICL — High-level Automated Reasoning ICL via MCTS (2024) — `arXiv:2411.18478`
- **Core**: MCTS over **5 atomic reasoning actions** to build reusable "thought cards." Shifts ICL from examples to **abstract reasoning patterns**.
- **Result**: MATH **79.6% with 7B** — beats GPT-4o and Claude 3.5.
- **Connection**: ancestor of [[rlad]]'s abstraction-discovery; ICL-side construction of high-level structure.

### MR-Ben — Meta-Reasoning Benchmark (2024) — `arXiv:2406.13975`, NeurIPS 2024
- **Core**: benchmark requires **meta-reasoning** — locate and analyze potential errors in auto-generated reasoning steps. 5,975 expert-curated questions across physics, chemistry, logic, coding.
- **Finding**: o1 series excels (scrutinizes solution space); many SOTA models fall significantly behind.
- **Connection**: empirical eval for [[meta-cot]] / [[scrit]] / [[critique-fine-tuning]] line — measures the critic-skill that those methods try to instill.

### ReGenesis — LLMs Grow into Reasoning Generalists via Self-Improvement (2024) — `arXiv:2410.02108`, Salesforce, ICLR 2025
- **Core**: self-synthesize reasoning paths **progressing abstract → concrete**, guided by various reasoning principles. Fixes the OOD weakness of STaR-style methods.
- **Result**: +6.1% over base on OOD reasoning; existing self-synth methods *lose* 4.6% on average on OOD.
- **Connection**: pre-R1 self-improvement; refines the STaR family with abstraction structure.

### DualFormer — Randomized Reasoning Traces (2024) — `arXiv:2410.09918`, Meta FAIR
- **Core**: single Transformer integrates **System 1 (fast) + System 2 (slow)** via training on **randomized reasoning traces** — drop random parts of the trace.
- **Result**: configurable fast/slow at inference; auto-decides mode if not specified. Improves speed + accuracy + diversity.
- **Connection**: trace-augmentation analogue to [[no-thinking]]'s prompting trick; embedded in the model.

### Thought Cloning (2023) — `arXiv:2306.00323`, NeurIPS 2023 Spotlight
- **Core**: imitation learning of **both behaviors and thoughts**. Train on human demonstrations *with* the language-based reasoning of the demonstrator.
- **Setup**: BabyAI synthetic gridworld with synthetic thought data — proof of concept.
- **Connection**: ancestor of the entire reasoning-RL line. Articulated the thesis that language-mediated thinking is what RL agents lack.

**Group synthesis (hierarchy/search)**:
- **Two paths to structure**: *imposed* (MCTS in [[rstar-math]], evolutionary in [[mind-evolution]], search trees in [[hiaricl]]) vs *learned* (HICRA's planning tokens, RLAD's abstractions, RA3's latent action subspace).
- **Search vs scaled-up RL**: [[rstar-math]] gets to o1-level via search; [[deepseek-r1]] gets there via pure RL. [[meta-cot]] argues they're convergent paths.
- **Substrate finding**: [[intelligence-edge-chaos]] + [[codei-o]] suggest synthetic complex domains (ECA, code) are sufficient pretraining substrates for reasoning. [[reasoning-from-getgo]] takes this maximalist position.

---

## 7. Cross-Cutting Synthesis

> Themes that span all six sections:

### A. The "what changed in 2025" thesis
The field collectively learned that:
1. **Outcome-only RL works** if your base has the cognitive primitives ([[deepseek-r1]] + [[cognitive-behaviors]]).
2. **Data quantity is overrated** — 1K curated examples can match 100K ([[limo]], [[s1]], [[limr]], [[structure-not-content]]).
3. **Verifier-based > verifier-free** under any fixed compute budget ([[ttc-without-verifier]]).
4. **More reasoning ≠ better** — there's a sweet spot ([[thoughtology]], [[underthinking]], [[no-thinking]]).
5. **What's learned is structure, not content** ([[structure-not-content]], [[cognitive-behaviors]]) — and that structure may already be in the base model from pretraining ([[rethinking-reflection]]).

### B. The diagnostic stack
A canonical reading list for "did this RL run actually do anything":
1. **Sober Look** — is your eval honest?
2. **Spurious Rewards / Anchor-Adapter** — are gains from memorization activation?
3. **Understanding R1-Zero / Dr. GRPO** — is your algorithm structurally biased (e.g., length inflation)?
4. **Cognitive Behaviors** — does your base have the primitives that make RL succeed?
5. **Climbing the Ladder** — what tier of problem are you actually moving?

### C. PRM ↔ verifier convergence
The classical PRM line (token-level scores from a separate model) is collapsing into the verifier line (reasoning model that scores at inference). [[deepseek-grm]] + [[eval-time-compute]] + [[rlv]] all point at the same target: **the verifier is itself a reasoning model that scales with compute**. [[prime]]'s implicit reward removes the data-annotation bottleneck. The 2026 follow-ups (RLAC adversarial critic, RubricEM stage-structured) extend this further.

### D. The length-control story
Across [[s1]] (budget forcing), [[l1]] (LCPO), [[mrt]] (meta-RL), [[orion]] (SLPO), [[chain-of-drafts]] (≤5 words), [[asap]] (surprisal pruning), [[no-thinking]] (skip thinking), [[self-train-concise]] (BoN concise), the field is converging on:
- Length should be **adaptive to problem difficulty**.
- Models **internally know** when to stop ([[hidden-state-probe]], [[certaindex]]).
- The right axis is **focus / structure**, not raw length.

### E. The "what's the unit of credit" debate
| Unit | Examples (24-25) | Connection to 2026 wave |
|---|---|---|
| Token | OREAL token-level RM, CGPO, GSPO sequence-ratio | OPD line, SAS |
| Step | PRIME, PAVs, VinePPO, SWEET-RL, FG-PRM, Critical Tokens | SRL, InT, RLAD |
| Trajectory | DeepSeek-R1, ORZ, BOLT, Logic-RL | Vanilla RLVR |
| Block / window | MRT (regret over output blocks), L1 (length window) | TCOD curriculum |
| Hierarchy | HICRA planning tokens, HRM, RA3 abstraction | RubricEM stage-GRPO |

Convergent answer: **step-level dense reward** is the sweet spot when you can get it cheaply (PRIME's implicit rewards make this practical).

### F. The o1 replication map
A timeline of "we can do this too":
- Sep 2024 — Snell TTC scaling
- Oct 2024 — O1 Replication Journey (journey learning), VinePPO, ReGenesis, Critical Tokens
- Jan 2025 — DeepSeek-R1, Kimi 1.5, s1, Sky-T1, rStar-Math, Critique Fine-Tuning, Mind Evolution, Meta-CoT survey, Certaindex
- Feb 2025 — PRIME, OREAL, LIMO, LIMR, Logic-RL, CodeI/O, Structure-not-content, BOLT, Frontier of Learnability, Cognitive Behaviors, Spurious Rewards (early signal)
- Mar 2025 — Dr. GRPO, MRT, L1, Self-Evolving Critic, DAgger-LLM, HRM, Open-Reasoner-Zero, Empirical Study R1-like, UPFT
- Apr 2025 — DeepSeek-GRM, Sober Look, Rethinking Reflection, Thoughtology, Climbing Ladder, No-Thinking, Compute-Optimal Verify, Seed-Thinking-v1.5, Hidden-State Probe
- Jul 2025 — GSPO, Search-Theoretic Perspective
- Aug 2025 — Rubric Anchors, DuPO, Pruning Unsurprising
- Sep 2025 — Emergent Hierarchical, EMPG, Eval-time Compute, AggLM, Teaching LLMs to Plan, What Characterizes Effective Reasoning

### G. What's still open (24–25 vintage)
- **Cognitive primitives**: how do you get verification / backtracking / subgoal-setting into a base model in the first place? [[reasoning-from-getgo]] argues for pretraining-from-scratch RL; nobody has shown it works at scale.
- **Faithful CoT**: explicit cues are verbalized ~20%, implicit cues much less ([[faithfulness-papers]]). CoT may not be the reasoning, just a post-hoc rationalization.
- **OOD reasoning**: [[generalization-tax]] in 2026 argues state-richness is what generalizes; [[sft-memorize-rl-generalize]] says RL generalizes — but these don't reach algorithmic tasks in esoteric languages ([[reasoning-from-getgo]]).
- **Reproducibility**: [[sober-look]]'s indictment hasn't been answered by most subsequent papers. Standard eval norms remain weak.
- **Small-model gap**: [[small-models-struggle]] shows distillation isn't a free path for <3B models — open question what to do for edge deployment.
