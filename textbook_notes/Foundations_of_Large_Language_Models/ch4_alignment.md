# Chapter 4: Alignment

> **Alignment** = aligning model outputs with human expectations (instructions, intents, values, safety). Pre-training alone insufficient because we cannot collect data covering all tasks + preferences. Two main paradigms used together: **instruction alignment** (SFT) and **human preference alignment** (RLHF).

---

## 4.1 Overview of LLM Alignment

- **Pipeline**: Pre-training → Instruction Alignment (SFT) → Human Preference Alignment (RLHF) → Prompting (inference).
- **Three approaches**:
	- **SFT** — fine-tune on labeled `(instruction, output)` pairs. Specialized, smaller than pre-training dataset. Good when input-output relationship is easy to describe/annotate.
	- **Reward-model-based RL (RLHF)** — train scorer from human preference data, then RL-train LLM against scorer. Needed because values/expectations are hard to fully express via direct labels; teaches *which output is better* rather than fitting a target.
	- **Inference-time alignment** — prompting, reranking outputs via reward/scoring model. No training required; cheap.
- Pre-train then align splits LLM development into two stages: pre-training stage + alignment stage.

---

## 4.2 Instruction Alignment

- **Goal**: tune LLM to accurately respond to instructions/intents (vs. just continuing the text). Also called **instruction fine-tuning**.

### 4.2.1 Supervised Fine-Tuning (SFT)

- **Data**: dataset $\mathcal{D}$ of `(x, y)` where $x$ = instruction + user input, $y$ = expected output.
- **Objective**: maximize conditional log-likelihood of output given input:
	$$\tilde{\theta} = \arg\max_\theta \sum_{(x,y)\in \mathcal{D}} \log \Pr_\theta(y|x)$$
	- Decomposed token-wise: $\log \Pr_\theta(y|x) = \sum_{i=1}^n \log \Pr_\theta(y_i | x, y_{<i})$ — equivalent to cross-entropy loss on output tokens.
- **Trick**: concatenate `[x, y]` into one sequence. Run forward pass as normal language modeling, **mask loss for x tokens to 0** (only compute loss on $y$ tokens).
	- Decomposition: $\log \Pr_\theta(\text{seq}_{x,y}) = \underbrace{\log \Pr_\theta(x)}_{\text{set to 0}} + \underbrace{\log \Pr_\theta(y|x)}_{\text{loss}}$.
	- Lets you reuse standard LM training infra.
- **Multi-turn / conversational SFT**:
	- $K$ rounds $\{x^1, y^1, ..., x^K, y^K\}$.
	- Loss only on chatbot responses $y^k$, conditioned on full prior conversation history.
	- Implementation: single forward pass on the whole conversation; zero out loss for all user inputs $x^k$, keep loss on assistant responses $y^k$.
- **Practical issues**:
	- **Data is labeled, not raw text** → annotation is harder than for traditional NLP since it requires multi-step crafted outputs.
	- **Computationally expensive** for large models → motivates **PEFT** (LoRA, adapters, soft prompts).
	- **Post-training** nature → risk of **catastrophic forgetting** / overfitting. Mitigations: smaller LR, regularization, early stopping, diverse data.

### 4.2.2 Fine-tuning Data Acquisition

#### Manually Generated Data

- Recruit annotators to write `(instruction, input, output)` triples.
	- Harder than classification labeling — more steps, creativity, and consistency across annotators.
- **Prompt template strategies**:
	- Write task-specific templates with slots like `{*text*}`, `{*translation*}`, then fill with existing pairs.
	- Reuse **established NLP benchmarks** (translation, classification, QA): cheap mass-generation via templates.
	- Crowdsource real user questions; collect (or LLM-generate then human-correct) answers — captures real distributions of "new" problems.
- **Diversity matters** — more diverse prompts/tasks → better robustness + generalization.

#### Automatically Generated Data

- Use a well-tuned LLM to label inputs collected via crowdsourcing or scrape. Similar to data augmentation.
- **Limitation**: relies on existing inputs → won't cover open-ended user requests.
- **Self-instruct** (Wang et al. 2023):
	- Maintain pool of `(instruction, input, output)`; seeded with hand-crafted samples.
	- **Loop**:
		1. **Sample** a few instructions from pool (mix hand-written + generated for diversity).
		2. **Instruction generation**: prompt LLM in-context to write a NEW instruction.
		3. **Sample generation**: prompt LLM to complete the new sample (input + output).
		4. **Filter** (similarity heuristics) → add to pool.
	- **Input inversion** trick: specify output (e.g. class label) first, then prompt LLM to generate input — counters class-bias in synthetic data.
- Used in many real SFT pipelines (Ouyang 2022, Alpaca, Vicuna). Evolutionary prompt strategies (Xu 2024) further diversify.
- **In practice**: small expert-collected seed + self-instruct expansion is a common bootstrap recipe.

### 4.2.3 Fine-tuning with Less Data

- Trend: large datasets (e.g., **FLAN**, 15M samples, 1836 tasks) work but are expensive.
- **Counter-trend** — quality > quantity:
	- **LIMA** (Zhou 2023a): only **1,000 carefully curated examples** → LLaMa 65B competitive with much more heavily fine-tuned models.
	- Use GPT-3.5 etc. to **score sample quality** → select top-k for fine-tuning (Chen 2024).
	- Other methods: heuristic filtering, prioritizing high-influence samples.
- **Superficial alignment hypothesis** (Zhou 2023a):
	- Learning happens primarily during **pre-training**; alignment phase mostly surfaces existing abilities.
	- Implies very small alignment datasets can be enough.
	- Extreme: Hewitt 2024 — fine-tune on **responses only** (no instructions) and still elicit instruction-following.
- **Sample efficiency**: SFT is sample-efficient vs. pre-training. RL methods (RLHF) similarly benefit from efficient sampling (reward-model-aided preference sampling, etc.).

### 4.2.4 Instruction Generalization

- Need to generalize over **two axes**: new instructions AND new inputs.
	- Within-task generalization: $\frac{1}{|\mathcal{Z}|}\sum_{z'\in\mathcal{Z}} P(c^*, z', y') > \epsilon$
	- Cross-task: average over all unseen `(c', z')` pairs.
- **Two strategies**:
	1. **Scale data diversity**: e.g., 100+ NLP tasks, 1M+ samples. Multi-task instruction tuning datasets.
		- Limitation: bounded by data diversity/quality; doesn't naturally capture subtle preferences → motivates RLHF later.
	2. **Small but well-chosen datasets** when pre-training already covers OOD variation. Modest fine-tune suffices.
- **Interesting phenomenon**: fine-tuning on math doesn't specialize the model — it may still write poetry on demand. LLM's pre-training-imposed "general-purpose executor" nature is sticky. Sometimes adapting it requires more diverse data.

### 4.2.5 Using Weak Models to Improve Strong Models

- Motivation: as LLMs surpass humans/experts, supervision sources get scarce.
- **Weak-to-strong generalization**: smaller/weaker model supervises stronger model.
	- Burns et al. 2023 — fine-tuning **GPT-4** with **GPT-2-level supervision** improved generalization on NLP tasks.
- **Metric — Performance Gap Recovered (PGR)**:
	$$\text{PGR} = \max\left\{0,\ \frac{P_{w\to s} - P_w}{P_{\text{ceiling}} - P_w}\right\}$$
	- $P_w$ = weak model, $P_{w\to s}$ = strong model fine-tuned on weak labels, $P_{\text{ceiling}}$ = strong model with ground-truth labels.
	- PGR = 1: gap fully recovered. Burns reports ~0.8 PGR on 22 classification tasks.
- **Methods using small models to improve large models** (Fig 4.5):
	- **(a) Synthetic data from weak model** → fine-tune strong on $(x, \hat{y}^w)$.
		- Objective: $\arg\max_\theta \sum_{x\in X} \log \Pr_\theta^s(\hat{y}|x)$.
		- Equivalent to knowledge distillation.
		- Risk: strong model imitates weak model's errors.
	- **(b) KD loss**: add KL divergence between weak and strong distributions to training:
		$$\mathcal{L}_{\text{kd}} = \text{KL}(\Pr^w(\cdot|x)\ \|\ \Pr_\theta^s(\cdot|x))$$
		$$\tilde{\theta} = \arg\max_\theta \sum_{(x,y)\in\mathcal{D}} \log \Pr_\theta^s(y|x) - \lambda \cdot \mathcal{L}_{\text{kd}}$$
		- Anneal $\lambda$ downward as the strong model trains.
	- **(c) Data selection/filtering** with small models via likelihood/cross-entropy criteria.
	- **(d) Ensembling** small models (majority vote, weighted avg, stacking) → strong combined model.
	- **(e) Cascading at inference time**: small model handles easy inputs; pass hard ones to large model. Saves compute/latency.

---

## 4.3 Human Preference Alignment: RLHF

- SFT can't capture subtle/ethical preferences (e.g. "don't explain how to hack").
- **RLHF idea**: train a **reward model** from human preference comparisons, then RL-train the LLM to maximize reward.

### 4.3.1 Basics of Reinforcement Learning

- **Components**:
	- **Agent** = LLM.
	- **Environment** = (mostly conceptual) framework providing rewards.
	- **State** $s$ = context tokens $(x, y_{<t})$.
	- **Action** $a$ = predicted token $y_t$.
	- **Reward** $r(s, a, s')$ or $r(s_t, a_t)$ (Markov: $s'$ determined).
	- **Policy** $\pi(a|s) = \Pr(y_t | x, y_{<t})$ — the LLM itself.
	- **Value functions**:
		- State value: $V(s) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0=s, \pi\right]$
		- Action value: $Q(s,a) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0=s, a_0=a, \pi\right]$
- **Trajectory** $\tau = \{(s_1,a_1),...,(s_T,a_T)\}$; **return** $R(\tau) = \sum_t r_t$.
- **Performance function**: $J(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}}[R(\tau) | \pi_\theta] = \sum_\tau \Pr_\theta(\tau) R(\tau)$.
- **Sparse rewards in NLP**: only get a reward at end-of-sequence ($r_t = 0$ for $t < T$).
- **Policy gradient** (REINFORCE):
	$$\frac{\partial J}{\partial \theta} = \sum_\tau \Pr_\theta(\tau) \frac{\partial \log \Pr_\theta(\tau)}{\partial \theta} R(\tau)$$
	- Reward function need not be differentiable (just samples + scalar reward).
- **Markov decomposition**:
	- $\log \Pr_\theta(\tau) = \sum_t \log \pi_\theta(a_t|s_t) + \sum_t \log \Pr(s_{t+1}|s_t, a_t)$.
	- Dynamics term has no $\theta$ dependence (env independent) → drops out.
	- Final: $\frac{\partial J}{\partial \theta} = \frac{1}{|\mathcal{D}|}\sum_\tau \frac{\partial}{\partial \theta}\left(\sum_t \log \pi_\theta(a_t|s_t) \sum_t r_t\right)$.
- **Variance reduction**: subtract baseline $b$ from returns.
	- Useful identity: future reward $\sum_{k=t}^T r_k$ doesn't depend on past actions → past rewards can be dropped.
	- **Advantage**: $A(s_t, a_t) = \sum_{k=t}^T r_k - V(s_t)$ — how much better is this action vs. average.
- **A2C (Advantage Actor-Critic)**:
	- Actor: policy $\pi_\theta$, updated via $\frac{\partial}{\partial \theta}\left(\sum_t \log \pi_\theta(a_t|s_t) A(s_t,a_t)\right)$.
	- Critic: value function $V_\omega$, predicts expected return; trained to minimize TD error.
	- Two equivalent forms of advantage:
		- $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$
		- $A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$ ← the **TD error**.
	- Value loss: $\mathcal{L}_v(\omega) = \frac{1}{M}\sum (r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t))^2$.

### 4.3.2 Training Reward Models

- **Reward model** $r = \text{Reward}(x, y)$: maps `(input, output)` token sequences → scalar.
	- Acts on **complete** outputs (semantic content); intermediate token rewards = 0.
- **Architecture** (Fig 4.8):
	- Pre-trained LLM (Transformer decoder) consumes `[x, y]`.
	- Take **last token representation** $\mathbf{h}_{\text{last}}$ (e.g. EOS), linear map to scalar: $r(x,y) = \mathbf{h}_{\text{last}} W_r$.
- **Human feedback formats**:
	- **Pairwise comparison** — pick better of two; encoded as $y_a \succ y_b$. Most common.
	- **Rating** — scalar score per output (e.g. 1–5 stars).
	- **Listwise ranking** — order N outputs.
- **Bradley–Terry model** for pairwise preference probability:
	$$\Pr(y_a \succ y_b | x) = \frac{e^{r(x,y_a)}}{e^{r(x,y_a)} + e^{r(x,y_b)}} = \sigma(r(x,y_a) - r(x,y_b))$$
- **Loss**:
	$$\mathcal{L}_r(\phi) = -\frac{1}{|\mathcal{D}_r|}\sum_{(x,y_a,y_b)\in\mathcal{D}_r} \log \Pr_\phi(y_a \succ y_b | x)$$
- Trained as a regular LLM but with pairwise loss instead of CE. After training, apply pointwise to score each $(x, y)$ during RL.

### 4.3.3 Training LLMs (RL Policy Training)

- **Utility function** (sequence-level):
	$$U(\tau; \theta) = \sum_t \log \pi_\theta(a_t|s_t) A(s_t, a_t)$$
- **Loss**: $\mathcal{L}(\theta) = -\mathbb{E}_{\tau \sim \mathcal{D}}[U(\tau; \theta)]$.
- Reframed in LLM notation: $U(x, y; \theta) = \sum_t \log \pi_\theta(y_t | x, y_{<t}) A(x, y_{<t}, y_t)$, $\mathcal{L}(\theta) = -\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}[U(x,y;\theta)]$.
	- $\mathcal{D}$ is input-only (no gold outputs needed).
- **Importance sampling**:
	- Replace $\log \pi_\theta(a_t|s_t)$ with **ratio** $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{ref}}}(a_t|s_t)}$.
	- Allows sampling from older "reference" policy $\pi_{\theta_{\text{ref}}}$ and reweighting → cheaper rollouts.
	- $> 1$: action favored more by current; $< 1$: less.
	- Surrogate: $J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{ref}}}}\left[\frac{\Pr_\theta(\tau)}{\Pr_{\theta_{\text{ref}}}(\tau)} R(\tau)\right]$.
- **Clipping** to bound importance weights:
	$$U_{\text{clip}}(\tau; \theta) = \sum_t \text{Clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{ref}}}(a_t|s_t)}\right) A(s_t, a_t)$$
	- Constrains ratio to $[1-\epsilon, 1+\epsilon]$.
- **Trust region penalty** (keep $\pi_\theta$ near $\pi_{\theta_{\text{ref}}}$):
	$$\text{Penalty} = \sum_t \log \pi_\theta(a_t|s_t) - \sum_t \log \pi_{\theta_{\text{ref}}}(a_t|s_t)$$
- **PPO (Proximal Policy Optimization)** — final objective:
	$$U_{\text{ppo-clip}}(\tau; \theta) = U_{\text{clip}}(\tau; \theta) - \beta \cdot \text{Penalty}$$

#### Four models needed in RLHF

| Model | Symbol | Role | Updates? |
|---|---|---|---|
| **Reward model** | $r_\phi$ | Scores `(x,y)` from preference data | Trained first; **fixed** during policy training |
| **Value model** | $V_\omega$ | Predicts expected sum of rewards from a state | Trained jointly with policy via MSE on TD |
| **Reference model** | $\pi_{\theta_{\text{ref}}}$ | Baseline policy (e.g., SFT model); KL anchor | **Fixed** |
| **Target policy** | $\pi_\theta$ | The LLM being aligned | Trained via PPO loss |

- **Training order**: init reward+value from pre-trained LLM; init ref+target from SFT model. Collect prefs → train reward → freeze. Then jointly: update value via MSE, update policy via PPO. Ref stays fixed.

---

## 4.4 Improved Human Preference Alignment

### 4.4.1 Better Reward Modeling

#### Supervision Signals (ranking variants)

- **Listwise pairwise sum**: accumulate pairwise loss over all ordered pairs in a list:
	$$\mathcal{L}_{\text{list}} = -\mathbb{E}\left[\frac{1}{N(N-1)}\sum_{y_a \succ y_b} \log \Pr(y_a \succ y_b | x)\right]$$
- **Plackett-Luce model** — generalizes Bradley-Terry to full rankings:
	- Worth: $\alpha(y) = \exp(r(x,y))$.
	- $\Pr(y \text{ selected} | x, Y) = \frac{\exp(r(x,y))}{\sum_{y'} \exp(r(x,y'))}$.
	- Ordered list log-prob: $\log \Pr(\mathring{Y}|x) = \sum_k \log \Pr(y_{j_k} | x, \mathring{Y}_{\geq k})$ — soft-max over unselected items at each stage.
	- Loss: $\mathcal{L}_{\text{pl}} = -\mathbb{E}[\log \Pr(\mathring{Y}|x)]$.
- **Pointwise** (regression on absolute scores):
	$$\mathcal{L}_{\text{point}} = -\mathbb{E}[\varphi(x,y) - r(x,y)]^2$$
	- Pros: conceptually simple. Cons: noisy human annotations cause variance; doesn't model relative comparisons → poor generalization.
- **Regularization** to mitigate underdetermination (Eisenstein 2023):
	$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{pair}} - \mathbb{E}[r(x,y_a) + r(x,y_b)]^2$$
- Other learn-to-rank models exist (RankNet, ListNet).

#### Sparse vs Dense Rewards

- RLHF rewards are sparse (end of sequence). But advantage / value-function gives per-step **dense supervision** because future return propagates back.
- **Reward shaping** (Ng 1999):
	$$r'(s_t, a_t, s_{t+1}) = r(s_t, a_t, s_{t+1}) + f(s_t, a_t, s_{t+1})$$
	with $f = \gamma \Phi(s_{t+1}) - \Phi(s_t)$. If $\Phi := V$:
	$$r'(s_t, a_t, s_{t+1}) = r(s_t,a_t,s_{t+1}) + \gamma V(s_{t+1}) - V(s_t)$$
	→ exactly the **advantage**! Advantage = shaped reward.
- Sparse rewards work in RLHF because human end-of-sequence judgments are highly **informative and accurate**.

#### Fine-grained Rewards

- Reward per **segment** $\bar{y}_k$, total $r(x,y) = \sum_k r(x, y, \bar{y}_k)$.
- **Difficulty**: hard to get segment-level human prefs.
- **Workaround for rating-style**: have strong LLM score prefix sequences, define segment score as **difference** of consecutive prefix scores: $s(\bar{y}_k) = s(\bar{y}_1...\bar{y}_k) - s(\bar{y}_1...\bar{y}_{k-1})$. Then regression loss.
- **Classification framing** (e.g. ethical/unethical): hinge loss
	$$\mathcal{L}_{\text{hinge}} = \max(0, 1 - r(x, y, \bar{y}_k)\cdot \hat{r})$$
- Segmentation strategies: fixed-length, sentence boundary, topic shift, reward-change boundary, LLM-identified breaks.

#### Combining Reward Models

- **Overoptimization / reward hacking / Goodhart's law**: when a single reward model is treated as truth, LLM games it → degrades real-world performance.
- Mitigation: **ensemble**:
	$$r_{\text{combine}} = \frac{1}{K}\sum_k w_k \cdot r_k(x, y)$$
- Build diverse reward models from different data subsets, or different alignment dimensions (factuality, completeness, etc.).
- Also: use **off-the-shelf strong LLMs as reward models** — competitive with custom RMs (Lambert 2024).

### 4.4.2 Direct Preference Optimization (DPO)

- **Idea**: skip explicit reward model. Optimize policy directly from preference pairs.
- **Derivation**:
	- PPO-style RLHF objective: $\arg\min_\theta \mathbb{E}_{x,y\sim\pi_\theta}[-r(x,y) + \beta(\log\pi_\theta(y|x) - \log\pi_{\theta_{\text{ref}}}(y|x))]$.
	- Rearrange to isolate $\pi_\theta$:
		$$\arg\min_\theta \mathbb{E}\left[\log\pi_\theta(y|x) - \log\pi_{\theta_{\text{ref}}}(y|x)\exp\left(\tfrac{1}{\beta}r(x,y)\right)\right]$$
	- Define normalized $\pi^*$: $\pi^*(y|x) = \frac{\pi_{\theta_{\text{ref}}}(y|x)\exp(\frac{1}{\beta}r(x,y))}{Z(x)}$ with $Z(x) = \sum_y \pi_{\theta_{\text{ref}}}(y|x)\exp(\frac{1}{\beta}r(x,y))$.
	- Objective becomes $\arg\min_\theta \mathbb{E}[\text{KL}(\pi_\theta(\cdot|x)\ \|\ \pi^*(\cdot|x))]$ → minimized when $\pi_\theta = \pi^*$.
	- Solving for $r$ in closed form:
		$$r(x, y) = \beta\left(\log \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{ref}}}(y|x)} + \log Z(x)\right)$$
- **Plug into Bradley–Terry** — $\log Z(x)$ cancels in differences:
	$$\Pr_\theta(y_a \succ y_b | x) = \sigma\left(\beta \log \frac{\pi_\theta(y_a|x)}{\pi_{\theta_{\text{ref}}}(y_a|x)} - \beta \log \frac{\pi_\theta(y_b|x)}{\pi_{\theta_{\text{ref}}}(y_b|x)}\right)$$
- **DPO loss**:
	$$\mathcal{L}_{\text{dpo}}(\theta) = -\mathbb{E}_{(x,y_a,y_b)\sim\mathcal{D}_r}[\log \Pr_\theta(y_a \succ y_b | x)]$$
- **Pros**:
	- No separate reward model → simpler.
	- No PPO sampling loop → cheaper, more sample-efficient.
	- Supervised-learning-style training.
- **Caveat**: DPO is **offline RL** (fixed preference dataset, no exploration). Online RL (PPO) keeps exploring + can adapt to new states — potentially better generalization on complex tasks.

```python
# Sketch of DPO loss
def dpo_loss(policy_logp_a, policy_logp_b,
             ref_logp_a, ref_logp_b, beta=0.1):
    pi_logratio_a = policy_logp_a - ref_logp_a
    pi_logratio_b = policy_logp_b - ref_logp_b
    logits = beta * (pi_logratio_a - pi_logratio_b)
    return -F.logsigmoid(logits).mean()
```

### 4.4.3 Automatic Preference Data Generation

- **AI Feedback (RLAIF)**: prompt LLM to act as a preference labeler.
	- Example prompt: "You are reviewing customer service responses. Pick which is more courteous, clear, concise…" → returns A or B.
	- Improve with CoT rationales, few-shot examples.
- Extract **label probabilities** from LLM softmax over "A"/"B" tokens → pointwise supervision signals.
- Diversify by varying: LLMs, prompts, ICL demonstrations. Variability in pairwise data is important (Cui 2024, Dubois 2024).
- **Trade-offs**:
	- AI feedback → scalable, objective-friendly, suited for well-defined tasks.
	- Human feedback → better for subjective/value-laden tasks.
	- Hybrid often best.

### 4.4.4 Step-by-step Alignment

- Reward on whole outputs is insufficient for **complex reasoning** (final answer right but steps wrong).
- **Two paradigms** (Uesato 2022):
	- **Outcome-based**: supervise only at final answer (standard RLHF).
	- **Process-based (PRM)**: supervise each reasoning step.
- **Data collection**: collect reasoning paths; humans annotate each step as `correct/incorrect[/neutral]`. Active learning for efficiency — label model-confident-but-wrong steps.
- **Reward model**: Transformer + Softmax classifier; input = `(x, ȳ_{≤k})`; output = distribution over label set.
- **Path-level reward**:
	$$r(x,y) = \sum_{k=1}^{n_s} \delta(\text{correct}, C(x, \bar{y}_{\leq k}))$$
	or use log-probs:
	$$r(x,y) = \sum_k \log \Pr(\text{correct} | x, \bar{y}_{\leq k})$$
- **Connection**: closely related to fine-grained reward modeling (§4.4.1.3). Difference: process-based conditions on **preceding steps** (chain context); fine-grained evaluates each segment **independently**.
- **Big deal for reasoning LLMs**: GPT-o1, o3, DeepSeek R1 use long internal CoT — process supervision is increasingly critical.
- Also helps reduce **redundant reasoning** → cheaper inference.

### 4.4.5 Inference-time Alignment

- Avoid expensive training: align at inference via reranking.
- **Best-of-N (BoN) sampling**:
	1. Sample $N$ outputs: $\{\hat{y}_1, ..., \hat{y}_N\} = \arg\text{TopN}_y \Pr(y|x)$ (sampling or beam search).
	2. Pick best by reward: $\hat{y}_{\text{best}} = \arg\max\{r(x, \hat{y}_1), ..., r(x, \hat{y}_N)\}$.
- **Diversity tradeoff**: if N-best are near-duplicates, BoN gain is limited. Tune sampling hyperparams (temp, top-p) or use multiple LLMs to diversify.
- **Rejection sampling** (related, but training): use BoN to pick best outputs, then **SFT on those** — bakes preference into model w/o RL. Used by WebGPT, LLaMA-2.

---

## 4.5 Summary

- **Coverage**: SFT + RLHF + DPO + inference-time alignment.
- Fine-tuning is **cheaper than pre-training** and addresses problems pre-training can't (e.g., human values).
- Broader context: **AI alignment** predates LLMs (Wiener 1960).
	> "If we use, to achieve our purposes, a mechanical agency with whose operation we cannot efficiently interfere ... we had better be quite sure that the purpose put into the machine is the purpose which we really desire." — Norbert Wiener
- Challenges:
	- Human values are diverse, context-dependent → no single objective fits.
	- **Misaligned outcomes**: systems may satisfy the surrogate objective in harmful ways (reward hacking).
- As LLM capability grows, AI safety + alignment become more urgent (Russell 2019, Bengio 2024).

---

## Key Equations Recap

| Concept | Formula |
|---|---|
| SFT objective | $\arg\max_\theta \sum_{(x,y)} \log \Pr_\theta(y\|x)$ |
| Bradley-Terry | $\Pr(y_a \succ y_b\|x) = \sigma(r(x,y_a) - r(x,y_b))$ |
| Policy gradient | $\nabla J = \mathbb{E}[\nabla \log \pi_\theta(\tau) R(\tau)]$ |
| Advantage (TD) | $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| PPO objective | $\sum_t \text{Clip}(\rho_t) A_t - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$ |
| DPO loss | $-\log \sigma(\beta \log \frac{\pi_\theta(y_a)}{\pi_{\text{ref}}(y_a)} - \beta \log \frac{\pi_\theta(y_b)}{\pi_{\text{ref}}(y_b)})$ |
| PGR (weak→strong) | $\max\{0, (P_{w\to s} - P_w) / (P_{\text{ceiling}} - P_w)\}$ |
