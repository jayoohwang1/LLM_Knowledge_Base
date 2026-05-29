# Continuous Autoregressive Language Models (CALM)

> **arXiv** 2510.27688 · WeChat AI (Tencent) + Tsinghua · Shao, Li, Meng, Zhou
> **TL;DR** Replace discrete next-**token** prediction with continuous next-**vector** prediction. An autoencoder compresses $K$ tokens $\to$ 1 continuous vector (recon >99.9% acc), so AR steps drop by factor $K$. Softmax/likelihood gone $\Rightarrow$ build a full **likelihood-free** toolkit: energy-score generative head (single-step), **BrierLM** eval metric, exact + approximate **temperature sampling** from a black-box sampler.

---

## 1. Motivation / Core Idea

- **Bottleneck**: LLM cost scales with **sequence length** in tokens. Token-by-token is the foundational inefficiency.
- **History as the design axis**: char-level $\to$ subword (BPE) each step **increased semantic bandwidth** per unit. CALM = continue that trajectory past the discrete limit.
- **Discrete is a dead end for scaling bandwidth**:
  - Token carries only $\log_2 V \approx 15$–$18$ bits (V = 32K–256K).
  - To pack a whole phrase, vocab must grow **exponentially** $\Rightarrow$ softmax becomes untenable.
- **Continuous fix**: capacity of a vector grows **gracefully** with its dimensionality $l$ $\to$ scale $K$ by scaling $l$.
- **New scaling lever**: beyond params & data, scale **semantic bandwidth $K$** per generative step.
  - *Analogy*: instead of speaking one phoneme at a time, emit a whole compressed "syllable vector" each step; vector dimension = how much you can cram in.

---

## 2. Autoencoder (compress $K$ tokens $\to$ 1 vector)

**Goal**: bijective-ish map $f_{enc}: \mathcal{V}^K \to \mathbb{R}^l$, $g_{dec}: \mathbb{R}^l \to \mathcal{V}^K$ with high-fidelity recon of $\mathbf{x}_{1:K}$.

- **Context-free** by design: each chunk encoded **independently** of neighbors (simplicity + compute). Context-aware AE = future work.

### 2.1 Architecture (mirror enc/dec)
- **Encoder**: embed $K$ tokens $\to$ pos-wise FFN per token $\to$ flatten $\mathbb{R}^{Kd}$ $\to$ linear compress to $\mathbb{R}^d$ $\to$ FFN $\to$ project to latent $\mathbf{z}\in\mathbb{R}^l$.
- **Decoder**: linear+FFN $\mathbb{R}^l\to\mathbb{R}^d$ $\to$ linear expand to $\mathbb{R}^{Kd}$ $\to$ reshape to $K$ hidden states $\to$ FFN $\to$ project to **logits via tied input-embedding matrix** $\to$ argmax.
- **Recon loss** = sum CE over $K$ positions:
$$\mathcal{L}_{ae}(\mathbf{x}_{1:K}) = -\sum_{i=1}^{K}\log p_{dec}(x_i \mid \mathbf{z}=f_{enc}(\mathbf{x}_{1:K}))$$
- **Lightweight**: $K=4$, $l=10$ already gives >99.9% token acc; shallow net, $d=512$. Overhead negligible vs the LM.

### 2.2 Robust Vector Representation (the hard part)
> A recon-only AE produces a **brittle, irregular** latent manifold — small generative errors $\to$ totally wrong tokens. Must make $\mathbf{z}$ **robust** to perturbation, else downstream LM untrainable.

- **Variational (VAE)**: encoder outputs $\boldsymbol{\mu},\boldsymbol{\sigma}$; $\mathbf{z}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2\mathbf{I})$. Smooths manifold (à la latent diffusion).
$$\mathcal{L}_{total}=\mathcal{L}_{ae}+\beta\cdot\mathcal{L}_{KL},\quad \beta=0.001$$
$$\mathcal{L}_{KL}=-\tfrac12\sum_{i=1}^{l}(1+\log\sigma_i^2-\sigma_i^2-\mu_i^2)$$
- **Posterior collapse** problem: some dims collapse to prior $\to$ become pure-noise dims $\to$ inject **chaotic signal** that destabilizes the downstream LM. Worse than just losing capacity.
  - **Fix: KL clipping** (Kingma 2016 free-bits) — clip each dim's KL at a floor $\lambda_{KL}=0.5$:
  $$\mathcal{L}_{KL}^{clip}=\sum_{i=1}^{l}\max(\lambda_{KL},\mathcal{L}_{KL,i})$$
  - Forces every dim to participate $\Rightarrow$ dense, no collapse. (Ablation: removing it $\to$ 71/128 dims collapsed.)
- **Dropout for robustness** (training-only, disabled at LM train/infer):
  - **DropLatent** $p=0.15$ on $\mathbf{z}$ before decoder $\to$ redundancy, robust to LM prediction error.
  - **DropToken** $p=0.15$ mask input tokens $\to$ CBOW-like; AE must infer masked tokens from context $\Rightarrow$ enriches $\mathbf{z}$ with **chunk semantics**, not just token-index compression.
- **Result** ($K=4$, $l=128$): learned $\sigma_i\approx0.3$, so sampling perturbs $\boldsymbol{\mu}$ with substantial noise $\boldsymbol{\sigma}\approx0.3\mathbf{I}$ yet decoder keeps >99.9% acc. **High fidelity + high robustness.**

---

## 3. Likelihood-Free Language Modeling

### 3.1 Next-Vector Prediction setup
- Sequence $T$ tokens $\to L=T/K$ non-overlapping chunks. $\mathbf{Z}=(\mathbf{z}_1,\dots,\mathbf{z}_L)$, $\mathbf{z}_i=f_{enc}(x_{(i-1)K+1},\dots,x_{iK})$.
- AR objective: $p(\mathbf{Z})=\prod_{i=1}^{L}p(\mathbf{z}_i\mid\mathbf{z}_{<i})$.
- **Two challenges** from predicting in uncountable $\mathbb{R}^l$ (no softmax):
  - **Training**: density $p(\mathbf{z}_i\mid\mathbf{z}_{<i})$ intractable $\to$ no MLE / CE.
  - **Eval**: Perplexity needs likelihood $\to$ dead. (See §5 for BrierLM.)

### 3.2 Generative Head
- Backbone Transformer emits conditioning hidden state $\mathbf{h}_{i-1}\in\mathbb{R}^d$; head is a **stochastic fn** drawing $\mathbf{z}_i\sim p(\cdot\mid\mathbf{h}_{i-1})$.
$$\mathbf{h}_{i-1}=\text{Transformer}(\mathbf{z}_{1:i-1}),\quad \mathbf{z}_i\sim p(\cdot\mid\mathbf{h}_{i-1})$$
- **Why not diffusion / flow matching?** They need **dozens–hundreds of NFEs** per vector $\Rightarrow$ kills the speedup from fewer AR steps. CALM needs **single-step** generation $\Rightarrow$ **Energy Transformer** (Shao et al. 2025b).

### 3.3 Energy Transformer

#### Strictly Proper Scoring Rules
- Scoring rule $S(P,y)$ scores prediction dist $P$ vs outcome $y$. Expected score $S(P,Q)=\mathbb{E}_{y\sim Q}[S(P,y)]$.
- **Proper**: $S(P,Q)\le S(Q,Q)$ — truthful reporting maximizes score. **Strictly proper**: equality iff $P=Q$.
- Training by maximizing a strictly proper score = **generalization of MLE** (log score = NLL is one special case). Works even when density intractable.

#### Energy Loss (Energy Score, Székely 2003)
- Likelihood-**free**, only needs **samples** from head (no density). For pred dist $P$, truth $\mathbf{y}$:
$$S(P,\mathbf{y})=\mathbb{E}_{\mathbf{x}',\mathbf{x}''\sim P}[\|\mathbf{x}'-\mathbf{x}''\|^\alpha]-2\,\mathbb{E}_{\mathbf{x}\sim P}[\|\mathbf{x}-\mathbf{y}\|^\alpha]$$
  - **Term 1 (+)**: rewards **diversity** — penalizes collapsed/overconfident identical samples.
  - **Term 2 (−)**: rewards **fidelity** — pulls samples toward truth.
  - Strictly proper for $\alpha\in(0,2)$; default $\alpha=1$. ($\alpha=2$ degenerates: only proper, not strict $\Rightarrow$ BrierLM collapses to 0; $\alpha<1$ training fails — gradient explosion.)
- **MC estimator** = the practical **energy loss**. Two sample sets per step $i$:
  - $N$ **model samples** $\{\tilde{\mathbf z}_{i,1..N}\}$ from the head.
  - **Key trick**: AE gives a **Gaussian posterior** $\mathbf{z}_i\sim q(\cdot\mid x_{(i-1)K+1:iK})$, not a point. Draw $M$ **target samples** $\{\mathbf z_{i,1..M}\}$ from it to cut variance (cheap, since posterior known).
$$\mathcal{L}_{energy}=\sum_{i=1}^{L}\Big(\frac{2}{NM}\sum_{n}\sum_{m}\|\mathbf z_{i,m}-\tilde{\mathbf z}_{i,n}\|-\frac{1}{N(N-1)}\sum_{n\neq k}\|\tilde{\mathbf z}_{i,n}-\tilde{\mathbf z}_{i,k}\|\Big)$$
- Defaults **$N=8$** (small — each is a head fwd pass, scales train cost), **$M=100$** (large — nearly free, stabilizes).
- **Flexibility**: only needs *samples*, near-zero architectural constraints on head.

#### Model Architecture
- **Backbone**: standard Transformer (LLaMA-style: RMSNorm, SwiGLU, RoPE).
- **Energy-Based Generative Head** (single-step):
  - Inputs: hidden $\mathbf{h}_{i-1}$ + random noise $\boldsymbol{\varepsilon}\in\mathbb{R}^{d_{noise}}$, each dim $\sim\mathcal{U}(-0.5,0.5)$.
  - Both projected (independent linears) to internal dim $=d$.
  - Stack of $L$ **residual MLP blocks**: fuse current rep $\boldsymbol{\varepsilon}_l$ with $\mathbf{h}$ via 2 linears $\to$ **SwiGLU** (inter dim $d$) $\to$ residual add. Final linear $\to$ output $\mathbf{z}_i\in\mathbb{R}^l$.
  - One MLP block $\approx 6d^2$ params; #blocks = ¼ of #Transformer layers $\Rightarrow$ head $\approx$ **10% of total params**.
  - *Sampling diversity comes purely from $\boldsymbol{\varepsilon}$* — single forward pass, no iteration.

#### Discrete Token Input (crucial design)
- Naively feeding predicted $\mathbf{z}_{i-1}$ (linear proj to $d$) into backbone **degrades** — model can't unpack the tiny brittle vector.
- **Fix: ground input in discrete space.** Each step: take previous step's $K$ tokens $\to$ embed $\to$ **input compression module** (2-layer MLP) $\to$ single input rep $\to$ backbone.
- **Inference loop**:
  1. **Input**: prev $K$ tokens embedded + compressed $\to$ Transformer.
  2. **Continuous predict**: backbone $\to \mathbf{h}_{i-1}$, head $\to \mathbf{z}_i$.
  3. **Discrete feedback**: $\mathbf{z}_i$ through **frozen AE decoder** $g_{dec}$ $\to$ next $K$ tokens (fed back to step 1).
- Ablation (Table 5): **Discrete input** BrierLM 4.70 vs Continuous 3.25 vs Both 4.40 $\Rightarrow$ discrete clearly best.

---

## 4. BrierLM Evaluation (likelihood-free)

- Perplexity needs explicit likelihood $\Rightarrow$ inapplicable. Energy loss magnitude is **latent-space-dependent** $\Rightarrow$ not comparable. Need a **model-agnostic, likelihood-free** metric.
- **Principle**: metric uniquely minimized at $P=Q$ (can't be hacked). PPL satisfies this via $\mathbb{E}[-\log P]=D_{KL}(Q\|P)+H(Q)$. Naive raw-likelihood $P(y)$ fails (rewards overconfident argmax).

### Brier Score (strictly proper)
$$\text{Brier}(P,y)=2P(y)-\sum_x P(x)^2$$
- Decomposition: $\mathbb{E}_{y\sim Q}[\text{Brier}]=-\underbrace{\sum_x(P(x)-Q(x))^2}_{\text{sq err, min at }P=Q}+\underbrace{\sum_x Q(x)^2}_{\text{const}}$. Balances accuracy + calibrated uncertainty.
- **Likelihood-free unbiased estimator** (two samples from model):
$$\text{Brier}(P,y)\approx \mathbb{1}\{x_1=y\}+\mathbb{1}\{x_2=y\}-\mathbb{1}\{x_1=x_2\},\quad x_1,x_2\sim P$$
  - $\sum_x P(x)^2$ = **collision prob** of 2 indep samples $\to$ $\mathbb{1}\{x_1=x_2\}$.
  - $P(y)$ $\to$ $\mathbb{1}\{x=y\}$.

### BrierLM (n-gram composite)
- Single-token Brier ignores the other $K-1$ tokens $\to$ define **Brier-n** over n-grams (treat n-gram as atomic outcome).
- Geometric mean of Brier-1..4, scaled to 0–100:
$$\text{BrierLM}=100\cdot\Big(\prod_{n=1}^{4}\text{Brier-}n\Big)^{0.25}$$
- **Validated**: vs CE loss, Pearson **−0.966**, Spearman **−0.991** $\Rightarrow$ trustworthy PPL alternative.
- **Universal**: works for any sampler — standard softmax LMs (sample from softmax) **and** diffusion LMs (avoids loose ELBO/PPL estimates). Fair cross-family comparison.

---

## 5. Likelihood-Free Temperature Sampling

> Conventional temp = rescale pre-softmax logits — needs explicit dist. CALM has only a **black-box sampler**. Build temperature sampling from samples alone via **rejection sampling**.

### 5.1 Exact algorithm (Alg 1)
- Target $P_T(x)\propto P(x)^{1/T}$, $T\in(0,1)$. Sample $x$ = a chunk of $K$ tokens.
- **Intuition**: $T=1/n$ (integer $n$) $\Rightarrow$ $P_T\propto P(x)^n$ = prob that $n$ indep draws are **all identical** $\to$ draw $n$ samples, accept iff all equal, else restart.
- **General $T$**: decompose $1/T = n + \alpha$, $n=\lfloor 1/T\rfloor$, $\alpha\in[0,1)$.
  - **Stage 1 (integer $n$)**: draw $n$ samples, must all equal $\to$ candidate $x^*$ (prob $P(x^*)^n$).
  - **Stage 2 (fractional $\alpha$)**: **Bernoulli Factory** (Keane–O'Brien, Mendo 2019) simulates a coin with success prob $P(x^*)^\alpha$ using only base-sampler draws (iterative: draw $x\sim S$, accept if $x=x^*$, else reject w.p. $\alpha/i$).
  - Accept iff both stages pass.
- **Theorem 1**: output $P_T(x)=P(x)^{1/T}/Z_T$, $Z_T=\sum_x P(x)^{1/T}$. **Exact.**

### 5.2 Cost
- **Theorem 2**: $\mathbb{E}[N_{total}]=\dfrac{n+\mathbb{1}(\alpha>0)\sum_x P(x)^{1/T-1}}{Z_T}$.
- **Corollary 2.1** bound: $\le \frac{1+n}{Z_T}$ if $T\le0.5$; $\le\frac{1+|\mathcal X|^{2-1/T}}{Z_T}$ if $0.5<T<1$.
- **Practicality is $T$-sensitive**:
  - $T\to1$: cost scales with sample-space size $|\mathcal X|=|\mathcal V|^K$ (high-temp regime — avoid).
  - Low $T$: needs $n$ identical draws $\to$ vanishing prob $\to$ astronomical rejection. $\Rightarrow$ need approximation.

### 5.3 Batch Approximation (Alg 2) — used in practice
- For $T=1/n$: instead of one risky $n$-tuple trial, draw **batch of $N\gg n$**, use combinatorics.
  - Count occurrences $c_x$ of each unique $x$. Candidate set = those with $c_x\ge n$, weight $w_x=\binom{c_x}{n}$ (# successful $n$-tuples).
  - Sample output $\propto w_x$. **Fallback**: if none $\ge n$, decrement $m\!:\!n\to1$ until non-empty.
  - *Example* $T=0.5,n=2$, batch $\{A,C,A,D,B,E,A,F,B,G\}$: $A$ (×3) $w=\binom32=3$, $B$ (×2) $w=\binom22=1$ $\to$ $P(A)=3/4,P(B)=1/4$.
- **Biased for finite $N$** (ratio-of-expectations $\ne$ expectation-of-ratio), but **Theorem 3**: asymptotically unbiased, $\lim_{N\to\infty}P_{alg}(x;N)=P_T(x)$.
- **$N$ = accuracy/diversity knob.** Black-box $\Rightarrow$ universal controlled decoding for implicit models.
- **Temp sampling analysis** (decompose Brier estimator):
  - **Accuracy** $\mathbb{E}[\mathbb1\{x=y\}]$; **Collision rate** $\mathbb{E}[\mathbb1\{x_1=x_2\}]$ (inverse diversity proxy).
  - Larger $N$ / lower $T$ $\to$ sharper dist (higher acc, higher collision). **$N$ dominates** the trade-off (effect of $T$ capped by finite-batch info).
  - CALM (tune $N$) traces **nearly identical** acc-diversity curve to Transformer (tune $T$): $T=0.6\approx N\!\approx\!100$, $T=0.5\approx N\!=\!200$.

---

## 6. Experiments

- **Data**: train on **Pile** (uncopyrighted), Llama-3 tokenizer, ~230B tokens. Eval **WikiText-103** via BrierLM.
- **Model**: LLaMA family (RMSNorm, SwiGLU, RoPE). Scales S/M/L/XL.
- **Two-stage training**:
  - AE: 15B-token Pile subset, $d=512$, latent $=32K$ (i.e. $l$ scales with $K$), ~75M params, 30k steps, batch 512k tokens.
  - CALM: remaining data, 250k steps, batch **2M tokens** (context 2048 $\Rightarrow$ $2048K$ tokens), AdamW $\beta=(0.9,0.95)$, lr $3\times10^{-4}$ constant, warmup 2000, wd 0.1, grad clip 1.0.

### Main results (Table 1, $K=4$, lower FLOPs / higher BrierLM = better)
| Model | #Params | Train FLOPs (1e20) | Infer FLOPs/tok (1e8) | BrierLM |
|---|---|---|---|---|
| Transformer-S | 281M | 6.6 | 4.4 | 6.05 |
| Transformer-M | 465M | 11.9 | 7.9 | 7.07 |
| Transformer-L | 849M | 22.5 | 15.0 | 8.98 |
| **CALM-M** (K=4) | 371M | **3.7** | **2.9** | 5.72 |
| **CALM-L** (K=4) | 735M | 7.7 | 4.6 | 6.58 |
| **CALM-XL** (K=4) | 1.82B | 19.5 | 9.4 | 8.53 |

- CALM-M ≈ Transformer-S BrierLM, at **44% fewer train FLOPs, 34% fewer infer FLOPs**. CALM establishes a better **performance–compute frontier**; scales with model size like Transformers.
- **Effect of $K$** (Fig 4): $K=1$ much worse (continuous task harder, no compression win) — shows room for improvement. $K=1\to2$ halves cost, marginal perf drop. **$K=4$ surpasses** baseline frontier. $K=8$ worse $\Rightarrow$ capacity limit (bigger models may unlock higher $K$).
- **Learning curve** (Fig 5): CALM-XL slower start (must learn complex continuous dist vs simple token) but **steeper, sustained** improvement.

### Ablations
- **AE regularization** (Table 2): recon-only BrierLM 3.99; naive VAE **hurts** (collapse); **KL-clip** is the key remedy; DropToken+DropLatent add orthogonal gains $\to$ 4.70.
- **$\beta$** (Fig 6): $\beta=0$ baseline; small $\beta$ helps a lot (smooths manifold, recon ~unaffected); $\beta=0.1$ too aggressive (recon $\to$99%, BrierLM drops). Choose **$\beta=0.001$**.
- **Latent dim $l$** (Fig 7): recon high everywhere, but BrierLM **peaks at $l=128$**. Too small ($l=32$) brittle; too large $\to$ encodes noise the head must model. (Dropout scaled with $l$: 0.05/0.1/0.15/0.2 for 32/64/128/256.)
- **AE scaling**: doubling layers/hidden/data gives **no** BrierLM gain $\Rightarrow$ AE task is inherently simple $\Rightarrow$ keep it tiny/negligible.
- **Head: Energy vs Diffusion vs Flow** (Figs 8–9): Energy & Flow > Diffusion. Energy = best ceiling **with zero iterative steps**. Flow midpoint sampler decent in 2–4 steps; diffusion needs ~100 steps.
- **Energy loss $N,M$** (Table 3): more $N$ → better but ~linear cost; $M$ cheaper. $N=8,M=100$ balanced.
- **$\alpha$** (Table 4): $\alpha<1$ Fail; best $\alpha=1$ (4.70); $\alpha=2$ → 0 (not strictly proper).

---

## 7. Related Work (positioning)

- **AE / latent gen**: VAE, latent diffusion, VQ-VAE. CALM = discrete→continuous map prioritizing **robust smooth manifold** (most prompt-compression work prioritizes only fidelity).
- **Text compression**: RNN hidden state, gist/ICAE prompt compression, 500xCompressor, 1568x cramming (Kuratov), DeepSeek-OCR. CALM differs by emphasizing robustness for **generation**.
- **Continuous AR**: GIVT (Gaussian mixture head, limited), MAR/Fluid (diffusion head, slow), Energy Transformer (Shao 2025b). CALM = Energy head + improvements (head arch, energy loss w/ posterior targets, discrete input).
- **Parallel tokens**: NAR-MT, multi-token prediction (Gloeckle), speculative decoding, hierarchical (MegaByte, **Large Concept Models** = SONAR sentence embeddings — heavy AE + diffusion bottleneck; CALM's robust continuous space could help these), block diffusion.
- **Eval**: BLEU/ROUGE/MAUVE/LLM-judge lack scoring-rule guarantees; CALM's Brier is strictly proper + sample-only.
- **Temp sampling**: Bernoulli Factory; VAE/GAN truncation, diffusion noise scaling — all heuristic + white-box. CALM = provably exact black-box.

---

## 8. Future Work
- **AE**: semantically-grounded latent (proximity = semantic similarity); context-aware/AR encoder.
- **Model**: end-to-end generative Transformer; other strictly proper scoring rules.
- **Sampling**: lighter heuristic alternatives to rejection (e.g. scale input noise, or fine-tune to steer).
- **Scaling laws**: unified law with $K$ as a third variable beyond params/data $\Rightarrow$ pick optimal $K$ per compute budget.
- **Algorithmic toolkit**: re-derive RL policy-grad (needs log-prob of rewarded samples) & KD (needs KL over full PMF) in a **sample-based** regime — both currently intractable for CALM.

---
---

## Implementation (codebase)

Repo: `calm/`. HF-`transformers`-based, LLaMA backbone reused. Two-stage: train AE, then train CALM with frozen AE.

### File structure
| Path | Role |
|---|---|
| `models/modeling_autoencoder.py` | VAE: `Encoder`, `Decoder`, `Autoencoder` (K tokens ↔ vector) |
| `models/configuration_autoencoder.py` | AE config (patch/latent/KL/dropout) |
| `models/modeling_calm.py` | `CALM` base: `generate`, `temperature_sampling`, `eval_brier` |
| `models/configuration_calm.py` | `CALMConfig` (patch/noise/samples/beta/...) |
| `models/modeling_energy.py` | `EnergyTransformer` + `MLPGenerator` (energy head + loss) — **default** |
| `models/modeling_flow.py` | `FlowTransformer` + `FlowLoss` (flow-matching head, ablation) |
| `models/modeling_diffusion.py` | `DiffusionTransformer` + `DiffLoss` (diffusion head, ablation) |
| `models/diffusion/` | DiT-style gaussian-diffusion utils (respace, losses) |
| `train/train_autoencoder.py`, `train/train_calm.py` | HF `Trainer` driver scripts |
| `train/*.sh` | launch configs (energy/flow/diffusion/ae/ar) |
| `llama3_tokenizer/` | Llama-3 tokenizer (vocab ~128K) |
| `data/process.py`, `data/get_data.sh` | Pile/WikiText prep |

### Autoencoder — encode/decode K tokens (`modeling_autoencoder.py`)
- **`AELayer`** (`:19`): pre-norm residual `LlamaMLP` (SwiGLU) block. No attention (context-free).
- **`Encoder.forward`** (`:60`):
  - reshape input to `(B*num_patches, patch_size)` then embed — chunks processed **independently** (`:66-69`).
  - **2 stages** of `num_encoder_layers//2` AELayers each; after stage 0, **squeeze**: `view(B*num_patches,1,patch*hidden)` → `squeeze_layer: Linear(patch*hidden → hidden)` (`:81-83`). This is the **flatten+linear compression** $\mathbb{R}^{Kd}\to\mathbb{R}^d$.
  - `hidden_to_latent: Linear(hidden → latent*2)` outputs **mean+log_std concatenated** (`:47`,`:86`).
- **`Decoder.forward`** (`:106`): `latent_to_hidden` (`:98`) → stage-0 layers → **`expand_layer: Linear(hidden → patch*hidden)`** then reshape to `seq*patch` (`:121-123`) → stage-1 layers → `F.linear(hidden, lm_head_weight)` logits with **tied embedding matrix** (`:126`, tie at `:138`).
- **`Autoencoder.forward`** (`:152`) = full VAE training step:
  ```python
  if self.training:                                   # DropToken
      mask = torch.rand_like(input_ids.float()) > self.ae_dropout
      input_ids = input_ids * mask.long()
  latent_states = self.encoder(input_ids=input_ids)
  mean, log_std = torch.chunk(latent_states, 2, dim=-1)
  std = torch.exp(log_std)
  latent_states = mean + torch.randn_like(mean) * std     # reparam
  latent_states = F.dropout(latent_states, p=self.ae_dropout, ...)  # DropLatent
  kl_loss = 0.5 * (mean**2 + std**2 - 1 - log_std*2)
  kl_loss = torch.clamp(kl_loss, min=self.kl_clamp)       # KL free-bits / clip
  kl_loss = kl_loss.sum(-1).mean()
  logits = self.decoder(latent_states).float()
  loss = CE(logits, labels)
  if self.training:
      loss = loss * self.patch_size + kl_loss * self.kl_weight
  ```
  - Matches paper: reparam, KL-clip (`kl_clamp=0.5`), DropToken+DropLatent both = `ae_dropout=0.15`, $\beta=$`kl_weight=1e-3`. Note CE is **mean** then `*patch_size` ≈ sum over K.
- **Config** (`configuration_autoencoder.py:111`): `patch_size=4, latent_size=128, hidden_size=512, intermediate_size=1280, num_encoder_layers=2, num_decoder_layers=2, ae_dropout=0.15, kl_clamp=0.5, kl_weight=1e-3`.

### Energy generative head (`modeling_energy.py`) — default CALM
- **`MLPBlock`** (`:50`): the residual block. `LayerNorm(x)` → concat with conditioning `y` → `Linear(2c→c)→SiLU→Linear(c→c)→SiLU→Linear(c→2c)` → split `gate,up` → **SwiGLU gating** `SiLU(gate)*up` → `down_proj` → `x + step` (`:77-83`). `y` = hidden state, `x` = evolving noise.
- **`MLPGenerator`** (`:100`): the single-step head.
  - Embeds **uniform noise** `noise = rand(...) - 0.5` ∈ U(−0.5,0.5) (`:131`), `noise_embd: Linear(noise_size→hidden)`, separate `hidden_embd` for $\mathbf h$, both LayerNorm'd (`:135-136`).
  - `for block in mlp_blocks: noise_embds = block(noise_embds, hidden_states)` then `final_layer → latent_size` (`:139-144`). **`sample()` is one forward pass — no iteration.**
  - `final_layer` last linear **zero-init** (`:124-125`).
- **`EnergyTransformer`** (`:146`):
  - Loads + **freezes** AE (`:158-165`), `LlamaModel` backbone (`:167`), `embed_proj` = **input compression MLP** `Linear(patch*hidden→2*hidden)→SiLU→Linear(2*hidden→hidden)→LayerNorm` (`:175-180`).
  - **`distance`** (`:188`): $\|x_1-x_2\|_2^\beta$ (β = α in paper, `beta=1.0`).
  - **`energy_score`** (`:191`):
    ```python
    distance_x = pairwise ||x_i - x_j|| .sum / (n_x*(n_x-1))   # diversity term (model-model)
    y = mean + randn(n_y=100,...)*std                          # M=100 target samples from AE posterior
    distance_y = ||x - y||.mean over (n_x, n_y)                # fidelity term (model-target)
    score = distance_x - 2*distance_y
    ```
    Hardcoded **`n_y=100`** = $M$; $N$ = `num_samples` rows of `x`. **Loss = −score** (`:245`).
  - **`forward`** (`:210`) training step:
    - GT latent from **frozen AE encoder** on `labels` shifted by `patch_size` (`:221-228`) → `mean, log_std`.
    - Input: embed `input_ids` → reshape to `(B, latent_len, patch*hidden)`, drop last → `embed_proj` (`:231-232`) — **discrete-grounded input** (the chosen scheme).
    - Backbone → hidden, gather valid patch positions via `patch_mask` (`:237-238`).
    - **Repeat hidden `num_samples` times**, `generative_head.sample()` → N latent predictions (`:241-242`).
    - `loss = -energy_score(...).mean()`.
- **Config** (`configuration_calm.py:108`): `model_type='energy', patch_size=4, num_mlp_layers=4, num_samples=8 (=N), beta=1.0 (=α), noise_size=64, latent_size=128, hidden_size=768, num_hidden_layers=12,...`.

### Flow / Diffusion heads (ablation alternatives)
- **`FlowLoss`** (`modeling_flow.py:50`): **rectified-flow / flow-matching**. `xt = (1-t)x0 + t*target`, predict velocity `v_target = target - x0`, MSE (`:76-82`). `sample()` (`:86`) integrates ODE: **midpoint** solver (2 NFE/step, 10 steps) or Euler (`:93-116`). Net = **`SimpleMLPAdaLN`** (DiT-style: `TimestepEmbedder` + adaLN-zero `ResBlock`s, `:211`).
- **`DiffLoss`** (`modeling_diffusion.py:50`): cosine-schedule gaussian diffusion (`create_diffusion`), VLB loss → net out dim `latent*2` (`:63`); `sample` = `p_sample_loop` **100 steps** (`:94`), supports CFG.
- Both reuse same backbone + `embed_proj`; `FlowTransformer.forward` (`:330`) draws `num_samples` posterior targets and calls `generative_head(z=hidden, target=latent)` instead of energy loss.

### BrierLM eval (`modeling_calm.py:eval_brier :70`)
- Likelihood-free Brier: `E[1{x1=y} + 1{x2=y} - 1{x1=x2}]`, x1/x2 = first 2 of the N model samples decoded via **frozen AE decoder → argmax** (`:91-95`).
- **patch_size ≥ 4** (default): direct — `cumprod` of token-matches gives Brier-1..4 (n-gram = prefix accuracy via cumprod) (`:99-103`).
- **patch_size < 4**: must AR-generate extra patches; accuracy via concatenated shifted patches; **collision term** $\mathbb1\{x_1=x_2\}$ computed by AR-rolling both sample paths through backbone+head+decoder with KV-cache (cache only follows x1's path) (`:107-175`).
- Returns `brier1..4`; **BrierLM** assembled in `train_calm.py:compute_metrics` (`:566`): `(b1*b2*b3*b4)^0.25` (paper scales ×100).

### Temperature sampling (`modeling_calm.py:temperature_sampling :186`) — Alg 2 batch approx
- Requires `1/T` ≈ integer `n` (`:213`). $T=1$ → single sample (`:224`).
- Else: repeat hidden `num_samples`(N, default 200) → head sample → AE-decode → argmax → `(B,N,patch)` token tuples (`:236-241`).
- Per batch elem: `Counter` over tuples; **cascade** `n→1`: candidates with `count≥n`, weight `math.comb(count,n)` $=\binom{c_x}{n}$, `random.choices` weighted (`:255-266`). Exactly paper's Alg 2 (fallback decrement built in).
- **`generate`** (`:278`): custom AR loop (HF `GenerationMixin` incompatible). Pads prompt to multiple of `patch_size`; each step embeds last `patch_size` tokens → `embed_proj` → backbone (KV-cache) → `temperature_sampling` on last hidden → append `patch_size` tokens; EOS/pad bookkeeping; cleans padding at end (`:300-376`).

### Training configs (`train/*.sh`)
- **AE** (`train_autoencoder.sh`): overrides `latent_size=128,num_encoder_layers=2,num_decoder_layers=2,patch_size=4`; block 2048, 30k steps, batch `8*4` per-dev*accum *8gpu, lr 3e-4 constant, warmup 1000, bf16.
- **CALM energy** (`train_energy.sh`): overrides `latent_size=128,num_mlp_layers=4,patch_size=4,hidden_size=1024,intermediate_size=2752,num_hidden_layers=16,num_attention_heads=16` (≈ "L"), block 8192, **250k steps**, batch `4*8*8gpu`, lr 3e-4 constant, warmup 2000, wd 0.1, grad-clip 1.0, **flash_attention_2**, streaming over Pile shards 02–29, eval on WikiText.
- `train_calm.py`: HF `Trainer`; dispatches `model_type` → Energy/Diffusion/Flow (`:433-440`); `group_texts` packs into `block_size` chunks; `labels = input_ids.copy()` (`:525`); WANDB disabled.
