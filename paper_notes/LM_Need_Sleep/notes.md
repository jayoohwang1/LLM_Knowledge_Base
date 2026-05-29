# Language Models Need Sleep

> **TL;DR.** When the context window fills up and the KV cache must be evicted, let the model "sleep": run $N$ extra **offline recurrent passes** over the accumulated context to iteratively refine the **fast weights** (the SSM state) inside its SSM blocks via a learned local update, *then* clear the cache. Bigger $N$ (longer sleep) = deeper reasoning over evicted context, with **no extra wake-time latency** (prediction is still a single forward pass). Shifts compute from inference → consolidation.

- **Authors:** Sangyun Lee, Giulia Fanti (CMU); Sean McLeish, Tom Goldstein (UMD). arXiv 2605.26099, May 2026. Preprint. No code released.
- **One-line thesis:** Scalable *memory* (fixed-size fast weights) ≠ scalable *reasoning*. The bottleneck for reasoning over evicted context is **compute available to transform context into internal state**, not storage capacity. Recurrence is useful not just for prediction but for **memory consolidation**.

---

## 1. Problem & Motivation

- **Transformer scaling pain:** attention compute is $O(T^2)$, KV cache grows $O(T)$ with context length → bad for long-horizon tasks.
- **Hybrid SSM-attention models** (Samba, Hymba, Griffin, Jamba-style, frontier models) interleave:
  - **Attention blocks** $\mathcal{B}^{\text{attn}}$: high-fidelity access to recent tokens, growing KV cache.
  - **SSM / linear-recurrent blocks** $\mathcal{B}^{\text{ssm}}$: fixed-size fast-weight memory for compressed info beyond the window.
- **Key empirical finding (the gap):** vanilla SSM-attention hybrids **degrade as required reasoning depth increases, even when info-to-store is held fixed.**
  - ⇒ failure is **not** memory-capacity-limited (contra prior "SSMs can't copy/recall" work [Jelassi, Arora]).
  - ⇒ failure is **compute-limited**: not enough computation to turn evicted context into a useful internal state.
- **Biological analogy (the "sleep" framing):**
  - Hippocampal **replay** consolidates short-term hippocampal memories into cortical synaptic weights, especially during **sleep** [McClelland; Rasch & Born].
  - Sleep makes animals unresponsive to external stimuli → must provide enough cognitive benefit to justify that cost.
  - **Mapping:** short-term memory = KV cache; long-term memory = fast weights (SSM state); replay = offline recurrent passes; "unable to respond to stimuli" = no new input tokens during sleep.
- **Secondary motivation — depth-recurrent / looped nets:** dynamic-depth models beat fixed-depth ones on sequential reasoning, can solve hard instances fixed-depth models can't. **Insight:** *forming* useful weight memory from observed tokens is itself nontrivial computation (cf. gradient descent improves via iterative weight updates) → allocate recurrence to it.
  - **Difference vs looped models:** prior looped models loop *at prediction time*. Here the loop happens *before* prediction (during sleep), so **prediction stays single-pass**.

---

## 2. Preliminaries / Background

### 2.1 Sequence mixers

- **Softmax attention** [Vaswani]. For token rep $\boldsymbol{x}_t$:
  $$\boldsymbol{q}_t = \mathbf{W}_Q\boldsymbol{x}_t,\quad \boldsymbol{k}_t = \mathbf{W}_K\boldsymbol{x}_t,\quad \boldsymbol{v}_t = \mathbf{W}_V\boldsymbol{x}_t,\qquad \boldsymbol{q}_t,\boldsymbol{k}_t,\boldsymbol{v}_t\in\mathbb{R}^d.$$
  Stack $\mathbf{K}_t=[\boldsymbol{k}_1,\dots,\boldsymbol{k}_t]^\top\in\mathbb{R}^{t\times d}$, $\mathbf{V}_t=[\boldsymbol{v}_1,\dots,\boldsymbol{v}_t]^\top$:
  $$\boldsymbol{o}_t = \mathbf{V}_t^\top\,\mathrm{softmax}\!\left(\frac{\mathbf{K}_t\boldsymbol{q}_t}{\sqrt{d}}\right).$$
  - **Cost:** KV cache $(\mathbf{K}_t,\mathbf{V}_t)$ grows linearly in $t$.

- **Linear recurrent / SSM layer** (Mamba2-style, gated Hebbian-like outer-product update):
  $$\mathbf{S}_t = \alpha_t\,\mathbf{S}_{t-1} + \beta_t\,\boldsymbol{v}_t\boldsymbol{k}_t^\top,\qquad \boldsymbol{o}_t = \mathbf{S}_t\boldsymbol{q}_t.$$
  - $\alpha_t\in(0,1)$ = data-dependent **forget gate**; $\beta_t\in(0,1)$ = data-dependent **input gate**; both computed from $\boldsymbol{x}_t$.
  - **Fast-weight state $\mathbf{S}_t$** does **not** grow with $t$ → memory-efficient but **lossy** (past must be compressed into fixed-size matrix).
  - **In experiments they use Gated Delta Networks (GDN)** [S. Yang et al.] which add a **delta-rule correction**. *The exact update rule doesn't matter for the argument.*

- **Blocks.** A block = sequence mixer + norm + residual + MLP. $\mathcal{B}^{\text{attn}}_\ell$ (attention mixer), $\mathcal{B}^{\text{ssm}}_\ell$ (linear-recurrent mixer).
  - **Attention-only LM:** $\text{Embed}\to\mathcal{B}^{\text{attn}}_0\to\cdots\to\mathcal{B}^{\text{attn}}_{D-1}\to\text{OutProj}$.
  - **Hybrid LM:** $\text{Embed}\to\mathcal{B}^{\text{attn}}_0\to\mathcal{B}^{\text{ssm}}_1\to\mathcal{B}^{\text{attn}}_2\to\mathcal{B}^{\text{ssm}}_3\to\cdots\to\text{OutProj}$.

### 2.2 Synthetic reasoning tasks

- **Rule 110** [Cook]: 1-D binary cellular automaton, fixed local transition rule. Predicting the state after $t$ steps is **P-complete** [Neary & Woods], no general parallel shortcut → clean test of deep sequential computation.
- **Depo** [Allen-Zhu & Li]: multi-hop knowledge retrieval. Sequence = shuffled directed cycle + queries; each query asks for the node reached after $k$ outgoing edges from a start node. Larger $k$ ⇒ deeper graph traversal.
- **Why these:** vary **reasoning demand** while holding **sequence length fixed** → isolate reasoning capability from retrieval/storage capacity.

---

## 3. Motivating Example: Hybrids Fail on Context They Can't Attend To

**Setup (Rule 110):**
- 4 independent length-24 binary strings = 4 initial states; char-level tokenizer ('0'/'1'). States are **unrelated** (not unrolls of each other).
- After all four states (length $T:=24\times4=96$), predict the **first bit** of each state after $t$ transitions → 4 label tokens. Total seq length $T=100$.
- Example:
  ```
  0101...1101 | 1101...1000 | 1101...0110 | 0011...0110 | 1 0 1 0
   state0        state1        state2        state3        labels
  ```
- **$t$ controls reasoning depth.** $t=0$ ⇒ trivial first-bit retrieval. Larger $t$ ⇒ must unroll Rule 110 $t$ times → harder.

**Hard-eviction constraint:**
- Context window size $L=24$; **clear the KV cache every $L=24$ tokens** (denote boundary "|"). So the model sees **one state at a time** and must encode it into fast weights $\mathbf{S}_t$ before $\mathbf{K}_t,\mathbf{V}_t$ are **fully evicted**.
- Splits sequence into:
  - **Consolidation phase** (first 96 tokens): encode context into $\mathbf{S}_t$.
  - **Prediction phase** (last 4 tokens): predict answers.
- **Prediction-phase latency constraint:** each answer token predicted with a **single standard forward pass**. No extra loops / chain-of-thought at prediction time. ⇒ everything needed must already be consolidated into fast weights.

**Result / why it matters:**
- **Plain transformer** can't beat random under hard eviction (KV cache destroyed before prediction).
- **Hybrid (4-layer GDN-attention, layout $\text{attn}\to\text{GDN}\to\text{attn}\to\text{GDN}$)** *can* in principle: store first bit of each evolved state in fast weights. But Fig 2a shows accuracy **drops rapidly as $t$ increases** despite $T$ fixed.
  - ⇒ Not memory capacity (info-to-store fixed). The bottleneck = **deep sequential computation** to simulate $t$ steps, which a fixed-depth model can't scale.
- **"Task failure" caveat:** failure = under a **fixed training-token budget**, not "unlearnable with infinite compute/data." Budget matters because reasoning-intensive data is sparse in web corpora; budget-controlled synthetic tasks surface trends that show up later/less clearly at scale.

---

## 4. Method: LLM Sleep — Offline Recursive Memory Consolidation

### 4.1 Core idea

- When context window fills, **enter sleep**: perform $N$ **offline recurrent passes** over the current context, each refining the fast weights $\mathbf{S}$ in the SSM blocks via the learned local rule (Eq. for $\mathbf{S}_t$), **before** evicting the KV cache.
- Looping over all $D$ blocks:
  $$\text{Embed}\to\big[\,\mathcal{B}^{\text{attn}}_0\to\mathcal{B}^{\text{ssm}}_1\to\cdots\to\mathcal{B}^{\text{attn}}_{D-1}\,\big]^{\times N}\to\text{OutProj}.$$
  - $\times N$ = $N$ looped passes over the architecture.
  - **$N=1$ ⇒ reduces to a vanilla SSM-attention hybrid.**
- Initialize from an SSM-attention hybrid w/ fixed window $L$. The fast weights $\mathbf{S}$ carry across the loops (each pass updates $\mathbf{S}$); after $N$ passes, evict KV cache, process next $L$ tokens.
- **At prediction:** single forward pass over refined memory + current context. The extra compute was already spent during sleep ⇒ **wake-time latency preserved**.

### 4.2 Training (Algorithm 1) — end-to-end backprop through the whole sleep process

```
Algorithm 1  LLM sleep training with hard eviction.
Require: tokens x, loss mask m, window size L, sleep passes N
 1  Zero-initialize SSM fast weights S
 2  Split x, m into non-overlapping chunks of length ≤ L
 3  for each token chunk c and its loss mask m_c do
 4      h ← Embed(c)
 5      if m_c is all-zero then              ▷ consolidation phase
 6          for n = 1, ..., N do
 7              h, S ← Blocks(h, S)          ▷ N recurrent passes refine fast weights
 8          end for
 9      else                                 ▷ prediction phase
10          h, S ← Blocks(h, S)              ▷ single pass
11          L ← MaskedCE(OutProj(h), c, m_c) ▷ masked cross-entropy loss
12      end if
13  end for
14  Backpropagate L and take an optimizer step
```

- **Consolidation chunks** (loss mask all-zero) get $N$ passes; **prediction chunks** get a single pass and contribute the loss.
- **Loss:** masked cross-entropy on prediction tokens only.
- **Gradient flow — key distinction from prior depth-recurrent models:**
  - Backprop through the **entire computational graph** (all $N$ passes), like other depth-recurrent / looped models.
  - **But** gradient flows through the **refined fast weights $\mathbf{S}$** (not through recursively refined *feature vectors* $h$), because **$h$ is discarded after sleep** — only $\mathbf{S}$ persists across the eviction boundary.
- **Architecture diagram (Fig 1):** at eviction boundary, hybrid does $N$ offline recurrent passes over current context, updating fast weights (green) in SSM blocks; then discards attention cache (purple). Later prediction uses consolidated context with **no wake-time looping**. (Example: "Mary has 2 children. Each child has 4 bags..." consolidated → "...bags does Mary have? Answer:" predicted in single pass.)

### 4.3 Two eviction strategies

| Strategy | Behavior | $N=1$ reduces to | Peak inference memory |
|---|---|---|---|
| **Hard eviction** | Clear *entire* KV cache every $L$ tokens; model sees one chunk at a time | vanilla SSM-attention hybrid | capped at $L$ |
| **Sliding-window eviction** | After sleep, keep most recent $L-1$ tokens, evict only older tokens | standard **SWA-SSM hybrid** (sliding-window attention + SSM) | capped at $L$ (same as SWA) |

- Sliding-window: does **not** increase peak memory; $N>1$ just adds recursive consolidation before old context leaves the window.

---

## 5. Experiments

**Common setup:**
- **Optimizer:** Muon for all experiments (following McLeish et al.). AdamW LR fixed $5\mathrm{e}{-5}$; tune only Muon LR. Tune Muon LR on $N=1$ baseline, use that value for looped models.
- **Compute:** automaton < 1 A6000 GPU-day; Depo & GSM-Infinite ≈ 1–2 H100 GPU-days/run.
- **Batch sizes:** 512 (automaton), 128 (Depo), 256 (GSM-Infinite).
- **Fixed random seeds** ⇒ identical data ordering across runs (fair comparison).
- **Models:**
  - §6.1/6.2: 4-layer GDN-attention hybrid, $d=256$ (automaton); 10-layer model $d=512$ from scratch (Depo).
  - §6.3: **Jet-Nemotron 2B** (SSM-attention hybrid fine-tuned from Qwen 2.5 1.5B, replaces some attn layers w/ Jet layers using dynamic convolution); **Ouro 1.4B** (looped attention-only model; insert 6 Jet layers w/o MLP to add fast-weight memory, +<10% params). Muon LR $1\mathrm{e}{-3}$.

### 5.1 Cellular automaton (Rule 110)

- **Fig 2a (effect of rollout step $t$, vanilla hybrid $N=1$):** harder/larger $t$ ⇒ slower convergence & lower final accuracy. 4-/8-step converge fast (early-stopped); 12-/16-step converge slowly. Confirms compute (not capacity) is the bottleneck.
- **Fig 2b (effect of $N$ at hard $t=32$):** "no loop"/2/3/4 loops over ~5B tokens.
  - **no loop:** stuck near random guessing (~10% exact accuracy).
  - **2 loops:** ~20%.
  - **3 & 4 loops:** >30%.
  - Context length, eviction rule, prediction-phase compute all **fixed** across runs ⇒ gains come purely from **consolidation-time compute during sleep**.

### 5.2 Depo (multi-hop retrieval)

- **Setup:** each cycle ≤75 nodes, padded to 300 tokens; query-answer portion = 10 QA pairs ≤60 tokens; total $T=360$. Window $L=75$ ⇒ each cycle **fragmented across 4 cache windows**. At prediction, cycle context already evicted.
- **Harder than automaton** because: (1) cycle spans 4 windows (not 1); (2) must form **query-agnostic** representation ($k$ and start node randomly sampled, vs $t$ fixed in automaton).
- $k$ = difficulty (hops). Train: $k\sim\text{Unif}[1,16]$. Eval test loss on held-out $k\in\{1,2,4,8,16\}$.
- **Fig 3 (test loss, $N\in\{1,2,4\}$):**
  - More offline loops ⇒ faster learning, **especially for higher-hop (deeper) queries**.
  - **1-loop:** little progress on 4-hop+.
  - **2-loop:** stalls on 8-hop+.
  - **Only 4-loop** begins to improve on hardest **16-hop** within budget.

### 5.3 GSM-Infinite (realistic math reasoning, pretrained LLMs)

- **Benchmark** [Zhou et al.]: synthetic, modeled after GSM8K; controls problem length via distractor tokens + difficulty via number of arithmetic operations. Requires **both** long-context processing **and** multi-step reasoning (simple retrieval-augmented baselines / RULER-style fail).
- **Setup:** problems 2,000–3,300 tokens; #operations $\sim\text{Unif}[1,8]$. **Question placed *before* context**, CoT traces **excluded** ⇒ forced to final answer in single prediction pass; question-first lets model **selectively consolidate** relevant info, ignore filler. Window $L=2000$ ⇒ full problem doesn't fit. Eval = 1,600 held-out examples.
- **Two ways to instantiate from pretrained models:**
  - **Jet-Nemotron 2B:** loop over **middle 14 of 28 blocks** (middle-block looping is standard in depth-recurrence). $N\in\{1,2,4,6\}$.
  - **Ouro 1.4B:** loop over **entire blocks** (already pretrained looped). $N\in\{1,2,4\}$ (memory-constrained).
- **Fig 4 (accuracy vs training steps, grouped by #ops 2/4/6/8):**
  - Easy (op2, op4): saturate regardless of loops (esp. Jet, more fast-weight capacity).
  - **Hard (op6, op8): gap widens with more loops.**
    - **Jet:** 6 loops improves op6 $0.742\to0.812$ (9%); op8 $0.351\to0.388$ (11%).
    - **Ouro:** 4 loops improves op6 $0.419\to0.615$; op8 $0.210\to0.272$. (wider gap — maybe reflects depth-recurrent pretraining.)
  - ⇒ sleep-time compute supports multi-step reasoning even on realistic math data w/ pretrained LLMs.

### 5.4 Sliding-window eviction (Ouro 1.4B, GSM-Infinite, $L=512$, $T\approx 4\text{–}6\times$ window)

- $N=1$ = standard **SWA-SSM hybrid** baseline.
- **Warm-up trick:** new Jet layers underutilized if given sliding-window KV cache (cf. Cabannes et al.); so **warm up only Jet layers 1 epoch with hard eviction**, then train full model 2 epochs. **Hard eviction in warm-up is crucial** for $N>1$ to learn to refine fast weights.
- **Fig 5 ($N\in\{1,2,4\}$):** increasing $N$ improves accuracy at **all** operation counts.
  - Unlike $L=2000$ case, this baseline ($L=512$, much smaller than $T$) is **bad even on op2** (most retrieval-heavy under distractors): loops improve op2 $0.596\to0.905$ (**52% improvement**); also op4 +10%, op6 +27%, op8 +18%.
  - **Conclusion:** when active window $\ll$ sequence length, **longer sleep helps not just multi-step reasoning, but also compressing & retrieving relevant context.**

### 5.5 Training throughput (Ouro 1.4B, H200, seq len 12,000)

- **Recurrence across context windows** (sequential dependency: window $j+1$ needs $\mathbf{S}$ from window $j$'s $N$ sleep passes) ⇒ **loses full sequence-axis parallelism** (vs teacher-forced transformers).
  - **But** Fig 6a: if $L$ large enough to keep GPU saturated, **windowed ≈ full-parallel** wall-clock (minimal overhead). Relevant in long-context regimes where $T,L$ both large.
- **Recurrent-depth cost** (Fig 6b): throughput **roughly inversely proportional to $N$** (cost grows ~linearly in $N$), like other depth-recurrent models. Uses activation checkpointing across context-chunk axis to avoid OOM. FlashAttention-2 used.
- **Net:** depth recurrence costs more compute, but consistently improves task performance vs non-recurrent.

---

## 6. Discussion & Limitations

- **Latency win is not free:** during *training*, $N$ deeper forward+backward passes ⇒ slow & potentially unstable training.
  - **Mitigations (future work):** implicit gradients (deep equilibrium models), truncated BPTT, looped-LM training-stabilization techniques.
- **Sequentiality is the point:** sleep makes training sequential across context *and* depth — but that's *why* it helps on these (inherently sequential) tasks. Forcing fully parallel computation on sequential tasks ⇒ brittle shortcut solutions [Liu et al.; serial-scaling hypothesis].
- **Scope of claims:** all "failure"/improvement claims are **under fixed training-token budget**, not infinite-data limits. Evaluated on synthetic tasks + modest-scale pretrained models.

---

## 7. Relation to Adjacent Work

- **Fast weights / linear RNNs / SSMs** (linear attention = online fast-weight matrix-valued memory [Katharopoulos; Schlag — "linear transformers are secretly fast weight programmers"]; delta-rule + gates: GDN, GLA, DeltaNet, Mamba2). Prior work: these *struggle with copying/recall* due to fixed memory [Jelassi; Arora]. **This paper:** they fail as **reasoning depth** grows **even with capacity held fixed** ⇒ different (compute) bottleneck.
- **Context compression** (Ge et al. in-context autoencoder; Eyuboglu et al. Cartridges via offline self-study). Shared goal: spend offline compute once to turn long context into reusable compact state. **Diff:** those *shorten what remains in attention*; this transfers **evicted** context into weight-based memory.
- **Context distillation** (Snell; Caccia; Chen "Generative Adapter"; Tack; Cao Infiniteicl) — distill active context into weights via predefined losses (imitate teacher, reconstruct, predict continuation, answer questions). **Diff:** here a **learned recurrent forward pass** transfers context to weights, not gradient descent on predefined losses.
- **Test-time training** (Tandon et al. end-to-end TTT for long context): replace full attention w/ sliding-window + one gradient step per chunk on standard CE, storing long-range info in temporary param updates. **Diff:** this uses a **learned recurrent forward pass** as the memory-update rule ⇒ more flexible consolidation than a one-step GD on a fixed scalar objective. Also: they eval perplexity on web text (retrieval+reasoning entangled); this uses synthetic tasks that **independently control reasoning depth & problem length**, showing sleep helps most when **reasoning depth increases**. (Zhang et al.: LoRA adapter from context chunk in RL setting — updates weights only once per chunk.)
- **Depth-recurrent / looped models** (Universal Transformers; ACT [Graves]; depth-adaptive transformer; Geiping "scaling test-time compute with latent reasoning"; Ouro; Parcae; iso-depth scaling laws; McLeish "retrofitted recurrence"; Merrill & Sabharwal "log-depth → Turing-complete"). **Diff:** prior looped models loop *at prediction time*; here recurrence is **offline (sleep)**, so prediction stays single-pass; and the loop refines **fast weights** (discarded features) not feature vectors.
- **Offline planning / neuroscience** (Tolman cognitive maps; Momennejad offline replay supports planning; Chalvidal recursive Hebbian-like weight updates for adaptation; Lin et al. "sleep-time compute"). **Diff:** recursively update fast weights during sleep-like offline phase to improve reasoning over evicted context **under a strict prediction-phase latency constraint.**

---

## 8. Key Takeaways

- **Reframing:** the limit on reasoning-over-evicted-context is **consolidation compute**, not memory size.
- **Mechanism:** $N$ offline recurrent passes refine fixed-size fast weights $\mathbf{S}$ before KV eviction; gradient flows through $\mathbf{S}$ (features discarded).
- **Free lunch at inference:** prediction is single-pass — extra cost is paid during "sleep," not "wake."
- **Scaling law (empirical):** increasing sleep duration $N$ monotonically improves reasoning, with **largest gains on the hardest / deepest instances** (high-$t$ automaton, high-$k$ Depo, high-op GSM-Infinite); easy instances saturate regardless of $N$.
- **Generality:** works from scratch (automaton, Depo) and as a **post-training add-on** to pretrained hybrids (Jet-Nemotron) and looped LMs (Ouro); compatible with both **hard** and **sliding-window** eviction (latter helps retrieval too, no extra peak memory).
- **Cost:** training throughput ∝ $1/N$ (depth-recurrent cost); window-sequential dependency adds little overhead when $L$ large enough to saturate GPU.
