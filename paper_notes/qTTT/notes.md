# Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs

> **arXiv** 2512.13898v1 (15 Dec 2025) · Bansal, Zhang et al. (Meta, Harvard/Kempner, OpenAI, UC Berkeley, UT Austin) · code: none released

**TL;DR:** Long-context LLMs can *consume* more text than they can reliably *use*. Static self-attention suffers **score dilution** at long $T$ (the needle's attention mass vanishes), and inference-time "thinking" tokens can't fix it because they reuse the *same* static attention. Fix = **query-only test-time training (qTTT)**: one prefill to cache K/V, then a handful of gradient steps on *only* the query projection matrices $W_Q$ over short sampled spans, keeping K/V frozen. Provably raises the target–distractor logit margin. Under FLOP-matched budgets beats thinking by big margins (+12.6 / +14.1 avg pts for Qwen3-4B on LongBench-v2 / ZeroScrolls subsets).

---

## 1. Problem & Motivation

- **Long context is the dream use-case:** scientific corpora, whole books, multi-turn histories, multi-file code repos ($10^4$–$10^6$ tokens).
- **Reality:** models miss buried clauses, overlook function defs deep in repos, fail to retrieve facts from prior turns even when content is literally "in context" (lost-in-the-middle).
- **Two ways to spend more compute:**
  1. **Bigger context window** (RoPE scaling, sparse/structured attention, longer-window training, RAG) — but *putting things in context* isn't enough.
  2. **Inference-time / "thinking" compute** (CoT, best-of-$n$, self-consistency, Quiet-STaR) — but **all generate extra tokens with the same static attention** that already mis-allocated mass to the evidence.
- **Central claim:** Both fail structurally at long context because of **score dilution** in static finite-precision self-attention. The cure isn't *more tokens*; it's *changing the $q_i^\top k_j$ similarities* via a tiny amount of context-specific training.

---

## 2. Vanilla Compute-Scaling Strategies Fail for Long Contexts (§2)

### 2.1 Two synthetic sandbox tasks (needle fixed, haystack grows)

Design principle: hold the relevant "needle" fixed, only grow surrounding distractors, isolating effect of length $T$.

| Task | Setup | Output | Length knob |
|------|-------|--------|-------------|
| **Bug Localization in Code Repo** | Inject single-line logical bug into OLMo repo (e.g. missing softmax temp scaling, layernorm misplacement). Sample $L$ lines around bug, extend to other files. | exact `file:line` | $L$ = 5 → 10000 lines |
| **Error in Transaction Log** | Synth multi-account banking logs, initial state + ops, each line `old→new` balances + `TX_ID`. Inject exactly one anomaly. | `{bug_type, TX_ID}` | $n$ = 25 → 500 ops ($\mathcal{O}(10^2)$–$\mathcal{O}(10^4)$ tokens) |

- **Bug types (log task):** `CALC_ERROR` (bad arithmetic), `NEGATIVE_BAL` (over-debit), `LOST_UPDATE` (stale write overwrites prior commit), `DUPLICATE_TXN` (same payment twice).
- Models: **Qwen3 1.7B–8B**. Fig 1 = Qwen3-4B.

**Findings (Fig 1):**
- **(i)** As context grows, **plain in-context accuracy drops sharply & monotonically**.
- **(ii)** **Thinking tokens help at short context but saturate fast**, asymptotically converging to plain model perf at long context.
- **(iii)** **qTTT consistently improves across all lengths** (no diminishing returns).

> **Empirical takeaway:** growing haystack ⇒ sharp monotonic ICL drop; thinking gives only short-horizon gains with clear saturation ⇒ structural limit of static attention.

### 2.2 Preliminaries

Per layer $\ell$, token $i$:
$$q_i^{(\ell)} = W_Q^{(\ell)} h_i,\quad k_j^{(\ell)} = W_K^{(\ell)} h_j,\quad v_j^{(\ell)} = W_V^{(\ell)} h_j$$
$$z_{i,j} := \frac{q_i^\top k_j}{\sqrt{d_k}},\quad \alpha_{i,j} := \frac{\exp(z_{i,j})}{\sum_{\ell=1}^T \exp(z_{i,\ell})},\quad o_i = \sum_{j=1}^T \alpha_{i,j} v_j$$

- **Thinking tokens:** append $M\ge0$ aux tokens at indices $t\in\{i{+}1,\dots,i{+}M\}$, answer at $a=i{+}M{+}1$. Each generated with **static params + same attention kernel** over augmented length $T'=T+M$.

**Definition 2.1 (Retrieval).** Needle = pair $(k_{j^\star},v_{j^\star})$ at $j^\star<i$. Threshold $\tau\in(0,1)$. Retrieval succeeds iff $\alpha_{i,j^\star}\ge\tau$. Margin form $\gamma_i := z_{i,j^\star} - \log\sum_{j\ne j^\star} e^{z_{i,j}}$; succeeds iff
$$\gamma_i \ge \log\!\Big(\frac{\tau}{1-\tau}\Big)$$
All other $j\ne j^\star$ = **distractors**.

### 2.3 Theoretical limits of static attention + thinking

**Lemma 2.2 (Score dilution).** If $\ge m$ distractor keys satisfy $z_{i,j}\ge z_{i,j^\star}-\Delta$ for $\Delta\ge0$:
$$\alpha_{i,j^\star} \le \frac{1}{1+m e^{-\Delta}}$$
- If $m\ge cT$ ($c>0$) and $\Delta=O(1)$ ⇒ $\alpha_{i,j^\star}\to0$ as $T\to\infty$.
- **Intuition:** when a constant fraction of tokens are within $O(1)$ logit of the needle, the softmax budget can't concentrate; needle mass vanishes with length. Even a *unique max* logit gets vanishing attention.

**Lemma 2.3 (Logarithmic margin requirement).** Fix $\varepsilon\in(0,1)$. If
$$\min_{j\ne j^\star}(z_{i,j^\star}-z_{i,j}) \ge \log\!\Big(\frac{(T-1)(1-\varepsilon)}{\varepsilon}\Big)$$
then $\alpha_{i,j^\star}\ge1-\varepsilon$. ⇒ guaranteeing fixed target mass vs worst-case distractors needs **margin $\Omega(\log T)$**.
- Achieving a $\log T$-scaling margin is *hard for static weights* (they don't depend on $T$).

**Proposition 2.4 (Needle-signal bound for thinking tokens).** For any thinking token $t$ and any $u$:
$$\langle u, o_t\rangle \le \alpha_{t,j^\star}\langle u, v_{j^\star}\rangle + (1-\alpha_{t,j^\star})\max_{j\ne j^\star}\langle u, v_j\rangle$$
- Any generated token carries **at most its own attention mass on the needle**.

**Corollary 2.5 (Specialization under small margin).** If $\gamma_t\le\log(\varepsilon/(1-\varepsilon))$ (i.e. $\alpha_{t,j^\star}\le\varepsilon$):
$$\langle u, o_t\rangle \le \varepsilon\langle u,v_{j^\star}\rangle + (1-\varepsilon)\max_{j\ne j^\star}\langle u,v_j\rangle$$
- Under dilution the needle-fraction in any thinking token's output is **provably tiny** ⇒ attending to thinking tokens can't materially raise the answer's effective margin unless some intermediate token *first* puts non-trivial attention on the needle.

> **Takeaways (§2.3):**
> - **(i)** Worst-case retrieval needs margin $\Omega(\log T)$; fixed weights ⇒ dilution & vanishing $\alpha_{i,j^\star}$.
> - **(ii)** Autoregressively generating more tokens with the same static attention does NOT repair missing access to evidence.
> - **(iii)** Any successful inference-time strategy must **change $q_i^\top k_j$** (e.g. update queries), not just sample more tokens.

---

## 3. qTTT: Efficient Test-Time Adaptation via Query-Only Updates (§3)

### Why naïve full TTT is infeasible
- **Full-parameter TTT** (update FFN + $W_Q,W_K,W_V$) on $x_{1:T}$ ⇒ every update alters keys/values ⇒ **invalidates KV cache**, forces fresh forward+backward over *entire* context each step. Prohibitive compute + activation memory.
- **FLOP accounting (App C):** one full-param TTT step over $T$ tokens ≈ generating $\approx 1.2T$ decoding tokens. For $T\approx10^5$ ⇒ one step ≈ generating ~120K tokens. Untenable.

### 3.1 The qTTT method

**Core idea:** single expensive prefill caches K/V (frozen forever), then cheap targeted gradient steps **only on $W_Q$** over short sampled spans, reusing the cache. Keeps the *evidence pathway* (K,V) fixed while *reshaping access* to it.

**Algorithm 1 (Query-Only TTT):**
```
Input: model f_θ, long context x_{1:T}, #steps N_TTT, span length k, step size η
1: {K^(ℓ), V^(ℓ)}_{ℓ=1..L} ← ForwardPassAndCache(f_θ, x_{1:T})   # single O(T^2) op
2: for n = 1 to N_TTT do
3:     sample random span x_s = x_{t:t+k} from x_{1:T}
4:     compute L_TTT(θ; x_s) using FROZEN {K^(ℓ), V^(ℓ)}
5:     update ONLY queries: {W_Q^(ℓ)} ← {W_Q^(ℓ)} − η ∇_{W_Q^(ℓ)} L_TTT
6: end for
7: return adapted f_{θ'} to generate final answer
```

**Two phases (Fig 2):**
1. **Phase 1 — Single-Pass KV Cache Generation.** One full forward pass with pretrained $f_\theta$; store $K^{(\ell)}\in\mathbb{R}^{T\times d_k}$, $V^{(\ell)}\in\mathbb{R}^{T\times d_v}$ per layer. Frozen for whole adaptation. ("expensive, computed only once")
2. **Phase 2 — Span-Sampled Query-Only Objective.** $N_{\text{TTT}}$ steps of grad descent on $W_Q$ only. Standard next-token loss on a small random contiguous span $x_s = x_{t:t+k}$, $k\ll T$:
$$\mathcal{L}_{\text{TTT}}(\theta; x_s) = -\sum_{i=t}^{t+k-1} \log p_\theta\big(x_{i+1}\mid x_{1:i};\{K^{(\ell)},V^{(\ell)}\}_{\ell=1}^L\big) \tag{3.1}$$
   Gradients $\nabla_\theta\mathcal{L}_{\text{TTT}}$ applied **only to $\{W_Q^{(\ell)}\}$**; all other weights + KV cache untouched. ("cheap and fast, computed $N_{\text{qTTT}}$ times")

### 3.2 Why qTTT works — the margin argument (the load-bearing theory)

Targets the §2 bottleneck *directly*: keep evidence (K,V) fixed, reshape query so $q_i^\top k_j$ favors the needle.

**Proposition 3.1 (Query update).** For loss $\ell_i = -\log\alpha_{i,j^\star}$ with fixed $K$, gradient w.r.t. $q_i$:
$$\nabla_{q_i}\ell_i = \frac{1}{\sqrt{d_k}}\Big(\underbrace{\sum_{\ell=1}^T \alpha_{i,\ell}k_\ell}_{\mu_i} - k_{j^\star}\Big)$$
- **Descent step** $q_i\leftarrow q_i - \eta\nabla_{q_i}\ell_i$ moves $q_i$ **toward $k_{j^\star}$** (the needle key) and **away from the attention-weighted mean $\mu_i$** (the distractor centroid). Explicitly counteracts dilution. (Per head; aggregates across heads.)
- **Fig 3 picture:** small margin $\Delta$ ⇒ diffuse attention; grad step $\Delta q\propto t^\star-\mu$ moves query toward needle ⇒ $\Delta$ grows to $\log T$ scale ⇒ attention concentrates.

**Lemma 3.2 (Margin improvement).** Margin $M_i(q_i):=-\ell_i(q_i)$. For small $\eta>0$:
$$M_i\big(q_i - \eta\nabla_{q_i}\ell_i\big) = M_i(q_i) + \eta\|\nabla_{q_i}\ell_i\|_2^2 + O(\eta^2)$$
- **Margin strictly increases** whenever $\nabla_{q_i}\ell_i\ne0$. Gain $\propto \|k_{j^\star}-\mu_i\|_2^2$.
- ⇒ **Improvements are largest exactly when attention is most diffuse** — i.e. the long-context regime where dilution is worst. (If $\ell_i$ is $L$-Lipschitz, $\eta\in(0,1/L)$ guarantees $M_i(q^+)\ge M_i(q)+\frac{\eta}{2}\|\nabla\ell_i\|^2$.)

> **Takeaway:** qTTT reallocates inference compute into **margin-raising** updates: each step provably increases target–distractor logit margin, most when attention is most diffuse, *without* re-encoding context or growing the KV cache.

- **Multi-head:** all statements hold per head; head aggregation is concat + output projection (linear) ⇒ directional & margin arguments preserved.

### 3.3 FLOP equivalence: thinking vs qTTT

Two ways to spend compute after one prefill: (A) generate $T_{\text{think}}$ thinking tokens (frozen), or (C) run $N_{\text{qTTT}}$ query-only updates on spans $k\ll T$ reusing cache. Rule of thumb (App C):
$$\boxed{T_{\text{think}} \approx 2\,N_{\text{qTTT}}\,k}\qquad(\text{long }T,\ k\ll T) \tag{3.2}$$

**App C derivation (cost coefficients, $L$ layers, hidden $d$, MLP ratio $r$):**
- $C_{\text{quad}} = 2Ld$ (quadratic attn), $C_{\text{tok}} = (4{+}2r)Ld^2$ (per-token proj/MLP).
- Prefill: $F_{\text{prefill}}(T) = C_{\text{quad}}T^2 + C_{\text{tok}}T$.
- **Case A (thinking + KV cache):** $F_{\text{gen}}(T_{\text{think}};T) = C_{\text{quad}}\big(T_{\text{think}}T + \tfrac{T_{\text{think}}(T_{\text{think}}-1)}{2}\big) + C_{\text{tok}}T_{\text{think}}$.
- **Case C (qTTT):** per-pass $G_{\text{partial}}(k;T) \approx 2\big(C_{\text{quad}}kT + (2{+}2r)Lk d^2\big)$; total $F_C = F_{\text{prefill}} + N_{\text{qTTT}}G_{\text{partial}}$.
- Cancelling shared prefill and equating gen costs, for $T\gg d$, $k\ll T$ ⇒ $T_{\text{think}}\approx 2N_{\text{qTTT}}k$.

**Numeric instantiation (~8B dense, $T=10^5$):** budget = 8K thinking tokens after prefill ⇒ matched qTTT: $N_{\text{qTTT}}=16$ for $k=128$, or $N_{\text{qTTT}}=8$ for $k=512$. Thinking grows cache by thousands of positions; qTTT keeps cache fixed at $T$ and uses matched FLOPs to *reshape queries* against existing K/V. (Sanity: $L{=}32,d{=}4096,r{=}4,T{=}10^5$, $T_{\text{think}}{=}8000$ ⇒ e.g. $N_{\text{qTTT}}{=}10,k{=}400$ since $2\cdot10\cdot400\approx8000$.)

---

## 4. Experiments (§4)

### Setup & protocol
- **Models:** Qwen3 **1.7B / 4B / 8B** (+ 32B in App F). Native tokenizers, max context windows.
- **Benchmarks (15+ datasets):** **LongBench-v2** (6 subsets), **ZeroScrolls** (8 datasets).
- **Baselines:** (i) **In-context** (plain decode, no extra tokens); (ii) **Thinking** (CoT, compute-matched via §3.3). Uses Qwen3 `/think` and `/no_think` tokens.
- **Default budget:** $T_{\text{think}}=8192$, $k=128$, $N_{\text{qTTT}}=32$, final-answer budget 512 tokens.
- **qTTT hyperparams (App D):** update **only $W_Q$** in all attn layers; **AdamW** (wd 0.01); LR sweep $\{3e{-}4,3e{-}5,1e{-}5,3e{-}6,1e{-}6,3e{-}7\}$, best per-dataset LR on held-out val; batch size 1; grad clip 1.0; bf16; spans sampled uniformly over $[1,T{-}k]$. **Not very sensitive to LR** between $[1e{-}5,1e{-}6]$; only collapses at extremes (Table 1).
- **Decoding:** Thinking temp 0.6 / top-p 0.95 / top-k 20; non-thinking temp 0.7 / top-p 0.8 / top-k 20.

### LongBench-v2 (Fig 4, Table 4)
Multiple-choice across context types: Code Repositories, Long Dialogue History, Long Structured Data, Long In-Context, Multi-Document QA, Single-Document QA.

**Full Qwen3 results (Table 4), bold = best per row/condition:**

| Subset | 1.7B IC / Think / **qTTT** | 4B IC / Think / **qTTT** | 8B IC / Think / **qTTT** |
|---|---|---|---|
| Code Repositories | 26.0 / 18.0 / **26.0** | 25.0 / 28.0 / **32.0** | 30.0 / 44.0 / **52.0** |
| Long Dialogue History | 23.1 / 30.8 / **46.2** | 20.5 / 30.8 / **43.6** | 33.3 / 53.8 / **58.5** |
| Long Structured Data | 27.3 / **30.3** / **30.3** | 30.3 / **35.3** / 35.3 | 34.3 / 38.2 / **42.4** |
| Long In-Context | 18.0 / 20.0 / **28.0** | 21.0 / 25.0 / **33.0** | 32.0 / 40.0 / **44.0** |
| Multi-Document QA | 26.0 / 26.0 / **42.0** | 30.0 / 40.0 / **46.0** | 32.0 / 34.0 / **50.0** |
| Single-Document QA | 32.0 / 34.0 / **38.0** | 35.0 / 42.0 / **48.0** | 32.0 / 44.0 / **46.0** |
| **Average** | **25.4 / 26.5 / 35.1** | **27.0 / 33.5 / 39.6** | **32.3 / 42.3 / 48.8** |

- qTTT wins/ties every subset/model. Biggest gains where evidence is most diffuse (Long Dialogue, Multi-Doc QA). Code Repos scales strongly with model size (Qwen3-8B: 30→44→52).
- gains *grow with model size*.

### ZeroScrolls (Fig 5/9, Table 5)
Grouped: **multi-hop reasoning** (MuSiQue, QASPER, NarrativeQA), **summarization** (GovReport, QMSum, SQuALITY... benchmark mix), **long-passage comprehension** (QuALITY MCQ).

**Full Qwen3 results (Table 5):**

| Dataset | 1.7B IC/Think/**qTTT** | 4B IC/Think/**qTTT** | 8B IC/Think/**qTTT** |
|---|---|---|---|
| GovReport | 22.5 / 21.8 / **26.0** | 24.9 / 20.2 / **33.5** | 22.0 / 17.8 / **29.8** |
| MuSiQue | 11.6 / 22.6 / **26.2** | 17.1 / 7.5 / **30.5** | 22.5 / 43.9 / **48.9** |
| NarrativeQA | 15.0 / 8.9 / **11.7** | 11.0 / 30.0 / **38.0** | 18.9 / 35.1 / **42.8** |
| QASPER | **25.7** / 21.4 / 31.1* | 23.2 / 24.7 / **34.0** | 19.6 / 21.1 / **26.1** |
| QMSum | 6.2 / 7.5 / **9.5** | **10.9** / 7.7 / 8.6 | **9.8** / 8.6 / 8.6 |
| QuALITY | 47.6 / 61.9 / **76.2** | 40.5 / 76.2 / **87.0** | 71.4 / 90.5 / **94.5** |
| SQuALITY | 9.2 / 14.6 / **18.0** | 9.9 / 16.8 / **18.7** | 18.1 / 15.3 / **18.3** |
| SummScreen-FD | 8.2 / 7.2 / **7.4** | **9.9** / 8.3 / **9.9** | **10.4** / 7.9 / 7.9 |
| **Average** | **18.3 / 20.7 / 25.8** | **18.4 / 23.9 / 32.5** | **24.1 / 30.0 / 34.6** |

- **Largest gains on retrieval/multi-hop** (validates score-dilution diagnosis). On **summarization** gains are smaller/comparable to thinking ⇒ when *generation quality* (not retrieval) is the bottleneck, reweighting attention helps less.

> **Takeaways (§4):**
> - **(i)** Consistent gains across benchmarks & sizes; best avg under matched FLOPs.
> - **(ii)** Retrieval-driven tasks benefit most ⇒ confirms dilution diagnosis + margin theory.
> - **(iii)** Thinking is unreliable — sometimes helps, sometimes trails plain In-context (esp. very long contexts, e.g. GovReport/MuSiQue 4B where Think < IC).
> - **(iv)** qTTT = more effective use of inference compute, no change to arch/data/pretraining.

---

## 5. Appendices (key extras)

### App A — Synthetic task examples
- **Code bug (Fig 6):** NL bug description + line-numbered code; output `olmo/model.py:L345` (line where attn weights computed without normalization).
- **Transaction log (Fig 7):** initial state + invariants + transfer log; output `{bug_type: NEGATIVE_BAL, bug_location: TX004}`.

### App B — Proofs (§2)
- Lemma 2.2 proof: $\sum_\ell e^{z_{i,\ell}} \ge e^{z_{i,j^\star}}(1+m e^{-\Delta})$.
- Lemma 2.3 proof: $\sum_{j\ne j^\star}e^{z_{i,j}}\le(T-1)e^{z_{i,j^\star}-\gamma}$ ⇒ $\alpha_{i,j^\star}\ge \frac{1}{1+(T-1)e^{-\gamma}}$.
- Prop 3.1 / Lemma 3.2 proofs use $\partial z_{i,\ell}/\partial q_i = k_\ell/\sqrt{d_k}$ ⇒ $\nabla_{q_i}\ell_i = \frac{1}{\sqrt{d_k}}(\mu_i - k_{j^\star})$, $\|\nabla_{q_i}\ell_i\|^2 = \frac{1}{d_k}\|k_{j^\star}-\mu_i\|^2$.

### App C — FLOP derivations (see §3.3 above).

### App D — Experimental details
- Inputs delimited with explicit headers (`[CONTEXT]`, `[QUESTION]`); UTF-8.
- **Compute matching:** Thinking $T_{\text{think}}=8192$; qTTT $(k,N_{\text{TTT}})=(128,32)$ so $T_{\text{think}}\approx 2N_{\text{TTT}}k$. Self-consistency/best-of-$n$ **disabled** by default to keep FLOPs matched.
- Templates: non-thinking base ("You are a careful assistant. Use only the provided context."), thinking adds `[SCRATCHPAD]` + `Final:` tag (answer extracted after `Final:`).
- Metrics: EM/F1 for QA, ROUGE-{1,2,L} or benchmark metric for summarization, MCQ accuracy for QuALITY.

**Table 1 — LR sensitivity (accuracy on synthetic tasks):** optimal $\eta\in[1e{-}6, 1e{-}5]$.

| Task / Context | 1e-4 | 3e-5 | 1e-5 | 3e-6 | 1e-6 | 3e-7 |
|---|---|---|---|---|---|---|
| Bank 512 | 4.2 | 26.5 | **28.0** | 27.2 | 26.8 | 15.5 |
| Bank 9,560 | 0.0 | 7.8 | **8.4** | 7.9 | 7.0 | 1.2 |
| Code 512 | 8.5 | 42.0 | 44.5 | **45.7** | 43.2 | 22.0 |
| Code 10,000 | 1.0 | 18.2 | **20.2** | 19.5 | 17.8 | 5.2 |
- Extreme high $\eta$ ⇒ instability (collapses to ~0–4); extreme low ⇒ insufficient adaptation.

### App E — Score-dilution evidence (the mechanism check)
- **Attention-mass metric:** for decode step $t$, layer $\ell$, head $h$, target indices $\mathcal{T}$: mass $= \sum_{\tau\in\mathcal{T}} A^{(\ell,h)}_{t,\tau}$, averaged over layers/heads/output steps.
- **RoPE ablation:** disable RoPE ($Q/K$ identity rotation) at inference, no fine-tuning.
- **Tables 2/3 (Qwen3-4B) — accuracy & target attention mass vs context:**

  | Ctx (Bank) | Think(RoPE) Acc/Mass | Think(No-RoPE) | **qTTT** |
  |---|---|---|---|
  | 512 | 36.0 / 0.46 | 34.0 / 0.44 | 28.0 / 0.42 |
  | 2,536 | 6.0 / 0.22 | 5.0 / 0.20 | 14.4 / 0.41 |
  | 5,120 | 2.5 / 0.11 | 0.8 / 0.03 | 10.0 / 0.36 |
  | 9,560 | 1.0 / 0.04 | 0.5 / 0.01 | 8.4 / 0.25 |

  (OLMo Code Bugs same trend: Think mass 0.64→0.12 over 512→10k; qTTT holds 0.58→0.35.)
- **Findings:** thinking accuracy + attention mass both **decay sharply with length**; disabling RoPE *accelerates* the collapse (but it happens even *with* RoPE). qTTT **sustains markedly higher attention mass** ⇒ supports score dilution (not training-data scarcity) as dominant failure mode.

### App F — All models/subsets (incl. Qwen3-32B)
- **Tables 6/7 (Qwen3-32B):** qTTT still wins most subsets, e.g. LongBench-v2 Code Repos IC/Think/qTTT = 36/61/**74**; Multi-Doc QA 35/41/**56**. ZeroScrolls MuSiQue 28.9/54.9/**59.2**, NarrativeQA 27.7/42.4/**49.6**. (A few summarization rows favor In-context/Thinking, e.g. GovReport, QMSum, Long Dialogue.) ⇒ trend holds at 32B.

### App G — Additional test-time scaling baselines (BoN, Beam) under strict FLOP parity
- All matched to extra budget $T_{\text{think}}=8192$: **SC-$N$** (self-consistency / best-of-$N$ majority vote) allots $\approx8192/N$ tokens/sample; **Beam-$k$** allots $\approx8192/k$ tokens/beam.
- **Tables 8/9 (Qwen3-4B), avg accuracy:**

  | Method | LongBench-v2 avg | ZeroScrolls avg |
  |---|---|---|
  | Thinking | 33.5 | 23.9 |
  | **qTTT** | **39.7** | **32.5** |
  | SC-8 | 36.6 | 28.2 |
  | SC-16 | 31.7 | 24.7 |
  | SC-32 | 19.8 | 18.3 |
  | Beam-8 | 33.5 | 26.0 |
  | Beam-16 | 27.8 | 21.4 |
  | Beam-32 | 16.4 | 15.3 |
- **qTTT beats all matched BoN/Beam.** SC-$N$ helps only when single-run accuracy already high (e.g. Single-Doc QA, QuALITY) but **degrades when per-sample acc < 50%** (more samples of bad reasoning = more bad votes; higher $N$ ⇒ fewer tokens/sample ⇒ worse). Beam-$k$ gives modest gains (correlated beams, imperfect ranking), usually trails qTTT.

### App H — Latency / wall-clock (A100, Qwen3-1.7B/4B/8B; Tables 10–12)
- Metrics: $t_{\text{ICL}}$ (vanilla pass ≈ prefill), $t_{\text{think}}$, $t_{\text{BoN}}$, $t_{\text{qTTT}}$ (=$N_{\text{qTTT}}{=}32$ steps, $k{=}128$); $N_{\text{think}}$, $N_{\text{BoN}}$ = #tokens/trajectories matched to $F_{\text{qTTT}}$.
- **Wall-clock for qTTT ≈ thinking ≈ BoN** at matched FLOPs (e.g. Qwen3-8B @ 32k: $t_{\text{qTTT}}$=109.0s, $t_{\text{think}}$=109.0s).
- **Prefill ($\approx t_{\text{ICL}}$) dominates total time**, esp. at long $T$ (8B @128k: $t_{\text{ICL}}$=354s of ~375s) ⇒ **motivates frozen K/V**: without it, prefill would be recomputed every training step (full TTT = death).

---

## 6. Relation to prior work

- **Long-context LLMs:** million-token windows (Gemini 1.5), RoPE scaling (PI, YaRN, Longrope), sparse/structured attn (Longformer, BigBird, Transformer-XL), eval suites (LongBench/-v2, ZeroScrolls, RULER, SWE-bench). Lost-in-the-middle (Liu '24), needle-in-haystack. This work targets the *retrieval* failure via **how attention mass is allocated**.
- **Inference-time scaling:** CoT, self-consistency, best-of-$n$, Quiet-STaR — scale *decoding*, compute-heavy, diminishing returns at long context.
- **Test-time training:** TTT (Sun '20), TTT on nearest neighbors (Hardt & Sun '24), Akyürek '24 (few-shot), TTRL, SEAL/self-adapting, TTT-RNNs (Sun '24). Usually for distribution shift; recent work on long-context. **This paper = first to re-purpose TTT to the micro-distribution of a *single input* via a query-only variant tailored to long context.**
- **Composable:** qTTT is inference-time ⇒ stacks on top of sliding window, adaptive positional encoding, longer-window training tweaks, or RAG.

## 7. Limitations / future directions

- Evaluated a **single point on the $(k, N_{\text{TTT}})$ trade-off** — budget schedules over span size / steps unexplored.
- Compute-matched baseline focused on thinking tokens; integrating self-consistency / best-of-$n$ into the same framework is future work (App G is a start).
- **Gains are task-dependent** (retrieval >> summarization). Need simple predictors for *when to prefer qTTT vs decoding-based scaling*.
- Summarization/generation-quality-bound tasks see limited returns (reweighting attention doesn't fix generation).
- Per-dataset LR selected on held-out val (extra tuning knob, though low sensitivity).

---

## One-line mental model

> **Don't generate more tokens through a clogged attention pipe; spend the same FLOPs to re-aim the query so the pipe actually points at the evidence.** Static attention's $\Omega(\log T)$ margin requirement is unmet by fixed weights and unfixable by thinking (which reuses the same kernel); $W_Q$-only gradient steps on frozen K/V provably move $q_i$ toward $k_{j^\star}$ and away from the distractor centroid $\mu_i$, raising the margin most where attention is most diffuse.
