# Understanding the Physics of KV Cache Compression through Attention Dynamics

**Authors**: Samhruth Ananthanarayanan, Ayan Sengupta, Tanmoy Chakraborty (IIT Delhi) · 2026 · arXiv 2603.01426
**TL;DR**: Reframes KV cache compression from an *engineering memory-reduction trick* into a **structural probe of attention geometry**. Core claim: attention is a **routing system**, not a passive store; keeping a KV pair ≠ keeping it *reachable*. Identifies a universal **hallucination "safety cliff" near 90% compression** = a **phase transition in semantic reachability**, governed by the survival of sparse **token-route lottery tickets (TR-LTs)**.

---

## 1. Central Framing — Compression as Controlled Perturbation of Routing

- **Standard view (what they argue against)**: KV cache is redundant memory; eviction/quantization/clustering remove 70-90% with little benchmark loss → "the cache is mostly redundant."
  - This conflates **storage** with **function**. Benchmarks (LongBench, RULER, InfiniteBench) only report aggregate pass/fail accuracy → coarse, no mechanistic signal.
- **Their view**: self-attention constructs **dynamic token-level computational pathways across heads and layers** (cf. induction heads, circuit-level studies). Reasoning depends on *structured cross-layer routes*, not isolated token storage.
  - **Retaining a KV pair does not guarantee functional accessibility** — a token can remain stored yet become **unreachable** if routing pathways are severed.
  - So KV compression = a **controlled ablation of token-route structures** within self-attention. Physics-inspired (Allen-Zhu "Physics of LLMs" methodology: controlled synthetic settings that expose internal mechanisms).
- **Central research question**: *Does KV compression merely remove redundant memory, or does it reveal and eventually destroy latent sparse token routes that govern reasoning?*

### Three distinguished notions (the key conceptual move)
Benchmarks collapse these; the paper insists on separating them:

| Notion | Question | Failure if violated |
|---|---|---|
| **Retention** (storage) | Is the info still in the cache? | token physically evicted |
| **Accessibility** (reachability) | Is it reachable via a surviving attention pathway? | route severed → **representational erasure** |
| **Utilization** (adaptivity) | Does the model actually route to + use it? | routing too rigid → **representational rigidity** |

- A model may answer correctly because redundancy compensates for routing damage, or fail for reasons unrelated to compression. Separating these enables **causal attribution** of failures.

---

## 2. Central Hypothesis — Sparse Token-Route Lottery Tickets (TR-LTs)

> Within dense attention layers exist **sparse cross-head / cross-layer pathways** — minimal routing backbones — that preserve the **semantic reachability** necessary for correct generation.

- **Mild compression may *expose*** these latent routes (strips interfering/redundant routes → transient improvement).
- **Extreme compression *destroys*** them → not gradual decay but a **structural phase transition** in semantic accessibility.
- **Relationship to Lottery Ticket lineage**:
  - **Frankle & Carbin LTH**: sparse *parameter* subnetworks that train to full accuracy.
  - **Strong LTH for attention (SLTH, Otsuka et al.)**: a sparse *parameter mask* over a wide attention module can approximate dense behavior *without retraining* — statement about **weights**.
  - **TR-LT (this paper)**: complementary **inference-time, token-level** notion. Parameters untouched; compression removes **rows of KV memory** (tokens attention can read), changing **token-to-token connectivity**. Sparsity is defined on the **token-connectivity graph** induced during the forward pass, not on weights.

---

## 3. Two Failure Modes (the core dichotomy)

Compression fails by either *erasing* evidence or leaving it *present but poorly used*:

1. **Representational Erasure** — answer tokens **globally evicted across all heads** → no surviving route → reachability collapse. Diagnosed by **GER** (below).
2. **Representational Rigidity** — tokens **survive** (low eviction) but **excessive head-level consensus** collapses routing flexibility → model can't re-route when its focal set is pruned. Diagnosed by **head consensus**.

> Compression tolerance depends not on raw token count but on **effective route capacity** within attention.

---

## 4. Structural Metrics (the diagnostic instruments)

Let $T$ = context token set, $H$ = head set, $\ell$ = layer, $\alpha$ = compression level, $t^{(\alpha,h)}$ = surviving-token set for head $h$.

- **Survival indicator**: $S_h(t,\alpha) = 1$ if $t \in t^{(\alpha,h)}$ else $0$.
- **Globally evicted** (token gone from *every* head):
  $$\text{GlobalEvicted}(t,\alpha) = \mathbb{1}\!\left[\textstyle\sum_{h\in H} S_h(t,\alpha) = 0\right]$$
- **Eviction Rate**: $\text{EvictionRate}(\alpha) = \frac{1}{|T|}\sum_{t\in T}\text{GlobalEvicted}(t,\alpha)$.
- **Global Eviction Ratio (GER)** — restrict to **answer-relevant tokens** $T_{\text{ans}}$:
  $$\boxed{\text{GER}(\alpha) = \frac{1}{|T_{\text{ans}}|}\sum_{t\in T_{\text{ans}}}\text{GlobalEvicted}(t,\alpha)}$$
  - **Measures route deletion at the evidence level**. High GER → no surviving access path to ground-truth tokens → **hallucination structurally likely**. This is the erasure diagnostic.
- **Head consensus** (coordination / rigidity diagnostic): with $A_{\ell,h}(t)$ = normalized attention weight, top-attended token $t^*_{\ell,h} = \arg\max_t A_{\ell,h}(t)$:
  $$\text{Consensus}(\ell) = \frac{\big|\{t^*_{\ell,h}\}_{h\in H}\big|}{|H|}$$
  - **Low** value = strong agreement (many heads focus on same token → rigid). **High** = diversity (flexible re-routing capacity).
- **Compression susceptibility** (phase-transition signature): $\chi = \partial H / \partial \alpha$ (rate of change of hallucination) — **peaks sharply near $\alpha \approx 0.9$**, consistent with a critical transition in the error landscape.

---

## 5. Synthetic Probing Tasks (controlled datasets)

**Why synthetic**: naturalistic benchmarks can't isolate structural effects. Two construction paradigms: (1) **controlled generative synthesis** (frontier LLM prompted with strict entity-attribute/mention/bidirectional constraints), (2) **deterministic template slot-filling** (eliminates linguistic confounders).

**Four design constraints**: (1) *route sensitivity* — answers depend on specific cross-layer pathways; (2) *minimal redundancy* — prevent accidental recoverability after eviction; (3) *bidirectional querying* — both subject→attribute and attribute→subject; (4) *failure interpretability* — classifiable into erasure vs rigidity.

| Task | Stressor | Probes |
|---|---|---|
| **Base task** | short passage, 1 subject, tightly-bound attributes | eviction precision under minimal redundancy |
| **Knowledge manipulation** | slot-filled biographies (template) | factual-binding stability, systematic eviction bias |
| **Multi presence** | repeated mentions of same entity | **instance disambiguation**, positional robustness |
| **Multi entity** | multiple semantically-linked entities | **cross-entity / cross-head routing integrity** |
| **Long context** | 400-1300+ word passages, distant spans | **route capacity at scale** |
| **Coreference** | pronoun perturbations, ground truth = "I don't know" | fine-grained routing failure + **hallucination/overconfidence** |
| **Hops** | chain-structured entities, multi-hop | preservation of **sequential semantic routes** |

- Context regimes span **150–1300+ tokens**. Stats: Base 200 passages/1199 Q; Knowledge manip 4000 passages/52000 Q (template, not LLM-gen); Coreference 8000/72000; Hops 100/1600; Multi-presence/entity/Long 100/1000 each.
- **Two orthogonal tagging axes** for fine-grained analysis:
  - **Answer-type**: Person, Thing, Organization, Creature, Location, Numerals, Date/Time, Event.
  - **Question-difficulty**: *Standard* (direct retrieval), *Manipulated* (implicit transformation), *Part* (aggregate info across multiple regions).

---

## 6. Experimental Methodology

- **Library**: NVIDIA **KVPress**. Inference-time compression, **no fine-tuning**. Compression ratios mild (10%) → aggressive (90% eviction). Single RTX A6000, greedy decoding, answers matched with LLaMA-3 8B tokenizer for consistency.
- **Scoring function**: **Expected Attention** (Devoto et al.) — token importance from aggregated expected attention weights across decode steps/layers.
- **Two pruning strategies (presses)**:
  - **FINCH / Chunk Press**: chunk-level pruning — partition context into contiguous segments, remove lowest-scoring within each. Coarse, contiguous, cheap.
  - **AdaKV Press**: head-wise *global* pruning — rank KV entries across all heads simultaneously, remove globally lowest, regardless of position. Non-contiguous, finer-grained, more overhead.
  - Comparison reveals whether collapse depends on **local chunk structure** (Chunk) vs **global head importance** (Ada).
- **Two inference regimes**:
  - **Question-agnostic (AGN)**: prune *before* seeing the query → realistic deployment, eviction independent of retrieval demand.
  - **Question-aware (AWR)**: candidate questions known before pruning → **upper bound** of compression tolerance when routing can be query-optimized.
- **Models (2 families, 3B–14B)**: LLaMA-3.2 3B, LLaMA-3 8B; Qwen-2.5 3B / 7B / 14B (all instruct). HF checkpoints, unmodified.
- **Linear probing** (Algorithm 2): train frozen-model linear classifier on pooled hidden states to predict gold answer's concept tag (macro-F1) → measures **what remains linearly accessible in the residual stream** = presence/accessibility, NOT use.

---

## 7. Key Empirical Findings

### 7.1 Moderate compression → large redundancy margins
- Base F1 ~70% (LLaMA), ~68-69% (Qwen) at 0% compression. Performance generally declines with compression but with **localized non-monotonic spikes** at intermediate ratios (esp. Qwen, AWR).
  - **Interpretation**: moderate pruning can *remove interfering/redundant routes* → temporarily improves alignment → direct evidence of **sparse TR-LT substructures surviving partial eviction**.
- **Knowledge manipulation**: both models stay >90% F1 until ~40% compression. Qwen degrades more *gradually* under high compression (esp. AWR) → robustness from **instruction-conditioned reasoning**; LLaMA's strength is **stable factual retention**.

### 7.2 The Hallucination Safety Cliff (the headline result)
- **Universal across all models/presses**: near $\alpha \approx 0.9$, hallucination rates **spike sharply** (Figure 22) — a *cliff*, not a slope.
- **Strongly correlated with GER spikes**: Pearson **r = 0.79–0.93** (p ≤ 0.008) between GER and hallucination rate.
- **Mechanism**: not smooth quality decay but a **sharp rise in probability of global route deletion** — answer-relevant tokens become simultaneously evicted across *all* heads → no remaining pathway to evidence → forced reliance on parametric priors → hallucination.
  - Explains why error curves stay flat across $\alpha$ then break: stable while ≥1 viable route survives; collapses when global deletion becomes common → **phase transition in semantic reachability**.
- **Susceptibility $\chi = \partial H/\partial\alpha$ peaks sharply at ~0.9** → critical-transition signature (physics framing).
- **"unknown"/abstention behavior** more erratic under AdaKV near the cliff.

### 7.3 Architecture-dependent depth dynamics (LLaMA vs Qwen)
Measured via **layer-wise consensus**:
- **LLaMA = "inverted funnel"**: **early-layer consensus (agreement), late-layer diversification**. Early agreement acts like normalization → supports late specialization / parallel feature channels.
- **Qwen = "funnel"**: **early exploration (diversity), late-stage convergence/consolidation** onto a narrow token set (late-stage decision cascade).
- **Implication**: a universal "pyramid" prior (monotone entropy decrease with depth) **transfers poorly** — the fragile computation sits at *different depths* per family. Same depth prior can be safe for one model, brittle for another → **compression must be architecture-aware**.
- Rigidity is most plausible in **late-consolidating architectures** (Qwen-like): late consolidation means token survival can coexist with under-utilization.
- Attention heatmaps: Chunk vs Ada perturb routes differently at moderate compression (contiguous vs irregular removal) but **converge to similar large-scale route destruction at extreme compression** (0.9 overwhelms policy differences).

### 7.4 Eviction is broadly query-agnostic
- Eviction rates **rise ~linearly with $\alpha$** and are **uniform across tasks/model sizes** — pruning applies similar retention dynamics regardless of downstream objective (unless explicitly conditioned). So compression robustness arises from *intrinsic route redundancy*, not query-tuned eviction (except in AWR).

### 7.5 Retention ≠ Utilization (probing vs generation mismatch)
- **Probe scores can stay high while generation fails**: concept vectors remain linearly *decodable* even as hallucination spikes near the cliff.
  - → **representational presence is not the binding constraint**; routing *availability* and *flexibility* are.
- Probe scores can also drop *early* (Fig 25) while generation stays stable until the cliff → **mismatch in trajectories rules out smooth representational decay** → favors reachability-driven phase transition.

### 7.6 Semantic hierarchy of fragility (tag-level)
- **Stable categories**: *Person*, *Thing*, *Location* — concrete nouns, consistent surface forms, direct lexical anchoring in KV; surviving mentions often suffice.
- **Fragile categories**: *Event* (verbs distributed across morphological variants → distributed encoding fragments), *Organization* (nested relational/hierarchical structure → query-aware pruning collapses hierarchy into coarse labels), and especially *Creature* (degrades even at mild compression — a **baseline representational fragility** compression merely amplifies, not a new failure mode).
- **Pattern**: compression preferentially harms categories needing **cross-token relational binding** over those with isolated lexical anchors.
- **Question-type**: *Standard* retrieval stable until moderate (depends on token survival); *Part* retrieval degrades earliest + most **irregularly** (jagged, intermittent route failure — rigidity signature); *Manipulated* exposes architectural split (Qwen keeps AGN/AWR aligned = robust instruction-conditioning; LLaMA diverges under AWR).
- **Multi-hop & long-range reasoning collapse *structurally before* complete token loss** → failure of **semantic connectivity**, not simple memory removal. Relational pathways more fragile than token presence.

### 7.7 Directional asymmetry & question-aware effects
- **Multi presence**: forward queries degrade more sharply than reverse → **asymmetric routing fragility** when entities are repeated. **Multi entity**: more balanced (distributing attributes across distinct entities mitigates interference).
- **Question-aware (AWR) is double-edged**:
  - Helps by injecting an **early discriminative bias** — retains tokens that best separate the answer entity from competitors → improved probe separability, mitigates the cliff, *especially in late-consolidating (Qwen) models*.
  - Hurts in **Coreference**: biases models toward *premature commitment* → reduces ability to correctly abstain ("I don't know") → **more overconfident errors** (LLaMA 75.6→60.9 AGN→AWR; Qwen 78.9→52.0, a 26.9-point drop).
  - Fails for tasks needing **latent bridge tokens** (multi-hop, discrepancy detection) if relevance isn't locally query-aligned → robust AWR pruning must **preserve routes, not just query-similar tokens**.
- **Scale ≠ robustness**: larger models (14B) **not inherently more resilient**; collapse occurs at similar 70-90% thresholds. Compression sensitivity driven by **task structure + entity distribution**, not parameter count. (Qwen-2.5 3B is an exception with more *linear* Multi-entity degradation + markedly lower consensus stability = structurally more fragile internally.)

---

## 8. Theoretical Formalization (Section 6)

### Token-Attention Graph
- **Def 1 (Token-Attention Graph)** $G=(V,E)$: nodes $V=\{(\ell,i)\}$ (layer × position); directed edge $(\ell,i)\to(\ell,j)$ if $\exists h$ with $A_{\ell,h}(i,j) > \varepsilon$. Summarizes which tokens can route info into which at each layer.
- **Def 2 (Compressed Graph)** $G_\alpha$: at level $\alpha$, each head keeps surviving positions $U^{(\alpha)}_{\ell,h}$; remove edges routing through pruned positions. Equivalently: zero the pruned columns of $A_{\ell,h}$ and renormalize.
- **Def 3 (Token-Route Lottery Ticket, TR-LT)**: a subgraph $H_\alpha \subseteq G_\alpha$ such that (i) ∃ answer token $t\in T_{\text{ans}}$ with a directed path from query $q$ to $t$, and (ii) restricting routing to $H_\alpha$'s edges suffices to preserve correct output. = **token-level sparse routing backbone** surviving compression (complements SLTH's parameter-level backbone).

### Reachability propositions
- **Reachability event**: $\mathcal{C}_\alpha(q) := (T_{\text{ans}} \cap R_\alpha(q) \neq \emptyset)$ where $R_\alpha(q)$ = tokens reachable from $q$ in $G_\alpha$. If it fails → no answer token reachable via attention → can't condition final states on evidence.
- **Prop 1 (Redundancy → robustness)**: if ≥ $k$ head-disjoint paths $q\to t$ exist and each head's token survives w.p. ≥ $p$:
  $$\Pr[t \in R_\alpha(q)] \geq 1-(1-p)^k$$
  → probability *all* routes destroyed decays **exponentially in redundancy $k$** → why moderate compression preserves behavior: ≥1 route tends to survive.
- **Prop 2 (Erasure → hallucination)**: under a *no-leakage* assumption (answer not pre-copied into other tokens), if $\exists \alpha^*$ with $T_{\text{ans}} \cap R_{\alpha^*}(q) = \emptyset$, then no self-attention routing can condition on the answer → model defaults to parametric priors → hallucination structurally likely. Each head output is a weighted combination of *surviving* value vectors; if no answer token survives reachably, no aggregation can incorporate its value.
- **Prop 3 (High agreement → reduced re-routing capacity)**: agreement fraction $\rho_\ell = \max_t \frac{1}{H}|\{h: t^*_{\ell,h}=t\}|$. If the consensus token $t^*$ is pruned, a $\rho_\ell$-fraction of heads must shift to secondary tokens; if those exclude $T_{\text{ans}}$, effective routing-to-evidence prob falls $\propto \rho_\ell$. → **rigidity**: evidence retained in memory but no routing flexibility to *use* it.

### Minimal failure decomposition
$$\Pr(\mathcal{F}_\alpha) = \underbrace{\Pr(\mathcal{F}_\alpha \mid \mathcal{E}_\alpha)}_{\approx 1}\Pr(\mathcal{E}_\alpha) + \underbrace{\Pr(\mathcal{F}_\alpha \mid \neg\mathcal{E}_\alpha)}_{\text{rigidity}}\Pr(\neg\mathcal{E}_\alpha)$$
where $\mathcal{E}_\alpha$ = "answer route-deleted across heads."
- **Term 1 = route existence** (erasure): $\Pr(\mathcal{E}_\alpha)$ estimated by **GER**.
- **Term 2 = route usability** (rigidity): proxied by **head-consensus trends**.
- Aligns with empirics: cliff driven primarily by **global route deletion at extreme $\alpha$**; intermediate degradations reflect **survive-but-not-used** in consolidation-heavy regimes.

### TR-LT as inference-time counterpart of SLTH
- SLTH: sparse *parameter* subnetworks approximate dense attention.
- This work: robustness governed by survival of sparse *token-route* subnetworks in the compressed graph. Extreme compression fails when (i) reachability collapses (high GER, Prop 2) or (ii) routing too rigid (high consensus, Prop 3). **It's not the amount of KV retained but whether the compressed routing graph preserves ≥1 viable, usable route to answer evidence.**

---

## 9. Implications for Designing Compression Methods

- **Stop optimizing for token-count retention; optimize for route capacity.** Heavy-hitter / importance heuristics implicitly assume skewed attention + redundant tokens — they ignore whether *routing pathways* survive.
- **Preserve cross-head redundancy** (Prop 1): head-disjoint routes give exponential robustness — pruning that collapses all heads onto the same tokens is dangerous even at low eviction (rigidity).
- **Architecture-aware pruning**: no universal depth/"pyramid" prior — stabilization vs consolidation layers differ by family (LLaMA early / Qwen late). Pruning policy must respect where fragile computation lives.
- **Question-aware pruning must preserve routes, not query-similar tokens** — helps discriminative tasks but breaks latent-bridge / multi-hop reasoning and increases overconfident errors (worse abstention).
- **Co-design over post-hoc pruning**: training objectives could explicitly encourage redundant cross-head routing / controlled consensus to maintain effective route capacity under compression → principled efficient-attention designs preserving structural reachability.
- **GER + consensus as deployment monitors**: GER predicts the hallucination cliff; consensus flags rigidity.

---

## 10. Limitations & Future Work

- **Synthetic datasets** isolate routing-sensitive behaviors but **abstract away** natural-language heterogeneity / large-scale corpora. Need to test reachability-rigidity decomposition on long-doc QA, RAG, code, multimodal.
- **External memory / tool use** may introduce other routing resilience/fragility modes not captured by internal KV pruning.
- **Metrics are explanatory, not yet tight**: GER + consensus show strong empirical alignment but lack formal guarantees. Future: spectral characterizations of attention graphs, connectivity thresholds in layered routing, formal multi-head redundancy bounds.
- Co-design of sparsity mechanisms + architecture is the open modeling direction.

---

## 11. Connections / Where This Sits

- **Methodology**: Allen-Zhu "Physics of LLMs" (controlled synthetic probes of internal mechanisms) + statistical-physics phase-transition language.
- **Theory lineage**: Frankle & Carbin LTH → SLTH for attention (Otsuka) → **TR-LT** (inference-time token-route version).
- **Mechanistic interp**: induction heads, circuit analysis (Elhage, Olsson) — attention as routing/circuits not storage.
- **KV compression methods referenced** as objects of study: H2O, StreamingLLM, Scissorhand, CurDKV (eviction/heavy-hitter); quantization; FINCH (Chunk) + AdaKV (the two presses used); Expected Attention (scoring). Efficient-attention context: Sparse Transformers, Longformer, BigBird, Linformer, Performer, Mamba, FlashAttention, MQA/GQA.
- **Reframes the redundancy debate**: "70-90% of the cache is redundant" is true *for accuracy* but obscures that the surviving 10-30% must contain a viable route backbone — and that backbone is what shatters at the cliff.

---

## Related Notes

> This is an **analysis** paper; the methods below are the *objects of study* whose behavior its theory (token-route lottery tickets, the ~90% GER hallucination cliff, retention≠utilization) aims to explain.

- **[Expected Attention](../Expected_Attention/Expected_Attention.md)** — Directly referenced; one of the scoring presses studied. Its future-query-attention scores are exactly the kind of "accessibility" signal this paper formalizes.
- **[TriAttention](../TriAttention/TriAttention.md)** / **[Compactor](../Compactor/Compactor.md)** / **[SAGE-KV](../SAGE-KV/SAGE-KV.md)** / **[R-KV](../R-KV/R-KV.md)** / **[KVzap](../KVzap/KVzap.md)** — Eviction/heavy-hitter methods whose aggressive budgets this paper contextualizes: the ~90% cliff predicts where they should break, and "retention ≠ utilization" explains why high-attention tokens aren't always the safe ones to keep (cf. R-KV's redundancy fix).
- **[KVTC](../KVTC/KVTC.md)** — The non-eviction (lossy-encode-all) regime; this paper's redundancy-margin finding is the theoretical justification for codec-style approaches that perturb *all* tokens slightly rather than dropping a subset.
