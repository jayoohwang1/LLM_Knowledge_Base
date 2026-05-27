# Additional Textbook Suggestions

Gaps not covered by the current library (Bishop & Bishop, Prince, Mathematics for ML) or Sutton & Barto RL.

---

## Tier 1 — Clear Gaps, High Priority

**Speech and Language Processing — Jurafsky & Martin (3rd ed.)**
- Free draft: https://web.stanford.edu/~jurafsky/slp3/
- Fills: NLP fundamentals, tokenization history, linguistic structure, evaluation metrics, ASR/TTS, coreference, semantic parsing
- 3rd ed. is current: DPO in ch. 9, updated transformer/LLM sections, new ASR/TTS chapters
- Read selectively: embeddings, LM, transformer, and alignment chapters are highest priority; syntax/parsing chapters are lower priority unless you do structured prediction

**Convex Optimization — Boyd & Vandenberghe**
- Free PDF: https://web.stanford.edu/~boyd/cvxbook/
- Fills: optimization theory depth — Lagrangians, duality, KKT conditions, proximal methods
- Relevant for: PPO's clipped objective, constrained fine-tuning formulations, regularization theory
- Don't read cover to cover — chapters 1–5 and the duality chapter are the high-value parts

**Elements of Information Theory — Cover & Thomas**
- Fills: information theory beyond the brief entropy/KL coverage in Bishop
- Relevant for: perplexity as bits-per-character, compression/generalization connections, mutual information in representation learning and interpretability, rate-distortion (quantization, tokenization theory)
- Prioritize the first six chapters

---

## Tier 2 — Important, Use Selectively

**Understanding Machine Learning — Shalev-Shwartz & Ben-David**
- Free PDF available online
- Fills: statistical learning theory — PAC learning, VC dimension, Rademacher complexity, generalization bounds
- Relevant for: reasoning about why scaling laws hold, data requirements, theory-adjacent research
- More useful for sharpening intuitions than day-to-day research; most frontier lab researchers don't use this daily

**The Elements of Statistical Learning — Hastie, Tibshirani, Friedman**
- Free PDF: https://hastie.su.domains/ElemStatLearn/
- Fills: classical ML reference — ensemble methods, regularization paths, kernel methods, boosting
- Best used as a reference rather than a cover-to-cover read
- Useful when classical techniques (gradient boosting, kernel tricks, regularization paths) appear in papers

---

## Tier 3 — Niche but Growing Importance

**Elements of Causal Inference — Peters, Janzing, Schölkopf**
- Free PDF from MIT Press
- Fills: causal reasoning — counterfactuals, interventions, spurious correlation vs. causal structure
- Relevant for: understanding what LLMs learn vs. exploit, alignment/evaluation methodology, Schölkopf-school research
- Not mainstream in LLM engineering today, but increasingly relevant for alignment and evaluation work

**Foundations of Large Language Models — Xipeng Qiu et al. (2025)**
- Specifically covers: pretraining, generative models, prompting, alignment, inference as a unified treatment
- Structured around frontier lab problem areas
- Weaker endorsement — check the table of contents before committing

---

## What to Skip

- **Koller & Friedman "Probabilistic Graphical Models"** — encyclopedic, most relevant content is already partially in Bishop & Bishop; only worth it if you work heavily on structured prediction or variational inference theory
- **Bishop's original PRML** — superseded for most purposes by Bishop & Bishop (2024)
- **Raschka "Build an LLM from Scratch"** — implementation-focused, won't add foundational knowledge

---

## Suggested Reading Order

1. **Jurafsky & Martin** — biggest gap; needed to read LLM papers fluently
2. **Boyd & Vandenberghe** (ch. 1–5 + duality) — optimization depth
3. **Cover & Thomas** (ch. 1–6) — information theory foundations
4. The rest are reference texts, not cover-to-cover reads
