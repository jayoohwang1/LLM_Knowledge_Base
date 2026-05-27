# PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents (2026)

**Authors**: Zhuohan Gu, Qizheng Zhang, Omar Khattab, Samuel Madden
**arXiv**: [2605.19932](https://arxiv.org/abs/2605.19932)

---

## TL;DR

- **Problem**: LLM agents acting over **recurring long external contexts** (doc corpora, code repos) waste tokens/iterations re-scanning the same material across tasks.
- **Idea**: Maintain a **small, constant-sized "context map"** in the agent's prompt — a persistent *orientation cache* about the external context (not trajectory replay, not raw access).
- **Mechanism**: A programmable cache policy with 3 modules — **Distiller** (extract knowledge from inference signals), **Cartographer** (translate into structured edits), **Evictor** (priority eviction under fixed token budget).
- **Result**: +6.3–34.0% over strong baselines, 93–145 fewer iterations, **1.7–5.8x cheaper than ACE** on long-context reasoning/aggregation; +6–14% solving rate at 1.4x lower cost on context learning.

---

## Motivation

- **Long-context agents repeat work**: each new task on the same corpus re-discovers structure, file layout, key entities.
	- Trajectory-style memory (e.g. caching prior runs verbatim) is bulky and **task-specific** — doesn't transfer to new queries.
	- Pure RAG / raw access doesn't give the agent a **mental map**.
- **PEEK's framing**: separate two kinds of knowledge.
	- **Trajectory knowledge** — what *I* did → not the target.
	- **Orientation knowledge** — what the *world* looks like (structure, where things live, semantic links) → transferable across tasks.
- Analogy: like a **subway map** vs. a recording of past commutes — the map is small, stable, and reusable.

---

## Method

### Context Map (the artifact)

- **Small fixed-size text artifact** injected into the agent prompt every turn.
- Stores **structured summary** of the external context: document topology, section/file hierarchies, semantic links, key entities.
- **Not** the corpus itself; **not** trajectory logs. Acts as an *index* the agent consults before deciding what to actually retrieve.
- Updated **online** during inference — grows in fidelity as the agent encounters more of the corpus.

### Cache Policy (3 modules)

> Think of it as a write-back cache for orientation knowledge, with custom write/admit/evict policies driven by LLM-extracted signals.

**1. Distiller**
- Watches agent's **inference-time signals** (tool calls, retrieved chunks, intermediate reasoning).
- Extracts **transferable knowledge** — facts about the context that will be useful beyond the current task.
- Filters out task-specific noise.

**2. Cartographer**
- Takes distilled knowledge → produces **structured edits** to the context map.
- Operations like: add node, link sections, annotate region, update summary.
- Keeps the map's schema coherent rather than appending free-form text.

**3. Evictor**
- Enforces **fixed token budget** on the map.
- **Priority-based eviction** — drops low-utility entries when budget exceeded.
- Priorities derived from access frequency, recency, and structural centrality.

### Inference Loop

1. Agent receives task + current context map in prompt.
2. Uses map to **orient** — decides what to fetch from the long context.
3. Acts (retrieves, reasons, calls tools).
4. Distiller/Cartographer/Evictor update the map from this turn's signals.
5. Map persists across tasks → next task benefits from accumulated orientation.

---

## Experiments

### Tasks evaluated

- **Long-context reasoning** — questions requiring multi-hop traversal of large docs.
- **Information aggregation** — gather + synthesize spread-out facts.
- **Context learning** — solve programming/rubric-graded tasks over a repo.
- Tested with multiple LLMs incl. **OpenAI Codex** → method is model-agnostic.

### Baselines

- **ACE** (prompt-learning framework) — primary strong baseline.
- Naive full-context retention.
- Sliding-window context.
- Trajectory-style memory.

### Headline numbers

| Setting | PEEK vs. baselines |
|---|---|
| Long-context reasoning / aggregation | **+6.3–34.0%** accuracy |
| Iterations to solve | **93–145 fewer** |
| Cost vs. ACE | **1.7–5.8x lower** |
| Context learning — solving rate | **+6.0–14.0%** |
| Context learning — rubric accuracy | **+7.8–12.1%** |
| Context learning — cost vs. ACE | **1.4x lower** |

- Big efficiency wins come from **not re-scanning** — the map shortcuts navigation.

### Ablations

- Removing each of Distiller / Cartographer / Evictor degrades performance — all three contribute.
- Map size sensitivity: fixed budget is key; unbounded growth defeats the cache purpose.

---

## Why this is interesting

- **Reframes agent memory**: shifts focus from *trajectory replay* (most prior work) to *world-orientation*. Cleaner separation of concerns.
- **Programmable cache abstraction**: Distiller/Cartographer/Evictor mirror classical cache design (admit/write/evict) — bridges systems thinking and agent design (Madden is a systems/db person — shows).
- **Constant-size artifact in prompt** → predictable cost, plays well with prompt caching.
- Related to **[[Cartridges]]** (context compression via learned artifacts) and **context-compression** literature ([[LLM_context_compression_papers]]) — but PEEK's artifact is **textual, structured, and online-updated**, not learned weights.
- vs. **RAG**: complementary — PEEK tells the agent *where to look* before retrieval fires.
- vs. **ACE**: ACE rewrites task-specific prompts; PEEK accumulates **cross-task** orientation → bigger lift on recurring-corpus regimes.

---

## Limitations / open questions

- Fixed hierarchical map may struggle with **unstructured or dynamic** corpora.
- Map schema is presumably hand-designed — extending to new domains may need re-tuning.
- Unclear how well it scales to **multi-agent** settings or contexts that change mid-task.
- Distiller quality bounded by the underlying LLM's ability to extract transferable signal — could compound errors over time without correction.
