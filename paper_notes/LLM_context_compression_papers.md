# LLM Context Compression — Paper Collection

> Compiled April 2026. Covers prompt-level compression, KV cache compression, and soft/learned compression methods.

---

## Taxonomy

Context compression methods fall into three broad families:

1. **Hard prompt compression** — Remove or rephrase tokens in the text prompt before sending to the LLM. The prompt remains natural language.
2. **Soft prompt / learned compression** — Train a model to compress context into dense "memory" embeddings or special tokens that the LLM conditions on. The compressed representation is no longer human-readable.
3. **KV cache compression** — Operate on the key-value cache during inference, evicting, quantizing, or merging KV entries to reduce memory and compute without changing the input text.

---

## 1. Hard Prompt Compression

### LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
- **Authors:** Jiang et al. (Microsoft Research)
- **Venue:** EMNLP 2023
- **Link:** https://arxiv.org/abs/2310.05736
- **Method:** Uses a small LM (e.g., GPT-2, LLaMA-7B) to compute per-token perplexity, then drops low-information tokens. Budget controller allocates compression ratio across prompt segments (instruction, demonstrations, question). Iterative token-level compression.
- **Results:** Up to 20x compression with minimal performance loss on reasoning and in-context learning benchmarks.
- **Notes:**
  - *Related: information theory (entropy/perplexity).* Token-level perplexity as a saliency signal connects to the information content h(x) = -log p(x) from Shannon's framework — tokens with low perplexity carry less "surprise" and are candidates for removal.
  - *Related: the "lost in the middle" problem.* LLMLingua helps mitigate the finding (Liu et al., 2023) that LLMs underweight information in the middle of long contexts, by concentrating the remaining tokens on the most informative parts.
  - *Limitation:* Compression decisions are made by a smaller model — its saliency judgments may diverge from what the target LLM actually attends to.

### LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression
- **Authors:** Jiang et al. (Microsoft Research)
- **Venue:** ACL 2024
- **Link:** https://aclanthology.org/2024.acl-long.91/
- **Method:** Extends LLMLingua to long-context scenarios. Uses question-aware coarse-to-fine compression: first ranks documents/segments by relevance (conditioned on the question), then applies fine-grained token-level pruning within selected segments. Also introduces a document reordering strategy and a subsequence recovery algorithm for faithful generation.
- **Results:** Up to 21.4% performance improvement with 4x fewer tokens on long-context benchmarks.
- **Notes:**
  - *Related: RAG pipelines.* Directly applicable as a post-retrieval compression step — after retrieving K documents, compress them before feeding to the LLM. Reduces cost without re-retrieving.
  - *Related: attention sink phenomenon.* The document reordering strategy accounts for position-dependent attention biases in LLMs.

### LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression
- **Authors:** Pan et al. (Microsoft Research)
- **Venue:** ACL 2024 Findings
- **Link:** https://arxiv.org/abs/2403.12968
- **Method:** Reformulates prompt compression as a token classification problem (keep/drop). Distills compression decisions from GPT-4 into a small model (XLM-RoBERTa or mBERT) that performs binary classification per token. Task-agnostic — no question conditioning needed at compression time.
- **Results:** 3-6x faster compression than LLMLingua, 1.6-2.9x end-to-end speedup at 2-5x compression ratios. Better out-of-domain generalization.
- **Notes:**
  - *Related: knowledge distillation.* The data distillation from GPT-4 to a small classifier is a form of KD where the teacher's compression behavior (which tokens it considers important) is transferred to the student.
  - *Related: token pruning in vision transformers.* The keep/drop binary classification is analogous to token pruning methods in ViTs (e.g., DynamicViT), where a lightweight predictor decides which image patches to drop.

### Context-Aware Prompt Compression (CPC)
- **Authors:** Byun et al.
- **Venue:** Preprint 2024
- **Link:** https://arxiv.org/abs/2409.01227
- **Method:** Sentence-level compression. Embeds each sentence and the query, removes sentences with low relevance to the query. Operates at a coarser granularity than token-level methods, which makes it faster.
- **Results:** Outperforms LongLLMLingua on both performance and latency.
- **Notes:**
  - *Related: retrieval and re-ranking.* CPC is essentially a re-ranking step applied to sentences within the prompt rather than to documents in a retrieval corpus. The same embedding models (e.g., E5, BGE) can serve both purposes.

### Prompt Compression in the Wild
- **Authors:** (2026 preprint)
- **Link:** https://arxiv.org/abs/2604.02985
- **Method:** Empirical study testing prompt compression across three providers (GPT-4o-mini, Claude-3.5-Sonnet, DeepSeek-Chat) on five benchmarks. Measures latency, rate adherence (does the compressed prompt actually hit the target compression ratio?), and quality.
- **Notes:**
  - *Related: practical deployment.* Important for understanding how compression interacts with different model providers' tokenizers and architectures. Compression ratios measured in characters may differ from ratios in tokens across providers.

---

## 2. Soft Prompt / Learned Compression

### Gist Tokens: Learning to Compress Prompts with Gist Tokens
- **Authors:** Mu et al. (Stanford)
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2304.08467
- **Method:** Fine-tunes an LLM to compress an arbitrary prompt into a small number of "gist tokens" — special virtual tokens whose activations summarize the prompt. Uses a modified attention mask during training: the gist tokens attend to the prompt, but subsequent generation tokens only attend to the gist tokens (not the original prompt). The gist token KV cache can be precomputed and stored.
- **Results:** Up to 26x compression, ~40% FLOPs reduction, 4.2% wall-time speedup, minimal quality loss.
- **Notes:**
  - *Related: prefix tuning and prompt tuning.* Gist tokens are conceptually similar to learned soft prompts (Lester et al., 2021; Li & Liang, 2021), but here the soft prompt is generated dynamically from the input context rather than learned as fixed parameters for a task.
  - *Related: KV cache reuse.* The gist token KV entries can be cached and reused across different queries sharing the same system prompt or context — directly applicable to reducing cost for system prompts that remain constant across API calls.
  - *Related: information bottleneck.* Compressing a long context through a small number of gist tokens is a concrete instantiation of the information bottleneck principle: maximize I(gist; output) while minimizing I(gist; input).

### In-Context Autoencoder (ICAE) for Context Compression in a Large Language Model
- **Authors:** Ge et al.
- **Venue:** ICLR 2024
- **Link:** https://arxiv.org/abs/2307.06945
- **Method:** Autoencoder architecture where the encoder (LLM + LoRA adapters) compresses context into a fixed number of "memory slots," and the decoder (original LLM, frozen) conditions on these memory slots for generation. Two-stage training: (1) pretrain with autoencoding + language modeling objectives on unlabeled text, (2) fine-tune on instruction data.
- **Results:** 4x compression with ~1% additional parameters. 2048 memory slots perform comparably to 4096 context tokens, saving ~20GB GPU memory.
- **Notes:**
  - *Related: autoencoders and the ELBO.* The autoencoding pretraining objective connects directly to the VAE framework (DLFC Ch 19) — the memory slots are a learned latent representation, and the reconstruction loss is the autoencoding term. The key difference: ICAE uses deterministic encoding rather than stochastic sampling.
  - *Related: LoRA and parameter-efficient fine-tuning.* Only the encoder is adapted (via LoRA), keeping the decoder frozen. This means any model that can use the base LLM can also use ICAE's compressed representations without further adaptation.
  - *Related: memory-augmented transformers.* ICAE's memory slots are related to Memorizing Transformers (Wu et al., 2022) and memory tokens in Transformer-XL, but here the memory is explicitly compressed from context rather than accumulated over time.

### 500xCompressor: Generalized Prompt Compression for Large Language Models
- **Authors:** Li et al.
- **Venue:** ACL 2025
- **Link:** https://arxiv.org/abs/2408.03094
- **Method:** Compresses natural language context into as few as one special token. Adds ~0.3% parameters. Key insight: storing compressed information as KV cache entries outperforms storing as embeddings at high compression ratios. Pretrained on ArxivCorpus, fine-tuned on ArxivQA, evaluated on unseen cross-domain QA tasks.
- **Results:** 6x–500x compression. Retains 70–74% F1, 77–84% EM. 27–90% computation reduction, 55–83% memory savings (generating 100–400 tokens).
- **Notes:**
  - *Related: extreme compression and rate-distortion theory.* At 500x compression, this pushes toward the rate-distortion limit — how much information can you preserve per bit of compressed representation? The fact that KV cache format outperforms embeddings at high ratios suggests that the key-value structure provides a better inductive bias for representing context than a flat embedding.
  - *Related: cross-domain generalization.* Trained on arxiv, tested on other domains — demonstrates that compression can learn domain-agnostic representations, similar to how pretraining learns transferable features.

### Activation Beacon: Long Context Compression with Activation Beacon
- **Authors:** Zhang et al.
- **Venue:** Preprint 2024 (updated for COLM)
- **Link:** https://arxiv.org/abs/2401.03462
- **Method:** Plug-in module that compresses long-range activations into compact "beacon" tokens at configurable compression ratios (2x, 4x, 8x). Processes context in sliding windows; at the end of each window, compresses the KV states into beacon tokens. Supports progressive compression for very long contexts (up to 400K tokens).
- **Results:** 2x inference speedup, 8x KV cache memory reduction.
- **Notes:**
  - *Related: sliding window attention.* Activation Beacon extends the sliding window idea (used in Mistral's architecture) by adding a compression step at each window boundary, effectively giving the model access to compressed history beyond the window.
  - *Related: recurrent/state-space models.* The progressive compression of past context into fixed-size representations is functionally similar to what recurrent models and SSMs (Mamba) do — maintain a finite-size state that summarizes the past. The difference is that Activation Beacon does this post-hoc on a pretrained transformer rather than requiring a recurrent architecture.

### In-Context Former: Lightning-fast Compressing Context for Large Language Model
- **Authors:** Huang et al.
- **Venue:** Preprint 2024
- **Link:** https://arxiv.org/abs/2406.13618
- **Method:** Faster variant of the ICAE approach, optimized for speed of the compression step itself.
- **Notes:**
  - *Related: amortized inference.* The speed improvement comes from amortizing the compression cost, relevant when the same compressed context is reused across multiple downstream queries.

---

## 3. KV Cache Compression

### StreamingLLM: Efficient Streaming Language Models with Attention Sinks
- **Authors:** Xiao et al. (MIT, Meta)
- **Venue:** ICLR 2024
- **Link:** https://arxiv.org/abs/2309.17453
- **Method:** Discovers the "attention sink" phenomenon — initial tokens receive disproportionately high attention regardless of content. Maintains a small set of sink tokens (first ~4 tokens) plus a sliding window of recent tokens. Enables streaming inference on arbitrarily long sequences without fine-tuning.
- **Results:** Stable perplexity on sequences up to 4M tokens with fixed KV cache budget.
- **Notes:**
  - *Related: softmax saturation and attention distribution.* The attention sink phenomenon is connected to the softmax function's behavior: when no key is particularly relevant, attention mass "collects" on a default token (usually the first). This is related to the observation that LLMs often place high attention on delimiter/BOS tokens.
  - *Related: positional encoding.* StreamingLLM requires re-indexing positions in the sliding window, which interacts non-trivially with RoPE and other positional encodings. Understanding positional encoding schemes (DLFC 12.1.9) is essential for understanding why naive sliding windows fail without the sink tokens.
  - *Limitation:* The evicted middle tokens are permanently lost — the model cannot recall information from outside the window. This is a hard tradeoff vs. compression methods that retain a lossy summary.

### SnapKV: LLM Knows What You are Looking for Before Generation
- **Authors:** Li et al.
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2404.14469
- **Method:** Observes that attention patterns in the prompt phase predict which KV entries will be important during generation. Clusters attention features at each layer and selects a subset of KV entries per head based on attention scores from a small observation window at the end of the prompt. Applied per-layer and per-head.
- **Results:** Maintains performance at 3-6x KV cache compression on long-context benchmarks.
- **Notes:**
  - *Related: attention head specialization.* SnapKV operates per-head, reflecting the finding that different heads attend to different types of information (positional, syntactic, semantic). This connects to multi-head attention analysis (DLFC 12.1.6).
  - *Related: structured pruning.* Selecting which KV entries to keep is analogous to structured pruning of neurons/channels in CNNs, but applied dynamically at inference time rather than statically.

### PyramidKV: Dynamic KV Cache Compression Based on Pyramidal Information Funneling
- **Authors:** Cai et al.
- **Venue:** Preprint 2024
- **Link:** https://arxiv.org/abs/2406.02069
- **Method:** Observes that attention becomes increasingly sparse in deeper layers. Allocates more KV cache budget to lower layers (which need more entries to represent broad attention) and less to upper layers (which attend more selectively). Creates a "pyramid" shaped cache budget allocation.
- **Results:** Maintains performance with only 12% KV cache, outperforms SnapKV, H2O, and StreamingLLM at all tested cache sizes.
- **Notes:**
  - *Related: layer-wise representation analysis.* The pyramidal budget reflects a fundamental property of transformer representations: early layers capture low-level, distributed features (requiring more keys to represent), while deep layers capture high-level, sparse features. This connects to the information bottleneck view of deep networks.
  - *Related: mixture-of-depths.* The idea that different layers need different computational budgets resonates with Mixture-of-Depths (Raposo et al., 2024), where different tokens skip different layers.

### KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
- **Authors:** Hooper et al. (UC Berkeley)
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2401.18079
- **Method:** Quantizes KV cache entries to low precision (down to 3-bit). Uses non-uniform quantization (NUQ) that accounts for sensitivity (not just magnitude). Handles outlier channels via per-channel quantization and dense-and-sparse quantization.
- **Results:** <0.1 perplexity degradation at 3-bit quantization, 4.8x memory reduction. Enables 10M context length on a single A100-80GB GPU with LLaMA-7B.
- **Notes:**
  - *Related: weight quantization.* The techniques (NUQ, outlier handling) build on the weight quantization literature (GPTQ, AWQ, SqueezeLLM). The key difference: KV cache values have different distributional properties than weights — they are input-dependent and have heavier-tailed distributions.
  - *Related: mixed-precision training.* Running the model in fp16/bf16 while keeping the cache in int3/int4 is a form of mixed-precision inference. Understanding numerical precision and quantization error is essential.

### CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving
- **Authors:** Liu et al.
- **Venue:** ACM SIGCOMM 2024
- **Link:** https://dl.acm.org/doi/10.1145/3651890.3672274
- **Method:** Compresses KV cache for network transmission in distributed/cloud LLM serving. Treats KV cache tensors as data to be compressed with a learned codec, streamed from storage to the GPU.
- **Results:** 3.5–4.3x KV cache size reduction, 3.2–3.7x total delay reduction.
- **Notes:**
  - *Related: systems-level optimization.* CacheGen targets a different bottleneck than most compression papers — network bandwidth rather than GPU memory. Relevant for disaggregated serving architectures (e.g., Mooncake, DistServe) where prefill and decode may happen on different machines.

### MiniCache: KV Cache Compression in Depth Dimension for Large Language Models
- **Authors:** Liu et al.
- **Venue:** NeurIPS 2024
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd0705710bf01b88a60a3d479ea341d9-Abstract-Conference.html
- **Method:** Exploits high similarity of KV cache states between adjacent layers in the middle-to-deep portion of LLMs. Merges KV caches across layers by sharing entries between similar layers.
- **Notes:**
  - *Related: weight sharing and parameter tying.* Cross-layer KV sharing is analogous to parameter sharing across transformer layers (as in Universal Transformers or ALBERT). The observation that adjacent layers have similar KV representations suggests significant redundancy in the depth dimension.

### ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference
- **Authors:** (2025)
- **Link:** https://openreview.net/forum?id=20JDhbJqn3
- **Method:** Treats semantic chunks (phrases, clauses) rather than individual tokens as the basic compression unit. Preserves complete linguistic structures under aggressive compression.
- **Results:** Up to 8.7% precision improvement over token-level methods at the same compression ratio.
- **Notes:**
  - *Related: chunking in RAG.* The chunk-level granularity mirrors the chunking decisions made in RAG pipelines — both face the question of what constitutes a semantically coherent unit.

### DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Attention Sinks and Tokens
- **Authors:** (2025)
- **Venue:** ACL 2025 Findings
- **Link:** https://aclanthology.org/2025.findings-acl.1055.pdf
- **Method:** Dynamically allocates attention sink tokens and cache budget based on context, rather than using a fixed number of sinks (as in StreamingLLM).
- **Notes:**
  - *Related: adaptive computation.* Dynamic allocation of cache budget per-input is a form of adaptive computation, related to early-exit methods and mixture-of-experts.

---

## 4. Surveys

### Prompt Compression for Large Language Models: A Survey
- **Authors:** Li et al.
- **Venue:** NAACL 2025 (Main, Selected Oral)
- **Link:** https://arxiv.org/abs/2410.12388
- **GitHub:** https://github.com/ZongqianLi/Prompt-Compression-Survey
- **Coverage:** Systematic taxonomy of hard vs. soft prompt compression. Covers both token-level and sentence-level hard methods, and embedding-based vs. KV-based soft methods.

### KV Cache Compression for Inference Efficiency in LLMs: A Review
- **Authors:** (2025)
- **Link:** https://arxiv.org/pdf/2508.06297
- **Coverage:** Analyzes three primary strategies: selective compression (eviction), quantization, and attention compression.

### Awesome-KV-Cache-Compression (GitHub)
- **Link:** https://github.com/October2001/Awesome-KV-Cache-Compression
- Continuously updated paper list.

### Awesome-LLM-Compression (GitHub)
- **Link:** https://github.com/HuangOwen/Awesome-LLM-Compression
- Broader scope: covers weight quantization, pruning, distillation, and KV cache compression.

---

## Key Themes and Connections

### Compression vs. Retrieval
Prompt compression and RAG are complementary strategies for the same problem (context window budget). Compression shrinks what's already in the context; retrieval selects what goes in. In practice they compose: retrieve K documents, then compress them before feeding to the LLM.

### The Rate-Distortion Tradeoff
All compression methods face a fundamental rate-distortion tradeoff: higher compression (lower rate) means more information loss (higher distortion). The KL divergence (DLFC 2.5.5) between the original and compressed representations quantifies this loss. Methods differ in where they sit on this curve and what distortion metric they implicitly optimize.

### Attention Sparsity as Compressibility Signal
A recurring finding across KV cache papers (SnapKV, PyramidKV, H2O) is that attention patterns are sparse — most generation steps attend to a small subset of context tokens. This sparsity is what makes compression possible without catastrophic quality loss. It connects to the broader observation that transformer attention is low-rank (Treviso et al., 2022).

### Soft Compression and the Information Bottleneck
Gist tokens, ICAE memory slots, and Activation Beacon tokens can all be understood through the information bottleneck framework: the compressed representation Z should maximize I(Z; Y) (mutual information with the output) while minimizing I(Z; X) (information about the input), where the balance is controlled by the compression ratio.

### Connection to Recurrence and State Space Models
Soft compression methods that produce fixed-size context summaries are functionally equivalent to recurrent state updates. Activation Beacon's progressive compression is especially close to SSMs like Mamba — both maintain a fixed-size state that summarizes the history. The difference is architectural: Activation Beacon retrofits compression onto a pretrained transformer, while SSMs build recurrence into the architecture from scratch.

### Practical Deployment Considerations
- **Hard compression** (LLMLingua) works with any LLM API — no model modification needed.
- **Soft compression** (Gist, ICAE) requires model fine-tuning or adapter training.
- **KV cache compression** (SnapKV, KVQuant) requires inference framework modifications but no model retraining.
- Cost considerations depend on whether you're bottlenecked by compute (FLOPs), memory (KV cache size), latency (time-to-first-token), or API cost (input token pricing).
