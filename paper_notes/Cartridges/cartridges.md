# Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study

- **Authors**: Eyuboglu, Ehrlich, Arora et al. (Stanford, Caltech, UBuffalo) — HazyResearch group
- **Core idea**: train a small KV cache offline on a corpus, load it at inference to replace full ICL
- **Setting**: many diverse queries over same large corpus (100k-1M tokens) — legal filings, codebases, medical records, textbooks

## Method

### Cartridge Parameterization
- **Cartridge** $Z \in \mathbb{R}^{L \times p \times d \times 2}$ = trainable key/value vectors across all layers, simplified prefix-tuning
	- $p$ controls size — equivalent memory footprint of $p$ tokens
	- $Z$ initialized from KV cache of first $p$ tokens of corpus $\mathcal{C}$ (not random — critical)
		- random vectors: 29.9% acc on LongHealth
		- random KV tokens: 51.3%
		- real corpus KV tokens: 55.3%
	- first token frozen (attention sink) — improves training stability
- **All model params frozen**, backprop only into $Z$'s keys and values
- **Prefix-tuning > LoRA** for this setting
	- LoRA accuracy drops precipitously as size increases (54.7 → 45.3 on MMLU)
	- prefix-tuning degrades gracefully (54.7 → 54.3)
	- prefix-tuning outperforms memory-matched LoRA by 4.5 chrF on MTOB

### Self-Study Training Recipe (two steps)

#### Step 1: Synthetic Data Generation (Alg 1)
- **Chunking**: sample random subcorpus $\hat{c}$ (512-4096 tokens) from $\mathcal{C}$
	- lets model focus on different parts of corpus
	- enables corpora longer than context window
- **Seed prompts**: sampled from 5 generic types — structuring, summarization, question, use cases, creative
	- generic across all corpora (no domain-specific prompts)
	- diversity matters: +4.8 acc on LongHealth, +7.9 chrF on MTOB vs single prompt
	- no benefit on QASPER (less reasoning-intensive queries)
- **Self-play conversation**: model talks to itself with $\hat{c}$ in system prompt
	- two participants A and B (same model)
	- A sees: subcorpus + seed prompt + history with roles flipped → generates "user" message
	- B sees: subcorpus + history in natural order → generates "assistant" message with **top-k logprobs** (top-20, 99% prob mass)
	- these logprobs become the teacher signal for distillation
	- conversations avoid training on same text repeatedly → prevents overfitting

#### Step 2: Context Distillation
- **Teacher**: $\mathcal{F}(\cdot|\hat{c})$ — model with subcorpus in context (logprobs collected during synthesis)
- **Student**: $\mathcal{F}_Z(\cdot)$ — model with cartridge, no corpus in context
- **Loss**: $\arg\min_Z \sum D_{KL}\big(\mathcal{F}(\cdot|\hat{c} \oplus \mathbf{x}[:t]) \;\|\; \mathcal{F}_Z(\cdot|\mathbf{x}[:t])\big)$
	- sparse approximation — only compute over top-k teacher tokens covering 99% mass
	- substantially better than next-token prediction (+8.6 chrF on MTOB)
	- next-token prediction memorizes corpus perfectly ($107\times$ less memory) but doesn't generalize to diverse queries

### Serving
- cartridge loads directly into KV cache slots in existing inference servers
- decoding identical to serving a cached prefix of length $p$
- compatible with existing KV cache management infrastructure (SGLang, vLLM)

### Composition
- independently trained cartridges can be concatenated at inference — no joint retraining
- tested on pairs of 10-K filings (AMD, Pepsi, AMEX, Boeing) across sizes {512, 1024, 2048, 4096}
- outperforms single-cartridge and truncated ICL on multi-document QA
- works because KV caches live in sequence space (concatenation is lossless), unlike LoRA weight merging

## Key Results
- **38.6x less memory**, **26.4x higher throughput** vs full ICL, matching quality
	- up to 10x memory savings on LongHealth at comparable performance
	- up to 100x on QASPER
- **Context length extrapolation**: 128k context model processes 484k token textbook via chunked self-study (MTOB)
	- outperforms ICL over first 130k tokens by 11.0 chrF points
- **Compression baselines degrade fast**: prompt truncation, summarization, DuoAttention all see quality drops at >2x compression
- **Scaling**: steady positive correlation between training compute and quality across all cartridge sizes
- **Training cost**: ~30 min on 8xH100 for Llama-8B — not cheap, only amortized over many queries

## Ablations Summary (Fig 6)
| Design choice | Key finding |
|---|---|
| **Parameterization** | Prefix-tuning > LoRA, especially on corpus-related queries |
| **Initialization** | Real token KVs >> random KVs >> random vectors |
| **Frozen tokens** | Freezing first token (attention sink) improves stability |
| **Objective** | Context distillation >> next-token prediction |
| **Seed prompts** | 5 diverse types >> 1 generic prompt |

## Code Implementation Mapping

```
cartridges/
├── cache.py              # TrainableCache — the cartridge (trainable KV nn.Parameters, frozen tokens, seq_id=-1)
├── train.py              # Training loop — CacheAndModel wrapper, KL loss over sparse teacher logprobs
├── synthesize.py         # Async batch orchestration for data generation
├── datasets.py           # Sequence packing, chat template tokenization, teacher logprob storage
├── generation.py         # flex_generate() — autoregressive decoding with cartridge as KV prefix
├── evaluate.py           # Perplexity and generation eval, ICL baselines
├── synthesizers/
│   └── self_study.py     # SelfStudySynthesizer — Alg 1: A/B role-flip convos, logprob collection
├── initialization/
│   ├── text.py           # KVFromText — forward pass on first p corpus tokens (default init)
│   ├── random.py         # KVFromRandomVectors (ablation)
│   └── pretrained.py     # KVFromPretrained — load saved cartridge
├── data/
│   ├── resources.py      # TextResource + 6 seed prompt generators (structuring/summarization/question/use_case/creative/generic)
│   └── chunkers.py       # TokenChunker — random 512-4096 token spans
└── models/
    └── llama/modeling_llama.py  # FlexLlamaForCausalLM — flex_attention with cartridge-aware mask (seq_id=-1 attended by all)
```

**Key implementation details**:
- **Attention mask**: cartridge tokens get `seq_id = -1`, mask function: `(kv_seq_id == -1) | (same_seq & causal)` — enables packed multi-sequence training where all seqs share the cartridge
- **Training mode**: `skip_append=True` so input KVs aren't cached (recomputed each step); generation mode caches them
- **Loss**: `ce = -p(x).exp() * log_softmax(logits)[target_idx]` — sparse KL over top-20 teacher tokens
- **Packing**: multiple conversations packed to fixed `seq_length` (default 2048) with distinct `element_ids` to avoid flex_attention recompiles

## Commentary

### What works well
- **Self-study is the real contribution** — prefix-tuning is known; the insight that model self-quizzing produces general cartridges while raw corpus training memorizes is the key finding
- **Self-contained pipeline** — same model is teacher and student, no separate distillation model
- **Composition is undersold** — concatenating independent cartridges for multi-doc QA without joint training is a strong result with practical implications (load codebase + docs cartridges together)
- **Seed prompt diversity** — large gains from a tiny intervention, suggests training signal diversity is the bottleneck

### Limitations
- **30 min training cost** not cheap — only makes sense in "many queries over static corpus" regime; RAG handles dynamic corpora trivially
- **All evaluations are single-document** — LongHealth, MTOB, QASPER each one big doc; unclear how well this scales to many heterogeneous docs with cross-references
- **Memory comparison somewhat apples-to-oranges** — 38.6x headline doesn't amortize training compute
- **Structural reasoning unclear** — claims structural awareness but evidence is mostly content QA, not "compare section 3 vs section 7" type reasoning across distant parts

### Connections to Broader Research
- **Episodic vs semantic memory** (cog sci): ICL = raw episodic recall, cartridge = consolidated semantic representation; self-study mirrors "retrieval practice" / testing effect from learning science
- **Information bottleneck**: cartridge is explicitly a bottleneck; next-token prediction preserves too much surface form, context distillation shapes what information passes through
- **Learned index structures**: cartridge as learned database index over a document — replaces full KV "B-tree" with trained approximation
- **Adapter composition**: avoids LoRA weight merging interference because KV caches compose in sequence space (lossless concatenation) rather than parameter space
- **Continual learning**: naive NTP failure = overfitting to corpus distribution / forgetting how to be general; self-study = replay-based anti-forgetting
- **Compilation metaphor**: ICL = interpretation, cartridge training = compilation, serving = execution, composition = linking
	- naturally frames open questions: incremental compilation (corpus updates), optimization levels (training time vs quality), decompilation (privacy), type system for compatibility
- **Associative memory / Hopfield networks**: injecting KV pairs = programming attention attractors; real-token init starts near valid attractors


### Directions
Incremental updates. Train a cartridge once, then when the corpus changes (new commits, updated docs), do a 
  cheap delta update instead of retraining from scratch. Could look like continued training on synthetic data
  generated only from the diff, or some form of online/streaming prefix-tuning.                               
                                                                                                            
  Hierarchical cartridges. Instead of one flat cartridge per corpus, train a tree — coarse summary cartridge  
  at the root, progressively more detailed ones at leaves. Route queries to the right level of detail. This
  could handle the "many heterogeneous documents" gap in the current work.                                    
                                                                                                            
  Cartridge retrieval / routing. Maintain a library of cartridges and learn a lightweight router that selects 
  which ones to compose for a given query. Like RAG but over cartridges instead of text chunks. The
  composition result suggests this is feasible — the missing piece is the selection mechanism.                
                                                                                                            
  Cross-model cartridge transfer. Can you train a cartridge on a small model and transfer it to a larger one  
  (or vice versa)? KV spaces differ across models, but there might be a learned projection that maps between
  them. Would drastically reduce training costs if the cheap model's cartridge transfers.                     
                                                                                                            
  Adversarial / privacy analysis. How much of the original corpus can you recover from a cartridge? The       
  compilation metaphor suggests a "decompilation" attack. If cartridges are meant for sensitive data (medical
  records, legal docs), this matters. Conversely, could you train cartridges that are differentially private? 
                                                                                                            
  Adaptive cartridge size. Currently $p$ is a fixed hyperparameter. Could you learn which parts of the corpus 
  need more representation and allocate tokens non-uniformly? Maybe via some importance scoring during
  self-study, or pruning after training.                                                                      
                                                                                                            
  Self-study for other parameterizations. The self-study recipe (synthetic conversations + context            
  distillation) is independent of the KV cache parameterization. Could apply it to LoRA fine-tuning, soft
  prompts, or even full fine-tuning to improve generalization — the recipe might be more transferable than the
   specific cartridge format.                                                                               
                                                                                                            
  Agentic cartridges. The codebase already has tool integration in the synthesizer. Push this further — let   
  the model use retrieval tools during self-study to decide what information to absorb, effectively learning
  an active reading strategy. The cartridge becomes a summary of what the agent found useful.                 
                                                                                                            
  Multi-modal cartridges. Same idea but for vision-language models with image-heavy documents (charts,        
  diagrams, scanned PDFs). The KV cache representation is modality-agnostic in principle — the question is
  whether self-study works when the "corpus" includes visual tokens.                                          
                                                                                                            
  Cartridges as long-term memory for agents. Instead of ever-growing conversation histories, periodically     
  compile recent interactions into a cartridge. Stack them chronologically. This is the "long-term memory in
  chatbots" application they mention in passing but don't explore.