# Chapter 15: Neural Networks for Sequences

**Meta**: This is THE chapter for modern LLM research. Most of the chapter (15.1-15.3) is now mostly historical interest, but the **Attention (15.4) + Transformers (15.5) + LMs/Pretraining (15.7)** sections are the foundation of basically everything frontier labs do today. The book is from 2022/2025 update, so it's missing a *lot* of the post-2020 stuff that actually matters — covered at the end.

---

## 15.1 Introduction

- **Setup**: input is a sequence, output is a sequence, or both. Tasks: MT, ASR, classification, captioning.
- The book frames everything probabilistically: model $p(y_{1:T} | x)$ or $p(x_{1:T})$.

---

## 15.2 Recurrent Neural Networks (RNNs)

**Modern relevance**: Mostly historical. RNNs/LSTMs were the workhorse of seq modeling 2014-2017 before transformers killed them. Still worth understanding because (1) **state space models (Mamba, S4, RWKV, Linear Attention)** are essentially re-inventions of RNNs with structured state updates, and (2) interview/research culture still references them. But you will not use a vanilla RNN in production.

### 15.2.1 Vec2Seq (sequence generation)
- **Form**: $f_\theta : \mathbb{R}^D \to \mathbb{R}^{N_\infty C}$. Map a fixed vector to a variable-length sequence.
- **Generative model**: $p(y_{1:T}|x) = \prod_{t=1}^T p(y_t|h_t) p(h_t|h_{t-1}, y_{t-1}, x)$.
- **Output**: $p(y_t|h_t) = \text{Cat}(y_t | \text{softmax}(W_{hy} h_t + b_y))$ for discrete tokens.
- **Hidden state update**: $h_t = \varphi(W_{xh}[x; y_{t-1}] + W_{hh} h_{t-1} + b_h)$.
- **Why it matters historically**: First neural model that's Turing-complete in principle (unbounded latent state). Karpathy's "Unreasonable Effectiveness of RNNs" (2015) was a cultural moment.
- **Applications**: Language modeling (unconditional, $x = \emptyset$), image captioning (CNN encodes image $\to$ RNN decoder), pen stroke generation.

### 15.2.2 Seq2Vec (sequence classification)
- **Form**: $f_\theta: \mathbb{R}^{TD} \to \mathbb{R}^C$.
- **Simple**: classifier on final hidden $h_T$.
- **Bidirectional RNN (biRNN)**: $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$ — forward and backward RNNs concatenated.
  - **Modern parallel**: BERT's bidirectional context is the spiritual successor.
  - Average-pool over $h_t$'s to get final rep.

### 15.2.3 Seq2Seq
- **Aligned**: $T = T'$, dense sequence labeling (POS tagging, NER pre-BERT).
- **Unaligned (encoder-decoder)**: encode $x_{1:T} \to c = f_e(x_{1:T})$, then decode $y_{1:T'} = f_d(c)$.
  - **The bottleneck**: stuffing all of $x$ into one vector $c$ kills long-sequence MT performance.
  - This bottleneck is exactly what motivated **attention** — single most important architectural move in modern ML.

### 15.2.4 Teacher Forcing
- Train with ground-truth $w_{t-1}$ as input instead of model's own prediction.
- **Exposure bias problem**: model never sees its own mistakes at train time, so test-time errors compound.
- **Scheduled sampling**: mix in model samples increasingly over training.
- **Modern note**: Transformer LMs still use teacher forcing (next-token prediction on ground truth). Exposure bias is real but mostly addressed by (a) massive scale making mistakes rare and (b) **RLHF/DPO** which optimizes on-policy.

### 15.2.5 Backpropagation Through Time (BPTT)
- Unroll the RNN graph and backprop.
- **Cost**: $O(T^2)$ per step (gradient depends on all previous hidden states).
- **Truncated BPTT**: chop to last $K$ steps, carry hidden state across minibatches.
- $\frac{\partial h_t}{\partial w_h} = \frac{\partial f}{\partial w_h} + \sum_{i=1}^{t-1} \left( \prod_{j=i+1}^t \frac{\partial f_j}{\partial h_{j-1}} \right) \frac{\partial f_i}{\partial w_h}$.
- **Modern note**: This product-of-Jacobians is exactly why **transformers are easier** — no recurrent gradient path of length $T$.

### 15.2.6 Vanishing & Exploding Gradients
- Repeated multiplication by $W_{hh}$ Jacobians $\to$ activations + gradients explode or decay.
- **Fixes**:
  - **Gradient clipping** (still standard in transformer training; clip global norm to ~1).
  - Control spectral radius $\lambda$ of $W_{hh}$.
  - **Echo state networks / liquid state machines / reservoir computing**: don't learn $W_{hh}$, just $W_{ho}$. Convex problem. Curiosity, not used.
  - Orthogonal init.
  - **Additive updates** (residual connections) — the actual modern answer (LSTM/GRU, ResNet, transformer).

### 15.2.7 Gating and Long-Term Memory
- **Key idea**: replace multiplicative update with **additive** update gated by sigmoids.

#### 15.2.7.1 GRU (Gated Recurrent Unit)
- **Reset gate**: $R_t = \sigma(X_t W_{xr} + H_{t-1} W_{hr} + b_r)$.
- **Update gate**: $Z_t = \sigma(X_t W_{xz} + H_{t-1} W_{hz} + b_z)$.
- **Candidate**: $\tilde{H}_t = \tanh(X_t W_{xh} + (R_t \odot H_{t-1}) W_{hh} + b_h)$.
- **New state**: $H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H}_t$.
- If $Z_t = 1$: pass through unchanged (long-term memory).

#### 15.2.7.2 LSTM (Long Short-Term Memory)
- Adds an explicit **memory cell** $c_t$ separate from hidden $h_t$.
- **Gates**: output $O_t$, input $I_t$, forget $F_t$ — all sigmoids of $X_t, H_{t-1}$.
- **Cell update**: $C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$ (additive, no vanishing).
- **Hidden output**: $H_t = O_t \odot \tanh(C_t)$.
- **Trick**: init forget gate bias to large value so info passes through cell easily.
- **Modern relevance**: Architectures like **Mamba/SSMs** explicitly resurrect the cell-state-as-memory pattern but with linear-time scans. The "selective" in Mamba ≈ input-dependent gating.

### 15.2.8 Beam Search
- **Greedy decoding**: $\hat{y}_t = \arg\max_y p(y | \hat{y}_{1:t-1}, x)$. Suboptimal globally.
- **Beam search**: keep top $K$ partial sequences, expand each, keep top $K$ overall.
- Time: $O(KV)$ per step, much faster than exact $O(V^T)$ Viterbi.
- **Stochastic beam search**: Gumbel noise on partial scores → sample top-$K$ without replacement.
- **Diverse beam search**: penalize beams that look like each other.
- **Modern relevance**: Beam search still used in MT, summarization, and ASR. For chatbot-style LLMs, **temperature sampling + top-p (nucleus) + top-k** dominate; beam is known to produce bland, repetitive text in open-ended generation (the "likelihood trap"). The book doesn't cover nucleus sampling, min-p, typical sampling, or modern decoding tricks like **speculative decoding** / **medusa** which are huge for inference cost.

---

## 15.3 1D CNNs (sequence convolutions)

**Modern relevance**: Mostly displaced by transformers for text. But causal/dilated 1D convs live on in (a) **WaveNet-style** audio gen, (b) **convolutional position embeddings** in conformers/hybrid models, (c) **Mamba/SSM hybrids** that use local conv layers.

### 15.3.1 1D CNNs for sequence classification
- Convolve filter $w_d \in \mathbb{R}^k$ over sequence: $z_i = \sum_d x_{i-k:i+k,d}^T w_d$.
- Multiple kernel widths capture different scales (TextCNN [Kim14]).
- Max-pool over time $\to$ fixed vector $\to$ classifier.

### 15.3.2 Causal 1D CNNs for generation
- **Causal/masked conv**: $y_t$ only depends on $y_{t-k:t-1}$.
- Like an RNN but parallel and finite memory.
- **WaveNet** [Oord+16]: stacked dilated causal convs with dilation rates $1, 2, 4, ..., 512$ → receptive field 1024. SOTA TTS in 2016, distilled to **Parallel WaveNet** for production.
- **Modern parallel**: **Mamba/S4** uses a similar "long-range with structured state" mindset but with linear scan instead of dilated conv.

---

## 15.4 Attention

**This is THE section.** Attention is the single most important architectural innovation in modern deep learning. Everything in LLMs is downstream of this.

### 15.4 Setup
- Standard NN layer: $Z = \varphi(XW)$ with fixed $W$.
- **Attention**: $Z = \varphi(VW(Q,K))$ — weights are computed from the inputs themselves. **Multiplicative interaction**.
- Three projections: queries $Q = W_q X$, keys $K = W_k X$, values $V = W_v X$.

### 15.4.1 Attention as Soft Dictionary Lookup
- $\text{Attn}(q, (k_{1:m}, v_{1:m})) = \sum_{i=1}^m \alpha_i(q, k_{1:m}) v_i$
- Weights via softmax: $\alpha_i = \text{softmax}_i([a(q,k_1), ..., a(q,k_m)]) = \frac{\exp(a(q,k_i))}{\sum_j \exp(a(q,k_j))}$.
- **Differentiable soft lookup**: this is the conceptual core. Hard lookup (dict) → soft weighted avg.
- **Masked attention**: set scores to $-10^6$ for invalid positions (padding, future tokens for causal LM). Critical detail for batched training and decoder-only models.

### 15.4.2 Kernel Regression as Non-Parametric Attention
- Nadaraya-Watson: $f(x) = \sum_i \alpha_i(x, x_{1:n}) y_i$ with Gaussian kernel $\alpha_i \propto \exp(-\beta^2 (x - x_i)^2 / 2)$.
- **Pedagogically nice connection** but not central to modern practice. The kernel-attention view does motivate **Performer/Linear Attention** (see 15.6.4).
- **Frontier connection**: kernel-attention view → **Linear Attention** (Katharopoulos+ 2020) and **Performer** (Choromanski+ 2020) → spiritually leads to **Mamba/RWKV** which are linear-time attention variants.

### 15.4.3 Parametric Attention
- **Additive attention** (Bahdanau): $a(q,k) = w_v^T \tanh(W_q q + W_k k)$. MLP scoring.
- **Scaled dot-product attention** (the one that matters):
  $$a(q, k) = \frac{q^T k}{\sqrt{d}}$$
  - $\sqrt{d}$ scaling: keeps variance of $q^T k$ ≈ 1 regardless of dim (assuming q,k zero-mean unit-var → $q^T k$ has variance $d$).
  - Without it, softmax saturates → gradients vanish for large $d$. **This is the detail you must know.**
- **Batched form**: $\text{Attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V$, $Q \in \mathbb{R}^{n \times d}$, $K \in \mathbb{R}^{m \times d}$, $V \in \mathbb{R}^{m \times v}$.

### 15.4.4 Seq2Seq with Attention
- The original (Bahdanau 2014, Luong 2015): RNN encoder produces $h^e_{1:T}$, decoder hidden $h^d_{t-1}$ is the query, encoder states are keys/values.
- Dynamic context: $c_t = \sum_i \alpha_i(h^d_{t-1}, h^e_{1:T}) h^e_i$.
- **Why huge**: fixed-vector bottleneck eliminated; explicit alignment learned (visible as attention heatmaps); SOTA NMT in 2015.
- **Modern relevance**: This is the seed crystal of the entire transformer revolution. The "Attention Is All You Need" paper (2017) realized you can drop the RNN entirely.

### 15.4.5-15.4.6 Attention for classification and pair classification
- **Seq2Vec with attention**: attention pool over hidden states. Used in EHR mortality prediction.
- **Decomposable attention for NLI** (Parikh 2016, SNLI): premise A attends to hypothesis B and vice versa; cross-attention before transformers were a thing. Conceptually the same as transformer **cross-attention** in encoder-decoder models.

### 15.4.7 Soft vs Hard Attention
- **Soft**: weighted average, differentiable. Default.
- **Hard**: pick one location, sample with REINFORCE. Less popular because non-diff.
- **Interpretability caveat**: attention weights ≠ explanation. The "attention is not explanation" debate (Jain & Wallace 2019). Frontier labs largely use **mechanistic interpretability** (sparse autoencoders, circuit analysis) now rather than attention viz.

---

## 15.5 Transformers

**This is THE architecture.** Vaswani et al. 2017. Everything modern is based on this.

### 15.5.1 Self-Attention
- **Definition**: $y_i = \text{Attn}(x_i, (x_1, x_1), ..., (x_n, x_n))$ — token attends to all tokens (itself + others).
- **Decoder usage**: set $n = i - 1$ (causal mask), so position $i$ only attends to $\leq i$.
- **Key win over RNNs**: at train time, all positions parallelize since all tokens are known (with masking). RNNs are inherently sequential.
- **Linguistic motivation**: coreference resolution. "The animal didn't cross the street because it was tired" vs "...too wide" — "it" attends to animal vs street differently based on context. Self-attention learns this without explicit supervision.

### 15.5.2 Multi-Head Attention (MHA)
- **Why**: one attention matrix = one notion of similarity. Use $h$ heads to capture syntactic, semantic, positional, coref patterns separately.
- **Per head $i$**: $h_i = \text{Attn}(W_i^{(q)} q, \{W_i^{(k)} k_j\}, \{W_i^{(v)} v_j\}) \in \mathbb{R}^{p_v}$.
- **Concat + project**: $\text{MHA} = W_o [h_1; ...; h_h] \in \mathbb{R}^{p_o}$.
- Set $p_q h = p_k h = p_v h = p_o$ to parallelize all heads with one matmul.
- **Modern variants** (book misses these):
  - **Multi-Query Attention (MQA)**: share K, V across heads → less KV cache memory.
  - **Grouped-Query Attention (GQA)**: middle ground, used in Llama 2/3, Mistral, etc.
  - **Multi-head Latent Attention (MLA)** in DeepSeek-V2/V3: compress KV via low-rank latent.

### 15.5.3 Positional Encoding
- Self-attention is **permutation-equivariant** → blind to order. Must inject position.
- **Sinusoidal PE** (the book's focus):
  $$p_{i, 2j} = \sin(i / C^{2j/d}), \quad p_{i, 2j+1} = \cos(i / C^{2j/d}), \quad C = 10000$$
- Combined with embeddings via **addition**: $\text{POS}(\text{Embed}(X)) = X + P$.
- **Key property**: $p_{t+\phi}$ is a linear transform of $p_t$ — relative positions are computable. Inductive bias for shift-equivariance.
- **CRITICAL OMISSION**: The book does not cover **RoPE (Rotary Position Embedding)**, which is what literally every frontier LLM uses (Llama, Mistral, Gemini, DeepSeek, Qwen, etc.). RoPE applies rotation matrices to Q and K such that $q_m^T k_n$ depends only on $m - n$. Also no **ALiBi** (used in MPT, BLOOM), no **NoPE**, no length-extrapolation tricks like YaRN/PI/NTK-aware scaling.

### 15.5.4 Putting It All Together: The Transformer
**Encoder block** (post-norm version in book; modern code uses **pre-norm** for stability):
```python
def EncoderBlock(X):
    Z = LayerNorm(MultiHeadAttn(Q=X, K=X, V=X) + X)
    E = LayerNorm(FeedForward(Z) + Z)
    return E
```
- **Pre-norm** (modern): `Z = X + MHA(LayerNorm(X))` then `E = Z + FFN(LayerNorm(Z))`. Trains deeper models without warmup tricks. Used in GPT-2+, Llama, etc.
- **Components**:
  - **MHA**: mixes info **across positions**.
  - **FFN** (typically 2-layer MLP with 4× hidden dim): mixes info **across features at each position independently**. Where most params live. Conjectured to be where "world knowledge" is stored (Meng+22, ROME paper).
  - **LayerNorm**: standard normalization. Modern: **RMSNorm** (just rescale, no mean centering) is faster and used in Llama, Gemma.
  - **Residual connections**: enables deep training.
- **Encoder**: $N$ stacked blocks on input embeddings.
- **Decoder block**: masked self-attn (causal) + cross-attn over encoder + FFN.
- **Inference**: decoder is autoregressive; new tokens added to attention K/V each step. **KV caching** is the standard optimization (book doesn't cover).

### 15.5.5 Comparing Transformer/CNN/RNN

| Layer | Complexity | Sequential ops | Max path |
|-------|-----------|---------------|----------|
| Self-attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k n d^2)$ | $O(1)$ | $O(\log_k n)$ |

- **Transformer trade**: $O(n^2)$ compute (bad for long seq) but $O(1)$ path (great for gradient flow) and $O(1)$ sequential (great for parallelism on GPUs).
- The $n^2$ wall motivates all of §15.6 + modern stuff (FlashAttention, SSMs).

### 15.5.6 Transformers for Images (ViT)
- **ViT**: chop 224×224 image into 16×16 patches → 196 patch embeddings + [CLS] + position embeddings → standard transformer encoder.
- **Inductive bias trade**: ViT has less inductive bias than CNNs → needs more data. Beats ResNets only at JFT scale (303M images).
- **Modern relevance**: ViTs are the standard for vision now (SAM, CLIP, DALL-E, Stable Diffusion image encoders). **DINOv2/DINOv3** for self-supervised vision. Vision-language models (LLaVA, GPT-4V, Gemini, Claude) all use ViT-like image encoders.

### 15.5.7 Other Variants (briefly)
- **GShard** [Lepikhin+21]: **Mixture of Experts** in FFN layers. Sparse conditional compute. **Hugely important now** — used in GPT-4 (rumored), Mixtral, DeepSeek-V3, Gemini, etc. Allows 10-100× param count with same FLOPs.
- **Conformer**: conv + transformer hybrid for ASR.

---

## 15.6 Efficient Transformers

**Modern relevance**: The book's taxonomy (Tay+2020 survey) is now somewhat dated. In practice, frontier labs have largely **not** adopted sub-quadratic attention variants — they instead use:
1. **FlashAttention** (Dao+22): exact attention but IO-aware tiling → 2-5× faster, used everywhere now.
2. **State Space Models / Mamba** (Gu & Dao 2023): genuinely sub-quadratic, gaining traction.
3. **Sliding window + global attention** (Mistral, Longformer).
4. Just throw more compute at it.

### 15.6.1 Fixed Localized Patterns
- Restrict each token to attend within a block / window. $O(N^2) \to O(N^2/K)$.
- **Sliding window attention**: used in **Mistral**, **Longformer**, **Big Bird**.
- **Dilated/strided patterns**: like dilated convs.

### 15.6.2 Learnable Sparse Patterns
- **Reformer** [KKL20]: LSH hashing → tokens in same bucket attend. $O(N \log N)$.
- **Clustering transformer**: K-means cluster tokens.
- In practice rarely used now — fiddly, doesn't help much vs flash attention.

### 15.6.3 Memory + Recurrence
- **Transformer-XL**: recurrence across blocks. Predecessor to modern long-context tricks.
- **Compressive Transformer**: hierarchical memory.

### 15.6.4 Low-Rank + Kernel Methods
- **Linformer** [Wang+20]: project K,V to lower-rank via random projection. $O(N)$ but loses softmax exactness.
- **Performer** [Cho+20]: approximate softmax via random Fourier features:
  $$A_{i,j} = \exp(q_i^T k_j) = \mathbb{E}[\phi(q_i)^T \phi(k_j)]$$
  - Decompose $A = \mathbb{E}[Q'(K')^T]$ → compute $AV$ in $O(N)$ via $Q' ((K')^T V)$.
  - Beautiful math, modest practical impact.
- **Spiritual descendants**: **Linear Attention** (Katharopoulos+ 20), **RWKV**, **RetNet**, **Mamba** — all exploit some form of linear/kernelized attention or selective state space.

---

## 15.7 Language Models and Unsupervised Representation Learning

**THIS is what made LLMs happen.** Pretrain on huge unlabeled corpus → fine-tune (or just prompt) for downstream tasks.

### 15.7.1 Non-Generative LMs (encoder-only)

#### 15.7.1.1 ELMo
- Two LSTMs (forward + backward) trained as LMs. Concatenate hidden states for contextual word embeddings.
- $\mathcal{L} = -\sum_t [\log p(x_t | x_{1:t-1}) + \log p(x_t | x_{t+1:T})]$.
- Task-specific weighting across layers: lower layers = syntax (POS), higher = semantics (WSD).
- **Historical importance**: Showed pretrained reps transfer. Killed by BERT a year later.

#### 15.7.1.2-15.7.1.5 BERT
- **Bidirectional Encoder Representations from Transformers** [Dev+19]. The encoder-only transformer.
- **Masked Language Modeling (MLM)**: randomly mask 15% of tokens, predict them. Like a denoising autoencoder.
  $$\mathcal{L} = \mathbb{E}_{x \sim D} \mathbb{E}_m \sum_{i \in m} -\log p(x_i | x_{-m})$$
- **Next Sentence Prediction (NSP)**: binary classify if B follows A. (Later shown to be useless — RoBERTa dropped it.)
- **Inputs**: token emb + segment emb (A/B) + positional emb, all summed.
- **Fine-tuning**: add task head, train end-to-end.
  - **CLS classification**: sentence-level (sentiment, NLI).
  - **Token tagging**: NER, parsing.
  - **Span extraction**: QA (predict start + end indices).
- **What BERT learns**: probes reveal it implicitly does POS/parsing/NER/SRL/coref in different layers (Tenney+19, BERTology).
- **Modern relevance**: Encoder-only models lost mindshare to decoder-only (GPT-style) for general LM, but BERT-family still dominates:
  - **Embeddings & retrieval** (sentence-transformers, BGE, E5, Cohere embed).
  - **Reranking** in RAG pipelines.
  - **Classification** when you need fast small models.
  - **ModernBERT** (2024) is the latest revival.

### 15.7.2 Generative (causal) LLMs

#### 15.7.2.1 GPT family
- Decoder-only masked transformer.
- **GPT-1**: joint LM loss + classification loss on small labeled set.
- **GPT-2** [Rad+19]: WebText, scale-up, ditched task-specific training. Zero-shot via prompting.
- **GPT-3** [Bro+20]: 175B params, in-context learning emerges.
- **GPT-4** [Ope23]: multimodal, much better.
- **ChatGPT/InstructGPT**: **RLHF** — train reward model from human preferences, fine-tune with PPO.
- **Scaling hypothesis** [Kap+20]: bigger model + more data + more compute = better performance, predictably. Book mentions this in a footnote but it deserves a whole section (see Critical Omissions).

#### 15.7.2.2 Applications
- **Zero-shot prompting**: model performs task from prompt alone.
- **Prompt engineering**: TL;DR-style triggers.
- **Few-shot / in-context learning**: examples in prompt.
- **Code generation**: Copilot, Codex.
- **Chatbots**: ChatGPT.

#### 15.7.2.3 T5
- Encoder-decoder transformer; everything is text-to-text.
- Pretrained on C4 (750GB clean web text) with span-corruption denoising:
  - Input: "Thank you `<X>` me to your party `<Y>` week"
  - Target: "`<X>` for inviting `<Y>` last `<EOS>`"
- **FLAN-T5**: instruction-tuned on 1800+ tasks. Predecessor to modern instruction tuning.
- **Modern relevance**: Encoder-decoder is mostly dead for general LMs (decoder-only won), but T5-style span corruption + instruction tuning are still influential. **UL2/PaLM/Gemini** mix denoising objectives.

#### 15.7.2.4 Discussion
- Book hedges: "doubt about whether such systems 'understand'." Standard 2022-era skeptic disclaimer.

---

## What the Book Misses (Important for Modern LLM Research)

The book is good on fundamentals (≤2020) but woefully incomplete on what actually matters in 2024-2026 LLM research. Honest take:

### Architecture
- **RoPE (Rotary Position Embedding)** [Su+21]: the de facto position encoding. Apply rotation matrices to Q/K. $\langle q_m, k_n\rangle = f(m - n)$.
- **ALiBi**: linear bias on attention scores instead of position embedding. Used in MPT, BLOOM.
- **RMSNorm** vs LayerNorm: faster, used in Llama family.
- **SwiGLU / GeGLU**: gated FFN variants. Standard in modern transformers (Llama, PaLM, Gemma). The book just says "feedforward layer".
- **Grouped-Query Attention (GQA)** and **Multi-Query Attention (MQA)**: reduce KV cache.
- **Mixture of Experts (MoE)**: only mentioned in passing. Now central — DeepSeek-V3, Mixtral, GPT-4 (rumored), Gemini 1.5.
- **State Space Models (S4, S5, Mamba, Mamba-2)**: subquadratic alternative to attention, gaining serious traction. Selective SSMs (Mamba) are the most promising non-transformer architecture in years.
- **Hybrid models**: Jamba, Zamba, Samba — mix attention + SSM/Mamba layers.

### Training & Optimization
- **FlashAttention** [Dao+22] and **FlashAttention-2/3**: IO-aware exact attention. Trains 2-3× faster, enables much longer context. **Universally adopted.**
- **Scaling laws** [Kap+20, Hoffmann+22 Chinchilla]: $L \approx (N_c / N)^{\alpha_N} + (D_c / D)^{\alpha_D}$. Predicts loss as function of compute/params/data. **Compute-optimal** training ratio (~20 tokens per parameter, Chinchilla scaling).
- **Mixed precision** (bf16, fp8 training).
- **ZeRO/FSDP**, **tensor/pipeline parallelism**, **3D parallelism**: how you actually train >1B param models.
- **Muon, Sophia, Lion** optimizers vs AdamW.

### Inference
- **KV caching**: stash K, V from previous tokens; only compute new ones.
- **PagedAttention / vLLM**: virtual-memory-style KV cache mgmt.
- **Speculative decoding**: small draft model proposes, big model verifies. 2-3× speedup.
- **Medusa / EAGLE / lookahead decoding**: multi-token prediction heads.
- **Quantization**: INT8, INT4 (GPTQ, AWQ), even sub-2-bit. Critical for serving.
- **Continuous batching** (Orca).

### Post-training
- **RLHF**: mentioned but not explained. PPO with KL constraint to reference policy.
- **DPO** (Direct Preference Optimization) [Rafailov+23]: skip the reward model, optimize prefs directly via log-likelihood ratio. Now dominant.
- **Constitutional AI**, **RLAIF**: AI feedback instead of human.
- **Instruction tuning** at scale (FLAN, T0, Alpaca-style SFT).
- **Reasoning RL** (o1/o3/R1 paradigm): RL on chain-of-thought to train reasoning. **The biggest thing happening in 2024-2026 LLM research.** DeepSeek-R1 showed pure RL from base models can elicit reasoning.

### Capabilities + Applications
- **Chain-of-Thought** prompting [Wei+22]: "let's think step by step".
- **In-context learning** theory: still poorly understood but practically central.
- **Tool use / function calling / agents**: LLMs calling external APIs, browsers.
- **RAG (Retrieval-Augmented Generation)**: retrieve + condition. Standard for grounded QA.
- **Long context**: 1M+ tokens (Gemini 1.5, Claude). Needle-in-haystack, RULER benchmarks.
- **Multimodal**: vision-language (LLaVA, GPT-4V, Gemini), audio (Whisper + LLM), video.
- **Mechanistic interpretability**: sparse autoencoders, circuit analysis, induction heads.

### What's actually used in practice (frontier lab POV)
- Decoder-only transformer, pre-norm, RMSNorm, SwiGLU, RoPE (or variants), GQA, FlashAttention-2/3.
- AdamW + cosine LR + warmup + grad clipping.
- Bf16 training, fp8 forward in newer setups.
- Multi-stage training: pretrain → SFT → RLHF/DPO → reasoning RL.
- MoE for biggest models (DeepSeek, GPT-4, Gemini).
- For long context: position interpolation/extrapolation (YaRN, NTK), sliding window, or just train at long context.
- Inference: vLLM, TensorRT-LLM, SGLang. Speculative decoding, paged attention, quantization.

### What's overrated / mostly dead
- Vanilla RNNs/LSTMs (except as conceptual ancestors to Mamba).
- Encoder-decoder for general LMs (T5 → dead; only kept alive for retrieval/MT niches).
- BERT-style MLM for general LMs (but alive in embeddings).
- Sub-quadratic attention approximations (Performer, Reformer, Linformer) — replaced by FlashAttention + Mamba.
- Beam search for open-ended gen.
- Sinusoidal absolute positional embeddings.

---

## Quick Cheat-Sheet of What to Care About from This Chapter

| Section | Importance for LLMs | Why |
|---------|--------|-----|
| 15.2 RNNs/LSTMs | Low (historical) | Mamba/SSM revival worth knowing |
| 15.2.5 BPTT | Low | Conceptual; explains why transformers won |
| 15.2.8 Beam search | Medium | Still used in MT/ASR |
| 15.3 1D CNNs | Low | WaveNet historical; conv hybrids in ASR |
| **15.4 Attention** | **Critical** | Foundation of everything |
| 15.4.3 Scaled dot product | **Critical** | $\sqrt{d}$ scaling, the formula |
| **15.5 Transformers** | **Critical** | THE architecture |
| 15.5.2 MHA | **Critical** | + modern GQA/MQA/MLA variants |
| 15.5.3 Positional encoding | High (but learn RoPE) | Sinusoidal is obsolete; concept matters |
| 15.5.4 Full block | **Critical** | The blueprint |
| 15.5.5 Comparison | High | Why $O(n^2)$ is the central tradeoff |
| 15.5.7 MoE | **Critical** (under-covered) | Modern frontier models all do MoE |
| 15.6 Efficient transformers | Medium | Mostly replaced by FlashAttention + SSMs |
| **15.7 LMs & pretraining** | **Critical** | The paradigm shift |
| 15.7.1 BERT | Medium-High | Encoder-only niche (embeddings, classification) |
| 15.7.2 GPT/LLMs | **Critical** | The whole game |

**Bottom line**: master attention + transformer block + autoregressive LM training/inference, then go read papers from 2022-2026 (Chinchilla, FlashAttention, Mamba, RoPE, DPO, R1) for what actually ships.
