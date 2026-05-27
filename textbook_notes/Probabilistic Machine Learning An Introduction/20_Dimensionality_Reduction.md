# Chapter 20: Dimensionality Reduction

## TL;DR for frontier-lab researchers

- **What's actually load-bearing in modern LLM work**: sparse autoencoders (mech interp), VAEs as latent compressors for diffusion (Stable Diffusion, MovieGen), word/contextual embeddings (which became BGE/E5/OpenAI text-embedding-3), PCA/SVD on activations (probing, low-rank fine-tuning, model diff), t-SNE/UMAP (diagnostic plots only — never trust the geometry), and the linear-representation hypothesis (vector arithmetic → activation steering / DPO direction edits).
- **What's mostly historical**: classical FA/PPCA, Isomap, LLE, Laplacian eigenmaps, MDS, kPCA, MVU. Worth knowing the shapes of, but you won't reach for them.
- **Single most important conceptual thread**: the idea that high-dim data sits on a low-dim manifold, that this manifold can be approximately linearized in feature space, and that **linear structure in the latent is meaningful** (analogies, steering, concept directions). Modern interp basically asks "which subspaces are causal?" — direct descendant of this chapter.

---

## 20.1 PCA (the workhorse — still everywhere)

- **Setup**: linear orthogonal projection $x \in \mathbb{R}^D \to z \in \mathbb{R}^L$ minimizing reconstruction $\mathcal{L}(W) = \frac{1}{N}\sum_n \|x_n - W W^\top x_n\|_2^2$ with $W$ orthonormal.
- **Closed form**: $\hat W = U_L$, top $L$ eigenvectors of empirical cov $\hat\Sigma = \frac{1}{N} X_c^\top X_c$. Equivalently rows-of-data SVD: $X = U_X S_X V_X^\top$, $W = V_X$, **PC scores** $Z = U_X S_X$.
- **Two equivalent objectives**: minimize $\ell_2$ reconstruction error OR maximize variance of projected data ($\mathbb{V}[z_1] = w_1^\top \hat\Sigma w_1 = \lambda_1$).
- **Computational notes**:
	- If $D > N$, work with $N \times N$ Gram matrix $XX^\top$ instead (kernel trick foreshadow).
	- **Randomized SVD** (`sklearn.decomposition.PCA(svd_solver='randomized')`) is $O(NL^2) + O(L^3)$ vs $O(ND^2)$ exact. Use this on activation matrices — what you actually run.
	- **Standardize** (correlation matrix vs covariance) when features have different scales — otherwise PCA chases units, not signal.
- **Choosing $L$**:
	- **Reconstruction error**: $\mathcal{L}_L = \sum_{j=L+1}^D \lambda_j$. Doesn't U-shape on test — PCA isn't a generative model so more dims always help reconstruction (caveat — see PPCA).
	- **Scree plot / variance explained** $F_L = \sum_{j\le L}\lambda_j / \sum \lambda_j$.
	- **Profile likelihood**: change-point model on eigenvalue sequence, gives a clean elbow.

### Why this still matters in 2025

- **Activation analysis on LLM internals**: top PCs of residual stream / MLP activations are how you find "the truth direction" (Burns et al.), refusal direction (Arditi et al.), sycophancy directions, harmfulness directions. Removing them via projection (concept erasure / RLACE / LEACE) is literally just "PCA on activations + null out".
- **Low-rank adaptation**: LoRA's whole premise is that fine-tuning updates are low-rank — $\Delta W = AB^\top$ with $A \in \mathbb{R}^{d \times r}$. Picking $r$ is the same trade-off PCA gives you (scree).
- **Model diffing / SVD of weight matrices**: e.g. Anthropic SVD-on-OV-circuits work; "what does $W_O W_V$ project onto in the unembedding".
- **Embedding compression / Matryoshka embeddings**: training an embedding so its first $k$ dims are themselves a usable PCA-like prefix.
- **PCA-whitening of text embeddings** before retrieval — improves cosine-sim alignment, especially for sentence-BERT family.

### 20.1.4 worth-remembering tricks

- **Whitening**: $Z W^{-1/2}$ → unit covariance. Used in many diffusion latent-spaces (NF, VAE) so the prior is well-conditioned.
- **Profile-likelihood elbow detection** generalizes to "how many SAE features matter", "how many heads to keep when pruning".

---

## 20.2 Factor Analysis (PCA's probabilistic cousin)

- **Generative model**: $z \sim \mathcal{N}(0, I)$, $x | z \sim \mathcal{N}(Wz + \mu, \Psi)$ with $\Psi$ **diagonal**.
	- Induces $p(x) = \mathcal{N}(\mu, WW^\top + \Psi)$: a **low-rank + diagonal** Gaussian, $O(LD)$ params instead of $O(D^2)$.
	- $\Psi_d$ is **uniqueness** — variance specific to dim $d$ not explained by factors.
- **PPCA**: $\Psi = \sigma^2 I$ and $W$ has orthonormal cols. MLE: $W_{\text{mle}} = U_L (L_L - \sigma^2 I)^{1/2} R$, $\sigma^2_{\text{mle}} = \frac{1}{D-L}\sum_{i>L}\lambda_i$ (avg leftover variance). As $\sigma \to 0$, recovers PCA.
- **EM for FA/PPCA**: alternates posterior $p(z|x)$ E-step (linear regression-flavored) and weight M-step. Useful for **missing data**, **online streaming**, and gives the "rigid rod attached by springs" mental picture.
- **Identifiability**: $W$ only identifiable up to orthogonal rotation $R$, since $z$ is isotropic Gaussian. Fixes:
	- Force $W$ orthonormal (PCA).
	- Lower-triangular $W$ (Bayesian, choose founder variables).
	- **Sparsity-promoting priors** ($\ell_1$, ARD, spike-and-slab) → sparse factor analysis.
	- **Varimax rotation** for interpretable factors.
	- Non-Gaussian prior on $z$ (→ ICA, identifiable).

### Honest take

- You will basically never use vanilla FA. **But**:
	- The "$\Psi$ diagonal + $W$ low-rank" decomposition is conceptually the same as **low-rank + per-token bias** in transformer weight analyses.
	- **Sparsity-promoting priors on $W$** is a direct ancestor of **sparse autoencoders** (Section 20.3.4 → modern SAE work).
	- Identifiability discussion is the spiritual root of **disentanglement** — why $\beta$-VAE alone doesn't disentangle without additional constraints.

### 20.2.5–7 Nonlinear / mixture / expfam FA

- **Nonlinear FA**: $p(x|z) = \mathcal{N}(f(z;\theta), \Psi)$ with $f$ a neural net. Intractable posterior → **this is literally the VAE setup** (20.3.5).
- **Mixture of FAs**: $K$ local linear manifolds approximating a curved one. Conceptual ancestor of **mixture-of-experts** in low-rank land.
- **Exponential family PCA / categorical PCA**: replace Gaussian likelihood with expfam. Binary PCA $p(x|z) = \prod_d \text{Ber}(\sigma(w_d^\top z))$ — note the resemblance to a **shallow language-model decoder**.

### 20.2.8 Paired-data variants (CCA & cousins)

- **Supervised PCA**: shared $z$ generating both $x$ and $y$.
- **PLS (partial least squares)**: shared $z^s$ plus a private $z^x$ for inputs.
- **CCA**: fully symmetric — shared + two private latents. MLE = classical CCA.
	- **Deep CCA** (Andrew et al. 2013) — predecessor of **contrastive multimodal models** (CLIP, ALIGN, SigLIP). The shared-latent picture is exactly what CLIP learns; the "two private noise sources" is what makes modality-specific encoders necessary.

---

## 20.3 Autoencoders (THE section for modern LLM/diff researchers)

- **Generic AE**: encoder $f_e: x \to z$, decoder $f_d: z \to \hat x$, loss $\|x - f_d(f_e(x))\|^2$ (or any expfam NLL).
- **Linear AE = PCA** (Baldi & Hornik 1989). Adding nonlinearities → strictly stronger.
- **Bottleneck regularization options**:
	- Undercomplete ($L \ll D$).
	- Overcomplete + denoising / sparsity / contractive penalty.

### 20.3.2 Denoising autoencoders (DAE)

- Train $r(\tilde x) \approx x$ where $\tilde x = x + \epsilon$.
- **Theory result** (Alain & Bengio 2014): residual $e(x) = r(x) - x \approx \sigma^2 \nabla_x \log p(x)$ as $\sigma \to 0$.
	- Translation: **DAE learns the score function**. This is the entire conceptual basis of **score-based diffusion models** (Song & Ermon, DDPM). DAE → score matching → diffusion is a straight line.
- Modern echo: **NoisePred / Diffusion as denoising** in latent space of Stable Diffusion = DAE composed at many noise levels.

### 20.3.3 Contractive autoencoders

- Penalty $\lambda \|\partial f_e / \partial x\|_F^2$ — Frobenius norm of encoder Jacobian.
- Pushes encoder to be locally constant except along directions of true data variation.
- Spiritual cousin of **input-gradient regularization** in adversarial robustness and **Jacobian-based interpretability**.

### 20.3.4 Sparse autoencoders ⭐ (MECH INTERP'S CURRENT WORKHORSE)

- Add **activity regularizer** $\Omega(z) = \lambda \|z\|_1$ to push latent activations sparse.
- Alternative: KL penalty between empirical activation rate $q_k$ and target rate $p$ — closer to what biological neurons look like (~70% off, no permanently dead units).
- **Why this is THE most important section in the chapter for current LLM research**:
	- **Anthropic SAE work** (Bricken et al. "Towards Monosemanticity", Templeton et al. "Scaling Monosemanticity to Claude 3 Sonnet", "Scaling and evaluating sparse autoencoders" OpenAI 2024, Gemma Scope DeepMind 2024):
		- Train a **wide overcomplete** SAE ($L \gg D$, often $L = 32 \times D_{\text{model}}$ to $L = 64\times D_{\text{model}}$) on transformer residual stream activations with **$L_1$ or TopK sparsity**.
		- Each latent unit is interpreted as a **monosemantic feature** ("Golden Gate Bridge", "code in Python", "deceptive intent").
		- Decoder columns are **feature directions** in the residual stream.
	- **TopK SAEs** (Gao et al. OpenAI 2024) replace $L_1$ with a hard-top-k activation — better reconstruction, fewer dead features.
	- **Gated SAEs / JumpReLU SAEs** (DeepMind) address the shrinkage that $L_1$ causes.
	- The "dead unit" problem (units that never activate) — directly the issue Murphy mentions with $L_1$ penalty driving 70% of units permanently off. Resurrection tricks (auxiliary loss on dead features) come straight from this.
	- **Activation steering** then becomes: find SAE feature → add its decoder direction to activations → steer model behavior.
- **Use case examples**:
	- Circuit discovery (find which SAE features mediate a behavior).
	- Concept-level monitoring at inference (detect "lying" feature firing).
	- Targeted unlearning / refusal modulation.
- Code sketch (TopK SAE, the modern version):
	```python
	# x: (B, d_model) batch of residual stream activations
	pre = encoder(x)               # (B, n_features), n_features >> d_model
	z = topk(pre, k=64)             # keep top-k, zero rest
	x_hat = decoder(z)              # (B, d_model)
	loss = ((x_hat - x)**2).sum(-1).mean()
	# decoder rows are unit-normalized after each step
	```

### 20.3.5 Variational Autoencoders ⭐ (LATENT DIFFUSION'S BACKBONE)

- **Generative model**: $p_\theta(x|z) = \mathcal{N}(f_d(z;\theta), \sigma^2 I)$, prior $p(z) = \mathcal{N}(0, I)$.
- **Inference / recognition net**: $q_\phi(z|x) = \mathcal{N}(f_{e,\mu}(x), \text{diag}(f_{e,\sigma}(x)))$.
- **ELBO**: $\mathcal{L}(\theta,\phi;x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)\,\|\,p(z))$
	- Expected log-likelihood (reconstruction) + KL pull-toward-prior (regularization).
	- KL has closed form for Gaussian posterior: $-\frac{1}{2}\sum_k(\log \sigma_k^2 - \sigma_k^2 - \mu_k^2 + 1)$.
- **Reparameterization trick**: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$. Lets gradients flow through sampling.
- **AE vs VAE**:
	- AE embeds points → delta-functions in latent → can't sample new data.
	- VAE embeds **Gaussians** → smooth latent → enables interpolation and sampling.
	- VAE latent space approximately zero-curvature → **linear interpolation $z = \lambda z_1 + (1-\lambda)z_2$ traces meaningful images**. This is the manifold-flattening property exploited by every latent-space editing method.

#### Why VAEs matter NOW (not for sampling — for compression)

- **Stable Diffusion / SDXL / Flux**: the "VAE" you hear about is a **KL-regularized AE** that maps $512\times512\times3$ images to a $64\times64\times4$ latent. Diffusion happens entirely in latent space. Without this, training at full res is infeasible.
	- The encoder is the LDM **first-stage model**; VQ-VAE-style discrete latents (VQ-GAN, MoVQ) and continuous KL latents both descend from this.
- **Sora / Veo / MovieGen** etc.: same idea but a 3D causal VAE compressing video spatially+temporally (e.g. 8x8x4 compression) → diffusion on tokens.
- **Audio**: Encodec / SoundStream / DAC — same recipe (residual vector quantization VAE) for audio LMs.
- **Speech LMs** like Mimi (Moshi) use a neural audio codec built on this template.
- **VQ-VAE** (Oord et al.) is the discrete-latent variant — replaces the Gaussian prior with a learned codebook. Used in Parti, MaskGIT, and is the conceptual root of using "tokenizers" for non-text modalities in multimodal LLMs.

#### VAE gotchas

- **Posterior collapse**: when decoder is too powerful, $q(z|x) \to p(z)$ and $z$ becomes uninformative. Fix with KL annealing, free bits, or weakening the decoder.
- **Blurry reconstructions**: Gaussian likelihood + mean prediction = mode averaging. Modern latent-space models use GAN loss or perceptual loss on top (LPIPS) — this is what makes SD's VAE actually sharp.

### Comparison ranking for "what should I actually use"

| Use case | Pick |
|---|---|
| Compress images/video for diffusion | KL-VAE or VQ-VAE |
| Interpret a frozen LLM | Sparse autoencoder (TopK) |
| Denoise / fill in missing data | DAE (or its descendant, a diffusion model) |
| Linear probe a layer | PCA on activations |
| Visualize embeddings | t-SNE / UMAP |

---

## 20.4 Manifold Learning (mostly historical, but the conceptual hypothesis is everything)

- **Manifold hypothesis** (FMN16, but folklore older): natural high-dim data lies on a low-dim manifold.
	- Random $64\times57$ image looks like noise; real digit-6 images live on a 2D manifold (rotation angle, etc.).
	- **Intrinsic dimensionality** $d$ ≪ **ambient dimensionality** $D$.
- This hypothesis is **THE** justification for why neural nets generalize, why diffusion works, why low-rank fine-tuning works, why activation steering uses a few directions, why SAEs find a small number of active features.

### 20.4.4 MDS

- **Classical MDS**: preserve pairwise inner products via top eigenvectors of centered Gram matrix → equivalent to PCA (!).
- **Metric MDS**: preserve raw distances (stress objective, SMACOF algorithm).
- **Non-metric MDS**: preserve only **rankings** of distances. Isotonic regression + alternating opt.
- **Sammon mapping**: down-weight large distances, focus on local structure.

### 20.4.5 Isomap

- KNN graph + Dijkstra shortest paths approximate **geodesic distance** along the manifold, then MDS.
- Fragile to **topological instability** (short-circuits through noise).

### 20.4.6 Kernel PCA

- PCA on $\phi(x)$ via the kernel trick. Eigendecompose Gram $K = \phi\phi^\top$ instead of cov.
- With RBF kernel, **expands** the feature space — not great for reduction directly, but motivates Maximum Variance Unfolding.

### 20.4.7 MVU / 20.4.8 LLE / 20.4.9 Laplacian eigenmaps

- All variants of "preserve local structure via the eigendecomposition of some matrix derived from KNN graph". Mostly historical.
- **Laplacian eigenmaps** uses the **graph Laplacian** $L = D - W$. The Dirichlet energy $f^\top L f = \frac{1}{2}\sum_{ij} w_{ij}(f_i - f_j)^2$ measures function smoothness on a graph.
	- **Why this concept survives**: **graph neural networks** (GCN, GAT) are basically learnable spectral filters on graph Laplacians. The "smoothness over a graph" inductive bias from this section directly motivates GNN message passing. And spectral clustering = normalized cuts uses the same machinery.

### 20.4.10 t-SNE & UMAP ⭐ (you will use these every week)

- **SNE**: convert distances to conditional probabilities $p_{j|i} \propto \exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)$ in input space, $q_{j|i}$ similarly in low-dim, minimize $\sum_i \text{KL}(P_i\|Q_i)$.
- **Symmetric SNE**: single KL of joints, simpler.
- **t-SNE innovation**: use **Student-t with $\nu=1$ (Cauchy)** for $q_{ij}$ in low-dim → heavier tails → fixes the **crowding problem** (the squashing of moderately-far points). Gradient $\nabla_{z_i} \mathcal{L} = 4\sum_j (p_{ij} - q_{ij})(z_i - z_j)(1 + \|z_i - z_j\|^2)^{-1}$ — the $(1+\cdot)^{-1}$ acts like inverse-square repulsion, gives well-separated clusters.
- **Perplexity** $= 2^{H(P_i)}$ — soft "effective number of neighbors". Default 30 in sklearn. **Very sensitive** — always run with multiple perplexities.
- **Barnes-Hut** approximation → $O(N \log N)$ but only for 2D output.
- **UMAP**: similar to t-SNE but preserves global structure better, much faster, more interpretable hyperparams (`n_neighbors`, `min_dist`). Widely the default now.

#### Honest take on t-SNE/UMAP for LLM work

- **Use for**: quick diagnostic plots of embeddings, activation clustering, "do my classes separate".
- **Do not use for**: claims about distance, cluster sizes, density, or geometry. Cluster size in t-SNE/UMAP is meaningless. Distance between clusters is meaningless.
- **Common real uses**:
	- Visualize SAE feature activation patterns across a corpus.
	- See if a fine-tune moved certain examples in embedding space.
	- Plot per-token residual stream at layer $L$ across prompt categories.
	- Sanity-check that a sentence-embedding model separates intent classes.
- For **quantitative geometry**, prefer PCA, CKA, or linear probes.

---

## 20.5 Word Embeddings ⭐⭐⭐ (the direct ancestor of every modern embedding/retrieval system)

- **Distributional hypothesis** (Harris '54, Firth '57): "a word is characterized by the company it keeps". Two words are similar if they appear in similar contexts.
- Everything in this section is the conceptual scaffolding for: **sentence/document embeddings (Sentence-BERT, E5, BGE, GTE, Cohere embed, OpenAI text-embedding-3)**, **retrieval (RAG)**, **dense ANN search (FAISS, ScaNN)**, **vector databases**, **CLIP/multimodal embeddings**.

### 20.5.1 LSI / LSA — SVD of count matrix

- Build term-document or term-context matrix $C$, truncated SVD $C \approx USV^\top$, define word embedding $u_i$ and context embedding $s \odot v_j$.
- **Cosine similarity** $\text{sim}(q, d) = q^\top d / (\|q\| \|d\|)$ — still the dominant retrieval metric in 2025.
- **PMI / PPMI weighting** instead of raw counts → much better embeddings (BL07 showed PPMI + SVD matches word2vec on many tasks).

### 20.5.2 Word2vec

- **CBOW**: predict center word from averaged context window.
- **Skip-gram**: predict each context word from center word. $\log p(w_o | w_c) = u_o^\top v_c - \log \sum_{i \in V} \exp(u_i^\top v_c)$.
- **Skip-gram with Negative Sampling (SGNS)**: replace expensive softmax with binary classification — true (positive) word vs $K$ noise words from a $\text{freq}(w)^{3/4}$ unigram. The 3/4 power dampens common words (this trick was important and reappears everywhere — e.g. softmax-temperature in contrastive losses).
	- $p(D=1|w_t, w_{t+j}) = \sigma(u_{w_{t+j}}^\top v_{w_t})$.

### 20.5.3 GloVe

- Factorize log-co-occurrence matrix directly with weighted least squares:
- $\mathcal{L} = \sum_{i,j} h(x_{ij})(u_j^\top v_i + b_i + c_j - \log x_{ij})^2$ where $h(x) = (x/c)^{0.75}$ capped at 1.
- Faster than skip-gram; similar quality.

### 20.5.4 Word analogies ⭐ (THE SECTION THAT BIRTHED ACTIVATION STEERING)

- Famous empirical observation: $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$.
- Conjecture (Pennington et al.): analogy $a:b::c:d$ holds iff $\frac{p(w|a)}{p(w|b)} \approx \frac{p(w|c)}{p(w|d)}$ for all $w$.
- **This is the foundational evidence for the linear representation hypothesis**: meaningful concepts correspond to **directions** in embedding space, and concept composition is linear.

#### Why this is the most important conceptual section for current alignment/interp research

- **Activation steering / representation engineering** (Turner et al., Zou et al. "Representation Engineering"):
	- Compute $\Delta = \bar h_{\text{positive examples}} - \bar h_{\text{negative examples}}$ on residual stream.
	- Add $\alpha \Delta$ at inference → steers behavior.
	- **This is literally the word-analogy trick lifted to LLM activations.**
- **Concept vectors / steering vectors / control vectors** (anthropic, MATS, etc.) all rely on this linearity.
- **DPO/IPO/KTO** implicitly find a "preference direction"; **ablation studies** project out a single direction (Arditi et al. on refusal).
- **Truth direction** (Burns et al. CCS) — find a single direction in activation space such that its sign tracks truthfulness.
- **Bias mitigation** — project out the gender direction (Bolukbasi et al. 2016, "Man is to Computer Programmer as Woman is to Homemaker") — same technology as analogy arithmetic.

#### RAND-WALK theoretical model (20.5.5)

- Log-bilinear LM with latent context vector $z_t$ that random-walks slowly. Self-normalization $Z(z_t) \approx Z$ const.
- Predicted $\text{PMI}(w, w') \approx v_w^\top v_{w'} / D$.
- Connects skip-gram, GloVe, PPMI+SVD as variants of a frequency-weighted SVD on PMI — this **unifies all classical word embedding methods**.

### 20.5.6 Contextual word embeddings (one paragraph but it's the whole point)

- A word's vector depends on its context. ELMo → BERT → GPT.
- This is where the chapter cuts off — but everything downstream (sentence embeddings, multilingual embeddings, RAG, CLIP) is just "contextual embeddings at the document level, trained with contrastive or denoising objectives".

#### Modern embedding-model pipeline (where this stuff actually lives now)

- **Architecture**: pretrained encoder (BERT-family, T5-encoder, decoder-only mean-pool) → pooling → optional linear head.
- **Objective**: contrastive loss (InfoNCE), often with **mined hard negatives** — direct descendant of SGNS negative sampling.
- **State of the art** (mid-2025): GTE-large/Qwen3-embed, BGE-m3, E5-mistral, OpenAI text-embedding-3-large, Cohere embed-v3, Voyage-large. All trained with contrastive + cross-encoder distillation.
- **Matryoshka representation learning**: train so first $k$ dims are themselves a usable embedding — bakes in PCA-like dim-reduction structure.
- **Retrieval**: still cosine similarity, ANN via HNSW/IVFPQ in FAISS or ScaNN or pgvector.
- **Multimodal**: CLIP/SigLIP = contrastive aligned image-text embedding. Conceptual line: distributional hypothesis → SGNS → CLIP. Just generalize "context" from "neighboring words" to "paired modality".

---

## What to skip on a first read / what to come back for

- **Skip on first read**: 20.2.3 (EM derivations), 20.4.4–9 (classical manifold methods unless you specifically need them), 20.5.5 (RAND-WALK theory unless you care why analogies work formally).
- **Come back for / internalize**:
	- 20.1 (PCA + SVD computation)
	- 20.3.4 (sparse AE — re-read alongside Bricken et al.)
	- 20.3.5 (VAE — re-read alongside LDM paper)
	- 20.5.4 (analogies — re-read alongside Turner steering paper)
	- t-SNE/UMAP caveats (always)

## Master analogy table: chapter concept → modern LLM use

| Chapter concept | Modern descendant |
|---|---|
| PCA / SVD | Activation analysis, LoRA, model diffing, embedding compression |
| Sparse FA | Sparse autoencoders for mech interp (Anthropic, OpenAI, DeepMind) |
| PPCA | LoRA, low-rank weight decompositions |
| Nonlinear FA | VAE → latent diffusion (Stable Diffusion, Sora, Veo) |
| Denoising AE | Score matching → diffusion models |
| Mixture of FAs | Local linear approximations / experts in MoE-style decompositions |
| CCA / Deep CCA | CLIP, SigLIP, multimodal contrastive learning |
| Laplacian eigenmaps | Graph NNs, spectral clustering |
| t-SNE / UMAP | Diagnostic plots only (with caveats) |
| LSA / LSI | Modern dense retrieval, RAG, vector DBs |
| Word2vec / SGNS | Contrastive learning, hard-negative mining, InfoNCE |
| Word analogies | Activation steering, refusal direction, truth direction, concept erasure |
| Distributional hypothesis | All of self-supervised pretraining; CLIP; embedding models |
| Manifold hypothesis | Why everything works; justification for low-rank / sparse interp |
