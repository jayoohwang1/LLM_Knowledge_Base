# Ch 10-11: Convolutional Networks & Residual Networks

## Importance Map for LLM Research

Before diving in -- honest take on what matters here for modern LLM work:

- **Residual connections**: **THE** most important concept in these chapters. Every transformer is a residual network. Without skip connections, you cannot train anything at the depth that modern LLMs require. This is non-negotiable foundational knowledge.
- **Normalization (LayerNorm specifically)**: BatchNorm is the historical entry point taught here, but the real payoff is understanding *why* normalization helps -- smoother loss surfaces, stable forward/backward passes, enabling higher learning rates. LLMs use LayerNorm (and variants like RMSNorm), not BatchNorm, but the theory transfers directly.
- **1x1 convolutions / channel mixing**: The concept of pointwise linear projections across channels is exactly what MLP blocks in transformers do. Think of "channels" as "hidden dimensions."
- **Bottleneck blocks**: The reduce-process-expand pattern (1x1 -> 3x3 -> 1x1) directly inspired efficient transformer designs (e.g., SwiGLU FFN blocks, low-rank adapters like LoRA).
- **Encoder-decoder architectures / U-Nets**: Relevant to diffusion models (which are almost always U-Net based), and the encoder-decoder structure maps to seq2seq transformers (T5, original transformer).
- **Convolutions themselves**: Less directly relevant to LLMs (transformers replaced them for sequence modeling), but still critical for vision encoders in multimodal models (CLIP, vision transformers with conv stems), and 1D convolutions show up in some efficient architectures (Mamba's conv layers, Hyena).
- **Shattered gradients**: Important conceptual understanding of *why* naive depth fails -- relevant to understanding why architectural innovations like residual connections and normalization were necessary.
- **Classic CNN architectures (AlexNet, VGG)**: Mostly historical context. Good to know the narrative arc but you won't use this directly.
- **Object detection, semantic segmentation specifics**: Niche unless you're doing vision. The *architectural patterns* (multi-scale processing, encoder-decoder) matter more than the specific systems.

---

## Ch 10: Convolutional Networks

### 10.1 Invariance and Equivariance

- **Three problems with fully connected nets on images**:
	- Images are high-dimensional ($224 \times 224 \times 3 = 150{,}528$ dims) -- FC hidden layers would need $150{,}528^2 \approx 22$ billion weights even for a shallow net
	- FC nets have no notion of spatial locality -- treat every pixel relationship identically
	- FC nets aren't aware of translation -- must re-learn "cat" at every possible position independently
- **Invariance**: $f[\mathbf{t}[\mathbf{x}]] = f[\mathbf{x}]$ -- output unchanged by transformation
	- Needed for classification: shifted mountain is still a mountain
- **Equivariance**: $f[\mathbf{t}[\mathbf{x}]] = \mathbf{t}[f[\mathbf{x}]]$ -- output transforms the same way as input
	- Needed for segmentation: shifted image should give shifted segmentation map
- **Convolutions are equivariant to translation by construction** -- this is the key inductive bias
	- *[LLM relevance: transformers achieve a different kind of equivariance -- they're equivariant to permutation of input tokens (before positional encoding). The idea that architecture should encode known symmetries of the data is the deeper principle here, and it's what drove development of both CNNs and attention.]*

### 10.2 Convolutional Networks for 1D Inputs

#### 10.2.1 1D Convolution Operation
- **Convolution**: weighted sum of nearby inputs using shared kernel $\boldsymbol{\omega} = [\omega_1, \omega_2, \omega_3]^T$
	- $z_i = \omega_1 x_{i-1} + \omega_2 x_i + \omega_3 x_{i+1}$
	- Same weights at every position -- equivariant to translation
	- Technically a cross-correlation (weights not flipped), but ML convention calls it convolution

#### 10.2.2 Padding
- **Zero-padding**: assume input is zero outside valid range, output same size as input
- **Valid convolutions**: only compute where kernel fully overlaps input, output shrinks
- Other options: circular padding, reflection padding

#### 10.2.3 Stride, Kernel Size, and Dilation
- **Stride**: skip positions when evaluating kernel -- stride 2 halves output size
- **Kernel size**: how many inputs contribute to each output -- larger = wider receptive field but more params
- **Dilated/atrous convolutions**: intersperse zeros in kernel to expand receptive field without adding params
	- Dilation rate = number of zeros between weights + 1
	- *[LLM relevance: dilated convolutions are conceptually interesting because they solve the same problem attention does -- integrating distant information efficiently. WaveNet used stacked dilated convolutions for autoregressive generation before transformers took over.]*

#### 10.2.4 Convolutional Layers
- Full layer: convolution + bias + activation function
- $h_i = a\left[\beta + \sum_{j=1}^{3} \omega_j x_{i+j-2}\right]$
- **Conv layer is a special case of FC layer** where most weights are zero and remaining weights are tied
	- FC layer with $D$ inputs and $D$ hidden units: $D^2$ weights + $D$ biases
	- Conv layer: just $K$ weights + 1 bias (kernel size $K$)
	- Conv structure acts as a regularizer -- restricts the function family to translation-equivariant ones

#### 10.2.5 Channels
- Single convolution loses information (averaging + ReLU clipping)
- **Multiple convolutions in parallel** -- each produces a **feature map** or **channel**
- With $C_i$ input channels, kernel size $K$, $C_o$ output channels: $C_i \times C_o \times K$ weights + $C_o$ biases
	- *[LLM relevance: "channels" in CNNs are directly analogous to the hidden dimension $d_{model}$ in transformers. Multiple attention heads are like multiple convolutional filters -- different learned patterns applied in parallel.]*

#### 10.2.6 Receptive Fields
- **Receptive field**: region of original input that influences a given hidden unit
- Grows with depth: each layer with kernel size 3 adds 2 to the receptive field
- Stride > 1 accelerates receptive field growth
- Information is gradually integrated from across the input through successive layers
	- *[LLM relevance: this is the fundamental limitation that motivated attention. CNNs need $O(\text{depth})$ layers for full-context integration. Attention gives $O(1)$ -- every position attends to every other position in one layer. But recent efficient architectures (state space models, RWKV) revisit the idea of building up context gradually, more like convolutions.]*

### 10.3 Convolutional Networks for 2D Inputs
- 2D kernel $\boldsymbol{\Omega} \in \mathbb{R}^{K \times K}$ slides over 2D input grid
- $h_{ij} = a\left[\beta + \sum_{m=1}^{3}\sum_{n=1}^{3} \omega_{mn} x_{i+m-2, j+n-2}\right]$
- For RGB input (3 channels): kernel is $3 \times 3 \times 3$, applied to all channels at each position
- To produce $C_o$ output channels from $C_i$ input channels with $K \times K$ kernels: $C_i \times C_o \times K \times K$ weights + $C_o$ biases

### 10.4 Downsampling and Upsampling

#### 10.4.1 Downsampling
- **Sub-sampling**: retain every other position (stride 2 convolution)
- **Max pooling**: take max of $2 \times 2$ block -- induces some translation invariance
- **Average/mean pooling**: take mean of block
- All reduce spatial dims by 2x while keeping channel count
	- *[LLM relevance: pooling is how CNNs compress information spatially. The transformer analogue is the [CLS] token or mean pooling over sequence positions for getting a fixed-size representation. In multimodal models, vision encoder output is often spatially pooled before feeding to the LLM.]*

#### 10.4.2 Upsampling
- **Duplication**: repeat each value (nearest neighbor)
- **Max unpooling**: redistribute values to their original max positions
- **Bilinear interpolation**: smooth fill between known values
- **Transposed convolution**: reverse of strided convolution -- each input contributes to multiple outputs
	- Weight matrix is the transpose of the downsampling convolution's weight matrix
	- Used heavily in decoders (image generation, segmentation)

#### 10.4.3 Changing the Number of Channels
- **1x1 convolution**: weighted sum of all channels at each spatial position + bias + activation
	- Equivalent to running the same FC network independently at every spatial position
	- Weights: $1 \times 1 \times C_i \times C_o$
	- *[LLM relevance: **this is exactly what the FFN/MLP block in a transformer does** -- a pointwise (position-wise) fully connected network applied independently to each token. The 1x1 conv is the CNN precursor to this idea. Understanding this connection is genuinely useful.]*

### 10.5 Applications

#### 10.5.1 Image Classification
- **ImageNet**: 1.28M training images, 1000 classes, $224 \times 224$ RGB input
- **AlexNet** (2012): 8 layers (5 conv + 3 FC), ~60M params
	- First deep net to dominate ImageNet: 16.4% top-5 error (vs ~25% prior SOTA)
	- Used ReLU, dropout, data augmentation, SGD+momentum
	- Kick-started the modern deep learning era
- **VGG** (2014): 19 layers, 144M params, 6.8% top-5 error
	- Key insight: **depth matters** -- just stack more $3 \times 3$ convs
	- Pattern: spatial dims decrease, channels increase, end with FC layers
- Trend: deeper networks consistently improved until they didn't (motivating Ch 11)

#### 10.5.2 Object Detection (YOLO)
- Goal: identify + localize multiple objects with bounding boxes
- YOLO: 24 conv layers -> $7 \times 7$ grid, predicts class + bounding boxes at each cell
- Transfer learning from ImageNet classification
- Non-max suppression to deduplicate overlapping boxes

#### 10.5.3 Semantic Segmentation
- Goal: per-pixel classification
- **Encoder-decoder** architecture: downsample (encode) then upsample (decode)
	- Encoder: VGG-like convs + pooling to compress to low-res global representation
	- Decoder: transposed convolutions + unpooling to reconstruct full-resolution output
	- Also called **hourglass networks** due to their shape
	- *[LLM relevance: the encoder-decoder pattern is everywhere. T5, BART, the original transformer -- all encoder-decoder. Diffusion models for image generation use U-Net, which is an encoder-decoder with skip connections. If you work on multimodal generation, you will encounter this.]*

---

## Ch 11: Residual Networks

### Priority: This entire chapter is high-priority for LLM research

*Residual connections are arguably the single most important architectural innovation for training deep networks. Every modern LLM is a stack of residual blocks. If you understand one thing from these two chapters, understand this.*

### 11.1 Sequential Processing

- Standard deep net: $\mathbf{h}_1 = f_1[\mathbf{x}, \boldsymbol{\phi}_1]$, $\mathbf{h}_2 = f_2[\mathbf{h}_1, \boldsymbol{\phi}_2]$, ..., $\mathbf{y} = f_K[\mathbf{h}_{K-1}, \boldsymbol{\phi}_K]$
- Equivalent to nested function composition: $\mathbf{y} = f_K[\ldots f_2[f_1[\mathbf{x}, \boldsymbol{\phi}_1], \boldsymbol{\phi}_2] \ldots, \boldsymbol{\phi}_K]$

#### 11.1.1 Limitations of Sequential Processing
- **Empirical observation**: VGG (19 layers) beats AlexNet (8 layers), but adding more layers *hurts* -- 56-layer CNN does worse than 20-layer on CIFAR-10
- **Surprising**: the degradation is on **both train and test** -- not an overfitting problem, it's an optimization problem
- **Shattered gradients**: the loss gradient w.r.t. parameters in early layers changes unpredictably
	- Derivative of output w.r.t. first layer: $\frac{\partial \mathbf{y}}{\partial \mathbf{f}_1} = \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_1} \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_2} \frac{\partial \mathbf{f}_4}{\partial \mathbf{f}_3}$
	- Changing $\mathbf{f}_1$ shifts where **all** subsequent derivatives are evaluated
	- Finite step size means the gradient computed at the current point may be irrelevant after the update
	- **Autocorrelation of gradients drops to zero** for deep nets -- nearby inputs have completely unrelated gradients
	- Loss surface looks like "an enormous range of tiny mountains" instead of a smooth landscape
	- *[This is the deep reason SGD fails on very deep nets. It's not just vanishing/exploding gradients -- even with He initialization keeping magnitudes stable, the landscape itself is chaotic. This distinction matters for understanding why residual connections help.]*

### 11.2 Residual Connections and Residual Blocks

- **The fix**: add the input back to the output of each layer (skip/residual connection)
	- $\mathbf{h}_1 = \mathbf{x} + f_1[\mathbf{x}, \boldsymbol{\phi}_1]$
	- $\mathbf{h}_2 = \mathbf{h}_1 + f_2[\mathbf{h}_1, \boldsymbol{\phi}_2]$
	- Each function $f_k$ learns an **additive update** to the current representation
	- Input and output of each block must have same dimensionality
- **Ensemble interpretation**: unraveling the recursion shows the output is a **sum of $2^K$ paths** through $K$ blocks
	- For 4 blocks: $\mathbf{y} = \mathbf{x} + f_1[\mathbf{x}] + f_2[\mathbf{x} + f_1[\mathbf{x}]] + f_3[\ldots] + f_4[\ldots]$ -- sum of the input plus 4 "sub-networks" of varying depth
	- Can think of it as an **implicit ensemble** of shallow networks
	- *[LLM relevance: **this is exactly the structure of a transformer**. Each transformer layer computes attention + FFN as additive updates to the residual stream. The "residual stream" interpretation of transformers (Elhage et al., 2021) -- where information flows through the residual and layers read/write to it -- comes directly from this. Mechanistic interpretability heavily relies on this view.]*

- **Gradient benefits**: derivative of output w.r.t. first layer now includes an **identity term** plus shorter chains
	- $\frac{\partial \mathbf{y}}{\partial \mathbf{f}_1} = \mathbf{I} + \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_1} + \left(\frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_1} + \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_1}\frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_2}\right) + \ldots$
	- Identity term means early layers always get direct gradient signal regardless of depth
	- Shorter chains of derivatives are better behaved
	- **Less shattered gradients** -- loss surface is smoother

#### 11.2.1 Order of Operations in Residual Blocks
- **Naive order** (Linear -> ReLU -> add): residual block can only add non-negative values (ReLU output >= 0)
- **Pre-activation order** (ReLU -> Linear -> add): both positive and negative updates possible
	- Typically: activation first, then linear transformation
	- Need initial linear layer before first block (in case input is all negative, ReLU would kill it)
- **In practice**: blocks usually contain multiple layers (e.g., ReLU -> Linear -> ReLU -> Linear)
	- *[LLM relevance: transformers use pre-norm (LayerNorm before attention/FFN) which is the same principle as pre-activation residual blocks. The original transformer used post-norm, but pre-norm is now standard because it's more stable to train -- exactly the lesson from ResNet v2.]*

### 11.3 Exploding Gradients in Residual Networks

- Even with residual connections, depth isn't free
- **Problem**: each residual block adds variance to the representation
	- With He init + ReLU, the residual branch output has variance = input variance
	- Adding residual branch to skip connection **doubles the variance**
	- After $K$ blocks: variance grows as $2^K$ -- **exponential blowup**
	- Same applies to gradients in the backward pass
- **Partial fix**: scale residual block output by $1/\sqrt{2}$ -- keeps variance stable
- **Better fix**: batch normalization (next section)
	- *[LLM relevance: this variance accumulation problem is real and shows up in LLM training. DeepSeek-V2 and other very deep models use specific initialization schemes (e.g., scaling residual branches by $1/\sqrt{2N}$ where $N$ is depth) to manage this. GPT-2 scales residual connections by $1/\sqrt{N}$ where $N$ is the number of layers. Understanding why this is needed comes from exactly this analysis.]*

### 11.4 Batch Normalization

- **BatchNorm**: for each hidden unit $h$, normalize across the batch $\mathcal{B}$:
	1. Compute batch mean: $m_h = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} h_i$
	2. Compute batch std: $s_h = \sqrt{\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}}(h_i - m_h)^2}$
	3. Standardize: $h_i \leftarrow \frac{h_i - m_h}{s_h + \epsilon}$
	4. Scale and shift with learned params: $h_i \leftarrow \gamma h_i + \delta$
- Applied independently to each hidden unit
- For conv nets: applied per channel, statistics computed over batch AND spatial positions
	- $K$ layers, $C$ channels each: $KC$ learned $\delta$ offsets and $KC$ learned $\gamma$ scales
- **At test time**: use running statistics from training (no batch to compute over)

#### 11.4.1 Costs and Benefits of Batch Normalization

- **Cost**: creates redundancy (network becomes invariant to weight/bias rescaling since normalization compensates), adds params ($\gamma$, $\delta$ per unit)
- **Stable forward propagation**: with $\delta = 0$, $\gamma = 1$ at init, every activation has unit variance
	- In residual nets: variance increases **linearly** (not exponentially) with depth -- $k$-th block adds 1 unit of variance to existing $k$
	- **Side effect at init**: later layers contribute less relative variation, so network effectively starts shallow and **learns its own depth** as $\gamma$ values increase during training
	- *[LLM relevance: this "growing depth" behavior at initialization is beautiful and has a direct analogue in transformers. With proper init, a 96-layer transformer initially behaves like a shallow network and progressively deepens during training. This is connected to why warmup is important -- you need to let the network gradually learn to use its depth.]*
- **Higher learning rates**: BatchNorm makes loss surface smoother + gradient more predictable -> can use larger steps
- **Regularization**: batch statistics inject noise (different batch = different normalization), acts as implicit regularizer

#### Variations of Batch Normalization (from Notes section)

- **Ghost BatchNorm**: compute stats on subsets of the batch for more noise/regularization
- **Batch renormalization**: running average of stats for stability when batch size is small/noisy
- **LayerNorm**: normalize each example separately, across channels + spatial positions
	- No dependence on batch -- works with any batch size, works for sequences
	- *[LLM relevance: **this is what transformers use**. LayerNorm normalizes across the hidden dimension for each token independently. BatchNorm is unusable for autoregressive LLMs because (a) batch sizes vary, (b) sequences have variable length, (c) you can't normalize across the batch at each sequence position during generation. RMSNorm (used in LLaMA, Gemma, etc.) is LayerNorm without the mean subtraction -- just divide by RMS, even simpler.]*
- **GroupNorm**: split channels into groups, normalize within each group
	- Compromise between LayerNorm and InstanceNorm
- **InstanceNorm**: normalize each channel separately across spatial positions only
	- Popular in style transfer
- **Why BatchNorm actually helps** -- not fully understood:
	- Original claim: reduces "internal covariate shift" -- but Santurkar et al. (2018) showed this isn't the real reason
	- **Actual reason**: makes loss surface smoother and its gradient change more slowly
	- Theoretical result: with BatchNorm, distance to nearest optimum is smaller for any initialization

### 11.5 Common Residual Architectures

#### 11.5.1 ResNet
- **ResNet block**: BN -> ReLU -> Conv $3 \times 3$ -> BN -> ReLU -> Conv $3 \times 3$ -> add
- **Bottleneck ResNet block** (for very deep nets): three convolutions
	- $1 \times 1$ conv (reduce channels by 4x) -> $3 \times 3$ conv -> $1 \times 1$ conv (expand channels back)
	- Same $3 \times 3$ receptive field, far fewer parameters
	- *[LLM relevance: **this reduce-process-expand pattern is the blueprint for transformer FFN blocks**. The standard FFN does: project up to 4x hidden dim, apply activation, project back down. SwiGLU (used in LLaMA, PaLM, etc.) is a gated version of this. LoRA also exploits this -- it approximates weight updates with a low-rank bottleneck. The idea that you can process information efficiently through a narrow bottleneck and then expand back is one of the most reusable patterns in deep learning.]*
- **ResNet-200**: 7x7 conv -> maxpool -> bottleneck blocks with periodic downsampling (stride 2) + channel doubling -> average pool -> FC -> softmax
	- 4.8% top-5 error on ImageNet, one of first to exceed human performance
	- Current SOTA is transformer-based (ViT variants)

#### 11.5.2 DenseNet
- **Alternative to residual addition**: **concatenate** all previous layer outputs
	- Layer input = concatenation of all prior layer outputs
	- Direct contribution from every earlier layer (not just previous one)
	- More flexible than simple addition -- can learn weighted combinations
- **Problem**: channel count grows rapidly -> use $1 \times 1$ convs to compress before $3 \times 3$ conv
- Concatenation breaks at downsampling boundaries (different spatial sizes)
- Competitive with ResNet at comparable parameter counts
	- *[LLM relevance: DenseNet-style connections are rare in LLMs, but the idea appears in some places. DenseFormer (2024) adds weighted connections from all previous layers. Some mixture-of-experts routing can be seen as selective dense connections. The standard residual approach won out for simplicity and scalability.]*

#### 11.5.3 U-Nets and Hourglass Networks
- **U-Net**: encoder-decoder with **residual connections between matching scales**
	- Encoder: repeated downsampling (convs + pooling)
	- Decoder: repeated upsampling (transposed convolutions)
	- Skip connections: concatenate encoder representations to decoder at each scale
	- Fully convolutional -- works on any input size
	- Originally for medical image segmentation
	- *[LLM relevance: **U-Net is THE architecture for diffusion models** (Stable Diffusion, DALL-E 2, Imagen). If you work on image generation at all, you will encounter U-Nets constantly. The skip connections between encoder and decoder are critical -- without them, high-frequency details are lost in the bottleneck. Newer diffusion models (DiT) replace U-Net with a transformer, but the multi-scale processing idea persists.]*
- **Hourglass networks**: similar but skip connections **add** processed features instead of concatenating
	- Stacked hourglass = series of encoder-decoders for iterative refinement
	- Used for human pose estimation (predict heatmap per joint)

### 11.6 Why Do Nets with Residual Connections Perform So Well?

- **Not just depth**: shallower, wider ResNets sometimes outperform deeper, narrower ones at same param count
- **Evidence against "depth is everything"**: gradients don't propagate through very long paths effectively
	- In a 54-block ResNet, almost all gradient updates come from paths of length 5-17 (only 0.45% of total paths)
	- Very deep ResNet may act more like an **ensemble of moderately deep networks** than a truly deep network
- **Loss surface smoothness**: ResNet loss surfaces near minima are smoother and more predictable than equivalent nets without skip connections
	- Smoother surface -> easier optimization -> better generalization
	- *[LLM relevance: this "implicit ensemble of shallow networks" view is relevant to how we think about transformer depth. There's evidence that not all layers in a trained LLM are equally important -- some can be pruned with minimal performance loss (layer pruning / "depth pruning"). This connects to the finding that effective path lengths in ResNets are much shorter than the total depth.]*

### 11.7 Key Results from Notes Section

- **L2 regularization behaves differently** in residual vs vanilla nets
	- Vanilla: encourages constant output (bias-only)
	- Residual: encourages identity (residual block -> zero) + bias
	- *[LLM relevance: weight decay in transformers pushes residual block contributions toward zero, effectively simplifying the network. This is connected to why weight decay acts as a regularizer differently in residual architectures.]*
- **Regularization methods for residual nets**: ResDrop, stochastic depth, RandomDrop
	- **Stochastic depth**: randomly drop entire residual blocks during training
	- At test time, scale block output by survival probability
	- Essentially dropout applied to whole blocks instead of individual units
	- *[LLM relevance: **stochastic depth is actively used in training modern LLMs and ViTs**. It's a standard regularization technique in large-scale training. The idea generalizes -- you can also drop attention heads, drop layers, or use progressive training where you start with fewer layers and add more.]*

---

## Summary: What to Take Away for LLM Work

**Concepts that are load-bearing in modern LLM research:**
1. **Residual connections** -- the entire transformer is built on this. The "residual stream" is the core abstraction for mechanistic interpretability.
2. **Normalization** -- specifically LayerNorm/RMSNorm. Understand why it helps (smoother loss surface, stable variance) even if BatchNorm itself isn't used in LLMs.
3. **Bottleneck / reduce-expand pattern** -- shows up in FFN blocks, attention (project to lower-dim Q/K/V then back up), LoRA, MoE routing.
4. **1x1 convolutions as pointwise transforms** -- this IS the transformer FFN, just applied to sequence positions instead of spatial positions.
5. **Encoder-decoder + skip connections (U-Net)** -- backbone of diffusion models, relevant to any multimodal work.
6. **Stochastic depth** -- standard regularization in large-scale training.
7. **Shattered gradients / loss surface smoothness** -- conceptual foundation for understanding why depth requires architectural support.

**Concepts that are good background but less directly actionable:**
- Convolution mechanics (stride, padding, dilation, pooling) -- know them, but you probably won't design new conv architectures
- AlexNet/VGG specifics -- historical context for the "depth" narrative
- DenseNet -- interesting alternative that didn't become dominant
- Object detection / segmentation specifics -- unless you're in vision
