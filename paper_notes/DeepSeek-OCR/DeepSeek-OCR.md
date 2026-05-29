# DeepSeek-OCR: Contexts Optical Compression

> Wei, Sun, Li (DeepSeek-AI), arXiv:2510.18234, Oct 2025. Code: `github.com/deepseek-ai/DeepSeek-OCR`.

---

## TL;DR / Core Thesis

- **Big idea**: use the **vision modality as a compression medium for *text*** — not for images, for text.
	- Render a long text/document to a **2D image**, encode it with a **small number of vision tokens**, then have an LLM **decode the text back out** (OCR = the decompression).
	- **"A picture is worth a thousand words"** made literal: how few vision tokens are needed to losslessly reconstruct $N$ text tokens?
- **Why it matters**: LLM attention is $O(n^2)$ in sequence length. If 1 image of rendered text = far fewer tokens than the raw text, you get cheap long-context.
	- OCR is the ideal **proof-of-concept testbed** because it's a natural compression↔decompression mapping with **quantitative** ground truth (edit distance / precision).
- **Headline compression results** (Fox benchmark, English docs 600-1300 text tokens):
	- **< 10× compression → ~97% OCR precision** (near-lossless).
	- **~20× compression → ~60% precision** (degrades but usable).
- **Practical SOTA**: on **OmniDocBench**, beats **GOT-OCR2.0** (256 tok/page) using only **100 vision tokens**, and beats **MinerU2.0** (6000+ tok/page) with **< 800 vision tokens**.
- **Throughput**: **200k+ pages/day on a single A100-40G**; **33M pages/day on 20 nodes** (8×A100-40G each) for LLM/VLM pretraining-data generation.
- **Speculative payoff**: a model of **LLM memory / forgetting** — render old context to images, progressively downscale older images → fewer tokens + blurrier text = biological-style memory decay.

---

## Architecture

Two parts: **DeepEncoder** (vision tokenizer + compressor) → **DeepSeek3B-MoE** decoder (text generator).

```
Image → [ SAM-base 80M (window attn) ] → [ Conv 16× downsample ] → [ CLIP-large 300M (global attn) ] → vision tokens → [ DeepSeek3B-MoE-A570M decoder ] → text
        \________ local/perception ________/                       \____ knowledge ____/
                            DeepEncoder (~380M)
```

### DeepEncoder (~380M params)

Design requirements current open encoders fail to *jointly* satisfy: (1) high-res capable; (2) **low activation memory** at high res; (3) **few output vision tokens**; (4) multi-resolution; (5) moderate param count.

- **Serial (not parallel) composition of two pretrained encoders bridged by a conv compressor**:
	1. **SAM-base** [Kirillov SAM], patch-16, ~80M — **perception**, dominated by **window attention**.
		- Processes the *full* high token count (e.g. 4096 tokens for 1024×1024) but window attention keeps **activation low**, and it's only 80M.
	2. **16× conv compressor** (borrowed from **Vary**): **2-layer conv**, kernel 3, stride 2, pad 1, channels **256 → 1024**. Each layer halves H,W → **÷16 total** token count.
		- e.g. 1024×1024 → 4096 patch tokens → **256 tokens** before global attention.
	3. **CLIP-large** ViT, ~300M — **knowledge / dense global attention**.
		- **First patch-embed layer removed** (input is already tokens from the compressor, not raw pixels).
- **Key trick — order matters**: window-attention SAM eats the *large* token count cheaply; compressor cuts tokens **16×**; only then does the *expensive* dense-global-attention CLIP run → activation memory stays controllable.
	- Contrast with the 3 typical VLM encoder families (all flawed):
		- **Dual-tower (Vary)**: dual preprocessing, hard PP parallelism, hard deploy.
		- **Tile-based (InternVL2.0)**: low native res → over-fragmentation → too many tokens.
		- **Adaptive/NaViT (Qwen2-VL)**: huge activations, very long seq lengths, slow.

### MoE Decoder — DeepSeek3B-MoE-A570M

- **DeepSeekMoE** 3B; **inference activates 6 of 64 routed experts + 2 shared experts → ~570M activated params**.
	- Sweet spot: **3B expressiveness, ~500M-model inference speed**. Good for "domain-centric" (OCR) VLM.
- **Decoder = learned nonlinear decompressor**:
$$f_{\text{dec}}: \mathbb{R}^{n\times d_{\text{latent}}} \to \mathbb{R}^{N\times d_{\text{text}}}, \quad \hat{\mathbf{X}} = f_{\text{dec}}(\mathbf{Z}), \quad n \le N$$
	- $\mathbf{Z}$ = compressed vision tokens; $\hat{\mathbf{X}}$ = reconstructed text; $n$ vision tokens ≤ $N$ text tokens (the compression).
	- Conjecture: full LLMs with the right pretraining would acquire this decode ability *more* naturally.

---

## Multi-Resolution Modes

Why: to *vary the compression ratio* (= vary vision-token count) for a fixed text length and probe the limit. Achieved by **dynamic interpolation of positional encodings**; all modes trained jointly into one model.

### Native resolution (single image)

| Mode  | Resolution | Vision Tokens | Process |
|-------|-----------|---------------|---------|
| **Tiny**  | 512×512   | **64**  | resize |
| **Small** | 640×640   | **100** | resize |
| **Base**  | 1024×1024 | **256** | **padding** |
| **Large** | 1280×1280 | **400** | **padding** |

- **Tiny/Small** small enough → **direct resize** (no aspect-ratio preservation needed).
- **Base/Large** → **pad** to preserve aspect ratio. After padding, *valid* tokens < actual tokens:
$$N_{\text{valid}} = \big\lceil N_{\text{actual}} \times \big[\, 1 - \tfrac{\max(w,h) - \min(w,h)}{\max(w,h)} \,\big] \big\rceil$$

### Dynamic resolution (tiles + global)

- **Gundam** = $n$ **local tiles** (640×640) **+ 1 global view** (1024×1024). Tiling follows InternVL2.0.
	- Vision tokens = $n \times 100 + 256$, with **$n \in [2,9]$** (tile count capped to avoid fragmentation; native res is large so docs don't shatter).
	- $n=0$ (w,h both < 640) → degrades to Base.
- **Gundam-Master** = $n$ × **1024×1024 local** + **1280×1280 global** → $n\times256 + 400$ tokens.
	- Obtained by **continued training** on the finished DeepSeek-OCR (its res is too large to co-train without slowing everything; done separately for load balancing).
- Tiling = "secondary window attention" → reduces activation for ultra-high-res inputs (e.g. newspapers).

---

## Compression-Ratio vs OCR-Accuracy (Fox benchmark)

Setup: English Fox docs with 600-1300 text tokens (100 pages); tokenized with DeepSeek-OCR's ~129k-vocab tokenizer; prompt `<image>\nFree OCR.`. Tested in **Tiny (64)** and **Small (100)** modes only.

| Text Tokens | **64 vis tok** Precision / Compression | **100 vis tok** Precision / Compression |
|-------------|----------------------------------------|------------------------------------------|
| 600-700   | 96.5% / 10.5× | 98.5% / 6.7× |
| 700-800   | 93.8% / 11.8× | 97.3% / 7.5× |
| 800-900   | 83.8% / 13.2× | 96.8% / 8.5× |
| 900-1000  | 85.9% / 15.1× | 96.8% / 9.7× |
| 1000-1100 | 79.3% / 16.5× | 91.5% / 10.6× |
| 1100-1200 | 76.4% / 17.7× | 89.8% / 11.3× |
| 1200-1300 | 59.1% / 19.7× | 87.1% / 12.6× |

- **≤ 10× → ~97%** (near-lossless). **~20× → ~60%**.
- **Two suspected causes of >10× degradation**:
	1. Long-doc **layout** gets complex.
	2. Text **blurs** at 512/640 res. → fixable by (a) rendering text onto a single clean layout page, (b) higher res; and conceptually this blur *is* the desired "forgetting" feature.

---

## Training Data ("Data Engine")

Mix: **OCR 70% / general-vision 20% / text-only 10%**. Seq length **8192**.

- **OCR 1.0 (traditional OCR)**:
	- **30M PDF pages, ~100 languages** (Chinese+English ~25M, others ~5M).
		- **Coarse labels**: text extracted directly with `fitz`.
		- **Fine labels**: 2M Chinese + 2M English, **interleaved layout+text** (box coords + label precede each text block; all coords normalized to **1000 bins**). Built with PP-DocLayout + MinerU + GOT-OCR2.0.
		- Minority langs: train a GOT-OCR2.0 on `fitz` small-patch data, then **model flywheel** → 600K samples.
		- Coarse vs fine selected via **different prompts**.
	- **3M Word docs** → high-quality image-text without layout (helps formulas + HTML tables).
	- **Scene OCR**: LAION + Wukong, PaddleOCR-labeled, 10M Chinese + 10M English.
- **OCR 2.0 (parsing complex synthetic images)** (à la GOT-OCR2.0):
	- **Charts** → render 10M (pyecharts/matplotlib); task = **image → HTML table** (saves tokens vs OneChart dict).
	- **Chemical formulas** → SMILES (PubChem) rendered via RDKit → **5M** pairs.
	- **Plane geometry** → **1M**, Slow-Perception style (perception-ruler size 4 per line segment), translation-invariant augmentation.
- **General vision (20%)**: caption/detection/grounding (DeepSeek-VL2 style). Only to **preserve a general-vision interface** — not a general VLM.
- **Text-only (10%)**: in-house, all to 8192 tokens, to keep language ability.

### Training Pipeline

- **Two stages**: (a) train **DeepEncoder** standalone; (b) train full **DeepSeek-OCR**.
- **(a) DeepEncoder**: next-token prediction w/ a compact LM (à la Vary). OCR 1.0+2.0 + 100M LAION. **2 epochs, bs 1280, AdamW, cosine, lr 5e-5, seq len 4096**.
- **(b) Full model**: on **HAI-LLM** platform, **4-way pipeline parallelism (PP)**:
	- **PP0**: SAM + compressor = "vision tokenizer", **frozen**.
	- **PP1**: CLIP part as input-embedding layer, **trainable**.
	- **PP2/PP3**: 6 + 6 of the 12 MoE decoder layers.
	- **20 nodes (8×A100-40G), DP=40, global bs 640, AdamW, step scheduler, lr 3e-5.**
	- Speed: **90B tokens/day** (text-only), **70B tokens/day** (multimodal).

---

## OmniDocBench Results (vs SOTA)

Metric = **edit distance, lower = better**. "Tokens" = avg vision tokens/page; DeepSeek values in parens = *valid* tokens.

| Model | Tokens | EN overall | ZH overall |
|-------|--------|-----------|-----------|
| GOT-OCR2.0 | 256 | 0.287 | 0.411 |
| MinerU2.0 | 6790 | 0.133 | 0.238 |
| dots.ocr⁺200dpi | 5545 | 0.125 | 0.160 |
| **DeepSeek Tiny** | **64** | 0.386 | 0.361 |
| **DeepSeek Small** | **100** | 0.221 | 0.284 |
| **DeepSeek Base** | 256 (182) | 0.137 | 0.240 |
| **DeepSeek Large** | 400 (285) | 0.138 | 0.208 |
| **DeepSeek Gundam** | 795 | 0.127 | 0.181 |
| **DeepSeek Gundam-M⁺200dpi** | 1853 | **0.123** | **0.157** |

- **100 tokens beats GOT-OCR2.0** (256 tok); **<800 tokens beats MinerU2.0** (~7000 tok). Higher compression ⇒ higher research ceiling.
- **Per-doc-type (Table 4)**: token need scales with text density.
	- **Slides** OK at **64**; **books/reports** OK at **100** (most have <1000 text tokens → <10× compression).
	- **Newspapers** need **Gundam / Gundam-M** (4-5k text tokens/page, far beyond 10×).

### Extra capabilities (qualitative)

- **"Deep parsing"**: one unified prompt → secondary model calls to parse charts (→HTML table), chemical (→SMILES), geometry, dense natural-image captions within a doc.
- **Multilingual** (~100 langs, incl. Arabic/Sinhala), layout & non-layout via prompt.
- **General vision** retained: description, detection, grounding. **No SFT stage** → not a chatbot; some abilities need completion-style prompts.

---

## Implications: LLM Context Compression & Memory Forgetting

- **Multi-turn dialogue**: optically compress history beyond $k$ rounds → ~10× efficiency, *for free* (multimodal systems already carry a vision encoder).
- **Biological forgetting analogy** (Fig 13): three parallel decay axes
	- **Memory** ↓ over **time** (just-happened → 1 year).
	- **Vision** ↓ over **distance** (10cm → 20m).
	- **Text** ↓ over **resolution mode** (Text-token → Gundam → Large → Base → Small → Tiny).
	- Recent context = high-res / high-fidelity; older context = progressively **downscaled images** → fewer tokens, blurrier → **textual forgetting**. Path toward "theoretically unlimited context."
- **Limitations / caveats**:
	- Honest **proof-of-concept**; **OCR alone insufficient** to validate true optical context compression.
	- Future: **digital-optical interleaved pretraining**, **needle-in-a-haystack** tests.
	- >10× accuracy drops with complex layouts and small-res blur.
	- Gundam-Master needs separate continued training (load balancing).
	- No SFT → limited instruction following.

---

## Implementation (codebase analysis)

Repo `DeepSeek-OCR-master/` has two inference backends: **`DeepSeek-OCR-vllm/`** (vLLM, batch/PDF/eval) and **`DeepSeek-OCR-hf/`** (transformers `.infer()`). Env: cuda11.8 + torch2.6.0, vllm 0.8.5, flash-attn 2.7.3.

### Encoder components

- **`deepencoder/sam_vary_sdpa.py` — SAM + the 16× compressor**
	- `ImageEncoderViT` = SAM-base ViT: `embed_dim=768, depth=12, num_heads=12, patch_size=16, window_size=14, global_attn_indexes=[2,5,8,11]`, built by `build_sam_vit_b()` → `_build_sam(...)`.
		- Window attention everywhere except the 4 global-attn block indices; rel-pos embeddings (`use_rel_pos=True`), interpolated for arbitrary res.
		- `Block.forward`: `window_partition` → `Attention` (uses `F.scaled_dot_product_attention`) → `window_unpartition`.
	- **The 16× compressor lives *inside* SAM's `forward`** as the `neck` + two stride-2 convs:
		```python
		self.neck = nn.Sequential(Conv2d(768,256,1), LayerNorm2d, Conv2d(256,256,3,pad=1), LayerNorm2d)
		self.net_2 = nn.Conv2d(256, 512,  3, stride=2, padding=1)   # ÷2
		self.net_3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)   # ÷2  → ÷4 area total per layer pair... 
		# forward: x→patch_embed→blocks→neck→net_2→net_3  (returns 1024-ch, H/4×W/4 spatial = 16× fewer tokens vs 1024 patch grid)
		```
		- For 1024×1024: patch grid 64×64 = 4096 → after net_2,net_3 → **16×16 = 256** tokens, channels 1024.
- **`deepencoder/clip_sdpa.py` — CLIP-large (`build_clip_l()`)**
	- `VitModel` wrapping `CLIPVisionEmbeddings` + `NoTPTransformer` (24 layers, hidden 1024, 16 heads, ffn 4096, patch 14, quick-GELU).
	- **Patch-embed bypassed**: `CLIPVisionEmbeddings.forward(pixel_values, patch_embeds)` — when `patch_embeds` is passed (the SAM/compressor output) it skips its own `patch_embedding` conv and just flattens + prepends the CLS token + adds interpolated abs-pos (`get_abs_pos`, bicubic).
	- So CLIP is called as `self.vision_model(patches, local_features_1)` — image arg is only for batch size; the real input is the compressed tokens.
- **`deepencoder/build_linear.py` — `MlpProjector`**
	- Many projector types; DeepSeek-OCR instantiates the simplest: `MlpProjector(projector_type="linear", input_dim=2048, n_embed=1280)`.
	- `input_dim=2048` because features are **concatenated**: CLIP output (1024) ⊕ SAM/compressor output (1024). `n_embed=1280` = decoder embed dim.
	- (Other types e.g. `downsample_mlp_gelu` implement the `F.unfold` "4-to-1 concat" downsample-by-`downsample_ratio` pattern; not used in the linear default path here.)

### Main model wiring — `deepseek_ocr.py` (`DeepseekOCRForCausalLM`, vLLM)

- `__init__`: `self.sam_model = build_sam_vit_b()`, `self.vision_model = build_clip_l()`, `self.projector = MlpProjector(linear, 2048→1280)`, plus learnable `image_newline` and `view_seperator` params (`tile_tag="2D"`). Decoder = `init_vllm_registered_model(...)` over `text_config` (DeepseekV2/V3/`DeepseekForCausalLM` chosen by `topk_method`/`use_mla`).
- **Encoder forward path** in `_pixel_values_to_embedding`:
	```python
	local_features_1  = self.sam_model(patches)              # SAM+compressor on local tiles
	local_features_2  = self.vision_model(patches, local_features_1)   # CLIP on compressed tokens
	local_features    = cat(local_features_2[:,1:], local_features_1.flatten(2).permute(0,2,1), dim=-1)  # drop CLS, concat → 2048-d
	local_features    = self.projector(local_features)       # → 1280-d decoder space
	# same for global_features on the padded global view (image_ori)
	```
	- **Feature fusion = channel concat of CLIP (high-level) + SAM-compressor (low-level)** then linear proj. SAM features added via `flatten(2).permute(0,2,1)` (B,C,H,W → B,HW,C).
	- **2D layout reassembly**: global features reshaped to `h×w`, an `image_newline` token appended per row; local tiles arranged `(height_crop_num, width_crop_num, h2, w2)` → row-major grid with newlines; final sequence = `[local_features, global_features, view_seperator]`. No-crop path = `[global_features, view_seperator]`.
- **Decoder integration**: `get_multimodal_embeddings` → `merge_multimodal_embeddings` splices vision embeds into the `<image>`-token positions of the text embedding sequence; then standard causal LM forward.
- **Weight loading** (`load_weights`): names containing `sam_model/vision_model/projector/image_newline/view_seperator` are stripped of `model.`; everything else prefixed `language.` → mapped to `language_model.`.

### Resolution config in code

- **`config.py`** drives everything via 3 globals:
	```python
	BASE_SIZE = 1024   # global view size
	IMAGE_SIZE = 640   # tile size
	CROP_MODE = True   # → this combo = Gundam mode
	MIN_CROPS=2; MAX_CROPS=6   # n tiles range (max 9)
	```
	- Mode cheat-sheet (from comments): Tiny `512/512/False`, Small `640/640/False`, Base `1024/1024/False`, Large `1280/1280/False`, **Gundam `1024/640/True`**.
- **Token-count math** (`DeepseekOCRProcessingInfo.get_num_image_tokens`, mirrors `tokenize_with_images`):
	```python
	patch_size=16; downsample_ratio=4
	h = ceil((base_size//16)/4)          # global grid side, e.g. 1024→16
	h2 = ceil((image_size//16)/4)        # tile grid side, e.g. 640→10
	global_views_tokens = h * (w + 1)    # +1 col = image_newline per row
	local_views_tokens  = (n_h*h2) * (n_w*w2 + 1)   # if cropping
	total = global + local + 1           # +1 = view_seperator
	```
	- So **Base** global = 16×(16+1)=272 raw (16×16=256 "valid"); **Small/Tiny** = 10×(10+1)=110 / 8×9=72 raw with the newline tokens (paper's 100/64 = the valid grid).
- **Tiling**: `count_tiles` / `dynamic_preprocess` (`image_process.py`) pick best aspect ratio from `target_ratios` (all (i,j) with `MIN_CROPS ≤ i*j ≤ MAX_CROPS`), `find_closest_aspect_ratio`, then crop into `image_size`-square tiles. Images ≤640×640 → `crop_ratio=[1,1]` (no tiling, degrades to Base).
- **Preprocessing** (`tokenize_with_images`): global view = `ImageOps.pad` to `base_size` (pad color = mean); normalize mean/std=(0.5,0.5,0.5); local tiles transformed individually. Builds the interleaved `<image>`-token id sequence with per-row newline tokens + view separator to match encoder output layout.

### Inference / generation

- **vLLM** (`run_dpsk_ocr_image.py`): registers `DeepseekOCRForCausalLM`, `AsyncLLMEngine`, `max_model_len=8192`, `gpu_memory_utilization=0.75`, `tensor_parallel_size=1`. Streaming generate.
	- `SamplingParams(temperature=0.0, max_tokens=8192, skip_special_tokens=False)` + **`NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist={128821,128822})`** — kills repetition loops, whitelisting `<td>/</td>` so tables can repeat (`SKIP_REPEAT` in config). Upstream vLLM uses `NGramPerReqLogitsProcessor`.
	- **Grounding postproc**: regex `<\|ref\|>...<\|/ref\|><\|det\|>...<\|/det\|>` extracts labels + boxes; coords are `/999*dim` (normalized 1000-bin → pixels); `draw_bounding_boxes` overlays boxes, crops out `image`-labeled regions to `images/*.jpg`, rewrites markdown. Geometry output (`line_type` present) re-plotted via matplotlib.
- **Transformers** (`DeepSeek-OCR-hf/run_dpsk_ocr.py`): one-liner
	```python
	model.infer(tokenizer, prompt, image_file, output_path,
	            base_size=1024, image_size=640, crop_mode=True,
	            save_results=True, test_compress=True)
	```
	(`test_compress=True` prints the achieved vision-token compression.)
- **Prompts** (`config.py` / README):
	- doc→markdown: `<image>\n<|grounding|>Convert the document to markdown.`
	- free OCR (no layout): `<image>\nFree OCR.`
	- figure: `<image>\nParse the figure.` · general: `<image>\nDescribe this image in detail.` · locate: `<image>\nLocate <|ref|>xxx<|/ref|> in the image.`

---

## Related Notes

- **[OCR-Memory](../OCR-Memory/OCR-Memory.md)** — Direct application of this paper. Uses a **fine-tuned DeepSeek-OCR (3B) backbone** as the optical encoder for an LLM agent's memory: history rendered to annotated images, retrieved by "locate-and-transcribe." Inherits DeepSeek-OCR's >10× vision-token compression; adds Set-of-Mark anchors + multi-resolution "optical forgetting."
- **[LensVLM](../LensVLM/LensVLM.md)** — Builds on / cites DeepSeek-OCR but **targets its key weakness**: DeepSeek-OCR's accuracy collapses past ~10–20× because characters shrink below the encoder's effective resolution (one-way compression). LensVLM keeps the compressed view for scanning but adds a reversible `EXPAND` tool to selectively decompress relevant regions → pushes past the legibility ceiling.
- **[MM-SeR](../MM-SeR/MM-SeR.md)** — Loosely adjacent (lightweight VLM, not text-compression). Shares the theme of doing more with small vision-language models; MM-SeR's glance→sharpen refinement is a different axis (perceptual grounding for captioning) than optical *context* compression.
- **[Knowing When to Stop](../Knowing_When_to_Stop/Knowing_When_to_Stop.md)** — Contrast: also reduces token budget for long context, but by **early-exiting** (stop reading once info is sufficient) rather than **compressing** the context into a denser medium. Orthogonal, composable idea.
