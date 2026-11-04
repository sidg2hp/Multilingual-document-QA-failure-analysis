# Research Protocol & Workshop Acceptance Strategy

## 1. Realistic Literature Check & Workshop Viability (2026 Landscape)

### Existing Prior Work You Must Cite and Differentiate Against
1. **Patram-Bench (Srinivasan et al., 2026)**: Evaluates 29 VLMs across 52,607 Indian documents (23 languages, 52 tasks) and introduces DUCLI (cross-lingual degradation) and SRS (script robustness).
2. **IndicVisionBench (Faraz et al., ICLR 2026)**: Benchmarks 10 Indic languages across VQA, MMT, and 876 Wikisource Level-4 verified OCR pages.
3. **DISCO (Benkirane et al., 2026, arXiv:2603.23511)**: Compares OCR pipelines vs. VLMs across document types.
4. **Locating Failure in Multi-Page VRDU (Xu et al., 2026, arXiv:2608.07943)**: Applies interventional error attribution (representation, selection, reasoning) on English multi-page documents.
5. **Perceptual Hallucination in VLMs / DocHallu (Hwang et al., ACL Findings 2026)**: Shows that visual degradation in document images triggers perceptual hallucinations in VLMs, especially on numerical and entity spans.

### Honest Verdict on the Initial Idea
- **If executed as originally proposed** (*running Tesseract, PaddleOCR, and a small quantized Qwen-VL on Patram-Bench/IndicVisionBench with oracle text injection*):
  - **Acceptance probability at top workshops (ICDAR, ACL/EMNLP Document Intelligence, CVPR/ECCV VDU)**: **~35–45% (Borderline)**.
  - **Reviewer critique**: *"Oracle text only tests the text-LLM half of a cascaded pipeline; it does not isolate whether an OCR-free VLM failed due to layout localization, abugida character perception, or reasoning. Furthermore, running only 3B quantized models on a small subset looks like a hardware-constrained benchmark reproduction."*

---

## 2. Upgraded Research Formulation (High Acceptance Potential: ~75–85%)

To make this a **strong paper for ICDAR, ACL/EMNLP MRL or Document Intelligence Workshop, or ICLR/NeurIPS Workshops**, we upgrade the study in **three specific ways**:

### Upgrade A: 5-Condition Counterfactual Mediation Protocol (Works on OCR-Free VLMs!)
Instead of only comparing $C_0$ (Image) vs. $C_3$ (Oracle Text), we introduce two intermediate counterfactual interventions that probe **inside** OCR-free VLMs:
1. **$C_1$ (Self-OCR Probe — Isolating Read-Reason Dissociation)**:
   - Prompt the VLM to first transcribe the visible text ($T_{\text{VLM}}$), and then answer $Q$ from $T_{\text{VLM}}$.
   - **Finding to test**: Often, an OCR-free VLM *can* accurately transcribe an Indic/code-mixed span when asked to read ($C_1$ succeeds), yet fails when asked to answer the question directly ($C_0$ fails) — proving a **cross-modal attention/grounding bottleneck** rather than a vision encoder resolution limit.
2. **$C_2$ (Oracle Layout BBox Crop — Isolating $\Delta_{\text{lay}}$ from $\Delta_{\text{rec}}$)**:
   - Crop the image to the ground-truth answer bounding box (`answer_bbox`) with 8% padding.
   - The accuracy jump $\Delta_{\text{lay}} = \text{ANLS}(C_2) - \text{ANLS}(C_0)$ measures **Layout Clutter & Localization Failure** directly in the visual domain, while $\Delta_{\text{rec}} = \text{ANLS}(C_3) - \text{ANLS}(C_2)$ isolates pure **Script/Character Recognition Failure**.

### Upgrade B: "Abstention vs. Matra-Blind Confident Hallucination" in Indic Scripts
- Indic scripts are **abugidas** where vowel diacritics (*matras*, Unicode category `Mn`/`Mc`) and conjunct ligatures carry critical lexical and numerical distinctions.
- Under Gaussian blur, low DPI, or JPEG artifacts:
  - **Cascaded OCR pipelines** produce high-entropy character garbage, causing the downstream LLM to **abstain** (`UNANSWERABLE`).
  - **OCR-Free VLMs** bypass explicit character thresholding and rely on autoregressive language priors—leading to **Matra-Blind Confident Hallucinations** (guessing a fluent but factually wrong word or number).
- We measure **Matra Error Rate (MER)** alongside CER and track the **Hallucination-to-Abstention Ratio (HAR)** across perturbation severities.

### Upgrade C: Turning the RTX 3050 (6GB VRAM) Constraint into a Scientific Variable
- Rather than apologizing for using 4-bit quantization on an RTX 3050, we turn **Quantization-Induced Script Degradation (NF4 vs. INT8 vs. FP16)** into a formal research question:
  - *Does 4-bit NF4 weight quantization disproportionately degrade visual character recognition ($\Delta_{\text{rec}}$) on low-resource Indic abugidas and code-mixed documents compared to monolingual Latin documents?*
- On a 6GB RTX 3050:
  - `Qwen2.5-VL-3B-Instruct` fits in **NF4 (~2.6 GB VRAM)** and **INT8 (~3.8 GB VRAM)** locally.
  - `SmolVLM-2.2B-Instruct` fits in **NF4 (~1.8 GB)**, **INT8 (~2.5 GB)**, and **FP16 (~4.6 GB)** locally on your RTX 3050!

---

## 3. Step-by-Step Execution Plan on RTX 3050 (6GB VRAM)

### Phase 1: Curate the Balanced Counterfactual Evaluation Slice (Week 1)
Do **not** download all 52,607 Patram-Bench images. Use `src/data/datasets.py` in streaming mode to build a fixed, deterministic **600-document Golden Diagnostic Set**:
- **200 samples** from **IndicVisionBench** (`ocr` + `vqa_indic` paired pages across Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati).
- **300 samples** from **Patram-Bench** (150 monolingual Indic, 150 high-SMI Code-Mixed Indic+English forms/circulars/invoices with bounding boxes).
- **100 samples** from **FUNSD** (monolingual English structured forms with ground-truth entity bounding boxes).

### Phase 2: Generate Controlled Perturbations & Layout Crops (Week 1)
Run `src/perturbations/degrade.py` offline on CPU to generate deterministic `mild`, `moderate`, and `severe` variants for:
- `gaussian_blur` (targets matras)
- `resolution_downscale` (targets small form fonts)
- `jpeg_artifacts` (targets stroke boundaries)
- `perspective_skew` (targets multi-column/table reading order)

### Phase 3: Sequential Model Inference on RTX 3050 (Week 2–3, While Waiting for PARAM Ganga)
Run one model at a time (strictly evicting VRAM via `.unload()`):
1. `Tesseract 5 (hin+eng / ben+eng / ...)` $\to$ saves extracted text to JSONL first.
2. `PaddleOCR (PP-OCRv4)` / `Surya OCR` $\to$ saves extracted text to JSONL first.
3. Load `Qwen2.5-3B-Instruct (NF4)` once and batch-evaluate all saved OCR transcripts + $C_3$ Oracle Text transcripts.
4. Load `Qwen2.5-VL-3B-Instruct (NF4 & INT8)` and evaluate $C_0, C_\delta, C_1, C_2, C_3$.
5. Load `SmolVLM-2.2B-Instruct (NF4, INT8, FP16)` and evaluate $C_0, C_\delta, C_1, C_2, C_3$.
6. Train and validate the **Adaptive OCR-VLM Router (`src/routing/router.py`)** on CPU using the 7 image/script features (`src/routing/features.py`) and plot the Accuracy–Latency Pareto curve.

### Phase 4: Top-Tier Conference Scaling on IITR PARAM Ganga (2x Tesla V100 16GB)
Once your PARAM Ganga account is approved (`sbatch slurm/run_param_ganga.sh` with `configs/param_ganga_config.yaml`):
- **Critical Hardware Compatibility Note for Tesla V100 (`sm_70` Volta)**:
  - Tesla V100 GPUs **do not support native `bfloat16`** or **`FlashAttention-2`** (which require Ampere `sm_80+`). Always run with `torch_dtype="float16"`, `attn_implementation="sdpa"`, and `device_map="auto"` across the 2x V100 GPUs (32GB combined VRAM).
- **Critical Multilingual Model Selection (Avoiding Reviewer Rejection)**:
  - Do **not** use `LLaVA-1.5-7B` as a primary Indic baseline—its Llama-2 tokenizer lacks Indic script vocabulary (splitting Hindi characters into raw bytes) and its 336px CLIP encoder cannot resolve dense document text.
  - Instead, run **Indic-capable 7B–8B VLMs in full FP16** across the 2x V100s:
    1. `bharatgenai/patram-7b-instruct` (India's document foundation VLM)
    2. `Qwen/Qwen2.5-VL-7B-Instruct`
    3. `OpenGVLab/InternVL2_5-8B`
  - Pair OCR pipelines (`Tesseract 5`, `PaddleOCR v4`, `Surya OCR`) with `Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Meta-Llama-3.1-8B-Instruct` (which includes native 128K multilingual Indic tokens).
- **Full 3-Ablation Matrix on PARAM Ganga**:
  1. **5-Level Degradation Spectrum (`level_1` to `level_5`)**: Gaussian blur, Salt-and-Pepper noise, JPEG artifacts, Resolution downscale, and Perspective skew.
  2. **Code-Mixing Ratio (`SMI` bins)**: `[0.00–0.05]`, `[0.05–0.20]`, `[0.20–0.35]`, and `[0.35–0.50]`.
  3. **Prompting Intervention**: Direct QA ($C_0$) vs. Step-wise Self-OCR Chain-of-Thought ($C_1$).
