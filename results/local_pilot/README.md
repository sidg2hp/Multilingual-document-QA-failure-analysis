# Local Pilot Results (RTX 3050)

**These are pilot artifacts, not paper evidence.** They were produced on the
author's laptop before PARAM Ganga access, without human annotators, without a
VLM that fits the paper protocol, and with a weak English-only OCR engine. Read
the limitations before citing any number.

## What ran

| Component | Value |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| OCR | Windows.Media.Ocr, en-GB/en-US recognizers only |
| Reader | Qwen2.5-1.5B-Instruct, Ollama `qwen2.5:1.5b`, Q4_K_M |
| Reader digest | `ae57435a742bf355973ad0e07db8247ad9ea6c2da5aa945d7dace1891fc67857` |
| Ollama | 0.5.7 (local server, workspace model store) |
| Real data | FUNSD test, local rows/images cache; 50 documents available |
| Synthetic data | 10 rendered invoice pages, 40 field questions |
| VLM | **not run**: Ollama 0.5.7 rejects `qwen2.5vl` (newer runtime required); the LLaVA-Phi3 projector download did not complete |

## 1. OCR-only degradation pilot (real FUNSD)

20 FUNSD documents, 26 conditions each (clean + 5 families × 5 levels), 520 OCR
calls. Gold is the FUNSD word sequence ordered by estimated line then x.

| Condition | Mean CER [95% CI] | Mean WER |
| :--- | :--- | :--- |
| clean | **0.597 [0.543, 0.645]** | 0.730 |
| Gaussian blur L1 → L5 | 0.652 → 0.965 → 0.997 → 1.000 → 1.000 | 0.785 → 1.000 |
| Resolution downscale L1 → L5 | 0.804 → 0.923 → 0.984 → 0.998 → 1.000 | 0.878 → 1.000 |
| JPEG L1 → L5 | 0.613 → 0.623 → 0.677 → 0.794 → 0.896 | 0.749 → 0.951 |
| Salt & pepper L1 → L5 | 0.612 → 0.645 → 0.695 → 0.780 → 0.889 | 0.741 → 0.956 |
| Perspective skew L1 → L5 | 0.691 → 0.689 → 0.704 → 0.696 → 0.685 | 0.814 → 0.801 |

**Interpretation.** The clean CER is high because Windows OCR misses a large
fraction of words on these low-quality fax forms (OCR output ~743 chars vs ~1345
gold chars on document 0). Blur and downscale destroy it completely; JPEG and
salt-and-pepper degrade gradually; the current perspective operator is nearly
flat, so it is not a useful severity axis as implemented. This is a **perturbation
calibration check**, not a DocQA result.

Files: `ocr_pilot_report.json`, `ocr_pilot_summary.csv`, `ocr_raw.jsonl`.

## 2. Controlled synthetic cascade (pipeline validation, not evidence)

10 rendered invoice pages, 40 field questions, Windows OCR → Qwen2.5-1.5B.
Every sample is marked synthetic; the report is `reportable: false`.

| Condition | Mean ANLS [95% CI] | Failure modes |
| :--- | :--- | :--- |
| C0 clean end-to-end | **0.733 [0.733, 0.733]** | 75% correct, 25% confident hallucination |
| C2 oracle crop | **1.000 [1.000, 1.000]** | 100% correct |
| C3 oracle text | **1.000 [1.000, 1.000]** | 100% correct |
| C3b shuffled text | **1.000 [1.000, 1.000]** | 100% correct |
| Cδ severe blur | **0.000** | all abstained |

**Interpretation.** The harness behaves as designed on clean, known text: the
reader answers from gold text and the gold crop, and OCR clean performance is
below the oracle ceiling. Shuffling whole lines does **not** break label–value
pairs, so C3b is a weak control in this form and must be redesigned (e.g.,
sentence/word shuffle) before the paper.

Files: `synthetic/tables.md`, `synthetic/analysis.json`, `synthetic/report.json`,
`synthetic/predictions.jsonl`.

## 3. Real FUNSD cascade (negative pilot)

60 questions from 20 FUNSD documents, derived from entity adjacency and filtered
to unique label/answer pairs, Windows OCR → Qwen2.5-1.5B.

| Condition | Mean ANLS [95% CI] |
| :--- | :--- |
| C0 clean end-to-end | 0.073 [0.015, 0.144] |
| C2 oracle crop | 0.091 [0.035, 0.150] |
| C3 oracle text | 0.077 [0.020, 0.148] |
| C3b shuffled text | 0.103 [0.042, 0.162] |
| Cδ severe blur | 0.000 |

**Interpretation.** This is a **negative, low-confidence pilot**, not evidence
that the method fails. The FUNSD questions are derived by a spatial heuristic,
many field labels are ambiguous (multiple fax/phone numbers), Windows OCR misses
a large fraction of characters, and Qwen-1.5B is far below the paper's 3B/7B
targets. The high confident-hallucination share (~65%) is therefore a statement
about this weak stack, not about document QA systems in general. It does show
that the report pipeline, clustering, and failure-mode accounting work on real
data.

Files: `funsd/tables.md`, `funsd/analysis.json`, `funsd/report.json`,
`funsd/predictions.jsonl`.

## What cannot be concluded

- No read–reason dissociation: no VLM was run.
- No OCR-vs-VLM attribution: no paired VLM.
- No Indic result: Windows OCR has no Indic recognizers.
- No paper claim about FUNSD difficulty or Qwen: the questions, OCR, and reader
  are all pilot-grade.
- No annotation-dependent HAR validation.

## What PARAM Ganga unlocks

1. Qwen2.5-VL-3B/7B and InternVL/Patram under FP16, so C0/C1/C2/C3b are real.
2. Tesseract/PaddleOCR/Surya cascades, so the OCR arm is a real OCR arm.
3. 3B/7B text readers, so oracle-text results are not floor-limited by a 1.5B
   model.
4. Enough throughput for a 600-item pre-registered run and 3 seeds.

Reproduce the local artifacts with the commands in
[`../../README.md`](../../README.md) and [`../../REPRODUCE.md`](../../REPRODUCE.md).
