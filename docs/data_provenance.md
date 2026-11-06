# Data & Model Provenance

**Access date for every check below: 6 November 2026.**
This document records what the project actually depends on, what was verified,
and what remains unverifiable. It exists because the previous project notes
asserted dataset sizes, splits, and model capabilities without checking them; at
least one assertion (the Patram-Bench Hugging Face ID) is not usable, and one
(the role of IndicVisionBench) misdescribes the benchmark.

Anything in this file that says "unverified" must be resolved before a paper
submission. A citation to a dataset the authors cannot access is not a
reproducibility artifact.

---

## 1. Datasets

### 1.1 Patram-Bench — **access unresolved; primary dependency**

- **Paper:** "PATRAM-BENCH: A Multi-task, Multi-domain and Multi-lingual
  Benchmark for Indian Document Image Understanding", ICDAR 2026 (verified via
  the ICDAR 2026 program and a DBLP record).
- **Claimed scale (from the paper/project notes):** 52,607 documents, 23
  languages, 13 domains, 52 tasks, 2.4M QA pairs.
- **Repository ID used in this project:** `bharatgenai/Patram-Bench`.
- **Verification:** the Hugging Face datasets-server returns
  `401 / does not exist or is not accessible without authentication`; the HF
  dataset search API for `patram` returns `[]`; arXiv has no paper with this
  title. The model repository `bharatgenai/patram-7b-instruct` exists and is
  gated, which is a different artifact.
- **Status:** **not publicly accessible under the configured ID as of the access
  date.** The loader for Patram-Bench did not exist in the code at commit
  `8f18b0c`; the config only named the repo.
- **Action required:** contact the authors for the dataset/release terms; if a
  gated or local copy is obtained, record its revision/hash and license here.
  Until then, no Patram-Bench results may be reported and no Patram-Bench
  numbers may appear in configs as if runnable.
- **Contingency:** re-scope to public data (IndicVisionBench `ocr` split + FUNSD
  + a derived probe set), or make Patram-Bench an optional extension.

### 1.2 IndicVisionBench — **verified, but not a document-QA benchmark**

- **Paper:** Faraz et al., "IndicVisionBench: Benchmarking Cultural and
  Multilingual Understanding in VLMs", arXiv:2511.04727, ICLR 2026.
- **HF dataset:** `krutrim-ai-labs/IndicVisionBench`, revision
  `45269760b15938c6a345f74373f56c7a5ab7ecc8` (last modified 2026-03-11).
- **Verified configs and sizes (datasets-server):**

  | Config | Rows | Schema | What it is |
  | :--- | ---: | :--- | :--- |
  | `ocr` | 876 | `id, image, text, language, page_url` | Wikisource Level-4 verified page images + transcript. **No QA, no bboxes.** |
  | `vqa_indic` | 1,007 | `id, image, topic, State/UT, language, short_q1/a1, short_q2/a2, mcq, mcq_a, mcq_opt1..4, true_false_q/a, long_q/a, adversarial_q/a, source_url` | Cultural VQA over **natural images**, not documents. |
  | `vqa_en` | 4,117 | same as above (English) | Cultural VQA. |
  | `vqa_parallel` | 1,166 | same as above, parallel languages | Cultural VQA. |
  | `mmt` | 106 | `English, Hindi, Bengali, …` | Multimodal translation. |

- **Consequence:** `vqa_indic` is **not** document QA. It has no page transcript;
  `row["text"]` does not exist. A loader that treats it as DocQA and sets
  `oracle_text` to a question+answer string leaks the label into `C3`.
- **What is actually usable for this project:**
  - `ocr` split pages: real Indic document images with gold transcripts,
    suitable for OCR metrics (CER/WER/MER), degradation curves, and
    `C0`/`C1`/`C3` **if** questions are generated and `C3` uses the true
    transcript. No gold answer boxes; `C2` requires a detector-derived box or a
    different dataset.
  - `vqa_indic`: can be used as a **multilingual visual-reasoning control**, not
    as document QA. Do not mix it into DocQA claims.
- **License:** see `LICENSE.md` in the HF repository; record the exact terms in
  the paper's data statement. Website images have mixed Wikimedia licenses;
  `source_url` is provided per row.

### 1.3 FUNSD — **verified; English layout control only**

- **HF dataset:** `nielsr/funsd`, revision
  `7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5` (last modified 2025-06-23).
- **Splits:** train 149, test 50.
- **Schema:** `id: string`, `words: [string]`, `bboxes: [[int]]`, `ner_tags:
  [class]` with labels `O, B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER,
  I-ANSWER`, `image: Image`. No `linking` column in this version.
- **What it is:** scanned English forms with word boxes and entity tags.
- **Use in this project:** English/Latin control for the **full layout ladder**
  (`C2` crop and coord text are possible). It cannot support claims about Indic
  scripts. The original loader's token-index question is not a valid task; build
  `QUESTION`/`ANSWER` entity pairs instead.
- **License:** FUNSD is distributed for research use; check the original
  `guillaumejaume.github.io/FUNSD` terms and cite Jaume et al., 2019.

### 1.4 Optional public data to consider

These are not currently wired in; they are candidate replacements if
Patram-Bench access fails. Verify each schema and license before use.

| Dataset | Why it might help | Caveat |
| :--- | :--- | :--- |
| DocVQA / InfographicVQA | Standard DocQA with boxes, strong baselines | English only |
| XFUND | Multilingual form understanding with boxes | No Indic language |
| IIIT-AR-13K | Indic document images | Classification labels, not QA |
| IndicSTR / MLT-2019 | Indic scene text | Scene text, not documents |
| DocHallu (IB99/DocHallu) | Original–damaged document pairs | English; hallucination focus |

---

## 2. Models

| Model ID | Status | Revision | License | Loading notes |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Public, verified | `66285546d2b821cf421d4f5eb2576359d3770cd3` | Qwen research license | `Qwen2_5_VLForConditionalGeneration`; use `process_vision_info`; dynamic pixels. |
| `Qwen/Qwen2.5-3B-Instruct` | Public (assumed; verify revision at run time) | pin at run time | Qwen research license | Use `apply_chat_template`; quantize with bitsandbytes. |
| `bharatgenai/patram-7b-instruct` | **Gated** (auto-approval), custom code | `6a31983daa739b18ba029c767c65d3f68d9be06c` | Apache-2.0 | `trust_remote_code=True`, custom `Patram1_1ForCausalLM`, non-ChatML `User:/Assistant:` template. The current `AutoModelForImageTextToText` path will not work. |
| `HuggingFaceTB/SmolVLM-Instruct` | **Stale ID**, verify | — | — | Use an exact SmolVLM/SmolVLM2 checkpoint and test processor behavior; do not assume Qwen-style `images=[image]`. |
| `OpenGVLab/InternVL2_5-8B` | Public, but custom preprocessing | pin at run time | verify | Requires InternVL-specific image tiling/`trust_remote_code`; not supported by the current runner. |

**Rule:** every model reported in the paper must have an exact revision, a
license statement, a prompt template, a quantization configuration, and a
hardware/driver record. "Qwen2.5-VL-3B" is not a reproducible model identity.

---

## 3. Literature (verified)

Use these full references, not the summaries in `PROJECT_MASTER_NOTES.txt`.

1. K. Benkirane, D. Goldwater, M. Asenov, A. Ghodsi. *DISCO: Document
   Intelligence Suite for COmparative Evaluation.* arXiv:2603.23511, 2026.
   ICLR 2026 Workshop on Multimodal Intelligence.
2. L. Xu, Y. Ding, Z. Xu, D. Y. Su, D. Liu, S. Luo, Y. Peng, W. Liu. *Locating
   Failure in Multi-Page Visually Rich Document Understanding: An Empirical
   Attribution.* arXiv:2608.07943, 2026.
3. T. Hwang, I. Heo, S. J. Lee, S. Jung. *Perceptual Hallucination in
   Vision–Language Models: Definition, Analysis and Verification.* Findings of
   ACL 2026, pp. 24710–24725.
4. A. Faraz, Akash, S. Khan, R. Kolla, A. Patidar, S. Goswami, A. Ravi, C.
   Khatri, S. Agarwal. *IndicVisionBench: Benchmarking Cultural and Multilingual
   Understanding in VLMs.* arXiv:2511.04727, ICLR 2026.
5. *PATRAM-BENCH: A Multi-task, Multi-domain and Multi-lingual Benchmark for
   Indian Document Image Understanding.* ICDAR 2026. (Full author list and DOI
   to be copied from the camera-ready; do not guess.)
6. G. Jaume, H. K. Ekenel, J.-P. Thiran. *FUNSD: A Dataset for Form
   Understanding.* ICDARW 2019.

Positioning consequence: DISCO already compares OCR and VLM pipelines; Xu et al.
already perform interventional attribution; Hwang et al. already study
degradation-induced hallucination. The paper must cite and differentiate from
all three in the introduction, not claim that prior work only did oracle-text
injection.

---

## 4. Verification commands

A snapshot of the audit run on 6 Nov 2026 is committed at
`docs/provenance_audit_2026-11-06.json`. It records the IndicVisionBench OCR/VQA
row counts and schemas, the FUNSD split, and the Patram-Bench 401.

Run these before a submission and paste the outputs into the artifact appendix.

```bash
# Dataset availability and schema
curl -s "https://datasets-server.huggingface.co/splits?dataset=krutrim-ai-labs/IndicVisionBench"
curl -s "https://huggingface.co/api/datasets/krutrim-ai-labs/IndicVisionBench"
curl -s "https://huggingface.co/api/datasets/nielsr/funsd"
curl -s "https://huggingface.co/api/datasets?search=patram"

# Model availability and revisions
curl -s "https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct"
curl -s "https://huggingface.co/api/models/bharatgenai/patram-7b-instruct"

# Literature
curl -s "https://arxiv.org/abs/2511.04727"
curl -s "https://arxiv.org/abs/2603.23511"
curl -s "https://arxiv.org/abs/2608.07943"
curl -s "https://aclanthology.org/2026.findings-acl.1237/"
```

Record the outputs (or their hashes) in `outputs/provenance/<date>/`. Do not
commit large files; commit the small JSON/text outputs only.

---

## 5. Data statement template for the paper

> We use the `ocr` split of IndicVisionBench (876 pages, 10 languages;
> revision `<hash>`; license `<license>`), the test split of FUNSD (50 forms;
> revision `<hash>`; license `<license>`), and `<Patram-Bench or none>`. We
> generate `<N>` extractive questions from the IndicVisionBench transcripts using
> the templates in Appendix `<X>`; the generation code and frozen manifest are
> released. Page images are not redistributed; we release sample IDs, prompts,
> and model outputs. No personally identifiable information was intentionally
> collected; FUNSD forms may contain synthetic names/addresses from the original
> dataset. Human validation was performed by `<N>` annotators fluent in
> `<languages>`; the annotation guideline and agreement are in Appendix `<Y>`.

If Patram-Bench is used, its license and redistribution terms must be quoted
verbatim, and the paper must state whether the data are public or available on
request.
