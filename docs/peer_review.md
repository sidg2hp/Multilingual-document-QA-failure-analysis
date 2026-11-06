# Adversarial Peer Review & Venue Assessment

**Artifact reviewed:** `sidg2hp/Multilingual-document-QA-failure-analysis` at commit
`8f18b0c` (6 Nov 2026).
**Review date:** 6 Nov 2026.
**Reviewer stance:** I am reviewing this as I would review a submission to a
top-tier main track (ACL/EMNLP/NeurIPS/ICLR/ICDAR), and separately as a workshop
submission and as a resource/demo paper. I have read every source file, run the
test suite, executed the smoke pipeline, and checked every external dependency
(datasets, model IDs, cited papers) against primary sources on the same date.

> **Response:** `docs/response_to_review.md` maps each defect below to the commit
> that fixes it (or states why it cannot be fixed by code). This review remains a
> snapshot of the artifact at `8f18b0c`; the verdict is unchanged until a real
> confirmatory run exists.

**Bottom line:** this is not yet a paper. It is a well-organized *protocol and
harness* for a paper that has not been run, on data that cannot currently be
obtained in the form the code assumes. The engineering is respectable; the
evidence is absent; several design choices would invalidate the intended causal
claims even after the runs are completed. The current artifact is not
submittable to any peer-reviewed venue, and its stated acceptance estimates are
not supported. The good news is that the core idea can become a credible
workshop paper, and possibly a strong findings-track paper, if the data and
measurement problems below are fixed before spending GPU time.

---

## 1. What kind of work is this?

This is an **empirical measurement / error-analysis study** with a **systems
harness** contribution, not a new model, not a new benchmark, and not a
theoretical paper.

The intended contributions are:

1. A **five-condition counterfactual intervention protocol** (`C0`, `C_δ`, `C1`,
   `C2`, `C3`) applied to both cascaded OCR+LLM pipelines and OCR-free VLMs, to
   attribute DocQA errors to visual degradation, layout/localization,
   recognition, and semantic reasoning.
2. **Abugida-aware metrics**, notably a Matra Error Rate (MER) alongside CER/WER,
   and a hallucination-to-abstention ratio (HAR).
3. A **read–reason dissociation probe** (`C1`): ask the VLM to transcribe first
   and answer second; if it reads correctly but answers incorrectly, the failure
   is in grounding/reasoning rather than perception.
4. A **cost-aware OCR-vs-VLM router** with an accuracy–compute Pareto analysis.
5. A reproducible harness for consumer hardware and IITR PARAM Ganga.

This framing is legitimate and potentially useful. It is also crowded: DISCO
(ICLR 2026 MMIntelligence workshop) already compares OCR pipelines and VLMs on
parsing and QA; "Locating Failure in Multi-Page VRDU" (arXiv 2608.07943) already
performs interventional attribution; DocHallu (ACL Findings 2026) already studies
perceptual hallucination under damaged document images. The differentiators here
are the **Indic/abugida focus**, the **read–reason probe**, and the **joint
OCR/VLM ladder**. Those differentiators are not yet demonstrated; they are only
implemented as code paths.

**Honest classification:** a methods-and-harness paper at the "protocol" stage.
It has no empirical result of its own. The dry-run outputs are synthetic and
must never be presented as findings.

---

## 2. Evidence status: claims vs. what exists

| Claim in README / docs | Evidence in repository | Status |
| :--- | :--- | :--- |
| "We implement a counterfactual stage-wise attribution framework" | Code exists; smoke test passes | **Partial** — implementation exists, but the decomposition is mis-specified for cascaded pipelines (§4.2) |
| "Evaluated on Patram-Bench, IndicVisionBench, and FUNSD" | FUNSD loader exists; Patram loader is absent; IndicVisionBench loader targets the wrong split for DocQA | **False as stated** |
| "Five controlled conditions per document-question pair" | All five enum values exist; `C1` is not meaningful for cascaded OCR; `C3` silently falls back to the ground-truth answer | **Partial / invalid** |
| "State-of-the-art baselines" | 3B quantized Qwen only; no 7B run; no multi-model run | **False** |
| "Statistical rigor: bootstrap, paired tests, Holm" | Implemented, but i.i.d. bootstrap ignores multiple questions per document; no McNemar; no power analysis | **Partial** |
| "Router evaluated with cross-validation to prevent leakage" | CV exists, but router features include `sample.metadata["smi"]` computed from the oracle transcript, and labels use degraded images while features use clean images | **Invalid** |
| "Reproducible" | Seeds exist; requirements unpinned; no dataset/model revisions; cache key omits prompts/revisions | **Weak** |
| "Paper readiness assessment" | `docs/paper_readiness_and_improvements.md` is candid about no results, but the research protocol claims 75–85% acceptance | **Overconfident** |

There are **zero** real model predictions in the repository. `outputs/` is
git-ignored; the committed code and tests only exercise synthetic samples. The
dry-run report in the local workspace shows `ANLS(C0) = 1.0`, `ANLS(C3) = 1.0`,
and `perturbation_error = 1.0` for the OCR pipeline — numbers produced by the
dry-run's hard-coded "return the ground truth" behaviour, not by a model.

---

## 3. Verified external dependencies (6 Nov 2026)

This check matters because the entire experiment design depends on datasets and
model checkpoints that the previous documentation asserted without verification.

### 3.1 Datasets

| Dependency | Verification | Consequence for the project |
| :--- | :--- | :--- |
| **Patram-Bench** | Real paper: "PATRAM-BENCH: A Multi-task, Multi-domain and Multi-lingual Benchmark for Indian Document Image Understanding" (ICDAR 2026 program; DBLP entry). The Hugging Face ID used in `configs/experiment_config.yaml` (`bharatgenai/Patram-Bench`) returns **HTTP 401 / does not exist**; the HF dataset search API returns no `patram` dataset; arXiv has no paper with that title. | **Blocking.** The primary 300-sample slice cannot be built from the stated source. Access must be requested from the authors, or the study must be re-scoped to public data. |
| **IndicVisionBench** | Real: arXiv 2511.04727, ICLR 2026. Verified splits and schemas via the HF datasets-server. `ocr` split: **876** pages with `id, image, text, language, page_url` — document transcripts, **no QA pairs, no bounding boxes**. `vqa_indic` split: **1,007** rows of *cultural* VQA over natural images (festivals, food, monuments), fields `short_q1/short_a1`, `mcq`, etc. — **not documents, no transcript**. | **Design-breaking.** The loader uses `vqa_indic` as if it were document QA; `row.get("text")` is `None`, and the fallback `f"{question} {answer}"` leaks the answer into the oracle. The 876-page `ocr` split has transcripts but no questions or boxes, so the five-condition protocol cannot be applied to it as-is. |
| **FUNSD** | Real; `nielsr/funsd` revision `7e7eeeed…`; test split 50 documents, train 149; columns `id, words, bboxes, ner_tags, image`. Verified via HF API. | Usable as an **English layout control**, but the current question ("value near token index N") is an artifact of the code, not a realistic information need; it also leaks positional metadata. |
| **IndicVisionBench "~5K images / 37K QA"** | The paper's aggregate figure is real (all configs combined), but the repository's claim implies document QA capacity that this benchmark does not have. | Citation must be precise; do not describe cultural VQA as document QA. |

### 3.2 Models

| Dependency | Verification | Consequence |
| :--- | :--- | :--- |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Exists, public, revision `6628554…`, architecture `Qwen2_5_VLForConditionalGeneration`. | Usable. |
| `bharatgenai/patram-7b-instruct` | Exists but is **gated** (auto-approval) and uses `trust_remote_code=True` with a custom `Patram1_1ForCausalLM` class; ~61 GB repository storage; tokenizer chat template is a simple `User: … Assistant:` format, not ChatML. | The current `AutoModelForImageTextToText` fallback will not load it; the HPC plan as written is not executable. A dedicated adapter is required. |
| `HuggingFaceTB/SmolVLM-Instruct` | The config ID is stale; the current family is `HuggingFaceTB/SmolVLM-*` (e.g. `SmolVLM-500M-Instruct`, `SmolVLM2-2.2B-Instruct`). | Must be pinned to an exact checkpoint and revision, or removed. |
| `tesseract 5.x`, `paddleocr PP-OCRv4` | Tools exist; but `ocr_pipeline.py` hard-codes PaddleOCR `lang="en"` and the Tesseract language map covers only 3 languages. | Indic OCR baselines are not actually multilingual in code. |

### 3.3 Literature

Verified as real and relevant:

- **DISCO: Document Intelligence Suite for COmparative Evaluation**, Benkirane et
  al., arXiv:2603.23511, ICLR 2026 MMIntelligence workshop. Compares OCR vs VLM
  on parsing and QA across handwritten, multilingual, medical, infographic, and
  multi-page documents. Directly overlaps the proposed comparison.
- **Locating Failure in Multi-Page Visually Rich Document Understanding: An
  Empirical Attribution**, Xu et al., arXiv:2608.07943. Interventional
  attribution for representation, selection, and reasoning on MMLongBench-Doc.
  Directly overlaps the proposed attribution framing (English, multi-page).
- **Perceptual Hallucination in Vision–Language Models: Definition, Analysis and
  Verification**, Hwang et al., Findings of ACL 2026, pp. 24710–24725. Introduces
  DocHallu (2,671 original–damaged pairs) and shows numerical content is more
  hallucination-prone. Directly overlaps the degradation/hallucination angle.
- **IndicVisionBench**, Faraz et al., arXiv:2511.04727, ICLR 2026.
- **Patram-Bench**, ICDAR 2026 (paper verified via conference program/DBLP; data
  availability unverified).

The related-work section must position against these three directly, not
pretend the space is empty. The claim "unlike prior comparisons that only inject
oracle text into cascaded pipelines" is too strong: Xu et al. already do
controlled interventions, and DISCO already compares OCR and VLMs.

---

## 4. Fatal and major technical flaws

The flaws below are ordered by how badly they damage the intended claims. Line
numbers refer to commit `8f18b0c`.

### 4.1 No executed results (fatal)

There are no real predictions. The smoke pipeline fabricates the ground-truth
answer for `C0`, `C1`, `C2`, and `C3`
(`src/models/ocr_pipeline.py:169-178`, `src/models/vlm_runner.py:189-197`). A
paper cannot be written from this. Every downstream statistic, Pareto curve,
and failure-mode distribution currently reflects the fabricator, not a model.

### 4.2 The causal ladder is mis-specified for cascaded OCR (fatal for the main claim)

`decompose_telescoping_error` assumes the nesting

```
degraded < clean < C2 (oracle layout) < C3 (oracle text)
```

(`src/evaluation/attribution.py:67-75`). This is coherent for an OCR-free VLM:
`C2` crops to the answer region, so it adds localization information but still
requires visual recognition; `C3` adds the transcript.

It is **not** coherent for a cascaded OCR+LLM pipeline. There, `C2` is defined
as *ground-truth text with ground-truth bounding boxes* (`C2` oracle layout),
while `C3` is *ground-truth text in flat reading order*. `C2` therefore contains
strictly more information than `C3` (perfect text + perfect layout vs. perfect
text), so the assumed order is reversed. The code feeds both pipelines through
the same decomposition and labels `C3 − C2` as "recognition error"; for the
cascade that quantity is the effect of *removing* layout information, not
recognition error. The reported causal story would be backwards.

The implementation is also internally inconsistent: for the cascade, `C2` crops
the image and re-runs OCR (`src/models/ocr_pipeline.py:155-165`), which is a
*localization + recognition* intervention, not the coordinate-text intervention
described in `format_layout_aware_oracle_text` and the docs. The protocol and the
code disagree.

**Fix direction:** define an architecture-specific ladder and report only
identified components.

- VLM: `E_total = (C0 − Cδ) + (C2 − C0) + (C3 − C2) + (1 − C3)` — valid.
- Cascade: `E_total = (C0 − Cδ) + (C3 − C0) + (1 − C3)`, where `C3 − C0` is a
  joint extraction error (recognition + reading order). Do not split it without
  an additional intervention. Report the layout-representation effect
  `C2_coord − C3_flat` separately as a diagnostic, not as an error component.
- Add a **placebo crop** (`C2b`, random region of matched size) and a
  **shuffled transcript** (`C3b`, same text, wrong order) to subtract the
  "cropping/removing distractors" and "any transcript helps" effects.

### 4.3 Oracle-text leakage makes `C3` trivially correct (fatal)

When `sample.oracle_text` is missing, `C3` falls back to the ground-truth answer:

```python
extracted_text = sample.oracle_text or (
    sample.ground_truth_answers[0] if sample.ground_truth_answers else ""
)
```

(`src/models/ocr_pipeline.py:151-153`; the VLM has the same problem at
`src/models/vlm_runner.py:199-205` where it passes an empty transcript if
`oracle_text` is `None`). For IndicVisionBench `vqa_indic`, the loader sets
`oracle_text = row.get("text")`, which is `None`, so:
`C3` receives the answer, scores 1.0, `E_sem = 0`, and the whole decomposition
collapses. Any dataset without a verified transcript must mark `C3` as
unsupported, not silently substitute the label.

### 4.4 The router has two independent leakage/mismatch bugs (fatal for the routing claim)

1. **Oracle feature leakage.** `extract_routing_features` receives
   `script_mixing_index=float(sample.metadata["smi"])`
   (`scripts/run_experiment.py:472-478`), and `metadata["smi"]` is computed from
   the *oracle transcript* in `src/data/datasets.py:89` and `:189`. At inference
   time the transcript is unavailable — that is the whole point of routing. The
   router is therefore trained on a feature that encodes the label side. The
   reported Pareto curve is optimistic by construction.
2. **Train/test condition mismatch.** Features are extracted from
   `sample.image_obj` (clean image), but the regression targets are
   `attr.anls_c_delta_degraded` — performance on the *degraded* image
   (`scripts/run_experiment.py:473-480`). A router that sees the clean image
   cannot predict degraded performance from blur features it never observes.
   Either extract features from the actual input (the degraded image) or predict
   clean performance; do not mix.

Additionally, the "cost" is a hard-coded relative constant
(`src/routing/router.py:44-46`), not the measured `latency_ms` that is already
recorded in every `PredictionRecord`. The Pareto frontier is therefore not an
accuracy–compute frontier.

### 4.5 `C1` does not measure what it claims (major)

`C1` asks the VLM to transcribe, then asks a *second, text-only* call to answer
from that transcript (`src/models/vlm_runner.py:219-230`). The code records only
the final answer; it never scores the transcript against the oracle, so it
cannot distinguish:

- "the VLM read the Indic text correctly but failed to use it" (the claimed
  read–reason dissociation), from
- "the VLM hallucinated a fluent transcript and then reasoned over its own
  hallucination" (a perception failure).

Without `CER/MER(C1_transcript, T*)`, the `Δ_dissoc = Acc(C1) − Acc(C0)` quantity
is uninterpretable. It also uses a different prompt and a different context
length from `C0`; these are confounds unless controlled.

**Fix direction:** record the `C1` transcript, compute CER/MER against `T*`,
report `P(read correct)` and `P(answer correct | read correct)`, and compare
`C0` vs `C1` with a paired McNemar test stratified by transcript quality.

### 4.6 `C2` crop is a bundled intervention, not "layout" (major)

Cropping to the gold answer box removes distractors, multi-column context,
headers, and the need to localize — all at once. It can also make a question
unambiguous that was ambiguous in the full page. The accuracy jump is therefore
`localization + clutter removal + disambiguation`, not pure layout. A **placebo
crop** (random box of the same size, or a box over a non-answer region) is
required to estimate the crop effect; without it `Δ_lay` is inflated and
unfalsifiable.

### 4.7 No answer parsing; raw generations are scored (major)

The models are prompted to answer "concisely", but there is no extraction step.
A reply such as `"The total is Rs. 12,450."` is scored against `"Rs. 12,450"` and
ANLS drops to zero, even though the answer is correct. Modern instruct models
routinely add prefixes, markdown, or explanations. The resulting numbers
measure formatting compliance, not document understanding. There is also no
handling of unanswerable questions, refusals, or truncated generations.

### 4.8 Degradation manipulations are not controlled (major)

- `apply_perspective_skew` rotates with `expand=False` and a white fill
  (`src/perturbations/degrade.py:112-117`), so corners are cropped. Content loss
  is confounded with layout distortion. Use `expand=True` and crop back to the
  original canvas, or measure and report the lost area.
- JPEG is applied to images that may already be JPEG; the "clean" baseline is
  not passed through the same encode/decode path. Add an identity/re-encode
  control.
- Severity levels are absolute (`radius=3.2`, `quality=5`) rather than scaled to
  estimated text height/DPI. The same `level_5` is trivial on a 300-DPI scan and
  catastrophic on a 72-DPI screenshot. Report the induced CER/MER as a
  manipulation check, not just the nominal level.
- There is no perturbation-specific hypothesis or calibration; "5 levels × 5
  families" is a grid, not an experiment.

### 4.9 Statistical inference ignores document clustering (major)

Multiple questions can come from the same document (FUNSD entity pairs;
IndicVisionBench pages). `bootstrap_ci` resamples individual items i.i.d.
(`src/evaluation/statistics.py:55-57`), which understates uncertainty and
inflates significance. Cross-validation for the router is also `KFold`, not
`GroupKFold` by document. Use document-clustered bootstrap, grouped CV, and
report the number of documents and items separately.

### 4.10 Cache keys can silently mix incompatible predictions (major)

`PredictionCache.make_key` includes model name, quantization, sample, condition,
perturbation type, and severity, but **not** the prompt text/version, model
revision, dataset revision, parser version, image hash, or code version
(`src/utils/prediction_cache.py:35-44`). Changing a prompt or model silently
reuses stale predictions, which is exactly the kind of error that produces
irreproducible paper numbers. The cache also has no schema version and no way to
invalidate a subset.

### 4.11 Dataset manifests and revisions are absent (major)

`datasets.load_dataset(..., streaming=True)` does not pin a revision
(`src/data/datasets.py:74-79`, `:112`). Streaming order, dataset contents, and
model files can change. A paper needs a frozen manifest of sample IDs, dataset
revision hashes, image hashes, and model revisions. None exists.

### 4.12 The FUNSD question is not a real task (major)

`stream_funsd_samples` asks "What is the value recorded near token index N?"
(`src/data/datasets.py:135`). This is a positional lookup keyed to the loader's
internal indexing, not a document understanding question; the answer is
recoverable from the question alone by counting tokens, and the "field" has no
semantic identity. Use FUNSD's `ner_tags` and reading order to build
question/answer entity pairs (e.g., "What is the value of <field label>?"), or
use a different form-QA dataset.

### 4.13 Prompt and configuration inconsistencies (major)

- `max_new_tokens` is 96 in the VLM constructor
  (`src/models/vlm_runner.py:30`) but 128 in the config; the runner passes the
  config value only for the VLM, not the OCR reader.
- The OCR reader uses raw text, not the model's chat template
  (`src/models/ocr_pipeline.py:121-128`), while the VLM does use a chat template.
  This confounds architecture with prompt format.
- `C0`, `C2`, and `C3` use different instructions; `C3` lacks the abstention
  instruction that `C0`/`C2` have. Any difference across conditions includes
  prompt differences.
- `C2` silently degrades to `C0` when `answer_bbox` is missing
  (`src/models/ocr_pipeline.py:155` falls through to full-image extraction).
  This makes "C2" identical to "C0" on datasets without boxes, without any flag.
- The config lists quantization modes and multiple VLMs, but the runner hard-codes
  `quantization="nf4"` and a single primary VLM
  (`scripts/run_experiment.py:367-372`).

### 4.14 Model loading is not portable (major)

`inputs.to(self._model.device)` (`src/models/vlm_runner.py:163-165`) is fragile
for `device_map="auto"` sharded/quantized models; use the input embedding device.
Patram-7B requires `trust_remote_code=True` and a custom class; the code's
`AutoModelForImageTextToText` fallback will fail. SmolVLM's processor chat
template and image handling differ from Qwen's; the `is_qwen` branch assumes all
non-Qwen models accept `images=[image]` and a ChatML-style template. This will
silently produce garbage or crash for several planned models.

### 4.15 Hardware/quantization claims are untested (major)

The docs claim NF4/INT8/FP16 comparisons and a 7B run on PARAM Ganga. The code
only runs NF4 on one 3B VLM. There is no memory measurement, no throughput
measurement, no OOM fallback, no per-model unload verification, and no evidence
the VLM even loads on the 6 GB RTX 3050. The "quantization as a scientific
variable" contribution is currently a config comment.

### 4.16 Reproducibility and packaging gaps (moderate)

- `requirements.txt` is unpinned, mixes CPU/GPU/OCR stacks, and has no lockfile.
- `PYTHONHASHSEED` is set after interpreter start
  (`src/utils/seed.py:10`), which has no effect on hash randomization.
- No CI, no lint/type checks, no coverage, no `pyproject.toml`.
- No `LICENSE` file despite the MIT badge in the README
  (`README.md:6`).
- No `CITATION.cff`, no dataset license/provenance table.
- The SLURM script does not load modules, activate an environment, pin the
  config's model, run the sweep, or use `--resume`; it runs a single
  `indic_vision_bench` setting (`slurm/run_param_ganga.sh:29-32`).
- `outputs/` is git-ignored, but the README references specific output files;
  there is no JSON schema or example report for readers to validate.

### 4.17 Documentation overclaims (moderate)

`docs/research_protocol.md:19` states an upgraded acceptance potential of
"~75–85%" for ICDAR/ACL workshops; `docs/paper_readiness_and_improvements.md`
gives 45–60% for a strong workshop *conditional on executing the study*. The
75–85% figure is not defensible for a first-time measurement study with 3B
quantized models, no human validation, and a single dataset family. Overclaiming
in a public repo is a credibility risk; reviewers who see it will discount the
work.

---

## 5. Mock reviews

### Reviewer 1 — empirical methods / statistics

> **Summary:** The authors propose a counterfactual protocol to localize DocQA
> failures across OCR and VLM pipelines. The protocol is sensible in spirit, but
> the paper reports no real experiments, and the causal decomposition is not
> identified for the cascaded pipeline. The statistical analysis is i.i.d. and
> underpowered for the per-language claims.
>
> **Strengths:** The five-condition idea is intuitive; the HAR/MER metrics target
> real gaps; the code is readable.
>
> **Weaknesses:** (1) The decomposition's `C2`/`C3` ordering is architecture-
> dependent and reversed for cascades. (2) `C3` can receive the answer. (3) The
> router has oracle leakage and a condition mismatch. (4) No answer parsing;
> ANLS measures formatting. (5) No power analysis or clustered inference.
>
> **Score:** 2/10 (reject). **Confidence:** 4/5.

### Reviewer 2 — multilingual / document understanding

> **Summary:** The paper aims to explain failures on Indian documents. However,
> the primary dataset is not available at the cited location, and the secondary
> "document" benchmark is actually a cultural VQA benchmark over natural images.
> The only verified document set with layout boxes is English FUNSD. The
> multilingual claims are therefore unsupported by the data.
>
> **Strengths:** The matra-aware metric is a genuinely useful idea; the Indic
> script-mixing analysis is relevant; the authors correctly avoid claiming a new
> benchmark.
>
> **Weaknesses:** (1) Patram-Bench inaccessible. (2) IndicVisionBench split
> misuse. (3) FUNSD question is artificial. (4) English-only prompts for Indic
> documents are not justified or analyzed. (5) No native-speaker validation of
> generated questions or hallucination labels.
>
> **Score:** 2/10 (reject). **Confidence:** 4/5.

### Reviewer 3 — systems / reproducibility

> **Summary:** The harness is clean but has not been run at scale. The router's
> Pareto analysis uses hard-coded costs, and the cache can silently reuse stale
> predictions. The claimed 7B/FP16 and multi-model comparisons are not
> implemented.
>
> **Strengths:** Resumable cache, config files, tests, and an explicit
> consumer-hardware plan are above average for a student project.
>
> **Weaknesses:** (1) No dataset/model revisions. (2) Cache key incomplete.
> (3) Model loader not portable to Patram/SmolVLM. (4) No CI/lockfile/license.
> (5) The SLURM script does not implement the described experiment matrix.
>
> **Score:** 3/10 (reject; would be a solid artifact badge after fixes).
> **Confidence:** 5/5.

### Meta-review

**Recommendation: Reject** for the main track and for a strong workshop *in the
current state*. There is nothing to evaluate empirically, and the central
measurement is not yet valid. This is not a "weak reject because the idea is
incremental"; it is a "not yet a paper" because the evidence does not exist and
the data plan is broken. With the fixes in §6 and a completed 600-item study,
the work could become a credible workshop submission; with 7B models, multiple
seeds, human validation, and a crisp mechanistic claim, it could reach a
findings-track submission.

---

## 6. Venue assessment (calibrated)

Probabilities are for **acceptance of a paper**, not of the code artifact. They
assume competent writing and no fatal flaw beyond those fixed.

| State of the work | Main track (ACL/EMNLP/NeurIPS/ICLR/ICDAR) | Strong domain workshop (ICDAR/WACI/CVPR-VDU/ACL-DocInt) | Findings / journal | Resource/demo track |
| :--- | :---: | :---: | :---: | :---: |
| **Current artifact (no real results)** | **0%** — not submittable | **0%** — not submittable | 0% | 10–15% (artifact only, if packaged) |
| Harness fixed, 600-item study executed with 3B models, 5 conditions, CIs, per-language breakdowns | <5% | 30–45% | 10–15% | 55–70% |
| Above + one 7B FP16 model, 3 seeds, human validation of 100 hallucination cases, leakage-free router, held-out test | 5–10% | 50–65% | 30–40% | 70–80% |
| Above + a surprising, replicated mechanistic claim (e.g., read–reason dissociation is large and script-dependent; or quantization selectively destroys matras) and a clear decision-theoretic contribution | 10–15% | 60–75% | 40–50% | 75–85% |

The previous documents' "75–85%" estimate corresponds only to the last row, and
even then it is optimistic for a first paper from a group without a track record
at these venues. A realistic target is a **strong workshop** (ICDAR 2027
workshops, WACI, CVPR VDU, ACL/EMNLP Document Intelligence) or **Findings**,
with the main track as a stretch goal after the effect replicates.

**What would change the verdict most:** not more code, but one clean empirical
claim that survives controls. The most promising candidates, in order:

1. **Read–reason dissociation in Indic VLMs.** If `C1` transcripts have low CER
   but `C0` QA is much worse than `C1` QA, and the gap is larger for Devanagari/
   Bengali/Tamil than for Latin, that is a mechanistic, actionable result. It
   requires transcript scoring, which the current code lacks.
2. **Quantization × script interaction.** If NF4 selectively degrades matra
   recognition relative to FP16 on Indic scripts while leaving Latin unchanged,
   that is a deployable finding. It requires the multi-precision runs the code
   does not yet perform.
3. **Selective prediction / routing.** If a text-free image router can capture a
   meaningful fraction of the OCR/VLM oracle gap at lower cost, that is a
   practical contribution. It requires fixing the leakage and using measured
   costs.

Any one of these, executed cleanly, is worth more than the entire current
five-condition matrix on synthetic data.

---

## 7. Threats to validity the paper must discuss

- **Construct validity:** template-generated questions may not reflect real
  information needs; ANLS on short spans rewards string matching, not reasoning;
  "confident hallucination" is a proxy, not a human judgment.
- **Internal validity:** prompt differences across conditions; crop placebo;
  re-encoding; severity calibration; model version drift; cache mixing.
- **External validity:** 3B quantized models may not represent 7B/70B systems;
  IndicVisionBench Wikipedia pages are not representative of forms/invoices;
  FUNSD is English and 1990s-era forms.
- **Statistical validity:** document clustering; multiple comparisons; small
  per-language cells; seed variance; selective reporting of the best
  perturbation.
- **Data provenance:** dataset licenses, redistribution, Wikisource provenance,
  PII in forms, model licenses (Qwen research license; Patram Apache-2.0 but
  gated).
- **Ecological validity:** a 6 GB GPU forces 4-bit quantization; the paper's
  conclusions may be artifacts of quantization, not architecture.

---

## 8. Prioritized remediation plan

### P0 — before spending any GPU time

1. **Resolve data access.** Obtain Patram-Bench from the authors (written
   request; ICDAR 2026 paper) or re-scope to public data. Do not code against a
   nonexistent HF ID.
2. **Fix the dataset abstraction:** verified schemas, explicit
   `supports_text_oracle` / `supports_layout_oracle` flags, no answer leakage,
   no silent fallbacks, pinned revisions, frozen manifest.
3. **Fix the attribution ladder** (architecture-specific identities, placebo
   conditions, no missing-condition zeros).
4. **Add answer parsing and versioned prompts**; hold prompts constant except
   for the intervention.
5. **Fix the router**: text-free features only, features from the same image as
   labels, measured costs, grouped CV.
6. **Add transcript scoring to `C1`** and condition the dissociation analysis on
   it.

### P1 — the minimum credible study

7. Freeze a 600-item manifest stratified by script/language/dataset, with
   hashes and revisions.
8. Run 3B models across the five conditions and five perturbation families with
   5 levels; include a placebo crop, shuffled transcript, and re-encode control.
9. Report document-clustered bootstrap CIs, paired permutation/McNemar tests,
   Holm correction over a pre-registered family, and a power/MDE analysis.
10. Human-validate 100 "confident hallucination" cases with a bilingual
    annotator; report agreement.
11. Report per-script and per-language cells with CIs and cell sizes; do not
    overinterpret cells with n < 30.

### P2 — the competitive version

12. Add one 7B FP16 model (Qwen2.5-VL-7B) and one 7B text reader on PARAM Ganga;
    implement the Patram adapter or drop it.
13. Run three seeds for any stochastic decoding; report seed variance.
14. Calibrate perturbations to estimated text height; report induced CER as the
    manipulation check.
15. Build the router on measured latency and clean/degraded strata; report
    calibration and risk–coverage, not just a Pareto curve.
16. Release the manifest, prompts, code, and a reproducibility script; add CI,
    lockfile, license, and citation metadata.

---

## 9. What is genuinely good and should be preserved

- The **idea of a nested intervention ladder** is sound for VLMs and, with the
  architecture-specific fix, can be a clean measurement tool.
- **Matra Error Rate** is a novel and appropriate metric for abugidas. It needs
  script-restricted extraction and alignment, but the motivation is right.
- **HAR** addresses a real gap: abstention and fabrication are different
  failures, and aggregate ANLS hides that.
- The **read–reason probe** is the most promising mechanistic idea in the
  project. It deserves to be the paper's centerpiece, not a side condition.
- The **engineering hygiene** (dataclasses, cache, tests, configs) is better
  than most research prototypes and makes the fixes tractable.
- The **honesty** in `docs/paper_readiness_and_improvements.md` about the lack of
  results is commendable; the overconfident acceptance estimate in
  `docs/research_protocol.md` is the exception, not the rule.

---

## 10. Summary judgement

**Can the current progress be accepted at a good conference? No.** It has no
empirical results, its primary dataset is not accessible as configured, its
secondary dataset is not a document-QA benchmark, its oracle condition can leak
the answer, its causal decomposition is reversed for cascaded pipelines, and its
router is trained with oracle features. Any one of these would be a major
reviewer objection; together they make the current claims unsupported.

**Is the project worth continuing? Yes, with a narrower and more honest scope.**
The right next step is not to run the existing matrix faster. It is to (a) secure
data that actually supports the protocol, (b) fix the measurement so the numbers
mean what the paper says, and (c) pick one mechanistic claim — most likely
read–reason dissociation or quantization × script — and test it properly. A
well-executed 600-item study with a single clean claim is a credible workshop
paper. The current repository is a good starting point for that, not a paper.

---

### Appendix A: Reproduction of this review

```bash
git rev-parse HEAD                      # 8f18b0c
python -m pytest tests/ -q              # 18 passed
python scripts/run_experiment.py --mode dry_run --resume --seed 42
```

Verified external endpoints (6 Nov 2026):

- `https://datasets-server.huggingface.co/splits?dataset=krutrim-ai-labs/IndicVisionBench`
- `https://huggingface.co/api/datasets/krutrim-ai-labs/IndicVisionBench`
- `https://huggingface.co/api/datasets/nielsr/funsd`
- `https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct`
- `https://huggingface.co/api/models/bharatgenai/patram-7b-instruct`
- `https://arxiv.org/abs/2511.04727`, `2603.23511`, `2608.07943`
- `https://aclanthology.org/2026.findings-acl.1237/`
- `https://huggingface.co/api/datasets?search=patram` (empty)

### Appendix B: File-level defect index

| File | Lines | Issue |
| :--- | :--- | :--- |
| `src/evaluation/attribution.py` | 67–75 | Reversed `C2`/`C3` ladder for cascades |
| `src/evaluation/attribution.py` | 135–161 | Missing conditions scored as 0 / silently substituted |
| `src/evaluation/attribution.py` | 172–184 | Primary-stage rule can label semantic failure when `C3` is absent |
| `src/models/ocr_pipeline.py` | 151–153 | `C3` falls back to the gold answer |
| `src/models/ocr_pipeline.py` | 155–165 | `C2` is a crop+OCR intervention, not coordinate text |
| `src/models/ocr_pipeline.py` | 102 | PaddleOCR hard-coded `lang="en"` |
| `src/models/ocr_pipeline.py` | 121–128 | No chat template for the reader LLM |
| `src/models/vlm_runner.py` | 163–165 | Fragile `.to(model.device)` for sharded models |
| `src/models/vlm_runner.py` | 219–230 | `C1` transcript never scored |
| `src/models/vlm_runner.py` | 189–197 | Dry-run fabricates ground truth |
| `src/data/datasets.py` | 86–88 | `vqa_indic` treated as DocQA; answer in fallback oracle |
| `src/data/datasets.py` | 98 | `oracle_text=row.get("text")` is `None` for VQA |
| `src/data/datasets.py` | 135 | FUNSD token-index question |
| `src/data/datasets.py` | 74–79, 112 | No dataset revision pinning |
| `scripts/run_experiment.py` | 472–480 | Router oracle SMI + clean-feature/degraded-label mismatch |
| `scripts/run_experiment.py` | 367–372 | Single model, hard-coded NF4 |
| `src/routing/router.py` | 44–46 | Hard-coded costs; `KFold` not grouped |
| `src/utils/prediction_cache.py` | 35–44 | Cache key omits prompt/revision/version |
| `src/utils/seed.py` | 10 | `PYTHONHASHSEED` set too late |
| `src/perturbations/degrade.py` | 112–117 | Rotation crops content (`expand=False`) |
| `configs/experiment_config.yaml` | 12–13 | Nonexistent Patram-Bench HF ID |
| `README.md` | 6 | MIT badge without `LICENSE` |
| `docs/research_protocol.md` | 19 | Unsupported 75–85% acceptance claim |
