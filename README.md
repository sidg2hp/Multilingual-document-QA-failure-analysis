# Where Does Document Understanding Break?

**A counterfactual error-attribution harness for OCR-based and OCR-free document QA on multilingual documents.**

> **Status (7 Nov 2026): research harness and pre-registered protocol, not a paper and not a validated empirical result.**
> The confirmatory VLM run has not happened. The smoke pipeline is explicitly
> non-reportable, and the laptop pilot is instrumentation evidence only. See
> [`docs/peer_review.md`](docs/peer_review.md) for the adversarial review and
> [`docs/claim_evidence_matrix.md`](docs/claim_evidence_matrix.md) for exactly
> what may and may not be claimed today.

> **Formal report:** [Project report and PARAM Ganga compute request](reports/Document_QA_Failure_Analysis_Project_Report.pdf) — 26 pages, including the protocol, pilot evidence, compute plan, risk register, and endorsement page.

[![CI](https://github.com/sidg2hp/Multilingual-document-QA-failure-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/sidg2hp/Multilingual-document-QA-failure-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. The problem

A document QA system can return a wrong answer for very different reasons:

- the page was blurred, compressed, or photographed at an angle;
- the answer region was never localized;
- the OCR engine misread a character, a matra, or a reading order;
- the model read the text but could not reason over it;
- the model never read the text and produced a confident hallucination.

Aggregate QA metrics such as ANLS or exact match collapse all of these into one
number. That number is useful for a leaderboard but not for a deployment
decision. If a system fails because of OCR, buying a larger language model is the
wrong fix. If it fails because of reasoning, a better OCR engine is the wrong fix.
The practical question is therefore **where** the failure enters, not only **how
often** it occurs.

The problem is sharper for Indian documents. Indic scripts are abugidas: vowel
diacritics (matras), virama/halant, and conjunct ligatures carry lexical and
numeric meaning. A single combining-mark error can change a name, a date, or an
amount. Documents are also form-like, code-mixed, scanned at inconsistent
quality, and often processed with English-centric prompts. A measurement protocol
designed for clean English pages does not automatically transfer.

## 2. What this project is — and is not

**This project is a measurement and error-analysis study with a systems
harness.** It asks where failures enter two DocQA architectures:

1. a **cascaded OCR + text-LLM pipeline**, which exposes an explicit text
   interface; and
2. an **OCR-free vision-language model (VLM)**, which reads the page image and
   answers end-to-end.

It implements a counterfactual ladder, negative controls, abugida-aware metrics,
document-clustered statistics, and a cost-aware router. The intended output is a
diagnostic account of failure stages, not a new model or a new leaderboard entry.

**This project is not:**

- a new benchmark or a new foundation model;
- an accepted paper;
- evidence that one architecture is better than the other;
- a claim that Patram-Bench, 23 languages, or "first Indic attribution" has been
  evaluated;
- a place to cite the pilot numbers as findings.

The project deliberately removed earlier overclaims. The current policy is:
**every public number must be traceable to a prediction file, a manifest, and a
script; every hypothesis must be stated before the confirmatory run.**

## 3. Research questions

| ID | Question | Primary evidence |
| :--- | :--- | :--- |
| **RQ1** | Which failure stage dominates for OCR cascades versus OCR-free VLMs? | Architecture-specific counterfactual decomposition with document-clustered CIs. |
| **RQ2** | Do Indic VLMs read a span correctly but fail to use it when answering directly? | `C1` transcript CER/MER, `C0` versus `C1` QA accuracy, stratified by script. |
| **RQ3** | Do recognition errors concentrate in matras relative to base characters under degradation? | Matra Error Rate versus base-character CER across calibrated severity levels. |
| **RQ4** | Does 4-bit quantization disproportionately degrade Indic recognition relative to Latin? | Matched NF4/INT8/FP16 comparison with a script interaction. |
| **RQ5** | Can a text-free image router capture the OCR/VLM oracle gap at lower measured cost? | Out-of-fold, page-grouped routing with measured latency and bootstrap gain. |

The primary claim for the paper is **RQ2**, because it is mechanistic and does
not depend on a layout oracle. RQ3 and RQ4 are the fallback claims. RQ1 is the
framework that makes the others measurable; it is not, by itself, a surprising
result. This ordering is fixed in [`docs/preregistration.md`](docs/preregistration.md).

## 4. Approach

### 4.1 End-to-end measurement loop

```text
verified data
    -> frozen manifest (IDs, hashes, revisions, capability flags)
    -> counterfactual conditions (C0, C_delta, C1, C2, C2b, C3, C3b)
    -> VLM and/or OCR-cascade runner
    -> per-item metrics and raw predictions
    -> document-clustered statistics and attribution
    -> tables, figures, routing analysis
    -> paper / artifact
```

The manifest is the unit of reproducibility. It fixes the dataset revision, page
and sample IDs, question template, gold answers, transcript hash, image hash, and
which oracle conditions are legal for that item. A condition that the data or
model cannot support is marked **unsupported** and raises; it is never silently
replaced by `C0`, `C3`, or the gold answer.

### 4.2 Conditions and what they isolate

The protocol uses seven conditions. The labels are shared across architectures,
but their meaning is deliberately **not** shared.

| Condition | OCR-free VLM input | Cascaded OCR + LLM input | What it isolates |
| :--- | :--- | :--- | :--- |
| `C0` | Clean image + question | Clean image -> OCR -> LLM | Baseline end-to-end accuracy |
| `Cδ` (`C_delta`) | Degraded image + question | Degraded image -> OCR -> LLM | Sensitivity to visual degradation |
| `C1` | Transcribe, then answer from own transcript; transcript is scored | **Not defined** | Read-reason dissociation (VLM only) |
| `C2` | Gold answer-region crop | Coordinated gold text | Localization and clutter removal (VLM); full extraction oracle (cascade) |
| `C2b` | Placebo crop of matched size | **Not defined** | Crop/context negative control (VLM) |
| `C3` | Gold transcript | Flat gold transcript | Recognition oracle |
| `C3b` | Shuffled gold transcript | Shuffled gold transcript | Transcript-content / order control |

**Why the ladder is architecture-specific.** For a VLM, `C2` is a crop: it
removes distractors and supplies the answer region, but the model must still read
the pixels. `C3` then supplies the transcript, so the VLM sequence
`C0 -> C2 -> C3` is nested.

For a cascade, the sequence is different. `C2` is coordinated gold text, which
contains both perfect text and perfect layout; `C3` is flat gold text, which
contains perfect text but not layout. `C2` therefore contains strictly more
information than `C3`, and the VLM ordering would be reversed. The cascade
decomposition stops at a **joint extraction** term and reports the layout effect
separately. This is the central methodological correction in the project.

`C1` is deliberately undefined for the cascade: `C0` already performs recognition
followed by reasoning, so a second "read first" prompt would duplicate `C0`
rather than create an independent intervention. `C2b` is undefined because a
placebo crop is an image-side control.

### 4.3 The identified decompositions

For an OCR-free VLM:

```text
E_total = (Acc(C0) - Acc(Cδ))
        + (Acc(C2) - Acc(C0))
        + (Acc(C3) - Acc(C2))
        + (1 - Acc(C3))
```

The four terms correspond to perturbation, localization, recognition, and
semantic reasoning. For a cascaded pipeline:

```text
E_total = (Acc(C0) - Acc(Cδ))
        + (Acc(C3) - Acc(C0))
        + (1 - Acc(C3))
```

The middle term is **joint extraction error** (recognition plus reading order).
It is not split, because the available interventions cannot identify recognition
and order separately. The coordinate-text diagnostic `Acc(C2_coord) -
Acc(C3_flat)` is reported as a separate layout-representation result, not as a
term in the identity.

Three rules follow:

1. **Missing conditions are `None`, never zero and never the gold answer.** They
   are excluded from the relevant aggregate and counted in a coverage table.
2. **Negative components are retained.** If an oracle intervention hurts
   accuracy, that is a finding, not a number to clip.
3. **The identity is reported with its assumptions.** The decomposition is an
   accounting identity over the measured conditions, not a claim that the
   interventions are perfectly pure.

### 4.4 Why the negative controls matter

`C2` is not a clean "layout" intervention: cropping to the answer region removes
distractors, may disambiguate the question, and can change the effective
resolution. `C2b` is a placebo crop of matched size from the same page, excluding
the answer region. The net localization effect is reported as a
difference-in-differences, not as a raw `C2 - C0` jump.

`C3b` is a shuffled transcript. It controls for the possibility that "any
transcript helps" rather than "the correct transcript helps". In the laptop pilot,
line-level shuffling preserved label-value pairs and produced a weak control; the
confirmatory version must use a stronger sentence/word shuffle and report the
manipulation check.

`C3` must use a verified transcript. If a dataset has no transcript, `C3` is
unsupported. The previous implementation could fall back to the gold answer,
which made `C3` trivially correct and collapsed the decomposition; that fallback
has been removed.

### 4.5 Metrics

| Metric | Meaning and use |
| :--- | :--- |
| **ANLS** | Normalized Levenshtein similarity with the standard 0.5 cutoff; primary QA metric. |
| **Exact Match (EM)** | Normalized exact match against any accepted answer. |
| **Field F1** | Token-level F1 for form-field extraction. |
| **CER / WER** | Character and word error rate of OCR or `C1` transcripts against gold. |
| **Base CER** | CER after removing combining marks; separates consonant errors from diacritic errors. |
| **Matra Error Rate (MER)** | Edit distance on Indic combining marks only; Latin diacritics are excluded. |
| **HAR** | Confident hallucinations divided by explicit abstentions, with counts and a saturation flag. |
| **Latency** | Measured wall-clock milliseconds per condition; used by the router cost model. |

ANLS is reported with conservative normalization by default. Numeric or
punctuation canonicalization is an explicit secondary analysis and is labelled;
it is never silently applied. The hallucination proxy is a screening signal, not
ground truth. Human validation is required before any causal claim about
"matra-blind hallucination".

### 4.6 Statistics

The item is the unit of observation, but multiple questions can share a page.
All intervals therefore use a **document-clustered percentile bootstrap** (10,000
resamples by page ID), not an i.i.d. bootstrap. The analysis includes:

- paired permutation tests and McNemar tests on exact correctness;
- Wilcoxon as a sensitivity check;
- Cohen's d and minimum detectable effects (MDE) for every headline cell;
- Holm-Bonferroni correction over the five pre-registered confirmatory
  hypotheses;
- Wilson intervals for proportions, ECE for calibration, and risk-coverage/AURC
  for selective prediction;
- explicit reporting of cell sizes, with cells below 30 items treated as
  descriptive only.

The confirmatory family is fixed in advance. Exploratory slices are labelled as
exploratory and are not described as "significant after correction".

### 4.7 The router

The router asks a practical question: can a cheap, **text-free** signal predict
when the VLM is worth its extra cost over OCR? It uses image-only and
declared-language features, measured latency, and `GroupKFold` by page. It must
not use transcript-derived features such as a script-mixing index computed from
the oracle text: that would leak the label side and produce an optimistic Pareto
curve. The router is evaluated on `C0` labels with features from the same input
image, and the cost is the measured wall-clock latency recorded per prediction,
not a hard-coded constant.

The router is a secondary contribution. It is reported only after the primary
read-reason claim, and only with out-of-fold predictions and a bootstrap CI on
the captured gap.

### 4.8 Provenance and reproducibility

Every prediction records the model ID and revision, quantization, prompt hash,
dataset revision, image hash, parser version, latency, token counts, and raw
output. The cache key includes all of these fields, so a prompt or model change
cannot silently reuse a stale prediction. The manifest records content hashes
and capability flags. The analysis script refuses synthetic predictions unless
explicitly overridden, and every figure is generated from the analysis outputs
rather than hand-edited.

This is deliberately more provenance than a typical research prototype. The
reason is simple: a diagnostic paper is only as credible as its ability to show
that the conditions, prompts, and models were the ones it claims.

## 5. Novelty and positioning

The space is not empty, and the README should not pretend otherwise.

- **DISCO** (Benkirane et al., 2026) compares OCR pipelines and VLMs across
  document types and finds that the better architecture depends on the document.
- **Xu et al.** (2026) perform interventional attribution for multi-page visually
  rich document understanding on English documents.
- **DocHallu / Hwang et al.** (Findings of ACL 2026) study perceptual
  hallucination under damaged document images and show that numerical content is
  especially vulnerable.
- **IndicVisionBench** and **Patram-Bench** already occupy the Indian document
  benchmark space.

Against that background, the contribution here is **not** "the first
counterfactual study" and **not** "the first Indian document benchmark". The
defensible differentiators are:

1. **A joint but architecture-specific ladder.** The same condition template is
   not reused across architectures. The cascade's `C2`/`C3` ordering is corrected,
   `C1` is explicitly undefined, and the extraction term is left joint rather
   than over-split. This is a measurement contribution: it prevents a reversed
   causal story.
2. **A transcript-scored read-reason probe.** `C1` is not merely "ask the model to
   read first". The transcript is stored and scored with CER/MER, and the analysis
   reports `P(read correct)` and `P(answer correct | read correct)`. That turns a
   vague dissociation story into a testable conditional quantity.
3. **Abugida-aware error measurement.** MER restricts combining-mark extraction
   to Indic blocks and reports it next to base-character CER, so a Latin
   diacritic cannot contaminate the Indic metric. The hypothesis that matras
   degrade faster than base characters is pre-registered, not post hoc.
4. **Negative controls plus clustered inference plus a leakage-guarded router.**
   Placebo crop, shuffled transcript, document-clustered bootstrap, grouped CV,
   and measured latency are integrated into one protocol rather than added as
   isolated checks.

The honest framing is: **a carefully specified measurement protocol with an
Indic focus and a promising mechanistic probe, not yet an empirical result.** The
novelty claim is only as strong as the confirmatory run that follows.

## 6. What is implemented today

| Area | State | Evidence |
| :--- | :--- | :--- |
| Protocol and pre-registration | Implemented | `docs/preregistration.md`; architecture-specific identities |
| Data adapters | Implemented for verified public data | IndicVisionBench OCR, FUNSD; Patram fails loudly |
| VLM and cascade runners | Implemented; confirmatory run pending | Capability-checked conditions, versioned prompts, answer parsing |
| Metrics | Implemented | ANLS, EM, CER, WER, base CER, MER, field F1, HAR, latency |
| Statistics | Implemented | Cluster bootstrap, paired tests, McNemar, Holm, Wilson, ECE, AURC, MDE |
| Router | Implemented; not run at scale | Text-free features, provenance guard, measured cost, GroupKFold |
| Reporting | Implemented | Tables and figures generated from predictions; synthetic guard |
| CI and tests | Green | 58 offline tests; GitHub Actions on Python 3.10 and 3.11 |
| Formal report | Complete | 26-page PDF and editable DOCX in `reports/` |
| Confirmatory evidence | **Not yet** | Requires PARAM Ganga and the frozen manifest |

The repository has **more than 30 commits since the original scaffold `8f18b0c`**
(32 at the time of this revision). The exact count is always
`git rev-list --count 8f18b0c..HEAD`.

## 7. What has actually been run (pilot only)

The laptop pilot exists to validate the measurement loop, not to test the
hypotheses. It used:

- **Windows.Media.Ocr** (`en-GB`/`en-US` only) — no Indic recognizers;
- **Qwen2.5-1.5B-Instruct Q4_K_M** via Ollama — far below the paper's 3B/7B
  readers;
- **FUNSD test** as real English forms, and **10 rendered synthetic invoice
  pages / 40 field questions** for controlled pipeline checks.

### 7.1 OCR degradation calibration

20 FUNSD documents, 26 conditions each, 520 OCR calls. Because Windows OCR is a
weak engine on low-quality fax forms, the absolute clean CER is high (0.597);
the useful signal is the **relative** increase over clean:

- Gaussian blur and resolution downscaling saturate quickly;
- JPEG and salt-and-pepper noise degrade gradually;
- the current perspective-skew operator is nearly flat and must be redesigned
  before it can serve as a severity axis.

This is a perturbation-calibration check. Absolute numbers are engine-specific
and are not paper findings.

### 7.2 Synthetic cascade validation

On 10 rendered pages and 40 questions, the corrected cascade moves from
`C0 = 0.733` to `C2 = 1.000` and `C3 = 1.000`, while severe degradation collapses
to `0.000`. That is the expected oracle-ceiling behavior and shows that the
condition plumbing, prompt injection, cache, and report path work. The line-level
`C3b` control is weak because it preserves label-value pairs; it must become a
sentence/word shuffle in the confirmatory run.

### 7.3 Real FUNSD cascade — a negative pilot

60 entity-derived questions from 20 FUNSD documents produced very low ANLS
(around 0.07). This is a **negative, question-quality-limited pilot**, not
evidence that the method fails: the questions came from a spatial heuristic,
many labels are ambiguous, Windows OCR misses many characters, and the reader is
1.5B. The useful outcome was engineering: the pilot found a real bug in which the
cascade's `C0` prompt described an image even though the reader only receives OCR
text, so the reader ignored the OCR. After the fix, the synthetic `C0` moved from
0.0 to 0.733.

**What the pilot cannot establish:** no VLM result, no Indic result, no
OCR-versus-VLM attribution, no human-validated hallucination labels, and no paper
claim about FUNSD difficulty or Qwen. The full caveats are in
[`results/local_pilot/README.md`](results/local_pilot/README.md).

## 8. Data status

| Dataset | What it contains | Status | Use in this project |
| :--- | :--- | :--- | :--- |
| **IndicVisionBench `ocr`** | 876 Indic document pages, 10 languages, gold transcripts; no QA, no boxes | Public, pinned revision | Multilingual recognition, CER/MER, `C0`/`C1`/`C3` with generated questions |
| **IndicVisionBench `vqa_indic`** | 1,007 cultural VQA rows over natural images; no transcript | Public | Multilingual visual-reasoning control only; **not DocQA** |
| **FUNSD** | 50 test forms, word boxes, BIO entity tags | Public, pinned revision | English/Latin layout control and field-value QA |
| **Patram-Bench** | ICDAR 2026 Indian document benchmark | Not publicly accessible at the configured ID on 6 Nov 2026; access request planned | Optional primary extension; never reported unless access is granted |

The public fallback is frozen so the study remains executable if Patram access is
delayed: IndicVisionBench OCR for the multilingual arm and FUNSD for the layout
control. The paper must state exactly which arm was used. No Patram numbers are
reported until there is a real, licensed access path.

Dataset/model revisions, licenses, and the audit commands are in
[`docs/data_provenance.md`](docs/data_provenance.md).

## 9. Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -e ".[data,gpu,ocr,dev]"   # CPU-only: pip install -e ".[dev]"

pytest -q                              # 58 offline tests; no network/GPU/data
python scripts/run_experiment.py --mode smoke --max-samples 3 --seed 42
```

The smoke run writes `outputs/smoke_report.json` with `reportable: false`. It is
a pipeline test, not evidence.

For real data, manifests, analysis tables, and the PARAM Ganga template, see
[`REPRODUCE.md`](REPRODUCE.md). For the laptop pilot:

```bash
python scripts/prepare_local_data.py --dataset funsd --max-rows 50
python scripts/prepare_local_data.py --dataset indic_ocr --per-language 3
python scripts/run_ocr_degradation_pilot.py --max-docs 20
```

## 10. Repository map

```text
configs/
  experiment_config.yaml    # model profiles and dataset revisions
  datasets.yaml             # verified revisions, licenses, supported conditions
  param_ganga_config.yaml   # 2x V100 FP16 profile
docs/
  peer_review.md            # adversarial review and venue assessment
  data_provenance.md        # verified datasets/models/literature
  preregistration.md        # hypotheses, endpoints, exclusions, analysis plan
  claim_evidence_matrix.md  # what may and may not be claimed today
  response_to_review.md     # issue -> fix mapping
  commit_series.md          # progressive commit log
paper/
  README.md, outline.md, claims.yaml
reports/
  Document_QA_Failure_Analysis_Project_Report.pdf
  Document_QA_Failure_Analysis_Project_Report.docx
  build_report.py, make_figures.py, assets/
scripts/
  run_experiment.py         # data -> conditions -> report
  analyze_results.py        # predictions + manifest -> tables
  build_probe_set.py        # frozen manifest
  audit_datasets.py         # live provenance audit
  run_ocr_degradation_pilot.py
src/
  data/                     # schema, adapters, QA generation, manifests
  perturbations/            # degradations, oracle/placebo crops
  models/                   # OCR cascade, VLM runner, prompts, parsing
  routing/                  # text-free features, measured-cost router
  evaluation/               # metrics, attribution, statistics
results/local_pilot/        # pilot artifacts and limitations
tests/                      # 58 offline tests
```

## 11. The confirmatory study

The target is a frozen 600-item diagnostic set:

- **500 Indic items** across at least five languages from IndicVisionBench OCR
  (or Patram-Bench if access is granted);
- **100 FUNSD items** as the Latin-script layout control;
- every supported condition on every item;
- a 5-family x 5-level perturbation sweep for `Cδ`;
- a 3B/7B VLM comparison, a Tesseract/PaddleOCR cascade comparison, and a 7B
  FP16 scale check.

The working estimate is **250 GPU-hours**; the request is **300 GPU-hours** on
one node with **2x V100 16 GB**, 12 CPUs, 64 GB RAM, 50 GB scratch, over 13 weeks.
The full plan, risk register, and endorsement page are in the
[formal report](reports/Document_QA_Failure_Analysis_Project_Report.pdf).

## 12. Limitations and claims policy

### Limitations that must appear in any paper

- No confirmatory results yet; the pilot uses a weak English-only OCR engine and
  a 1.5B reader.
- The Indic arm has no gold boxes; layout claims are tested on FUNSD or on a
  separately verified subset.
- Generated questions may not reflect natural information needs.
- English prompts over multilingual documents are a deliberate but limited
  setting.
- Line-level `C3b` is a weak control; it must be strengthened.
- Quantization may confound architecture comparisons; NF4/INT8/FP16 runs and seed
  variance are required.
- Patram-Bench access is unresolved; the public fallback is weaker on natural QA
  and layout.
- Human validation is planned but not yet performed.

### Claims allowed today

- "We implement a counterfactual measurement harness for OCR cascades and VLMs."
- "We provide a pre-registered protocol with architecture-specific
  decompositions, negative controls, and abugida-aware metrics."
- "We report a laptop pilot that validates the instrumentation and found a real
  cascade prompt bug."

### Claims not allowed today

- "Evaluated on Patram-Bench" or any Patram number.
- "State-of-the-art baselines."
- "Read-reason dissociation is demonstrated."
- "Leakage-free router" as an empirical result.
- "First" claims about Indic document attribution or benchmarks.
- Any acceptance probability or venue guarantee.

## 13. Roadmap

1. **Weeks 1-2:** secure PARAM Ganga access; submit the Patram-Bench request;
   freeze the public manifest and perturbation calibration.
2. **Weeks 3-4:** run FUNSD as the integration test; verify every condition and
   the coverage report.
3. **Weeks 4-6:** run the 3B VLM and the OCR cascades; produce the first
   condition table with clustered CIs.
4. **Weeks 7-9:** run the 7B FP16 scale check and the quantization comparison.
5. **Weeks 9-11:** human-validate 100 failures with two bilingual volunteers;
   compute agreement and the final HAR analysis.
6. **Weeks 11-13:** freeze tables and figures; write the paper around **one**
   mechanistic claim; release the artifact.

## 14. Documentation index

- [Formal report (PDF)](reports/Document_QA_Failure_Analysis_Project_Report.pdf)
- [Formal report (DOCX)](reports/Document_QA_Failure_Analysis_Project_Report.docx)
- [Pre-registration and analysis plan](docs/preregistration.md)
- [Verified data and model provenance](docs/data_provenance.md)
- [Claim-evidence matrix](docs/claim_evidence_matrix.md)
- [Adversarial peer review](docs/peer_review.md)
- [Response to review](docs/response_to_review.md)
- [Progressive commit series](docs/commit_series.md)
- [Reproduction guide](REPRODUCE.md)
- [Local pilot results and caveats](results/local_pilot/README.md)
- [Paper outline](paper/outline.md)

## 15. Glossary

- **ANLS:** average normalized Levenshtein similarity; a soft string-match metric
  for QA.
- **CER / WER:** character/word error rate between a hypothesis and a gold
  transcript.
- **MER:** Matra Error Rate; edit distance restricted to Indic combining marks.
- **HAR:** hallucination-to-abstention ratio; confident wrong answers divided by
  explicit refusals, reported with counts.
- **`C1`:** the read-reason probe: transcribe first, answer from the transcript,
  and score the transcript separately.
- **`C2b`:** placebo crop; a matched-size crop that does not contain the answer,
  used to subtract the generic effect of cropping.
- **`C3b`:** shuffled transcript; same text in a wrong order, used to test
  whether correct content or merely "some text" drives the oracle gain.
- **Document-clustered bootstrap:** resampling pages, not individual questions, so
  multiple questions per page do not create false precision.

## 16. License and citation

Code is MIT-licensed ([`LICENSE`](LICENSE)). Dataset and model terms are their
own; see [`docs/data_provenance.md`](docs/data_provenance.md). Citation metadata
is in [`CITATION.cff`](CITATION.cff), but **do not cite this as a validated
result** until the confirmatory run exists.

The most useful citation today is to the protocol and harness, not to a finding:

> Siddhartha. *Where Does Document Understanding Break? A counterfactual
> error-attribution harness for multilingual document QA.* Research artifact,
> 2026. https://github.com/sidg2hp/Multilingual-document-QA-failure-analysis
