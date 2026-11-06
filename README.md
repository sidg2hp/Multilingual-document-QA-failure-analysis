# Where Does Document Understanding Break?

**A counterfactual error-propagation harness for OCR-based and OCR-free document QA on multilingual documents.**

> **Status (6 Nov 2026): research harness, not a paper, and not a validated empirical result.**
> There has been no VLM confirmatory run. The smoke pipeline produces `[SMOKE]`
> outputs and marks its report `reportable: false`. A **laptop pilot** (Windows
> OCR + Qwen2.5-1.5B, no VLM, no annotator) is in
> [`results/local_pilot/`](results/local_pilot/README.md); it is explicitly not
> paper evidence. Read [`docs/peer_review.md`](docs/peer_review.md) for the
> adversarial review and a calibrated venue assessment before citing anything.

> **Formal report (6 Nov 2026):** [Project report and PARAM Ganga compute request](reports/Document_QA_Failure_Analysis_Project_Report.pdf) — 26-page PDF for faculty review.

[![CI](https://github.com/sidg2hp/Multilingual-document-QA-failure-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/sidg2hp/Multilingual-document-QA-failure-analysis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this is

A measurement harness for asking **where** a document QA system fails, not just
how often:

- visual degradation,
- localization / reading order,
- script and character recognition (including Indic matras),
- semantic / cross-lingual reasoning.

It implements an **architecture-specific counterfactual ladder** for both
cascaded OCR+LLM pipelines and OCR-free VLMs, with explicit negative controls, a
transcript-scored readÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“reason probe, matra-aware metrics, clustered statistics,
a leakage-guarded router, and a provenance-aware prediction cache.

## What this is not

- It is **not** a new benchmark, model, or leaderboard entry.
- It is **not** an accepted paper. The current artifact has no confirmatory
  results, and its original data plan was not executable (see below).
- It does **not** support claims about Patram-Bench, 23 languages, or "first"
  Indic attribution; those claims were removed.

## Verified data status

| Dataset | What it actually contains | Status | Use here |
| :--- | :--- | :--- | :--- |
| **IndicVisionBench `ocr`** | 876 Indic document pages with gold transcripts, 10 languages; **no QA, no boxes** | Public, revision `45269760ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦` | Transcript-grounded generated QA; CER/MER and readÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“reason conditions |
| **IndicVisionBench `vqa_indic`** | 1,007 cultural VQA rows over **natural images**, no transcript | Public, same revision | Multilingual visual-reasoning control only; **not DocQA** |
| **FUNSD** | 50 test forms, word boxes, BIO entity tags | Public, revision `7e7eeeeeÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦` | English/Latin layout control; field-value QA derived from entities |
| **Patram-Bench** | ICDAR 2026 paper; no public HF dataset at the configured ID (HTTP 401 on 2026-11-06) | **Blocked** | Not used until the authors release it |

The audit is reproducible:

```bash
python scripts/audit_datasets.py --out outputs/provenance/dataset_audit.json
```

Details, licenses, and model revisions are in
[`docs/data_provenance.md`](docs/data_provenance.md).

## Protocol

### OCR-free VLM

| Condition | Input | Identifies |
| :--- | :--- | :--- |
| `C0` | clean image + question | baseline |
| `C_ÃƒÅ½Ã‚Â´` | perturbed image + question | degradation sensitivity |
| `C1` | transcribe, then answer from own transcript (transcript scored) | readÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“reason dissociation |
| `C2` | gold answer-region crop | localization + clutter removal |
| `C2b` | placebo crop of the same size | crop/context control |
| `C3` | gold transcript | recognition oracle |
| `C3b` | shuffled gold transcript | transcript-content control |

Identified decomposition:

```text
E_total = (C0 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ CÃƒÅ½Ã‚Â´) + (C2 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ C0) + (C3 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ C2) + (1 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ C3)
```

### Cascaded OCR + text LLM

`C1` is undefined (C0 already reads then reasons). `C2` is **coordinate gold
text**, not an image crop. The identified decomposition is:

```text
E_total = (C0 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ CÃƒÅ½Ã‚Â´) + (C3 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ C0) + (1 ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ C3)
```

The middle term is **joint extraction** (recognition + reading order); it is not
split. `C2_coord ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ C3_flat` is reported separately as a layout-representation
diagnostic. Missing conditions are `None`, never zero and never the gold answer.

## What is implemented

- Architecture-specific attribution, signed components, exact identity, and
  condition-coverage reporting.
- CER, WER, base-character CER, Matra Error Rate (Indic-only marks), ANLS, EM,
  field F1, failure modes, and HAR with counts/saturation.
- Document-clustered bootstrap, paired permutation, McNemar, Wilcoxon, Cohen's d,
  HolmÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“Bonferroni, Wilson intervals, ECE, riskÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“coverage/AURC, and MDE.
- Versioned prompts and answer parsing; cache keys include prompt hash, model
  revision, dataset revision, and image hash.
- Capability-checked runners that raise on unsupported conditions instead of
  silently falling back.
- A leakage-guarded router: text-free/declared-language features, measured
  latency, and GroupKFold by page.
- Frozen manifests, a provenance audit, and an analysis script that refuses
  synthetic predictions.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[data,gpu,ocr,dev]"   # CPU-only: pip install -e ".[dev]"

pytest -q                              # no network, GPU, or datasets needed
python scripts/run_experiment.py --mode smoke --max-samples 3 --seed 42
```

The smoke run writes `outputs/smoke_report.json` with `reportable: false`. It is
a pipeline test, not evidence.

See [`REPRODUCE.md`](REPRODUCE.md) for the real-data workflow, the frozen
manifest, the analysis tables, and the PARAM Ganga template.

## Repository map

```text
configs/
  experiment_config.yaml   # models, precision, dataset revisions
  datasets.yaml            # verified revisions, licenses, supported conditions
  param_ganga_config.yaml  # 2x V100 FP16 profile
docs/
  peer_review.md           # adversarial review, venue assessment, defect index
  data_provenance.md       # verified datasets/models/literature
  preregistration.md       # hypotheses, endpoints, exclusions, analysis plan
  claim_evidence_matrix.md # claims vs. evidence today
  research_protocol.md     # original protocol (partially superseded)
  paper_readiness_and_improvements.md
paper/
  README.md, outline.md, claims.yaml
scripts/
  run_experiment.py        # data -> conditions -> report
  analyze_results.py       # predictions + manifest -> tables
  build_probe_set.py       # frozen manifest
  audit_datasets.py        # provenance audit
src/
  data/                    # schema, adapters, QA generation, manifest
  perturbations/           # degradations, oracle/placebo crops
  models/                  # OCR cascade, VLM runner, prompts, parsing
  routing/                 # text-free features, cost-aware router
  evaluation/              # metrics, attribution, statistics
tests/                     # 58 offline tests
```

## Local pilot on the RTX 3050 (not paper evidence)

Until PARAM Ganga access arrives, the repository includes a hardware-constrained
pilot path:

- **Windows OCR** (`Windows.Media.Ocr`, English en-GB/en-US only) + **Qwen2.5-1.5B
  Q4_K_M** via Ollama for a cascaded pipeline.
- **Real FUNSD** forms via a local rows/images cache (no `datasets`/PyArrow).
- **Controlled synthetic forms** for pipeline validation only.
- An **OCR-only degradation pilot** that measures CER/WER under the five
  perturbation families.

```bash
python scripts/prepare_local_data.py --dataset funsd --max-rows 50
python scripts/prepare_local_data.py --dataset indic_ocr --per-language 3
python scripts/run_experiment.py --config configs/local_cascade.yaml \
  --mode funsd_local --models ocr --max-samples 60 \
  --manifest manifests/local_funsd.json --resume
python scripts/run_ocr_degradation_pilot.py --max-docs 20
```

Caveats are not optional: Windows OCR has no Indic recognizers, Qwen-1.5B is not
a paper-scale reader, and the synthetic forms are never evidence. The pilot
results and their limitations are in
[`results/local_pilot/README.md`](results/local_pilot/README.md).

## What still must happen before a paper

1. Obtain Patram-Bench or finalize the public-data scope; freeze the manifest.
2. Run real inference on ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥600 items, both pipelines, all supported conditions,
   with calibrated perturbations and a 7B FP16 scale check.
3. Add human validation of 100 failure cases and report agreement.
4. Replace every placeholder with numbers generated by `scripts/analyze_results.py`.
5. Write around **one** mechanistic claim (readÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“reason dissociation, matra
   vulnerability, or routing), not around the framework.

The honest assessment is in [`docs/peer_review.md`](docs/peer_review.md):
the current artifact is not submittable; a strong workshop paper is plausible
after a clean, pre-registered study.

## License and citation

Code is MIT-licensed ([`LICENSE`](LICENSE)). Dataset and model terms are their
own; see [`docs/data_provenance.md`](docs/data_provenance.md). Citation metadata
is in [`CITATION.cff`](CITATION.cff), but do not cite this as a validated result
until the confirmatory run exists.
