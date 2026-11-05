# Paper Readiness Assessment & Implemented Improvements

This document has two purposes:

1. Give an honest, calibrated assessment of whether the current project can be
   accepted at a good AI conference workshop if written up as a paper.
2. Record the concrete code and documentation changes made to this repository to
   close the biggest gaps between "interesting protocol" and "publishable study".

---

## 1. Executive verdict

### 1.1 As the repository stands today

**No venue will accept the project in its current form, because there are no
executed results.** The repository is a well-structured *protocol and harness*:
dataset loaders, perturbation operators, counterfactual conditions, metrics, a
router, and a dry-run pipeline. The dry-run mode fabricates predictions that are
almost always equal to the ground truth, so the reported numbers are not
evidence about any real model.

This is not a criticism of the idea. It is a statement about evidence: a paper
needs measured model behaviour, not a demonstration that the measurement loop
runs.

### 1.2 If the planned experiments are actually executed

Assuming a clean, completed run on the 600-document diagnostic slice described
in `docs/research_protocol.md`, with real model outputs, confidence intervals,
and the ablations described below:

| Venue tier | Examples | Realistic odds | Main condition for acceptance |
| :--- | :--- | :--- | :--- |
| Main conference (NeurIPS/ICLR/CVPR/ACL) | — | <5% | Not competitive: 3B quantized models, modest benchmark slice, incremental framing |
| Strong domain workshop | ICDAR workshops, CVPR VDU, ACL/EMNLP Document Intelligence, NeurIPS/ICLR MRL workshops | 45–60% | A crisp, surprising finding (e.g. read–reason dissociation or quantization-specific Indic degradation) plus honest statistics |
| General AI workshop / student track | AAAI/ACL student workshops, regional venues | 60–75% | Clean execution and a reproducible artifact; novelty can be modest |

The single highest-leverage change is **not more code**. It is producing one
clear empirical claim that survives controls. The current five-condition design
is capable of producing such a claim; it has not yet produced one.

### 1.3 What is genuinely promising in the design

- **Five nested counterfactual conditions** applied to both cascaded OCR and
  OCR-free VLMs, rather than oracle-text injection for OCR pipelines only.
- **Read–reason dissociation probe ($C_1$)**: asking the VLM to transcribe
  first and reason second. If $C_1 \gg C_0$, the bottleneck is cross-modal
  grounding, not the vision encoder. That is a publishable, mechanistic result.
- **Matra-aware error analysis**: a metric family tailored to Indic abugidas,
  where Unicode combining marks carry lexical and numeric meaning.
- **Confident hallucination vs. abstention (HAR)**: a failure-mode axis that
  aggregate ANLS hides.
- **Quantization as an experimental variable** (NF4 vs. INT8 vs. FP16): turns a
  hardware constraint into a research question.
- **Cost-aware OCR/VLM router** with an accuracy–latency Pareto analysis: gives
  the work a practical contribution beyond diagnosis.

### 1.4 Predictable reviewer objections (and what must be done)

1. *"This is a benchmark reproduction on small quantized models."*
   - Mitigate with a mechanism claim, not a leaderboard claim. Report
     read–reason dissociation, HAR, and quantization deltas. Include at least
     one 7B FP16 model on PARAM Ganga.
2. *"Oracle text only tests the text-LLM half."*
   - Already addressed in principle by $C_1$ and $C_2$, but both must be
     reported for OCR-free models, not just implemented.
3. *"No statistical evidence."*
   - Add paired bootstrap confidence intervals, Wilcoxon signed-rank tests, and
     multiple-comparison correction (implemented in this repository now).
4. *"The router is trained and evaluated on the same samples."*
   - This was true before these changes. The router now supports cross-validated
     out-of-fold evaluation; use it for every reported Pareto number.
5. *"Dry-run numbers are synthetic."*
   - Never report dry-run numbers in a paper. Use them only for pipeline tests.
6. *"Single dataset / single seed / no human check."*
   - Report at least three seeds where decoding is stochastic, separate
     calibration and test splits, and manually inspect a stratified sample of
     confident hallucinations.
7. *"English-only forms (FUNSD) do not test Indic claims."*
   - Keep FUNSD as a Latin-script control. Do not use it to support claims about
     Indic scripts.
8. *"Licensing / provenance."*
   - Document dataset licenses, snapshot hashes, and exact loader configuration.

### 1.5 Minimum experiment package for a credible workshop submission

- 600-sample frozen diagnostic set (as specified in `research_protocol.md`),
  with a saved manifest of sample IDs and dataset revisions.
- Real inference for at least: Tesseract + Qwen2.5-3B, PaddleOCR + Qwen2.5-3B,
  Qwen2.5-VL-3B (NF4 and INT8), SmolVLM (NF4/INT8/FP16).
- All five conditions ($C_0, C_\delta, C_1, C_2, C_3$) for every VLM.
- Five perturbation levels for each of at least four perturbation families.
- Per-language and per-script breakdowns, with confidence intervals.
- Paired significance tests for the central comparisons.
- A qualitative appendix containing matched success/failure examples.
- Released code, configs, frozen sample IDs, and prompts.

---

## 2. Improvements implemented in this repository

The following changes were made to convert the harness into a study that can
support paper claims. Each subsection states the problem, the change, and the
files touched. A commit-by-commit summary is at the end of the document.

### 2.1 Correct, identity-preserving error decomposition

**Problem.** The original `decompose_telescoping_error` mixed up the roles of
$C_2$ (oracle layout crop) and $C_3$ (oracle text). The arithmetic identity held,
but the components were labelled as if $C_3$ measured layout error and $C_2$
measured recognition error. A paper would have reported the opposite causal
story.

**Change.** The decomposition now follows the nested oracle order explicitly:

```text
degraded -> clean -> oracle-layout (C2) -> oracle-text (C3)
E_total = (acc_clean  - acc_degraded)   perturbation loss
        + (acc_C2     - acc_clean)      layout/localization loss
        + (acc_C3     - acc_C2)         recognition loss
        + (1          - acc_C3)         semantic reasoning loss
```

The function reports the identity check, whether the sequence is monotonic, and
signed components. Negative components are retained because they are findings
(an oracle intervention hurt performance), not errors to hide.

### 2.2 Hallucination-to-abstention ratio and failure modes

**Problem.** HAR was described in the protocol but never computed. A pipeline
that abstains and one that confidently invents a number were both just "wrong".

**Change.** Added explicit failure-mode classification (`correct`, `abstained`,
`confident_hallucination`), HAR aggregation, and a conservative
`is_indic_confident_hallucination` helper. The helper is deliberately named and
documented as a proxy: it flags confident, high-edit-distance answers on Indic
references; it does not prove that a matra was the specific cause.

### 2.3 Statistical rigor

**Problem.** The original code compared mean ANLS values with no uncertainty
estimates and no significance testing. Workshop reviewers routinely reject this.

**Change.** Added `src/evaluation/statistics.py` with:

- percentile bootstrap confidence intervals;
- paired bootstrap tests for model differences;
- Wilcoxon signed-rank tests when SciPy is available;
- paired Cohen's $d$ effect size;
- Holm–Bonferroni correction.

The experiment runner now reports bootstrap intervals for every headline metric
and paired tests for OCR-vs-VLM comparisons.

### 2.4 Leakage-free router evaluation

**Problem.** `run_experiment.py` fitted the router and then evaluated the same
samples, producing optimistic Pareto curves.

**Change.** `AdaptiveDocQARouter.evaluate_pareto_frontier_cv` performs
cross-validated out-of-fold score prediction before routing. Reports include the
evaluation protocol (`cross_validated` vs `in_sample`) so the number cannot be
silently overstated.

### 2.5 Resumable, cache-backed inference

**Problem.** On a 6 GB GPU or a shared HPC queue, a crash or preemption forced a
full re-run of expensive inference.

**Change.** Added a JSONL prediction cache keyed by model, quantization, sample,
condition, perturbation type, and severity. Runs can be resumed with
`--resume`, and repeated conditions are skipped.

### 2.6 Perturbation sweep with shared baselines

**Problem.** The original runner evaluated a single perturbation/severity pair,
and would recompute $C_0, C_1, C_2, C_3$ for every perturbation if extended
naively.

**Change.** Added `scripts/run_sweep.py`, which computes the perturbation-
independent conditions once per sample and only recomputes $C_\delta$ for each
perturbation family and severity. This reduces inference calls by roughly 4x and
produces a CSV/JSON degradation curve suitable for plotting.

### 2.7 Correct multimodal loading

**Problem.** `AutoModelForVision2Seq` is not the correct loader for Qwen2.5-VL,
and the processor call did not use `qwen_vl_utils.process_vision_info`, so the
reported VLM pipeline could fail or silently use the wrong preprocessing.

**Change.** Added architecture-aware model resolution (Qwen2.5-VL class when
available, `AutoModelForImageTextToText` fallback, legacy
`AutoModelForVision2Seq` last), optional `process_vision_info`, and explicit
generation settings.

### 2.8 Reproducibility

**Problem.** Seeds existed in configs but were not applied globally, and output
reports did not record the protocol used.

**Change.** Added a global seeding helper and recorded protocol metadata and
cache usage in experiment reports.

---

## 3. What still must be done before submission

These are intentionally *not* solved by code changes alone:

1. Run real inference on the frozen diagnostic set and replace every dry-run
   number.
2. Recruit one annotator to verify a sample of "confident hallucination" cases;
   report inter-annotator agreement if two annotators are available.
3. Produce the qualitative appendix and the degradation-curve figure.
4. Add at least one 7B FP16 model from the PARAM Ganga plan.
5. Write the paper around one mechanistic claim, not around the framework.
6. Archive the exact dataset revisions and model commit hashes.

---

## 4. Commit-by-commit change log

See the repository history. The commits are ordered so each one is reviewable:

1. `docs: add paper readiness assessment and improvement plan`
2. `fix(eval): correct telescoping decomposition and add HAR failure modes`
3. `feat(eval): add bootstrap, paired tests, and multiple-comparison correction`
4. `feat(runtime): add resumable cache, perturbation sweep, and VLM loader fix`
5. `fix(routing): evaluate router Pareto frontier with cross-validation`
6. `test: cover statistics, failure modes, cache, and CV routing`
7. `docs: finalize improvement changelog and README usage`
