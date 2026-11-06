# Paper Outline

Working title: **"Where Does Indic Document QA Break? A Counterfactual Study of
OCR and OCR-Free VLMs"**

> Do not write final prose until the confirmatory run exists. This outline marks
> what each section needs and where the evidence comes from.

## Abstract (150–200 words)

- One sentence on the gap: aggregate DocQA scores hide whether errors come from
  recognition, localization, or reasoning, and the evidence is mostly English.
- One sentence on the protocol: architecture-specific counterfactual ladder with
  a placebo crop and shuffled transcript, applied to Indic and Latin documents.
- One sentence on the headline result (must be filled from the run; no placeholders
  in a submitted version).
- One sentence on the implication for deployment or evaluation.

**Do not** claim a new benchmark. **Do not** claim SOTA.

## 1. Introduction

- Motivation: multilingual document QA deployments in India; failure diagnosis
  matters more than a single accuracy number.
- Gap: DISCO compares OCR and VLMs but does not isolate stages per architecture;
  Xu et al. do interventional attribution on English multi-page VRDU; DocHallu
  studies degradation hallucinations but not Indic scripts or OCR cascades.
- Contributions (3, in order):
  1. an architecture-specific counterfactual protocol with negative controls;
  2. a transcript-scored read–reason probe and matra-aware error analysis on
     real Indic document pages;
  3. a leakage-free, cost-aware routing evaluation on measured latency.
- Scope statement: generated extractive questions over real transcripts; layout
  claims limited to the FUNSD control unless Indic boxes are obtained.

## 2. Related work

- DocQA benchmarks: DocVQA, InfographicVQA, FUNSD, XFUND, Patram-Bench,
  IndicVisionBench. State exactly which split is used.
- OCR vs VLM pipelines: DISCO; report its findings and differentiate.
- Error attribution: Xu et al.; oracle-text mediation; why the cascade's
  extraction term is joint.
- Hallucination under degradation: DocHallu; HAR; numerical content.
- Routing / cascade systems: cost-aware selection; why hard-coded costs and
  oracle features are insufficient.

## 3. Data

- IndicVisionBench OCR: 876 pages, 10 languages, transcripts, no QA/boxes.
- FUNSD test: 50 forms, entity boxes, English control.
- Patram-Bench: state access status; do not report numbers if unavailable.
- Question generation: templates, uniqueness constraint, English-query scope,
  unanswerable probes, human verification sample.
- Manifest: hashes, revisions, counts, exclusions.

## 4. Protocol

- Conditions per architecture (table).
- Why C1 is undefined for the cascade.
- Placebo crop and shuffled transcript.
- Identified decompositions (VLM vs cascade).
- What is *not* identified and why.
- Prompts: shared system/answer rule; versioned prompt IDs/hashes.

## 5. Models and implementation

- Exact model IDs, revisions, licenses, precision, quantization, max pixels/tokens.
- OCR engines and language packs.
- Hardware, batch size, decoding, seeds.
- Cache and resume policy; no silent reuse.

## 6. Metrics and statistics

- ANLS, EM, CER/WER, base CER, MER, HAR, failure modes.
- Document-clustered bootstrap, paired permutation, McNemar, Holm, MDE.
- Human validation and agreement.

## 7. Results

- Table 1: model × condition with CIs and coverage.
- Table 2: decomposition by architecture; negative components retained.
- Figure 1: degradation curves (ANLS and CER/MER) with CIs.
- Figure 2: read–reason dissociation (transcript CER vs QA gain), per script.
- Table 3: per-language/script cells with n and MDE.
- Table 4: HAR and human agreement.
- Figure 3: router accuracy–cost curve with measured latency and grouped CV.
- Every result labelled confirmatory or exploratory.

## 8. Discussion

- What the decomposition does and does not identify.
- Why generated questions limit external validity.
- Deployment implication (if any) with cost caveats.
- Negative results and surprises.

## 9. Limitations

- Data access and scope; generated questions; English prompts.
- 3B/4-bit models vs 7B/FP16; single hardware profile.
- Layout oracle only where boxes exist.
- Human validation sample size.
- No claim about all 23 languages or all document types.

## 10. Ethics and data statement

- Licenses, redistribution, PII, Wikisource provenance, annotator compensation.
- Model licenses (Qwen research license; Patram Apache-2.0 but gated).

## Appendix

- A: prompts and template IDs.
- B: manifest hash and counts.
- C: perturbation calibration table and induced CER.
- D: per-item examples (success/failure).
- E: compute budget and runtime.
- F: negative results and deviations from pre-registration.

## Submission checklist

- [ ] Confirmatory run completed; `reportable=true`.
- [ ] Manifest frozen and hashed.
- [ ] All conditions reported, including placebos and negatives.
- [ ] Cluster-aware CIs and paired tests for every headline claim.
- [ ] Human validation with agreement.
- [ ] 7B FP16 scale check (or an explicit limitation).
- [ ] Artifact and reproduction script.
- [ ] Ethics/limitations/data statement.
- [ ] No claim that the data do not support.
