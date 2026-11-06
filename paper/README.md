# Paper Plan

**Status: no paper draft and no results.** This directory is a plan for the
paper that the harness could support after a real confirmatory run. Writing the
paper before the data is a recipe for a desk reject; see `docs/peer_review.md`.

## Target

- **Primary target:** a strong domain workshop (ICDAR workshops, WACI, CVPR
  VDU, ACL/EMNLP Document Intelligence) or Findings.
- **Stretch:** main track, only if the primary claim replicates across scripts,
  models, and seeds and the effect is large.
- **Do not target:** a main-track submission built on 3B NF4 models, 600 items,
  generated questions, and no human validation.

## One-sentence claims (pick one after the run)

1. **Read–reason dissociation.** "Indic OCR-free VLMs often transcribe the
   answer-bearing span correctly yet fail to use it when answering directly; the
   gap is larger for Devanagari/Bengali/Tamil than for Latin and is not explained
   by transcript CER."
2. **Matra vulnerability.** "Under blur and downscaling, recognition errors
   concentrate in Indic combining marks relative to base characters, and 4-bit
   quantization amplifies that concentration."
3. **Routing.** "A text-free image router captures a substantial fraction of the
   OCR/VLM oracle gap at lower measured cost, and the gain survives
   document-grouped out-of-fold evaluation."

A paper with one of these, cleanly measured, is stronger than a paper with all
three measured badly.

## Required evidence

- Frozen manifest with ≥600 items, ≥100 documents, ≥5 Indic languages plus a
  Latin control.
- Real predictions for at least one 3B VLM and one OCR cascade; ideally one 7B
  FP16 model.
- All supported conditions, including placebo crop and shuffled transcript.
- Document-clustered CIs, paired permutation/McNemar, Holm correction over the
  pre-registered family, and MDE for every cell.
- Human validation of 100 failure cases with agreement.
- Released prompts, manifest, predictions, analysis script, and environment.

## Files

- `outline.md` — section-by-section plan with data sources and TODOs.
- `claims.yaml` — machine-readable claim/evidence/experiment map.
- `../docs/preregistration.md` — hypotheses, endpoints, exclusions.
- `../docs/claim_evidence_matrix.md` — what is and is not supported today.
- `../docs/peer_review.md` — adversarial review and venue assessment.

## Rule

Every number in the abstract, tables, and figures must be reproducible by
`scripts/analyze_results.py` from a released prediction file and manifest. If a
number cannot be traced to that pipeline, it does not go in the paper.
