# Response to the Adversarial Review

This document maps the issues raised in `docs/peer_review.md` to the commits that
address them, and states what is still open. It is deliberately not a claim that
the paper is ready.

## Resolved in this repository

| Review issue | Fix | Commit |
| :--- | :--- | :--- |
| No executed results; smoke outputs treated as results | Smoke mode is explicit, emits `[SMOKE]`, sets `reportable: false`, and the analysis script refuses synthetic predictions | `fix(runtime)`, `feat(report)`, `feat(analysis)` |
| Patram-Bench HF ID does not exist | Loader fails loudly; local-export adapter; config marks the dataset unavailable; audit confirms HTTP 401 | `fix(data)`, `feat(data)` |
| IndicVisionBench `vqa_indic` misused as DocQA | Separate `ocr` and `vqa` adapters; VQA has `supports_text_oracle=False` and is marked not document QA | `fix(data)` |
| C3 silently receives the gold answer | C3/C3b require a verified transcript; unsupported conditions raise | `fix(runtime)` |
| Reversed C2/C3 ladder for cascades | Architecture-specific decompositions; cascade's extraction term is joint; layout-representation effect is a separate diagnostic | `fix(eval)` |
| Missing conditions scored as 0 | Condition scores are `Optional`; summaries report coverage; decompositions require complete subsets | `fix(eval)` |
| C1 transcript never scored | C1 stores the transcript and computes CER/MER; `c1_read_correct` is thresholded and reported | `fix(eval)`, `fix(runtime)` |
| C2 crop is a bundled intervention | Added C2b placebo crop; `delta_lay_net` reported | `fix(eval)`, `fix(runtime)` |
| No transcript-content control | Added C3b shuffled transcript | `fix(eval)`, `fix(runtime)` |
| Router trained on oracle SMI | Feature provenance guard; oracle source raises; declared-language only | `fix(routing)` |
| Router features from clean image, labels from degraded | Router evaluated on C0 labels with clean-image features; measured latency; GroupKFold by page | `fix(routing)`, `feat(report)` |
| Hard-coded router costs | Measured per-item latency supported; cost model labelled | `fix(routing)` |
| Raw generations scored | Versioned answer parser; raw output preserved | `fix(runtime)` |
| Prompt inconsistency across conditions | Shared system/answer rule; versioned PromptSpecs; prompt hash in cache key | `fix(runtime)` |
| Cache can silently reuse stale predictions | Cache key includes prompt hash, model/dataset revision, image hash; old schema ignored | `fix(runtime)` |
| No dataset/model revisions or manifest | Pinned revisions, manifest builder with content hashes, provenance audit | `fix(data)`, `feat(data)` |
| FUNSD token-index question | Entity-derived field-value questions with derived pairing caveat | `fix(data)` |
| i.i.d. bootstrap ignores documents | Cluster bootstrap by page; grouped CV; McNemar/permutation | `fix(eval)` |
| HAR saturation returns a fake ratio | HAR summary with counts and `saturated` flag | `fix(eval)` |
| Indic MER contaminated by Latin diacritics | Indic-block-restricted mark extraction; base CER; breakdown | `fix(eval)` |
| Dry-run fabricates ground truth | Removed; explicit `[SMOKE]` | `fix(runtime)` |
| Single model / hard-coded NF4 | Config-driven runner; model registry fixes; precision parameter | `fix(runtime)`, `feat(report)` |
| No CI/license/lockfile | MIT license, pyproject extras, GitHub Actions, reproduction guide | `chore(repro)` |
| Unsupported acceptance estimate | Withdrawn; calibrated table and pre-registration | `docs` |

## Still open (cannot be fixed by code alone)

1. **Patram-Bench access.** The dataset is not public at the configured ID. The
   paper cannot use it until the authors release it or the scope changes.
2. **Real confirmatory inference.** No model has been run on real data. The
   harness is ready; the GPU time is not spent.
3. **Perturbation calibration.** The five levels are still absolute parameters;
   they must be calibrated to induced CER and frozen.
4. **Human validation.** The confident-hallucination proxy has not been checked
   against bilingual annotators.
5. **7B FP16 scale check.** The code path exists; the PARAM Ganga run does not.
6. **Paper writing.** `paper/` is an outline, not a draft.

## What would change the verdict

A completed 600-item study with document-clustered CIs and one replicated
mechanistic claim would move the work from "not submittable" to a plausible
strong-workshop submission. More code without that evidence will not.
