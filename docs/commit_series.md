# Progressive Commit Series

This repository was changed in small, reviewable steps. The series below is the
actual local history; the table records the intent of each commit so the
progression is auditable even before the branch is pushed.

**Current state:** the phase series below is published on both local `main` and
`origin/main`, after the original scaffolding commit `8f18b0c`. The exact count
is whatever `git rev-list --count 8f18b0c..HEAD` reports; this document is part
of the series.

## Phase 0 — Evidence and review

| Commit | Purpose |
| :--- | :--- |
| `docs: add adversarial peer review, data provenance, and preregistration` | Mock reviews, calibrated venue assessment, verified datasets/models, analysis plan, claim–evidence matrix. |

## Phase 1 — Make the measurement valid

| Commit | Purpose |
| :--- | :--- |
| `fix(eval): make attribution architecture-specific and stop imputing missing conditions` | Separate VLM and cascade ladders; placebo/sham controls; C1 transcript scoring; no gold-answer fallback. |
| `fix(eval): make metrics script-aware and statistics cluster-aware` | Indic-only MER; base CER; cluster bootstrap; permutation/McNemar/ECE/AURC/MDE. |

## Phase 2 — Make the data executable

| Commit | Purpose |
| :--- | :--- |
| `fix(data): replace unverified loaders with schema-checked adapters and a frozen manifest` | Correct IndicVisionBench split use, FUNSD entity QA, Patram fail-loud, manifests. |
| `feat(data): add provenance audit and frozen probe-set builder` | Live HF API audit; reproducible manifests. |

## Phase 3 — Make inference honest

| Commit | Purpose |
| :--- | :--- |
| `fix(runtime): version prompts, parse answers, and key the cache by provenance` | Versioned prompts, answer parsing, capability checks, cache provenance. |
| `fix(runtime): preserve multi-line C1 transcripts and guard single-unit C3b` | Correct C1 transcript handling; no no-op control. |
| `fix(runtime): hash the actual crop and text-only inputs in cache lookups` | Cache keys match the real model input. |
| `fix(runtime): align C2 cache hashing with truncated coordinate text` | Cascade C2 matching. |
| `fix(runtime): take default precision from the model config` | FP16 profile no longer silently runs NF4. |

## Phase 4 — Router and report

| Commit | Purpose |
| :--- | :--- |
| `fix(routing): remove oracle leakage and evaluate with measured costs and document groups` | Text-free features; measured cost; GroupKFold. |
| `feat(report): rewrite the runner around provenance, capability checks, and honest reporting` | Smoke gating, provenance, ID-joined tests, architecture-specific decomposition. |

## Phase 5 — Analysis and reproducibility

| Commit | Purpose |
| :--- | :--- |
| `feat(analysis): generate result tables and paired tests from released predictions` | Tables from JSONL + manifest; refuses synthetic data. |
| `fix(analysis): write sweep predictions and select a single setting for condition tables` | Audit trail for sweeps; no duplicated baselines. |
| `feat(analysis): generate figures from the analysis tables` | Figures cannot diverge from tables. |
| `chore(repro): add packaging, CI, license, and a runnable PARAM Ganga profile` | Package, CI, MIT license, HPC template. |
| `docs(paper): align the README, review response, and paper plan with the actual evidence` | README, response-to-review, paper outline. |

## Phase 6 — Laptop pilot

| Commit | Purpose |
| :--- | :--- |
| `feat(local): add an Ollama pilot path and a real FUNSD/Indic local cache` | Ollama runner, cache builder, local manifests. |
| `fix(cascade): inject the OCR transcript into the reader prompt` | Critical bug: cascade C0/Cδ ignored OCR text. |
| `feat(local): add Windows OCR, a rule lower bound, synthetic validation forms, and an OCR degradation pilot` | Windows OCR, rule reader, synthetic forms, CER pilot. |
| `fix(local): make the rule reader stop at the next field label` | Correct field-value boundaries. |
| `docs(results): add the RTX 3050 pilot, its negative findings, and the cascade prompt bug` | Curated results, raw predictions, OCR text, limitations. |
| `docs: clarify the local pilot scope` | Remove ambiguous wording. |

## Push status

The series is published to `origin/main`. The local branch and the GitHub remote
match; no force-push or history rewrite was used.

```bash
git fetch origin
git log -1 --oneline origin/main
git rev-list --count origin/main..HEAD   # 0
```

For future changes from this machine, the HTTPS backend may need:

```bash
git -c http.sslBackend=openssl push origin main
```

Git Credential Manager can open its prompt from a normal interactive terminal;
this agent session cannot create the named objects it needs for interactive
authentication, but the stored credential worked for the publish above.

## Verify the series locally

```bash
git log --oneline 8f18b0c..HEAD
git rev-list --count 8f18b0c..HEAD
python -m pytest -q          # 58 tests
```
