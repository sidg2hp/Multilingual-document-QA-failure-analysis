# Pre-registration & Analysis Plan

**Status:** draft to be frozen before the confirmatory run.
**Purpose:** prevent the most common failure mode of diagnostic papers — running
a large matrix, then choosing the slicing that looks significant. The plan below
fixes the primary claim, the conditions, the exclusion rules, and the statistical
family in advance.

A frozen copy must be committed with a tag (e.g. `prereg-v1`) and a hash before
the confirmatory predictions are generated. Exploratory analyses are allowed and
must be labelled as such.

---

## 1. Research questions

**RQ1 (primary).** When a multilingual document QA system fails, is the failure
better explained by recognition, localization, or reasoning — and does the
answer differ between cascaded OCR+LLM pipelines and OCR-free VLMs?

**RQ2.** Does an OCR-free VLM exhibit **read–reason dissociation** on Indic
scripts: can it transcribe the relevant span (low CER/MER) but fail to use it
when answering directly?

**RQ3.** Under controlled visual degradation, does recognition error concentrate
in Indic combining marks (matras) relative to base consonants, and does that
concentration differ between OCR and VLM pipelines?

**RQ4.** Does 4-bit weight quantization disproportionately degrade Indic
recognition relative to FP16/INT8, compared with Latin text?

**RQ5.** Can a text-free, image-only router capture a useful fraction of the
per-item oracle gap between OCR and VLM at lower measured cost?

---

## 2. Hypotheses and primary endpoints

| ID | Hypothesis | Primary endpoint | Direction | Family |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Read–reason dissociation exists and is larger for Indic than Latin | `Acc(C1) − Acc(C0)` (ANLS), paired by item; interaction with script group | > 0; Indic > Latin | Confirmatory |
| **H2** | Matra errors grow faster than base-character errors under blur/downscale | `MER(C0 OCR, T*) − CER_base(C0 OCR, T*)` slope across severity | > 0 | Confirmatory |
| **H3** | Oracle localization helps beyond a placebo crop | `[Acc(C2) − Acc(C0)] − [Acc(C2b) − Acc(C0)]` | > 0 | Confirmatory |
| **H4** | NF4 degrades Indic recognition more than Latin, relative to FP16 | `ΔMER(NF4 − FP16)` Indic minus Latin | > 0 | Confirmatory |
| **H5** | A text-free router captures ≥ 50% of the oracle gap at ≤ 60% of VLM cost | `(Acc(router) − Acc(always_OCR)) / (Acc(oracle) − Acc(always_OCR))` and mean measured cost | ≥ 0.5 at ≤ 0.6 | Confirmatory |
| **H6** | VLM hallucination-to-abstention ratio exceeds OCR's at severe degradation | HAR difference at level 4–5 | > 0 | Exploratory |

`C2b` is a placebo crop: a random region of the same area as the gold answer
box, sampled from the same page and excluding the answer region. `C3b` is a
shuffled transcript (same lines, deterministic wrong order) used as a
"transcript helps at all" control.

**Primary claim for the paper:** H1, because it is mechanistic and does not
depend on the layout oracle. H2/H4 are the second paper if H1 is null.

---

## 3. Conditions

### 3.1 OCR-free VLM

| Condition | Input | Identifies |
| :--- | :--- | :--- |
| `C0` | clean image + question | baseline |
| `Cδ` | perturbed image + question | degradation sensitivity |
| `C1` | clean image; transcribe, then answer from own transcript (transcript logged and scored) | read–reason dissociation |
| `C2` | gold answer-region crop + question | localization + clutter removal |
| `C2b` | placebo crop + question | crop/context effect |
| `C3` | gold transcript + question (text-only call) | recognition oracle |
| `C3b` | shuffled gold transcript + question | transcript-content control |

### 3.2 Cascaded OCR + text LLM

| Condition | Input | Identifies |
| :--- | :--- | :--- |
| `C0` | clean image → OCR → LLM | baseline |
| `Cδ` | perturbed image → OCR → LLM | degradation sensitivity |
| `C2` | gold transcript **with coordinates** → LLM | full extraction oracle (not layout alone) |
| `C3` | gold transcript, flat reading order → LLM | extraction oracle |
| `C3b` | shuffled gold transcript → LLM | order sensitivity with perfect recognition |

`C1` is **not defined** for the cascade: `C0` already performs recognition then
reasoning, and a second "read first" prompt does not create an independent
intervention. The implementation must mark it unsupported rather than
duplicating `C0`.

**Identified decomposition.**

- VLM: `E_total = (AccC0 − AccCδ) + (AccC2 − AccC0) + (AccC3 − AccC2) + (1 − AccC3)`.
- Cascade: `E_total = (AccC0 − AccCδ) + (AccC3 − AccC0) + (1 − AccC3)`, where the
  middle term is **joint extraction error**. Report
  `AccC2_coord − AccC3_flat` as a separate layout-representation diagnostic; do
  not add it to the identity.
- The placebo conditions are reported as contrasts, not as decomposition terms.

Cases where the nested accuracy sequence is non-monotonic are reported, not
clipped. Negative components are findings (an oracle intervention hurt).

---

## 4. Data and sample

### 4.1 Intended set

A frozen 600-item diagnostic set:

- 500 items from Indic document pages (5 languages × 100), drawn from
  IndicVisionBench `ocr` or Patram-Bench if access is obtained.
- 100 items from FUNSD test (English/Latin layout control).

Each item = one (page, question, gold answer, gold transcript, optional gold
box). Multiple questions per page are allowed but the document ID is the
clustering unit. The manifest records:

- dataset name, revision/hash, split;
- sample ID, page ID, language, script;
- question template ID and version;
- answer text, answer aliases, answer box (if any);
- image SHA-256, transcript SHA-256;
- `supports_text_oracle`, `supports_layout_oracle`;
- inclusion/exclusion flags and reason.

### 4.2 If Patram-Bench access is not obtained

Do **not** fabricate a Patram-Bench split. Pre-register the fallback:

- Indic set from IndicVisionBench `ocr` (876 pages available; sample 500 with a
  deterministic hash order, stratified by language);
- questions generated by the released template generator, with the template
  family and random seed fixed;
- layout conditions only on the subset where a gold box can be obtained from an
  independent detector + manual verification (target ≥ 200 pages, 50 per
  language); otherwise H3 is tested on FUNSD only and the paper says so.

### 4.3 Power

With 600 paired items and α = 0.05 (two-sided), a paired t-test on ANLS has
~80% power for a standardized mean difference of ≈ 0.115; McNemar on accuracy
has ~80% power for a discordant-pair odds ratio around 1.5 when discordance is
~30%. Per-language cells (n ≈ 100) only detect large effects (≈ 0.28 SD); they
are descriptive, not confirmatory. Report minimum detectable effects for every
reported cell rather than leaving readers to guess.

---

## 5. Perturbations

Five families: Gaussian blur, salt-and-pepper, JPEG, resolution downscale, and
perspective skew. Five levels each. Levels must be calibrated so that the
induced OCR CER brackets the interesting range (roughly 5–60%) rather than being
defined by an arbitrary pixel radius. The calibration procedure:

1. Run the reference OCR (Tesseract) on 50 calibration pages.
2. For each family, choose parameters targeting CER ≈ {5, 10, 20, 35, 50}%.
3. Freeze the parameter table in the manifest.
4. Report the measured CER for each level as a manipulation check; if the curve
   is non-monotonic, fix the calibration before the confirmatory run.

Negative controls:

- `identity`: re-encode the clean image through the same JPEG/PNG path, no
  degradation;
- `placebo_crop` (above);
- `shuffled_transcript` (above).

---

## 6. Models

Minimum confirmatory set:

| Role | Model | Precision | Notes |
| :--- | :--- | :--- | :--- |
| OCR-free VLM (primary) | Qwen2.5-VL-3B-Instruct | NF4, INT8, FP16 where it fits | exact revision pinned |
| OCR-free VLM (scale check) | Qwen2.5-VL-7B-Instruct | FP16 on 2×V100 | optional but strongly recommended |
| Text reader | Qwen2.5-3B-Instruct (NF4) | NF4 | chat template, not raw prompt |
| Text reader (scale check) | Qwen2.5-7B-Instruct | FP16 | 2×V100 |
| OCR engines | Tesseract 5, PaddleOCR PP-OCRv4 (language-correct), Surya | — | record versions |
| Secondary VLM | SmolVLM2-2.2B or InternVL2.5-8B | verify | optional; only if loader is tested |

Every model gets a JSON record with revision, license, dtype, quantization,
device map, max pixels/tokens, prompt version, and decoding parameters.

---

## 7. Analysis plan

### 7.1 Primary

- Item-level ANLS per condition; mean and 95% **document-clustered** bootstrap CI
  (10,000 resamples, cluster = page ID).
- H1: paired difference `C1 − C0`; paired bootstrap CI, paired permutation test,
  and McNemar on exact-match correctness. Interaction with script group via a
  cluster-bootstrap stratified difference (Indic minus Latin).
- H3: difference-in-differences `(C2 − C0) − (C2b − C0)` with cluster bootstrap.
- H5: out-of-fold, document-grouped router predictions; accuracy–cost curve;
  bootstrap CI on the gap captured; report measured wall-clock cost.

### 7.2 Secondary

- H2/H4: mixed-effects or cluster-robust regression of MER/CER on severity,
  script, model, and their interactions; report marginal effects with CIs. If a
  simpler analysis is used, say so and justify it.
- H6: HAR with Wilson CIs; no ratio is reported without its numerator and
  denominator.

### 7.3 Multiplicity

Five confirmatory hypotheses → Holm–Bonferroni at family-wise α = 0.05. All
other tests are exploratory and reported as raw p-values with a clear label. No
"significant after correction" claim for exploratory slices.

### 7.4 Missing data

- Missing condition (model crashed, dataset lacks support) → excluded from that
  condition's analysis and counted in a coverage table. Never imputed as 0 or as
  the gold answer.
- Failed generation (empty output after parsing) → counted as an abstention, not
  silently dropped.
- Model load/OOM failures → reported as a system result, not excluded from the
  paper.

### 7.5 Exclusions (pre-specified)

- Blank or unreadable page (manual flag) — exclude before running.
- Gold answer not present verbatim in the transcript — exclude from oracle-text
  conditions and record why.
- Duplicate page across splits — deduplicate by image hash.
- Fewer than 30 items in a cell — descriptive only.

---

## 8. Human validation

- Sample 100 failures stratified by script and model.
- Two annotators fluent in the relevant language label: correct, abstention,
  confident hallucination, or ambiguous.
- Report Cohen's κ; disagreements adjudicated by a third annotator.
- Do not use the automatic `is_indic_confident_hallucination` proxy as ground
  truth in the paper; report its agreement with human labels.

---

## 9. Stopping and reporting rules

- Fixed sample; no optional stopping. If a model cannot run, report it.
- Report every condition and every perturbation family, not only the best.
- Report negative and null results; the decomposition must include negative
  components.
- The paper's abstract may contain only claims that survived the pre-registered
  analysis.
- All figures must be generated by a script from the released prediction files;
  no hand-edited numbers.

---

## 10. Artifacts to release

- Frozen manifest (IDs, hashes, revisions).
- Prompt templates with version hashes.
- Prediction JSONL with raw outputs, parsed answers, transcripts, latencies,
  token counts, and model metadata.
- Analysis script and figure script.
- Environment lockfile and hardware record.
- Annotation guideline and de-identified human labels.
- A `REPRODUCE.md` with exact commands and expected runtime.

---

## 11. Freeze checklist

- [ ] Data access confirmed and license recorded.
- [ ] Manifest built, hashed, and committed (IDs only; no restricted images).
- [ ] Prompt templates frozen and hashed.
- [ ] Perturbation levels calibrated and frozen.
- [ ] Models/revisions pinned and load-tested.
- [ ] Analysis script reviewed against this plan.
- [ ] Tag `prereg-v1` created and hash recorded in the paper.
- [ ] Any deviation documented in a dated changelog.
