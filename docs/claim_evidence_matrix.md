# Claim–Evidence Matrix

Every claim the project might make, the evidence it needs, and whether that
evidence exists today. A paper may only claim what is in the "Supported?" column
after the confirmatory run.

Legend: **No** = no evidence; **Partial** = a code path or synthetic output
exists; **Yes** = real, verified evidence exists.

| # | Candidate claim | Required evidence | Current status | Supported? | Minimum work |
| :--- | :--- | :--- | :--- | :---: | :--- |
| C1 | "We introduce a counterfactual protocol that localizes DocQA failures across OCR and VLM pipelines." | Executed protocol; condition coverage; per-condition accuracies; architecture-specific decomposition | Code paths exist; no real runs; cascade ladder corrected but not executed | **Partial** | Run 600 items |
| C2 | "Indic OCR-free VLMs can read matras but fail to reason over them (read–reason dissociation)." | `C1` transcript CER/MER; `C0` vs `C1` QA; stratified by script; paired test | Transcript scoring implemented; no runs | **Partial** | Run + inspect |
| C3 | "Matra errors grow faster than base-character errors under blur/downscale." | MER and base CER curves; 5 calibrated levels; CIs | MER fixed; levels not calibrated; no runs | **Partial** | Calibrate, run |
| C4 | "Layout localization is a distinct failure stage, not just clutter removal." | `C2` vs placebo `C2b`; boxes for Indic subset | Placebo implemented; no Indic boxes | **Partial** | Obtain boxes, run |
| C5 | "4-bit quantization disproportionately degrades Indic recognition." | NF4/INT8/FP16 same model; script interaction; 3 seeds | Precision path exists; no runs | **Partial** | Multi-precision runs |
| C6 | "OCR-free VLMs hallucinate more than cascaded OCR under severe degradation." | HAR with counts and CIs; human validation | HAR fixed; dry-run only; proxy unvalidated | **Partial** | Run + human labels |
| C7 | "A text-free image router captures most of the OCR/VLM oracle gap at lower cost." | Leakage-free features; measured costs; grouped CV; held-out test; CI on gain | Guard, measured cost, grouped CV implemented; no runs | **Partial** | Run router |
| C8 | "The framework generalizes across 10 Indic languages and 23 languages." | Per-language n ≥ 100; CIs; multiple scripts | Data not available at claimed scale | **No** | Data access |
| C9 | "The pipeline is reproducible on 6 GB consumer hardware and PARAM Ganga." | End-to-end run on both; memory/latency logs; lockfile | CI, license, and package exist; no GPU run | **Partial** | Execute + pin lock |
| C10 | "We release a diagnostic benchmark / probe set." | Frozen manifest, generation code, licenses, human check | Manifest builder and generator exist; no real frozen set | **Partial** | Build & release |

---

## Claims that must **not** be made

These appeared in the project notes or are implied by the current README; remove
them until evidence exists.

1. **"Evaluated on Patram-Bench."** The dataset is not accessible at the
   configured ID and no run exists.
2. **"5,000 images / 37,000 QA pairs of document QA from IndicVisionBench."**
   The 37K figure is the benchmark's aggregate across cultural VQA/MMT/OCR; it
   is not 37K document QA pairs.
3. **"State-of-the-art baselines."** A single 3B NF4 model is not a baseline
   suite.
4. **"Exact causal decomposition"** for cascaded pipelines. Only the joint
   extraction term is identified.
5. **"Read–reason dissociation"** without transcript CER. The current `C1` can
   be explained by hallucinated transcription.
6. **"Leakage-free router"** while `script_mixing_index` comes from the oracle
   transcript.
7. **"75–85% acceptance probability."** Unsupported and inconsistent with the
   evidence.
8. **"First"** claims about Indic document benchmarks or attribution. Patram-
   Bench, IndicVisionBench, DISCO, and Xu et al. already occupy adjacent space.

---

## Evidence-to-figure map (target paper)

| Figure / table | Content | Data source | Status |
| :--- | :--- | :--- | :--- |
| Fig. 1 | Protocol diagram (conditions, architecture-specific ladders) | This repo | Exists as Mermaid; needs redraw |
| Fig. 2 | Degradation curves: ANLS and CER/MER vs severity, per pipeline | `predictions.jsonl` | No data |
| Fig. 3 | Stage decomposition with cluster-bootstrap CIs, VLM vs cascade | Attribution + manifest | No data |
| Fig. 4 | Read–reason dissociation: transcript CER vs QA gain, per script | `C1` transcripts | No data |
| Fig. 5 | Router accuracy–cost Pareto with measured latency | Router eval | No data |
| Table 1 | Dataset and model provenance | `docs/data_provenance.md` | Partial |
| Table 2 | Main results by condition and model, with CIs | Predictions | No data |
| Table 3 | Per-language/script breakdown with cell sizes and MDE | Predictions | No data |
| Table 4 | HAR and failure-mode distribution, with human agreement | Predictions + labels | No data |
| Appendix | Prompts, templates, manifest hash, compute, negative results | Repo | Partial |

---

## Reviewer-proofing checklist

- [ ] Every number in the abstract is generated by `scripts/analyze_results.py`
      from a released prediction file.
- [ ] Every condition is reported, including negative components.
- [ ] Every comparison has a paired test, an effect size, and a CI.
- [ ] Every per-language cell reports n and a minimum detectable effect.
- [ ] Every model has a revision, license, precision, and prompt hash.
- [ ] Every dataset has a revision, license, and manifest hash.
- [ ] The placebo and negative controls are reported next to the oracle results.
- [ ] The human-validation protocol and agreement are reported.
- [ ] Limitations include data access, model scale, English prompts, and
      template-generated questions.
