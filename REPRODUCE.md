# Reproducing the harness

This file reproduces the **engineering pipeline**, not a paper result. As of
commit `fca2cb2` there has been no real-model confirmatory run. Read
`docs/peer_review.md` first.

## 1. Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[data,gpu,ocr,dev]"
```

CPU-only, for the tests and smoke pipeline:

```bash
pip install -e ".[dev]"
```

The GPU/OCR extras are only needed for real inference. Pin `torch`,
`transformers`, CUDA, and the model revisions in the run record; the ranges in
`pyproject.toml` are not a lockfile.

## 2. Tests

```bash
pytest -q
```

Expected: all tests pass without network, GPU, or the `datasets` package.

## 3. Provenance audit

```bash
python scripts/audit_datasets.py --out outputs/provenance/dataset_audit.json
```

This confirms the dataset/model revisions and the unresolved Patram-Bench
access. It never downloads images or weights.

## 4. Smoke pipeline (not evidence)

```bash
python scripts/run_experiment.py --mode smoke --max-samples 3 --seed 42 \
  --write-manifest manifests/smoke_manifest.json
```

Expected: `outputs/smoke_report.json` with `"reportable": false`, and
`outputs/predictions_smoke.jsonl` containing `[SMOKE]` outputs. Do not analyze
these as results.

## 5. Frozen probe set (real data, no model)

```bash
# English/layout control
python scripts/build_probe_set.py --mode funsd --max-samples 200

# Indic transcript-grounded questions (requires dataset access)
python scripts/build_probe_set.py --mode indic_ocr --max-pages 600 --per-group 100
```

Expected: `manifests/probe_set_<mode>.json` with item IDs, hashes, and counts.
The manifest contains no page images.

## 6. Real run (requires GPU and data access)

```bash
# English/layout control, both pipelines, all supported conditions
python scripts/run_experiment.py \
  --mode funsd \
  --manifest manifests/probe_set_funsd.json \
  --max-samples 200 \
  --resume \
  --seed 42

# A perturbation sweep
python scripts/run_experiment.py \
  --mode funsd \
  --manifest manifests/probe_set_funsd.json \
  --sweep \
  --resume \
  --seed 42
```

Expected: `outputs/attribution_report_funsd.json`,
`outputs/predictions_funsd.jsonl`, and (for the sweep)
`outputs/sweep_funsd.json|csv`. The report includes the git commit, config
hash, model/dataset revisions, prompt hashes, measured latency, condition
coverage, clustered CIs, and architecture-specific decompositions.

If the report says `"reportable": false`, stop; it is a smoke or misconfigured
run.

## 7. Analysis tables

```bash
python scripts/analyze_results.py \
  --predictions outputs/predictions_funsd.jsonl \
  --manifest manifests/probe_set_funsd.json \
  --out-dir outputs/analysis_funsd
```

Expected: `analysis.json`, `tables.md`, `condition_metrics.csv`, and
`degradation_curves.csv`. Every number in the paper should come from this
script.

## 8. PARAM Ganga

`slurm/run_param_ganga.sh` is a template. Before submitting:

1. Fill in the module load / conda activation lines for your account.
2. Pin the model and dataset revisions in the config.
3. Use `--resume` and a shared cache path.
4. Run `funsd` first as an integration test, then `indic_ocr`.
5. Do not run `patram_local` until the data are obtained from the authors.

See `docs/preregistration.md` for the analysis plan and stopping rules.
