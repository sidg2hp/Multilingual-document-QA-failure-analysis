# Manifests

A manifest freezes the items used in a run: sample IDs, page IDs, question
templates, answers, content hashes, and capability flags. It contains no page
images, so it can be committed even when the source data cannot be redistributed.

Build one with:

```bash
python scripts/build_probe_set.py --mode funsd --max-samples 200
python scripts/build_probe_set.py --mode indic_ocr --max-pages 600 --per-group 100
```

Do **not** commit a manifest generated from smoke/synthetic data as if it were a
probe set. The smoke manifest is for pipeline testing only.

A real manifest must record the dataset revision and the generator version. If
the underlying dataset changes, rebuild and version the manifest; do not reuse a
stale one.
