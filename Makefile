.PHONY: test smoke sweep audit manifest analyze clean

test:
	python -m pytest -q

smoke:
	python scripts/run_experiment.py --mode smoke --max-samples 3 --seed 42 --resume

sweep:
	python scripts/run_experiment.py --mode smoke --sweep --max-samples 3 --seed 42 --resume

audit:
	python scripts/audit_datasets.py --out outputs/provenance/dataset_audit.json

manifest:
	python scripts/build_probe_set.py --mode funsd --max-samples 200

analyze:
	python scripts/analyze_results.py --predictions outputs/predictions_funsd.jsonl --manifest manifests/probe_set_funsd.json

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('outputs/analysis', 'outputs/analysis_smoke')]"
