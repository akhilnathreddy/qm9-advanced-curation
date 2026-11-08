.PHONY: sync lint format typecheck test check repro repro-ci fixture gate0 gate1 gate2 gate3 gate4 gate5 gate6 gate7 gate8

sync:
	uv sync --locked

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

repro:
	uv run dvc repro

# CI has no copy of the raw data; it runs only stages that don't need it.
repro-ci:
	uv run dvc repro env_report

# Rebuild the committed 1k test fixture from the raw archive (deterministic).
fixture:
	uv run python -m qm9_curation.fixture

# Phase 0 exit gate: locked install, all checks, pipeline runs, result recorded.
gate0: sync check repro
	uv run python -m qm9_curation.reporting.gates --phase 0

# Phase 1 exit gate: raw files verified, second download matches, inventory holds.
gate1: sync check repro
	uv run python -m qm9_curation.acquire --verify-remote
	uv run python -m qm9_curation.reporting.gates --phase 1

# Phase 2 exit gate: lossless parse, schemas valid, and a forced second parse
# must reproduce the Parquet files byte-for-byte.
gate2: sync check repro
	cp reports/data_quality/parse_report.json /tmp/qm9_parse_report_first.json
	uv run dvc repro --force parse
	uv run python -c "import json,sys; a,b=(json.load(open(p))['output_sha256'] for p in ('/tmp/qm9_parse_report_first.json','reports/data_quality/parse_report.json')); print('PASS  parquet byte-identical across runs' if a==b else 'FAIL  parquet differs across runs'); sys.exit(a!=b)"
	uv run python -m qm9_curation.reporting.gates --phase 2

# Phase 3 exit gate: profile + triage, and a forced second profile run must
# reproduce every output byte-for-byte.
gate3: sync check repro
	cp reports/data_quality/profile.json /tmp/qm9_profile_first.json
	uv run dvc repro --force --single-item profile
	uv run python -c "import json,sys; a,b=(json.load(open(p))['output_sha256'] for p in ('/tmp/qm9_profile_first.json','reports/data_quality/profile.json')); print('PASS  profile outputs byte-identical across runs' if a==b else 'FAIL  profile outputs differ across runs'); sys.exit(a!=b)"
	uv run python -m qm9_curation.reporting.gates --phase 3

# Phase 4 exit gate: all checks run on every record, tolerances set, and a
# forced second validation run reproduces the flag table byte-for-byte.
gate4: sync check repro
	cp reports/validation/validation_report.json /tmp/qm9_validation_first.json
	uv run dvc repro --force --single-item validate
	uv run python -c "import json,sys; a,b=(json.load(open(p))['output_sha256'] for p in ('/tmp/qm9_validation_first.json','reports/validation/validation_report.json')); print('PASS  validation table byte-identical across runs' if a==b else 'FAIL  validation table differs across runs'); sys.exit(a!=b)"
	uv run python -m qm9_curation.reporting.gates --phase 4

# Phase 5 exit gate: tiers + standardization conditions, and a forced second
# curation run must reproduce the curated table byte-for-byte.
gate5: sync check repro
	cp reports/curation/curation_report.json /tmp/qm9_curation_first.json
	uv run dvc repro --force --single-item curate
	uv run python -c "import json,sys; a,b=(json.load(open(p))['output_sha256'] for p in ('/tmp/qm9_curation_first.json','reports/curation/curation_report.json')); print('PASS  curated table byte-identical across runs' if a==b else 'FAIL  curated table differs across runs'); sys.exit(a!=b)"
	uv run python -m qm9_curation.reporting.gates --phase 5

# Phase 6 exit gate: feature conditions, and a forced second feature run must
# reproduce every table byte-for-byte.
gate6: sync check repro
	cp reports/features/feature_report.json /tmp/qm9_features_first.json
	uv run dvc repro --force --single-item features
	uv run python -c "import json,sys; a,b=(json.load(open(p))['output_sha256'] for p in ('/tmp/qm9_features_first.json','reports/features/feature_report.json')); print('PASS  feature tables byte-identical across runs' if a==b else 'FAIL  feature tables differ across runs'); sys.exit(a!=b)"
	uv run python -m qm9_curation.reporting.gates --phase 6

# Phase 7 exit gate: review conditions, limitations recorded, and a forced
# second review run must reproduce both tables byte-for-byte.
gate7: sync check repro
	grep -q "^## Unresolved after Phase 7" docs/limitations.md
	cp reports/statistical/statistical_report.json /tmp/qm9_review_first.json
	uv run dvc repro --force --single-item review
	uv run python -c "import json,sys; a,b=(json.load(open(p))['output_sha256'] for p in ('/tmp/qm9_review_first.json','reports/statistical/statistical_report.json')); print('PASS  review tables byte-identical across runs' if a==b else 'FAIL  review tables differ across runs'); sys.exit(a!=b)"
	uv run python -m qm9_curation.reporting.gates --phase 7

# Phase 8 exit gate: leakage assertions for every split and seed, analysis
# plan frozen, and a forced second split run must reproduce both tables.
gate8: sync check repro
	grep -q "^- \*\*Frozen:\*\* yes" docs/analysis_plan.md
	cp reports/splits/split_report.json /tmp/qm9_splits_first.json
	uv run dvc repro --force --single-item splits
	uv run python -c "import json,sys; a,b=(json.load(open(p))['output_sha256'] for p in ('/tmp/qm9_splits_first.json','reports/splits/split_report.json')); print('PASS  split tables byte-identical across runs' if a==b else 'FAIL  split tables differ across runs'); sys.exit(a!=b)"
	uv run python -m qm9_curation.reporting.gates --phase 8
