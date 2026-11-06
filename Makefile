.PHONY: sync lint format typecheck test check repro repro-ci fixture gate0 gate1 gate2 gate3

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
