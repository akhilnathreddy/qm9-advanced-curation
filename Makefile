.PHONY: sync lint format typecheck test check repro repro-ci fixture gate0 gate1

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
