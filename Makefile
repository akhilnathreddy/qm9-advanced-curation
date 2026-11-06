.PHONY: sync lint format typecheck test check repro gate0

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

# Phase 0 exit gate: locked install, all checks, pipeline runs, result recorded.
gate0: sync check repro
	uv run python -m qm9_curation.reporting.gates --phase 0
