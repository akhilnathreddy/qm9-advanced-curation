# 0003 — Environment and pipeline tooling

- **Status:** Accepted
- **Date:** 2026-11-06
- **Phase:** 0

## Context
Every curated value must be rebuildable from raw files with identical
results, and a new contributor must be able to set up in minutes.

## Decision
- **uv** with `pyproject.toml` and a committed `uv.lock`; Python 3.12.
- **DVC** (installed as a dev dependency through uv, not globally) for data
  versioning and the stage graph in `dvc.yaml`.
- **ruff** (lint and format), **mypy --strict** on `src/`, **pytest** with
  **Hypothesis** for property-based tests.
- **pre-commit** with local hooks that call `uv run`, so hook and CI tool
  versions both come from `uv.lock`.
- ML extras (scikit-learn, LightGBM, MLflow) live in an optional `ml` extra
  until Phase 9.

## Alternatives considered
- *conda / mamba*: needed historically for RDKit, but RDKit now ships PyPI
  wheels for macOS arm64 and Linux.
- *Poetry / pip-tools*: slower, and no Python-version management.

## Consequences
- `uv sync --locked` in CI fails if `pyproject.toml` and `uv.lock` disagree.
- Constants are pinned in `units.py` (CODATA 2018), independent of SciPy.
