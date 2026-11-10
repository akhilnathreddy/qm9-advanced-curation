# Handoff — where the project stands (2026-11-10)

A new working session starts here. Read this file, then `CLAUDE.md` (working
rules), `docs/plan_v2.md` (the plan) and `docs/analysis_plan.md` (frozen
analysis plan plus amendments).

## Status by phase

| Phase | Branch (GitHub) | Commit | State |
|---|---|---|---|
| 0 Setup | `phase-0/setup` | 4e3492b | done, Gate 0 |
| 1 Acquisition | `phase-1/acquisition` | 42edd75 | done, Gate 1 |
| 2 Parser | `phase-2/parser` | 0ba7727 | done, Gate 2 |
| 3 Profiling | `phase-3/profiling` | 9121ca0 | done, Gate 3 |
| 4 Validation | `phase-4/validation` | 485bad7 | done, Gate 4 |
| 5 Curation and tiers | `phase-5/curation` | 8a0cd93 | done, Gate 5 |
| 6 Features | `phase-6/features` | 8a80814 | done, Gate 6 |
| 7 Statistical validation | `phase-7/statistical-validation` | 678e758 | done, Gate 7 |
| 8 Splits | `phase-8/splits` | 4b08cc5 | done, Gate 8; analysis plan frozen; CI green |
| **9 Benchmark** | `phase-9/benchmark` (local only) | — | **done, Gate 9; uncommitted** (waiting for the user's go-ahead) |
| 10 Release | — | — | not started |
| 11 Uncharacterized recovery (optional) | — | — | not started |

- Branches are stacked: each starts from the previous one, so
  `phase-8/splits` contains Phases 0–8.
- GitHub's default branch is `phase-0/setup`, and there is no `main`. On
  2026-11-09 the user chose to leave the default branch as it is. Create
  `main` before the Phase 10 release.

## Phase 9 — what exists (uncommitted, on `phase-9/benchmark`)

All code passes ruff, mypy --strict and the full test suite: 192 tests, about
2.7 min.

- **`src/qm9_curation/benchmark/`**
  - `data.py` — aligned inputs, split rows, tier subsampling.
  - `models.py` — ladder levels 0–3:
    - mean;
    - dressed atom;
    - histogram GBM, using scikit-learn instead of LightGBM;
    - Laplacian KRR on Coulomb eigenvalues, capped at 20k training molecules.

    Levels 2–5 learn the residual of level 1 (Δ-learning).
  - `gnn.py` — D-MPNN (Chemprop architecture) and SchNet in plain PyTorch.
  - `metrics.py` — MAE, RMSE, R², fraction within chemical accuracy;
    stratified and paired bootstraps; Holm correction.
  - `tracking.py` — MLflow file store (`mlruns/`), provenance tags, run keys.
  - `run.py` — the grid, in order: tuning (validation only), then RQ3 +
    stress, then RQ2 out-of-fold, then RQ5. Resumable: runs are keyed by spec
    and parameters and cached in `data/benchmark/runs/`. `--models` and
    `--experiments` select a subset, so several processes can split the work.
  - `analysis.py` — decision rules for RQ1–RQ5, verdicts, summary.md,
    figures and the Gate 9 conditions.
- **`configs/benchmark.yaml`**
  - frozen-plan hash, model parameters, tuning grids;
  - `coverage` (which level runs which experiment) and `not_run` (reasons);
  - statistics settings, including the leakage sanity alarm.
- **Wiring**
  - DVC stages `benchmark` and `benchmark_analysis`;
  - `make gate9`;
  - Gate 9 in `reporting/gates.py`;
  - `config.CONFIG_NAMES` includes `benchmark`.
- **Environment**
  - `pyproject.toml`: the `ml` extra is scikit-learn, mlflow-skinny and
    torch, with CPU-only torch wheels on Linux.
  - `make sync` and CI use `uv sync --locked --extra ml`.
- **Tests:** `tests/unit/test_benchmark.py`, `tests/unit/test_gnn.py`,
  `tests/pipeline/test_benchmark_fixture.py` (end-to-end on the 1k fixture).
- **`docs/analysis_plan.md`:** dated amendments (2026-11-09), including a
  disclosure of test numbers seen in exploratory runs. The frozen text above
  `## Amendments` is unchanged and its sha256 still matches the config.

## Phase 9 — results (2026-11-10)

- **Grid:** 682 runs, 40.6 h of compute, all cells present; `make gate9`
  passes.
  - Each run is a Parquet + JSON pair in `data/benchmark/runs/` (DVC).
  - MLflow store: `mlruns/`, git-ignored.
  - Analysis: `reports/benchmark/`.
- **Coverage (Option B, the user's choice, 2026-11-09):** D-MPNN does not run
  RQ5. The amendment and ADR 0013 record this.
- **Verdicts:** RQ1, RQ2, RQ4 and RQ5 supported; RQ3 mixed (only levels 0–1
  and one KRR cell are not inflated).
  - Best `random` U0 MAE: D-MPNN 0.050 eV.
  - Best gap MAE: SchNet 0.107 eV.
  - See `docs/benchmark.md`.
- **Written:**
  - ADR 0013;
  - `docs/benchmark.md`;
  - data-dictionary entries;
  - `## After Phase 9` in `docs/limitations.md`;
  - the README phase line.
- **Code changes during the grid:**
  - RQ5 now respects `coverage`, in both the runner and the analysis.
  - RQ1's `listed_passing_identity` uses `identity_class`. It was 0 before,
    because `n_failed_identity` counts list membership itself.
- **Open:**
  - recompute the Phase 7 suspicion score with the best model;
  - the ZPE-separated RQ4 comparison;
  - the stray `Users/` folder in the repo root, 435 MB and untracked, left
    over from the move. Delete it or git-ignore it before committing.
- **Next:** commit and push `phase-9/benchmark` when the user asks. Then
  Phase 10.

## Phases 10–11 and what the user asked for at the end

- **Phase 10, release.** The user wants the dataset published:
  - exports: Parquet, extended XYZ, a PyG loader;
  - a datasheet and Croissant metadata;
  - `make reproduce` with a hash check.

  Choose a code license (still open). Both data sources are CC0-1.0 (QM9 and
  Kim et al. 2019). The Zenodo DOI deposit goes through the **user's own
  account**: prepare everything, and the user uploads or approves.
- **At the end:** a single consolidated project report covering:
  - every phase and its steps;
  - each decision with its alternatives (ADRs 0001+);
  - the RQ1–RQ5 results;
  - limitations;
  - exact reproduction steps.

  It goes in the repo and, if the user wants, as a shareable page.

## Local environment notes

- The project moved from a temporary app folder to
  `~/qm9-curation-project/qm9-advanced-curation` on 2026-11-09. The git
  history, data, DVC cache and environment were restored, and the tests pass.
- DVC: stages for Phases 0–8 are up to date. `env_report` is stale because
  `uv.lock` changed. The Phase 9 stages have not run yet.
- MLflow file store: `MLFLOW_ALLOW_FILE_STORE=true` is set in `tracking.py`.
