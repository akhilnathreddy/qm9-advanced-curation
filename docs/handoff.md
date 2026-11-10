# Handoff — where the project stands (2026-11-10, Phase 10)

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
| 9 Benchmark | `phase-9/benchmark` | 8bddedf | done, Gate 9 |
| **10 Release** | `phase-10/release`, `main` | b7c1b03 + verification fix | **done, Gate 10; tag `v1.0.0`; two least-squares tables verified to 1e-9 (user's choice); published on Zenodo, doi:10.5281/zenodo.24105666** |
| 11 Uncharacterized recovery (optional) | — | — | not started |

- Branches are stacked: each starts from the previous one, so
  `phase-8/splits` contains Phases 0–8.
- GitHub's default branch is `phase-0/setup`, and there is no `main`. On
  2026-11-09 the user chose to leave the default branch as it is. Create
  `main` before the Phase 10 release.

## Phase 9 — what exists (committed in 8bddedf)

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
- **Next:** commit and push `phase-9/benchmark` when the user asks. Then
  Phase 10.

## Phases 10–11 and what the user asked for at the end

- **Phase 10, release (2026-11-10).** User decisions:
  - code MIT, dataset CC0-1.0;
  - PyG loader as an optional `pyg` extra;
  - author credited by GitHub handle only;
  - the stray `Users/` folder was moved to the Trash.

  Built:
  - `data/release/qm9-curated-v1.0.0/`;
  - the published manifest `release/qm9-curated-v1.0.0.sha256`;
  - `make release`, `make reproduce`, `make zenodo` and `make gate10`;
  - `.github/workflows/reproduce.yml` (the Linux reference build);
  - ADR 0014, `docs/release.md`, `docs/datasheet.md`.

  First Linux run (2026-11-10):
  - every pipeline check passed;
  - 5 of 7 tables were byte-different from the Mac build:
    - 3 differed only in Parquet chunking, now fixed by the canonical layout;
    - 2 (`review`, `g4mp2_xref`) differ by up to 4.5e-11, from BLAS rounding.

  The published manifest is now the predicted Linux build: the release builder
  run on the Linux CI artifact's tables. On macOS, `verify --compare-with`
  checks equivalence (ADR 0014).

  Open:
  1. Run 42796901845 reproduced every file byte-identically. The later runs
     (tag `v1.0.0`, merge of PR #8) landed on other machines, where
     `review` and `g4mp2_xref` differed by up to 5e-12 (OpenBLAS kernels).
     The user chose a tolerance rule over pinning the BLAS: every file except
     those two must be byte-identical, and those two within 1e-9 of the
     published archive (ADR 0014).
  2. The Zenodo bundle in `data/zenodo/` was built from that run's artifact
     (243 MB, sha256 `e3ef7b45…`). The user uploads it from their own
     account (`data/zenodo/UPLOAD.md`). Done: published 2026-11-10 as
     https://zenodo.org/records/24105666, version DOI 10.5281/zenodo.24105666, concept DOI 10.5281/zenodo.24105665. `make reproduce` uses the
     Zenodo file as its reference archive.
  3. Done: `docs/report.md`, the consolidated project report.
  4. Done: `main` and the annotated tag `v1.0.0` point at the report commit;
     the tag triggers a Linux reproduce run.
  5. Optional: make `main` the default branch (GitHub setting, the user's
     call).
- **At the end (done, `docs/report.md`):** a single consolidated project report covering:
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
