# 0013 — Benchmark ladder, protocol and compute budget

- **Status:** Accepted
- **Date:** 2026-11-09
- **Phase:** 9

## Context
Phase 9 answers RQ2–RQ5 of the analysis plan, frozen at commit 4b08cc5,
before any Phase 9 model was trained. The plan fixes these:
- the research questions, populations and splits;
- the decision rules;
- the baseline ladder, levels 0–5 (plan v2 §7.1).

The plan does not fix how each level is built or how much compute each one
gets. The work runs on one laptop: an M2 Pro with 10 cores, 16 GB of RAM and
an Apple GPU (MPS), no CUDA. Every deviation from the plan is a dated
amendment in `docs/analysis_plan.md`. This ADR records those amendments and
the reasons behind them.

## Decision — the ladder (`src/qm9_curation/benchmark/models.py`, `gnn.py`)

| Level | Model | Inputs | Notes |
|---|---|---|---|
| 0 | Training mean | — | Floor |
| 1 | Dressed atom | Element counts + intercept, least squares | Stoichiometric baseline |
| 2 | Histogram gradient boosting (scikit-learn) | 184 RDKit descriptors + ECFP4 count bits present in ≥ 1% of training molecules | Primary RQ5 model |
| 3 | Laplacian kernel ridge | Sorted Coulomb-matrix eigenvalues | Exact solve, ≤ 20,000 training molecules |
| 4 | D-MPNN (Chemprop architecture), plain PyTorch | Phase 6 molecular graph | Hidden 300, depth 3, CPU |
| 5 | SchNet, plain PyTorch | Coordinates | 6 interactions, cutoff 5 Å, MPS |

- **Δ-learning:** levels 2–5 learn the residual of level 1, fitted on the same
  training rows, so every model has the stoichiometric part removed the same
  way.
- **Readout:** sum for U0 atomization, which is extensive, and mean for the
  gap, which is intensive.
- **Neural levels:** a fixed 30-epoch budget. Adam with linear warm-up and
  exponential decay. The epoch with the best validation MAE is kept.
- **Level 2:** the number of boosting iterations is chosen on validation.
- **Tuning:** on `random` seed 0, validation rows only.
  - Level 2: `max_leaf_nodes` ∈ {31, 127}.
  - Level 3: kernel width × λ, a 6 × 4 grid, on 5,000 training molecules.
  - Levels 4 and 5: architecture defaults from their papers, no tuning grid.
- **No packages beyond the ladder:**
  - scikit-learn replaces LightGBM, because the LightGBM macOS wheel needs
    Homebrew `libomp`, which this machine lacks.
  - D-MPNN and SchNet are written in plain PyTorch. They do not use the
    Chemprop, Lightning or PyG packages, which would add heavy, fast-moving
    dependencies for two models.

## Decision — protocol (`src/qm9_curation/benchmark/run.py`, `analysis.py`)
- **What one run is:** one cell of experiment × model × target × split ×
  seed × training tiers × N × fold.
- **Run key:** a hash of the spec and the model parameters, including a
  per-model `version`.
  - A finished run is never repeated, and a changed model never reuses a
    cached run.
  - So the grid is resumable, and processes on disjoint `--models` can run
    at once.
- **Test rows:**
  - Tuning runs predict validation rows only.
  - Every other run predicts `val ∩ Gold` and `test ∩ Gold` once.
  - RQ2 runs predict their out-of-fold training rows.
- **Tracking:** every run is logged to a local MLflow file store (`mlruns/`),
  tagged with `git_commit`, `git_dirty`, `code_sha256` and
  `dataset_version`.
  - `code_sha256` hashes the benchmark source and configuration.
  - It exists because the runs come before the Phase 9 commit.
- **Statistics** (`metrics.py`):
  - percentile bootstrap: 2,000 resamples, seed 0;
  - RQ3: unpaired, molecules resampled within each seed's test set, MAE
    pooled over seeds;
  - RQ5: paired, on the same Gold test molecules;
  - Holm correction within each RQ;
  - one-sided Mann–Whitney for RQ2.
- **Combining verdicts:** a cell needs its CI condition and a Holm-adjusted
  p < 0.05. The analysis-plan amendment of 2026-11-09 gives the rules.
- **Leakage alarm:** any `random` Gold-test MAE below half the published
  SchNet MAE halts reporting until the cause is investigated.

## Decision — compute budget and coverage
Measured on this machine, each process alone:
- **Level 2:** about 80 s per full-data run with 6 threads, about 2 min with
  10.
- **Level 3:** about 30 s per run at N = 20,000.
- **Level 4:** 23.6 s per epoch on CPU, so about 12 min per 30-epoch run.
  This came from one full-data `tuning` run on validation rows only:
  `random` seed 0, U0. The best epoch was the last, with Gold validation MAE
  0.049 eV.
- **Level 5:** about 85 s per epoch on MPS, so about 45 min per run.

The full grid as first configured would have needed about 30–35 h of wall
time. The user chose to drop level 4 from RQ5:
- RQ5's decision rule uses level 2, the primary model, which runs every RQ5
  cell. Level 4 was only the secondary learning curve, and cost about 12.5 h.

Final coverage, set in `configs/benchmark.yaml` (`coverage`, `not_run`):

| Level | RQ3 (5 splits) | Stress (3 splits) | RQ5 | RQ2 |
|---|---|---|---|---|
| 0, 1, 3 | ✓ | ✓ | — | — |
| 2 | ✓ | ✓ | ✓ | if best |
| 4 | ✓ | ✓ | not run (compute) | if best |
| 5 | `random` only | not run (compute) | — | if best |

Every covered cell uses 5 seeds and both targets. RQ2 uses the level with the
best mean `random` validation MAE, chosen separately for each target.

The grid ran as three processes on disjoint models:
- levels 0–3 on CPU with 6 OpenMP threads;
- level 4 on CPU with 4 threads;
- level 5 on MPS with 2 CPU threads.

- **RQ2:** it waits for every `random` run. The SchNet process ran the gap
  folds. A separate `--models dmpnn --experiments rq2` process ran the U0
  folds, because the first D-MPNN process had read the tuning winners at
  start-up, before levels 2 and 3 were tuned.
- **Final pass:** `dvc repro benchmark` ran a plain
  `python -m qm9_curation.benchmark.run`, which found every cell cached and
  trained nothing.
- **Heat:** SchNet was paused for about 9 h to cut the machine's heat.
- **Speed vs results:** thread counts change speed, not results. Every run's
  seconds are in `run_metrics.parquet`.

## Alternatives considered
- **Full grid, level 4 in RQ5:** about 30–35 h. It adds a secondary curve
  from an under-trained model, and no decision rule depends on it.
- **Also drop level 4 from the stress tests:** about 11–12 h. Rejected
  because the stress tests are the only view of a learned graph model under
  an element or property-tail shift.
- **Fewer seeds or epochs:** rejected. The plan requires ≥ 5 seeds, and
  fewer epochs would widen the gap to the literature further.
- **Rent a CUDA GPU:** not chosen, to keep every result reproducible on the
  project's own machine and environment.
- **LightGBM, Chemprop, PyG:** rejected for the platform and dependency
  reasons above.

## Consequences
- Levels 4 and 5 are under-trained relative to the literature: level 4's
  best epoch was the last of 30. Their numbers are lower bounds on what the
  architectures can do. They are not reproductions of published results.
- RQ5 speaks for level 2 only. Level 4's learning curve is listed as not run.
- Level 5 enters RQ3 only on `random`, as the RQ3/RQ4 reference.
- Any change to a model's code needs a `version` bump in
  `configs/benchmark.yaml`. Otherwise the cached runs would be reused.
- Phase 7 left two items for this phase:
  - recompute the suspicion score with the best model before any manual
    review;
  - an electronic-only (ZPE-separated) RQ4 comparison.

  Both stay open; see `docs/limitations.md`.
