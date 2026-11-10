# Analysis plan

- **Frozen:** yes — 2026-11-07, at the end of Phase 8, before any Phase 9 model is trained.
- **Frozen at commit:** the last commit that changes this file before the first
  Phase 9 result (`git log -1 --format=%H -- docs/analysis_plan.md`); Gate 9
  records that hash and fails if this file changed afterwards without an
  amendment below.

Changes after freezing are allowed only as dated amendments at the bottom,
each with a reason, and must be reported alongside the results.

## Research questions

| ID | Question | Hypothesis | Primary metric | Test |
|---|---|---|---|---|
| RQ1 | How many records fail identity/physics checks beyond the 3,054 uncharacterized? | A non-trivial number do | Flag counts, overlap with uncharacterized list | Descriptive |
| RQ2 | Do flagged records hurt models? | Flagged records have larger out-of-fold residuals | Residual distribution by tier | Mann–Whitney U |
| RQ3 | How much do split choices inflate performance? | Random-split MAE < isomer/scaffold/cluster/size-split MAE | Gold-test MAE with 95% bootstrap CI | Unpaired bootstrap (different test sets) |
| RQ4 | What is the label-noise floor? | Δ(B3LYP − G4MP2) exceeds best random-split MAE | Δ distribution vs. MAE | Descriptive |
| RQ5 | Does curated training help on a fixed Gold test set? | Gold-only ≤ all-records MAE at matched N | Gold-test MAE | Paired bootstrap |

## Targets
- HOMO–LUMO gap (eV)
- U0 atomization energy (eV), using the release's atom-reference energies

## Splits
Fixed by `configs/splits.yaml` and `data/splits/splits.parquet` (ADR 0012,
`docs/splits.md`); seeds 0–4. Every split keeps each leakage group in one
subset.

| Use | Splits |
|---|---|
| RQ3 headline comparison | `random`, `isomer_group`, `scaffold_generic`, `cluster_ecfp`, `size_extrapolation` |
| Stress tests (reported, no hypothesis) | `element_holdout`, `property_tail_gap`, `property_tail_u0_atomization` |
| RQ2, RQ4, RQ5 | `random` |

## Populations
- **Evaluation:** always `test ∩ Gold`. Validation: `val ∩ Gold`, used only
  for early stopping and hyperparameters.
- **Training (default):** `train ∩ {Gold, Silver, Bronze}`. Quarantine (5
  records, not true minima) is never used for training or evaluation.
- **RQ5 training sets:** from `train`: {Gold, Silver, Bronze} vs. {Gold,
  Silver} vs. {Gold}, each subsampled (seeded) to the same N at every
  learning-curve point that all three can reach.

## Models
Baseline ladder levels 0–5 (plan v2 §7.1). No models beyond the ladder in v1.0.
Every level that completes is reported; a level that cannot be run is
reported as not run, with the reason. RQ5 uses level 2 (primary) and level 4
(secondary).

## Protocol
- ≥ 5 seeds; hyperparameters tuned on validation only; test touched once per
  (model, split, target).
- Learning curves at N ∈ {1k, 2k, 5k, 10k, 25k, 50k, 100k}.
- Results much better than published numbers for the same model and split
  trigger a leakage investigation before reporting.

## Decision rules (α = 0.05; Holm correction within each RQ)
- **RQ2:** out-of-fold residuals (5 folds by leakage group over the training
  population) of the level with the best `random` validation MAE. One-sided
  Mann–Whitney, Silver vs. Gold and Bronze vs. Gold, for each target (4 tests).
  Effect size: P(non-Gold residual > Gold residual).
- **RQ3:** for each model and target, a split "inflates" if the upper 95% CI
  bound of `random` Gold-test MAE is below the lower bound of the other split.
  CIs: bootstrap over test molecules (2,000 resamples) pooled over seeds.
- **RQ4:** supported if the upper 95% CI bound of the best `random` Gold-test
  U0 atomization MAE is below 0.123 eV (the Gold B3LYP − G4(MP2) residual MAE
  from Phase 7).
- **RQ5:** paired bootstrap on the same Gold test molecules, per N, model and
  seed-averaged; "Gold-only ≤ all records" is supported if the upper CI bound
  of MAE(Gold) − MAE(all) is ≤ 0 at the largest common N.

## Preliminary results (descriptive, before freezing)
- RQ1 (Phase 4): beyond the 3,054 listed, 5 records fail the independent
  identity check; conversely 1,466 listed records pass it
  (`docs/validation.md`). Recorded here so the frozen plan can reference it.
- RQ1 corroboration (Phase 7): Kim et al.'s 229 multi-molecule records are
  exactly our 229 `fragmented` records; all 453 of our exact duplicates are in
  their 252 duplicate pairs (`docs/statistical_validation.md`, §2).
- RQ4 (Phase 7): B3LYP − G4(MP2) U0 residual after a per-element fit, Gold:
  SD 0.153 eV, MAE 0.123 eV (n = 130,621). Without N (valid atom references):
  AE difference mean +0.171 eV, MAE 0.196 eV (n = 50,929). Phase 9 compares
  these with model MAE; the comparison itself is not yet made.
- RQ2 (Phase 7, weak ridge model, exploratory only): Bronze has larger
  out-of-fold |residual| than Gold for U0 atomization (P = 0.60, p ≈ 10⁻⁴⁵);
  Silver does not (P = 0.50, p = 0.65). The confirmatory test uses the
  Phase 9 models.

- RQ3 context (Phase 8): a record-level random split would leave 774 of
  13,388 test records (5.8%) sharing a leakage group, and 43 an exact
  duplicate, with train/val (`docs/splits.md`). Seed-0 median test → train
  nearest-neighbour Tanimoto: random 0.62, isomer 0.63, cluster 0.57,
  scaffold 0.55, size 0.52.

## Amendments
Each entry was written before the benchmark grid ran. Each is also recorded in
ADR 0013 and reported with the results.

**Disclosure — test numbers seen before the grid (2026-11-09).** Exploratory
timing and smoke runs, made outside the tracked grid, printed Gold-test MAEs on
the `random` split:
- **Levels 0 and 1:** every RQ3 split and seed.
- **Level 3:** seed 0, after tuning on validation.
- **Levels 2, 4 and 5:** seed 0 only. Level 2 was run in two configurations;
  levels 4 and 5 for one epoch on 20,000 molecules, for timing.

The level 2 choices (Δ-learning on level 1, and the 1% ECFP bit-frequency
filter) were made after seeing level 2's seed-0 validation **and** test MAE
(U0 0.26 eV without Δ-learning, 0.09 eV with it). No other choice used these
numbers. All of these runs are discarded; the grid recomputes every cell.
Read the level 2 `random` result with this in mind.

- **2026-11-09 — Level 2 model.** scikit-learn `HistGradientBoostingRegressor`
  replaces LightGBM: the LightGBM macOS wheel needs a system OpenMP library
  (Homebrew `libomp`) that this machine lacks, and the scikit-learn model is
  the same histogram gradient-boosting algorithm with bundled OpenMP. Inputs:
  the 184 RDKit descriptors plus the ECFP4 count bits present in at least 1%
  of the training molecules (selected on training rows only). Ridge is not run
  as a separate level; Phase 7's ridge is the exploratory reference.
- **2026-11-09 — Level 3 inputs and size.** KRR uses the sorted Coulomb-matrix
  eigenvalues only: Phase 6 built no Bag-of-Bonds table. Its exact solve needs
  O(N²) memory, so each run trains on a seeded sample of at most 20,000
  training molecules; `n_train_used` is reported.
- **2026-11-09 — Levels 4 and 5.** D-MPNN (Chemprop's architecture) and
  SchNet are re-implemented in plain PyTorch on the Phase 6 graph tables and
  the coordinates (no Chemprop, Lightning or PyG packages), with a fixed
  30-epoch budget and the best epoch chosen on validation. They are
  under-trained relative to the literature, which the results state. Coverage
  is limited by compute, as `configs/benchmark.yaml` (`coverage`, `not_run`)
  records: SchNet runs only the `random` split (RQ3 reference and RQ4) and RQ2
  if it is the best level.
- **2026-11-09 — Δ-learning.** Levels 2–5 learn the residual of level 1
  (dressed atoms) fitted on the same training rows. The plan was silent on
  this; it applies to every level alike.
- **2026-11-09 — Records without RDKit features.** 22 Bronze records whose bond
  orders could not be perceived have no descriptors, fingerprints or graphs.
  They are excluded from every level's training set, so all levels train on
  the same molecules. Every Gold record has features, so evaluation sets are
  unaffected.
- **2026-11-09 — How verdicts combine (clarification).** A cell meets its
  decision rule when its CI condition holds *and* its Holm-adjusted one-sided
  p-value is below 0.05. The p-values are bootstrap p-values for RQ3 and RQ5,
  and Mann–Whitney p-values for RQ2. An RQ is `supported` when every cell meets
  its rule, `refuted` when none does, and `mixed` otherwise. RQ3 counts every
  model, level 0 included, as the plan says "for each model". For RQ5 the
  verdict uses the primary model (level 2): `refuted` needs Gold-only to be
  significantly worse, and `inconclusive` means neither direction is
  significant. The RQ3 bootstrap resamples molecules within each seed's test
  set and pools the MAE over seeds. RQ1 is descriptive: its verdict records
  whether any record beyond the 3,054 fails a hard or identity check.
- **2026-11-09 — Tracking.** Runs are logged to a local MLflow file store,
  since the SQLite store needs the full `mlflow` server package. Runs are made
  before the Phase 9 commit exists, so each also carries `code_sha256`, a hash
  of the benchmark source and configuration, alongside `git_commit` (the
  parent commit) and `git_dirty`.
- **2026-11-09 — Level 4 drops out of RQ5 (compute).** A full-data timing run
  of level 4 (`tuning` experiment, `random` seed 0, U0 atomization, validation
  rows only; no test rows predicted) took 23.6 s per epoch on this machine's
  CPU, about 12 min per 30-epoch run; the best epoch was the last (29), with
  Gold validation MAE 0.049 eV. At that rate the full grid needs about 30–35 h,
  of which level 4's 210 RQ5 runs are about 12.5 h. The user chose to drop
  level 4 from RQ5 only (`coverage`, `not_run` in `configs/benchmark.yaml`).
  RQ5's verdict already rests on level 2 (primary), which runs every RQ5 cell,
  so the decision rule is unchanged; the secondary level 4 learning curve is
  reported as not run, with the reason. Level 4 still runs every RQ3 and
  stress split with 5 seeds, and RQ2 if it is the best level.
