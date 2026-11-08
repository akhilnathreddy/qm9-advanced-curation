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
None.
