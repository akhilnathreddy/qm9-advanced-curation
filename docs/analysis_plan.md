# Analysis plan

- **Frozen:** no — must be frozen (commit hash recorded here) before Phase 9 runs.
- **Frozen at commit:** —

Changes after freezing are allowed only as dated amendments at the bottom,
each with a reason, and must be reported alongside the results.

## Research questions

| ID | Question | Hypothesis | Primary metric | Test |
|---|---|---|---|---|
| RQ1 | How many records fail identity/physics checks beyond the 3,054 uncharacterized? | A non-trivial number do | Flag counts, overlap with uncharacterized list | Descriptive |
| RQ2 | Do flagged records hurt models? | Flagged records have larger out-of-fold residuals | Residual distribution by tier | Mann–Whitney U |
| RQ3 | How much do split choices inflate performance? | Random-split MAE < isomer/cluster/size-split MAE | Test MAE with 95% bootstrap CI | Paired bootstrap |
| RQ4 | What is the label-noise floor? | Δ(B3LYP − G4MP2) exceeds best random-split MAE | Δ distribution vs. MAE | Descriptive |
| RQ5 | Does curated training help on a fixed Gold test set? | Gold-only ≤ all-records MAE at matched N | Gold-test MAE | Paired bootstrap |

## Targets
- HOMO–LUMO gap (eV)
- U0 atomization energy (eV), using the release's atom-reference energies

## Splits
To be fixed from `configs/splits.yaml` after Phase 8.

## Models
Baseline ladder levels 0–5 (plan v2 §7.1). No models beyond the ladder in v1.0.

## Protocol
- ≥ 5 seeds; hyperparameters tuned on validation only; test touched once per
  (model, split, target).
- Learning curves at N ∈ {1k, 2k, 5k, 10k, 25k, 50k, 100k}.
- Results much better than published numbers for the same model and split
  trigger a leakage investigation before reporting.

## Preliminary results (descriptive, before freezing)
- RQ1 (Phase 4): beyond the 3,054 listed, 5 records fail the independent
  identity check; conversely 1,466 listed records pass it
  (`docs/validation.md`). Recorded here so the frozen plan can reference it.

## Amendments
None.
