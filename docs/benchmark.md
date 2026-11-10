# Phase 9 — benchmark and curation ablation

- **Runs:** `data/benchmark/runs/`, 682 runs, one Parquet + JSON pair each.
  `data/benchmark/run_metrics.parquet` has one row per run.
- **Report:** `reports/benchmark/benchmark_report.json`, with `summary.md` and
  `figures/` beside it.
- **Tracking:** MLflow file store in `mlruns/`.
- **Decisions:** ADR 0013. The protocol is the frozen analysis plan
  (`docs/analysis_plan.md`) plus its dated amendments.

All MAEs are on `test ∩ Gold` in eV. Each is pooled over 5 seeds and has a
95% bootstrap CI. The CIs are narrow (±1% or less) and are in `summary.md`.

## Verdicts

| RQ | Question | Verdict | In one line |
|---|---|---|---|
| RQ1 | Records failing checks beyond the 3,054 listed | supported | 8 unlisted records fail a hard or identity check; 1,466 listed records pass the independent identity check |
| RQ2 | Do flagged records have larger residuals? | supported | Silver and Bronze out-of-fold errors exceed Gold's on both targets (4/4 tests, Holm p ≤ 10⁻³) |
| RQ3 | Does the split choice inflate performance? | mixed | Every learned level is inflated on every split except one cell; levels 0–1 are not, as expected |
| RQ4 | Is the best MAE below the label-noise floor? | supported | Best U0 MAE 0.050 eV [0.049, 0.050] < 0.123 eV |
| RQ5 | Does curated training help on Gold test? | supported | Gold-only ≤ all records at N = 100,000 for level 2, by about 1 meV |

The leakage alarm did not fire. No `random` MAE is below half the published
SchNet MAE (0.007 eV for U0, 0.032 eV for the gap).

## The ladder on `random` (Gold-test MAE, eV)

| Level | Model | Gap | U0 atomization |
|---|---|---|---|
| 0 | mean | 1.070 | 8.119 |
| 1 | dressed atom | 0.818 | 0.862 |
| 2 | GBM (descriptors + ECFP4) | 0.144 | 0.088 |
| 3 | KRR (Coulomb eigenvalues, 20k) | 0.622 | 0.552 |
| 4 | D-MPNN | 0.110 | **0.050** |
| 5 | SchNet | **0.107** | 0.051 |

- Levels 4 and 5 are close: SchNet leads on the gap, D-MPNN on U0. Both
  were still improving at epoch 30; about half their runs had their best
  epoch last. Published SchNet reaches 0.063 eV (gap) and 0.014 eV (U0)
  with ~110k training molecules and far longer training, so these numbers
  are budget-limited.
- Level 3 is far behind. Coulomb-matrix eigenvalues on 20,000 molecules are
  a weak representation (`docs/limitations.md`).
- Chemical accuracy is 0.043 eV. No level reaches it on average for U0.
  D-MPNN gets 62% of Gold validation molecules within it (timing run).

## RQ3 — split inflation (ratio of MAE to `random`)

| Model | Target | isomer_group | scaffold_generic | cluster_ecfp | size_extrapolation |
|---|---|---|---|---|---|
| GBM | gap | 1.30 | 1.55 | 1.08 | 1.97 |
| GBM | U0 | 1.44 | 2.92 | 1.11 | 3.36 |
| KRR | gap | 1.16 | 1.10 | 1.04 | 1.31 |
| KRR | U0 | 1.26 | 1.10 | 1.01 (n.s.) | 1.58 |
| D-MPNN | gap | 1.08 | 1.72 | 1.11 | 1.79 |
| D-MPNN | U0 | 1.08 | 3.76 | 1.15 | 3.09 |

**What `mixed` means here.**
- The verdict rule counts every model, level 0 included, because the plan
  says "for each model". Levels 0 and 1 are not inflated on several splits.
  - The training mean has no structure to overfit.
  - The dressed-atom fit depends only on composition.
- Among learned levels, 23 of 24 cells inflate. The one exception is KRR U0
  on `cluster_ecfp`.
- For the levels a reader would actually use, the answer is a clear yes.

**Which splits matter.**
- `scaffold_generic` and `size_extrapolation` are the hardest. They inflate
  D-MPNN and GBM U0 error 3–4×.
- `cluster_ecfp` changes little (1.04–1.15×). Phase 8 found its test set is
  barely less similar to train than `random`'s (median Tanimoto 0.57 vs 0.62).

## Stress tests (no hypothesis)

| Split | GBM gap | D-MPNN gap | GBM U0 | D-MPNN U0 |
|---|---|---|---|---|
| element_holdout (no F in train) | 0.411 | 0.384 | 5.82 | 6.19 |
| property_tail_gap (top 5% gap) | 0.435 | 0.384 | 0.120 | 0.067 |
| property_tail_u0_atomization (top 5% U0) | 0.171 | 0.120 | 0.133 | 0.045 |

- **Element hold-out:** every level fails on U0. The cause is the Δ-learning
  baseline: with no fluorine in training, the dressed-atom fit has no F
  coefficient, and every level learns its residual. It measures that blind
  spot, not the networks' transfer to a new element.
- **Property tails:** extrapolating to the extreme 5% of the gap costs both
  models 3–3.5× their `random` gap error. For the U0 tail, D-MPNN stays at
  its `random` U0 level (0.045 vs 0.050), while GBM grows 1.5×.

## RQ2 — do flagged records hurt models?
Out-of-fold absolute residuals come from 5 folds by leakage group over the
training population. Each target uses the level with the best `random`
validation MAE: SchNet for the gap, D-MPNN for U0.

| Target | Model | Tier | n | MAE tier | MAE Gold | P(tier > Gold) | p (Holm) |
|---|---|---|---|---|---|---|---|
| gap | SchNet | Silver | 1,339 | 0.153 | 0.125 | 0.549 | 5 × 10⁻¹⁰ |
| gap | SchNet | Bronze | 1,283 | 0.264 | 0.125 | 0.634 | 10⁻⁶¹ |
| U0 | D-MPNN | Silver | 1,339 | 0.061 | 0.053 | 0.525 | 9 × 10⁻⁴ |
| U0 | D-MPNN | Bronze | 1,283 | 0.126 | 0.053 | 0.645 | 10⁻⁷¹ |

- Bronze records are clearly harder: about 2× Gold's MAE on both targets.
  They break the structural assumptions the models rely on (e.g. rearranged
  frameworks, proton transfers).
- Silver records are only slightly harder: effect size 0.52–0.55.
- This changes Phase 7's exploratory finding, where Silver looked no
  different (P = 0.50) with a ridge model. The stronger models resolve a
  small Silver effect that ridge could not.

## RQ4 — the label-noise floor
- The best `random` U0 MAE is 0.050 eV [0.049, 0.050], from D-MPNN. SchNet
  reaches 0.051 and GBM 0.088.
- All three are below the 0.123 eV MAE between Gold B3LYP and G4(MP2) U0
  (Phase 7).
- So the best models already reproduce the B3LYP labels more closely than
  B3LYP agrees with a higher level of theory. Further gains on QM9 U0
  measure fidelity to B3LYP, not to the true energy.
- The electronic-only (ZPE-separated) comparison is still open
  (`docs/limitations.md`).

## RQ5 — curated training on a fixed Gold test set (level 2)
Paired bootstrap on the same Gold test molecules. Each row is the MAE
averaged over seeds.

| N | Gap: all | Gap: gold | Gap: gold − all [95% CI] | U0: all | U0: gold | U0: gold − all [95% CI] |
|---|---|---|---|---|---|---|
| 1,000 | 0.354 | 0.357 | +0.0035 [+0.0013, +0.0057] | 0.374 | 0.370 | −0.0046 [−0.0069, −0.0023] |
| 10,000 | 0.229 | 0.229 | +0.0008 [−0.0006, +0.0022] | 0.182 | 0.180 | −0.0019 [−0.0032, −0.0006] |
| 100,000 | 0.146 | 0.145 | −0.0006 [−0.0012, −0.0000] | 0.090 | 0.089 | −0.0011 [−0.0017, −0.0006] |

- At the largest common N (100,000), Gold-only is not worse on either target:
  Holm p = 0.025 for the gap and 0.001 for U0. The rule is met, so the
  verdict is supported.
- The effect is tiny. It is about 1 meV, under 1% of the MAE, and it changes
  sign at smaller N on both targets.
- The learning-curve slopes match across tier sets: log-MAE vs log-N is
  −0.20 for the gap and −0.32 for U0, for every tier set.
- At these sizes, the ~2.4% of non-Gold training records barely matter to
  level 2. Their effect shows in RQ2's residuals, not in Gold-test accuracy.
- The secondary level 4 curve was not run (ADR 0013).

## Compute
- **Total:** 40.6 h of fit-and-predict time, from 2026-11-09 14:47 to
  2026-11-10 10:42 IST. Three processes ran in parallel, with SchNet paused
  for about 9 h to reduce heat.
- **By level:** D-MPNN 19.9 h, SchNet 10.8 h, GBM 9.2 h, KRR 0.7 h.
- **Provenance:** every run carries the same `code_sha256` and parent commit
  4b08cc5.

## Figures
- `reports/benchmark/figures/rq3_mae_by_split.png`
- `reports/benchmark/figures/rq5_learning_curves.png`
