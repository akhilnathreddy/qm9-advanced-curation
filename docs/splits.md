# Phase 8 — split suite

- **Splits:** `data/splits/splits.parquet`, in long format `(qm9_id, split_name,
  seed, subset)`. There are 8 splits × 5 seeds, each assigning all 133,885
  records.
- **Groups:** `data/splits/split_groups.parquet`.
- **Report:** `reports/splits/split_report.json`.
- **Decisions:** ADR 0012.

Every split keeps each **leakage group** in one subset. A leakage group joins
records that share an exact structure, tautomer, GDB connectivity, or the
connectivity block of the InChIKey perceived from the geometry. There are
129,343 groups, the largest with 9 records. Without this grouping, a
record-level random split would leave **774 test records (5.8%)** sharing a
group with train/val, 43 of them exact duplicates.

## Sizes and shift (seed 0; shift on Gold, test vs. train)

| Split | Units | Train / val / test | NN Tanimoto test→train, median [p05, p95] | Gap: KS, W/SD | U0 atomization: KS, W/SD | Heavy atoms: W |
|---|---|---|---|---|---|---|
| random | 129,343 | 107,109 / 13,388 / 13,388 | 0.62 [0.48, 0.79] | 0.006, 0.01 | 0.010, 0.02 | 0.00 |
| isomer_group | 621 | 107,109 / 13,388 / 13,388 | 0.63 [0.47, 0.74] | 0.061, 0.09 | 0.263, 0.30 | 0.08 |
| scaffold_generic | 737 | 107,109 / 13,388 / 13,388 | 0.55 [0.41, 0.81] | 0.127, 0.25 | 0.072, 0.14 | 0.02 |
| cluster_ecfp | 6,850 | 107,109 / 13,388 / 13,388 | 0.57 [0.43, 0.70] | 0.062, 0.14 | 0.062, 0.15 | 0.01 |
| size_extrapolation | 129,343 | 19,815 / 2,476 / 111,594 | 0.52 [0.38, 0.71] | 0.070, 0.19 | 0.411, 1.01 | 1.23 |
| element_holdout | 129,343 | 117,087 / 14,635 / 2,163 | 0.57 [0.36, 0.66] | 0.396, 0.82 | 0.660, 1.58 | 0.02 |
| property_tail_gap | 129,343 | 113,024 / 14,128 / 6,733 | 0.59 [0.45, 0.80] | 1.000, 2.08 | 0.480, 1.18 | 0.22 |
| property_tail_u0_atomization | 129,343 | 113,059 / 14,132 / 6,694 | 0.64 [0.50, 0.83] | 0.532, 1.25 | 1.000, 2.34 | 0.20 |

W/SD is the Wasserstein-1 distance in units of the train SD.

## How to read the splits

- **random:** an interpolation baseline, comparable to the literature except
  that leakage is removed.
- **isomer_group:** whole formulas are held out, so U0 atomization, which is
  dominated by composition, shifts (KS 0.26). Structural similarity is *not*
  reduced (NN 0.63): the same scaffolds recur across formulas. It tests
  composition novelty, not structural novelty.
- **scaffold_generic and cluster_ecfp:** ring-system or fingerprint novelty.
  - Test → train similarity drops only modestly (medians 0.55 and 0.57 against
    0.62), because QM9 is a dense enumeration.
  - Merging units with leakage groups creates one large unit per split
    (23,418 and 15,018 records), which goes to train.
- **size_extrapolation:**
  - train on molecules with ≤ 8 heavy atoms (19,815 records), test on all
    111,594 with 9;
  - the strongest composition shift (W = 1.23 heavy atoms);
  - the lowest NN similarity.
- **element_holdout and the property tails:** stress tests, with no
  hypothesis attached (analysis plan). By construction, train contains no F
  in `element_holdout`, and target-space extrapolation in the tails has
  KS = 1.

## Use

```python
import polars as pl

splits = pl.read_parquet("data/splits/splits.parquet")
mol = pl.read_parquet("data/processed/molecules.parquet")
s = splits.filter((pl.col("split_name") == "random") & (pl.col("seed") == 0)).join(mol, on="qm9_id")
train = s.filter((pl.col("subset") == "train") & (pl.col("quality_tier") != "quarantine"))
test_gold = s.filter((pl.col("subset") == "test") & (pl.col("quality_tier") == "gold"))
```
