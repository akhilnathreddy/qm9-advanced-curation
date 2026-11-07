# Phase 7 — statistical, scientific and multi-fidelity validation

The numbers come from `reports/statistical/statistical_report.json`. The
per-record scores are in `data/processed/review.parquet`, and the G4(MP2)
values are in `data/processed/g4mp2_xref.parquet`. ADR 0011 records the
decisions. Every flag here is **soft**: 384 records carry a review reason,
and no tier changed.

## 1. Multi-fidelity cross-check (RQ4)

The external reference is the QM9-G4(MP2) set of Kim et al. (2019, CC0):

- 133,858 converged molecules and 27 not converged; every count matches the
  paper;
- the formula of every row matches the QM9 composition;
- recomputing the authors' `atomization.csv` from their own tables agrees to
  5 × 10⁻¹³ Ha.

**The published nitrogen reference is wrong.** `atom_ref/N.log` was run as a
doublet (`Multiplicity = 2`), but nitrogen's ground state is the ⁴S quartet.
- Fitting AE(B3LYP) − AE(G4(MP2)) on element counts with the published
  references gives **2.380 eV per N atom**.
- NIST gives 2.3835 eV for the N ²D–⁴S splitting.
- Every N-containing G4(MP2) atomization energy in the published tables
  is therefore too large in magnitude by about 2.4 eV per N.
- This pipeline withholds them, with the reason recorded, rather than
  inventing a corrected reference.

**Difference between the levels of theory, Gold, 0 K:**

| Quantity | n | Mean | SD | MAE | p01 … p99 |
|---|---|---|---|---|---|
| AE(B3LYP) − AE(G4(MP2)), molecules without N | 50,929 | +0.171 eV | 0.166 eV | 0.196 eV | −0.21 … +0.55 eV |
| Residual after the per-element fit, all Gold | 130,621 | 0.000 eV | **0.153 eV** | **0.123 eV** | −0.31 … +0.38 eV |

- **B3LYP underbinds on average:** AE(B3LYP) is less negative.
- **Fixed by a per-element correction:** the mean offset and the size-linear
  part of the error. A dressed-atom correction would absorb these.
- **Not fixed by it:** a residual with an SD of 0.153 eV (3.5 kcal/mol) and
  an MAE of 0.123 eV (2.8 kcal/mol). That is nearly 3 × "chemical accuracy"
  (1 kcal/mol = 0.043 eV).
- **Preliminary RQ4 answer:** this residual is the noise floor of QM9's
  energy labels relative to G4(MP2). A model whose random-split MAE on U0 is
  far below 0.12 eV is reproducing B3LYP, not approaching the G4(MP2) value.
  Phase 9 compares this number with the measured model MAEs.

**The largest residuals are chemistry, not data errors:**
- **Polyynes** such as `C#CC#CC#CC#CC` (B3LYP overbinds conjugated triple
  bonds) and **strained polycyclic cages**.
- Only one record exceeds |z| = 6: **133858** (Bronze, structure changed in
  Phase 4).
- By tier, the residual is similar across Gold (MAE 0.123 eV), Silver
  (0.117 eV) and Bronze (0.131 eV).

**Geometry agreement with the re-optimized structures:**
- **Changed by more than 1 kcal/mol:** 23 records, matching the paper.
  - 20 are Gold, 2 Bronze, and 1 Quarantine: 117523, the authors'
    trimolecular case at 20 kcal/mol.
  - These 23 are not same-geometry comparisons. They are excluded from the
    statistics and flagged `g4mp2_geometry_changed`.
- **Not converged:** 27 records. 8 of them are on the QM9 authors'
  hard-to-converge list, and 18 are Gold. They are flagged
  `g4mp2_not_converged`.

## 2. Corroboration of the Phase 4–5 findings (RQ1)

| Authors' finding | This pipeline | Agreement |
|---|---|---|
| 229 multi-molecule records (graph components and InChI dots) | 229 `fragmented` (Hückel perception) | **Identical sets** |
| 252 duplicate pairs (connectivity graph isomorphism) | 453 duplicate records in 222 exact-structure groups | All 453 of our duplicates are in their pairs; 231 of their 252 pairs share our structure group |

**The 21 pairs they call duplicates and we do not:**
- In every one, the two records have the same constitution from the
  geometry.
- 13 also have the same geometry SMILES; the other 8 differ only in
  stereochemistry.
- None was grouped by our exact-structure test. That test needs the same
  GDB label or a near-identical geometry, and these pairs have different
  GDB labels and are different conformers or stereoisomers. Some are
  separated by up to 15 kcal/mol.
- They are distinct calculations of the same molecule. **For Phase 8 this
  matters:** splits must group by the geometry-perceived identity as well,
  or the same molecule can appear on both sides of a split.

## 3. Composition residuals

- **The fit:** U0 atomization energy on element counts plus an intercept, on
  Gold. The residual SD is 1.09 eV and the MAE 0.87 eV. That residual is
  what bonding contributes beyond a per-atom average.
- **No record exceeds |z| = 6 within its size stratum;** the largest is 5.49.
- **The 25 largest are documented, not removed.**
  - They are mostly strained, bridged polycycles: 72 % have a 3- or
    4-membered ring (62 % of Gold), 68 % have bridgeheads (12 % of Gold),
    and 72 % have three or more rings (23 % of Gold).
  - The other 7 are N/O-rich rings with no strain or conjugation
    descriptor, such as `n1nnon1` and `O=[N+]([O-])c1nnon1`. One of them is a
    Quarantine record (129158).

## 4. Inter-target structure

- **Collinearity:** the U0, U298, H298 and G298 atomization energies have a
  minimum pairwise Pearson r of 0.99980, both in Gold and in all records.
- **Thermal energy (U298 − U0):** no record exceeds |z| = 6; the maximum is
  4.7.
- **−TS (G298 − H298):** 8 records exceed |z| = 6. All 8 are already Bronze
  `fragmented` complexes, small molecules held together with CO₂
  (e.g. `C=CC#N.N.O=C=O`). Several loosely bound molecules have more
  low-frequency motion and a larger entropy, so the flag is physically
  expected. It also independently confirms that those records are not single
  molecules.

## 5. Distributions by tier (vs. Gold)

- **All records vs. Gold:** the KS statistic is ≤ 0.011 for every target.
  Restricting to Gold barely changes the marginal distributions.
- **Silver and Bronze differ strongly from Gold:**
  - KS for atomization energies is 0.54 (Silver) and 0.25 (Bronze);
  - for the gap, 0.54 and 0.30;
  - for the dipole, 0.21 and 0.53.
- These are not random subsets: the uncharacterized-list and changed-
  structure records come from particular regions of chemical space.
- **Consequence for RQ5:** a curated training set changes chemical coverage
  as well as label quality, so the ablation must compare at matched N on a
  fixed Gold test set (as planned).

## 6. Model-based suspicion score

- **The model:** out-of-fold ridge on element counts plus ECFP4 counts, with
  folds that keep exact duplicates together. The OOF MAE is 0.32 eV for U0
  atomization and 0.40 eV for the gap. This is a weak, bond-additive model
  and only ranks records.
- **Flagged:** 329 records exceed |z| = 6 (293 Gold, 8 Silver, 27 Bronze,
  1 Quarantine).
- **What is at the top:** unusual chemistry for a linear fingerprint model,
  not label errors.
  - Records such as CF₄, the nine-membered ring `c1cccc[nH]ccc1`, and
    zwitterionic records.
  - 306 of the 329 have nine heavy atoms.
- Phase 9 recomputes the score with the strongest benchmark model before any
  record is reviewed by hand.

**Preliminary RQ2 (same weak model, one-sided Mann–Whitney on |OOF
residual|):**

| Comparison | U0 atomization: P(non-Gold > Gold) | p | Gap: P(> Gold) | p |
|---|---|---|---|---|
| Bronze vs. Gold | 0.60 | 1 × 10⁻⁴⁵ | 0.55 | 1 × 10⁻¹² |
| Silver vs. Gold | 0.50 | 0.65 | 0.54 | 4 × 10⁻¹⁰ |

- **Bronze:** Bronze records (changed or fragmented structures) are harder to
  predict.
- **Silver:** for energies, Silver records (on the uncharacterized list but
  consistent) are not harder to predict than Gold. That supports keeping them
  out of Quarantine.
- This is recorded before the analysis plan is frozen. The confirmatory test
  is in Phase 9.

## Reproduce

```bash
uv run dvc repro review
make gate7
```
