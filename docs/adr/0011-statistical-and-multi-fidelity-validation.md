# 0011 — Statistical, scientific and multi-fidelity validation

- **Status:** Accepted
- **Date:** 2026-11-07
- **Phase:** 7

## Context
Layers 1–3 (Phase 4) check each record against itself. Phase 7 checks
records against the rest of the dataset and against an independent,
higher-level calculation, to estimate how far B3LYP labels are from a
better reference (RQ4) and to find records that are unusual for their
chemistry. Plan v2 makes every such flag soft: it routes a record to
review and never changes a tier or removes a record.

## Decision — external reference
- **Source:** Kim, Park, Choi, *Sci. Data* 6, 109 (2019), figshare collection
  10.6084/m9.figshare.c.4351631.v1. **License CC0** (figshare API,
  2026-11-07), so the joined values can be redistributed.
- Only `data_publication_main.v2.tar.gz` (30 MB) is fetched, verified against
  the published MD5 and recorded with SHA-256 like the raw QM9 files
  (`data/external/kim2019/MANIFEST.json`). The 14 Gaussian-log archives
  (~10 GB) are not needed.
- Configured archive members are read in memory. Nothing is extracted and
  none of the authors' scripts is run.
- **Counts must match the paper:** 133,858 converged molecules, 27 not
  converged, 229 multi-molecule records, 252 duplicate pairs, and 23 records
  whose re-optimization lowered the B3LYP energy by more than 1 kcal/mol.
- **Join key:** the QM9 index. For every row, the formula must agree with the
  QM9 composition.

## Decision — atom references
- The authors' atom logs are parsed for spin multiplicity and compared with
  the ground states (H 2, C 3, N 4, O 3, F 2).
- **The published N atom is a doublet** (`Multiplicity = 2` in
  `atom_ref/N.log`); the ground state is the ⁴S quartet.
- Fitting the B3LYP − G4(MP2) atomization difference on element counts with
  the published references gives a per-N coefficient of 2.380 eV. NIST
  gives 2.3835 eV for the N ²D–⁴S splitting, which corroborates the diagnosis.
- G4(MP2) atomization energies are therefore **withheld (null, with the
  reason) for every N-containing molecule**. No replacement reference is
  invented.
- The primary RQ4 quantity does not depend on atom references at all; see the
  next section.

## Decision — what the multi-fidelity difference means
- `delta_u0_atomization_ev`: AE(B3LYP, QM9 references) − AE(G4(MP2)). Only
  where every reference is valid; 50,929 Gold records.
- `delta_residual_ev`: U0(B3LYP) − U0(G4(MP2)), minus a per-element linear
  term fitted on Gold.
  - An AE difference and a total-energy difference differ only by a
    per-element constant, so this residual is identical whichever atom
    references are used.
  - It is the part of the B3LYP error that a dressed-atom correction cannot
    absorb, and it is **the label-noise floor reported for RQ4**.
- Records whose geometry the authors' re-optimization changed by more than
  1 kcal/mol (the paper's criterion) are excluded from the statistics. They
  are not same-geometry comparisons.
- G4(MP2) scales the B3LYP ZPE by 0.9854, while QM9's U0 uses the unscaled
  ZPE. That difference is part of Δ by construction, since both are the
  respective method's own U0.

## Decision — Layer 4 statistics (soft)
- Every score is a robust z, (x − median)/(1.4826 · MAD), **within
  heavy-atom strata** (1–4 heavy atoms pooled, as in Phase 3). The review
  threshold is |z| > 6, the same as Phases 3–4.
- **Composition residual:** U0 atomization energy ~ element counts +
  intercept (fit on Gold). The 25 largest residuals are annotated (strained
  small ring, bridged polycycle, conjugated triple bonds, three or more
  rings), and those rates are compared with Gold.
  - These are heuristics. For example, RDKit counts no bridgeheads in cubane.
  - The annotations explain residuals and never exclude records.
- **Inter-target:** Pearson r among the four atomization energies, plus
  robust z of the thermal energy (U298 − U0) and of −TS (G298 − H298).
- **Distribution tests:** two-sample KS (asymptotic) and Wasserstein-1, also
  in units of the Gold SD. Each tier with at least 20 records, and all
  records, is compared with Gold for all 16 standardized targets.
- **Suspicion score:**
  - Model: out-of-fold ridge (α = 1, untuned) on element counts plus ECFP4
    counts, 5 folds assigned by `structure_group_id mod 5`, so exact
    duplicates share a fold.
  - Per target (U0 atomization, gap), the score is the robust z of the
    residual; the overall score is the maximum |z|.
  - This is a weak, bond-additive model. Phase 9 recomputes the score with
    the strongest benchmark model.

## Outputs
- `data/processed/review.parquet`: one row per record with every score,
  `review_reasons` and the tier, which is copied and never changed.
- `data/processed/g4mp2_xref.parquet`: the plan's `external_xref` table.
- `reports/statistical/statistical_report.json`.

## Consequences
- 384 records carry at least one review reason, and no tier changed.
- RQ4 has a preliminary answer before any model is trained: the Gold
  residual SD is 0.153 eV (3.5 kcal/mol). Phase 9 compares it with model MAE.
- Downstream users of Kim et al.'s `atomization.csv` should know that every
  N-containing value is shifted by about 2.4 eV per N atom (see
  `docs/limitations.md`).
