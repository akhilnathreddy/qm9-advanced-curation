# Known limitations and open uncertainties

Grows through the project. Each entry: what is uncertain, why it matters,
what was done, what remains.

- QM9 labels are B3LYP/6-31G(2df,p) values, not experimental or
  high-level reference values.
- One geometry per molecule; it is not guaranteed to be the global-minimum
  conformer.
- 433 records carry two different frequency sets; until Phase 4 decides,
  frequency-derived quantities for them are ambiguous
  (`docs/raw_data_observations.md`, O1).
- Five records are not true minima (imaginary frequencies); their
  thermochemistry is not well defined (O2).
- 453 molecules are exact-structure duplicates of another record (ADR 0007);
  until Phase 5 picks representatives, naive random splits leak.
- The authors' relaxed SMILES are Open Babel perceptions and unreliable for
  identity (746 unparseable, 12,075 with radicals); see docs/profiling.md.
- `is_linear` misses six physically linear molecules (P14).
- Identity from 3D depends on RDKit's extended-Hückel connectivity; for the
  5 framework rearrangements not on the authors' list, 3 are not corroborated
  by the authors' relaxed SMILES and need expert review.
- 1,466 records on the uncharacterized list look consistent by independent
  perception; whether to keep them is a Phase 5 decision, not a fact.

## Unresolved after Phase 7

Recorded for Gate 7. The evidence for each is in `docs/statistical_validation.md`.

- **The external nitrogen reference.**
  - *Uncertain:* the true G4(MP2) atomization energy of N-containing
    molecules. Kim et al.'s N atom is a doublet, which shifts their published
    atomization energies by about 2.4 eV per N.
  - *Done:* those values are withheld. RQ4 uses a residual that does not
    depend on atom references.
  - *Remaining:* raw AE differences for N-containing molecules need a
    ground-state N reference from a new G4(MP2) atom calculation or another
    G4(MP2) dataset. The Narayanan et al. 2019 set is still unverified
    (plan §15).
- **Δ mixes the two methods' ZPEs.**
  - *Uncertain:* how much of Δ is ZPE treatment rather than electronic
    energy. G4(MP2) uses a scaled B3LYP ZPE (0.9854) and QM9's U0 an unscaled
    one.
  - *Remaining:* separate the two using the authors' unscaled ZPE column if
    Phase 9 needs an electronic-only comparison.
- **The noise floor is relative to G4(MP2), not to experiment.**
  - *Uncertain:* the true label error. G4(MP2) has its own errors, of the
    order of 1 kcal/mol for small organics, and these are not quantified on
    QM9 itself.
  - *Consequence:* the 0.123 eV residual MAE is a lower bound on B3LYP's
    error only to the extent G4(MP2) is better.
- **Geometry changes and non-convergence.**
  - *Uncertain:* the reference values for these records. 23 records
    re-optimized more than 1 kcal/mol lower, and 27 did not converge at
    G4(MP2); 38 of these 50 are Gold.
  - *Done:* their QM9 labels may describe a non-minimum conformer, so they
    are flagged for review but kept in their tiers.
- **Same molecule, different records.**
  - *Uncertain:* the identity grouping used for splits. In 21 duplicate
    pairs from the external source, both records perceive to the same
    constitution, as different conformers or stereoisomers, but are not
    grouped by ADR 0007.
  - *Remaining:* Phase 8 must decide whether isomer or cluster splits group
    by geometry-perceived constitution, to avoid this leakage.
  - *Resolved in Phase 8:* every split groups by the geometry InChIKey
    connectivity block as well (ADR 0012).
- **A weak suspicion model.**
  - *Uncertain:* which records are genuinely suspicious. The ridge model has
    an OOF MAE of 0.32 eV for U0, and its top-ranked records are unusual
    chemistry, not demonstrated label errors.
  - *Remaining:* Phase 9 recomputes the score with the best model before any
    manual review.
- **Heuristic structure annotations.**
  - *Uncertain:* whether the ring-strain, bridgehead and polyyne labels
    explain the residuals; they are not a verdict. For example, RDKit's
    bridgehead count is 0 for cubane.
  - *Remaining:* the 25 largest composition residuals need expert review.
- **Silver and Bronze are not random subsets.**
  - *Uncertain:* how much an ablation measures label quality rather than
    coverage. Their target distributions differ strongly from Gold (KS up to
    0.54), so tier ablations change chemical coverage as well as label
    quality.
  - *Remaining:* RQ5 must keep the planned matched-N, fixed-Gold-test design.

## After Phase 8

- **Leakage groups merge more than duplicates.** Tautomers and stereoisomers
  are joined deliberately. They are distinct molecules with distinct labels,
  but training on one and testing on another overstates generalization. A
  study that wants them on both sides needs its own split.
- **Cluster and scaffold splits stay close to train.** Median test → train
  Tanimoto is 0.57 and 0.55 (random 0.62). QM9 is a dense enumeration, so
  these splits measure modest structural novelty, not out-of-distribution
  chemistry.
- **The cluster threshold is a choice.** Leader clustering at Tanimoto
  distance 0.65 on ECFP4 bits; another threshold or fingerprint gives
  different clusters. The threshold is in `configs/splits.yaml` and the
  report.
- **Holdout test sets are fixed across seeds.** Seeds vary only train/val,
  so seed variance underestimates test-set sampling variance for those
  splits. Bootstrap CIs over test molecules cover this (analysis plan).

## After Phase 9

- **Training budgets are below the literature's.**
  - Levels 4 and 5 train for 30 epochs; published models train for hundreds.
    Their best validation epoch was the last one in about half the runs
    (D-MPNN 52%, SchNet 47%).
  - Level 2's best boosting iteration equals its 1,000-iteration cap.
  - All three were still improving, so their MAEs are upper bounds for these
    architectures, not reproductions of published numbers.
- **Level 3 is weak by construction.** KRR uses only the sorted
  Coulomb-matrix eigenvalues, because Phase 6 built no Bag-of-Bonds table.
  Its exact solve is capped at 20,000 training molecules. The tuned λ (10⁻⁴)
  sits at the edge of its grid.
- **Coverage is incomplete, by choice (ADR 0013).**
  - SchNet runs only the `random` split.
  - D-MPNN does not run RQ5, so RQ5 speaks for level 2 only.
  - `not_run` in `configs/benchmark.yaml` gives the reasons.
- **The `element_holdout` stress test is dominated by the Δ-learning
  baseline.** Train contains no fluorine, so the dressed-atom fit has no F
  coefficient. Every level that learns its residual inherits an error of
  several eV on U0. That result measures the baseline's blind spot, not the
  models' ability to transfer to a new element.
- **Accuracy is relative to the B3LYP labels.** The best `random` U0 MAE is
  below the B3LYP − G4(MP2) disagreement (RQ4). Lower MAE past that point
  means fitting the DFT method more closely, not the true energy more
  closely.
- **RQ5's effect is small.** At N = 100,000, Gold-only training beats
  all-tier training by about 1 meV on both targets. That is significant on
  13k paired test molecules, but about 1% of the MAE. The learning-curve
  slopes are almost identical across tier sets.
- **Exploratory test numbers were seen before the grid.** The analysis-plan
  amendment of 2026-11-09 discloses them. Read the level 2 `random` result
  with that in mind.
- **Long-lived grid processes keep their tuning winners from start-up.** A
  process that starts before another level's tuning finishes cannot pick the
  RQ2 model, and waits instead. RQ2 is filled by a later process; the final
  plain run checks that every cell exists.
- **Still open from Phase 7:**
  - the suspicion score is not yet recomputed with the best model;
  - the electronic-only (ZPE-separated) RQ4 comparison was not run.

## After Phase 10

- **Byte identity holds only on the reference platform.** The published
  manifest is the Linux x86-64 build (ADR 0001).
  - On macOS, `review` and `g4mp2_xref` differ in their least-squares float
    columns by at most 4.5 × 10⁻¹¹, because Accelerate and OpenBLAS round
    differently.
  - The macOS build is therefore verified as *equivalent* (floats within
    10⁻⁹, everything else equal), not identical. That check needs the
    reference files (`make reproduce REFERENCE=…`).
  - Another BLAS, CPU or library version could move these digits again. The
    tolerance has five orders of magnitude of margin.
- **The extended XYZ is large (201 MB) because it is uncompressed.**
  Compressing it would make its hash depend on the zlib build. The Zenodo
  zip compresses it for download.
- **No DOI yet.** Minting one is the user's manual Zenodo upload. Adding
  the DOI to the citation text is a PATCH release.
- **Feature tables are not released.** Users who want the Phase 6 features
  rebuild them with the pipeline (`make repro`).
- **The PyG graphs have no bonds.** They carry atomic numbers and positions
  only. Bond graphs need the Phase 6 graph tables, which are not released,
  or a radius graph.
- **Raw-archive availability is outside this project's control.**
  `make reproduce` needs figshare to keep serving the original files. Their
  MD5s are pinned, so any change is detected, but it cannot be repaired from
  here.
