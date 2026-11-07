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
