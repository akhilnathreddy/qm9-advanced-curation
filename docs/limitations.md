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
