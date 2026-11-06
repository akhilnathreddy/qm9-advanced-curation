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
