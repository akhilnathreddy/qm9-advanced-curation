# 0007 — What counts as a duplicate

- **Status:** Accepted
- **Date:** 2026-11-06
- **Phase:** 3

## Context
"Duplicate" is ambiguous for QM9. Profiling found three very different
situations that naive deduplication would conflate:

1. 8,097 molecules (3,652 groups) share a **standard InChI**. They are
   tautomers: GDB-17 enumerates them separately and their energies differ by
   up to tens of kcal/mol. Standard InChI merges them by design (mobile-H layer).
2. 174 molecules (87 pairs) share an **RDKit canonical GDB-17 SMILES**. GDB-17
   listed two Kekulé forms of the same aromatic ring; both were computed and
   relaxed to the same structure.
3. Pairs with **near-identical relaxed geometries** but different GDB-17
   SMILES: mostly charge-separated resonance forms that DFT relaxed to the same
   neutral structure.

## Decision
- **Never deduplicate by InChI or InChIKey.** Tautomer groups are recorded
  (`tautomer_group_id`) and used only to build leakage-safe splits.
- An **exact-structure group** (`structure_group_id`) is the union-find closure
  of (2) and (3). These are true duplicates: the same molecule under more than
  one `qm9_id`. Phase 5 chooses a representative; Phase 8 keeps each group in
  one split.
- **Geometric match:** same formula; sorted Coulomb-matrix eigenvalues within
  **0.1** (candidate search), then every sorted interatomic distance within
  **0.05 Å** (confirmation).
- **Why these thresholds:** a scan (table below) shows a plateau. Up to
  0.1 / 0.05 Å every matched pair has |ΔU0| ≤ 0.023 kcal/mol, i.e. the same
  molecule; looser settings start pairing distinct molecules (|ΔU0| up to
  18.9 kcal/mol). The profile stage enforces this: Gate 3 fails if any
  geometric pair differs by more than 0.1 kcal/mol.

| Eigen radius | Distance tol (Å) | Pairs | Max \|ΔU0\| (kcal/mol) | Pairs > 0.1 kcal/mol |
|---|---|---|---|---|
| 0.01 | 0.01 | 228 | 0.014 | 0 |
| 0.1 | 0.01 | 235 | 0.021 | 0 |
| 0.1 | 0.02 | 238 | 0.021 | 0 |
| **0.1** | **0.05** | **239** | **0.023** | **0** |
| 0.5 | 0.1 | 243 | 0.779 | 2 |
| 1.0 | 0.3 | 553 | 18.873 | 290 |

## Alternatives considered
- *Deduplicate by InChIKey:* would delete ~4,400 distinct tautomers.
- *Permutation-aware RMSD:* more exact but quadratic in group size (up to
  6,095 isomers per formula); the two invariant fingerprints plus the energy
  check are sufficient here.

## Consequences
- 453 molecules in 222 exact-structure groups (largest: 3) need a
  representative choice in Phase 5.
- The geometric test cannot tell mirror images apart; a pair could be two
  enantiomers. GDB-17 has no stereo, so this does not arise from the
  enumeration itself, but it is stated as a limitation.
- One SMILES pair (25673/25677) differs by 0.28 Å in geometry (a conformer
  difference) but has the same SMILES and ΔU0 = 0.004 kcal/mol; it is caught
  by the SMILES rule, not the geometric one.
