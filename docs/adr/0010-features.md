# 0010 — Feature inputs, families and storage

- **Status:** Accepted
- **Date:** 2026-11-07
- **Phase:** 6

## Context
Phase 9 needs several input representations (plan v2 §7.1, levels 1–5).
The main risks are leakage (using outputs of the same quantum calculation
as inputs), features that describe the GDB-17 label rather than the computed
structure, and non-determinism.

## Decision — inputs (leakage guard)
Features may read only `qm9_id, atom_idx, element, atomic_number, x_ang,
y_ang, z_ang` (`features/inputs.py`). Every feature function rejects any
other column, so Mulliken charges, frequencies, ZPVE, rotational constants
and all targets cannot be used by accident. A test enforces this.

## Decision — which molecule
2D features describe the structure **perceived from the B3LYP geometry**
(same perception as Phase 5), not the GDB-17 SMILES. For Gold records they
coincide; for Bronze records the geometry is what the labels describe.
22 molecules (all Bronze) have no bond-order assignment; they have
composition and Coulomb features but no RDKit features, recorded in
`index.unavailable_reason`.

## Decision — families (`data/processed/features/`)

| Table | Content | Notes |
|---|---|---|
| composition | n_H, n_C, n_N, n_O, n_F, n_atoms, n_heavy | dressed-atom baseline |
| coulomb_eigenvalues | 29 sorted eigenvalues, a.u. (Bohr), zero-padded | invariant to pose and atom order |
| rdkit_descriptors | 184 of RDKit's 217 descriptors | 21 constant (S/P groups) and 12 with non-finite values (Gasteiger-charge based) dropped and listed |
| ecfp4 | Morgan radius 2, 2,048-bit counts, sparse | heavy-atom molecule |
| maccs | 167 keys, sparse | |
| graph_nodes / graph_edges | atom and bond features, bond lengths (Å) | `atom_idx` = source atom order, joins to `atoms` coordinates |

Bag-of-Bonds is implemented (`features/coulomb.py`) but computed on demand:
as a dense table it is ~1,000 columns per molecule. SOAP/ACSF are deferred.

**Not removed:** 13 descriptor pairs with |r| ≥ 0.99 are reported, not
dropped; collinearity handling is a modelling choice for Phase 9.

## Consequences
- Feature tables are deterministic (a test builds them twice and compares).
- Graph tables plus `atoms` coordinates are enough to build PyG `Data`
  objects in Phase 9 without re-running RDKit.
