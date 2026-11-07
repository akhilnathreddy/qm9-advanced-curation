# 0008 — Validation methods and tolerances

- **Status:** Accepted
- **Date:** 2026-11-07
- **Phase:** 4

## Context
Phase 4 runs 19 checks in four layers (`src/qm9_curation/checks/registry.py`).
Each tolerance must come from evidence (print precision of the source, or a
clear gap in the residual distribution), never from "what makes records
pass". Bond perception from 3D needed a method choice of its own.

## Decision — physics tolerances (`configs/validation.yaml`)

| Check | Tolerance | Basis | Observed |
|---|---|---|---|
| gap = LUMO − HOMO | 1.5e-4 Ha | 3 values printed to 4 decimals: 3 × 0.5e-4 | max 1.0e-4 |
| H − U = k_B T | 1.5e-6 Ha | 2 values printed to 6 decimals | max 8.2e-7 |
| G < H, U > U0 | strict | thermodynamics | 0 violations |
| Σ Mulliken = 0 | n_atoms × 5e-7 | each charge printed to 6 decimals | max 6e-6 (29 atoms allow 1.45e-5) |
| ZPVE = ½ Σ ν | 1e-4 relative | single-set max 2.3e-5; unscaled harmonic frequencies | max 1.2e-5 for the chosen set |
| Rotational constants | 1e-4 relative | recomputed from geometry with most-abundant isotope masses | max 1.4e-5 |
| Linear by geometry | smallest moment ≤ 0.1 amu·Å² | linear molecules ≤ 0.0022; next (water) 0.632 | 11 linear |
| "Infinite" A | > 1e4 GHz | largest non-linear A is 800 GHz (water) | 2 polyynes |
| Statistical review | \|robust z\| > 6 within heavy-atom strata | review threshold from Phase 3 | 1,451 records |

## Decision — bond perception from 3D (Layer 2)
Use `rdDetermineBonds.DetermineConnectivity(useHueckel=True, charge=0)`.

The default covalent-radius method bonded the bridgehead carbons of strained
cages (bicyclo[1.1.1]pentane and relatives; C···C 1.88 Å). That produced
6,812 false "rearranged" records not on the authors' list, all of which the
authors' relaxed SMILES and GDB-17 agreed on. Its `covFactor` argument had
no effect in RDKit 2026.09 (identical counts at 1.1–1.3). Extended-Hückel
connectivity reduced those 6,812 to 5 and left every class on the
uncharacterized list essentially unchanged. A test pins the
bicyclo[1.1.1]pentane case.

Graphs are compared as skeletons (all atoms including H, single bonds, no
charges or aromaticity), so bond-order and charge notation cannot cause a
mismatch but a moved proton can. A heavy-atom-only skeleton separates proton
transfer from framework changes.

## Alternatives considered
- *Tolerances from residual percentiles* (e.g. 99.9th): would label 0.1% of
  records as failures by construction.
- *Distance-based connectivity with a smaller factor:* not available (factor
  ignored), and any fixed distance cutoff trades cage false positives against
  long real bonds.
- *Trust the authors' relaxed SMILES:* 746 unparseable, 12,075 with radicals
  (Phase 3).

## Consequences
- Every physics invariant holds for all records except the 5 imaginary-
  frequency structures; the stored numbers are internally consistent to
  print precision.
- Severity (`hard` / `identity` / `soft`) is recorded per check; what happens
  to a record is decided by the Phase 5 tier rules, not here.
