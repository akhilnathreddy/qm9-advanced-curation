# 0009 — Standardization and quality tiers

- **Status:** Accepted
- **Date:** 2026-11-07
- **Phase:** 5

## Context
Phases 3–4 produced per-record evidence (duplicate groups, 19 check results
with severities). Phase 5 turns it into one curated table without deleting
anything, and must decide units, how to represent undefined values, which
duplicate to prefer, and what each tier means.

## Decision — representation
- **Never overwrite source values.** All 15 source properties are carried as
  `source_<column>` in source units, bit-identical to the parsed tables
  (checked by Gate 5).
- **Standard units:** energies in eV (×27.211386245988, CODATA 2018),
  α in Å³, ⟨R²⟩ in Å²; μ (D), rotational constants (GHz), Cv (cal/(mol·K))
  unchanged.
- **Atomization energies** for U0, U, H, G use only the release's
  `atomref.txt` (same level of theory): E_mol − Σ E_atom (negative).
  Methane: −0.631066 Ha = −17.172181 eV, checked by hand in a test.
- **Linear molecules** (11, by geometry): `rot_a_ghz` is null with
  `rot_a_missing_reason`; the source's 0 or huge value stays in
  `source_rot_a_ghz`.
- **Identifiers from the geometry:** canonical isomeric SMILES and standard
  InChIKey of the structure perceived from 3D (Hückel connectivity, bond
  orders for a neutral molecule, stereo from coordinates). 22 records fail
  bond-order assignment; all are Bronze.
- **Charge-separated GDB-17 SMILES** are kept as written and flagged
  (`gdb_charge_separated`), not neutralized: the geometry is the ground truth.

## Decision — duplicates
One representative per exact-structure group (ADR 0007): fewest failed hard,
then identity, then soft checks; ties to the smallest `qm9_id`. The choice
never looks at property values.

## Decision — tiers (`configs/curation_rules.yaml`)

| Rule | Tier | Matches | Records it alone moves |
|---|---|---|---|
| R1 hard check failed | quarantine | 5 | 5 |
| R2 geometry fragmented | bronze | 229 | 227 |
| R3 structure changed (proton moved / rearranged) | bronze | 1,364 | 1,364 |
| R4 on uncharacterized list, but 3D structure is the GDB-17 molecule | silver | 1,466 | 1,459 |
| R5 duplicate, not the representative | silver | 231 | 159 |
| R6 authors' difficult-to-converge list | silver | 11 | 5 |

Result: **gold 130,659 · silver 1,630 · bronze 1,591 · quarantine 5.**

Deliberately **not** tier rules (flags remain available):
- **Statistical outliers** (1,451): unusual is not wrong. Demoting them would
  make Gold more "typical" and flatter benchmark scores.
- **Relaxed-SMILES mismatch** (1,549): an error in the authors' annotation,
  not the data; the curated table carries identifiers from the geometry.
- **Linearity representation** (6): fixed by the standardized columns.

**R4 is a judgement call.** The plan's draft put every uncharacterized
record in Bronze. Phase 4 showed 1,466 of them have the intended structure
after optimization; the authors flagged them because the *starting*
geometry differed. Silver keeps them out of Gold while letting users opt in;
`uncharacterized_listed` lets anyone reproduce the conventional 130,831-record
subset exactly.

## Alternatives considered
- *Drop all 3,054 uncharacterized records* (common practice): discards 1,466
  structurally sound records and 5 problems it does not catch.
- *Free-form rule expressions in YAML:* rules choose among named predicates in
  code instead, so a config edit cannot inject arbitrary logic.

## Consequences
- Gold is duplicate-free, contains no uncharacterized-listed record, and
  every Gold record has a geometry-derived identifier.
- The tier of every record is explained by `tier_reasons`.
