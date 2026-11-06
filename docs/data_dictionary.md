# Data dictionary

## Raw schema (source files, as delivered)

Confirmed against `readme.txt` and the inventory of all 133,885 records
(`reports/data_quality/inventory.json`). Phase 2 adds dtypes, normalized
units and the curated tables.

### `.xyz` record layout (one file per molecule, tab-separated)

| Line | Content |
|---|---|
| 1 | Number of atoms *nₐ* |
| 2 | Tag, index, then properties 3–17 (below). The tag is `gdb` in every record (the readme says `gdb9`). The line ends with a trailing tab. |
| 3 … nₐ+2 | Element, x, y, z (Å), Mulliken partial charge (e) |
| nₐ+3 | Harmonic vibrational frequencies (cm⁻¹): 3nₐ−6, or 3nₐ−5 for linear molecules. **433 records have two sets on this line** (see below). |
| nₐ+4 | SMILES from GDB-17, then SMILES of the relaxed geometry (trailing tab) |
| nₐ+5 | InChI from GDB-17, then InChI of the relaxed geometry |

### Properties on line 2

| # | Name | Source unit | Meaning |
|---|---|---|---|
| 1 | tag | — | constant `gdb` |
| 2 | index | — | 1-based molecule ID, matches the file name |
| 3–5 | A, B, C | GHz | Rotational constants (A = 0 for linear molecules) |
| 6 | mu | Debye | Dipole moment |
| 7 | alpha | Bohr³ | Isotropic polarizability |
| 8 | homo | Hartree | HOMO energy |
| 9 | lumo | Hartree | LUMO energy (the readme mislabels it "lowest occupied") |
| 10 | gap | Hartree | LUMO − HOMO |
| 11 | r2 | Bohr² | Electronic spatial extent ⟨R²⟩ |
| 12 | zpve | Hartree | Zero-point vibrational energy |
| 13 | U0 | Hartree | Internal energy at 0 K |
| 14 | U | Hartree | Internal energy at 298.15 K |
| 15 | H | Hartree | Enthalpy at 298.15 K |
| 16 | G | Hartree | Free energy at 298.15 K |
| 17 | Cv | cal/(mol·K) | Heat capacity at 298.15 K |

All at B3LYP/6-31G(2df,p).

### Number format

Fortran-style exponents (`2.1997*^-6` meaning 2.1997 × 10⁻⁶) occur in
**176 records, only in atom coordinates**. The parser must convert `*^` to
`e` and log each occurrence.

### `uncharacterized.txt`

A free-text header, then one row per molecule: index, GDB-17 SMILES,
SMILES from the B3LYP geometry, SMILES from the Corina starting geometry,
and D_IJ (L1 norm of the Coulomb-matrix difference, a.u.). Data rows are
the lines whose first token is an integer: **3,054 unique indices**,
58 to 133,866.

### `atomref.txt`

Per-element (H, C, N, O, F) ZPVE, U(0 K), U, H, G (Hartree) and Cv
(cal/(mol·K)) for isolated atoms, used to compute atomization energies.

## Curated schema

Filled in Phase 2.
