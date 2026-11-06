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

## Interim tables (`data/interim/*.parquet`, Phase 2)

Built by `python -m qm9_curation.parse.build`, validated by the Pandera
schemas in `src/qm9_curation/schema.py`. Values are in **source units**;
the unit is part of each column name. Role: **id** identifier,
**meta** metadata/provenance, **struct** structure (model input),
**target** candidate prediction target, **qc** by-product of the quantum
calculation (not an input when predicting thermochemistry; plan v2 §4,
Phase 6).

### `molecules` — 133,885 rows, key `qm9_id`

| Column | Type | Unit | Role | Meaning |
|---|---|---|---|---|
| qm9_id | UInt32 | — | id | Source index (1-based) |
| n_atoms | UInt8 | — | struct | Atom count |
| n_heavy | UInt8 | — | struct | Non-hydrogen atom count (1–9) |
| formula | String | — | struct | Hill-order formula, e.g. `C2H2` |
| rot_a_ghz, rot_b_ghz, rot_c_ghz | Float64 | GHz | qc | Rotational constants; A = 0 for the 5 linear molecules |
| mu_debye | Float64 | D | target | Dipole moment |
| alpha_bohr3 | Float64 | Bohr³ | target | Isotropic polarizability |
| homo_ha, lumo_ha, gap_ha | Float64 | Ha | target | Frontier orbital energies and gap |
| r2_bohr2 | Float64 | Bohr² | target | Electronic spatial extent |
| zpve_ha | Float64 | Ha | target / qc | Zero-point vibrational energy |
| u0_ha, u298_ha, h298_ha, g298_ha | Float64 | Ha | target | U(0 K), U, H, G at 298.15 K |
| cv_cal_per_mol_k | Float64 | cal/(mol·K) | target | Heat capacity at 298.15 K |
| smiles_gdb, smiles_relaxed | String | — | struct / meta | GDB-17 SMILES; SMILES of relaxed geometry |
| inchi_gdb, inchi_relaxed | String | — | struct / meta | Same, as InChI |
| n_modes | UInt16 | — | meta | Modes per frequency set (3N−6, or 3N−5 if linear) |
| n_frequency_sets | UInt8 | — | meta | 1, or 2 for the 433 doubled records |
| is_linear | Boolean | — | struct | Frequency count is 3N−5. **Misses 6 physically linear molecules** (docs/profiling.md, P14) |
| has_fortran_exponent | Boolean | — | meta | Record contained a `*^` token |
| source_member | String | — | meta | File name inside the archive |
| source_sha256 | String | — | meta | SHA-256 of this record's source bytes |
| source_archive_sha256 | String | — | meta | SHA-256 of the archive (matches `MANIFEST.json`) |

### `atoms` — 2,407,753 rows, key (`qm9_id`, `atom_idx`)

| Column | Type | Unit | Meaning |
|---|---|---|---|
| qm9_id | UInt32 | — | Molecule |
| atom_idx | UInt8 | — | 0-based position in the source record |
| element, atomic_number | String, UInt8 | — | H/C/N/O/F and 1/6/7/8/9 |
| x_ang, y_ang, z_ang | Float64 | Å | Cartesian coordinates (B3LYP geometry) |
| mulliken_charge_e | Float64 | e | Mulliken partial charge (qc) |

### `frequencies` — 6,438,713 rows, key (`qm9_id`, `set_idx`, `mode_idx`)

| Column | Type | Unit | Meaning |
|---|---|---|---|
| qm9_id | UInt32 | — | Molecule |
| set_idx | UInt8 | — | 0, or 1 for the second set of a doubled record |
| mode_idx | UInt16 | — | 0-based mode position as written |
| wavenumber_cm1 | Float64 | cm⁻¹ | Harmonic frequency; negative = imaginary mode |

### `parse_log` — 609 rows

| Column | Type | Meaning |
|---|---|---|
| qm9_id | UInt32 | Molecule |
| event | String | `fortran_exponent` (176) or `doubled_frequencies` (433) |
| location | String | Where in the record, e.g. `atom[0].x` |

## Profile tables (`data/interim/`, Phase 3)

### `molecule_profile` — 133,885 rows, key `qm9_id`

| Column | Meaning |
|---|---|
| gdb_rdkit_ok, relaxed_rdkit_ok | RDKit parsed the GDB-17 / relaxed SMILES |
| canonical_smiles_gdb / _relaxed | RDKit canonical isomeric SMILES (null if unparseable) |
| flat_smiles_gdb / _relaxed | Same without stereo |
| inchikey_gdb / _relaxed | InChIKey from the source InChI strings |
| n_rings, ring_sizes | SSSR ring count and comma-separated sizes |
| n_aromatic_atoms | RDKit aromatic atom count |
| murcko_scaffold, generic_scaffold | Bemis–Murcko scaffold; `""` for acyclic |
| n_charged_atoms, n_radical_electrons, n_stereocentres | From the relaxed SMILES (GDB-17 if it won't parse) |
| min_interatomic_ang, max_interatomic_ang, radius_of_gyration_ang | Geometry statistics (Å) |
| connectivity_group_id / _size | InChIKey first block (skeleton) |
| tautomer_group_id / _size | Full standard InChIKey of GDB-17 molecule — **tautomers share it** |
| structure_group_id / _size | Exact-structure duplicates (ADR 0007) |
| in_smiles_duplicate, in_geometric_duplicate | Which rule placed it in a structure group |
| max_abs_robust_z | Largest \|robust z\| over the 15 properties, within heavy-atom strata |
| uncharacterized | Listed in the authors' `uncharacterized.txt` |

Group ids are the smallest `qm9_id` in the group; singletons have their own id.

### `geometric_duplicate_pairs`

`qm9_id_a < qm9_id_b`, `max_distance_diff_ang`: pairs with near-identical
relaxed geometry (ADR 0007).
