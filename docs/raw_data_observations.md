# Raw data observations (Phase 1)

What the Phase 1 inventory found in the source files, and which later phase
handles each finding. Nothing here has been changed in the data. Source:
`reports/data_quality/inventory.json`.

| # | Observation | Count | Handled in |
|---|---|---|---|
| O1 | **Two frequency sets on one line**: 2 × (3N−6) values whose halves differ slightly (e.g. 6620). Which set matches the reported ZPVE and thermochemistry is not documented. | 433 records | Phase 2 ✔ both sets kept (`set_idx`); Phase 4 (ZPVE invariant decides which set is consistent, ADR required) |
| O2 | **Imaginary (negative) frequencies**: 6620, 59818, 87037, 117523, 129158. All five are on the authors' "difficult to converge" list; 6620 and 59818 are stated saddle points. | 5 records | Phase 4 Layer 3 "real minimum" check, hard flag |
| O3 | Authors' readme lists 11 hard-to-converge molecules, 6 converged only at a low threshold. | 11 records | Phase 4 soft flag; stored in `configs/dataset.yaml` |
| O4 | Fortran `*^` exponents, only in coordinates. | 176 records | Phase 2 ✔ converted and logged in `parse_log` |
| O5 | Linear molecules with 3N−5 frequencies and A = 0. | 5 records | Phase 2 (A = 0 means undefined, not zero); Phase 4 rotational-constant check |
| O6 | Header tag is `gdb`, not `gdb9` as documented. | all | Phase 2 ✔ parser expects `gdb` |
| O7 | `uncharacterized.txt` compares B3LYP vs **Corina** geometries via Open Babel, not GDB-17 SMILES vs relaxed SMILES. | 3,054 | Phase 4 Layer 2: our three-way check is independent of theirs; compare outcomes |
| O8 | Size distribution is very skewed: 111,594 of 133,885 molecules (83%) have 9 heavy atoms; only 48 have 1–4. | — | Phase 8 size-extrapolation split; Phase 4 stratified statistics |

Structure-level checks that came back clean: every record has
nₐ + 5 lines, 15 properties, a header index that matches its file name,
and only H/C/N/O/F atoms.
