# Phase 4 — validation findings

Outputs: `data/interim/validation.parquet` (one row per molecule; a pass flag
for each of 19 checks, residuals, `flag_bitmask`, failure counts per
severity) and `reports/validation/validation_report.json`. Methods and
tolerances: ADR 0008. Nothing was removed or changed.

## Results by check

| Layer | Check | Severity | Failed |
|---|---|---|---|
| 1 | elements allowed, ≤ 9 heavy atoms, frequency count, finite coordinates | hard | 0 each |
| 2 | single fragment | identity | 229 |
| 2 | GDB-17 graph = graph from 3D | identity | 1,593 |
| 2 | not on authors' uncharacterized list | identity | 3,054 |
| 2 | authors' relaxed SMILES = graph from 3D | soft | 1,549 |
| 3 | gap, H − U, G < H, U > U0, Mulliken sum, ZPVE, rotational constants | hard | 0 each |
| 3 | real minimum (no imaginary frequency) | hard | 5 |
| 3 | linearity represented consistently | soft | 6 |
| 4 | \|robust z\| ≤ 6 | soft | 1,451 |
| 4 | not on authors' difficult-to-converge list | soft | 11 |

128,867 records (96.3%) pass every check.

## RQ1 — independent identity check vs the authors' uncharacterized list

| Identity from 3D | On the list | Not on the list |
|---|---|---|
| consistent with GDB-17 | 1,466 | 130,826 |
| proton moved | 1,010 | 0 |
| fragmented | 229 | 0 |
| framework rearranged | 349 | 5 |

- **Agreement:** every proton-transfer and fragmentation case found
  independently is on the authors' list.
- **1,466 listed records look consistent.** The authors compared their
  *Corina starting geometry* with the B3LYP result. For these records the
  starting geometry was the odd one out (e.g. 58: Corina built
  `[NH]C(=[NH2])C#N`), while the B3LYP structure, by our perception and by the
  authors' own B3LYP SMILES column, is the intended GDB-17 molecule. Their
  D_IJ is also smaller (median 7.7 vs 23.7). These are candidates to keep;
  Phase 5 decides.
- **5 rearrangements not on the list:** 128228, 133831, 133850, 133857,
  133859 (mostly strained cages). For 133857 and 133859 the authors' relaxed
  SMILES agrees with our graph; the other three need review.

## Other results

- **Doubled frequency sets resolved.** For 432 of 433 records both sets
  reproduce the stored ZPVE (median difference 2e-6), so the choice is
  immaterial. Only **129152** (on the authors' low-threshold list) differs:
  set 0 is 6.7% off; set 1 is consistent and recorded as
  `zpve_consistent_set`.
- **Linearity.** 11 molecules are linear by geometry: 4, 5, 23, 24, 25, 484,
  485, 486, 14562, 14563, 14564. Six are represented inconsistently in the
  source (A = 0 with 3N−6 frequencies, or a huge finite A).
- **Imaginary frequencies.** 6620, 59818, 87037, 117523, 129158 — all on the
  authors' difficult-to-converge list.
