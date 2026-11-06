# Phase 3 — profiling findings

Generated outputs: `reports/data_quality/profile.html` (read this first),
`reports/data_quality/profile.json`, `data/interim/molecule_profile.parquet`,
`data/interim/geometric_duplicate_pairs.parquet`. Nothing in the data was
changed or removed in this phase.

## Headline numbers

| | |
|---|---|
| Molecules | 133,885 |
| Distinct formulas | 621 (largest: C7H10O2, 6,095 isomers; 95 singletons) |
| Acyclic | 14,117 (10.5%) — all share the empty Murcko scaffold |
| Murcko / generic scaffolds | 22,804 / 1,024 |
| Exact-structure duplicate groups | 222 groups, 453 molecules |
| Tautomer groups (shared standard InChI) | 3,652 groups, 8,097 molecules |
| Relaxed SMILES RDKit cannot parse | 746 (GDB-17 SMILES: 0) |
| Relaxed structures that fell apart | 229 |
| Shortest interatomic distance anywhere | 0.959 Å (no clashes) |

## Issues and triage

Triage rules live in `src/qm9_curation/profiling/issues.py`; the HTML report
lists examples for each.

| ID | Issue | Count | Triage | Owner |
|---|---|---|---|---|
| P01 | Exact structure duplicates | 453 | needs investigation | Phase 5, 8 |
| P02 | Tautomers share standard InChI | 8,097 | expected chemistry | Phase 8 |
| P03 | Relaxed SMILES unparseable | 746 | needs investigation | Phase 4 |
| P04 | Relaxed SMILES with radicals | 12,075 | needs investigation | Phase 4 |
| P05 | Relaxed structure fragmented | 229 | needs investigation | Phase 4 |
| P06 | GDB vs relaxed SMILES differ (not in uncharacterized list) | 10,677 | needs investigation | Phase 4 |
| P07 | Charge-separated GDB-17 SMILES | 1,705 | expected chemistry | Phase 5 |
| P08 | Doubled frequency sets | 433 | known source issue | Phase 4 |
| P09 | Imaginary frequencies | 5 | known source issue | Phase 4 |
| P10 | Linear molecules, A = 0 | 5 | expected chemistry | Phase 5 |
| P14 | Linearity represented inconsistently | 6 | needs investigation | Phase 4, 5 |
| P11 | \|robust z\| > 6 within heavy-atom strata | 2,010 | needs investigation | Phase 7 |
| P12 | 83% of molecules have 9 heavy atoms | 111,594 | expected chemistry | Phase 7, 8 |
| P13 | Acyclic → empty scaffold | 14,117 | expected chemistry | Phase 8 |

## Notes on the main findings

**Relaxed SMILES are unreliable.** They come from Open Babel bond perception
on the B3LYP geometry. Many encode zwitterions as neutral radicals
(`[NH3]CC(=O)[O]`), 12,075 carry radical centres although every QM9 molecule
is a closed-shell singlet, and 746 cannot be parsed. String-level
disagreement with GDB-17 (P06) therefore over-counts real connectivity
changes. Phase 4 re-derives bonds from the coordinates independently.

**Fragmentation (P05).** 229 relaxed SMILES contain `.`, e.g.
`[NH3+]CC1=CC=C([O-])O1` relaxing to `N.C=C1C=CC(=O)O1` (ammonia leaves). These
are no longer the molecule in the label.

**Linearity (P14).** 9 molecules have A = 0, but only 5 of them list 3N−5
frequencies. Four (cyanogen 25, 485, 14563, 14564) are linear with 3N−6
frequencies, and the polyynes C6H2 (484) and C8H2 (14562) report A of
619,868 and 232,664 GHz instead of infinity. `is_linear` in the interim table
means "frequency count is 3N−5" and misses these six.

## Correction to Phase 2

Profiling exposed a bug in `hill_formula`: an `H` was always written after
`C`, so the 49 carbon-containing molecules with no hydrogen got a wrong
formula string (e.g. cyanogen `C2HN2` instead of `C2N2`). Formula groups were
unaffected (621 either way). Fixed in Phase 3 with regression tests; the
interim tables were rebuilt.
