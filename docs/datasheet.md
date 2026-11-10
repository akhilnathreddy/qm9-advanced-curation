# Datasheet — QM9 Curated v1.0.0

This datasheet follows Gebru et al., *Datasheets for Datasets* (Commun. ACM
64(12), 2021). The release ships it as `datasheet.md` beside the data. A
machine-readable Croissant 1.0 description (`croissant.json`) ships too.
Repository: https://github.com/akhilnathreddy/qm9-advanced-curation.

## Motivation

- **Purpose.** QM9 (Ramakrishnan et al., 2014) is the most-used benchmark
  for molecular property prediction. It is usually consumed through
  redistributions that filter records silently, convert units without
  saying so, and split at random. This release keeps every record. Each
  record is checked independently and tiered with a machine-readable reason
  instead of being deleted. The release also adds leakage-audited splits, so
  model results can be compared and trusted.
- **Creator.** Created by akhilnathreddy as an independent research project.
  No external funding.

## Composition

- **Instances:** 133,885 small organic molecules — every record of the
  source release. Each has at most 9 heavy atoms, from C, N, O and F, plus
  hydrogen. Each record holds one geometry and its computed properties.
  None are removed.
- **Tiers.** Assigned by rules applied in order (ADR 0009):

| Tier | Records | Meaning |
|---|---|---|
| gold | 130,659 | Passes every hard and identity check. Not a duplicate copy. Not on the authors' problem lists |
| silver | 1,630 | Usable with a documented caveat: 1,466 on the authors' list but consistent by independent check, 159 duplicate copies, 5 convergence concerns |
| bronze | 1,591 | The geometry is not the labelled molecule: the structure changed or fragmented |
| quarantine | 5 | Not a true minimum (imaginary frequency) |

- **Tables** (Parquet; the join key is `qm9_id`):

| Table | Rows | Content |
|---|---|---|
| `molecules` | 133,885 | Tier and reasons; identifiers (GDB-17 and geometry-perceived SMILES and InChIKey); 15 properties in eV/Å/Debye and **unchanged source values**; atomization energies; check results; lineage hashes |
| `atoms` | 2,407,753 | Element, Cartesian coordinates (Å), Mulliken charge |
| `frequencies` | 6,438,713 | Harmonic frequencies (cm⁻¹); 433 records have two sets in the source |
| `g4mp2_xref` | 133,885 | G4(MP2) energies of Kim et al. (2019) and the B3LYP − G4(MP2) difference |
| `review` | 133,885 | Soft statistical review scores and reasons; these never change a tier |
| `splits` | 5,355,400 | 8 splits × 5 seeds: train / val / test per record |
| `split_groups` | 133,885 | Leakage groups and split units |

  `qm9_curated.extxyz` has the same geometries in extended XYZ, one frame
  per record, with the properties in each frame header.
- **Labels.** Every property is computed at B3LYP/6-31G(2df,p), the source
  method. Nothing is measured experimentally. U0 atomization energies use
  the release's atom references.
- **Missing values.**
  - `rot_a_ghz` is null for linear molecules.
  - The geometry SMILES and InChIKey are null for 22 records whose bonds RDKit
    could not perceive (`geometry_identifier_error`).
  - G4(MP2) values are null for the 27 records that did not converge.
  - G4(MP2) atomization energies are withheld for every molecule that
    contains nitrogen, because the published N atom reference is a doublet.
- **Relationships.**
  - Exact duplicates are grouped by `structure_group_id`, with one
    representative per group in Gold.
  - Tautomers, GDB connectivity and geometry InChIKey blocks are grouped in
    `split_groups`. Every split keeps each leakage group in a single subset.
- **Splits.** Five splits are for the main comparison: `random`,
  `isomer_group`, `scaffold_generic`, `cluster_ecfp` and
  `size_extrapolation`. Three are stress tests: `element_holdout` (no F in
  train), and the upper 5% tails of the gap and of U0 atomization. Each has
  seeds 0–4. Evaluate on `test ∩ gold`.
- **Errors and noise.**
  - B3LYP differs from G4(MP2) by a mean absolute 0.123 eV on Gold U0, after
    a per-element fit.
  - 23 records re-optimized more than 1 kcal/mol lower at G4(MP2).
  - 384 records carry soft review flags.
  - `docs/limitations.md` lists every known issue.
- **Confidential or personal data:** none. The dataset describes molecules,
  not people.

## Collection process

- **Source.** The QM9 figshare collection
  (doi:10.6084/m9.figshare.c.978904.v5): the `.xyz` archive,
  `uncharacterized.txt`, `atomref.txt` and `readme.txt`. Every file is
  checked against figshare's published MD5. The G4(MP2) data come from Kim et
  al.'s figshare collection (doi:10.6084/m9.figshare.c.4351631.v1), also
  MD5-checked.
- **How the source was made.** Ramakrishnan et al. enumerated molecules from
  GDB-17 and optimized them with B3LYP/6-31G(2df,p) in Gaussian 09. No new
  calculations were run for this release.
- **When.** The source is from 2014; the external data from 2019. Files were
  retrieved on 2026-11-06 and 2026-11-07, and the hashes are in the
  repository.

## Preprocessing, cleaning and labelling

The order of the steps, with the ADR that records each decision:
1. **Lossless parsing (ADR 0006).** Every record round-trips to its exact
   source bytes.
2. **Four layers of checks (ADR 0008).**
   - format;
   - chemical identity: 3D-perceived graph vs. GDB-17 SMILES vs. the authors'
     list;
   - physics invariants;
   - statistical outliers.
3. **Duplicate grouping (ADR 0007).**
4. **Standardization and tiering (ADR 0009).**
5. **Statistical and multi-fidelity review against G4(MP2) (ADR 0011).**
6. **Splits (ADR 0012).**

Source values are kept unchanged in `source_*` columns beside the converted
ones, and every conversion can be inverted. The raw data are not
redistributed, because they are available from figshare. The pipeline
rebuilds this release from them with identical file hashes
(`make reproduce`).

## Uses

- **Intended:**
  - training and evaluating property-prediction models, on Gold test molecules;
  - studying split-induced optimism (`isomer_group`, `scaffold_generic`,
    `size_extrapolation`);
  - label-quality and curation ablations (tiers);
  - multi-fidelity work with the G4(MP2) cross-reference.
- **Benchmark (Phase 9, `docs/benchmark.md`).**
  - On the `random` split, the best baselines reach a Gold-test MAE of
    0.050 eV for U0 atomization and 0.107 eV for the gap.
  - The scaffold and size splits raise U0 error 3–4×.
  - Silver and Bronze records have larger out-of-fold errors than Gold.
- **Not suitable for:**
  - claims about experimental accuracy: the labels are DFT values;
  - molecules outside C/H/N/O/F with at most 9 heavy atoms;
  - conformer-dependent properties: there is one geometry per molecule,
    not guaranteed to be the global minimum;
  - treating Silver or Bronze as random subsets: their chemistry differs from
    Gold's.
- **Risks:** the dataset carries no social or privacy risk. Its main risk is
  over-reading a low random-split MAE as general accuracy; use the grouped
  splits.

## Distribution

- **License:** CC0-1.0. The source data and the G4(MP2) data are both
  CC0-1.0. The code that builds the release is MIT-licensed. Citing this
  release and the original QM9 paper is requested.
- **Where:** a versioned archive with a DOI (Zenodo), plus the code
  repository. `MANIFEST.sha256` lists every file's SHA-256.
- **Format:** Parquet (canonical), extended XYZ, a PyTorch Geometric loader
  in the code package, and Croissant JSON-LD metadata.

## Maintenance

- **Maintainer:** akhilnathreddy, through issues on the GitHub repository.
- **Versioning:** semantic.
  - MAJOR: the record set or a tier definition changes.
  - MINOR: tables, columns or features are added.
  - PATCH: documentation or metadata fixes only.
  - Every version is rebuilt from the raw files and gets its own manifest.
- **Errata:** listed in the repository's `docs/limitations.md` and fixed in
  a new version. Old versions stay available.
- **Possible future additions:** a separate, labelled table of recovered
  geometries for the uncharacterized records (plan v2, Phase 11). They
  would never be merged into the original record set.
