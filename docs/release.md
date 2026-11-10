# Phase 10 — dataset release

- **Release:** `data/release/qm9-curated-v1.0.0/`, built by `make release`
  (DVC stage `release`).
- **Published manifest:** `release/qm9-curated-v1.0.0.sha256`.
- **Reports:** `reports/release/release_report.json` (build) and
  `verify_report.json` (verification).
- **Decisions:** ADR 0014. **Datasheet:** `docs/datasheet.md`.

## Contents (v1.0.0)

| File | Rows | Size | Content |
|---|---|---|---|
| `molecules.parquet` | 133,885 | 29.5 MB | Curated table: tier, reasons, identifiers, properties (converted and source), checks, lineage |
| `atoms.parquet` | 2,407,753 | 66.7 MB | Coordinates (Å), Mulliken charges |
| `frequencies.parquet` | 6,438,713 | 27.6 MB | Harmonic frequencies (cm⁻¹) |
| `g4mp2_xref.parquet` | 133,885 | 6.6 MB | G4(MP2) cross-reference (Kim et al. 2019) |
| `review.parquet` | 133,885 | 11.2 MB | Soft review scores and reasons |
| `splits.parquet` | 5,355,400 | 14.4 MB | 8 splits × 5 seeds |
| `split_groups.parquet` | 133,885 | 5.3 MB | Leakage groups, split units |
| `qm9_curated.extxyz` | 133,885 frames | 201 MB | Geometries for ASE, with properties per frame |
| `croissant.json` | — | — | Croissant 1.0 metadata: 7 record sets, every column described |
| `datasheet.md`, `README.md`, `CITATION.cff`, `LICENSE` | — | — | Documentation; CC0-1.0 |
| `MANIFEST.sha256` | — | — | SHA-256 of every file (`sha256sum -c MANIFEST.sha256`) |

Tiers: Gold 130,659; Silver 1,630; Bronze 1,591; Quarantine 5.

## Loading

The Gold tier and a named split, in Polars:

```python
import polars as pl
mol = pl.read_parquet("qm9-curated-v1.0.0/molecules.parquet").filter(pl.col("quality_tier") == "gold")
split = pl.read_parquet("qm9-curated-v1.0.0/splits.parquet").filter(pl.col("split_name") == "scaffold_generic", pl.col("seed") == 0)
train = mol.join(split.filter(pl.col("subset") == "train"), on="qm9_id")
test = mol.join(split.filter(pl.col("subset") == "test"), on="qm9_id")
```

That gives 130,659 Gold molecules: 104,098 train and 13,253 test.

```python
from ase.io import read
frames = read("qm9-curated-v1.0.0/qm9_curated.extxyz", index=":")  # atoms.info["gap_ev"], ...

from qm9_curation.export.pyg import QM9Curated  # uv sync --extra ml --extra pyg
ds = QM9Curated("pyg_cache", "qm9-curated-v1.0.0", tiers=("gold",))
train = ds[ds.split_indices("scaffold_generic", seed=0)["train"]]
```

## Verification

`python -m qm9_curation.export.verify` (part of `make reproduce` and
`make gate10`) checks the build:
- every file's SHA-256 against the published manifest;
- the release's own manifest;
- that ASE reads all 133,885 frames with coordinates and atomic numbers
  equal to `atoms.parquet`;
- that the Croissant file validates with mlcroissant (no errors or
  warnings);
- that the README quick start runs, in at most 5 lines.

**Reproduction from the raw archives.**
- `make reproduce` on a clean checkout downloads the QM9 and Kim et al.
  files, checks their published MD5s, and runs every stage up to `release`.
- **Tested on 2026-11-10:** a clean copy of the repository on the M2 Pro
  (no data, no DVC cache, a new environment) ran `make reproduce` in
  11 min 20 s. Every file matched the published manifest, and every check
  passed.
- The Linux reference run is the `Reproduce release` workflow in GitHub
  Actions.

## Publishing on Zenodo

`make zenodo` writes three files to `data/zenodo/`:
- the release zipped deterministically;
- `zenodo_metadata.json` (title, creator, CC0, keywords, and related
  identifiers to QM9, Kim et al. and this repository);
- `UPLOAD.md`, with the steps.

The upload is done by the user, from their own account: publishing mints a
permanent DOI. DOI: **pending**.
