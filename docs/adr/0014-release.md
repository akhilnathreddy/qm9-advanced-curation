# 0014 — Dataset release: contents, formats, licenses and reproducibility

- **Status:** Accepted
- **Date:** 2026-11-10
- **Phase:** 10

## Context
Plan v2 (Phase 10) asks for:
- a versioned release with a datasheet and Croissant metadata;
- Parquet, extended-XYZ and PyTorch Geometric access;
- a `make reproduce` that rebuilds the release from the raw archives with
  identical hashes, checked by CI;
- a Zenodo DOI if the source license allows it.

Gate 10 needs two things:
- a clean machine reproduces the release with identical hashes;
- an outside reader loads the Gold tier and a named split in at most 5 lines.

Both sources are CC0-1.0: QM9 on figshare, and the Kim et al. 2019 G4(MP2)
data.

## Decision — contents (`configs/release.yaml`)
- **Version 1.0.0**, semantic:
  - MAJOR: the record set or a tier definition changes;
  - MINOR: tables or columns are added;
  - PATCH: documentation or metadata only.
- **Tables:** seven Parquet tables, **copied byte for byte** from the
  pipeline, so each release hash equals the pipeline's:
  - `molecules` (curated, every record, tiered);
  - `atoms`;
  - `frequencies`;
  - `g4mp2_xref`;
  - `review`;
  - `splits`;
  - `split_groups`.
- **Not released:**
  - **Feature tables:** they are derived, and the pipeline rebuilds them.
    RDKit descriptor floats are also the most likely to differ across
    platforms.
  - **Benchmark runs:** they are results, not data. `docs/benchmark.md` and
    the repository hold them.
- **Raw archives:** not redistributed. They are on figshare, MD5-verified,
  and `make reproduce` downloads them.

## Decision — formats
- **Parquet is canonical.** The 5-line quick start in the release README is
  plain Polars, so a reader needs none of this project's code.
- **Extended XYZ** (`qm9_curated.extxyz`):
  - one frame per record; element, position and Mulliken charge per atom;
  - `qm9_id`, `quality_tier`, `formula` and 16 properties per frame.
  - Floats are written with Python's shortest round-trip `repr`, so ASE reads
    back the exact float64 values.
  - **Not compressed:** gzip output depends on the zlib build (system zlib on
    Linux, other builds elsewhere), which would break cross-platform hash
    identity. The Zenodo zip compresses it for transport.
- **Croissant 1.0** (`croissant.json`):
  - one FileObject (with SHA-256) and one RecordSet per table, one field per
    column.
  - Field descriptions are parsed from `docs/data_dictionary.md`, so metadata
    and documentation cannot drift. The build fails its gate condition if any
    release column is undocumented.
  - Generated without mlcroissant; the gate validates it with mlcroissant.
- **PyTorch Geometric:** `qm9_curation.export.pyg.QM9Curated` is in the
  package, behind an optional `pyg` extra, so the core install stays light.
  - It reads the release Parquet in place, and caches per (tiers, targets).
  - `split_indices(split, seed)` maps release splits to dataset positions.
  - Graphs carry `z`, `pos` and `y`, with no bonds (models build their own
    neighbours).
- **Also in the release:** `datasheet.md` (Gebru et al.), `README.md`,
  `CITATION.cff`, a CC0 `LICENSE`, and `MANIFEST.sha256` in `sha256sum`
  format, so a download can be checked with `sha256sum -c`.

## Decision — licenses and credit
- **Code:** MIT.
- **Dataset:** CC0-1.0, matching both sources, so it mixes freely into
  other datasets. Citation is requested, not required.
- **Author:** credited by GitHub handle (`akhilnathreddy`), at the user's
  choice.

## Decision — reproducibility
- **Every output is deterministic.**
  - Tables are copied unchanged.
  - The extended XYZ, Croissant JSON and README are generated from sorted
    inputs.
  - The publication date is fixed in the config, not the build date.
  - A forced rebuild is byte-identical (Gate 10).
- **The published manifest is `release/<name>.sha256`, committed to git.**
  - `make reproduce` runs `uv sync --locked`, rebuilds the release stage and
    its ancestors (downloading the raw archives on a clean checkout), then
    runs `qm9_curation.export.verify`, which checks:
    - every file against the published manifest;
    - the release's own manifest;
    - that ASE reads every frame with exact coordinates;
    - that the Croissant metadata validates;
    - that the README quick start runs in at most 5 lines.
- **Reference platform:** per ADR 0001, Linux CI is the reference build. The
  `Reproduce release` workflow runs `make reproduce` on `ubuntu-latest`:
  - on demand, on `v*` tags, and monthly;
  - it uploads the build as an artifact.

  If macOS and Linux ever differ, the published manifest is the Linux one,
  and `verify --compare-with <artifact>` locates the differing columns.
- **Zenodo:** `make zenodo` writes three files to `data/zenodo/`:
  - a deterministic zip of the release;
  - Zenodo deposition metadata;
  - upload steps.

  Publishing mints a permanent DOI under the user's account, so the upload is
  done by the user. Nothing is sent automatically.

## Alternatives considered
- **Gzip or xz for the extended XYZ:** rejected; see above.
- **Shipping the feature tables:** would add about 65 MB, plus the risk of
  differences across platforms. They can be added later as a MINOR release
  if users ask.
- **torch_geometric as a core dependency:** heavy for readers who only want
  Parquet. An optional extra keeps both use cases cheap.
- **CC BY 4.0 for the dataset:** it would require attribution, but would add
  friction for people merging datasets. The sources chose CC0.
- **A custom loader API instead of plain Polars for the quick start:** it
  would force readers to install this package just to read files.

## Consequences
- Any change to a released table, the datasheet or the data dictionary
  changes the release hashes. That needs a new version and a new published
  manifest.
- The DOI is minted only after the user's upload. Recording it in the config
  and docs is a PATCH release (1.0.1), because the citation text inside the
  release changes.
- The monthly reproduce run detects upstream changes (figshare files are
  MD5-checked) and breakage from tool updates, but the locked environment
  pins every package.
