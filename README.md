# QM9 Advanced Curation

A research-grade, reproducible curation pipeline for the QM9 quantum-chemistry
dataset (Ramakrishnan et al., *Sci. Data* 2014,
[doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)).

The pipeline preserves the raw release, validates every record against
format, chemical-identity and physics checks, assigns quality tiers instead of
deleting records, and produces ML-ready tables, a split suite and baseline
benchmarks. The full plan is in `docs/plan_v2.md`.

**Current phase:** 10 — release v1.0.0 built and verified (Gate 10; `docs/release.md`, ADR 0014): all 133,885 records, tiered, with splits, as Parquet, extended XYZ and Croissant metadata. Rebuilt from the raw archives on clean machines: 12 of 14 files byte-identical, the two least-squares tables equal within 10⁻⁹. Published on Zenodo: [doi:10.5281/zenodo.24105666](https://doi.org/10.5281/zenodo.24105666).

**Project report:** `docs/report.md` covers every phase, decision, result and limitation, plus reproduction steps.

## Using the release

**QM9 Curated v1.0.0** contains all 133,885 records, each tiered with a
reason, plus geometries, frequencies, a G4(MP2) cross-reference and eight
leakage-audited splits. It ships as Parquet, extended XYZ and Croissant
metadata (`docs/release.md`, `docs/datasheet.md`).

Build it with `make release` (output in `data/release/`), or download the
archive from Zenodo, [doi:10.5281/zenodo.24105666](https://doi.org/10.5281/zenodo.24105666) (all versions: doi:10.5281/zenodo.24105665).
The same files are mirrored on Kaggle as
[akhilnathreddy/qm9-curated](https://www.kaggle.com/datasets/akhilnathreddy/qm9-curated),
byte-identical to the Zenodo archive. In a Kaggle notebook they are at
`/kaggle/input/qm9-curated/`.
Then load the Gold tier and a named split:

```python
import polars as pl

mol = pl.read_parquet("qm9-curated-v1.0.0/molecules.parquet").filter(
    pl.col("quality_tier") == "gold"
)
split = pl.read_parquet("qm9-curated-v1.0.0/splits.parquet").filter(
    pl.col("split_name") == "scaffold_generic", pl.col("seed") == 0
)
train = mol.join(split.filter(pl.col("subset") == "train"), on="qm9_id")
```

For PyTorch Geometric: `uv sync --extra ml --extra pyg`, then
`qm9_curation.export.pyg.QM9Curated`. To rebuild the release from the raw
archives and check every file hash: `make reproduce`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Git. uv installs Python 3.12
and every dependency from `uv.lock`.

```bash
uv sync
uv run pre-commit install
```

## Everyday commands

```bash
make check    # lint, format check, type check, tests
make repro    # run the DVC pipeline
make gate0    # Phase 0 exit gate
make gate1    # Phase 1 exit gate: verify raw files, re-download check, inventory
make gate2    # Phase 2 exit gate: lossless parse, schemas, byte-identical rebuild
make gate3    # Phase 3 exit gate: profile, triage, byte-identical rebuild
make gate4    # Phase 4 exit gate: all checks, tolerances set, byte-identical rebuild
make gate5    # Phase 5 exit gate: tiers, source values preserved, byte-identical rebuild
make gate6    # Phase 6 exit gate: features documented, leakage-guarded, byte-identical rebuild
make gate7    # Phase 7 exit gate: G4(MP2) cross-check, soft review flags, byte-identical rebuild
make gate8    # Phase 8 exit gate: leakage assertions per split and seed, plan frozen, byte-identical rebuild
make gate9    # Phase 9 exit gate: frozen plan intact, every planned cell and seed run, byte-identical analysis
make gate10   # Phase 10 exit gate: release complete and documented, matches the published manifest
make release  # build the dataset release in data/release/
make reproduce  # rebuild the release from the raw archives and verify every hash
make fixture  # rebuild the 1k test fixture from the raw archive
```

`make repro` on a fresh clone downloads the raw files from figshare (about
86 MB) and checks them against the published checksums.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Dataset facts, validation tolerances, tier rules, split settings |
| `data/` | `raw/` (immutable, DVC-tracked), `interim/`, `processed/`, `splits/`, `external/` |
| `src/qm9_curation/` | All code that produces released artifacts |
| `tests/` | `unit/`, `property/` (Hypothesis), `pipeline/`, `fixtures/qm9_1k/` |
| `reports/` | Generated reports; `gates/` holds each phase's exit-gate result |
| `docs/` | Plan, ADRs, analysis plan, data dictionary, datasheet, limitations |
| `notebooks/` | Exploration only; never the source of a released artifact |

## Rules

1. Raw files are never modified.
2. No record is deleted silently; every exclusion has a machine-readable reason.
3. Every unit conversion comes from `src/qm9_curation/units.py`.
4. Every non-trivial decision is an ADR in `docs/adr/`.
5. Each phase closes with a gate recorded in `reports/gates/`.

## License

- **Code:** MIT (`LICENSE`).
- **Released dataset:** CC0-1.0, like both of its sources.
- **QM9 source data:** CC0-1.0 (public domain), as recorded in
  `data/raw/MANIFEST.json`.

Please cite Ramakrishnan et al. (2014) and this release (`CITATION.cff`). The G4(MP2) cross-reference
(`data/processed/g4mp2_xref.parquet`) is derived from Kim, Park and Choi (2019,
*Sci. Data* 6, 109; CC0); cite them when using it.
