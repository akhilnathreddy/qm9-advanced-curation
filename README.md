# QM9 Advanced Curation

A research-grade, reproducible curation pipeline for the QM9 quantum-chemistry
dataset (Ramakrishnan et al., *Sci. Data* 2014,
[doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)).

The pipeline preserves the raw release, validates every record against
format, chemical-identity and physics checks, assigns quality tiers instead of
deleting records, and produces ML-ready tables, a split suite and baseline
benchmarks. The full plan is in `docs/plan_v2.md`.

**Current phase:** 7 complete — statistical, scientific and multi-fidelity review against G4(MP2) in `data/processed/review.parquet` (Gate 7; `docs/statistical_validation.md`). Next: Phase 8, split suite.

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

Code license: **not yet chosen.** The QM9 source data is **CC0-1.0**
(public domain), as recorded in `data/raw/MANIFEST.json`. Please cite
Ramakrishnan et al. (2014) when using it. The G4(MP2) cross-reference
(`data/processed/g4mp2_xref.parquet`) is derived from Kim, Park and Choi (2019,
*Sci. Data* 6, 109; CC0); cite them when using it.
