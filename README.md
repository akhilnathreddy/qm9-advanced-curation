# QM9 Advanced Curation

A research-grade, reproducible curation pipeline for the QM9 quantum-chemistry
dataset (Ramakrishnan et al., *Sci. Data* 2014,
[doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22)).

The pipeline preserves the raw release, validates every record against
format, chemical-identity and physics checks, assigns quality tiers instead of
deleting records, and produces ML-ready tables, a split suite and baseline
benchmarks. The full plan is in `docs/plan_v2.md`.

**Current phase:** 0 — setup. No data has been downloaded yet.

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
make gate0    # Phase 0 exit gate: sync, check and repro in one go
```

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

Code license: **not yet chosen.** Dataset license: inherited from the QM9
source and recorded in Phase 1.
