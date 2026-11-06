# 0004 — Validation framework

- **Status:** Accepted
- **Date:** 2026-11-06
- **Phase:** 0

## Context
Validation is the core of the project: table schemas plus about 25
chemistry and physics checks that must each be unit-tested and must store
residuals, not just pass/fail.

## Decision
- **Pandera** for table schemas (dtypes, ranges, nullability, uniqueness),
  usable with both Polars and pandas.
- Domain checks are **plain Python functions** in `qm9_curation.checks`,
  each returning `(passed, residual, detail)`, with tolerances in
  `configs/validation.yaml`.

## Alternatives considered
- *Great Expectations*: heavier configuration layer, weaker fit for per-record
  residuals and chemistry-specific logic, harder to unit-test.

## Consequences
- Check results feed the `flags` table directly.
- Tolerances need an ADR each when they are set in Phase 4.
