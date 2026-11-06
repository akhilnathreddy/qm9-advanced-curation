# 0006 — Lossless parsing and the interim table layout

- **Status:** Accepted
- **Date:** 2026-11-06
- **Phase:** 2

## Context
Every later check and transformation reads the parsed tables, so the parse
must neither drop nor alter information. The source is an ad-hoc XYZ
extension with no formal spec, Fortran `*^` exponents, a sign-padding
convention, trailing tabs, and 433 records with two frequency sets.
Plan v2 asks for a byte-level round-trip on every record.

## Decision
**"Lossless" is proven at two levels, both on all 133,885 records:**

1. **Text level.** `parse_record` keeps every field as its source token
   and then re-serializes the record; any difference from the input on any
   line is a `ParseError` with record name and line number. A record that
   parses therefore round-trips to its exact source bytes.
2. **Value level.** For every numeric token (18,078,000 in total), the
   decimal value of the token equals the decimal value of the float64 that
   the tables store. The tables thus hold every source value exactly at
   source precision.

What the tables do **not** carry is formatting: `0.` vs `0.0`, `*^` vs
`e`, and sign padding. That formatting has no scientific meaning, and the
raw archive (DVC-tracked, read-only, SHA-256 in each row) keeps it.

**Layout:** four tables in `data/interim/` (Parquet, zstd level 9,
deterministic row order): `molecules`, `atoms`, `frequencies`,
`parse_log`. Columns carry their **source** unit in their name
(`u0_ha`, `x_ang`); no unit conversion happens in Phase 2.

**Doubled frequencies:** both sets are kept (`set_idx` 0 and 1).
Choosing one is deferred to Phase 4, where the ZPVE invariant can decide
which set is consistent with the reported thermochemistry.

**Linear molecules:** `rot_a_ghz` is stored as the source wrote it (0).
Phase 5 decides how to represent "undefined".

## Alternatives considered
- *Store source tokens as strings alongside floats:* doubles table size and
  is unnecessary once value-level exactness is proven.
- *Store the full source text per molecule:* duplicates the raw archive.
- *Parse with a lenient regex and skip odd lines:* exactly the silent loss
  this project exists to avoid.

## Consequences
- A future change to the parser that loses or alters any value fails Gate 2.
- Phase 4 owns the choice between the two frequency sets for 433 molecules;
  until then, frequency-derived quantities for them are ambiguous.
