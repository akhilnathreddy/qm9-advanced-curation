# 0002 — Source of record for QM9

- **Status:** Accepted (exact URL and license to be confirmed in Phase 1)
- **Date:** 2026-11-06
- **Phase:** 0

## Context
QM9 is redistributed by many libraries (PyG, DeepChem, TorchDrug, Kaggle
mirrors), each with its own filtering, unit conversions and ID handling.
v1 of the plan listed `qm9.org`, which is not confirmed as an official source.

## Decision
The source of record is the **figshare collection linked from the 2014
*Scientific Data* paper** (doi:10.1038/sdata.2014.22): the `.xyz` archive,
the list of uncharacterized molecules, and the atom-reference energy file.
Redistributions are used only as independent cross-checks.

## Alternatives considered
- *PyG / DeepChem processed versions*: convenient but already transformed;
  their filtering would become invisible upstream decisions.
- *qm9.org*: provenance not established.

## Consequences
- Phase 1 must record the exact URL, file names, sizes, SHA-256 and the
  license string from the figshare record in `data/raw/MANIFEST.json`, and
  fill `configs/dataset.yaml`.
- Redistribution of the curated release depends on that license.
