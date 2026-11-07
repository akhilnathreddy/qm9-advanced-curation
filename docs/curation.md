# Phase 5 — curated dataset

`data/processed/molecules.parquet`: one row for each of the 133,885 source
records (none removed). Report: `reports/curation/curation_report.json`.
Decisions: ADR 0009.

## Tiers

| Tier | Records | Meaning |
|---|---|---|
| gold | 130,659 (97.6%) | Passes every hard and identity check; not a duplicate; not on the authors' problem lists |
| silver | 1,630 | Usable, with a documented caveat (listed-but-consistent, duplicate copy, convergence concern) |
| bronze | 1,591 | Geometry is not the labelled molecule (changed or fragmented) |
| quarantine | 5 | Not a true minimum (imaginary frequency) |

## Waterfall (rules applied in order)

| After rule | Matched | Newly out of Gold | Gold remaining |
|---|---|---|---|
| — | — | — | 133,885 |
| R1 hard check failed | 5 | 5 | 133,880 |
| R2 fragmented | 229 | 227 | 133,653 |
| R3 structure changed | 1,364 | 1,364 | 132,289 |
| R4 listed uncharacterized, consistent | 1,466 | 1,466 | 130,823 |
| R5 duplicate, not representative | 231 | 159 | 130,664 |
| R6 convergence concern | 11 | 5 | 130,659 |

## Selecting a subset

```python
import polars as pl

mol = pl.read_parquet("data/processed/molecules.parquet")
gold = mol.filter(pl.col("quality_tier") == "gold")
# The conventional "QM9 minus uncharacterized" subset (130,831 records):
conventional = mol.filter(~pl.col("uncharacterized_listed"))
```
