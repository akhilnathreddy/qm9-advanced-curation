"""Composition features: atom counts per element (basis of the dressed-atom baseline)."""

import polars as pl

from qm9_curation.features.inputs import require_allowed

ELEMENTS = ("H", "C", "N", "O", "F")


def composition(atoms: pl.DataFrame) -> pl.DataFrame:
    require_allowed(atoms)
    counts = atoms.group_by("qm9_id").agg(
        *((pl.col("element") == e).sum().cast(pl.UInt8).alias(f"n_{e}") for e in ELEMENTS),
        pl.len().cast(pl.UInt8).alias("n_atoms"),
    )
    return counts.with_columns(
        (pl.col("n_atoms") - pl.col("n_H")).cast(pl.UInt8).alias("n_heavy")
    ).sort("qm9_id")
