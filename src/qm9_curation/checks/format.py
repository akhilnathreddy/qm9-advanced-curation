"""Layer 1 — format and completeness, re-checked on the typed tables.

The strict parser already enforces most of this; repeating it here makes
every record's flag row complete and guards against later table edits.
"""

import polars as pl

from qm9_curation.schema import ELEMENTS


def run_format(
    mol: pl.DataFrame, atoms: pl.DataFrame, freq: pl.DataFrame, max_heavy: int
) -> pl.DataFrame:
    per_atoms = atoms.group_by("qm9_id").agg(
        pl.col("element").is_in(ELEMENTS).all().alias("elements_allowed__pass"),
        pl.all_horizontal(pl.col("x_ang", "y_ang", "z_ang").is_finite())
        .all()
        .alias("coordinates_finite__pass"),
    )
    per_set = (
        freq.group_by("qm9_id", "set_idx").len().join(mol.select("qm9_id", "n_atoms"), on="qm9_id")
    )
    n3 = pl.col("n_atoms").cast(pl.Int64) * 3
    counts = per_set.group_by("qm9_id").agg(
        ((pl.col("len") == n3 - 6) | (pl.col("len") == n3 - 5)).all().alias("frequency_count__pass")
    )
    return (
        mol.select("qm9_id", (pl.col("n_heavy") <= max_heavy).alias("heavy_atom_limit__pass"))
        .join(per_atoms, on="qm9_id", how="left")
        .join(counts, on="qm9_id", how="left")
        .sort("qm9_id")
    )
