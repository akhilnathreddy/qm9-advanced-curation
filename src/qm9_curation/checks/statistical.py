"""Layer 4 — statistical review flags and the authors' own annotations.

Both are soft: they route records to review and never by themselves mark a
record as wrong.
"""

from collections.abc import Iterable

import polars as pl


def run_statistical(
    profile: pl.DataFrame, robust_z_review: float, hard_to_converge: Iterable[int]
) -> pl.DataFrame:
    return profile.select(
        "qm9_id",
        pl.col("max_abs_robust_z").alias("not_statistical_outlier__residual"),
        (pl.col("max_abs_robust_z").fill_null(0) <= robust_z_review).alias(
            "not_statistical_outlier__pass"
        ),
        (~pl.col("qm9_id").is_in(sorted(hard_to_converge))).alias("not_hard_to_converge__pass"),
    ).sort("qm9_id")
