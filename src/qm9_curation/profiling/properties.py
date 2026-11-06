"""Distribution statistics for the 15 source properties (Phase 3).

Outlier scores are robust z-scores (median / MAD) computed **within
heavy-atom strata**, because most properties scale with molecule size; a
global z-score would mostly flag small molecules. Strata with one to four
heavy atoms are pooled (48 molecules in total). Scores only route records
to review; nothing is excluded here.
"""

from typing import Any

import polars as pl

from qm9_curation.schema import PROPERTY_COLUMNS

QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)
ROBUST_Z_REVIEW = 6.0
MAD_TO_SIGMA = 1.4826


def stratum() -> pl.Expr:
    return pl.when(pl.col("n_heavy") <= 4).then(4).otherwise(pl.col("n_heavy")).alias("stratum")


def robust_z(frame: pl.DataFrame) -> pl.DataFrame:
    """One robust z-score column per property, computed within heavy-atom strata."""
    exprs = []
    for col in PROPERTY_COLUMNS:
        med = pl.col(col).median().over("stratum")
        mad = (pl.col(col) - med).abs().median().over("stratum") * MAD_TO_SIGMA
        exprs.append(
            pl.when(mad > 0).then((pl.col(col) - med) / mad).otherwise(None).alias(f"z_{col}")
        )
    return (
        frame.with_columns(stratum())
        .with_columns(exprs)
        .select("qm9_id", *(f"z_{c}" for c in PROPERTY_COLUMNS))
    )


def property_stats(molecules: pl.DataFrame) -> dict[str, Any]:
    z = robust_z(molecules)
    stats: dict[str, Any] = {}
    for col in PROPERTY_COLUMNS:
        s = molecules[col]
        zc = z.select("qm9_id", pl.col(f"z_{col}").alias("z")).drop_nulls()
        top = (
            zc.with_columns(pl.col("z").abs().alias("az"))
            .sort(["az", "qm9_id"], descending=[True, False])
            .head(5)
        )
        stats[col] = {
            "count": s.len(),
            "null": s.null_count(),
            "nan": int(s.is_nan().sum()),
            "zero": int((s == 0).sum()),
            "negative": int((s < 0).sum()),
            "mean": float(s.mean()),  # type: ignore[arg-type]
            "std": float(s.std()),  # type: ignore[arg-type]
            "quantiles": {str(q): float(s.quantile(q, "nearest")) for q in QUANTILES},  # type: ignore[arg-type]
            "n_robust_z_above_review": int((zc["z"].abs() > ROBUST_Z_REVIEW).sum()),
            "most_extreme": [
                {"qm9_id": int(i), "robust_z": round(float(v), 2)}
                for i, v in top.select("qm9_id", "z").rows()
            ],
        }
    return stats


def pearson_matrix(molecules: pl.DataFrame) -> dict[str, dict[str, float]]:
    cols = list(PROPERTY_COLUMNS)
    corr = molecules.select(cols).corr()
    return {a: {b: round(float(corr[b][i]), 4) for b in cols} for i, a in enumerate(cols)}
