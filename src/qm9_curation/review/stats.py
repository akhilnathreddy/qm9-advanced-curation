"""Shared statistics: robust z-scores, per-element linear fits, two-sample tests."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from qm9_curation.profiling.properties import MAD_TO_SIGMA, stratum

ELEMENT_COUNTS = ("n_H", "n_C", "n_N", "n_O", "n_F")
QUANTILES = (
    ("min", 0.0), ("p01", 0.01), ("p05", 0.05), ("p25", 0.25), ("median", 0.5),
    ("p75", 0.75), ("p95", 0.95), ("p99", 0.99), ("max", 1.0),
)  # fmt: skip


def robust_z(values: pl.Expr, over: str | None = None) -> pl.Expr:
    """(x - median) / (1.4826 * MAD), optionally within groups; null where MAD is 0."""
    med = values.median() if over is None else values.median().over(over)
    mad = (values - med).abs().median()
    mad = (mad if over is None else mad.over(over)) * MAD_TO_SIGMA
    return pl.when(mad > 0).then((values - med) / mad).otherwise(None)


def robust_z_by_size(frame: pl.DataFrame, column: str) -> pl.Series:
    """Robust z within heavy-atom strata (1-4 heavy atoms pooled, as in Phase 3)."""
    return (
        frame.with_columns(stratum())
        .select(robust_z(pl.col(column), over="stratum").alias(f"z_{column}"))
        .to_series()
    )


def fit_per_element(
    fit_on: pl.DataFrame, y: str, intercept: bool = False
) -> tuple[dict[str, float], pl.Expr]:
    """Least-squares y ~ sum_e c_e * n_e; returns coefficients and a prediction expr."""
    cols = list(ELEMENT_COUNTS)
    x = fit_on.select(cols).to_numpy().astype(np.float64)
    if intercept:
        x = np.column_stack([x, np.ones(len(x))])
        cols.append("intercept")
    coef, *_ = np.linalg.lstsq(x, fit_on[y].to_numpy(), rcond=None)
    coefs = {c: float(v) for c, v in zip(cols, coef, strict=True)}
    pred = pl.lit(coefs.get("intercept", 0.0))
    for c in ELEMENT_COUNTS:
        pred = pred + pl.col(c).cast(pl.Float64) * coefs[c]
    return coefs, pred


def summary(values: np.ndarray) -> dict[str, Any]:
    v = values[np.isfinite(values)]
    q = np.quantile(v, [p for _, p in QUANTILES])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "mae": float(np.abs(v).mean()),
        "rmse": float(np.sqrt((v**2).mean())),
        "quantiles": {name: float(x) for (name, _), x in zip(QUANTILES, q, strict=True)},
    }


def compare_distributions(reference: np.ndarray, other: np.ndarray) -> dict[str, float | int]:
    """KS test and Wasserstein-1 distance; W1 also in units of the reference SD."""
    a, b = reference[np.isfinite(reference)], other[np.isfinite(other)]
    ks = stats.ks_2samp(a, b, method="asymp")
    w1 = float(stats.wasserstein_distance(a, b))
    return {
        "n_reference": int(a.size),
        "n_other": int(b.size),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": w1,
        "wasserstein_over_reference_sd": w1 / float(a.std()) if a.std() > 0 else float("nan"),
    }


def compare_groups(
    frame: pl.DataFrame,
    targets: Sequence[str],
    group_col: str,
    reference: str,
    groups: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Every target, every group vs the reference group (plus all records vs reference)."""
    ref = frame.filter(pl.col(group_col) == reference)
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for target in targets:
        r = ref[target].drop_nulls().to_numpy()
        out[target] = {"all": compare_distributions(r, frame[target].drop_nulls().to_numpy())}
        for g in groups:
            sub = frame.filter(pl.col(group_col) == g)[target].drop_nulls().to_numpy()
            out[target][g] = compare_distributions(r, sub)
    return out


def mann_whitney_greater(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    """One-sided test that values in ``a`` tend to exceed those in ``b``."""
    res = stats.mannwhitneyu(a, b, alternative="greater", method="asymptotic")
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "u_statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        # Probability that a random a exceeds a random b (common-language effect size).
        "prob_a_greater": float(res.statistic) / (a.size * b.size),
    }
