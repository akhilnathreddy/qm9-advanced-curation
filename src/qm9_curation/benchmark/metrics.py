"""Error metrics and the bootstrap procedures of the frozen analysis plan.

* :func:`unpaired_mae_ci` — RQ3: test sets differ between splits, so each
  split is resampled on its own. Resampling is stratified by seed (molecules
  are drawn within each seed's test set) and the MAE is pooled over seeds.
* :func:`paired_difference_ci` — RQ5: the same Gold test molecules for every
  training set of a seed; the per-molecule |error| difference is resampled
  within each seed and averaged over seeds.
* :func:`holm` — Holm step-down adjustment within one RQ.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

KCAL_PER_MOL_EV = 0.0433641  # 1 kcal/mol in eV (chemical accuracy)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": len(y),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": 1.0 - float((err**2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
        "within_chemical_accuracy": float(np.mean(np.abs(err) <= KCAL_PER_MOL_EV)),
    }


def _resampled_means(
    values: np.ndarray, n_boot: int, rng: np.random.Generator, chunk: int = 64
) -> np.ndarray:
    """Means of ``n_boot`` bootstrap resamples of ``values`` (in chunks, bounded memory)."""
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, len(values), size=(k, len(values)))
        out[start : start + k] = values[idx].mean(axis=1)
    return out


def _stratified_means(
    groups: Sequence[np.ndarray], n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap distribution of the pooled mean of ``groups`` (resampled within each)."""
    total = sum(len(g) for g in groups)
    sums = np.zeros(n_boot)
    for g in groups:
        sums += _resampled_means(g, n_boot, rng) * len(g)
    return sums / total


def unpaired_mae_ci(
    abs_errors_by_seed: Sequence[np.ndarray], n_boot: int, seed: int, level: float = 0.95
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    boot = _stratified_means(abs_errors_by_seed, n_boot, rng)
    pooled = np.concatenate(list(abs_errors_by_seed))
    lo, hi = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return {"mae": float(pooled.mean()), "ci_low": float(lo), "ci_high": float(hi), "boot": boot}


def unpaired_greater_pvalue(boot_other: np.ndarray, boot_reference: np.ndarray) -> float:
    """One-sided bootstrap p-value for MAE(other) > MAE(reference), independent resamples."""
    return float((np.sum(boot_other - boot_reference <= 0) + 1) / (len(boot_other) + 1))


def paired_difference_ci(
    diffs_by_seed: Sequence[np.ndarray], n_boot: int, seed: int, level: float = 0.95
) -> dict[str, float]:
    """Mean over seeds of the mean per-molecule difference, with a bootstrap CI."""
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot)
    for d in diffs_by_seed:
        boot += _resampled_means(d, n_boot, rng)
    boot /= len(diffs_by_seed)
    estimate = float(np.mean([d.mean() for d in diffs_by_seed]))
    lo, hi = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return {
        "estimate": estimate,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_greater_than_zero": float((np.sum(boot <= 0) + 1) / (n_boot + 1)),
        "p_less_than_zero": float((np.sum(boot >= 0) + 1) / (n_boot + 1)),
    }


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (monotone, capped at 1), in input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
